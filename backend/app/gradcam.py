"""Real Grad-CAM implementation using tf.GradientTape.

Returns base64-encoded PNGs:
  - heatmap_png_base64: standalone JET-colored heatmap
  - overlay_png_base64: original fundus image with 40% heatmap overlay
"""
from __future__ import annotations

import io
import base64
from typing import Tuple

import numpy as np
from PIL import Image

from .models import ModelBundle


def _png_b64(arr_u8: np.ndarray) -> str:
    img = Image.fromarray(arr_u8)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _apply_jet_colormap(gray01: np.ndarray) -> np.ndarray:
    """Convert a [0,1] grayscale heatmap to an RGB JET-like colormap (uint8)."""
    # Approximate matplotlib's 'jet' without requiring matplotlib at runtime.
    x = np.clip(gray01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def make_gradcam_overlay(
    bundle: ModelBundle,
    image_batch: np.ndarray,
    original_bytes: bytes,
    alpha: float = 0.4,
) -> Tuple[str, str]:
    """Compute Grad-CAM for the predicted class and return (heatmap, overlay)."""
    import tensorflow as tf

    model = bundle.model()
    x = bundle.preprocess(image_batch.copy())
    x_tf = tf.convert_to_tensor(x)

    try:
        last_conv = model.get_layer(bundle.last_conv_layer)
    except ValueError:
        # Fallback: find last 4D conv-like output.
        last_conv = None
        for layer in reversed(model.layers):
            try:
                if len(layer.output_shape) == 4:
                    last_conv = layer
                    break
            except Exception:  # pragma: no cover
                continue
        if last_conv is None:
            raise RuntimeError("No convolutional layer found for Grad-CAM")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(x_tf, training=False)
        top_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, top_idx]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
    conv_out = conv_out[0]  # (H, W, C)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (H, W)
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap = (heatmap / max_val).numpy()

    # Resize heatmap to the original image size for the overlay.
    original = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    W, H = original.size
    heatmap_resized = np.asarray(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    color = _apply_jet_colormap(heatmap_resized)
    base = np.asarray(original, dtype=np.float32)
    overlay = (base * (1 - alpha) + color.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return _png_b64(color), _png_b64(overlay)