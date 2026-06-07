"""Image preprocessing + prediction helpers."""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image

from .models import CLASS_NAMES, ModelBundle


def preprocess_image(raw_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Decode bytes → RGB → resize → float32 ndarray of shape (1,H,W,3).

    Model-specific normalization (mean/std) happens later via bundle.preprocess.
    """
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_single(bundle: ModelBundle, image_batch: np.ndarray) -> dict:
    """Run a single forward pass and return ranked probabilities."""
    model = bundle.model()
    x = bundle.preprocess(image_batch.copy())
    preds = model.predict(x, verbose=0)[0]  # shape (NUM_CLASSES,)
    preds = np.asarray(preds, dtype=np.float64)
    # Defensive softmax in case the head returns logits.
    if not np.isclose(preds.sum(), 1.0, atol=1e-3):
        e = np.exp(preds - preds.max())
        preds = e / e.sum()
    top_idx = int(np.argmax(preds))
    return {
        "predicted": CLASS_NAMES[top_idx],
        "confidence": float(preds[top_idx]),
        "probabilities": [
            {"disease": CLASS_NAMES[i], "prob": float(preds[i])} for i in range(len(CLASS_NAMES))
        ],
    }


def predict_all(bundles, image_batch: np.ndarray):
    return [(b.id, predict_single(b, image_batch)) for b in bundles]