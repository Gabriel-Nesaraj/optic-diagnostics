"""Model registry — loads Keras .h5 files lazily and caches them.

Each model declaration includes:
  - path to .h5 file (resolved from MODELS_DIR env var or ./models)
  - preprocessing function (matches the backbone the model was trained with)
  - the convolutional layer name used for Grad-CAM
  - benchmark metadata exposed via /models

Set USE_FAKE_WEIGHTS=1 to fall back to ImageNet-initialised backbones with a
fresh classifier head — useful for CI / first deploys before you upload your
trained weights. Production should ship real .h5 files in /app/models.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

log = logging.getLogger("eyevision.models")

# Disease classes — order MUST match the trained classifier head.
CLASS_NAMES: List[str] = ["Normal", "Glaucoma", "Diabetic Retinopathy", "Cataract", "AMD"]
NUM_CLASSES = len(CLASS_NAMES)

MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
USE_FAKE_WEIGHTS = os.getenv("USE_FAKE_WEIGHTS", "0") == "1"


@dataclass
class ModelBundle:
    id: str
    filename: str
    last_conv_layer: str
    preprocess: Callable[[np.ndarray], np.ndarray]
    description: str
    accuracy: float
    params_m: float
    auc: float
    # Loaded lazily:
    _model: object = field(default=None, repr=False)

    @property
    def path(self) -> str:
        return os.path.abspath(os.path.join(MODELS_DIR, self.filename))

    def public_info(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "accuracy": self.accuracy,
            "params_m": self.params_m,
            "auc": self.auc,
            "loaded": self._model is not None,
        }

    def model(self):
        if self._model is None:
            self._model = _load_or_build(self)
        return self._model


def _resnet_preprocess(x: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return preprocess_input(x)


def _densenet_preprocess(x: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.densenet import preprocess_input
    return preprocess_input(x)


def _efficientnet_preprocess(x: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(x)


MODEL_REGISTRY: Dict[str, ModelBundle] = {
    "ResNet50": ModelBundle(
        id="ResNet50",
        filename="resnet_model.h5",
        last_conv_layer="conv5_block3_out",
        preprocess=_resnet_preprocess,
        description="Skip connections for stable deep learning. Strong baseline.",
        accuracy=93.1,
        params_m=25.6,
        auc=0.962,
    ),
    "DenseNet121": ModelBundle(
        id="DenseNet121",
        filename="densenet_model.h5",
        last_conv_layer="conv5_block16_concat",
        preprocess=_densenet_preprocess,
        description="Dense feature sharing — excels on tiny retinal details.",
        accuracy=94.7,
        params_m=8.0,
        auc=0.974,
    ),
    "EfficientNetB0": ModelBundle(
        id="EfficientNetB0",
        filename="efficientnet_model.h5",
        last_conv_layer="top_conv",
        preprocess=_efficientnet_preprocess,
        description="Optimized CNN — high accuracy with low computational cost.",
        accuracy=95.9,
        params_m=5.3,
        auc=0.981,
    ),
}


def list_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def get_model(model_id: str) -> ModelBundle:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(model_id)
    return MODEL_REGISTRY[model_id]


def _load_or_build(bundle: ModelBundle):
    """Load a trained .h5 model; otherwise build a backbone-only fallback.

    The fallback exists so the API stays runnable in CI and on a fresh deploy
    before weights are uploaded. It is NOT for clinical use.
    """
    import tensorflow as tf

    if os.path.exists(bundle.path) and not USE_FAKE_WEIGHTS:
        log.info("Loading trained model %s from %s", bundle.id, bundle.path)
        return tf.keras.models.load_model(bundle.path, compile=False)

    log.warning(
        "Trained weights not found for %s at %s — building fallback backbone "
        "with random classifier head. Set USE_FAKE_WEIGHTS=0 and upload "
        "%s to enable real inference.",
        bundle.id, bundle.path, bundle.filename,
    )
    return _build_fallback(bundle.id)


def _build_fallback(model_id: str):
    """Build a backbone + classifier head with ImageNet weights."""
    import tensorflow as tf
    from tensorflow.keras import layers

    if model_id == "ResNet50":
        from tensorflow.keras.applications import ResNet50 as Backbone
    elif model_id == "DenseNet121":
        from tensorflow.keras.applications import DenseNet121 as Backbone
    elif model_id == "EfficientNetB0":
        from tensorflow.keras.applications import EfficientNetB0 as Backbone
    else:
        raise ValueError(model_id)

    backbone = Backbone(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return tf.keras.Model(backbone.input, out)