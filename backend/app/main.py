"""EyeVision AI — FastAPI backend.

Production-ready inference server for retinal fundus disease classification
with three CNN backbones (ResNet50, DenseNet121, EfficientNetB0) and real
Grad-CAM explainability via tf.GradientTape.

Designed for HuggingFace Spaces (Docker SDK), Render, and Azure.
"""
from __future__ import annotations

import io
import os
import time
import base64
import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .models import MODEL_REGISTRY, get_model, list_models, ModelBundle
from .inference import preprocess_image, predict_single, predict_all
from .gradcam import make_gradcam_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eyevision")

MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

app = FastAPI(
    title="EyeVision AI",
    description="Deep-learning ophthalmology inference API",
    version="1.0.0",
)

# CORS — HuggingFace Spaces are typically called from arbitrary frontends.
# Restrict ALLOWED_ORIGINS via env var in production.
_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
_origins = ["*"] if _origins_env.strip() == "*" else [o.strip() for o in _origins_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #
class ClassProbability(BaseModel):
    disease: str
    prob: float


class PredictionOut(BaseModel):
    model: str
    predicted: str
    confidence: float
    probabilities: List[ClassProbability]
    inference_ms: float


class CompareOut(BaseModel):
    results: List[PredictionOut]


class GradCamOut(BaseModel):
    model: str
    predicted: str
    confidence: float
    heatmap_png_base64: str
    overlay_png_base64: str


class HealthOut(BaseModel):
    status: str
    tensorflow_version: str
    models_loaded: List[str]
    gpu_available: bool


# --------------------------------------------------------------------------- #
# File validation
# --------------------------------------------------------------------------- #
async def _read_validated_image(file: UploadFile) -> bytes:
    """HIPAA-style upload guard. Validates mime, extension and size."""
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported media type: {file.content_type}")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file extension: {ext}")
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_FILE_BYTES} bytes")
    if len(data) < 256:
        raise HTTPException(400, "File too small to be a valid image")
    return data


def _resolve_model(model_id: str) -> ModelBundle:
    try:
        return get_model(model_id)
    except KeyError:
        raise HTTPException(400, f"Unknown model '{model_id}'. Available: {list_models()}")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    return HealthOut(
        status="ok",
        tensorflow_version=tf.__version__,
        models_loaded=list_models(),
        gpu_available=len(gpus) > 0,
    )


@app.get("/models")
def models_info():
    return {"models": [m.public_info() for m in MODEL_REGISTRY.values()]}


@app.post("/predict", response_model=PredictionOut)
async def predict(
    file: UploadFile = File(..., description="Retinal fundus image (JPG/PNG)"),
    model: str = Form("EfficientNetB0"),
):
    raw = await _read_validated_image(file)
    bundle = _resolve_model(model)
    arr = preprocess_image(raw, target_size=(224, 224))

    t0 = time.perf_counter()
    pred = predict_single(bundle, arr)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionOut(
        model=bundle.id,
        predicted=pred["predicted"],
        confidence=pred["confidence"],
        probabilities=[ClassProbability(**p) for p in pred["probabilities"]],
        inference_ms=round(elapsed_ms, 2),
    )


@app.post("/compare", response_model=CompareOut)
async def compare(file: UploadFile = File(...)):
    """Run inference on every loaded model in parallel-friendly sequence."""
    raw = await _read_validated_image(file)
    arr = preprocess_image(raw, target_size=(224, 224))

    results = []
    for bundle in MODEL_REGISTRY.values():
        t0 = time.perf_counter()
        pred = predict_single(bundle, arr)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append(
            PredictionOut(
                model=bundle.id,
                predicted=pred["predicted"],
                confidence=pred["confidence"],
                probabilities=[ClassProbability(**p) for p in pred["probabilities"]],
                inference_ms=round(elapsed_ms, 2),
            )
        )
    return CompareOut(results=results)


@app.post("/gradcam", response_model=GradCamOut)
async def gradcam(
    file: UploadFile = File(...),
    model: str = Form("EfficientNetB0"),
):
    raw = await _read_validated_image(file)
    bundle = _resolve_model(model)
    arr = preprocess_image(raw, target_size=(224, 224))
    pred = predict_single(bundle, arr)
    heatmap_b64, overlay_b64 = make_gradcam_overlay(bundle, arr, raw)
    return GradCamOut(
        model=bundle.id,
        predicted=pred["predicted"],
        confidence=pred["confidence"],
        heatmap_png_base64=heatmap_b64,
        overlay_png_base64=overlay_b64,
    )


@app.exception_handler(Exception)
async def unhandled(_req, exc):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})