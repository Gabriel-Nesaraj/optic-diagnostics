---
title: EyeVision AI
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# EyeVision AI — FastAPI Backend

Real TensorFlow/Keras inference server for retinal fundus disease
classification. Pairs with the EyeVision AI frontend.

## Endpoints

| Method | Path        | Purpose                                       |
| ------ | ----------- | --------------------------------------------- |
| GET    | `/health`   | Liveness + TF version + GPU + loaded models   |
| GET    | `/models`   | Model metadata for the comparison panel       |
| POST   | `/predict`  | Single-model inference (multipart upload)     |
| POST   | `/compare`  | All three models in one call                  |
| POST   | `/gradcam`  | Real Grad-CAM heatmap + overlay (base64 PNG)  |

`POST /predict` and `POST /gradcam` expect a multipart form:

```
file:  retinal_fundus.jpg     (image/jpeg | image/png, ≤10 MB)
model: EfficientNetB0         (ResNet50 | DenseNet121 | EfficientNetB0)
```

Response (`/predict`):

```json
{
  "model": "EfficientNetB0",
  "predicted": "Glaucoma",
  "confidence": 0.964,
  "probabilities": [
    { "disease": "Normal", "prob": 0.012 },
    { "disease": "Glaucoma", "prob": 0.964 },
    { "disease": "Diabetic Retinopathy", "prob": 0.011 },
    { "disease": "Cataract", "prob": 0.008 },
    { "disease": "AMD", "prob": 0.005 }
  ],
  "inference_ms": 84.21
}
```

## Folder layout

```
backend/
├── app/
│   ├── main.py        # FastAPI app + routes + upload validation
│   ├── models.py      # Model registry, lazy loading, fallback backbones
│   ├── inference.py   # Preprocess + predict_single + predict_all
│   └── gradcam.py     # Real Grad-CAM via tf.GradientTape
├── models/            # Drop your .h5 files here (see models/README.md)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Local development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (optional) place your trained .h5 files in ./models
uvicorn app.main:app --reload --port 7860
open http://localhost:7860/docs
```

No `.h5` files? The server still starts — it logs a warning and uses
ImageNet-initialised backbones with a random classifier head. Replace with
trained weights before any real use.

## Deploy → HuggingFace Spaces

1. Create a new Space → **SDK: Docker** → choose hardware (CPU Basic is fine
   for demo, GPU Small for production).
2. Push this `backend/` directory to the Space's git repo as the root. The
   YAML front-matter above is the Space's required header.
3. Upload your three `.h5` files via the Space UI under `/models` (or
   commit them — note Git LFS for >10 MB).
4. The Space exposes `https://<user>-<space>.hf.space`. Use that as
   `VITE_API_URL` in the frontend.
5. (Optional) Set the `ALLOWED_ORIGINS` Space secret to your frontend's
   domain in production.

## Deploy → Render

1. New → **Web Service** → connect repo → root directory `backend/`.
2. Environment: **Docker**. Render auto-detects the Dockerfile.
3. Add env vars: `MODELS_DIR=/app/models`, `ALLOWED_ORIGINS=https://your-frontend`.
4. Use a **disk** for `/app/models` and upload `.h5` files there, or commit
   them with Git LFS.

## Deploy → Azure Container Apps

```bash
az acr build -t eyevision:latest -r <YourRegistry> backend/
az containerapp create \
  --name eyevision-api \
  --resource-group <rg> \
  --environment <env> \
  --image <YourRegistry>.azurecr.io/eyevision:latest \
  --target-port 7860 --ingress external \
  --env-vars ALLOWED_ORIGINS=https://your-frontend \
  --cpu 2 --memory 4Gi
```

## Security notes (HIPAA-style)

- All uploads are validated: MIME type, extension, max 10 MB, min 256 B.
- No image is persisted to disk. The request body lives only in memory.
- The API has **no auth** by default — put it behind an API gateway, Cloudflare
  Access, HuggingFace private Space, or your own reverse proxy in production.
- This system is for **research and education only**. It is not a medical
  device and must not be used to make clinical decisions.

## Future extensions

The registry pattern in `app/models.py` was designed to absorb:
- **OCT** models — add a separate registry keyed on modality.
- **Multi-label classification** — change the head + replace argmax with
  threshold-based class selection in `inference.py`.
- **Segmentation** — add a `/segment` route returning a PNG mask using the
  same upload validation pipeline.
- **Patient records** — wire `POST /history` to a database (Postgres / Mongo)
  and persist anonymised metadata only.