# EyeVision AI — Deployment Guide

Full system = **frontend** (this Lovable project) + **backend** (`backend/`
folder, FastAPI + TensorFlow).

## 1. Backend — HuggingFace Spaces (recommended)

1. [Create a new Space](https://huggingface.co/new-space) → **SDK: Docker** →
   hardware: CPU Basic (free) for testing, T4 small GPU for production.
2. Clone the Space repo and copy everything inside `backend/` to the **root**
   of that repo (the `README.md` front-matter is required by HF).
3. Add your three trained `.h5` files to `models/`:
   - `models/resnet_model.h5`
   - `models/densenet_model.h5`
   - `models/efficientnet_model.h5`
   For files >10 MB, use **Git LFS** or upload via the Space's "Files" UI.
4. Commit and push. Build logs appear in the Space's "Logs" tab.
5. When the Space is **Running**, your endpoint is
   `https://<user>-<space>.hf.space`.
6. (Optional) In the Space's **Settings → Variables and secrets**, add
   `ALLOWED_ORIGINS=https://your-frontend-domain`.

## 2. Backend — Render

1. New → **Web Service** → connect this GitHub repo → Root directory: `backend`.
2. Environment: **Docker**. Plan: at least 2 GB RAM for TensorFlow.
3. Add a **Disk** mounted at `/app/models` and upload `.h5` files.
4. Env vars: `ALLOWED_ORIGINS=https://your-frontend-domain`.

## 3. Backend — Azure Container Apps

```bash
# 1. Build & push to Azure Container Registry
az acr build -t eyevision:latest -r <YourRegistry> backend/

# 2. Deploy
az containerapp create \
  --name eyevision-api \
  --resource-group <rg> \
  --environment <env> \
  --image <YourRegistry>.azurecr.io/eyevision:latest \
  --target-port 7860 --ingress external \
  --env-vars ALLOWED_ORIGINS=https://your-frontend \
  --cpu 2 --memory 4Gi
```

Use **Azure Files** mounted at `/app/models` for the trained weights.

## 4. Frontend — wire it up

1. In Lovable, open **Project → Settings → Build Variables** (or your local
   `.env` for self-hosted), set:
   ```
   VITE_API_URL=https://<user>-<space>.hf.space
   ```
2. Click **Update** in the publish dialog to deploy the frontend.
3. Verify the **"Live backend"** badge appears in the Analyze section.

## 5. Smoke test

```bash
curl https://<your-backend>/health
# → { "status": "ok", "tensorflow_version": "2.16.2", "models_loaded": [...] }

curl -X POST -F "file=@fundus.jpg" -F "model=EfficientNetB0" \
     https://<your-backend>/predict
```

## 6. Demo mode

When `VITE_API_URL` is unset or the backend is unreachable, the frontend
automatically falls back to **simulated inference** so the UI stays usable
for demos and screenshots. Each result is clearly labelled **LIVE** or
**DEMO** in the prediction card.