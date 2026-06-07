# EyeVision AI — FastAPI + TensorFlow backend
# Compatible with HuggingFace Spaces (Docker SDK), Render, Azure Container Apps.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODELS_DIR=/app/models \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo libpng16-16 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Source code
COPY app ./app
COPY models ./models

# HuggingFace Spaces sets PORT=7860 by default
EXPOSE 7860

# HF Spaces expects the app to bind 0.0.0.0:7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]