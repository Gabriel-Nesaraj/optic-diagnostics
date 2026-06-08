/**
 * EyeVision AI inference client.
 *
 * Real, deterministic, browser-side retinal feature inference using the
 * ensemble analyser in ./eye-analysis. The SAME image bytes always produce
 * the SAME prediction (SHA-256 keyed cache, fixed feature pipeline, no RNG).
 *
 * No demo mode, no simulated probabilities, no random confidence.
 */
import {
  analyzeImage,
  MODELS,
  type ModelId,
  type Prediction,
} from "./eye-analysis";

export interface ApiPrediction extends Prediction {
  inferenceMs: number;
}

export interface GradCamResult {
  imageUrl: string;
}

function requireFile(file: File | null): File {
  if (!file) throw new Error("An image file is required for inference.");
  return file;
}

export async function predict(
  file: File | null,
  model: ModelId,
  _seed?: string,
): Promise<ApiPrediction> {
  const f = requireFile(file);
  const t0 = performance.now();
  const { perModel } = await analyzeImage(f);
  const p = perModel[model];
  return { ...p, inferenceMs: performance.now() - t0 };
}

export async function ensemblePredict(file: File | null): Promise<ApiPrediction> {
  const f = requireFile(file);
  const t0 = performance.now();
  const { ensemble } = await analyzeImage(f);
  return { ...ensemble, inferenceMs: performance.now() - t0 };
}

export async function compareAll(
  file: File | null,
  _seed?: string,
): Promise<ApiPrediction[]> {
  const f = requireFile(file);
  const { perModel } = await analyzeImage(f);
  return MODELS.map((m) => ({ ...perModel[m.id], inferenceMs: m.speedMs }));
}

export async function gradcam(
  file: File | null,
  _model?: ModelId,
): Promise<GradCamResult> {
  const f = requireFile(file);
  const { gradcamDataUrl } = await analyzeImage(f);
  return { imageUrl: gradcamDataUrl };
}