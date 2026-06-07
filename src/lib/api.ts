/**
 * EyeVision AI API client.
 *
 * Talks to the FastAPI backend hosted on HuggingFace Spaces / Render / Azure.
 * Configure with VITE_API_URL=https://your-space.hf.space at build time.
 *
 * When VITE_API_URL is unset OR the backend is unreachable, every call falls
 * back to local mock inference so the UI stays interactive in demo mode.
 */
import { mockPredict, MODELS, type ModelId, type Prediction } from "./eye-analysis";
import gradcamSample from "@/assets/gradcam-sample.jpg";

export const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, "") || "";

export interface ApiPrediction extends Prediction {
  inferenceMs?: number;
  source: "live" | "demo";
}

export interface GradCamResult {
  imageUrl: string;
  source: "live" | "demo";
}

function isLive() {
  return API_URL.length > 0;
}

async function postFile(path: string, file: File, fields: Record<string, string> = {}) {
  const fd = new FormData();
  fd.append("file", file);
  for (const [k, v] of Object.entries(fields)) fd.append(k, v);
  const res = await fetch(`${API_URL}${path}`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`${path} → ${res.status} ${res.statusText}`);
  return res.json();
}

function seedFromFile(file: File | null, fallback: string) {
  return file ? `${file.name}-${file.size}` : fallback;
}

export async function predict(
  file: File | null,
  model: ModelId,
  seed: string,
): Promise<ApiPrediction> {
  if (isLive() && file) {
    try {
      const data = await postFile("/predict", file, { model });
      return {
        model: data.model,
        predicted: data.predicted,
        confidence: data.confidence,
        probabilities: data.probabilities,
        inferenceMs: data.inference_ms,
        source: "live",
      };
    } catch (err) {
      console.warn("[EyeVision] /predict failed, falling back to demo:", err);
    }
  }
  const mock = mockPredict(seedFromFile(file, seed), model);
  return { ...mock, source: "demo" };
}

export async function compareAll(file: File | null, seed: string): Promise<ApiPrediction[]> {
  if (isLive() && file) {
    try {
      const data = await postFile("/compare", file);
      return data.results.map((r: any) => ({
        model: r.model,
        predicted: r.predicted,
        confidence: r.confidence,
        probabilities: r.probabilities,
        inferenceMs: r.inference_ms,
        source: "live" as const,
      }));
    } catch (err) {
      console.warn("[EyeVision] /compare failed, falling back to demo:", err);
    }
  }
  return MODELS.map((m, i) => {
    const p = mockPredict(seedFromFile(file, seed) + i, m.id);
    return { ...p, inferenceMs: m.speedMs, source: "demo" as const };
  });
}

export async function gradcam(
  file: File | null,
  model: ModelId,
): Promise<GradCamResult> {
  if (isLive() && file) {
    try {
      const data = await postFile("/gradcam", file, { model });
      return {
        imageUrl: `data:image/png;base64,${data.overlay_png_base64}`,
        source: "live",
      };
    } catch (err) {
      console.warn("[EyeVision] /gradcam failed, falling back to demo:", err);
    }
  }
  return { imageUrl: gradcamSample, source: "demo" };
}

export function backendStatus() {
  return { live: isLive(), url: API_URL || null };
}