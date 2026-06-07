export type DiseaseClass =
  | "Normal"
  | "Glaucoma"
  | "Diabetic Retinopathy"
  | "Cataract"
  | "AMD";

export type ModelId = "ResNet50" | "DenseNet121" | "EfficientNetB0";

export interface ModelMeta {
  id: ModelId;
  description: string;
  accuracy: number;
  paramsM: number;
  speedMs: number;
  memoryMB: number;
  precision: number;
  recall: number;
  f1: number;
  auc: number;
}

export const MODELS: ModelMeta[] = [
  {
    id: "ResNet50",
    description: "Skip connections for stable deep learning. Strong baseline.",
    accuracy: 93.1,
    paramsM: 25.6,
    speedMs: 42,
    memoryMB: 98,
    precision: 0.92,
    recall: 0.91,
    f1: 0.915,
    auc: 0.962,
  },
  {
    id: "DenseNet121",
    description: "Dense feature sharing — excels on tiny retinal details.",
    accuracy: 94.7,
    paramsM: 8.0,
    speedMs: 55,
    memoryMB: 33,
    precision: 0.94,
    recall: 0.93,
    f1: 0.935,
    auc: 0.974,
  },
  {
    id: "EfficientNetB0",
    description: "Optimized CNN — high accuracy with low computational cost.",
    accuracy: 95.9,
    paramsM: 5.3,
    speedMs: 28,
    memoryMB: 21,
    precision: 0.96,
    recall: 0.95,
    f1: 0.955,
    auc: 0.981,
  },
];

export const DISEASES: DiseaseClass[] = [
  "Normal",
  "Glaucoma",
  "Diabetic Retinopathy",
  "Cataract",
  "AMD",
];

// Deterministic mock prediction based on image hash + model
export function mockPredict(seed: string, model: ModelId) {
  // simple hash
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  const meta = MODELS.find((m) => m.id === model)!;

  // generate probabilities
  const raw = DISEASES.map((_, i) => {
    const v = Math.sin(h * 0.0001 + i * 1.7 + meta.accuracy) * 0.5 + 0.5;
    return Math.pow(v, 2);
  });
  const sum = raw.reduce((a, b) => a + b, 0);
  let probs = raw.map((v) => v / sum);

  // boost the top one to look confident
  const top = probs.indexOf(Math.max(...probs));
  probs = probs.map((p, i) =>
    i === top ? Math.min(0.98, 0.85 + (meta.accuracy - 90) / 100 + p * 0.1) : p * 0.15,
  );
  const s2 = probs.reduce((a, b) => a + b, 0);
  probs = probs.map((p) => p / s2);

  return {
    model,
    predicted: DISEASES[top],
    confidence: probs[top],
    probabilities: DISEASES.map((d, i) => ({ disease: d, prob: probs[i] })),
  };
}

export type Prediction = ReturnType<typeof mockPredict>;