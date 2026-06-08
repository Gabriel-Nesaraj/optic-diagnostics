import { Upload, Cog, Cpu, Layers, Activity, Stethoscope } from "lucide-react";

const STEPS = [
  { Icon: Upload, label: "Upload Eye Image", sub: "JPG · PNG · JPEG" },
  { Icon: Cog, label: "Preprocessing", sub: "Resize 224×224 · Normalize" },
  { Icon: Cpu, label: "SOTA Ensemble", sub: "Swin / ConvNeXt / EfficientNetV2" },
  { Icon: Layers, label: "Feature Extraction", sub: "Deep representations" },
  { Icon: Activity, label: "Classification", sub: "5 disease classes" },
  { Icon: Stethoscope, label: "Clinical Prediction", sub: "Confidence + Grad-CAM" },
];

export function Workflow() {
  return (
    <section id="workflow" className="container mx-auto px-4 py-20">
      <div className="mb-10 text-center">
        <div className="text-xs uppercase tracking-widest text-primary">Pipeline</div>
        <h2 className="mt-2 text-3xl font-bold md:text-4xl">End-to-End Inference Workflow</h2>
        <p className="mx-auto mt-2 max-w-xl text-muted-foreground">
          Transparent, modular, and reproducible — from raw fundus capture to clinical-grade prediction.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
        {STEPS.map(({ Icon, label, sub }, i) => (
          <div key={label} className="relative">
            <div className="glass-panel rounded-xl p-5 text-center shadow-[var(--shadow-card)] transition-transform hover:-translate-y-1">
              <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-[var(--gradient-primary)] text-primary-foreground shadow-[var(--shadow-glow)]">
                <Icon className="h-5 w-5" />
              </div>
              <div className="text-sm font-semibold">{label}</div>
              <div className="mt-1 text-xs text-muted-foreground">{sub}</div>
              <div className="mt-3 text-[10px] font-mono text-primary">STEP 0{i + 1}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}