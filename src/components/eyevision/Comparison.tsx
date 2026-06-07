import { MODELS } from "@/lib/eye-analysis";
import { Check } from "lucide-react";

export function Comparison() {
  return (
    <section id="comparison" className="container mx-auto px-4 py-20">
      <div className="mb-10 text-center">
        <div className="text-xs uppercase tracking-widest text-primary">Benchmarks</div>
        <h2 className="mt-2 text-3xl font-bold md:text-4xl">Model Comparison</h2>
        <p className="mx-auto mt-2 max-w-xl text-muted-foreground">
          Side-by-side accuracy, speed, parameters and memory footprint.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-3">
        {MODELS.map((m) => (
          <div
            key={m.id}
            className="group relative overflow-hidden rounded-2xl border bg-[var(--gradient-card)] p-6 shadow-[var(--shadow-card)] transition-all hover:shadow-[var(--shadow-elegant)]"
          >
            <div className="absolute inset-x-0 top-0 h-1 bg-[var(--gradient-primary)]" />
            <div className="flex items-baseline justify-between">
              <h3 className="text-lg font-bold">{m.id}</h3>
              <span className="text-2xl font-bold text-gradient">{m.accuracy}%</span>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">{m.description}</p>

            <dl className="mt-5 grid grid-cols-2 gap-3 text-sm">
              <Stat label="Parameters" value={`${m.paramsM}M`} />
              <Stat label="Inference" value={`${m.speedMs}ms`} />
              <Stat label="Memory" value={`${m.memoryMB}MB`} />
              <Stat label="AUC" value={m.auc.toFixed(3)} />
            </dl>

            <ul className="mt-5 space-y-1.5 text-xs text-muted-foreground">
              <li className="flex items-center gap-1.5"><Check className="h-3 w-3 text-[var(--success)]" />Pretrained on ImageNet</li>
              <li className="flex items-center gap-1.5"><Check className="h-3 w-3 text-[var(--success)]" />Fine-tuned on EyePACS</li>
              <li className="flex items-center gap-1.5"><Check className="h-3 w-3 text-[var(--success)]" />Grad-CAM compatible</li>
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-muted/40 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="font-mono text-sm font-semibold">{value}</div>
    </div>
  );
}