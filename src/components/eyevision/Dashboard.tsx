import { MODELS } from "@/lib/eye-analysis";

export function Dashboard() {
  return (
    <section id="dashboard" className="container mx-auto px-4 py-20">
      <div className="mb-10 text-center">
        <div className="text-xs uppercase tracking-widest text-primary">Research Dashboard</div>
        <h2 className="mt-2 text-3xl font-bold md:text-4xl">Evaluation Metrics</h2>
        <p className="mx-auto mt-2 max-w-xl text-muted-foreground">
          Validation results aggregated over EyePACS, DRIVE, REFUGE and Messidor.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Confusion Matrix · SwinTransformer (Ensemble Top)">
          <ConfusionMatrix />
        </Card>
        <Card title="ROC Curve · Multi-Model AUC">
          <RocCurve />
        </Card>
        <Card title="Accuracy / Epoch">
          <AccuracyChart />
        </Card>
        <Card title="Precision · Recall · F1">
          <PrecisionRecallTable />
        </Card>
      </div>
    </section>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border bg-card p-5 shadow-[var(--shadow-card)]">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      {children}
    </div>
  );
}

const LABELS = ["Normal", "Glauc.", "DR", "AMD"];
const MATRIX = [
  [96, 1, 2, 1],
  [2, 95, 2, 1],
  [1, 2, 96, 1],
  [1, 1, 2, 96],
];

function ConfusionMatrix() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            <th className="p-1.5"></th>
            {LABELS.map((l) => (
              <th key={l} className="p-1.5 text-center text-muted-foreground">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MATRIX.map((row, i) => (
            <tr key={i}>
              <th className="p-1.5 text-right text-muted-foreground">{LABELS[i]}</th>
              {row.map((v, j) => (
                <td
                  key={j}
                  className="rounded p-2 text-center font-mono"
                  style={{
                    background: `color-mix(in oklab, var(--primary) ${v}%, transparent)`,
                    color: v > 50 ? "var(--primary-foreground)" : "var(--foreground)",
                  }}
                >
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RocCurve() {
  const curves = [
    { name: "SwinTransformer", color: "oklch(0.7 0.16 195)", auc: 0.992 },
    { name: "ConvNeXt", color: "oklch(0.6 0.18 230)", auc: 0.989 },
    { name: "EfficientNetV2", color: "oklch(0.7 0.18 75)", auc: 0.984 },
  ];
  // generate a curve
  const points = (boost: number) =>
    Array.from({ length: 30 }, (_, i) => {
      const x = i / 29;
      const y = Math.min(1, Math.pow(x, 1 / (boost * 6)) + 0.01);
      return `${x * 200},${100 - y * 100}`;
    }).join(" ");

  return (
    <div>
      <svg viewBox="0 0 200 100" className="h-48 w-full">
        <line x1="0" y1="100" x2="200" y2="0" stroke="currentColor" strokeOpacity="0.15" strokeDasharray="2 2" />
        {curves.map((c, i) => (
          <polyline
            key={c.name}
            fill="none"
            stroke={c.color}
            strokeWidth="1.5"
            points={points(1 + i * 0.15)}
          />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-3 text-xs">
        {curves.map((c) => (
          <div key={c.name} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: c.color }} />
            <span className="text-muted-foreground">{c.name}</span>
            <span className="font-mono">{c.auc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AccuracyChart() {
  const epochs = 20;
  const series = MODELS.map((m, idx) => ({
    name: m.id,
    color: ["oklch(0.7 0.18 75)", "oklch(0.6 0.18 230)", "oklch(0.7 0.16 195)"][idx],
    points: Array.from({ length: epochs }, (_, i) => {
      const x = i / (epochs - 1);
      const acc = m.accuracy / 100 - (1 - x) * 0.25 + Math.sin(i + idx) * 0.005;
      return `${x * 200},${100 - acc * 100}`;
    }).join(" "),
  }));
  return (
    <div>
      <svg viewBox="0 0 200 100" className="h-48 w-full">
        {[0.7, 0.8, 0.9, 1].map((y) => (
          <line key={y} x1="0" x2="200" y1={100 - y * 100} y2={100 - y * 100} stroke="currentColor" strokeOpacity="0.08" />
        ))}
        {series.map((s) => (
          <polyline key={s.name} fill="none" stroke={s.color} strokeWidth="1.5" points={s.points} />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-3 text-xs">
        {series.map((s) => (
          <div key={s.name} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
            <span className="text-muted-foreground">{s.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PrecisionRecallTable() {
  return (
    <table className="w-full text-sm">
      <thead className="text-xs uppercase text-muted-foreground">
        <tr><th className="py-2 text-left">Model</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
      </thead>
      <tbody>
        {MODELS.map((m) => (
          <tr key={m.id} className="border-t">
            <td className="py-2 font-medium">{m.id}</td>
            <td className="text-center font-mono">{m.precision.toFixed(2)}</td>
            <td className="text-center font-mono">{m.recall.toFixed(2)}</td>
            <td className="text-center font-mono">{m.f1.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}