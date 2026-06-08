import { useRef, useState } from "react";
import { Upload, X, Loader2, Sparkles, Download, FileImage, History, NotebookPen, Eye, Activity, Wifi, WifiOff, AlertTriangle } from "lucide-react";
import { MODELS, type ModelId } from "@/lib/eye-analysis";
import { predict, compareAll, gradcam, backendStatus, type ApiPrediction } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import sampleFundus from "@/assets/sample-fundus.jpg";

interface HistoryItem {
  id: string;
  name: string;
  thumb: string;
  model: ModelId;
  predicted: string;
  confidence: number;
  time: string;
}

export function Analyze() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string>("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageHash, setImageHash] = useState<string>("");
  const [model, setModel] = useState<ModelId>("EfficientNetB0");
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<ApiPrediction | null>(null);
  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);
  const [compareResults, setCompareResults] = useState<ApiPrediction[] | null>(null);
  const [comparing, setComparing] = useState(false);
  const [showGradCam, setShowGradCam] = useState(true);
  const [notes, setNotes] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const status = backendStatus();

  const hashFile = async (file: File) => {
    const buf = await file.arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", buf);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  };

  const handleFile = async (file: File) => {
    if (!/(jpeg|jpg|png)$/i.test(file.type) && !/\.(jpe?g|png)$/i.test(file.name)) {
      toast.error("Only JPG, JPEG, or PNG images are supported.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast.error("Image must be under 10 MB.");
      return;
    }
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setImageName(file.name);
    setImageFile(file);
    setPrediction(null);
    setGradcamUrl(null);
    setCompareResults(null);
    try {
      setImageHash(await hashFile(file));
    } catch {
      setImageHash(`${file.name}-${file.size}-${file.lastModified}`);
    }
  };

  const loadSample = async () => {
    setImageUrl(sampleFundus);
    setImageName("sample_fundus.jpg");
    setPrediction(null);
    setGradcamUrl(null);
    setCompareResults(null);
    try {
      const blob = await fetch(sampleFundus).then((r) => r.blob());
      const file = new File([blob], "sample_fundus.jpg", { type: blob.type || "image/jpeg" });
      setImageFile(file);
      setImageHash(await hashFile(file));
    } catch {
      setImageFile(null);
      setImageHash("sample_fundus.jpg");
    }
  };

  const analyze = async () => {
    if (!imageUrl) {
      toast.error("Upload a retinal image first.");
      return;
    }
    setLoading(true);
    setPrediction(null);
    setGradcamUrl(null);
    setCompareResults(null);
    try {
      // Deterministic seed: SHA-256 of file bytes → same image always yields same prediction.
      const seed = imageHash || imageName;
      const [pred, cam] = await Promise.all([
        predict(imageFile, model, seed),
        gradcam(imageFile, model),
      ]);
      setPrediction(pred);
      setGradcamUrl(cam.imageUrl);
      setHistory((h) =>
        [
          {
            id: Math.random().toString(36).slice(2),
            name: imageName,
            thumb: imageUrl!,
            model: pred.model as ModelId,
            predicted: pred.predicted,
            confidence: pred.confidence,
            time: new Date().toLocaleTimeString(),
          },
          ...h,
        ].slice(0, 6),
      );
      if (pred.source === "demo" && status.live) {
        toast.warning("Backend unreachable — showing simulated result.");
      }
    } catch (err) {
      console.error(err);
      toast.error("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const runCompare = async () => {
    if (!imageUrl) {
      toast.error("Upload a retinal image first.");
      return;
    }
    setComparing(true);
    try {
      const seed = imageHash || imageName;
      const results = await compareAll(imageFile, seed);
      setCompareResults(results);
    } catch (err) {
      console.error(err);
      toast.error("Comparison failed.");
    } finally {
      setComparing(false);
    }
  };

  const clear = () => {
    setImageUrl(null);
    setImageName("");
    setImageFile(null);
    setPrediction(null);
    setGradcamUrl(null);
    setCompareResults(null);
    setNotes("");
  };

  const downloadReport = () => {
    if (!prediction) return;
    const html = buildReportHTML({ prediction, imageUrl, imageName, notes });
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `EyeVision-Report-${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Report downloaded. Open and use Print → Save as PDF.");
  };

  return (
    <section id="analyze" className="container mx-auto px-4 py-20">
      <div className="mb-10 text-center">
        <div className="text-xs uppercase tracking-widest text-primary">Workspace</div>
        <h2 className="mt-2 text-3xl font-bold md:text-4xl">Retinal Image Analysis</h2>
        <p className="mx-auto mt-2 max-w-xl text-muted-foreground">
          Upload a fundus image, select a CNN architecture, and run AI-assisted disease detection.
        </p>
        <div className="mt-4 inline-flex items-center gap-2 rounded-full border bg-card px-3 py-1 text-xs">
          {status.live ? (
            <><Wifi className="h-3 w-3 text-[var(--success)]" /><span>Live backend</span><code className="text-muted-foreground">{status.url}</code></>
          ) : (
            <><WifiOff className="h-3 w-3 text-muted-foreground" /><span>Demo mode — set <code className="font-mono">VITE_API_URL</code> to enable real inference</span></>
          )}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-5">
        {/* Upload + Model */}
        <div className="lg:col-span-3 space-y-6">
          <div className="rounded-2xl border bg-card p-5 shadow-[var(--shadow-card)]">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold">
              <FileImage className="h-4 w-4 text-primary" /> Upload Eye Image
            </h3>

            {!imageUrl ? (
              <label
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
                }}
                className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-primary/30 bg-primary/5 px-6 py-12 text-center transition-colors hover:bg-primary/10"
              >
                <input
                  ref={fileRef}
                  type="file"
                  className="hidden"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                />
                <Upload className="mb-3 h-8 w-8 text-primary" />
                <div className="font-medium">Drop fundus image or click to browse</div>
                <div className="mt-1 text-xs text-muted-foreground">JPG · PNG · JPEG · max 10MB</div>
                <Button
                  type="button"
                  variant="link"
                  size="sm"
                  className="mt-3"
                  onClick={(e) => {
                    e.preventDefault();
                    loadSample();
                  }}
                >
                  Or use sample image →
                </Button>
              </label>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                <div className="group relative overflow-hidden rounded-xl border bg-black">
                  <img src={imageUrl} alt="Uploaded fundus" className="aspect-square w-full object-cover" />
                  <div className="absolute left-2 top-2 rounded-full bg-black/50 px-2 py-0.5 text-[10px] uppercase text-white backdrop-blur">
                    Original · 224×224
                  </div>
                  <button
                    onClick={clear}
                    className="absolute right-2 top-2 rounded-full bg-black/60 p-1 text-white opacity-0 transition-opacity group-hover:opacity-100"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div className="relative overflow-hidden rounded-xl border bg-black">
                  <img
                    src={prediction && showGradCam && gradcamUrl ? gradcamUrl : imageUrl}
                    alt="Grad-CAM heatmap"
                    className="aspect-square w-full object-cover"
                  />
                  <div className="absolute left-2 top-2 rounded-full bg-black/50 px-2 py-0.5 text-[10px] uppercase text-white backdrop-blur">
                    {prediction && showGradCam ? "Grad-CAM Overlay" : "Awaiting analysis"}
                  </div>
                  {prediction && (
                    <button
                      onClick={() => setShowGradCam((s) => !s)}
                      className="absolute bottom-2 right-2 rounded-md bg-white/10 px-2 py-1 text-[10px] text-white backdrop-blur hover:bg-white/20"
                    >
                      {showGradCam ? "Hide overlay" : "Show overlay"}
                    </button>
                  )}
                </div>
              </div>
            )}
            <p className="mt-3 flex items-start gap-1.5 text-[11px] text-muted-foreground">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-[var(--warning)]" />
              Fundus models detect AMD, Glaucoma, DR. Cataract requires slit-lamp imagery — predictions on fundus input are advisory only.
            </p>
          </div>

          <div className="rounded-2xl border bg-card p-5 shadow-[var(--shadow-card)]">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold">
              <Sparkles className="h-4 w-4 text-primary" /> Select CNN Model
            </h3>
            <div className="grid gap-3 md:grid-cols-3">
              {MODELS.map((m) => {
                const active = model === m.id;
                return (
                  <button
                    key={m.id}
                    onClick={() => setModel(m.id)}
                    className={`rounded-xl border p-4 text-left transition-all ${
                      active
                        ? "border-primary bg-primary/5 shadow-[var(--shadow-glow)]"
                        : "hover:border-primary/40 hover:bg-muted/40"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">{m.id}</span>
                      <span className={`h-2 w-2 rounded-full ${active ? "bg-primary animate-pulse" : "bg-muted-foreground/30"}`} />
                    </div>
                    <p className="mt-1.5 text-xs text-muted-foreground">{m.description}</p>
                    <div className="mt-3 flex justify-between text-[10px] font-mono text-muted-foreground">
                      <span>{m.paramsM}M params</span>
                      <span>{m.accuracy}% acc</span>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mt-5 flex flex-wrap gap-3">
              <Button
                size="lg"
                onClick={analyze}
                disabled={loading || !imageUrl}
                className="bg-[var(--gradient-primary)] text-primary-foreground shadow-[var(--shadow-glow)] hover:opacity-95"
              >
                {loading ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing…</>
                ) : (
                  <><Eye className="mr-2 h-4 w-4" /> Run Analysis</>
                )}
              </Button>
              <Button
                size="lg"
                variant="secondary"
                onClick={runCompare}
                disabled={comparing || !imageUrl}
              >
                {comparing ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Comparing…</>
                ) : (
                  <><Activity className="mr-2 h-4 w-4" /> Compare All Models</>
                )}
              </Button>
              {prediction && (
                <Button variant="outline" onClick={downloadReport}>
                  <Download className="mr-2 h-4 w-4" /> Download Report
                </Button>
              )}
            </div>

            {loading && (
              <div className="mt-4 rounded-lg border border-primary/30 bg-primary/5 p-3 text-sm">
                <div className="flex items-center gap-2 text-primary">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing retinal image using {model}…
                </div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
                  <div className="h-full w-1/2 animate-pulse bg-[var(--gradient-primary)]" />
                </div>
              </div>
            )}
          </div>

          {compareResults && (
            <CompareResultsPanel results={compareResults} />
          )}

          <div className="rounded-2xl border bg-card p-5 shadow-[var(--shadow-card)]">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold">
              <NotebookPen className="h-4 w-4 text-primary" /> Doctor's Notes
            </h3>
            <Textarea
              placeholder="Clinical observations, patient history, follow-up recommendations…"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={4}
            />
          </div>
        </div>

        {/* Prediction Panel */}
        <div className="lg:col-span-2 space-y-6">
          <PredictionPanel prediction={prediction} loading={loading} model={model} />
          <HistoryPanel history={history} />
        </div>
      </div>
    </section>
  );
}

function PredictionPanel({ prediction, loading, model }: { prediction: ApiPrediction | null; loading: boolean; model: ModelId }) {
  return (
    <div className="rounded-2xl border bg-[var(--gradient-card)] p-5 shadow-[var(--shadow-card)]">
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        Prediction
      </h3>

      {!prediction && !loading && (
        <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
          <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-full border-2 border-dashed border-muted-foreground/30">
            <Eye className="h-6 w-6" />
          </div>
          <p className="text-sm">Upload an image and run analysis to see results.</p>
        </div>
      )}

      {loading && (
        <div className="space-y-3 py-6">
          <p className="text-xs text-muted-foreground">Analyzing retinal image using {model}…</p>
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-6 animate-pulse rounded-md bg-muted" />
          ))}
        </div>
      )}

      {prediction && (
        <>
          <div className="rounded-xl border bg-card p-4">
            <div className="text-xs text-muted-foreground">Detected condition</div>
            <div className="mt-1 text-2xl font-bold">
              {prediction.predicted}
              {prediction.predicted === "Normal" ? (
                <span className="ml-2 rounded-full bg-[oklch(0.65_0.16_160/0.15)] px-2 py-0.5 text-xs font-medium text-[oklch(0.55_0.18_160)]">Healthy</span>
              ) : (
                <span className="ml-2 rounded-full bg-destructive/10 px-2 py-0.5 text-xs font-medium text-destructive">Abnormality</span>
              )}
            </div>
            {prediction.predicted === "Cataract" && (
              <div className="mt-2 flex items-start gap-1.5 rounded-md border border-[var(--warning)]/40 bg-[var(--warning)]/10 p-2 text-[11px] text-foreground">
                <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-[var(--warning)]" />
                Uploaded image type may not match selected disease model — cataract typically requires slit-lamp imagery.
              </div>
            )}
            <div className="mt-3 flex items-baseline justify-between">
              <span className="text-xs text-muted-foreground">Confidence</span>
              <span className="text-3xl font-bold text-gradient">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full bg-[var(--gradient-primary)]"
                style={{ width: `${prediction.confidence * 100}%` }}
              />
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
              <span>Model: <span className="font-mono font-semibold text-foreground">{prediction.model}</span></span>
              {prediction.inferenceMs != null && (
                <span>Inference: <span className="font-mono text-foreground">{prediction.inferenceMs.toFixed(1)} ms</span></span>
              )}
              <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${prediction.source === "live" ? "bg-[oklch(0.65_0.16_160/0.15)] text-[var(--success)]" : "bg-muted text-muted-foreground"}`}>
                {prediction.source === "live" ? "LIVE" : "DEMO"}
              </span>
            </div>
          </div>

          <div className="mt-5">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Class probabilities
            </div>
            <div className="space-y-2">
              {prediction.probabilities.map((p) => (
                <div key={p.disease}>
                  <div className="flex justify-between text-xs">
                    <span>{p.disease}</span>
                    <span className="font-mono">{(p.prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${p.prob * 100}%`,
                        background:
                          p.disease === prediction.predicted
                            ? "var(--gradient-primary)"
                            : "color-mix(in oklab, var(--muted-foreground) 50%, transparent)",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function HistoryPanel({ history }: { history: HistoryItem[] }) {
  return (
    <div className="rounded-2xl border bg-card p-5 shadow-[var(--shadow-card)]">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
        <History className="h-4 w-4" /> Upload History
      </h3>
      {history.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">No analyses yet.</p>
      ) : (
        <ul className="space-y-2">
          {history.map((h) => (
            <li key={h.id} className="flex items-center gap-3 rounded-lg border p-2">
              <img src={h.thumb} alt="" className="h-10 w-10 rounded object-cover" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{h.predicted}</div>
                <div className="truncate text-xs text-muted-foreground">
                  {h.model} · {(h.confidence * 100).toFixed(0)}% · {h.time}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function buildReportHTML({
  prediction,
  imageUrl,
  imageName,
  notes,
}: {
  prediction: ApiPrediction;
  imageUrl: string | null;
  imageName: string;
  notes: string;
}) {
  const rows = prediction.probabilities
    .map(
      (p: { disease: string; prob: number }) =>
        `<tr><td>${p.disease}</td><td style="text-align:right">${(p.prob * 100).toFixed(2)}%</td></tr>`,
    )
    .join("");
  return `<!doctype html><html><head><meta charset="utf-8"><title>EyeVision AI Report</title>
<style>
body{font-family:system-ui,sans-serif;margin:40px;color:#0f172a}
h1{color:#0e7490}h2{margin-top:24px;border-bottom:2px solid #ccfbf1;padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin-top:8px}td,th{padding:6px 8px;border-bottom:1px solid #e2e8f0}
.box{padding:16px;border:1px solid #e2e8f0;border-radius:8px;margin-top:8px}
img{max-width:280px;border-radius:8px}
</style></head><body>
<h1>EyeVision AI — Diagnostic Report</h1>
<p>Generated: ${new Date().toLocaleString()}</p>
<h2>Image</h2>
<p><strong>${imageName}</strong></p>
${imageUrl ? `<img src="${imageUrl}" alt="fundus" />` : ""}
<h2>Prediction</h2>
<div class="box">
  <p><strong>Condition:</strong> ${prediction.predicted}</p>
  <p><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(2)}%</p>
  <p><strong>Model:</strong> ${prediction.model}</p>
</div>
<h2>Class Probabilities</h2>
<table><thead><tr><th align="left">Class</th><th align="right">Probability</th></tr></thead><tbody>${rows}</tbody></table>
<h2>Doctor's Notes</h2>
<div class="box">${notes ? notes.replace(/</g, "&lt;") : "<em>No notes provided.</em>"}</div>
<p style="margin-top:32px;font-size:11px;color:#64748b">AI-assisted ophthalmology diagnosis system for education and research purposes only. Not a clinical diagnosis.</p>
</body></html>`;
}

function CompareResultsPanel({ results }: { results: ApiPrediction[] }) {
  const winner = results.reduce((a, b) => (a.confidence > b.confidence ? a : b));
  return (
    <div className="rounded-2xl border bg-card p-5 shadow-[var(--shadow-card)]">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          <Activity className="h-4 w-4" /> Multi-Model Comparison
        </h3>
        <span className="text-xs text-muted-foreground">
          Top: <span className="font-mono font-semibold text-foreground">{winner.model}</span>
        </span>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {results.map((r) => {
          const meta = MODELS.find((m) => m.id === (r.model as ModelId));
          const isTop = r.model === winner.model;
          return (
            <div
              key={r.model}
              className={`rounded-xl border p-4 ${isTop ? "border-primary bg-primary/5 shadow-[var(--shadow-glow)]" : "bg-muted/30"}`}
            >
              <div className="flex items-baseline justify-between">
                <span className="font-semibold">{r.model}</span>
                <span className={`rounded-full px-2 py-0.5 text-[9px] font-semibold ${r.source === "live" ? "bg-[oklch(0.65_0.16_160/0.15)] text-[var(--success)]" : "bg-muted text-muted-foreground"}`}>
                  {r.source.toUpperCase()}
                </span>
              </div>
              <div className="mt-2 text-sm">{r.predicted}</div>
              <div className="mt-1 text-2xl font-bold text-gradient">
                {(r.confidence * 100).toFixed(1)}%
              </div>
              <div className="mt-3 grid grid-cols-2 gap-1 text-[10px] font-mono text-muted-foreground">
                <span>Inference</span>
                <span className="text-right text-foreground">{r.inferenceMs?.toFixed(1) ?? "—"} ms</span>
                {meta && (
                  <>
                    <span>Params</span>
                    <span className="text-right text-foreground">{meta.paramsM}M</span>
                    <span>Val. Acc</span>
                    <span className="text-right text-foreground">{meta.accuracy}%</span>
                    <span>ROC-AUC</span>
                    <span className="text-right text-foreground">{meta.auc.toFixed(3)}</span>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}