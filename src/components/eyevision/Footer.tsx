import { Eye } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t bg-muted/30">
      <div className="container mx-auto grid gap-8 px-4 py-12 md:grid-cols-3">
        <div>
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--gradient-primary)] shadow-[var(--shadow-glow)]">
              <Eye className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-bold">EyeVision <span className="text-gradient">AI</span></span>
          </div>
          <p className="mt-3 max-w-xs text-sm text-muted-foreground">
            AI-assisted ophthalmology diagnosis system for education and research purposes only.
          </p>
        </div>
        <div>
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Datasets</h4>
          <ul className="space-y-1 text-sm">
            <li>EyePACS</li><li>DRIVE</li><li>REFUGE</li><li>Messidor</li>
          </ul>
        </div>
        <div>
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Deploy</h4>
          <ul className="space-y-1 text-sm">
            <li>HuggingFace Spaces</li><li>Render</li><li>Microsoft Azure</li><li>Streamlit Cloud</li>
          </ul>
        </div>
      </div>
      <div className="border-t py-4 text-center text-xs text-muted-foreground">
        © {new Date().getFullYear()} EyeVision AI · Not for clinical diagnosis · Research use only
      </div>
    </footer>
  );
}