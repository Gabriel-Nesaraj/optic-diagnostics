import { Moon, Sun, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  dark: boolean;
  onToggleDark: () => void;
}

export function Header({ dark, onToggleDark }: Props) {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b">
      <div className="container mx-auto flex items-center justify-between px-4 py-3">
        <a href="#top" className="flex items-center gap-2.5">
          <div className="relative flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--gradient-primary)] shadow-[var(--shadow-glow)]">
            <Eye className="h-5 w-5 text-primary-foreground" />
          </div>
          <div className="leading-tight">
            <div className="text-base font-bold tracking-tight">EyeVision <span className="text-gradient">AI</span></div>
            <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Ophthalmology Suite</div>
          </div>
        </a>
        <nav className="hidden items-center gap-6 text-sm text-muted-foreground md:flex">
          <a href="#analyze" className="hover:text-foreground">Analyze</a>
          <a href="#workflow" className="hover:text-foreground">Workflow</a>
          <a href="#comparison" className="hover:text-foreground">Models</a>
          <a href="#dashboard" className="hover:text-foreground">Research</a>
        </nav>
        <Button variant="ghost" size="icon" onClick={onToggleDark} aria-label="Toggle theme">
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  );
}