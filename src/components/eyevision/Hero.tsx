import { Activity, Brain, ShieldCheck } from "lucide-react";
import heroImg from "@/assets/retina-hero.jpg";

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden">
      <div
        className="absolute inset-0 -z-10"
        style={{ background: "var(--gradient-hero)" }}
      />
      <div className="absolute inset-0 -z-10 opacity-30 mix-blend-screen">
        <img src={heroImg} alt="" className="h-full w-full object-cover" />
      </div>
      <div className="absolute inset-0 -z-10 bg-gradient-to-b from-transparent via-transparent to-background" />

      <div className="container mx-auto px-4 py-20 md:py-28">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div className="text-primary-foreground">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1 text-xs backdrop-blur-md">
              <span className="h-1.5 w-1.5 rounded-full bg-[oklch(0.8_0.18_160)] animate-pulse" />
              Deep Learning · SwinTransformer · ConvNeXt · EfficientNetV2
            </div>
            <h1 className="text-4xl font-bold leading-[1.05] tracking-tight md:text-6xl">
              AI Eye Disease<br />
              <span className="bg-gradient-to-r from-[oklch(0.85_0.16_195)] to-[oklch(0.95_0.05_200)] bg-clip-text text-transparent">
                Detection System
              </span>
            </h1>
            <p className="mt-5 max-w-xl text-lg text-white/80">
              Deep Learning-based Ophthalmology Assistant. Upload a retinal fundus or optic
              nerve image and receive instant CNN-powered diagnostic insight with
              explainable Grad-CAM visualization.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a
                href="#analyze"
                className="inline-flex items-center justify-center rounded-md bg-white px-6 py-3 text-sm font-semibold text-[oklch(0.25_0.08_235)] shadow-[var(--shadow-glow)] transition-transform hover:scale-[1.02]"
              >
                Start Analysis
              </a>
              <a
                href="#comparison"
                className="inline-flex items-center justify-center rounded-md border border-white/25 bg-white/5 px-6 py-3 text-sm font-medium text-white backdrop-blur-md hover:bg-white/10"
              >
                Compare Models
              </a>
            </div>

            <div className="mt-10 grid max-w-md grid-cols-3 gap-4 text-center">
              {[
                { Icon: Brain, label: "3 CNN Models" },
                { Icon: Activity, label: "5 Conditions" },
                { Icon: ShieldCheck, label: "Grad-CAM XAI" },
              ].map(({ Icon, label }) => (
                <div
                  key={label}
                  className="rounded-lg border border-white/15 bg-white/5 px-2 py-3 backdrop-blur-md"
                >
                  <Icon className="mx-auto mb-1 h-5 w-5 text-[oklch(0.85_0.16_195)]" />
                  <div className="text-xs text-white/85">{label}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="relative mx-auto aspect-square w-full max-w-md">
            <div className="absolute inset-0 animate-pulse-ring rounded-full" />
            <div className="absolute inset-4 overflow-hidden rounded-full border-2 border-white/30 shadow-[var(--shadow-glow)]">
              <img
                src={heroImg}
                alt="Retinal fundus scan with AI overlay"
                className="h-full w-full object-cover"
                width={1920}
                height={1080}
              />
              <div className="pointer-events-none absolute inset-x-0 h-[20%] animate-scan bg-gradient-to-b from-transparent via-[oklch(0.85_0.18_195)]/40 to-transparent" />
            </div>
            <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 rounded-full border border-white/20 bg-black/40 px-4 py-1.5 text-xs text-white/90 backdrop-blur-md">
              Live retinal analysis
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}