import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Header } from "@/components/eyevision/Header";
import { Hero } from "@/components/eyevision/Hero";
import { Analyze } from "@/components/eyevision/Analyze";
import { Workflow } from "@/components/eyevision/Workflow";
import { Comparison } from "@/components/eyevision/Comparison";
import { Dashboard } from "@/components/eyevision/Dashboard";
import { Footer } from "@/components/eyevision/Footer";
import { Toaster } from "@/components/ui/sonner";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "EyeVision AI — Deep Learning Ophthalmology Assistant" },
      {
        name: "description",
        content:
          "AI-powered detection of AMD, glaucoma and diabetic retinopathy from retinal fundus images using a deterministic SwinTransformer + ConvNeXt + EfficientNetV2 ensemble.",
      },
      { property: "og:title", content: "EyeVision AI — Ophthalmology Assistant" },
      { property: "og:description", content: "Deep learning-based detection of eye diseases from retinal fundus images." },
    ],
  }),
  component: Index,
});

function Index() {
  const [dark, setDark] = useState(true);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header dark={dark} onToggleDark={() => setDark((d) => !d)} />
      <main>
        <Hero />
        <Analyze />
        <Workflow />
        <Comparison />
        <Dashboard />
      </main>
      <Footer />
      <Toaster />
    </div>
  );
}
