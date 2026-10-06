"use client";

import { useState } from "react";
import type { KeyboardEvent } from "react";
import type { PretrainingFigure } from "@/content/pretrainingFigures";
import { NextIcon, PreviousIcon } from "@/components/ui/ControlIcons";
import { InteractiveSvg, useHydrated } from "./InteractiveSvg";
import styles from "./InteractiveFigure.module.css";

const order = ["a", "b", "c"] as const;
type ArchitectureStep = (typeof order)[number];

const steps: Record<ArchitectureStep, { label: string; description: string }> = {
  a: {
    label: "1 · 20-layer stack",
    description: "HobbyLM has 20 decoder layers. Layer 00 uses a dense feed-forward block; layers 01–19 use MoE blocks.",
  },
  b: {
    label: "2 · One MoE layer",
    description: "One MoE layer lifts out. It holds 64 routed experts and one shared expert; every MoE layer has the same structure.",
  },
  c: {
    label: "3 · One token route",
    description: "For one token, the router selects 8 of the 64 routed experts. The shared expert runs on every token, and both outputs are added to the residual stream.",
  },
};

export function ArchitectureFigure({ figure }: { figure: PretrainingFigure }) {
  const hydrated = useHydrated();
  const [step, setStep] = useState<ArchitectureStep>("c");
  const index = order.indexOf(step);

  function move(nextIndex: number) {
    setStep(order[Math.max(0, Math.min(order.length - 1, nextIndex))]);
  }

  function handleKeys(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "ArrowRight") {
      move(index + 1);
      event.preventDefault();
    }
    if (event.key === "ArrowLeft") {
      move(index - 1);
      event.preventDefault();
    }
  }

  return (
    <div className={styles.interactiveFigure}>
      <div className={styles.architecture} data-state={step}>
        <InteractiveSvg desktopSrc={figure.desktopSrc} mobileSrc={figure.mobileSrc} alt={figure.alt} namespace="figure-03" />
      </div>
      {hydrated && <>
        <div className={styles.controls} role="group" aria-label="Architecture steps" onKeyDown={handleKeys}>
          {order.map((key) => (
            <button type="button" key={key} aria-pressed={step === key} onClick={() => setStep(key)}>{steps[key].label}</button>
          ))}
          <span className={styles.separator} aria-hidden="true" />
          <button className={styles.iconButton} type="button" onClick={() => move(index - 1)} disabled={index === 0} aria-label="Previous architecture step"><PreviousIcon /></button>
          <button className={styles.iconButton} type="button" onClick={() => move(index + 1)} disabled={index === order.length - 1} aria-label="Next architecture step"><NextIcon /></button>
        </div>
        <p className={styles.status} aria-live="polite">{steps[step].description}</p>
      </>}
    </div>
  );
}
