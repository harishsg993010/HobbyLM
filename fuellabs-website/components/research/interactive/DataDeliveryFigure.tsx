"use client";

import { useEffect, useState } from "react";
import type { PretrainingFigure } from "@/content/pretrainingFigures";
import { NextIcon, PauseIcon, PlayIcon, PreviousIcon, ReplayIcon } from "@/components/ui/ControlIcons";
import { InteractiveSvg, useHydrated } from "./InteractiveSvg";
import styles from "./InteractiveFigure.module.css";

const descriptions = [
  "Before the explanation: the same training-sequence axis, empty.",
  "Step 1 of 4. In the proxy run, source families reached the model one after another.",
  "Step 2 of 4. Proxy evaluation exposed the mismatch: training loss kept falling while broader evaluation moved the wrong way.",
  "Step 3 of 4. Delivery was corrected and the proxy experiment repeated. The implementation is internal.",
  "Step 4 of 4. Before scaling, families were verified to reach the loader across the whole sequence. Proportions and ordering are not literal.",
] as const;

const hold = [2400, 3000, 2600, 2400] as const;

export function DataDeliveryFigure({ figure }: { figure: PretrainingFigure }) {
  const hydrated = useHydrated();
  const [step, setStep] = useState(4);
  const [playing, setPlaying] = useState(false);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    if (!playing) return;
    if (step >= 4) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const delay = step === 0 && reduceMotion ? 600 : step === 0 ? 250 : hold[step - 1];
    const timer = window.setTimeout(() => {
      const nextStep = step + 1;
      setStep(nextStep);
      if (nextStep === 4) setPlaying(false);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [playing, step]);

  const playLabel = playing ? "Pause explanation" : step === 4 && started ? "Replay explanation" : step < 4 && started ? "Resume explanation" : "Play explanation";

  function togglePlay() {
    if (playing) {
      setPlaying(false);
      return;
    }
    setStarted(true);
    if (step === 4) setStep(0);
    setPlaying(true);
  }

  function move(nextStep: number) {
    setPlaying(false);
    setStarted(true);
    setStep(Math.max(0, Math.min(4, nextStep)));
  }

  return (
    <div className={styles.interactiveFigure}>
      <div className={styles.dataDelivery} data-step={step}>
        <InteractiveSvg desktopSrc={figure.desktopSrc} mobileSrc={figure.mobileSrc} alt={figure.alt} namespace="figure-02" />
      </div>
      {hydrated && <>
        <div className={styles.controls} role="group" aria-label="Data-delivery explanation controls">
          <button className={styles.iconButton} type="button" onClick={togglePlay} aria-label={playLabel} aria-pressed={playing}>
            {playing ? <PauseIcon /> : step === 4 && started ? <ReplayIcon /> : <PlayIcon />}
          </button>
          <button className={styles.iconButton} type="button" onClick={() => move(step - 1)} disabled={step === 0} aria-label="Previous step"><PreviousIcon /></button>
          <button className={styles.iconButton} type="button" onClick={() => move(step + 1)} disabled={step === 4} aria-label="Next step"><NextIcon /></button>
        </div>
        <p className={styles.status} aria-live="polite">{descriptions[step]}</p>
      </>}
    </div>
  );
}
