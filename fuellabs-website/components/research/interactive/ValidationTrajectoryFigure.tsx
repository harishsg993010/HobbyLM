"use client";

import { useEffect, useId, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import type { PretrainingFigure } from "@/content/pretrainingFigures";
import { NextIcon, PreviousIcon } from "@/components/ui/ControlIcons";
import { finalValidation, validationPoints } from "@/content/pretrainingValidation";
import { useHydrated } from "./InteractiveSvg";
import styles from "./InteractiveFigure.module.css";

const charts = {
  desktop: {
    viewBox: "0 0 864 453.6",
    width: 864,
    height: 453.6,
    plot: { x: 60.48, y: 54.432, width: 768.96, height: 281.232 },
  },
  mobile: {
    viewBox: "0 0 302.4 410.4",
    width: 302.4,
    height: 410.4,
    plot: { x: 45.36, y: 49.248, width: 226.8, height: 229.824 },
  },
} as const;

export function ValidationTrajectoryFigure({ figure }: { figure: PretrainingFigure }) {
  const hydrated = useHydrated();
  const sliderId = useId();
  const [index, setIndex] = useState(0);
  const [isFinal, setIsFinal] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const point = validationPoints[index];
  const chart = isMobile ? charts.mobile : charts.desktop;
  const x = chart.plot.x + (point.step / 81000) * chart.plot.width;
  const y = chart.plot.y + ((5 - point.loss) / 1.8) * chart.plot.height;
  const tooltipX = Math.max(chart.plot.x, Math.min(x + 12, chart.plot.x + chart.plot.width - 146));
  const tooltipY = Math.max(chart.plot.y + 4, Math.min(y - 56, chart.plot.y + chart.plot.height - 48));

  useEffect(() => {
    const media = window.matchMedia("(max-width: 40rem)");
    const update = () => setIsMobile(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  function select(nextIndex: number) {
    setIndex(Math.max(0, Math.min(validationPoints.length - 1, nextIndex)));
    setIsFinal(false);
  }

  function inspectPointer(event: PointerEvent<HTMLDivElement>) {
    if (event.pointerType === "touch") return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const pointerX = ((event.clientX - bounds.left) / bounds.width) * chart.width;
    if (pointerX < chart.plot.x || pointerX > chart.plot.x + chart.plot.width) return;
    const step = ((pointerX - chart.plot.x) / chart.plot.width) * 81000;
    select(Math.round(step / 250) - 1);
  }

  function inspectKeys(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "ArrowLeft") {
      select(index - 1);
      event.preventDefault();
    }
    if (event.key === "ArrowRight") {
      select(index + 1);
      event.preventDefault();
    }
    if (event.key === "Home") {
      select(0);
      event.preventDefault();
    }
    if (event.key === "End") {
      select(validationPoints.length - 1);
      event.preventDefault();
    }
  }

  const readout = isFinal
    ? `Final validation · ${finalValidation.loss.toFixed(4)} · after logged step ${finalValidation.lastPrecedingLoggedStep.toLocaleString("en-US")}`
    : `Step ${point.step.toLocaleString("en-US")} · loss ${point.loss.toFixed(4)} · ${index + 1} / ${validationPoints.length}`;

  return (
    <div className={styles.interactiveFigure}>
      <div
        className={styles.validationChart}
        onPointerMove={inspectPointer}
        onKeyDown={inspectKeys}
        tabIndex={0}
        role="group"
        aria-label="Validation trajectory. Use Left and Right Arrow keys to inspect recorded points."
      >
        <picture className={styles.chartPicture}>
          <source media="(max-width: 40rem)" srcSet={figure.mobileSrc} />
          <img src={figure.desktopSrc} alt={figure.alt} loading="lazy" decoding="async" />
        </picture>
        {hydrated && !isFinal && (
          <svg className={styles.chartOverlay} viewBox={chart.viewBox} aria-hidden="true">
            <line x1={x} x2={x} y1={chart.plot.y} y2={chart.plot.y + chart.plot.height} className={styles.crosshair} />
            <circle cx={x} cy={y} r="3" className={styles.point} />
            <g transform={`translate(${tooltipX} ${tooltipY})`}>
              <rect width="146" height="44" className={styles.tooltipBackground} />
              <text x="10" y="17" className={styles.tooltipText}>Step {point.step.toLocaleString("en-US")}</text>
              <text x="10" y="33" className={styles.tooltipText}>Loss {point.loss.toFixed(4)}</text>
            </g>
          </svg>
        )}
      </div>

      {hydrated && <>
        <div className={styles.inspector}>
          <output className={styles.readout} aria-live="polite">{readout}</output>
          <div className={styles.controls} role="group" aria-label="Validation point controls">
            <button className={styles.iconButton} type="button" onClick={() => select(isFinal ? validationPoints.length - 1 : index - 1)} disabled={!isFinal && index === 0} aria-label="Previous recorded point"><PreviousIcon /></button>
            <button className={styles.iconButton} type="button" onClick={() => select(isFinal ? validationPoints.length - 1 : index + 1)} disabled={!isFinal && index === validationPoints.length - 1} aria-label="Next recorded point"><NextIcon /></button>
            <button type="button" onClick={() => setIsFinal(true)} aria-pressed={isFinal}>Final validation pass</button>
          </div>
        </div>

        <label className={styles.sliderLabel} htmlFor={sliderId}>Inspect a recorded step</label>
        <input
          className={styles.slider}
          id={sliderId}
          type="range"
          min="0"
          max={validationPoints.length - 1}
          value={index}
          step="1"
          aria-valuetext={`Step ${point.step}, validation loss ${point.loss.toFixed(4)}`}
          onChange={(event) => select(Number(event.currentTarget.value))}
        />
        <p className={styles.help}>Move across the chart or use the slider. Arrow keys select adjacent recorded points. The line joins samples; intermediate values are not measurements.</p>
      </>}
    </div>
  );
}
