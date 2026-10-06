import type { PretrainingFigure } from "@/content/pretrainingFigures";
import { ArchitectureFigure } from "./interactive/ArchitectureFigure";
import { DataDeliveryFigure } from "./interactive/DataDeliveryFigure";
import { ValidationTrajectoryFigure } from "./interactive/ValidationTrajectoryFigure";
import styles from "./ArticleFigure.module.css";

export function ArticleFigure({ figure }: { figure: PretrainingFigure }) {
  const media = figure.id === "pretraining-data-delivery"
    ? <DataDeliveryFigure figure={figure} />
    : figure.id === "pretraining-architecture"
      ? <ArchitectureFigure figure={figure} />
      : figure.id === "pretraining-validation"
        ? <ValidationTrajectoryFigure figure={figure} />
        : (
          <picture>
            <source media="(max-width: 40rem)" srcSet={figure.mobileSrc} />
            {/* The source SVGs contain their own accessible title and description. */}
            <img src={figure.desktopSrc} alt={figure.alt} loading="lazy" decoding="async" />
          </picture>
        );

  return (
    <figure className={styles.figure} id={figure.id}>
      {media}
      <figcaption>
        <span>Figure {figure.publishedNumber}</span>
        {figure.caption}
      </figcaption>
    </figure>
  );
}
