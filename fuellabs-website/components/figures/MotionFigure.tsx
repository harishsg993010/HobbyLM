"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { PauseIcon, PlayIcon } from "@/components/ui/ControlIcons";
import styles from "./MotionFigure.module.css";

type MotionFigureProps = {
  animatedSrc: string;
  stillSrc: string;
  width: number;
  height: number;
  alt: string;
  caption: string;
  format?: "image" | "video";
};

export function MotionFigure({
  animatedSrc,
  stillSrc,
  width,
  height,
  alt,
  caption,
  format = "image",
}: MotionFigureProps) {
  const figureRef = useRef<HTMLElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [userPaused, setUserPaused] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(true);
  const [inView, setInView] = useState(true);
  const [documentVisible, setDocumentVisible] = useState(true);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const syncPreference = () => setReducedMotion(media.matches);
    const syncVisibility = () => setDocumentVisible(document.visibilityState === "visible");
    const observer = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), { rootMargin: "10%" });

    syncPreference();
    syncVisibility();
    if (figureRef.current) observer.observe(figureRef.current);
    media.addEventListener("change", syncPreference);
    document.addEventListener("visibilitychange", syncVisibility);

    return () => {
      observer.disconnect();
      media.removeEventListener("change", syncPreference);
      document.removeEventListener("visibilitychange", syncVisibility);
    };
  }, []);

  const isPlaying = !userPaused && !reducedMotion && inView && documentVisible;

  useEffect(() => {
    if (format !== "video" || !videoRef.current) return;

    if (isPlaying) {
      void videoRef.current.play().catch(() => undefined);
    } else {
      videoRef.current.pause();
    }
  }, [format, isPlaying]);

  return (
    <figure className={styles.figure} ref={figureRef}>
      <div className={styles.media}>
        {format === "video" ? (
          <video
            ref={videoRef}
            width={width}
            height={height}
            poster={stillSrc}
            muted
            loop
            playsInline
            preload="metadata"
            aria-label={alt}
          >
            <source src={animatedSrc} type="video/mp4" />
          </video>
        ) : (
          <Image
            unoptimized
            src={isPlaying ? animatedSrc : stillSrc}
            width={width}
            height={height}
            sizes="(max-width: 800px) 100vw, 58vw"
            alt={alt}
          />
        )}
      </div>
      <div className={styles.meta}>
        <figcaption>{caption}</figcaption>
        {!reducedMotion ? (
          <button
            type="button"
            aria-label={userPaused ? "Play motion" : "Pause motion"}
            aria-pressed={userPaused}
            onClick={() => setUserPaused((paused) => !paused)}
          >
            {userPaused ? <PlayIcon /> : <PauseIcon />}
          </button>
        ) : null}
      </div>
    </figure>
  );
}
