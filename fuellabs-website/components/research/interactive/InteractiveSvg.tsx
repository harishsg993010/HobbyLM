"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import styles from "./InteractiveFigure.module.css";

const svgCache = new Map<string, string>();
const subscribe = () => () => undefined;

export function useHydrated() {
  return useSyncExternalStore(subscribe, () => true, () => false);
}

function namespaceSvgIds(source: string, namespace: string) {
  const ids = [...source.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
  const replacements = new Map(ids.map((id) => [id, `${namespace}-${id}`]));
  let svg = source;

  for (const [id, replacement] of replacements) {
    svg = svg
      .replaceAll(`id="${id}"`, `id="${replacement}"`)
      .replaceAll(`url(#${id})`, `url(#${replacement})`)
      .replaceAll(`href="#${id}"`, `href="#${replacement}"`)
      .replaceAll(`xlink:href="#${id}"`, `xlink:href="#${replacement}"`);
  }

  return svg.replace(/aria-labelledby="([^"]+)"/g, (_, labels: string) => {
    const namespaced = labels.split(/\s+/).map((id: string) => replacements.get(id) ?? id).join(" ");
    return `aria-labelledby="${namespaced}"`;
  });
}

export function InteractiveSvg({
  desktopSrc,
  mobileSrc,
  alt,
  namespace,
}: {
  desktopSrc: string;
  mobileSrc: string;
  alt: string;
  namespace: string;
}) {
  const [svg, setSvg] = useState<string | null>(null);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 40rem)");
    let cancelled = false;
    let controller: AbortController | null = null;

    const load = async () => {
      const src = media.matches ? mobileSrc : desktopSrc;
      const cached = svgCache.get(src);
      if (cached) {
        setSvg(namespaceSvgIds(cached, namespace));
        return;
      }

      controller?.abort();
      controller = new AbortController();
      try {
        const response = await fetch(src, { signal: controller.signal });
        if (!response.ok) throw new Error(`Unable to load ${src}`);
        const source = await response.text();
        svgCache.set(src, source);
        if (!cancelled) setSvg(namespaceSvgIds(source, namespace));
      } catch (error) {
        if (!cancelled && !(error instanceof DOMException && error.name === "AbortError")) setSvg(null);
      }
    };

    void load();
    media.addEventListener("change", load);
    return () => {
      cancelled = true;
      controller?.abort();
      media.removeEventListener("change", load);
    };
  }, [desktopSrc, mobileSrc, namespace]);

  if (!svg) {
    return (
      <picture className={styles.staticFallback}>
        <source media="(max-width: 40rem)" srcSet={mobileSrc} />
        <img src={desktopSrc} alt={alt} loading="lazy" decoding="async" />
      </picture>
    );
  }

  return <div className={styles.inlineSvg} dangerouslySetInnerHTML={{ __html: svg }} />;
}
