"use client";

import { useEffect, useRef, useState } from "react";
import { Wordmark } from "@/components/brand/Wordmark";
import styles from "./CurtainReveal.module.css";

export function CurtainReveal() {
  const [state, setState] = useState<"enter" | "leave" | "done">(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "done" : "enter");
  const skipRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    skipRef.current?.focus();
    const leave = window.setTimeout(() => setState("leave"), 1800);
    const done = window.setTimeout(() => finish(), 2800);
    return () => { window.clearTimeout(leave); window.clearTimeout(done); };
  }, []);
  function finish() {
    const restoreFocus = document.activeElement === skipRef.current;
    setState("done");
    if (restoreFocus) window.requestAnimationFrame(() => document.querySelector<HTMLElement>("#main")?.focus());
  }
  if (state === "done") return null;
  return <div className={`${styles.curtain} ${state === "leave" ? styles.leave : ""}`} role="dialog" aria-modal="true" aria-label="fuellabs. introduction" onKeyDown={(event) => { if (event.key === "Tab") { event.preventDefault(); skipRef.current?.focus(); } }}><div className={styles.logo}><Wordmark large tabIndex={-1} /></div><button ref={skipRef} className={styles.skip} type="button" onClick={finish}>Skip introduction</button></div>;
}
