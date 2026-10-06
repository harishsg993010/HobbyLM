"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ReplayIcon } from "@/components/ui/ControlIcons";
import styles from "./MlxTerminal.module.css";

const session = [
  {
    text: "fuel ~ % git clone https://github.com/fuellabs/hobbyLM.git",
    prefix: "fuel ~ % ",
    tone: "command",
  },
  { text: "Cloning into 'hobbyLM'...", prefix: "", tone: "status" },
  {
    text: "fuel ~ % cd hobbyLM && hobbylm-mlx \\",
    prefix: "fuel ~ % ",
    tone: "command",
  },
  {
    text: '    --prompt "Explain sparse routing in one sentence."',
    prefix: "",
    tone: "continuation",
  },
  { text: "HobbyLM", tone: "speaker" },
  {
    text: "Sparse routing activates only the experts most relevant to each token, reducing computation while preserving model capacity.",
    tone: "answer",
  },
] as const;

const fullTranscript = session.map(({ text }) => text).join("\n");

function wait(milliseconds: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));
}

export function MlxTerminal() {
  const terminalRef = useRef<HTMLDivElement>(null);
  const runIdRef = useRef(0);
  const [lines, setLines] = useState<string[]>(() => session.map(({ text }) => text));
  const [running, setRunning] = useState(false);

  const runSession = useCallback(async () => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setLines(session.map(({ text }) => text));
      setRunning(false);
      return;
    }

    const runId = ++runIdRef.current;
    setRunning(true);
    setLines(session.map(() => ""));

    for (let lineIndex = 0; lineIndex < session.length; lineIndex += 1) {
      const { text, tone } = session[lineIndex];

      for (let characterIndex = 1; characterIndex <= text.length; characterIndex += 1) {
        if (runId !== runIdRef.current) return;

        while (document.hidden) {
          await wait(200);
          if (runId !== runIdRef.current) return;
        }

        setLines((current) => {
          const next = [...current];
          next[lineIndex] = text.slice(0, characterIndex);
          return next;
        });
        await wait(tone === "answer" ? 17 : 12);
      }

      await wait(tone === "status" || tone === "continuation" ? 360 : 180);
    }

    if (runId === runIdRef.current) setRunning(false);
  }, []);

  useEffect(() => {
    const terminal = terminalRef.current;
    if (!terminal) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        void runSession();
      },
      { threshold: 0.4 },
    );

    observer.observe(terminal);
    return () => {
      observer.disconnect();
      runIdRef.current += 1;
    };
  }, [runSession]);

  return (
    <div className={styles.terminal} ref={terminalRef} aria-label="Example HobbyLM MLX terminal session">
      <div className={styles.terminalBar}>
        <span className={styles.windowControls} aria-hidden="true"><i /><i /><i /></span>
        <span className={styles.terminalTitle}>Fuel — zsh</span>
        <button
          className={styles.replay}
          type="button"
          onClick={() => void runSession()}
          disabled={running}
          aria-label={running ? "Terminal animation running" : "Replay terminal animation"}
        >
          <ReplayIcon />
        </button>
      </div>

      <div className={styles.terminalBody} aria-hidden="true">
        {session.map(({ tone, ...line }, index) => (
          <p className={`${styles.line} ${styles[tone]}`} key={index}>
            {"prefix" in line && line.prefix ? (
              <>
                <span className={styles.shellPrompt}>{lines[index].slice(0, line.prefix.length)}</span>
                <span>{lines[index].slice(line.prefix.length)}</span>
              </>
            ) : lines[index]}
            {running && index === lines.findIndex((line, lineIndex) => line.length < session[lineIndex].text.length) ? (
              <span className={styles.cursor} />
            ) : null}
          </p>
        ))}
      </div>
      <pre className={styles.srOnly}>{fullTranscript}</pre>
    </div>
  );
}
