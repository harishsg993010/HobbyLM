#!/usr/bin/env python3
"""Tiny interactive wrapper around the remote Modal generation path.

This is intentionally simple:
- it talks to the checkpoint that already lives on Modal
- it keeps a lightweight local chat history
- it is for experimentation with a base model, not a polished assistant

Example:
  /private/tmp/HobbyLM/.venv/bin/python scripts/modal_base_chat.py \
      --run-name 30M_study_1000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODAL_TRAIN = REPO_ROOT / "training" / "modal_train.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def build_prompt(history: list[tuple[str, str]], user_msg: str) -> str:
    parts: list[str] = []
    for user, assistant in history:
        parts.append(f"User: {user}\nAssistant: {assistant}")
    parts.append(f"User: {user_msg}\nAssistant:")
    return "\n\n".join(parts)


def extract_completion(stdout: str, prompt: str) -> str:
    marker = "PROMPT:"
    if marker not in stdout:
        return stdout.strip()
    lines = stdout.splitlines()
    capture = False
    payload: list[str] = []
    for line in lines:
        if line.startswith("PROMPT:"):
            capture = True
            continue
        if capture:
            if line.startswith("Stopping app"):
                break
            payload.append(line)
    text = "\n".join(payload).strip()
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()
    return text.strip()


def run_remote_generate(args, prompt: str) -> str:
    cmd = [
        str(PYTHON),
        "-m",
        "modal",
        "run",
        str(MODAL_TRAIN),
        "--action",
        "generate",
        "--run-name",
        args.run_name,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "remote generation failed")
    return extract_completion(proc.stdout, prompt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="30M_study_1000")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    history: list[tuple[str, str]] = []
    print(f"Remote base-model chat using Modal run '{args.run_name}'.")
    print("Type '/quit' to exit, '/reset' to clear history.")
    print("This is a base model, so expect imperfect and sometimes strange replies.\n")
    while True:
        try:
            user_msg = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_msg:
            continue
        if user_msg == "/quit":
            return 0
        if user_msg == "/reset":
            history.clear()
            print("History cleared.\n")
            continue
        prompt = build_prompt(history, user_msg)
        try:
            reply = run_remote_generate(args, prompt)
        except Exception as exc:
            print(f"Model> [error] {exc}\n")
            continue
        history.append((user_msg, reply))
        print(f"Model> {reply}\n")


if __name__ == "__main__":
    sys.exit(main())
