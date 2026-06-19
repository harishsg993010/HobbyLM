# hobby-chat

A local, ChatGPT-style **desktop app** (Tauri) for the 500M MoE — the `hobby-rs` engine embedded in a
Rust backend with a vanilla HTML/CSS/JS frontend. No server, no cloud: the model runs on your CPU.

## Features

- **ChatGPT-style UI** — dark theme, streaming token-by-token responses, user/assistant bubbles, autosizing
  composer, "New chat".
- **Local inference** — loads a `hobbylm` GGUF (F32 or Q4_K_M) as Q8 (~0.56 GB RAM, ~40 tok/s) via `hobby-rs`.
- **Function calling (Tools toggle)** — flip *Tools* on and the model is forced to emit a tool call
  (`get_weather`, `calculate`); the app executes it, shows a tool card, feeds the result back, and the model
  summarizes. This implements the model's trained agent loop (forced-call to beat its over-abstention).

## Run

Prereqs: Rust, WebView2 runtime (Windows), and `tauri-cli` (`cargo install tauri-cli --version "^2"`).

```powershell
cd hobby-chat/src-tauri
cargo tauri dev          # dev window
# or a distributable build + installer:
cargo tauri build
```

In the app: paste the path to the GGUF (e.g. `..\joint12-hobbylm.gguf` or the 362 MB
`joint12-Q4_K_M.gguf`), click **Load**, then chat. Toggle **Tools** and ask e.g. *"what's the weather in Tokyo?"*
or *"what is (2+3)*4?"* to see the function-calling loop.

## How it works

- `src-tauri/src/main.rs` — Tauri commands: `load_model` (builds a `hobby_rs::Engine`, stored in app state) and
  `chat` (spawns a worker thread that streams `token` events; for Tools, runs the
  prompt→force-call→execute→summarize loop and emits `tool_call`/`tool_result`).
- `src-tauri/src/tools.rs` — the tool schema, a lenient `{"name":..,"arguments":..}` parser, and executors.
- `ui/` — the frontend; `app.js` calls `invoke()` and listens for the streamed events.

The SigLIP/Whisper encoders aren't embedded, so this app is text + tools; image/speech run via the CLI's
`--image`/`--speech` (precomputed embeddings) in `../hobby-rs`.
