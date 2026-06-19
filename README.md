# HobbyLM

HobbyLM is a small language-model family I built from scratch on a hobby budget — a 500M-parameter
sparse Mixture-of-Experts model, the training code that produced it, a from-scratch Rust engine that
runs it on a laptop CPU, and a desktop app that wraps the whole thing. No cluster, no borrowed
weights: just FineWeb, a handful of Modal H100 hours, and a lot of ablations.

The goal was to see how far you can actually get at the ~500M scale if you sweat the architecture and
the systems work — and to own every layer of the stack end to end, from the optimizer to the GGUF
reader to the click-to-run app.

## The models

Every variant shares one 500M sparse-MoE core and is published on Hugging Face under
[`rootxhacker`](https://huggingface.co/rootxhacker):

- **HobbyLM-Base** — the pretrained foundation model (FineWeb).
- **HobbyLM-Chat** — instruction / conversation tuned.
- **HobbyLM-Computer-Use** — function calling + an accessibility-tree GUI agent.
- **HobbyLM-Omni** — multimodal: text + image + audio (TinyLLaVA-style, with vision and speech projectors).
- **HobbyLM-Diffusion** — a masked-diffusion (LLaDA-style) variant for parallel, bidirectional decoding.
- **HobbyLM-Image** — a separate 1024px text-to-image diffusion model (a DiT, not a language model).

Each model repo ships `safetensors` + `config.json`; the GGUF builds live in **HobbyLM-gguf**.

## Architecture, briefly

A decoder-only transformer with the modern small-MoE recipe, where each piece was chosen by ablation
rather than vibes:

- **Sparse MoE FFN** — 36 fine-grained experts, top-6 routed, plus one always-on shared expert; the
  first layer stays dense (DeepSeekMoE style).
- **Sigmoid gating** with DeepSeek-V3's **aux-loss-free** load balancing — a learned bias steers which
  experts get picked while the gate weights stay raw, so nothing fights the language-modeling objective.
- **GQA** attention with **per-head QK-norm** applied before RoPE, and a decoupled head dimension.
- Pre-norm **RMSNorm**, **RoPE**, tied embeddings, GPT-2 byte-level BPE.
- Trained with **Muon** (on the 2-D and per-expert 3-D matrices) and AdamW for the rest.

There are 130M / 500M / 1B presets; 500M is the one that got the full treatment. The design notes and
the ablations behind each decision are in [`docs/ARCHITECTURE_RESEARCH.md`](docs/ARCHITECTURE_RESEARCH.md).

## Running it — `hobby-rs`

`hobby-rs/` is a from-scratch CPU inference engine in Rust. It memory-maps a HobbyLM GGUF, runs the
sparse MoE natively (only the active experts are computed), and streams tokens — no llama.cpp, no
Python, no ONNX at runtime. The matmul hot path is hand-written AVX2/SIMD, and every hyperparameter is
read from the GGUF metadata, so nothing is hardcoded.

```bash
cd hobby-rs
cargo build --release
./target/release/hobby-rs --model HobbyLM-Chat.gguf --prompt "The capital of France is" --n 48
```

The GGUFs declare a custom `hobbylm` architecture. They load directly in `hobby-rs`; stock llama.cpp
would need the `hobbylm` arch registered first.

## Chatting with it — `hobby-chat`

`hobby-chat/` is a small Tauri desktop app (Rust backend + web UI): a local, ChatGPT-style window that
embeds `hobby-rs` along with the multimodal encoders, computer-use (Windows UI-Automation accessibility
tree), and an MCP client for tool use. Point it at a GGUF and everything runs on your machine.

## Training it — `train.py` + Modal

The training stack is plain PyTorch, run on [Modal](https://modal.com) serverless GPUs (1–8× H100):
FineWeb in, checkpoints out, with the full ablation suite a flag away.

```bash
python count_params.py --smoke                                   # local CPU sanity check
python -m modal run modal_train.py --action train --preset 500M  # train on Modal H100s
python -m modal run modal_train.py --action ablate --steps 3000  # run the architecture ablations
```

Multimodal, tool-use, diffusion, and image-generation training live in `modal_mm.py`, `modal_tools.py`,
`modal_dreamlite.py`, and the `dreamlite/` package.

## Layout

```
config.py model.py moe.py optim.py train.py   core MoE training
modal_train.py modal_mm.py modal_tools.py     Modal training harnesses
dreamlite/  modal_dreamlite.py                text-to-image diffusion model
to_gguf.py  modal_hobbylm.py                  GGUF / safetensors export + HF release
hobby-rs/                                     Rust CPU inference engine
hobby-rs-cli/                                 standalone CLI build of the engine
hobby-chat/                                   Tauri desktop app
docs/                                         architecture notes + plans
```

## Honest status

This is a research / hobby project at the 500M scale. It's genuinely fluent and the multimodal and
agent pieces work, but it carries the capability ceiling of a small model — it isn't meant to compete
with frontier systems. Weights aren't checked into the repo; grab them from Hugging Face.

## License

Apache-2.0.
