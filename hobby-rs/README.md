# hobby-rs

A from-scratch, dependency-light **CPU inference engine in Rust** for our 500M sparse-MoE LLM
(`hobbylm` arch). It loads the F32 GGUF directly and generates text — no llama.cpp, no Python at runtime.

## What it implements

- **GGUF v3 reader** over an mmap (`gguf.rs`) — metadata + tensor directory, zero-copy F32 tensor views.
  Every hyperparameter is read from the file (the joint12 model uses `rope_theta=1e6`, 8k ctx).
- **GPT-2 byte-level BPE** (`tokenizer.rs`) — vocab + merges straight from the GGUF, GPT-2 pretok regex.
- **The exact forward pass** (`model.rs`): pre-norm residual stream; GQA attention with a **decoupled
  head_dim of 128** (Q=1536, KV=384), **per-head QK-norm before RoPE** (rotate-half), `1/√128` scale,
  causal KV cache; dense SwiGLU FFN on block 0.
- **Native sparse MoE** (`model.rs::ffn`) — the centerpiece graph-exporters can't do: sigmoid router,
  aux-free selection bias (`sel = scores + bias`), top-6 of 36 experts weighted by **raw scores** (no renorm),
  plus an always-on shared expert. Only the ~7 active experts run per token.
- f32 kernels (`ops.rs`) with rayon-parallel matmul; greedy / temperature / top-p sampling (`sample.rs`).

## Build & run

```powershell
cargo build --release
.\target\release\hobby-rs.exe --model ..\joint12-hobbylm.gguf --prompt "The capital of France is" --n 48 --temp 0
```

Flags: `--model <gguf>` `--prompt <s>` `--n <int>` `--temp <f>` (`0`=greedy) `--top-p <f>` `--seed <u64>` `--threads <n>`.

Get the F32 GGUF (~2 GB) from HF `rootxhacker/moe-omni-500m` (`export/joint12-hobbylm.gguf`) or the
Modal volume `fineweb10B:runs/500M_vlm_joint12/export/joint12-hobbylm.gguf`.

## Quantization

Reads **F32 / F16 / Q8_0 / Q5_0 / Q4_K / Q6_K** GGUFs (`--info` dumps the type histogram). `--quant q8`
(default) quantizes the big matmul weights to Q8_0 at load and runs int8×int8 matmul (router + norms stay F32);
`--quant f32` is exact. On a 12-core CPU: F32 ~8 tok/s / 2 GB → **Q8 ~40 tok/s / 0.56 GB, output identical**.
The 362 MB Q4_K_M file loads too (dequant → Q8 runtime), coherent with mild 4-bit quality loss.

## Multimodal (image / speech / audio)

The engine splices **precomputed encoder embeddings** at the marker position (inputs_embeds), exactly as the
model was trained. The SigLIP2 / Whisper encoders run in PyTorch (`modal_mm.py --action dump_embeds` →
`image_embeds.bin` 1024×768, `speech_embeds.bin` 750×768); the engine consumes the raw f32 `.bin`:

```powershell
hobby-rs.exe --model joint12-hobbylm.gguf --image image_embeds.bin  --prompt "USER: Describe this image.`nASSISTANT:"
hobby-rs.exe --model joint12-hobbylm.gguf --speech speech_embeds.bin --prompt "USER: What is being said?`nASSISTANT:"
```

Verified: cats photo → "two cats lying on a pink couch…"; jfk.wav → "your fellow citizens are being asked what
your country can do for you." (Using PyTorch embeds sidesteps the clip.cpp audio-downsample bug.)

## Status / TODO

Fully omni (text + image + speech), text verified token-for-token vs llama.cpp. Known wart: multimodal
**prefill is sequential** (~50 s for 1024 image positions) — a batched prefill GEMM is the main perf TODO.
Also future: in-Rust encoders, CLAP `--audio` embed dumping, SIMD micro-opt.
