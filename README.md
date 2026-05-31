# moe-lab

Clean, modular **Mixture-of-Experts** LLM codebase for building efficient models at
**130M / 500M / 1B** total params, with ablation knobs as config flags. Trains on
**Modal** GPUs (1–8× H100). Design rationale: [`docs/ARCHITECTURE_RESEARCH.md`](docs/ARCHITECTURE_RESEARCH.md).

## Architecture
Decoder-only transformer, pre-norm RMSNorm + **QK-norm**, **GQA** attention, **RoPE**, tied embeddings.
FFN = **fine-grained SwiGLU MoE** (first layer dense): dropless token-choice top-k routing via
`torch._grouped_mm`, **fp32 router**, DeepSeek-V3 **aux-loss-free bias** balancing + router z-loss.
Optimizer: **Muon** (hidden + per-expert 3D matrices, batched Newton-Schulz) + **AdamW**
(router, embeddings, head, norms). Mixed precision = fp32 master weights + bf16 autocast.

## Files
| file | role |
|---|---|
| `config.py` | `ModelConfig` (+ ablation flags), `TrainConfig`, `PRESETS` (130M/500M/1B) |
| `moe.py` | MoE layer: router, balancing, dropless expert compute (grouped/bmm/loop backends) |
| `model.py` | attention (GQA+QK-norm+RoPE), dense FFN, blocks, GPT wrapper, loss, param counter |
| `optim.py` | Muon (2D + batched 3D) + AdamW split |
| `data.py` | FineWeb `.bin` loader (modded-nanogpt format), DDP-sharded |
| `train.py` | training loop: LR schedule, grad accum, bias anneal, eval. Single-GPU or `torchrun` DDP |
| `count_params.py` | param targets + local CPU smoke test (`--smoke`) |
| `modal_train.py` | Modal harness: download data, GPU smoke, train, ablation suite |

## Usage
```bash
# local (CPU) sanity check — param counts + fwd/bwd
python count_params.py --smoke

# Modal: one-time data download (~1B tokens)
python -m modal run modal_train.py --action download --chunks 10

# Modal: GPU smoke (grouped_mm + Muon + compile)
python -m modal run modal_train.py --action smoke

# train one preset (1x H100)
python -m modal run modal_train.py --action train --preset 130M --steps 4000 --run-name baseline
# 8x H100
python -m modal run modal_train.py --action train --preset 1B --gpus 8 --steps 20000

# run the focused ablation suite (10 runs at 130M, in parallel)
python -m modal run modal_train.py --action ablate --steps 3000
python -m modal run modal_train.py --action results   # leaderboard by final val loss

# train with throughput optimizations (see below) — fused_ce is the win
python -m modal run modal_train.py --action train --preset 1B --gpus 8 --opts fused_ce --micro 32
```

## Ablations (each changes ONE thing vs the 130M baseline)
`softmax` gating · `aux_loss` (classic balance) · `shared1` (shared expert) · `no_qknorm` ·
`no_renorm` (top-k gate renorm) · `topk8` · `experts16` · `no_zloss` · `scale_emb`.
Winners get promoted into the 500M/1B configs.

## Throughput optimizations (nanogpt-inspired)
Flag-gated speedups; see `docs/ARCHITECTURE_RESEARCH.md` §8. Always-on (numerics-identical):
on-device loss accumulation (no per-micro sync), `expandable_segments` allocator, CUDA data
prefetch. Opt-in via `--opts`:
- **`fused_ce`** ✅ **the win** — chunked, checkpointed cross-entropy; never materializes the
  `(T,vocab)` fp32 logit tensor. Bit-identical loss, **+6% step time, −21% peak memory**, enables a
  ~2× larger batch (baseline OOMs where fused_ce fits). *Recommended for the 1B run.*
- **`polar`** — Polar Express orthogonalizer in Muon. Measured a wash at our scale (no speed/quality
  gain); kept as an option.
- **`fp8`** ⚠️ **experimental/broken** — FP8 lm_head. No speedup, +51M params, and a zero-gradient
  backward (model won't train). Do not use; needs a backward fix.
- **`all_safe`** = `fused_ce`+`polar` · **`all_max`** = `fp8`+`polar` (broken).

Measured results: see `docs/ARCHITECTURE_RESEARCH.md` §8. **Recommended: `--opts fused_ce`** (or
`all_safe`), then bump `--micro` to spend the freed memory.

```bash
# synthetic throughput probe at target scale (ms/step, tok/s, peak GB, speedup table)
python -m modal run modal_train.py --action speedtest --preset 1B
# quality ablation: short real 130M runs per variant -> final val loss
python -m modal run modal_train.py --action ablate_opts --preset 130M --steps 800
```
