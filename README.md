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
```

## Ablations (each changes ONE thing vs the 130M baseline)
`softmax` gating · `aux_loss` (classic balance) · `shared1` (shared expert) · `no_qknorm` ·
`no_renorm` (top-k gate renorm) · `topk8` · `experts16` · `no_zloss` · `scale_emb`.
Winners get promoted into the 500M/1B configs.
