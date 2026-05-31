# MoE Architecture Research & Design Spec

Consolidated from primary sources (DeepSeekMoE/V3/V3.2/**V4**, OLMoE, Qwen3, Mixtral, LFM2,
Switch/ST-MoE, Krajewski fine-grained scaling laws, Muon/Moonlight/Kimi-K2) + modded-nanogpt.
Target: **efficient MoE LLMs at ~130M, ~500M, ~1B total params**. Date: 2026-05-30.

---

## 1. Decisions that are settled (consensus, low-risk — bake in)

| Area | Decision | Source / rationale |
|---|---|---|
| Backbone | Decoder-only transformer, **pre-norm RMSNorm** (eps 1e-6) | universal |
| **QK-norm** | RMSNorm per-head on Q and K **before RoPE** | OLMo-2/OLMoE/Qwen3; cheapest stability win, bounds attn-logit blowup |
| Attention | **GQA**, NOT MLA (MLA not worth complexity <1B) | no sub-1B model uses MLA; KV cache tiny at this scale |
| head_dim | **64** (130M) → **128 decoupled from d_model** (500M/1B, Qwen3 trick) | Qwen3-0.6B: 16×128=2048 > d_model 1024 |
| Positional | **RoPE θ=10,000** for ≤4k context | larger θ only for long context |
| Expert FFN | **SwiGLU** (gate/up/down, 3 matrices) | all reference MoEs |
| Embeddings | **tied** (embed = lm_head), vocab **~50k** (GPT-2/FineWeb), ×√d_model scale | embeddings ≈22% of 130M model — tying mandatory |
| Shape | **deep-thin** (aspect ratio d_model/n_layers ~20–55) | MobileLLM +0.9-1.1%, Qwen3-0.6B ratio ~37 |
| MoE granularity | **fine-grained, G = dense_ffn/expert_ffn ≈ 8** (never G=1) | Krajewski scaling law: G=8@100M, G=16@1B |
| MoE placement | **first 1 layer dense**, rest MoE | DeepSeekMoE-16B; early routing unstable |
| Routing mode | **dropless** (no capacity/token-drop) | OLMoE ablation: dropless > dropping |
| Expert compute | **sort tokens by expert → `torch._grouped_mm` → scatter** | torchtitan pattern; fast on H100, compile-able, no EP needed <1B |
| Router precision | **fp32** logits, small init (**0.1×**) | ST-MoE; bf16 router = instability |
| Routing fn | **token-choice** top-k (NOT expert-choice) | OLMoE: token-choice wins downstream |
| Optimizer | **Muon** (hidden + expert matrices, batched per-expert NS) + **AdamW** (router, embed, lm_head, norms) | Moonlight/Kimi-K2/DeepSeek-V4 split |
| z-loss (final logits) | 1e-4 · mean(logsumexp²) | PaLM/OLMo |
| Init | std 0.02, residual-proj × 1/√(2·n_layers) | GPT-2/Megatron; critical for deep-thin |

## 2. Decisions to ABLATE (genuine forks in the literature)

1. **Shared expert: 0 vs 1.** DeepSeek says yes (helps at 256 experts); **OLMoE + Qwen3 found it slightly HURTS at ≤64 experts**. → ablate; default 0 @130M, 1 @500M/1B.
2. **Gating: sigmoid+top-k-renorm (DeepSeek-V3) vs softmax (OLMoE).** Sigmoid more stable at high expert count; softmax simpler. → ablate.
3. **Load balancing: aux-loss-free bias (DeepSeek-V3, u=0.001 anneal→0) + tiny global-batch aux (α≈1e-3) vs classic aux loss (α=0.01) + z-loss (β=1e-3) (OLMoE).** → ablate; bias method is the modern winner.
4. **num_experts / top_k:** 32/top4 → 64/top8. Diminishing returns past 64 (OLMoE). → ablate at small scale.
5. **MTP single head (λ≤0.1, curriculum):** helps big models, **SLMs struggle** w/o curriculum. → optional ablation, mainly for spec-decode draft head.
6. **LFM2 conv-attention hybrid (efficiency lever):** replace most attention with depthwise causal short-conv (k=3) + double gate, keep ~⅓ GQA layers. 2× CPU decode, fixed-size state vs growing KV cache. → optional efficiency ablation.
7. **Router z-loss β** and **balance over global batch vs micro-batch** (Qwen3: global wins).

## 3. Advanced ideas from DeepSeek-V4 / Kimi (higher effort, optional)

- **MuonClip / QK-Clip** (Kimi K2): cap Q·K logits → zero loss spikes at scale. Cheap to add to QK-norm.
- **Aux-loss-free bias** is now standard (V3→V4 keep it; V4 adds tiny seq-wise loss).
- **Hash routing for first MoE layers** (V4): token-ID hash → expert, zero routing compute, perfectly balanced. Alternative to dense-first-layers.
- **DSA distilled lightning-indexer** (V3.2): warm dense → KL-distill cheap ReLU indexer → top-k sparse attention. Best long-context trick; overkill <1B short context.
- **mHC (Manifold-Constrained Hyper-Connections)** + Muon (V4): deep-stability; plain residuals fine <1B.
- **MTP depth-1** retained through V4.

## 4. Recommended starting configs (tune exact dims with a param-counter to hit targets)

SwiGLU expert params ≈ `(n_exp+shared)·3·d_model·expert_ffn`. Attention(GQA)+embeds counted separately.
All: RMSNorm+QK-norm, RoPE θ=1e4, tied embeds vocab 50304, SwiGLU, dropless grouped_mm, Muon+AdamW.

| Config | d_model | n_layers (dense) | Q/KV heads | head_dim | dense_ffn | n_exp | top_k | shared | expert_ffn | G | ~Total | ~Active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **MoE-130M** | 512 | 12 (1) | 8/2 | 64 | 1536 | 32 | 4 | 0 | 256 | 6 | ~135M | ~22M |
| **MoE-500M** | 768 | 16 (1) | 12/3 | 128* | 2304 | 48 | 6 | 1 | 320 | 7 | ~510M | ~70M |
| **MoE-1B** | 1024 | 20 (1) | 16/8 | 128* | 2816 | 64 | 8 | 1 | 384 | 7.5 | ~1.05B | ~140M |

*head_dim 128 decoupled: heads×128 may exceed d_model (Qwen3 trick). Active/total ≈ 6–8× sparsity
(memory-bound at small scale; matches OLMoE 5.3×, DeepSeekMoE-16B 5.9×).

**Closest single blueprint = OLMoE** (1.3B active/6.9B total, 16L, d2048, 64 exp, top-8, expert_ffn 1024,
no shared, SwiGLU, QK-norm, dropless, aux 0.01 + z 0.001). Scale down for our budgets.

## 5. Optimizer config (Muon + AdamW)

```
Muon: hidden attn/mlp/EXPERT matrices (ndim>=2, not router/embed/head/norm)
  lr≈0.02 (or 2× adam matrix lr), momentum 0.95 (nesterov), wd 0.1, ns_steps 5,
  update = 0.2 · NewtonSchulz(momentum) · sqrt(max(d_in,d_out)) + wd·W
  3D expert stacks (E,d,h): BATCHED per-expert Newton-Schulz (bmm), with transpose branch for thin matrices
AdamW: router(gate), embeddings, lm_head, all RMSNorm/biases/scalars
  lr 3e-4, betas (0.9,0.95), eps 1e-8, wd 0.1
Router pitfall: Muon flattens singular values → destroys routing signal & amplifies noise on rare experts → ALWAYS AdamW.
```

## 6. Implementation notes (PyTorch, H100, Modal)

- **bf16** everywhere (grouped_mm requires it); **fp32** router logits + losses.
- `torch._grouped_mm(xs_sorted, w, offs=cumsum.int32)` — one kernel, dropless, no padding.
- `torch.compile`: `mark_dynamic(sorted_tokens, dim=0)` to avoid per-step recompiles from variable tokens/expert; or fixed-capacity for fully static graph; or compile only GEMM region.
- **No EP needed <1B** — fits one A100/H100. Multi-GPU = FSDP2 `fully_shard` or DDP. EP only for tens-of-B.
- Reuse from modded-nanogpt: Muon/NorMuon, RMSNorm, RoPE/YaRN, QK-norm, logit softcap, FineWeb data loader, FP8 (later).

## 6b. ABLATION RESULTS (130M, 3000 steps, FineWeb val loss) — 2026-05-30

10-way single-knob ablation on 10× H100. Baseline = 3.6688.

| run | val loss | Δ | conclusion |
|---|---|---|---|
| topk8 | 3.6441 | −0.025 | top-8 > top-4: more active experts help → adopt higher top_k |
| no_renorm | 3.6542 | −0.015 | **norm_topk_prob=False wins** (OLMoE-style) → flipped default |
| softmax | 3.6677 | −0.001 | sigmoid≈softmax (no diff) |
| shared1 | 3.6678 | −0.001 | shared expert neutral at 32 experts (matches OLMoE) |
| baseline | 3.6688 | 0 | — |
| scale_emb | 3.6775 | +0.009 | Gemma embed scaling slightly hurts → keep off |
| aux_loss | 3.7048 | +0.036 | classic aux-loss worse than aux-free bias → keep aux-free |
| no_zloss | 3.7097 | +0.041 | router z-loss helps → keep |
| experts16 | 3.7166 | +0.048 | fewer experts worse → keep ≥32 |
| no_qknorm | 4.4715 | **+0.80** | **QK-norm is critical** (biggest effect) |

**Promotions applied to config defaults:** `norm_topk_prob=False`; 130M `top_k 4→8`.
**Confirmed bake-ins:** QK-norm (essential), router z-loss, aux-loss-free bias, ≥32 experts, sigmoid.
**Dropped:** embedding scaling. **Neutral:** shared expert at 32 experts (kept 1 in 500M/1B per literature at higher expert counts).

## 6c. FLAGSHIP 130M RESULT (10B tokens) — 2026-05-30

Trained the ablation-winning 130M config (top-8, norm_topk_prob=False, QK-norm, aux-free bias, z-loss)
on **~10B FineWeb tokens**: 8×H100, 9537 steps × 1.05M tokens/step, batch-scaled LR (sqrt: muon 0.04 /
adam 6e-4), ~742ms/step, **~2h wall-clock**. **Final val loss = 3.3016** (vs 3.669 ablation baseline at
3000 steps — **−0.37**). Weights at `/data/runs/130M_10B/model.pt` (+ ckpt_2000/4000/6000/8000) on Modal
volume. Inference (`generate.py`): fluent coherent English, correct simple facts (e.g. "capital of France
is Paris"), with repetition loops + factual hallucinations as expected at 130M. Context = 1024 tokens.

## 6d. FLAGSHIP 500M RESULT (40B UNIQUE tokens) — 2026-05-31

Trained the 500M config (d768, 16L, 36 exp/top6, 1 shared, winning recipe) on **~40B UNIQUE FineWeb-100B
tokens**: 8×H100, 38147 steps × 1.05M tokens, lr sqrt-scaled ×2, ~1208ms/step, **~13h wall-clock**.
**Final val loss = 3.0281** vs 130M flagship 3.3016 (−0.27) and 130M ablation baseline 3.669 (−0.64).
Weights `/data/runs/500M_40B/model.pt` (+ ckpt_5000..35000). Scaling holds: 130M/10B → 3.30, 500M/40B → 3.03.

| model | params (total/active) | tokens | final val loss |
|---|---|---|---|
| 130M | 140M / 62M | 10B | 3.3016 |
| 500M | 500.8M / 169M | 40B unique | **3.0281** |

## 7. Data & eval
FineWeb / FineWeb-Edu 10B (modded-nanogpt `data/cached_fineweb10B.py`), GPT-2 tokenizer (vocab 50257→50304),
target metric = val cross-entropy (modded-nanogpt uses 3.28 on FineWeb val). Track active-param efficiency.

## 8. Throughput optimizations (nanogpt-inspired) — 2026-05-31
Ported the *applicable* speed techniques from modded-nanogpt's flagship `train_gpt.py` into the clean
codebase (its FP8/Triton/FlashAttn-3/varlen stack is too entangled to fork wholesale). All are flag-gated;
defaults preserve the exact original training math.

**Always-on (numerics-identical):**
- **On-device loss accumulation** — the old loop called `loss.item()` every micro-step, forcing a
  device->host sync that serialized grad accumulation. Now accumulate a GPU scalar, sync once per log.
- **`PYTORCH_ALLOC_CONF=expandable_segments:True`** — less allocator fragmentation -> fits a bigger batch.
- **CUDA data prefetch** (`CUDAPrefetcher`) — double-buffers the H2D copy on a side stream to overlap
  with compute (data loader yields pinned host tensors).

**Opt-in (`--opts` / config flags):**
- **`fused_ce`** (numerics-preserving) — vocab projection + CE done in row-chunks under activation
  checkpointing, so the full `(T, vocab)` fp32 logit tensor (~1.6 GB at the 1B micro-batch) is never
  materialized or saved for backward. Verified **bit-identical** loss & grads vs the baseline CE on CPU.
- **`polar`** — Polar Express orthogonalization schedule in Muon (arXiv:2505.16932): a tuned 5-iteration
  coefficient sequence that converges the Newton-Schulz step faster (tighter singular values toward 1).
- **`fp8`** — FP8 (`torch._scaled_mm`, e4m3 fwd / e5m2 bwd) on the lm_head, the single largest GEMM.
  Stored transposed for a natural grad layout (per @YouJiacheng); **unties** the head (+`vocab x d` params),
  copy-initialized from the embedding for a fair start. Numerics change -> validated by the quality ablation.
- **`all_safe`** = `fused_ce`+`polar` (recommended for the real 1B run) · **`all_max`** = `fp8`+`polar`.

Harness: `--action speedtest` (synthetic ms/step + tok/s + peak-mem table at any preset) and
`--action ablate_opts` (short real-data val-loss ablation across variants). CPU checks in `test_speedups.py`.

**MEASURED — 1B speed probe (H100, micro=8 × seq 1024, accum=1, synthetic):**

| opts | ms/step | tok/s | peak GB | speedup | params |
|---|---|---|---|---|---|
| **fused_ce** | **209.2** | **39,166** | **26.4** | **1.06×** | 1037M |
| all_safe (fused_ce+polar) | 210.1 | 38,988 | 26.4 | 1.06× | 1037M |
| baseline | 222.5 | 36,821 | 33.3 | 1.00× | 1037M |
| fp8 | 223.8 | 36,605 | 33.8 | 0.99× | 1089M |
| polar | 224.8 | 36,445 | 33.3 | 0.99× | 1037M |
| all_max (fp8+polar) | 228.7 | 35,819 | 33.8 | 0.97× | 1089M |

The memory saving is the real lever: at **micro=32**, baseline / fp8 / polar / all_max all **OOM** (even with
`expandable_segments`), while fused_ce / all_safe fit at 70.9 GB and reach **~74,400 tok/s** — roughly **2×**
the throughput of baseline's largest fitting batch (≈36.8k tok/s at micro=8).

**MEASURED — 130M quality ablation (1000 steps, FineWeb val loss):**

| variant | val loss | note |
|---|---|---|
| **fused_ce** | **4.0681** | numerics-neutral (≈ baseline, marginally better) |
| baseline | 4.0739 | reference |
| polar | 4.0790 | neutral (within noise) |
| all_safe | 4.0855 | neutral |
| fp8 / all_max | 10.90 / 10.94 | **broken — frozen at init (no grad through fp8 backward)** |

**Verdict:** ship **`fused_ce`** (free: +6% step time, −21% peak memory, ~2× batch headroom, quality-identical).
**Polar Express** was a wash here (no speed or quality gain at this scale/horizon; kept as an option — may help
over much longer runs per the paper). **FP8 head dropped:** no speedup, +51M params, and a zero-gradient
backward bug that prevents training. The always-on opts (no-sync accumulation, `expandable_segments`, prefetch)
are pure wins. For the 1B run: enable `fused_ce` and raise `micro_batch_seqs` to exploit the freed memory.

**MEASURED — 1B end-to-end training time (8×H100 DDP, 60-step probe, fused_ce, micro=32, 1.05M-tok batch):**
**~1.53 s/step** steady-state (~688k tok/s aggregate, ~86k/GPU; micro=32 fits in DDP, no OOM). At 1.05M
tokens/step, **100B tokens = 95,400 steps ≈ 41–42 h wall-clock (~$1,300 on Modal H100)** — ~10–20 h under the
paper estimate, since real DDP throughput beat the conservative efficiency assumption. Launch command:
`modal run modal_train.py --action train --preset 1B --gpus 8 --steps 95400 --opts fused_ce --micro 32
--batch-tokens 1048576 --data 100B --run-name 1B_100B --save-every 5000`.

## 9. Downstream eval — lm-evaluation-harness (0-shot) — 2026-05-31
EleutherAI lm-eval-harness (v0.4.9.1) via a custom `MoELMWrapper` (eval_harness.py; loglikelihood over
our MoE + gpt2 tokenizer). `modal run modal_train.py --action lmeval --run-name both`. acc_norm where
defined, else acc. (Active params: 130M→62M, 500M→169M; both near GPT-2-small/Pythia-160M class.)

| task | 130M_10B | 500M_40B | random |
|---|---|---|---|
| lambada_openai (acc) | 0.2996 | **0.3998** | 0 |
| hellaswag | 0.3253 | **0.4154** | 0.25 |
| arc_easy | 0.3771 | **0.4272** | 0.25 |
| arc_challenge | 0.2355 | 0.2235 | 0.25 |
| piqa | 0.6545 | **0.6964** | 0.50 |
| winogrande (acc) | 0.5264 | 0.5162 | 0.50 |
| openbookqa | 0.2820 | **0.2960** | 0.25 |
| sciq | 0.6020 | **0.7030** | 0.25 |
| boolq (acc) | 0.6125 | 0.5104 | ~0.62 (majority) |
| **average** | **0.4350** | **0.4653** | — |

500M wins on every LM-driven task (lambada/hellaswag/sciq/piqa/arc_easy, +0.05–0.10), consistent with its
lower val loss (3.03 vs 3.30). arc_challenge/winogrande sit at chance for both (expected at this scale); boolq
hovers near the majority-class baseline (noisy for base models). Results saved to `runs/<name>/lm_eval.json`.

**Sub-1B comparison — all run through OUR harness (`--action lmeval_hf`, identical 7-task protocol).**
**All rows are BASE / pretrained-only checkpoints (no instruction tuning), 0-shot — matching our base models**
(the widely-quoted Qwen3/Gemma-3/SmolLM2 *instruct* numbers are higher and would not be a fair comparison).
Validated by reproduction: MicroLlama 42.23 (card: 42.36), TinyLlama-1.1B-3T 52.75 (card: 52.99). Sorted by avg:

| # | model | params | tokens | hella | obqa | wino | arc_c | arc_e | boolq | piqa | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SmolLM2-360M | 360M | 4T | 56.41 | 38.00 | 59.35 | 38.48 | 68.22 | 62.08 | 71.49 | 56.29 |
| 2 | Qwen3-0.6B | 600M | 36T | 53.81 | 34.60 | 58.56 | 38.31 | 58.08 | 69.82 | 70.29 | 54.78 |
| 3 | SmolLM2-135M | 135M | 2T | 43.13 | 33.20 | 52.25 | 29.44 | 58.59 | 60.37 | 68.39 | 49.34 |
| 4 | gemma-3-270m | 270M | 6T | 41.28 | 30.60 | 53.43 | 28.33 | 57.07 | 57.77 | 68.12 | 48.09 |
| 5 | pythia-410m | 410M | 300B | 40.14 | 28.80 | 52.96 | 23.98 | 45.45 | 58.50 | 67.52 | 45.34 |
| 6 | **OURS-500M-MoE** | 500M/169M act | 40B | 41.56 | 29.80 | 51.30 | 22.44 | 42.76 | 50.98 | 69.53 | **44.05** |
| 7 | opt-350m | 350M | 180B | 36.55 | 28.40 | 52.80 | 24.74 | 40.49 | 57.61 | 64.69 | 43.61 |
| 8 | **OURS-130M-MoE** | 140M/62M act | 10B | 32.54 | 28.20 | 52.17 | 23.46 | 37.71 | 61.31 | 65.40 | **42.97** |
| 9 | MicroLlama-300M | 300M | 50B | 34.22 | 30.40 | 51.30 | 23.21 | 39.23 | 52.57 | 64.69 | 42.23 |
| 10 | gpt2-124m | 124M | ~10B | 31.15 | 27.40 | 51.46 | 22.53 | 39.60 | 49.94 | 62.24 | 40.62 |
| 11 | pythia-160m | 160M | 300B | 30.21 | 26.60 | 49.64 | 24.83 | 36.95 | 42.42 | 59.52 | 38.60 |

**Ranking ≈ sorted by pretrain tokens, not params.** The top 4 (SmolLM2/Qwen3/Gemma-3) use **2T–36T tokens
(50–900× ours)** — a different data regime. Within the *classic* regime (≤300B tok), ours lead per token:
OURS-500M (44.05, 40B tok) beats opt-350m / MicroLlama / gpt2 / pythia-160m and trails only pythia-410m (300B tok,
7.5×). **OURS-130M (42.97, just 10B tok) beats MicroLlama-300M (50B) and pythia-160m (300B)** —
out-scoring 300M-param models trained on 5–30× more data. Per-token efficiency is the win; absolute score is
gated by token budget → the 1B/100B run (and longer 130M/500M runs) is the lever. (≥1B refs: SmolLM2-1.7B 64.53,
Qwen3-1.7B 62.49, TinyLlama-1.1B-3T 52.75; omitted from this <1B view.)

*Eval-harness audit note:* our `MoELMWrapper` originally encoded context/continuation separately; fixed to match
lm-eval's `_encode_pair` (joint boundary tokenization + trailing-whitespace handling), as the HF reference models
use. Impact was negligible — 500M 44.07→44.05, 130M 43.05→42.97 (only winogrande moved, ~0.4) — so rankings and
conclusions are unchanged. Numbers above are post-fix. (The 9-task table in §9 above predates the fix; same ~0.1 shift.)

See [[moe-project]], [[modal-infra]].
