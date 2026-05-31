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

See [[moe-project]], [[modal-infra]].
