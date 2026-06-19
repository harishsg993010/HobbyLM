"""Model + training configuration for the MoE lab.

A single ModelConfig drives the architecture; ablation knobs are explicit fields so an
ablation = one config override. PRESETS hold the 130M / 500M / 1B starting points
(exact dims are validated against targets by count_params.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class ModelConfig:
    # ---- shape ----
    vocab_size: int = 50304            # GPT-2 (50257) padded to mult of 128
    d_model: int = 512
    n_layers: int = 12
    n_dense_layers: int = 1            # first-k layers use a dense FFN (DeepSeekMoE-16B style)
    # ---- attention (GQA + QK-norm) ----
    n_q_heads: int = 8
    n_kv_heads: int = 2                # GQA group = n_q_heads / n_kv_heads
    head_dim: int = 64                 # may be decoupled from d_model (Qwen3 trick)
    qk_norm: bool = True               # RMSNorm per-head on Q,K before RoPE
    rope_theta: float = 10000.0
    attn_softcap: float = 0.0          # 0 disables; logit softcap fallback if not using qk_norm
    # ---- FFN / experts (SwiGLU) ----
    dense_ffn: int = 1536              # intermediate dim of the dense FFN layers
    expert_ffn: int = 256              # intermediate dim of ONE routed/shared expert (fine-grained)
    n_experts: int = 32                # routed experts per MoE layer
    top_k: int = 4                     # active routed experts per token
    n_shared: int = 0                  # always-on shared experts (0 or 1 at this scale)
    # ---- routing / balancing (ablation forks) ----
    gating: Literal["sigmoid", "softmax"] = "sigmoid"
    norm_topk_prob: bool = False       # ablation winner: NOT renormalizing top-k gates (OLMoE-style) beat renorm by -0.015
    balancing: Literal["aux_free", "aux_loss"] = "aux_free"
    aux_loss_coef: float = 1e-3        # global-batch balance loss (safety net w/ aux_free; 1e-2 if aux_loss only)
    z_loss_coef: float = 1e-3          # router z-loss
    bias_update_rate: float = 1e-3     # aux-loss-free per-expert bias step (u); annealed to 0 near end
    router_init_std: float = 0.02      # router gets a small (0.1x-ish) init; see model init
    # ---- embeddings / output ----
    tie_embeddings: bool = True
    scale_embeddings: bool = False     # multiply token embeds by sqrt(d_model) (Gemma)
    final_z_loss_coef: float = 1e-4    # z-loss on final logits
    logit_softcap: float = 0.0         # 0 disables; e.g. 15-30 (Gemma-2)
    # ---- diffusion conversion (LLaDA/MDLM: bidirectional attn + masked-token objective) ----
    diffusion: bool = False            # True => full bidirectional attention + masked-diffusion loss (no AR)
    mask_token_id: int = 50257         # free sentinel in the 50304-padded vocab (>= GPT2_VALID, never an AR token)
    mask_eps: float = 1e-3             # min mask ratio: t ~ U(eps, 1) per sequence
    # ---- multi-token prediction (optional) ----
    n_mtp: int = 0                     # 0 disables; 1 = predict t+2 with one extra head
    mtp_weight: float = 0.1
    # ---- init ----
    init_std: float = 0.02
    # ---- impl ----
    expert_backend: Literal["grouped", "bmm", "loop"] = "grouped"  # grouped=GPU fast; bmm/loop=CPU-testable ref
    # ---- throughput opts (nanogpt-inspired; see docs/ARCHITECTURE_RESEARCH.md §8) ----
    fused_ce: bool = False             # chunked cross-entropy: never materialize the full (T,vocab) fp32 logits
    ce_chunk: int = 4096               # rows per CE chunk (fused_ce only)
    fp8_head: bool = False             # EXPERIMENTAL/BROKEN: FP8 lm_head (untied). No speedup at 1B + zero-grad
                                       # backward (loss frozen at init in the 130M ablation). Do not use.
    fp8_x_scale: float = 1.0           # fp8 activation/weight/grad scales (pre-head x is ~unit RMS, so 1.0 is safe)
    fp8_w_scale: float = 1.0
    fp8_grad_scale: float = 1.0

    def __post_init__(self):
        assert self.n_q_heads % self.n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"
        assert self.n_dense_layers <= self.n_layers
        assert self.top_k <= self.n_experts

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense_layers

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/fineweb10B"
    train_pattern: str = "fineweb_train_*.bin"
    val_pattern: str = "fineweb_val_*.bin"
    seq_len: int = 1024
    batch_tokens: int = 256 * 1024     # tokens per optimizer step (global)
    micro_batch_seqs: int = 16         # sequences per micro-batch per GPU
    # schedule
    max_steps: int = 4000
    warmup_steps: int = 100
    cooldown_frac: float = 0.4         # last fraction of steps linearly decays lr -> lr*final_frac
    final_lr_frac: float = 0.1
    # optimizer
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_wd: float = 0.1
    muon_ns_steps: int = 5
    orthogonalizer: Literal["ns5", "polar"] = "ns5"  # ns5=Newton-Schulz (baseline); polar=Polar Express schedule
    adam_lr: float = 3e-4
    adam_betas: tuple = (0.9, 0.95)
    adam_wd: float = 0.1
    grad_clip: float = 1.0
    # bias anneal
    bias_anneal_frac: float = 0.95     # disable aux-free bias updates after this fraction of steps
    # eval / logging
    val_every: int = 250
    val_tokens: int = 10 * 1024 * 1024
    log_every: int = 10
    # run
    seed: int = 1337
    compile: bool = True
    bf16: bool = True
    out_dir: str = "runs"
    run_name: str = "default"


# ---- preset architectures (starting points; tune with count_params.py) ----
PRESETS: dict[str, ModelConfig] = {
    # dims tuned so TOTAL params hit targets (see count_params.py); G = dense_ffn/expert_ffn
    "130M": ModelConfig(   # ~140M total / ~62M active, G=8; top_k bumped 4->8 (ablation: -0.025 val loss)
        d_model=512, n_layers=12, n_dense_layers=1,
        n_q_heads=8, n_kv_heads=2, head_dim=64,
        dense_ffn=1536, expert_ffn=192, n_experts=32, top_k=8, n_shared=0,
    ),
    "500M": ModelConfig(   # ~500M total, G=7.2
        d_model=768, n_layers=16, n_dense_layers=1,
        n_q_heads=12, n_kv_heads=3, head_dim=128,
        dense_ffn=2304, expert_ffn=320, n_experts=36, top_k=6, n_shared=1,
    ),
    "1B": ModelConfig(     # ~1.02B total, G=12.6
        d_model=1024, n_layers=20, n_dense_layers=1,
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        dense_ffn=2816, expert_ffn=224, n_experts=64, top_k=8, n_shared=1,
    ),
}


def get_config(preset: str) -> ModelConfig:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    # return a copy so callers can mutate for ablations
    return ModelConfig(**PRESETS[preset].to_dict())
