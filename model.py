"""The MoE transformer: GQA + QK-norm + RoPE attention, dense/MoE SwiGLU FFNs, tied head.

Pre-norm RMSNorm blocks. First cfg.n_dense_layers use a dense SwiGLU FFN; the rest use MoE.
Loss = cross-entropy + final-logit z-loss + sum of per-layer MoE aux/z losses.
"""
from __future__ import annotations

import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import ModelConfig
from moe import MoE, SwiGLUWeights


def rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    out = F.rms_norm(x, (x.size(-1),), eps=eps)
    return out * weight if weight is not None else out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


def precompute_rope(head_dim: int, max_seq: int, theta: float, device) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (S, head_dim/2)
    return freqs.cos(), freqs.sin()             # each (S, head_dim/2)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, S, D). Rotate-half formulation.
    S, D = x.shape[-2], x.shape[-1]
    cos = cos[:S].view(1, 1, S, D // 2)
    sin = sin[:S].view(1, 1, S, D // 2)
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.nq, self.nkv, self.hd = cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim
        self.rep = self.nq // self.nkv
        qkv_out = (self.nq + 2 * self.nkv) * self.hd
        self.qkv = nn.Linear(cfg.d_model, qkv_out, bias=False)
        self.proj = nn.Linear(self.nq * self.hd, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.nq * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = q.view(B, S, self.nq, self.hd).transpose(1, 2)    # (B, nq, S, hd)
        k = k.view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = v.view(B, S, self.nkv, self.hd).transpose(1, 2)
        if self.cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)             # per-head RMSNorm before RoPE
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # GQA: expand kv heads to match q heads
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, S, self.nq * self.hd)
        return self.proj(o)


class DenseFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w13 = nn.Linear(cfg.d_model, 2 * cfg.dense_ffn, bias=False)
        self.w2 = nn.Linear(cfg.dense_ffn, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.is_moe = layer_idx >= cfg.n_dense_layers
        self.ffn = MoE(cfg) if self.is_moe else DenseFFN(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        if self.is_moe:
            out, aux = self.ffn(self.ffn_norm(x))
            return x + out, aux
        return x + self.ffn(self.ffn_norm(x)), x.new_zeros(())


class MoETransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._rope_cache: dict = {}
        self.apply(self._init)
        self._scale_residual_init()

    # ---- init ----
    def _init(self, m: nn.Module):
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, SwiGLUWeights):
            nn.init.normal_(m.w13, mean=0.0, std=std)
            nn.init.normal_(m.w2, mean=0.0, std=std)

    def _scale_residual_init(self):
        # scale residual-projection weights by 1/sqrt(2*n_layers) (GPT-2/Megatron), critical deep-thin
        scale = (2 * self.cfg.n_layers) ** -0.5
        for blk in self.blocks:
            with torch.no_grad():
                blk.attn.proj.weight.mul_(scale)
                if isinstance(blk.ffn, DenseFFN):
                    blk.ffn.w2.weight.mul_(scale)
                else:
                    blk.ffn.experts.w2.mul_(scale)
                    if self.cfg.n_shared > 0:
                        blk.ffn.shared.w2.mul_(scale)
                    # small router init (~0.1x)
                    blk.ffn.gate.weight.mul_(0.1)

    def rope(self, S: int, device, dtype):
        key = (S, device, dtype)
        if key not in self._rope_cache:
            cos, sin = precompute_rope(self.cfg.head_dim, S, self.cfg.rope_theta, device)
            self._rope_cache[key] = (cos.to(dtype), sin.to(dtype))
        return self._rope_cache[key]

    def forward(self, idx: Tensor, targets: Tensor | None = None):
        B, S = idx.shape
        x = self.embed(idx)
        if self.cfg.scale_embeddings:
            x = x * (self.cfg.d_model ** 0.5)
        cos, sin = self.rope(S, idx.device, x.dtype)
        aux_sum = x.new_zeros(())
        for blk in self.blocks:
            x, aux = blk(x, cos, sin)
            aux_sum = aux_sum + aux
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if self.cfg.logit_softcap > 0:
            sc = self.cfg.logit_softcap
            logits = sc * torch.tanh(logits / sc)
        if targets is None:
            return logits, aux_sum
        logits = logits.float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        z = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        loss = ce + self.cfg.final_z_loss_coef * z + aux_sum
        return loss, {"ce": ce.detach(), "aux": aux_sum.detach(), "z": z.detach()}

    @torch.no_grad()
    def set_bias_update_rate(self, rate: float):
        for blk in self.blocks:
            if isinstance(blk.ffn, MoE):
                blk.ffn.bias_update_rate = rate

    @torch.no_grad()
    def sync_expert_bias(self):
        """Average aux-free bias buffers across DDP ranks so they stay identical
        (each rank updates from local token counts; DDP doesn't sync buffers)."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        for blk in self.blocks:
            if isinstance(blk.ffn, MoE):
                dist.all_reduce(blk.ffn.expert_bias, op=dist.ReduceOp.SUM)
                blk.ffn.expert_bias.div_(world)


def count_params(model: MoETransformer) -> dict:
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    # subtract tied head double-count is already avoided (shared weight counted once)
    embed = cfg.vocab_size * cfg.d_model * (1 if cfg.tie_embeddings else 2)
    # active = total - inactive routed experts. Per MoE layer, only top_k of n_experts run.
    per_expert = cfg.d_model * 2 * cfg.expert_ffn + cfg.expert_ffn * cfg.d_model
    inactive_per_moe = (cfg.n_experts - cfg.top_k) * per_expert
    active = total - cfg.n_moe_layers * inactive_per_moe
    return {"total": total, "active": active, "embed": embed,
            "active_pct": 100 * active / total, "sparsity": total / active}
