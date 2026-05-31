"""Mixture-of-Experts layer: fp32 router, dropless expert compute, load balancing.

Backends for the expert compute (selected by cfg.expert_backend):
  - "grouped": sort tokens by expert -> torch grouped_mm -> scatter. Fast on H100/A100, bf16.
  - "bmm":     loop-free reference using per-expert masked matmuls. CPU-testable.
  - "loop":    explicit python loop over experts. Slowest, clearest reference.

Balancing:
  - "aux_free":  DeepSeek-V3 gradient-free per-expert bias added to the SELECTION scores only
                 (not the gate weights), updated by sign(load error). Plus a tiny aux loss safety net.
  - "aux_loss":  classic Switch/OLMoE load-balance loss (coef ~1e-2) only.
  Both add a router z-loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import ModelConfig


def _grouped_mm(a: Tensor, b: Tensor, offs: Tensor) -> Tensor:
    """a: (T, d_in) expert-sorted; b: (E, d_in, d_out); offs: (E,) int32 cumulative row counts.
    grouped_mm requires bf16 inputs; cast here (autocast does not cover this custom op)."""
    fn = getattr(F, "grouped_mm", None) or getattr(torch, "_grouped_mm", None)
    if fn is None:
        raise RuntimeError("grouped_mm not available in this torch build; use expert_backend='bmm'")
    return fn(a.bfloat16(), b.bfloat16(), offs=offs)


class SwiGLUWeights(nn.Module):
    """A stack of E SwiGLU experts. w13: (E, d, 2f) gate+up fused; w2: (E, f, d) down."""
    def __init__(self, n_experts: int, d_model: int, ffn: int):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.ffn = ffn
        self.w13 = nn.Parameter(torch.empty(n_experts, d_model, 2 * ffn))
        self.w2 = nn.Parameter(torch.empty(n_experts, ffn, d_model))

    def expert_glu(self, x: Tensor, e: int) -> Tensor:
        h = x @ self.w13[e]
        gate, up = h.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.w2[e]


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.n_shared = cfg.n_shared
        self.backend = cfg.expert_backend

        # fp32 router (gate). Initialized small in model init.
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = SwiGLUWeights(cfg.n_experts, cfg.d_model, cfg.expert_ffn)
        if cfg.n_shared > 0:
            self.shared = SwiGLUWeights(cfg.n_shared, cfg.d_model, cfg.expert_ffn)

        # aux-loss-free per-expert routing bias (no grad), updated by sign(load error)
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))
        self.bias_update_rate = cfg.bias_update_rate  # set to 0 to freeze (anneal at end of training)

    # ---- routing (always fp32, even under bf16 autocast) ----
    def _route(self, x: Tensor):
        """x: (T, d). Returns topi (T,k) long, topv (T,k) fp32 gate weights, aux_loss scalar."""
        cfg = self.cfg
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.gate.weight.float())   # fp32 router
            scores = torch.sigmoid(logits) if cfg.gating == "sigmoid" else torch.softmax(logits, dim=-1)

            # selection scores: add gradient-free bias (aux_free) for balanced routing
            sel = scores + self.expert_bias.float() if cfg.balancing == "aux_free" else scores
            topi = torch.topk(sel, self.top_k, dim=-1).indices       # (T, k)
            topv = torch.gather(scores, -1, topi)                    # gate weights from ORIGINAL scores
            if cfg.norm_topk_prob:
                topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)

            # ---- balancing losses ----
            T = x.shape[0]
            counts = torch.bincount(topi.reshape(-1), minlength=self.n_experts).float()
            f_i = counts / (T * self.top_k)
            P_i = scores.mean(dim=0)
            aux = self.n_experts * (f_i.detach() * P_i).sum()
            z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
            aux_total = cfg.aux_loss_coef * aux + cfg.z_loss_coef * z_loss

            # update aux-free bias (no grad): under-loaded experts up, over-loaded down
            if cfg.balancing == "aux_free" and self.training and self.bias_update_rate > 0:
                with torch.no_grad():
                    ideal = T * self.top_k / self.n_experts
                    self.expert_bias.add_(self.bias_update_rate * torch.sign(ideal - counts))

        return topi, topv, aux_total

    # ---- expert compute backends ----
    def _experts_grouped(self, x: Tensor, topi: Tensor, topv: Tensor) -> Tensor:
        T, d = x.shape
        k = self.top_k
        flat_e = topi.reshape(-1)
        flat_tok = torch.arange(T, device=x.device).repeat_interleave(k)
        order = torch.argsort(flat_e)
        sort_e = flat_e[order]
        sort_tok = flat_tok[order]
        xs = x[sort_tok]                                            # (T*k, d)
        counts = torch.bincount(sort_e, minlength=self.n_experts)
        offs = torch.cumsum(counts, 0).to(torch.int32)
        h = _grouped_mm(xs, self.experts.w13, offs)                # (T*k, 2f) bf16
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up
        y = _grouped_mm(h, self.experts.w2, offs).to(x.dtype)      # (T*k, d) -> residual dtype
        y = y * topv.reshape(-1)[order].unsqueeze(-1).to(x.dtype)
        out = torch.zeros_like(x)
        out.index_add_(0, sort_tok, y)
        return out

    def _experts_bmm(self, x: Tensor, topi: Tensor, topv: Tensor) -> Tensor:
        # reference path: per-expert masked matmul (works on CPU)
        T, d = x.shape
        flat_e = topi.reshape(-1)
        flat_tok = torch.arange(T, device=x.device).repeat_interleave(self.top_k)
        flat_v = topv.reshape(-1)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            sel = (flat_e == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            toks = flat_tok[sel]
            ye = self.experts.expert_glu(x[toks], e).to(x.dtype) * flat_v[sel].unsqueeze(-1).to(x.dtype)
            out.index_add_(0, toks, ye)
        return out

    def _shared(self, x: Tensor) -> Tensor:
        out = x.new_zeros(x.shape)
        for e in range(self.n_shared):
            out = out + self.shared.expert_glu(x, e).to(x.dtype)
        return out

    def forward(self, x: Tensor):
        """x: (B, S, d) -> (out (B,S,d), aux_loss scalar)."""
        B, S, d = x.shape
        xf = x.reshape(-1, d)
        topi, topv, aux = self._route(xf)
        if self.backend == "grouped":
            out = self._experts_grouped(xf, topi, topv)
        else:
            out = self._experts_bmm(xf, topi, topv)
        if self.n_shared > 0:
            out = out + self._shared(xf)
        return out.reshape(B, S, d), aux
