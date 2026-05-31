"""Muon (hidden + expert matrices) + AdamW (router, embeddings, head, norms).

Muon orthogonalizes the momentum via Newton-Schulz. Works on 2D weights and, batched over the
leading dim, on 3D expert stacks (E, d_in, d_out). The router gate, embeddings, lm_head, and all
1D params (norms/biases) go to AdamW — Muon would destroy the router's routing signal.
"""
from __future__ import annotations

import torch
from torch import Tensor


def newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Orthogonalize over the last two dims; batched over any leading dims (e.g. experts)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.1, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            mom, wd, lr, steps = group["momentum"], group["weight_decay"], group["lr"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "mom" not in state:
                    state["mom"] = torch.zeros_like(g)
                buf = state["mom"]
                buf.lerp_(g, 1 - mom)
                g = g.lerp(buf, mom)                       # Nesterov
                o = newton_schulz(g, steps).to(p.dtype)
                m, n = p.shape[-2], p.shape[-1]
                scale = 0.2 * max(m, n) ** 0.5             # update RMS ~ AdamW (Moonlight)
                p.mul_(1 - lr * wd)                        # decoupled weight decay
                p.add_(o, alpha=-lr * scale)


def build_optimizers(model, tc):
    """Return (muon, adamw). Dedupes tied params; routes by name/shape."""
    muon_p, adam_p, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_router = name.endswith("gate.weight")
        is_embed = ("embed" in name) or ("lm_head" in name)
        if p.ndim >= 2 and not is_router and not is_embed:
            muon_p.append(p)
        else:
            adam_p.append(p)
    muon = Muon(muon_p, lr=tc.muon_lr, momentum=tc.muon_momentum,
                weight_decay=tc.muon_wd, ns_steps=tc.muon_ns_steps)
    adamw = torch.optim.AdamW(adam_p, lr=tc.adam_lr, betas=tc.adam_betas,
                              weight_decay=tc.adam_wd, eps=1e-8)
    return muon, adamw, (len(muon_p), len(adam_p))
