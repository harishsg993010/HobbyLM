"""DreamLite in-context DiT on a DC-AE f32c32 latent — the $300 path.

DC-AE compresses 32x spatially (vs VAE f8's 8x), so a 512px image -> 16x16x32 latent. The two-panel
canvas (target|source, width-concat) is 16x32x32 = ONLY 512 tokens (vs the conv U-Net's 64x128=8,192
positions). A DiT on 512 tokens is ~order-of-magnitude cheaper to train than the conv U-Net, which is
how the budget drops from ~$1.66k toward ~$300. Trade-off: f32 reconstruction is slightly weaker than
f8c16 on fine text (acceptable for the V0 "global + simple-local edits" scope).

Standard DiT (AdaLN-Zero on self-attn + MLP) + cross-attention to the frozen-VLM refiner tokens, flow-
matching velocity. Two-panel handled by a learned left/right panel embedding; loss uses the left half.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DiTConfig:
    in_channels: int = 34      # 32 DC-AE latent ch + 2 optional mask ch
    out_channels: int = 32
    latent_h: int = 16         # 512px / 32 = 16 (256px -> 8)
    panel_w: int = 16          # per-panel width; canvas width = 2*panel_w
    patch: int = 1             # DC-AE latent is already compressed -> patch 1
    d_model: int = 1024
    depth: int = 16
    heads: int = 16
    mlp_ratio: float = 3.0
    ctx_dim: int = 1024
    n_tasks: int = 3


def sinusoidal(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None, :]
    return torch.cat([a.cos(), a.sin()], dim=-1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, ctx_dim: int | None = None, qk_norm: bool = True):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.cross = ctx_dim is not None
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim or dim, 2 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        # QK-norm: RMSNorm on Q,K (over head_dim) BEFORE the dot product — the key training-stability
        # fix for DiTs (Inf-DiT/Lumina-Next/SD3); bounds the attention logits, prevents divergence.
        self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()

    def forward(self, x: Tensor, ctx: Tensor | None = None) -> Tensor:
        B, N, _ = x.shape
        src = ctx if self.cross else x
        q = self.qn(self.q(x).view(B, N, self.heads, self.hd)).transpose(1, 2)
        kv = self.kv(src).view(B, src.shape[1], 2, self.heads, self.hd)
        k = self.kn(kv[:, :, 0]).transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.o(o.transpose(1, 2).reshape(B, N, -1))


class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.sa = Attention(d, cfg.heads)
        self.lnc = nn.LayerNorm(d, eps=1e-6)
        self.ca = Attention(d, cfg.heads, ctx_dim=cfg.ctx_dim)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        h = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(approximate="tanh"), nn.Linear(h, d))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.adaln[-1].weight); nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: Tensor, cond: Tensor, ctx: Tensor) -> Tensor:
        sa_s, sa_sc, sa_g, ml_s, ml_sc, ml_g = self.adaln(cond).chunk(6, dim=-1)
        x = x + sa_g[:, None] * self.sa(modulate(self.ln1(x), sa_s, sa_sc))
        x = x + self.ca(self.lnc(x), ctx)                       # cross-attn to VLM tokens
        x = x + ml_g[:, None] * self.mlp(modulate(self.ln2(x), ml_s, ml_sc))
        return x


class HobbyImageDiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.canvas_w = 2 * cfg.panel_w
        self.n_tokens = (cfg.latent_h // cfg.patch) * (self.canvas_w // cfg.patch)
        self.patch_embed = nn.Conv2d(cfg.in_channels, d, cfg.patch, stride=cfg.patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d))
        self.panel = nn.Parameter(torch.zeros(2, d))            # left/right
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.task_emb = nn.Embedding(cfg.n_tasks, d)
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.depth)])
        self.ln_f = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.adaln_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.adaln_f[-1].weight); nn.init.zeros_(self.adaln_f[-1].bias)
        self.head = nn.Linear(d, cfg.patch * cfg.patch * cfg.out_channels)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: Tensor, t: Tensor, ctx: Tensor, task: Tensor | None = None) -> Tensor:
        """x: (B, in_ch, H, 2*panel_w) two-panel DC-AE latent. t: (B,). ctx: (B, M, ctx_dim).
        Returns velocity (B, out_ch, H, 2*panel_w)."""
        cfg = self.cfg
        B, _, H, W = x.shape
        h = self.patch_embed(x).flatten(2).transpose(1, 2)      # (B, N, d)
        h = h + self.pos
        # panel embedding: tokens in the left half vs right half (by column)
        gw = W // cfg.patch
        col = torch.arange(self.n_tokens, device=x.device) % gw
        left = (col < gw // 2)
        h = h + torch.where(left[None, :, None], self.panel[0], self.panel[1])
        cond = self.t_mlp(sinusoidal(t * 1000.0, cfg.d_model))
        if task is not None:
            cond = cond + self.task_emb(task)
        for blk in self.blocks:
            h = blk(h, cond, ctx)
        s, sc = self.adaln_f(cond).chunk(2, dim=-1)
        h = modulate(self.ln_f(h), s, sc)
        h = self.head(h)                                        # (B, N, p*p*out)
        # unpatchify
        gh = H // cfg.patch
        h = h.view(B, gh, gw, cfg.patch, cfg.patch, cfg.out_channels)
        h = h.permute(0, 5, 1, 3, 2, 4).reshape(B, cfg.out_channels, H, W)
        return h


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# 512px: DC-AE f32 -> 16x16x32 per panel; two-panel canvas 16x32; 512 tokens.
V0_DCAE_512 = DiTConfig(in_channels=34, out_channels=32, latent_h=16, panel_w=16, patch=1,
                        d_model=1024, depth=16, heads=16, mlp_ratio=3.0, ctx_dim=1024)
# 256px pilot: 8x8x32 per panel; canvas 8x16; 128 tokens.
V0_DCAE_256 = DiTConfig(in_channels=34, out_channels=32, latent_h=8, panel_w=8, patch=1,
                        d_model=1024, depth=16, heads=16, mlp_ratio=3.0, ctx_dim=1024)


if __name__ == "__main__":
    for name, cfg in [("512", V0_DCAE_512), ("256", V0_DCAE_256)]:
        m = HobbyImageDiT(cfg)
        x = torch.randn(2, cfg.in_channels, cfg.latent_h, 2 * cfg.panel_w)
        t = torch.rand(2); ctx = torch.randn(2, 256, cfg.ctx_dim); task = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            y = m(x, t, ctx, task)
        print(f"DiT {name}: {count_params(m)/1e6:.1f}M params | {m.n_tokens} tokens | "
              f"in {tuple(x.shape)} -> out {tuple(y.shape)}")
