"""DreamLite-style in-context latent U-Net (V0, 512px) — flow-matching velocity prediction.

Operates on a TWO-PANEL latent canvas concatenated along width:
  generation: [ noisy_target(64x64) | blank(64x64) ]   -> 64x128
  editing:    [ noisy_target(64x64) | source_latent ]   -> 64x128
At f8c16 a 512px image -> 64x64x16 latent; the panel pair is 64x128. Optional 2 mask channels
(edit/preserve) make the per-panel input 18ch. Output is 16ch velocity over the full canvas; only
the LEFT (target) half is used for the loss.

Conditioning: sinusoidal timestep embedding + task embedding (generate/edit/noop) + panel position
embedding, and CROSS-ATTENTION to the frozen-VLM refiner tokens C_vlm (B, 256, 1024). GQA cross-
attention with QK/RMS norm; NO self-attention at the highest (64x128) resolution (DreamLite trick).

Pure PyTorch, from-scratch (matches the repo's llm.c ethos). See moe_dreamlite_hq_architecture_*.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class UNetConfig:
    in_channels: int = 18           # 16 latent + 2 optional mask channels
    out_channels: int = 16          # velocity over the 16-ch latent
    stem_channels: int = 256
    block_channels: tuple = (256, 512, 768)        # per-stage widths
    resblocks_per_stage: int = 2
    transformer_per_stage: tuple = (0, 2, 4)       # cross/self-attn blocks; 0 at the 64x128 stage
    mid_transformer: int = 1
    head_dim: int = 64
    kv_heads: int = 2               # GQA
    ctx_dim: int = 1024             # VLM refiner token dim
    mlp_ratio: float = 2.0
    n_tasks: int = 3                # generate / edit / noop
    temb_dim: int = 1024
    qk_norm: bool = True


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None, :]
    return torch.cat([a.cos(), a.sin()], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv x2 with additive timestep conditioning, residual."""
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Optional self-attention (skipped at high res) + GQA cross-attention to C_vlm + MLP, on the
    flattened (B, H*W, C) feature map. QK-RMS-norm, flash SDPA."""
    def __init__(self, dim: int, cfg: UNetConfig, self_attn: bool):
        super().__init__()
        self.dim = dim
        self.heads = dim // cfg.head_dim
        self.kv_heads = cfg.kv_heads
        self.hd = cfg.head_dim
        self.self_attn = self_attn
        if self_attn:
            self.sn = nn.LayerNorm(dim)
            self.s_qkv = nn.Linear(dim, dim + 2 * self.kv_heads * self.hd, bias=False)
            self.s_o = nn.Linear(dim, dim, bias=False)
            self.s_qn = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
            self.s_kn = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.cn = nn.LayerNorm(dim)
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_kv = nn.Linear(cfg.ctx_dim, 2 * self.kv_heads * self.hd, bias=False)
        self.c_o = nn.Linear(dim, dim, bias=False)
        self.c_qn = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.c_kn = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        hidden = int(dim * cfg.mlp_ratio)
        self.mn = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _gqa(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, Nq = q.shape[0], q.shape[1]
        q = q.view(B, Nq, self.heads, self.hd).transpose(1, 2)
        k = k.view(B, -1, self.kv_heads, self.hd).transpose(1, 2)
        v = v.view(B, -1, self.kv_heads, self.hd).transpose(1, 2)
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v)
        return o.transpose(1, 2).reshape(B, Nq, self.dim)

    def forward(self, x: Tensor, ctx: Tensor, hw: tuple) -> Tensor:
        B, C, H, W = x.shape
        h = x.flatten(2).transpose(1, 2)                 # (B, H*W, C)
        if self.self_attn:
            y = self.sn(h)
            qkv = self.s_qkv(y)
            q, k, v = qkv.split([self.dim, self.kv_heads * self.hd, self.kv_heads * self.hd], dim=-1)
            q = self.s_qn(q.view(B, -1, self.heads, self.hd)).flatten(2)
            k = self.s_kn(k.view(B, -1, self.kv_heads, self.hd)).flatten(2)
            h = h + self.s_o(self._gqa(q, k, v))
        # cross-attention to VLM tokens
        y = self.cn(h)
        q = self.c_qn(self.c_q(y).view(B, -1, self.heads, self.hd)).flatten(2)
        kv = self.c_kv(ctx)
        k, v = kv.split([self.kv_heads * self.hd, self.kv_heads * self.hd], dim=-1)
        k = self.c_kn(k.view(B, -1, self.kv_heads, self.hd)).flatten(2)
        h = h + self.c_o(self._gqa(q, k, v))
        h = h + self.mlp(self.mn(h))
        return h.transpose(1, 2).reshape(B, C, H, W)


class DreamLiteUNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        td = cfg.temb_dim
        self.t_mlp = nn.Sequential(nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        self.task_emb = nn.Embedding(cfg.n_tasks, td)
        self.panel_emb = nn.Parameter(torch.zeros(2, cfg.stem_channels))  # left/right panel bias
        self.stem = nn.Conv2d(cfg.in_channels, cfg.stem_channels, 3, padding=1)

        chans = cfg.block_channels
        # ---- encoder ----
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = cfg.stem_channels
        skip_chs = [cfg.stem_channels]
        for si, ch in enumerate(chans):
            res = nn.ModuleList(); attn = nn.ModuleList()
            for _ in range(cfg.resblocks_per_stage):
                res.append(ResBlock(prev, ch, td)); prev = ch
                skip_chs.append(ch)
            for _ in range(cfg.transformer_per_stage[si]):
                attn.append(AttnBlock(ch, cfg, self_attn=(si >= 1)))
            self.down_res.append(res); self.down_attn.append(attn)
            if si < len(chans) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_chs.append(ch)
            else:
                self.downsamples.append(nn.Identity())

        # ---- mid ----
        mc = chans[-1]
        self.mid_res1 = ResBlock(mc, mc, td)
        self.mid_attn = nn.ModuleList([AttnBlock(mc, cfg, self_attn=True) for _ in range(cfg.mid_transformer)])
        self.mid_res2 = ResBlock(mc, mc, td)

        # ---- decoder (mirror, with skips) ----
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for si, ch in reversed(list(enumerate(chans))):
            res = nn.ModuleList(); attn = nn.ModuleList()
            n_res = cfg.resblocks_per_stage + 1   # +1 consumes the downsample/stem skip at this res
            for _ in range(n_res):
                res.append(ResBlock(prev + skip_chs.pop(), ch, td)); prev = ch
            for _ in range(cfg.transformer_per_stage[si]):
                attn.append(AttnBlock(ch, cfg, self_attn=(si >= 1)))
            self.up_res.append(res); self.up_attn.append(attn)
            if si > 0:
                self.upsamples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(32, prev)
        self.out_conv = nn.Conv2d(prev, cfg.out_channels, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor, ctx: Tensor, task: Tensor | None = None) -> Tensor:
        """x: (B, in_ch, 64, 128) two-panel latent. t: (B,) in [0,1]. ctx: (B, M, ctx_dim) VLM tokens.
        task: (B,) long in [0,n_tasks). Returns velocity (B, out_ch, 64, 128)."""
        cfg = self.cfg
        temb = self.t_mlp(sinusoidal_embedding(t * 1000.0, cfg.temb_dim))
        if task is not None:
            temb = temb + self.task_emb(task)
        h = self.stem(x)
        # panel position bias: left half gets panel_emb[0], right half panel_emb[1]
        W = h.shape[-1]
        h[:, :, :, :W // 2] += self.panel_emb[0][None, :, None, None]
        h[:, :, :, W // 2:] += self.panel_emb[1][None, :, None, None]

        skips = [h]
        for si in range(len(cfg.block_channels)):
            for rb in self.down_res[si]:
                h = rb(h, temb); skips.append(h)
            for ab in self.down_attn[si]:
                h = ab(h, ctx, h.shape[-2:])
            h = self.downsamples[si](h)
            if not isinstance(self.downsamples[si], nn.Identity):
                skips.append(h)

        h = self.mid_res1(h, temb)
        for ab in self.mid_attn:
            h = ab(h, ctx, h.shape[-2:])
        h = self.mid_res2(h, temb)

        for ui in range(len(cfg.block_channels)):
            si = len(cfg.block_channels) - 1 - ui
            for rb in self.up_res[ui]:
                h = rb(torch.cat([h, skips.pop()], dim=1), temb)
            for ab in self.up_attn[ui]:
                h = ab(h, ctx, h.shape[-2:])
            h = self.upsamples[ui](h)

        return self.out_conv(F.silu(self.out_norm(h)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


V0_512 = UNetConfig(
    in_channels=18, out_channels=16, stem_channels=384,
    block_channels=(384, 768, 1024), resblocks_per_stage=2,
    transformer_per_stage=(0, 2, 4), mid_transformer=1,
    head_dim=64, kv_heads=2, ctx_dim=1024, mlp_ratio=2.0,
)


if __name__ == "__main__":
    cfg = V0_512
    m = DreamLiteUNet(cfg)
    n = count_params(m)
    print(f"DreamLite V0-512 U-Net: {n/1e6:.1f}M params (target ~390M)")
    x = torch.randn(1, cfg.in_channels, 64, 128)
    t = torch.rand(1)
    ctx = torch.randn(1, 256, cfg.ctx_dim)
    task = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y = m(x, t, ctx, task)
    print(f"forward ok: in {tuple(x.shape)} -> out {tuple(y.shape)} (expect (1,16,64,128))")
