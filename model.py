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


# -----------------------------------------------------------------------------
# FP8 matmul for the lm_head (the single largest GEMM). Adapted from modded-nanogpt
# (@YouJiacheng): weight stored transposed (in, out) so the gradient w.r.t. the weight
# lands in the natural layout. Forward in e4m3, backward grad in e5m2. bf16 outputs.
# Wrapped as a custom op with an explicit autograd rule (torch._scaled_mm is not
# differentiable on its own).

@torch.library.custom_op("moelab::mm_t", mutates_args=())
def _mm_t(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    """y = x @ w with x:(M,in), w:(in,out). Returns (y_bf16, x_f8, w_f8) for backward reuse."""
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        w_col = w_f8.T.contiguous().T  # _scaled_mm needs column-major B
        out = torch._scaled_mm(x_f8, w_col, out_dtype=torch.bfloat16,
                               scale_a=x.new_tensor(x_s, dtype=torch.float32),
                               scale_b=x.new_tensor(w_s, dtype=torch.float32),
                               use_fast_accum=True)
        return out, x_f8, w_f8
    return impl(x, w)


@_mm_t.register_fake
def _(x: Tensor, w: Tensor, *_):
    return x @ w, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)


@torch.library.custom_op("moelab::mm_t_backward", mutates_args=())
def _mm_t_backward(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(g: Tensor, x_f8: Tensor, w_f8: Tensor):
        x_scale = g.new_tensor(x_s, dtype=torch.float32)
        w_scale = g.new_tensor(w_s, dtype=torch.float32)
        g_scale = g.new_tensor(grad_s, dtype=torch.float32)
        g_f8 = g.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(g_f8, w_f8.T, out_dtype=torch.bfloat16,
                                  scale_a=g_scale, scale_b=w_scale, use_fast_accum=False)
        grad_w = torch._scaled_mm(x_f8.T.contiguous(), g_f8.T.contiguous().T, out_dtype=torch.float32,
                                  scale_a=x_scale, scale_b=g_scale, use_fast_accum=False)
        return grad_x, grad_w
    return impl(g, x_f8, w_f8)


@_mm_t_backward.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)


def _mm_t_setup(ctx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = (x_s, w_s, grad_s)
    ctx.set_materialize_grads(False)


def _mm_t_bwd(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    gx, gw = torch.ops.moelab.mm_t_backward(grad_out, x_f8, w_f8, x_s, w_s, grad_s)
    return gx, gw, None, None, None


_mm_t.register_autograd(_mm_t_bwd, setup_context=_mm_t_setup)


class FP8Linear(nn.Module):
    """Bias-free linear with FP8 matmul in training (CUDA) and a bf16 fallback elsewhere.
    Weight is stored transposed (in_features, out_features)."""
    def __init__(self, in_features: int, out_features: int, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.x_s, self.w_s, self.grad_s = x_s, w_s, grad_s
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.is_cuda:
            flat = x.flatten(0, -2).bfloat16().contiguous()
            out = torch.ops.moelab.mm_t(flat, self.weight.bfloat16().contiguous(),
                                        self.x_s, self.w_s, self.grad_s)[0]
            return out.reshape(*x.shape[:-1], self.out_features)
        return x @ self.weight.type_as(x)


# -----------------------------------------------------------------------------
# Fused cross-entropy: process the vocab projection in row-chunks under activation
# checkpointing so the full (T, vocab) fp32 logit tensor is never materialized or
# saved for backward. Numerically identical to a plain CE + final z-loss.

def _ce_chunk(x_c: Tensor, weight: Tensor, tgt_c: Tensor, softcap: float, tied: bool):
    logits = (x_c @ weight.T) if tied else (x_c @ weight)   # tied: weight (V,d); fp8 head: weight (d,V)
    if softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1)                   # (c,)
    z_sum = (lse * lse).sum()
    valid = (tgt_c != -1)
    tgt_logit = logits.gather(-1, tgt_c.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    ce_sum = ((lse - tgt_logit) * valid).sum()
    return ce_sum, z_sum


def fused_cross_entropy(x: Tensor, weight: Tensor, targets: Tensor, *,
                        z_coef: float, softcap: float, chunk: int, tied: bool):
    """Returns (loss = mean_ce + z_coef * mean(lse^2), z_mean.detach()). Memory-light."""
    from torch.utils.checkpoint import checkpoint
    T = x.shape[0]
    n_valid = (targets != -1).sum().clamp_min(1)
    ce_sum = x.new_zeros((), dtype=torch.float32)
    z_sum = x.new_zeros((), dtype=torch.float32)
    for i in range(0, T, chunk):
        c_ce, c_z = checkpoint(_ce_chunk, x[i:i + chunk], weight, targets[i:i + chunk],
                               softcap, tied, use_reentrant=False)
        ce_sum = ce_sum + c_ce
        z_sum = z_sum + c_z
    ce = ce_sum / n_valid
    z_mean = z_sum / T
    return ce + z_coef * z_mean, z_mean.detach()


# Masked-diffusion (LLaDA/MDLM) loss: cross-entropy on the masked positions only, reweighted
# by 1/p_mask, summed and normalized by the total token count (B*L). Chunked + activation-
# checkpointed like fused_cross_entropy so the (T, vocab) logits never fully materialize.

def _dce_chunk(x_c: Tensor, weight: Tensor, tgt_c: Tensor, pm_c: Tensor, softcap: float, tied: bool):
    logits = (x_c @ weight.T) if tied else (x_c @ weight)
    if softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1)
    valid = (tgt_c != -1)                                    # unmasked positions carry target -1
    tgt_logit = logits.gather(-1, tgt_c.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    ce = (lse - tgt_logit) * valid / pm_c                    # 1/p_mask reweight; unmasked -> 0
    return ce.sum()


def diffusion_cross_entropy(x: Tensor, weight: Tensor, targets: Tensor, p_mask: Tensor, *,
                            softcap: float, chunk: int, tied: bool):
    """LLaDA loss = sum_{masked} CE / p_mask, normalized by B*L. Memory-light."""
    from torch.utils.checkpoint import checkpoint
    T = x.shape[0]
    ce_sum = x.new_zeros((), dtype=torch.float32)
    for i in range(0, T, chunk):
        ce_sum = ce_sum + checkpoint(_dce_chunk, x[i:i + chunk], weight, targets[i:i + chunk],
                                     p_mask[i:i + chunk], softcap, tied, use_reentrant=False)
    return ce_sum / T


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
        # diffusion (LLaDA) models see the whole noised canvas -> bidirectional; AR stays causal.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=not self.cfg.diffusion)
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
        # FP8 head must be untied (it stores the weight transposed, (d, vocab), for fp8 grad layout).
        self.fp8_head = cfg.fp8_head
        if cfg.fp8_head:
            import warnings
            warnings.warn("fp8_head is EXPERIMENTAL/BROKEN: zero gradient flows through the FP8 "
                          "backward (model won't train) and it gave no speedup. Use fused_ce instead.")
            self.lm_head = FP8Linear(cfg.d_model, cfg.vocab_size,
                                     x_s=cfg.fp8_x_scale, w_s=cfg.fp8_w_scale, grad_s=cfg.fp8_grad_scale)
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.embed.weight
        self._rope_cache: dict = {}
        self.apply(self._init)
        if cfg.fp8_head:
            # start the untied head from the (tied) embedding weights so the ablation isolates fp8,
            # not a different head initialization.
            with torch.no_grad():
                self.lm_head.weight.copy_(self.embed.weight.t())
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

    def forward(self, idx: Tensor | None = None, targets: Tensor | None = None,
                inputs_embeds: Tensor | None = None, p_mask: Tensor | None = None):
        # accept either token ids OR precomputed embeddings (inputs_embeds), for multimodal splicing.
        if inputs_embeds is None:
            x = self.embed(idx)
            if self.cfg.scale_embeddings:
                x = x * (self.cfg.d_model ** 0.5)
            device = idx.device
        else:
            x = inputs_embeds
            device = inputs_embeds.device
        B, S = x.shape[0], x.shape[1]
        cos, sin = self.rope(S, device, x.dtype)
        aux_sum = x.new_zeros(())
        for blk in self.blocks:
            x, aux = blk(x, cos, sin)
            aux_sum = aux_sum + aux
        x = self.final_norm(x)
        cfg = self.cfg
        sc = cfg.logit_softcap

        # ---- inference: return logits ----
        if targets is None:
            logits = self.lm_head(x)
            if sc > 0:
                logits = sc * torch.tanh(logits / sc)
            return logits, aux_sum

        # ---- training: compute loss ----
        if cfg.diffusion:
            # LLaDA masked-diffusion loss on the noised positions (input already masked upstream;
            # targets hold the original token at masked positions, -1 elsewhere; p_mask = per-token t).
            assert p_mask is not None, "diffusion forward needs p_mask (use diffusion.forward_mask)"
            loss_ce = diffusion_cross_entropy(
                x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1),
                p_mask.reshape(-1), softcap=sc, chunk=cfg.ce_chunk, tied=not cfg.fp8_head)
            loss = loss_ce + aux_sum
            return loss, {"ce": loss_ce.detach(), "aux": aux_sum.detach(), "z": x.new_zeros(())}

        if cfg.fused_ce and not cfg.fp8_head:
            # chunked CE on the tied weight (V, d); never materializes the full fp32 logits.
            loss_cez, z = fused_cross_entropy(
                x.reshape(-1, x.size(-1)), self.lm_head.weight, targets.reshape(-1),
                z_coef=cfg.final_z_loss_coef, softcap=sc, chunk=cfg.ce_chunk, tied=True)
            loss = loss_cez + aux_sum
            ce = (loss_cez - cfg.final_z_loss_coef * z).detach()
            return loss, {"ce": ce, "aux": aux_sum.detach(), "z": z}

        logits = self.lm_head(x)                       # fp8 head -> bf16, else nn.Linear
        if sc > 0:
            logits = sc * torch.tanh(logits / sc)
        logits = logits.float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        z = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        loss = ce + cfg.final_z_loss_coef * z + aux_sum
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
    # subtract tied head double-count is already avoided (shared weight counted once).
    # fp8_head forces an untied head, so it costs a second vocab x d_model matrix.
    tied = cfg.tie_embeddings and not cfg.fp8_head
    embed = cfg.vocab_size * cfg.d_model * (1 if tied else 2)
    # active = total - inactive routed experts. Per MoE layer, only top_k of n_experts run.
    per_expert = cfg.d_model * 2 * cfg.expert_ffn + cfg.expert_ffn * cfg.d_model
    inactive_per_moe = (cfg.n_experts - cfg.top_k) * per_expert
    active = total - cfg.n_moe_layers * inactive_per_moe
    return {"total": total, "active": active, "embed": embed,
            "active_pct": 100 * active / total, "sparsity": total / active}
