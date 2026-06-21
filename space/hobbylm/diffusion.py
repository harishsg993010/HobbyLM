"""Pure masked-diffusion (LLaDA / MDLM) conversion utilities for the MoE LLM.

The model itself only flips one thing for diffusion: attention becomes bidirectional
(model.py, gated on cfg.diffusion). Everything else lives here:

  forward_mask : the forward (noising) process used at train time.
  generate     : iterative-denoising sampler with semi-autoregressive blocks.

The TRAIN loss is model.diffusion_cross_entropy (fused/chunked for big batches). The
unfused `diffusion_loss` below is for tests / sanity checks only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def forward_mask(input_ids: Tensor, mask_id: int, eps: float = 1e-3,
                 prompt_lens: Tensor | None = None, generator: torch.Generator | None = None):
    """LLaDA forward process.

    One mask ratio t ~ U(eps, 1) per sequence; mask each token iid with prob t.
    Returns (noisy, labels, p_mask):
      noisy  : input with masked positions replaced by mask_id
      labels : original token at masked positions, -1 (ignore_index) elsewhere
      p_mask : per-token mask probability t (broadcast), used for the 1/p reweighting
    `prompt_lens` (B,) optionally protects a prompt prefix from being masked/scored (SFT).
    """
    b, l = input_ids.shape
    dev = input_ids.device
    t = torch.rand(b, device=dev, generator=generator) * (1 - eps) + eps   # (b,)
    p_mask = t[:, None].expand(b, l).contiguous()                          # (b, l)
    mask = torch.rand(b, l, device=dev, generator=generator) < p_mask
    if prompt_lens is not None:
        pos = torch.arange(l, device=dev)[None, :]
        mask &= pos >= prompt_lens[:, None]
    # guarantee >=1 masked token per sequence so no micro-batch contributes a zero loss
    none_masked = ~mask.any(dim=1)
    if none_masked.any():
        rows = none_masked.nonzero(as_tuple=True)[0]
        lo = 0 if prompt_lens is None else int(prompt_lens.min().item())
        j = torch.randint(lo, l, (rows.numel(),), device=dev, generator=generator)
        mask[rows, j] = True
    noisy = torch.where(mask, torch.full_like(input_ids, mask_id), input_ids)
    labels = torch.where(mask, input_ids, torch.full_like(input_ids, -1))
    return noisy, labels, p_mask


def diffusion_loss(logits: Tensor, labels: Tensor, p_mask: Tensor) -> Tensor:
    """Unfused LLaDA loss (tests): sum_{masked} CE / p_mask, normalized by B*L."""
    b, l, _ = logits.shape
    m = labels != -1
    if int(m.sum()) == 0:
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits[m].float(), labels[m], reduction="none")
    return (ce / p_mask[m]).sum() / (b * l)


def get_num_transfer_tokens(n: int, steps: int) -> list[int]:
    """Spread n unmask events as evenly as possible across `steps` (sums to n)."""
    base = n // steps
    out = [base] * steps
    for i in range(n - base * steps):
        out[i] += 1
    return out


def add_gumbel_noise(logits: Tensor, temperature: float,
                     generator: torch.Generator | None = None) -> Tensor:
    """Gumbel-max categorical sampling (LLaDA). argmax of this == a sample at `temperature`.
    temperature<=0 -> identity (argmax == greedy)."""
    if temperature <= 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand(logits.shape, dtype=torch.float64, device=logits.device, generator=generator)
    gumbel = (-torch.log(noise + 1e-12)) ** temperature
    return logits.exp() / gumbel


def _rep_penalty(blk: Tensor, present_ids: Tensor, penalty: float) -> Tensor:
    """CTRL-style penalty across the canvas: damp logits of tokens already present (prompt +
    committed) so the denoiser stops filling many slots with the same token. In-place on blk."""
    if penalty == 1.0 or present_ids.numel() == 0:
        return blk
    col = blk[:, present_ids]
    blk[:, present_ids] = torch.where(col > 0, col / penalty, col * penalty)
    return blk


@torch.no_grad()
def generate(model, prompt_ids: Tensor, gen_len: int = 256, block: int = 32, steps: int = 64,
             mask_id: int = 50257, temperature: float = 0.0, rep_penalty: float = 1.0,
             remask_steps: int = 0, remask_frac: float = 0.3, valid_vocab: int = 50257,
             eos_id: int | None = None, generator: torch.Generator | None = None) -> Tensor:
    """Semi-autoregressive iterative denoising. prompt_ids: (1, P). Returns generated ids (1, <=gen_len).

    Each block of `block` masked slots is filled over ~`steps*block/gen_len` steps, committing the
    highest-confidence still-masked positions each step (low-confidence-remasking selection). Then
    `remask_steps` refinement passes re-mask the lowest-confidence committed tokens and re-predict
    them with full bidirectional context — this is what lets the model fix repetition/mistakes.
    Sentinels (>= valid_vocab, incl. mask_id) are banned from being emitted. Blocks are causal
    w.r.t. each other (a block attends to the committed prefix + itself), bidirectional within.
    """
    was_training = model.training
    model.eval()
    dev = prompt_ids.device
    x = torch.cat([prompt_ids, torch.full((1, gen_len), mask_id, device=dev, dtype=prompt_ids.dtype)], dim=1)
    P = prompt_ids.shape[1]

    def block_logits(b1: int, b0: int) -> Tensor:
        """Forward prefix+block; return (blk_len, V) logits with sentinels banned + rep-penalty."""
        logits, _ = model(x[:, :b1])
        blk = logits[0, b0:b1].float()
        blk[:, valid_vocab:] = -float("inf")                  # never emit mask/sentinel ids
        present = torch.unique(x[0, :b1])
        present = present[(present < valid_vocab) & (present != mask_id)]
        return _rep_penalty(blk, present, rep_penalty)

    def predict(blk: Tensor):
        prob = blk.softmax(-1)
        pred = add_gumbel_noise(blk, temperature, generator).argmax(-1) if temperature > 0 else blk.argmax(-1)
        return pred, prob

    for b0 in range(P, P + gen_len, block):
        b1 = min(b0 + block, P + gen_len)
        blk_len = b1 - b0
        sb = max(1, round(steps * blk_len / gen_len))
        sched = get_num_transfer_tokens(blk_len, sb)
        # --- fill: commit the most-confident still-masked positions over sb steps ---
        for s in range(sb):
            pred, prob = predict(block_logits(b1, b0))
            conf = prob.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            still = x[0, b0:b1] == mask_id
            conf = torch.where(still, conf, torch.full_like(conf, -1.0))
            k = min(sched[s], int(still.sum()))
            if k <= 0:
                continue
            idx = conf.topk(k).indices
            x[0, b0 + idx] = pred[idx].to(x.dtype)
        # --- refine: re-mask the least-confident committed tokens and re-predict them ---
        for _ in range(remask_steps):
            blk = block_logits(b1, b0)
            prob = blk.softmax(-1)
            cur = x[0, b0:b1]
            cur_conf = prob.gather(-1, cur.unsqueeze(-1)).squeeze(-1)   # confidence in current tokens
            r = max(1, int(blk_len * remask_frac))
            x[0, b0 + cur_conf.topk(r, largest=False).indices] = mask_id
            pred, _ = predict(block_logits(b1, b0))
            still = (x[0, b0:b1] == mask_id).nonzero(as_tuple=True)[0]
            x[0, b0 + still] = pred[still].to(x.dtype)
        if eos_id is not None and bool((x[0, b0:b1] == eos_id).any()):
            rel = int((x[0, b0:b1] == eos_id).nonzero(as_tuple=True)[0][0].item())
            if was_training:
                model.train()
            return x[:, P:b0 + rel + 1]
    if was_training:
        model.train()
    return x[:, P:]
