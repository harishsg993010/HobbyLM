"""Top-k Sparse Autoencoder for mechanistic interpretability of HobbyLM activations.

A top-k SAE (Gao et al. 2024 / EleutherAI `sae`) over the residual stream: it reconstructs an
activation x as a sparse, non-negative combination of a learned overcomplete dictionary, keeping only
the k largest latents active. No L1 coefficient to tune (sparsity is exactly k), and an auxiliary loss
resurrects dead features so the dictionary stays fully used.

  x_centered = x - b_dec
  z          = TopK( relu(x_centered @ W_enc + b_enc) )        # exactly k non-zeros
  x_hat      = z @ W_dec + b_dec                                 # W_dec rows are unit-norm
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SAEConfig:
    d_in: int = 768          # model activation dim (residual stream)
    d_sae: int = 12288       # dictionary size (16x expansion)
    k: int = 32              # active latents per token
    k_aux: int = 256         # dead-latent auxiliary reconstruction width
    dead_after: int = 2_000_000   # a feature unused for this many tokens is "dead"
    aux_coef: float = 1.0 / 32.0  # weight on the auxk resurrection loss


def _topk(pre: Tensor, k: int) -> Tensor:
    """Keep the k largest values per row (after the relu in the caller), zero the rest."""
    vals, idx = pre.topk(k, dim=-1)
    out = torch.zeros_like(pre)
    return out.scatter_(-1, idx, vals)


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d, m = cfg.d_in, cfg.d_sae
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.b_enc = nn.Parameter(torch.zeros(m))
        # decoder rows unit-norm; encoder initialised as the decoder transpose (standard tied init)
        W_dec = F.normalize(torch.randn(m, d), dim=1)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())
        # tokens since each latent last fired (for dead-feature tracking); not a learned param
        self.register_buffer("last_fired", torch.zeros(m, dtype=torch.long))
        self.register_buffer("n_seen", torch.zeros((), dtype=torch.long))

    # ---- encode / decode ----
    def encode(self, x: Tensor) -> Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return _topk(F.relu(pre), self.cfg.k)            # (..., d_sae), exactly k non-zeros

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: Tensor):
        """Returns (x_hat, z, loss_dict). x: (N, d_in)."""
        pre = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        z = _topk(pre, self.cfg.k)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, x)

        # ---- dead-feature bookkeeping + auxk resurrection ----
        with torch.no_grad():
            fired = (z > 0).any(dim=0)                    # (d_sae,)
            self.last_fired += x.shape[0]
            self.last_fired[fired] = 0
            self.n_seen += x.shape[0]
            dead = self.last_fired > self.cfg.dead_after
        aux = x.new_zeros(())
        n_dead = int(dead.sum())
        if n_dead > 0 and self.cfg.k_aux > 0:
            # reconstruct the residual error using only the top-k_aux DEAD latents (revives them)
            resid = x - x_hat.detach()
            pre_dead = pre.masked_fill(~dead[None, :], 0.0)
            z_aux = _topk(pre_dead, min(self.cfg.k_aux, n_dead))
            resid_hat = z_aux @ self.W_dec                # no b_dec: model the centred residual
            aux = F.mse_loss(resid_hat, resid)
            loss = recon + self.cfg.aux_coef * aux
        else:
            loss = recon
        return x_hat, z, {"loss": loss, "recon": recon.detach(), "aux": aux.detach(),
                          "n_dead": n_dead}

    # ---- keep decoder rows unit-norm (call after each optimizer step) ----
    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def set_decoder_to_geometric_mean(self, x: Tensor):
        """Initialise b_dec to the data mean (standard) — call once on the first activation batch."""
        self.b_dec.data = x.mean(dim=0)


def fraction_variance_explained(x: Tensor, x_hat: Tensor) -> float:
    """1 - ||x - x_hat||^2 / ||x - mean(x)||^2  (per-batch FVU complement)."""
    num = (x - x_hat).pow(2).sum()
    den = (x - x.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
    return float((1.0 - num / den).detach())
