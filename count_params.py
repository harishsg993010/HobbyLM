"""Param counter + local CPU smoke test. Validates shapes, param targets, and a fwd/bwd pass.

Usage:
  python count_params.py            # count all presets
  python count_params.py --smoke    # also run a tiny CPU forward/backward (bmm backend)
"""
import argparse
import torch

from config import PRESETS, get_config
from model import MoETransformer, count_params


def fmt(n: int) -> str:
    return f"{n/1e6:.1f}M" if n < 1e9 else f"{n/1e9:.3f}B"


def report():
    print(f"{'preset':>7} | {'total':>9} | {'active':>9} | {'act%':>6} | {'sparsity':>8} | {'embed':>8}")
    print("-" * 64)
    for name in PRESETS:
        cfg = get_config(name)
        cfg.expert_backend = "bmm"
        model = MoETransformer(cfg)
        c = count_params(model)
        print(f"{name:>7} | {fmt(c['total']):>9} | {fmt(c['active']):>9} | "
              f"{c['active_pct']:>5.1f}% | {c['sparsity']:>7.2f}x | {fmt(c['embed']):>8}")
        del model


def smoke():
    print("\n=== CPU smoke test (130M, bmm backend) ===")
    torch.manual_seed(0)
    cfg = get_config("130M")
    cfg.expert_backend = "bmm"
    cfg.n_layers, cfg.n_experts, cfg.d_model = 4, 8, 128  # shrink for speed
    cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim = 4, 2, 32
    cfg.dense_ffn, cfg.expert_ffn = 256, 64
    model = MoETransformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    tgt = torch.randint(0, cfg.vocab_size, (2, 16))
    loss, parts = model(idx, tgt)
    loss.backward()
    gnorm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"loss={loss.item():.4f}  ce={parts['ce'].item():.4f}  "
          f"aux={parts['aux'].item():.4e}  z={parts['z'].item():.4f}  grad_norm={gnorm:.3f}")
    # check bias updated
    b = model.blocks[-1].ffn.expert_bias
    print(f"expert_bias updated: nonzero={int((b != 0).sum())}/{b.numel()}  range=[{b.min():.4f},{b.max():.4f}]")
    print("OK" if torch.isfinite(loss) else "FAIL: non-finite loss")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    report()
    if args.smoke:
        smoke()
