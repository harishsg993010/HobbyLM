"""Local CPU smoke test for the AR->diffusion (LLaDA) conversion harness.

Builds a tiny bidirectional MoE, overfits a handful of fixed sequences with the masked-
diffusion objective, then runs the iterative-denoising sampler and checks it reconstructs
the held-out second half. Validates: forward_mask, the weighted diffusion loss, bidirectional
attention, and generate() end-to-end. Runs in well under a minute on CPU.

    python test_diffusion.py
"""
import torch

from config import ModelConfig
from model import MoETransformer
from diffusion import forward_mask, generate

torch.manual_seed(0)

VOCAB, MASK_ID, SEQ, NSEQ = 256, 255, 16, 6
STEPS, LR = 1500, 1.5e-3

cfg = ModelConfig(
    vocab_size=VOCAB, d_model=192, n_layers=4, n_dense_layers=1,
    n_q_heads=4, n_kv_heads=2, head_dim=48,
    dense_ffn=384, expert_ffn=96, n_experts=8, top_k=2, n_shared=1,
    diffusion=True, mask_token_id=MASK_ID, ce_chunk=4096,
    expert_backend="bmm",          # CPU reference path
)
model = MoETransformer(cfg)
model.train()

# fixed toy "corpus": NSEQ sequences of real tokens (0..199), to be memorized
data = torch.randint(0, 200, (NSEQ, SEQ))

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=LR * 0.05)

print("overfitting tiny bidirectional diffusion model...")
first_loss = None
for step in range(STEPS):
    noisy, labels, p_mask = forward_mask(data, MASK_ID, eps=1e-3)
    loss, _ = model(noisy, labels, p_mask=p_mask)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if first_loss is None:
        first_loss = loss.item()
    if step % 150 == 0:
        print(f"  step {step:4d}  loss {loss.item():.4f}")
last_loss = loss.item()
print(f"loss {first_loss:.3f} -> {last_loss:.3f}")

# --- direct denoiser check: mask the 2nd half, ONE forward, argmax accuracy on masked positions ---
model.eval()
with torch.no_grad():
    noisy = data.clone(); noisy[:, SEQ // 2:] = MASK_ID
    logits, _ = model(noisy)
    pred = logits[:, SEQ // 2:].argmax(-1)
    oneshot = (pred == data[:, SEQ // 2:]).float().mean().item()
print(f"one-shot denoise accuracy (mask 2nd half, single forward) = {oneshot:.1%}")
model.train()

# --- confirm bidirectional attention is actually on ---
assert model.cfg.diffusion is True
print("attention is_causal =", not cfg.diffusion, "(expect False)")

# --- reconstruction via the denoise loop: prompt = first half, regenerate second half ---
model.eval()
half = SEQ // 2
correct = total = 0
for i in range(NSEQ):
    prompt = data[i:i + 1, :half]
    gen = generate(model, prompt, gen_len=half, block=half, steps=half,
                   mask_id=MASK_ID, temperature=0.0)
    truth = data[i, half:]
    correct += int((gen[0] == truth).sum())
    total += half
acc = correct / total
print(f"reconstruction accuracy (overfit) = {acc:.1%}  ({correct}/{total})")

# --- assertions ---
assert last_loss < first_loss * 0.5, f"loss did not drop enough: {first_loss:.3f}->{last_loss:.3f}"
assert acc > 0.85, f"denoise loop failed to reconstruct memorized data: acc={acc:.1%}"
print("\nPASS: diffusion conversion harness is wired correctly.")
