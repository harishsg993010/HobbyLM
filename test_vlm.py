"""CPU tests for the multimodal plumbing (Phase 0.5). Run: python test_vlm.py

Checks:
  1. model.forward(inputs_embeds=...) == forward(idx=...) for text-only (refactor is transparent).
  2. MoEVLM splices image features at the IMAGE_TOKEN sentinel: correct merged length, image
     positions get IGNORE_INDEX targets, loss is finite, gradient flows to the projector.
  3. Audio splice + image+audio together.
  4. Mixed batch (image sample + text-only sample) via right-padding.
  5. Stage-1 freeze: only projector params require grad.
"""
import torch

from config import get_config
from model import MoETransformer
from multimodal import MoEVLM, IMAGE_TOKEN, AUDIO_TOKEN, IGNORE_INDEX


def tiny_cfg():
    c = get_config("500M")
    c.expert_backend = "bmm"
    c.n_layers, c.n_experts, c.top_k, c.d_model = 2, 4, 2, 64
    c.n_q_heads, c.n_kv_heads, c.head_dim = 4, 2, 16
    c.dense_ffn, c.expert_ffn, c.n_shared = 128, 32, 0
    c.__post_init__()
    return c


def test_inputs_embeds_equiv():
    print("[1] forward(inputs_embeds) == forward(idx) for text-only")
    torch.manual_seed(0)
    m = MoETransformer(tiny_cfg())
    m.eval()  # eval mode: no aux-free bias buffer update between the two forwards (deterministic routing)
    idx = torch.randint(0, 50257, (2, 16))
    tgt = torch.randint(0, 50257, (2, 16))
    with torch.no_grad():
        l1, _ = m(idx, tgt)
        emb = m.embed(idx)
        l2, _ = m(inputs_embeds=emb, targets=tgt)
    d = abs(l1.item() - l2.item())
    print(f"    loss(idx)={l1.item():.6f}  loss(embeds)={l2.item():.6f}  |d|={d:.2e}")
    assert d < 1e-5
    print("    OK")


def test_image_splice():
    print("[2] image splice: length, ignored targets, finite loss, projector grad")
    torch.manual_seed(1)
    VIS_DIM, NI = 1152, 12
    vlm = MoEVLM(MoETransformer(tiny_cfg()), vision_dim=VIS_DIM)
    # sequence: [bos t t <image> t t t]
    ids = torch.tensor([[5, 6, 7, IMAGE_TOKEN, 8, 9, 10],
                        [1, 2, IMAGE_TOKEN, 3, 4, 5, 6]])
    tgt = torch.randint(0, 50257, ids.shape)
    feats = torch.randn(2, NI, VIS_DIM)
    embeds, new_tgt = vlm.build_inputs_embeds(ids, image_features=feats, targets=tgt)
    expect_len = ids.shape[1] - 1 + NI                       # one sentinel -> NI features
    print(f"    merged len={embeds.shape[1]} (expect {expect_len})")
    assert embeds.shape == (2, expect_len, vlm.d_model)
    # the NI image positions must be IGNORE_INDEX
    n_ignored = (new_tgt == IGNORE_INDEX).sum(dim=1)
    print(f"    ignored targets/sample={n_ignored.tolist()} (expect >= {NI})")
    assert (n_ignored >= NI).all()
    loss, parts = vlm(ids, image_features=feats, targets=tgt)
    loss.backward()
    g = vlm.mm_projector.net[0].weight.grad
    print(f"    loss={loss.item():.4f}  projector.grad finite={bool(torch.isfinite(g).all())}")
    assert torch.isfinite(loss) and g is not None and torch.isfinite(g).all()
    print("    OK")


def test_audio_and_both():
    print("[3] audio splice + image&audio together")
    torch.manual_seed(2)
    VIS, AUD, NI, NA = 1152, 768, 8, 5
    vlm = MoEVLM(MoETransformer(tiny_cfg()), vision_dim=VIS, audio_dim=AUD)
    ids = torch.tensor([[5, IMAGE_TOKEN, 6, AUDIO_TOKEN, 7, 8, 9]])
    tgt = torch.randint(0, 50257, ids.shape)
    img = torch.randn(1, NI, VIS)
    aud = torch.randn(1, NA, AUD)
    embeds, new_tgt = vlm.build_inputs_embeds(ids, image_features=img, audio_features=aud, targets=tgt)
    expect = ids.shape[1] - 2 + NI + NA
    print(f"    merged len={embeds.shape[1]} (expect {expect})  ignored={(new_tgt==IGNORE_INDEX).sum().item()}")
    assert embeds.shape[1] == expect
    assert (new_tgt == IGNORE_INDEX).sum().item() >= NI + NA
    loss, _ = vlm(ids, image_features=img, audio_features=aud, targets=tgt)
    assert torch.isfinite(loss)
    print(f"    loss={loss.item():.4f}  OK")


def test_mixed_batch():
    print("[4] mixed batch: image sample + text-only sample (right-pad)")
    torch.manual_seed(3)
    VIS, NI = 1152, 10
    vlm = MoEVLM(MoETransformer(tiny_cfg()), vision_dim=VIS)
    # sample 0 has an image; sample 1 is text-only (no sentinel)
    ids = torch.tensor([[5, IMAGE_TOKEN, 6, 7, 8],
                        [1, 2, 3, 4, 5]])
    tgt = torch.randint(0, 50257, ids.shape)
    feats = torch.randn(2, NI, VIS)   # sample 1's slice is simply never spliced (no sentinel)
    embeds, new_tgt = vlm.build_inputs_embeds(ids, image_features=feats, targets=tgt)
    # sample 0 -> 5-1+NI=14 ; sample 1 -> 5 ; padded to 14
    print(f"    padded len={embeds.shape[1]} (expect 14); text-only row pad targets="
          f"{int((new_tgt[1] == IGNORE_INDEX).sum())}/{embeds.shape[1]}")
    assert embeds.shape[1] == 5 - 1 + NI
    loss, _ = vlm(ids, image_features=feats, targets=tgt)
    assert torch.isfinite(loss)
    print(f"    loss={loss.item():.4f}  OK")


def test_stage1_freeze():
    print("[5] stage-1 freeze: only projector trains")
    vlm = MoEVLM(MoETransformer(tiny_cfg()), vision_dim=1152)
    vlm.set_llm_trainable(False)
    llm_grad = sum(p.requires_grad for p in vlm.llm.parameters())
    proj_grad = sum(p.requires_grad for p in vlm.projector_parameters())
    print(f"    llm params requiring grad={llm_grad} (expect 0); projector={proj_grad} (>0)")
    assert llm_grad == 0 and proj_grad > 0
    print("    OK")


if __name__ == "__main__":
    test_inputs_embeds_equiv()
    test_image_splice()
    test_audio_and_both()
    test_mixed_batch()
    test_stage1_freeze()
    print("\nALL VLM PLUMBING TESTS PASSED")
