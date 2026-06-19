"""Convert our vision stack (SigLIP2-so400m-patch16-512 + the joint12 mm_projector) -> a clip 'mmproj' GGUF
for llama.cpp's mtmd multimodal runtime. The mmproj = frozen SigLIP2 vision tower (V_ENC_* tensors) + our
2-layer MLP projector (V_MMPROJ, mm.0/mm.2). At runtime: image -> SigLIP2 -> 1024x1152 -> projector -> 1024x768
embeddings, spliced into the LLM at IMAGE_TOKEN positions.

    python to_mmproj.py --ckpt /data/runs/500M_vlm_joint12/model.pt --out mmproj-joint12.gguf

NOTE: our projector is a plain 2-layer MLP with NO token pooling (1024 image tokens). clip.cpp's siglip
projector types mostly pool (gemma3) or pixel-shuffle (idefics3); a no-pool MLP path may need a small clip.cpp
addition (projector_type). This converter emits the weights + metadata; the runtime graph is the next step.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

# SigLIP2-so400m-patch16-512 vision config
V = dict(hidden=1152, layers=27, heads=16, ffn=4304, image=512, patch=16, eps=1e-6)
SIGLIP = "google/siglip2-so400m-patch16-512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="joint12 model.pt (uses its 'projector' state-dict)")
    ap.add_argument("--out", default="mmproj-joint12.gguf")
    ap.add_argument("--proj_type", default="phi4")       # PHI4 = siglip vision + no-pool 2-layer GELU MLP
                                                          # (mm.0->GELU->mm.2) = EXACTLY our projector, no patch
    args = ap.parse_args()
    import gguf
    from transformers import AutoModel

    # frozen SigLIP2 vision tower (same weights used in training) + our trained projector
    vis = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float32).vision_model.state_dict()
    proj = torch.load(args.ckpt, map_location="cpu", weights_only=False)["projector"]
    print(f"siglip2 tensors={len(vis)}  projector keys={list(proj.keys())}", flush=True)

    w = gguf.GGUFWriter(args.out, "clip")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", args.proj_type)
    w.add_uint32("clip.vision.embedding_length", V["hidden"])
    w.add_uint32("clip.vision.feed_forward_length", V["ffn"])
    w.add_uint32("clip.vision.block_count", V["layers"])
    w.add_uint32("clip.vision.attention.head_count", V["heads"])
    w.add_float32("clip.vision.attention.layer_norm_epsilon", V["eps"])
    w.add_uint32("clip.vision.image_size", V["image"])
    w.add_uint32("clip.vision.patch_size", V["patch"])
    # PHI4 path: dynamic-resolution sizing by a total-pixel budget (W*H). Pin min==max==512*512 so it
    # always targets our training area (512px SigLIP2 -> ~32x32=1024 patches); resize_position_embeddings
    # interpolates the 32x32 pos grid to the actual patch grid. No tiling (n_merge=1).
    w.add_uint32("clip.vision.image_min_pixels", V["image"] * V["image"])
    w.add_uint32("clip.vision.image_max_pixels", V["image"] * V["image"])
    w.add_uint32("clip.vision.projection_dim", 768)       # our mm_projector output = d_model
    w.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])   # SigLIP normalization
    w.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    w.add_bool("clip.use_gelu", True)                     # SigLIP uses gelu_tanh

    def t(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr.detach().to(torch.float32).numpy()))

    # ---- vision embeddings ----
    t("v.patch_embd.weight", vis["embeddings.patch_embedding.weight"])   # (1152,3,16,16)
    t("v.patch_embd.bias", vis["embeddings.patch_embedding.bias"])
    t("v.position_embd.weight", vis["embeddings.position_embedding.weight"])  # (1024,1152)
    t("v.post_ln.weight", vis["post_layernorm.weight"])
    t("v.post_ln.bias", vis["post_layernorm.bias"])

    # ---- vision transformer blocks ----
    for i in range(V["layers"]):
        b = f"encoder.layers.{i}."
        for hf, gg in [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
                       ("self_attn.v_proj", "attn_v"), ("self_attn.out_proj", "attn_out"),
                       ("layer_norm1", "ln1"), ("layer_norm2", "ln2"),
                       ("mlp.fc1", "ffn_up"), ("mlp.fc2", "ffn_down")]:
            t(f"v.blk.{i}.{gg}.weight", vis[b + hf + ".weight"])
            t(f"v.blk.{i}.{gg}.bias", vis[b + hf + ".bias"])

    # ---- our 2-layer MLP projector (Projector.net = [Linear, GELU, Linear]) -> mm.0 / mm.2 ----
    pk = {k.split("net.")[-1] if "net." in k else k: v for k, v in proj.items()}
    t("mm.0.weight", pk["0.weight"]); t("mm.0.bias", pk["0.bias"])
    t("mm.2.weight", pk["2.weight"]); t("mm.2.bias", pk["2.bias"])

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out}  (clip mmproj: SigLIP2-so400m + 2-layer MLP, proj_type={args.proj_type})", flush=True)


if __name__ == "__main__":
    main()
