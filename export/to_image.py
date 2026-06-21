#!/usr/bin/env python3
"""Export the HobbyLM text-to-image pipeline (CLIP text encoder + HobbyImageDiT + DC-AE f32c32
decoder) to flat F32 safetensors that the from-scratch Rust engine (hobby-rs) reads, plus a
reference-activation bundle for token-for-token verification of the Rust port.

Outputs (default --out image_weights/):
  dit.safetensors          HobbyImageDiT weights (from model_1024flux4.pt), keys = state_dict keys
  dit_meta.json            cfg_dict + lat_std + sf + steps
  clip_text.safetensors    CLIP ViT-L/14 text-encoder weights (text_model.* keys)
  clip_meta.json           dims (layers/width/heads/eps) + special token ids + max_length
  clip_vocab.json          token -> id  (49408)
  clip_merges.txt          BPE merges (CLIP order)
  dcae_decoder.safetensors DC-AE decoder weights (decoder.* keys)
  dcae_meta.json           decoder structural config (channels / layers / qkv / scaling_factor)
  ref.safetensors          fixed-seed reference tensors (z0, ctx, uncond, vc0, vu0, z_final, pixels)
  ref_meta.json            prompt / neg / steps / cfg used for the reference

Run locally (needs torch + transformers + diffusers==0.32.x):
  python export/to_image.py --ckpt checkpoints/model_1024flux4.pt --out image_weights
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for hobby_image
from hobby_image.dit import HobbyImageDiT, DiTConfig

CLIP_ID = "openai/clip-vit-large-patch14"
DCAE_ID = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
NEG_DEFAULT = "blurry, low quality, watermark, signature, text, jpeg artifacts, deformed, distorted"
REF_PROMPT = "a red cube on a wooden table, studio lighting"


def save_safetensors(path, tensors: dict):
    from safetensors.torch import save_file
    save_file({k: v.contiguous().cpu().float() for k, v in tensors.items()}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model_1024flux4.pt")
    ap.add_argument("--out", default="image_weights")
    ap.add_argument("--ref_steps", type=int, default=8)
    ap.add_argument("--ref_cfg", type=float, default=5.0)
    ap.add_argument("--skip_ref", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cpu"
    torch.manual_seed(0)

    # ---- 1. DiT -----------------------------------------------------------------
    print(f"[dit] loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = DiTConfig(**ck["cfg_dict"])
    dit = HobbyImageDiT(cfg).to(dev).eval()
    dit.load_state_dict(ck["sd"])
    save_safetensors(os.path.join(args.out, "dit.safetensors"), dict(dit.state_dict()))
    json.dump({"cfg_dict": ck["cfg_dict"], "lat_std": float(ck["lat_std"]),
               "sf": float(ck["sf"]), "steps": int(ck.get("steps", 0))},
              open(os.path.join(args.out, "dit_meta.json"), "w"), indent=1)
    lat = cfg.latent_h               # 32 -> 1024px
    print(f"[dit] {sum(p.numel() for p in dit.parameters())/1e6:.1f}M | lat={lat} ctx_dim={cfg.ctx_dim}", flush=True)

    # ---- 2. CLIP text encoder ---------------------------------------------------
    print(f"[clip] loading {CLIP_ID}", flush=True)
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    clip = CLIPTextModel.from_pretrained(CLIP_ID, torch_dtype=torch.float32).to(dev).eval()
    tm = clip.text_model
    cc = clip.config
    save_safetensors(os.path.join(args.out, "clip_text.safetensors"),
                     {k: v for k, v in clip.state_dict().items() if k.startswith("text_model.")})
    json.dump({"vocab_size": cc.vocab_size, "hidden": cc.hidden_size, "layers": cc.num_hidden_layers,
               "heads": cc.num_attention_heads, "intermediate": cc.intermediate_size,
               "eps": cc.layer_norm_eps, "max_position": cc.max_position_embeddings,
               "act": cc.hidden_act, "bos": tok.bos_token_id, "eos": tok.eos_token_id,
               "pad": tok.pad_token_id, "max_length": 64},
              open(os.path.join(args.out, "clip_meta.json"), "w"), indent=1)
    # tokenizer vocab + merges
    vocab = tok.get_vocab()
    json.dump(vocab, open(os.path.join(args.out, "clip_vocab.json"), "w"))
    merges_src = tok.save_vocabulary(args.out)  # writes vocab.json + merges.txt
    # normalize merges file name
    for f in os.listdir(args.out):
        if f.endswith("merges.txt") and f != "clip_merges.txt":
            os.replace(os.path.join(args.out, f), os.path.join(args.out, "clip_merges.txt"))
    print(f"[clip] {cc.num_hidden_layers}L x {cc.hidden_size} | vocab {cc.vocab_size}", flush=True)

    def encode(texts):
        ids = tok(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            return clip(ids).last_hidden_state.float(), ids

    # ---- 3. DC-AE decoder -------------------------------------------------------
    print(f"[dcae] loading {DCAE_ID}", flush=True)
    from diffusers import AutoencoderDC
    ae = AutoencoderDC.from_pretrained(DCAE_ID, torch_dtype=torch.float32).to(dev).eval()
    aecfg = dict(ae.config)
    save_safetensors(os.path.join(args.out, "dcae_decoder.safetensors"),
                     {k: v for k, v in ae.state_dict().items() if k.startswith("decoder.")})
    json.dump({k: aecfg[k] for k in ["latent_channels", "attention_head_dim", "decoder_block_types",
                                     "decoder_block_out_channels", "decoder_layers_per_block",
                                     "decoder_qkv_multiscales", "upsample_block_type",
                                     "decoder_norm_types", "decoder_act_fns", "scaling_factor"]},
              open(os.path.join(args.out, "dcae_meta.json"), "w"), indent=1)
    print(f"[dcae] decoder {sum(p.numel() for p in ae.decoder.parameters())/1e6:.1f}M", flush=True)

    if args.skip_ref:
        print("done (weights only).", flush=True)
        return

    # ---- 4. reference activations (fixed seed) ----------------------------------
    print(f"[ref] prompt={REF_PROMPT!r} steps={args.ref_steps} cfg={args.ref_cfg}", flush=True)
    lat_std = float(ck["lat_std"]); sf = float(ck["sf"])
    ctx, ids_c = encode([REF_PROMPT])
    uncond, ids_u = encode([NEG_DEFAULT])
    task = torch.zeros(1, dtype=torch.long, device=dev)
    g = torch.Generator(device=dev).manual_seed(1234)
    z = torch.randn(1, 32, lat, lat, generator=g, device=dev)
    z0 = z.clone()
    zs = torch.zeros(1, 32, lat, lat, device=dev)
    em = torch.zeros(1, 1, lat, 2 * lat, device=dev)
    vc0 = vu0 = None
    with torch.no_grad():
        for i in range(args.ref_steps):
            tt = torch.full((1,), i / args.ref_steps, device=dev)
            inp = torch.cat([torch.cat([z, zs], dim=-1), em, em], dim=1)
            vc = dit(inp, tt, ctx, task)[..., :lat]
            vu = dit(inp, tt, uncond, task)[..., :lat]
            if i == 0:
                vc0, vu0 = vc.clone(), vu.clone()
            z = z + (vu + args.ref_cfg * (vc - vu)) / args.ref_steps
        z_final = z.clone()
        dec_in = (z * lat_std / sf)
        pixels = ae.decode(dec_in).sample.float().clamp(-1, 1)
    save_safetensors(os.path.join(args.out, "ref.safetensors"),
                     {"ids_cond": ids_c.float(), "ids_uncond": ids_u.float(),
                      "ctx": ctx, "uncond": uncond, "z0": z0, "vc0": vc0, "vu0": vu0,
                      "z_final": z_final, "dec_in": dec_in, "pixels": pixels})
    json.dump({"prompt": REF_PROMPT, "neg": NEG_DEFAULT, "steps": args.ref_steps,
               "cfg": args.ref_cfg, "seed": 1234, "lat": lat, "lat_std": lat_std, "sf": sf},
              open(os.path.join(args.out, "ref_meta.json"), "w"), indent=1)
    px = ((pixels[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
    try:
        from PIL import Image
        Image.fromarray(px).save(os.path.join(args.out, "ref_pixels.png"))
    except Exception:
        np.save(os.path.join(args.out, "ref_pixels.npy"), px)
    print(f"[ref] saved. pixels {tuple(pixels.shape)} range [{pixels.min():.3f},{pixels.max():.3f}]", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
