"""Convert our speech stack (frozen openai/whisper-small ENCODER + the joint12 speech_projector) -> a clip
'audio mmproj' GGUF for llama.cpp's mtmd runtime. Our speech path = Whisper encoder + stack-2 + 2-layer GELU
MLP projector + <speech> splice = textbook **Ultravox/Voxtral**. mtmd's PROJECTOR_TYPE_VOXTRAL graph is exactly
`build_ffn(mm_1 -> GELU_ERF -> mm_2)` (no norm, no gating) = our speech_projector, and clip.cpp does the whisper
mel + encode + frame-stacking itself (proj_stack_factor). So like the LLM (hobbylm) and vision (phi4), our
speech path is a structural twin of an existing type — emit the right tensors + metadata, no C++ patch.

    python to_aproj.py --ckpt /data/runs/500M_vlm_joint12/model.pt --out aproj-joint12.gguf

At runtime: wav -> (clip.cpp) 80-mel/30s -> whisper encoder -> 1500x768 -> stack 2 -> 750x1536 -> projector
-> 750x768 embeddings, spliced into the LLM where mtmd places the audio marker (= our [SPEECH_TOKEN] position).
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

WHISPER = "openai/whisper-small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="joint12 model.pt (uses its 'speech_projector' state-dict)")
    ap.add_argument("--out", default="aproj-joint12.gguf")
    ap.add_argument("--stack", type=int, default=2)        # our WhisperSpeech stack (1500 -> 1500/stack tokens)
    args = ap.parse_args()
    import gguf
    from transformers import WhisperModel, WhisperConfig

    cfg: WhisperConfig = WhisperConfig.from_pretrained(WHISPER)
    enc = WhisperModel.from_pretrained(WHISPER, torch_dtype=torch.float32).encoder.state_dict()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    proj = ck["speech_projector"]
    d = cfg.d_model; nl = cfg.encoder_layers; nh = cfg.encoder_attention_heads
    ff = cfg.encoder_ffn_dim; mels = cfg.num_mel_bins
    print(f"whisper d={d} layers={nl} heads={nh} ffn={ff} mels={mels} stack={args.stack}  "
          f"speech_proj keys={list(proj.keys())}", flush=True)

    w = gguf.GGUFWriter(args.out, "clip")
    w.add_bool("clip.has_vision_encoder", False)
    w.add_bool("clip.has_audio_encoder", True)
    w.add_string("clip.projector_type", "voxtral")          # build_ffn(mm_1->GELU_ERF->mm_2) = our projector
    w.add_uint32("clip.audio.embedding_length", d)
    w.add_uint32("clip.audio.block_count", nl)
    w.add_uint32("clip.audio.attention.head_count", nh)
    w.add_uint32("clip.audio.feed_forward_length", ff)
    w.add_uint32("clip.audio.num_mel_bins", mels)
    w.add_uint32("clip.audio.projector.stack_factor", args.stack)   # clip.cpp stacks 768 -> 768*stack
    w.add_float32("clip.audio.attention.layer_norm_epsilon", 1e-5)  # whisper eps
    w.add_uint32("clip.audio.projection_dim", 768)          # projector out = d_model

    def t(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr.detach().to(torch.float32).numpy()))

    def t16(name, arr):     # ggml CPU conv_1d (im2col) ASSERTS the kernel is F16 (src0->type == F16)
        w.add_tensor(name, np.ascontiguousarray(arr.detach().to(torch.float16).numpy()))

    # ---- whisper conv front-end (mel -> 1500x768) + positions ----
    # conv weight (OC,IC,K) torch -> ggml ne=[K,IC,OC] (correct for ggml_conv_1d). conv BIAS must be (OC,1)
    # -> ggml ne=[1,OC] so it broadcasts over the frame dim (after conv, ne0=frames not channels); a flat
    # (OC,) bias fails ggml_can_repeat([OC],[frames,OC]).
    t16("a.conv1d.1.weight", enc["conv1.weight"]); t("a.conv1d.1.bias", enc["conv1.bias"].reshape(-1, 1))
    t16("a.conv1d.2.weight", enc["conv2.weight"]); t("a.conv1d.2.bias", enc["conv2.bias"].reshape(-1, 1))
    t("a.position_embd.weight", enc["embed_positions.weight"])                              # (1500,768)
    t("a.post_ln.weight", enc["layer_norm.weight"]); t("a.post_ln.bias", enc["layer_norm.bias"])

    # ---- whisper encoder blocks (audio path = SEPARATE attn_q/k/v; whisper k_proj has NO bias) ----
    for i in range(nl):
        b = f"layers.{i}."
        t(f"a.blk.{i}.attn_q.weight", enc[b + "self_attn.q_proj.weight"])
        t(f"a.blk.{i}.attn_q.bias", enc[b + "self_attn.q_proj.bias"])
        t(f"a.blk.{i}.attn_k.weight", enc[b + "self_attn.k_proj.weight"])                    # k: no bias
        t(f"a.blk.{i}.attn_v.weight", enc[b + "self_attn.v_proj.weight"])
        t(f"a.blk.{i}.attn_v.bias", enc[b + "self_attn.v_proj.bias"])
        t(f"a.blk.{i}.attn_out.weight", enc[b + "self_attn.out_proj.weight"])
        t(f"a.blk.{i}.attn_out.bias", enc[b + "self_attn.out_proj.bias"])
        t(f"a.blk.{i}.ln1.weight", enc[b + "self_attn_layer_norm.weight"])
        t(f"a.blk.{i}.ln1.bias", enc[b + "self_attn_layer_norm.bias"])
        t(f"a.blk.{i}.ln2.weight", enc[b + "final_layer_norm.weight"])
        t(f"a.blk.{i}.ln2.bias", enc[b + "final_layer_norm.bias"])
        t(f"a.blk.{i}.ffn_up.weight", enc[b + "fc1.weight"]); t(f"a.blk.{i}.ffn_up.bias", enc[b + "fc1.bias"])
        t(f"a.blk.{i}.ffn_down.weight", enc[b + "fc2.weight"]); t(f"a.blk.{i}.ffn_down.bias", enc[b + "fc2.bias"])

    # ---- our 2-layer MLP speech_projector (net = [Linear(1536,768), GELU, Linear(768,768)]) -> mm.a.mlp.1/2 ----
    pk = {k.split("net.")[-1] if "net." in k else k: v for k, v in proj.items()}
    t("mm.a.mlp.1.weight", pk["0.weight"]); t("mm.a.mlp.1.bias", pk["0.bias"])
    t("mm.a.mlp.2.weight", pk["2.weight"]); t("mm.a.mlp.2.bias", pk["2.bias"])

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out}  (clip AUDIO mmproj: whisper-small enc + 2-layer MLP, proj_type=voxtral)", flush=True)


if __name__ == "__main__":
    main()
