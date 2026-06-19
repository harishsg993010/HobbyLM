"""Convert the joint12 MoE LLM core -> GGUF emitting the custom **`hobbylm`** architecture.

Our arch (GQA + per-head QK-norm + sigmoid-gated MoE + aux-free routing bias + 1 shared expert + first-layer
dense + full attention) is exported under `general.architecture = hobbylm`, which the `hobby-rs` engine (or a
patched llama.cpp registered for `hobbylm`) loads directly; ggml's `ggml_mul_mat_id` runs only the top-6 experts
(native sparse MoE, the thing the ONNX/LiteRT graph-exporters couldn't do). GPT-2 BPE tokenizer is embedded.
(The `--arch` flag can still emit a stock-llama.cpp-compatible build for the legacy export path.)

    python to_gguf.py --ckpt /data/runs/500M_vlm_joint12/model.pt --out joint12-hobbylm.gguf
"""
from __future__ import annotations

import argparse
import numpy as np
import torch


def _gpt2_tokenizer(vocab_size):
    """GPT-2 byte-level BPE: token strings, merges, types. Pads to vocab_size with user-defined sentinels
    (our 50257-50303 image/audio/speech tokens)."""
    import json
    from huggingface_hub import hf_hub_download
    vj = json.load(open(hf_hub_download("openai-community/gpt2", "vocab.json")))   # token -> id
    merges = [l.strip() for l in open(hf_hub_download("openai-community/gpt2", "merges.txt"),
                                      encoding="utf-8").read().splitlines()[1:] if l.strip()]
    id2tok = {i: t for t, i in vj.items()}
    toks, types = [], []
    for i in range(vocab_size):
        if i in id2tok:
            toks.append(id2tok[i]); types.append(1)              # NORMAL
        elif i == 50256:
            toks.append("<|endoftext|>"); types.append(3)        # CONTROL
        else:
            toks.append(f"<|extra_{i}|>"); types.append(4)       # USER_DEFINED (sentinels + padding)
    return toks, types, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="joint12-hobbylm.gguf")
    ap.add_argument("--arch", default="hobbylm",
                    help="GGUF general.architecture (also prefixes the arch KV keys). 'bailingmoe2' = stock "
                         "llama.cpp; 'hobbylm' = HobbyLM brand (needs hobby-rs / patched llama.cpp).")
    ap.add_argument("--block-size", type=int, default=32, help="diffusion decode block size (diffusion ckpts only)")
    args = ap.parse_args()
    import gguf

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]; c = ck["config"]
    d = c["d_model"]; L = c["n_layers"]; n_dense = c["n_dense_layers"]
    nq = c["n_q_heads"]; nkv = c["n_kv_heads"]; hd = c["head_dim"]
    n_exp = c["n_experts"]; top_k = c["top_k"]; n_shared = c["n_shared"]
    exp_ffn = c["expert_ffn"]; dense_ffn = c["dense_ffn"]; vocab = c["vocab_size"]
    theta = float(c.get("rope_theta", 1e6)); eps = float(c.get("rms_eps", 1e-5))
    print(f"d={d} L={L} dense={n_dense} q/kv={nq}/{nkv} hd={hd} E={n_exp}/top{top_k} "
          f"shared={n_shared} expff={exp_ffn} theta={theta}", flush=True)

    w = gguf.GGUFWriter(args.out, args.arch)
    w.add_name(f"HobbyLM ({args.ckpt.split('/')[-2] if '/' in args.ckpt else 'model'})")
    w.add_context_length(8192); w.add_embedding_length(d); w.add_block_count(L)
    w.add_feed_forward_length(dense_ffn)
    w.add_head_count(nq); w.add_head_count_kv(nkv)
    w.add_key_length(hd); w.add_value_length(hd)
    w.add_layer_norm_rms_eps(eps); w.add_rope_freq_base(theta)
    w.add_expert_count(n_exp); w.add_expert_used_count(top_k)
    w.add_expert_feed_forward_length(exp_ffn)
    w.add_expert_shared_count(n_shared); w.add_expert_shared_feed_forward_length(exp_ffn)
    w.add_leading_dense_block_count(n_dense)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_expert_weights_norm(bool(c.get("norm_topk_prob", False)))
    w.add_expert_weights_scale(1.0)
    w.add_vocab_size(vocab)
    try:
        w.add_uint32(f"{args.arch}.nextn_predict_layers", 0)      # no MTP
    except Exception:
        pass
    # ---- diffusion (LLaDA) metadata: read by hobby-rs to switch to bidirectional + denoise decode.
    # llama.cpp ignores these unknown keys (it would run the model autoregressively, incorrectly).
    if c.get("diffusion"):
        w.add_bool("diffusion.enabled", True)
        w.add_uint32("diffusion.mask_token_id", int(c.get("mask_token_id", 50257)))
        w.add_uint32("diffusion.block_size", int(args.block_size))
        print(f"  + diffusion metadata: mask_token_id={c.get('mask_token_id', 50257)} "
              f"block_size={args.block_size}", flush=True)

    # ---- tokenizer (GPT-2 BPE) ----
    toks, types, merges = _gpt2_tokenizer(vocab)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("gpt-2")
    w.add_token_list(toks); w.add_token_types(types); w.add_token_merges(merges)
    w.add_bos_token_id(50256); w.add_eos_token_id(50256)
    w.add_add_bos_token(False)
    # pass-through chat template: emit message content verbatim, no role tokens. Our VLM format
    # ([IMAGE]USER: q\nASSISTANT:) is supplied by the caller in -p; mtmd auto-prepends the image marker
    # to the FRONT (= our [IMAGE_TOKEN] position). This also satisfies mtmd-cli's "needs a chat template".
    w.add_chat_template("{% for message in messages %}{{ message['content'] }}{% endfor %}")

    def t(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr.detach().to(torch.float32).numpy()))

    t("token_embd.weight", sd["embed.weight"])
    t("output_norm.weight", sd["final_norm.weight"])
    t("output.weight", sd.get("lm_head.weight", sd["embed.weight"]))
    q_dim, kv_dim = nq * hd, nkv * hd
    for i in range(L):
        p = f"blocks.{i}."
        t(f"blk.{i}.attn_norm.weight", sd[p + "attn_norm.weight"])
        t(f"blk.{i}.attn_qkv.weight", sd[p + "attn.qkv.weight"])      # hobbylm uses FUSED qkv (our layout)
        t(f"blk.{i}.attn_q_norm.weight", sd[p + "attn.q_norm.weight"])
        t(f"blk.{i}.attn_k_norm.weight", sd[p + "attn.k_norm.weight"])
        t(f"blk.{i}.attn_output.weight", sd[p + "attn.proj.weight"])
        t(f"blk.{i}.ffn_norm.weight", sd[p + "ffn_norm.weight"])
        if i < n_dense:
            w13 = sd[p + "ffn.w13.weight"]
            t(f"blk.{i}.ffn_gate.weight", w13[:dense_ffn])
            t(f"blk.{i}.ffn_up.weight", w13[dense_ffn:])
            t(f"blk.{i}.ffn_down.weight", sd[p + "ffn.w2.weight"])
        else:
            t(f"blk.{i}.ffn_gate_inp.weight", sd[p + "ffn.gate.weight"])
            t(f"blk.{i}.exp_probs_b.bias", sd[p + "ffn.expert_bias"])
            ew13 = sd[p + "ffn.experts.w13"]; ew2 = sd[p + "ffn.experts.w2"]
            g, u = ew13[..., :exp_ffn], ew13[..., exp_ffn:]
            t(f"blk.{i}.ffn_gate_exps.weight", g.transpose(1, 2).contiguous())
            t(f"blk.{i}.ffn_up_exps.weight", u.transpose(1, 2).contiguous())
            t(f"blk.{i}.ffn_down_exps.weight", ew2.transpose(1, 2).contiguous())
            sw13 = sd[p + "ffn.shared.w13"]; sw2 = sd[p + "ffn.shared.w2"]
            t(f"blk.{i}.ffn_gate_shexp.weight", sw13[0, :, :exp_ffn].t().contiguous())
            t(f"blk.{i}.ffn_up_shexp.weight", sw13[0, :, exp_ffn:].t().contiguous())
            t(f"blk.{i}.ffn_down_shexp.weight", sw2[0].t().contiguous())

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    runtime = "stock llama.cpp" if args.arch == "bailingmoe2" else "hobby-rs / patched llama.cpp"
    print(f"wrote {args.out}  (arch={args.arch}, runs in: {runtime})", flush=True)


if __name__ == "__main__":
    main()
