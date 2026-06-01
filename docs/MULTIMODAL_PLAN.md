# Multimodal MoE (Image + Audio) — Plan

Turn the MoE base LLM into a **TinyLLaVA-style** vision-language model (then add audio), using our
MoE as the LLM. Decisions locked 2026-05-31.

## Locked decisions
| piece | choice |
|---|---|
| LLM (target) | **MoE 500M** (`500M_40B`, val 3.03); 1B later |
| Vision encoder | `google/siglip2-so400m-patch14-384` (~400M), **frozen**, full ~729 tokens |
| Context | **extend 500M 1024 → 2048 first** (RoPE), so 729 img + text fits |
| Connector | 2-layer MLP projector (GELU), TinyLLaVA default — trainable |
| Stage 1 (align) | LAION-CC-SBU-558K, **projector only** |
| Stage 2 (SFT) | TinyLLaVA_Factory full mix, projector + LLM |
| Audio (Phase 3) | **CLAP/BEATs** encoder (general audio), frozen + projector |
| Special tokens | reuse free vocab slots 50257–50303 (`<image>`,`<audio>`,`<im_start/end>`) — no tokenizer change |

## Key engineering facts / gotchas
- **Context is the gating constraint.** SigLIP2-so400m-384 ≈ 729 tokens > half of 1024 → must extend to 2048.
- **Precompute & cache frozen encoder features** to a Modal volume (vision frozen in BOTH stages) → training
  never runs the 400M ViT; stage 1 becomes ~free. Same for audio. (Big efficiency win; matches our ethos.)
- **MoE bonus:** router sees visual tokens too — log expert usage on image vs text tokens; on-ramp to
  *modality experts* (Phase 4) for ~free.
- **Audio data gap:** LAION/TinyLLaVA are image-only; audio needs its own data (AudioCaps/Clotho) + encoder
  + a 3rd training stage. Phase it after vision.

## Code changes to `MoETransformer` / harness
1. `forward(input_ids=None, inputs_embeds=None, ...)` — accept precomputed embeds for splicing.
2. **`MoEVLM` wrapper** (new file): text-embed → encoders+projectors (or load cached feats) → replace
   `<image>`/`<audio>` placeholder embeddings → call LLM.
3. **Checkpoint-resume in `train.py`** (needed first, for context extension): load weights + continue.
4. RoPE: 1D positions for image/audio tokens to start; 2D/M-RoPE is a later upgrade.

## Roadmap
- **Phase 0 — context extension — ✅ DONE (2026-05-31).** Added `--init_from`/`--lr_mult` to train.py;
  continued-pretrained `500M_40B` at `seq_len=2048`, fused_ce, lr_mult 0.5, 2000 steps (~2.1B tok, ~1h, 8×H100),
  θ kept at 1e4 (train-at-length, not extrapolation). **Result: `500M_ctx2048` val 2.9990 @ 2048 — BEATS the
  original 3.0281 @ 1024.** Checkpoint `/data/runs/500M_ctx2048/model.pt`. This is the VLM backbone. (Gotcha:
  Git-Bash mangled the `--init-from /data/...` POSIX path → use `MSYS_NO_PATHCONV=1`.)
- **Phase 0.5 — VLM plumbing — ✅ DONE (2026-05-31).** `model.forward(inputs_embeds=)` (bit-identical to
  idx path); `multimodal.py`: `MoEVLM` + `Projector` (mlp2x_gelu) + sentinels IMAGE_TOKEN=50257/AUDIO_TOKEN=50258;
  `build_inputs_embeds` splices projected feats at sentinels, right-pads (causal-safe) for mixed/text-only;
  `set_llm_trainable` for stage-1 freeze. `test_vlm.py` = 5 CPU tests pass. Vision encoder dim assumed 1152
  (SigLIP2 so400m); projector → d_model 768.
- **Phase 1 — vision — stage-1 ✅ DONE (2026-05-31).** vision.py (SigLIP2-so400m, 729×1152, frozen),
  multimodal.py (MoEVLM; fixed 2 bugs: autocast cat dtype + next-token last-feature target), modal_mm.py
  (moe-vlm app). **Images STREAMED from images.zip** (random-access read by name; unzip OOM'd container disk,
  so no extraction). DDP `vlm_stage1.py` on 8×H100. **Stage-1 projector: loss 6.66→3.02, 1500 steps (~0.7 epoch),
  ~20 min.** `500M_vlm_stage1/projector.pt`. **Captioning WORKS** (8/8 image-grounded, e.g. read the brand:
  "a white Hyundai car parked in front of a house"; "a boat in the harbor t shirt"). Backbone=`500M_ctx2048`,
  projector=1.48M params. NEXT: stage-2 SFT on TinyLLaVA mix (projector+LLM). Eval: VQAv2 / GQA / TextVQA / POPE.
- **Phase 2 — (folded into 0)** full 729 tokens at 2048 ctx.
- **Phase 3 — audio:** CLAP/BEATs encoder + AudioCaps/Clotho data + projector; joint image+audio SFT.
- **Phase 4 (optional, novel) — MoE modality experts** ablation.

## Compute (Modal, 8×H100)
- Context extension: a few hours (~1–3B tokens at 2048).
- Stage 1 (cached feats, projector only): minutes–~1h.
- Stage 2 SFT (~1.2M samples, LLM unfrozen): a few hours at 500M.
- Biggest effort = **data engineering** (TinyLLaVA mix = COCO/GQA/OCR-VQA/TextVQA/VG, hundreds of GB).

## Eval
VQAv2, GQA, TextVQA, **POPE** (hallucination), MME, COCO CIDEr — via lm-eval multimodal tasks.
