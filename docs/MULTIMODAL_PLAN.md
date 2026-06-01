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
  projector=1.48M params.
- **Phase 1 — stage-2 SFT ✅ DONE (2026-05-31).** Chose the **single most effective subset**: LLaVA-Instruct-150K
  (GPT-4 visual instructions) + **COCO train2017** (~19GB, streamed from train2017.zip; hit-rate 300/300, train2017
  is the right split). `vlm_stage2.py` DDP: unfreeze LLM (502M trainable), init projector from stage-1, AdamW
  (LLM 2e-5 / proj 1e-4) warmup+cosine, loss masked to assistant turns. **8×H100, micro 8, 1500 steps (~0.6 epoch,
  ~13 min): loss 2.64→1.9.** `500M_vlm_stage2/model.pt`. Test: `--action caption --stage2_run 500M_vlm_stage2`
  (USER:/ASSISTANT: chat format). Eval: VQAv2 / GQA / TextVQA / POPE (TODO).
- **Phase 1 — quality pass (2026-06-01).** Diagnosed the stage-2 "repetition/hallucination": **repetition was a
  pure DECODING artifact** → fixed FREE with rep-penalty(1.3)+no-repeat-3gram in `caption` (no retrain; "white white
  car" loops → clean 7/8). Residual **hallucination** (hard images, e.g. banknote→"building") is data/scale-limited →
  **full-epoch retrain** (2500 steps = 1 epoch, the standard SFT length; >1 risks overfit): loss→1.58 (vs 0.6-ep 1.9).
  Real levers for more: full 665K mix (~50GB) or 1B backbone.
- **Phase 2 — (folded into 0)** full 729 tokens at 2048 ctx.
- **Phase 3 — audio — stage-1 ✅ DONE (2026-06-01).** `audio.py` (frozen CLAP `laion/clap-htsat-unfused`
  → 64 tokens × 768; flatten HTSAT 2D map). `vlm_audio_data.ClothoAudio` reads the HF parquet via **pyarrow +
  soundfile directly** (the `datasets` Audio loader fought back: torchcodec dep, then a pyarrow/dataclass crash —
  bypassed entirely). Single-H100 `train_audio_stage1`: audio projector only (CLAP+LLM frozen), Clotho 3839 clips,
  1200 steps (~17 min): **loss 4.46→2.77.** `500M_vlm_audio_stage1/audio_projector.pt` (1.18M params). **Audio
  captioning WORKS — 7-8/8 correct sounds**, *discriminates* birds vs crickets vs cars vs pouring-water vs voices
  (`--action caption_audio`). Repetition = same decoding fix (rep-penalty + no-repeat-3gram). NEXT: joint
  image+audio SFT for one unified model; WavCaps for more audio data.
- **UNIFIED image+video+audio ✅ DONE (2026-06-01).** `video.py` SiglipVideo = sample frames → SigLIP2 →
  concat RAW 729 tok/frame (NOT pooled: pooled tokens are OOD for mm_projector → garbage; learned this the hard
  way). Video reuses `mm_projector` (VIDEO_TOKEN). `--action unified`: ONE MoEVLM = `500M_vlm_stage2` LLM +
  mm_projector (image & video) + `500M_vlm_audio_stage1` audio_projector. **All 3 modalities work 4/4** — video
  matches its source image (zero-shot, no video training); audio_projector trained on base LLM transfers to the
  stage-2 LLM fine (hears engines/birds/gear-shifting). Context limit: 2 frames × 729 = 1458 < 2048. Next for video:
  a video-SFT (or train mm_projector to accept pooled tokens for more frames).
- **Phase 4 (optional, novel) — MoE modality experts** ablation.

## Compute (Modal, 8×H100)
- Context extension: a few hours (~1–3B tokens at 2048).
- Stage 1 (cached feats, projector only): minutes–~1h.
- Stage 2 SFT (~1.2M samples, LLM unfrozen): a few hours at 500M.
- Biggest effort = **data engineering** (TinyLLaVA mix = COCO/GQA/OCR-VQA/TextVQA/VG, hundreds of GB).

## Eval
VQAv2, GQA, TextVQA, **POPE** (hallucination), MME, COCO CIDEr — via lm-eval multimodal tasks.
