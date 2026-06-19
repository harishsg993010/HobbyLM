# MoE-DreamLite-HQ: DreamLite-Style Architecture, VAE, and Training Data Plan

**Goal:** build a compute-efficient, DreamLite-style image generation and image editing model using a **frozen custom 500M MoE VLM** as the semantic backbone, a **high-fidelity VAE** as the image/appearance bottleneck, and a **compact in-context latent U-Net** as the trainable renderer/editor.

**Target constraint:** approximately **$500 compute budget**, assuming access to an 8-GPU GB200-class machine is available and charged only for actual training time/usage.

**Recommended first serious model:** `MoE-DreamLite-HQ-v1`

```text
Frozen 500M MoE VLM
        +
VLM token refiner (coarse semantic conditioning; planner dropped — Section 0)
        +
HQ f8c16 KL-VAE
        +
DreamLite-style in-context latent U-Net
        +
foreground-weighted edit loss
        +
T2I -> Edit -> Unified -> SFT -> Preference -> Distillation curriculum
```

---

## 0. Revision Note (2026-06-13): VLM Grounding Test — Scope Change

Before committing to the build, we **empirically tested** whether the frozen 500M MoE VLM can ground objects/regions — the prerequisite for the original Edit Planner / Mask Head (Section 6) and the VLM Ranker.

**Test:** prompted `500M_vlm_joint12` with explicit grounding/localization questions over real images, plus the existing `pope/gqa/vqav2` eval.

**Result — the VLM CANNOT reliably ground:**

```text
- Object ID: weak + prior-driven (2/5 correct, 2/5 confidently WRONG:
  banknote -> "picture frame", pinball scene -> "table tennis ball").
- Spatial localization: essentially absent (4/5 ignored the location
  instruction; the lone "center" answer is the trivial default).
- Benchmarks corroborate: GQA 0.206 (<= blind baseline),
  VQAv2 0.282 (~ language-prior baseline -> barely using the image),
  POPE ~chance (degenerate harness, but no signal of presence discrimination).
```

**Decisions (this revision applies them throughout):**

1. **DROP the VLM-driven Edit Planner / Mask Head (Section 6).** It cannot localize target/preserve regions from this backbone. Section 6 is replaced by a **diff/user mask strategy** that needs no VLM grounding.
2. **The VLM is a COARSE SEMANTIC conditioner only** — object identity, attributes, edit intent via cross-attention. Not a planner, grounder, or region reasoner. (Section 4 recalibrated.)
3. **Masks come from source↔target pixel diffs (training) and the user or soft foreground emphasis (inference)** — already the documented fallback; now the PRIMARY path. The foreground-weighted edit loss (Section 14) is unchanged and becomes the main mechanism for localized edits.
4. **VLM Ranker is downscoped to OPTIONAL/EXTERNAL** — judging preservation/identity needs the same image understanding the VLM lacks; use an external scorer or human eval for V1.
5. **Expectation reset:** instruction-following on complex/compositional edits is **capped by the VLM's weak image comprehension**. V1 realistic scope = global + simple-local edits driven by source-latent + instruction text + coarse VLM tokens.

**Fork if precise region control is required later:** either (a) **continue-train the VLM on referring/grounding data** (GRIT / RefCOCO-style) until POPE/GQA clear baseline, *then* re-introduce the planner; or (b) **swap in a stronger off-the-shelf VLM** as the conditioner (loses the "reuse my model" appeal, unblocks grounding). Do **not** build the planner on the current frozen 500M.

The rest of the document is retained but should be read through this lens: **VAE + in-context U-Net + refiner + diff-mask foreground loss is the viable V1; the planner/ranker "smart" modules are deferred.**

---

## 1. Core DreamLite Ideas to Reuse

DreamLite-style training is attractive because it uses one compact diffusion model for both text-to-image generation and image editing.

The most important ideas to copy are:

1. **Unified in-context latent layout**
   - Text-to-image generation:

     ```text
     [ noisy target latent | blank latent ]
     ```

   - Image editing:

     ```text
     [ noisy target latent | source image latent ]
     ```

   The model sees a two-panel latent canvas. The **left panel** is the target image being generated. The **right panel** is either blank for generation or the source image latent for editing.

2. **Task prompt tokens**
   - Generation prompt:

     ```text
     [Generate]: {prompt}
     ```

   - Editing prompt:

     ```text
     [Edit]: A diptych with two side-by-side images of the same scene. Compared to the right image, the left image has {edit_instruction}
     ```

3. **Staged training curriculum**

   ```text
   Stage 1: Text-to-image prior training
   Stage 2: Image-edit alignment training
   Stage 3: Unified generation + editing training
   Stage 4: High-quality SFT
   Stage 5: Reward/preference tuning or reranking
   Stage 6: Few-step distillation
   ```

4. **Foreground-emphasis edit loss**
   - Local edits affect small regions.
   - Uniform diffusion loss can cause the model to ignore small edits.
   - Use source-target differences to create a foreground edit mask and weight the loss more strongly in changed regions.

5. **Compact U-Net efficiency tricks**
   - Remove high-resolution self-attention.
   - Use mobile/expanded depthwise-separable convolutions where possible.
   - Use MQA/GQA instead of full multi-head attention for cross-attention.
   - Use QK/RMS normalization for stable attention.
   - Use fewer transformer blocks at high resolution.

6. **Improve the VAE**
   - DreamLite’s tiny VAE is good for speed but weak for text, identity, logos, and high-frequency details.
   - For this project, use a stronger f8c16 VAE.

---

## 2. Final Model Overview

### Model Name

```text
MoE-DreamLite-HQ-v1
```

### High-Level Architecture

```text
Text instruction + optional source image
        |
        v
+-----------------------------+
| Frozen 500M MoE VLM          |
| COARSE semantic conditioning |
| (NOT grounding -- Section 0) |
+--------------+--------------+
               |
               v
+-----------------------------+
| VLM Token Refiner            |
| (planner/mask head DROPPED)  |
+--------------+--------------+
               |
               v
        semantic condition tokens
               |
               v
Source image -> HQ VAE Encoder -> source latent
               |
   edit/preserve masks: from source-target diff (train)
                        or user / soft foreground (infer)
               |
Noisy target latent
               |
               v
+---------------------------------------------------+
| DreamLite-Style In-Context Latent U-Net            |
| generation input: [target_noise | blank]           |
| editing input:    [target_noise | source_latent]   |
+-------------------+-------------------------------+
                    |
                    v
             predicted velocity/noise
                    |
                    v
              target latent update
                    |
                    v
              HQ VAE Decoder
                    |
                    v
              generated / edited image
```

### Module Responsibilities

| Module | Frozen? | Purpose |
|---|---:|---|
| Custom 500M MoE VLM | Yes | **Coarse semantic conditioning only** (object identity, attributes, edit intent). NOT grounding/region reasoning — see Section 0. |
| VLM Token Refiner | No | Convert VLM hidden states into compact conditioning tokens for the U-Net |
| ~~Edit Planner / Mask Head~~ | — | **DROPPED (Section 0):** VLM can't localize. Masks now from source↔target diff (train) + user/soft-foreground (inference). |
| HQ f8c16 VAE | Frozen after training | Encode/decode images into high-fidelity latent space |
| In-Context Latent U-Net | No | Generate or edit image latents (the workhorse; learns edits from source latent + instruction + coarse VLM tokens) |
| VLM Ranker Head | **Deferred / external** | Candidate scoring needs image understanding the VLM lacks — use an external scorer or human eval for V1 |

---

## 3. Supported Tasks

The architecture should support these modes from day one:

```text
1. Text-to-image generation
2. Instruction-based image editing
3. Mask-guided image editing
4. No-op preservation / reconstruction
5. Local object edit
6. Background replacement
7. Style transfer
8. Text/logo-preserving edit
9. Product/fashion/ad creative edit
10. Candidate reranking and auto-retry
```

Optional later modes:

```text
1. Multi-image composition
2. Identity-preserving character editing
3. High-resolution tiled editing
4. Few-step mobile/edge distilled model
```

---

## 4. Frozen 500M MoE VLM

### Role

The frozen VLM is the semantic world model. It should not be trained end-to-end during the first architecture build.

**Recalibrated by the Section 0 grounding test.** Use it ONLY for what it can actually do:

```text
CAN (coarse semantic signal):
- text instruction encoding
- coarse source-image semantic encoding (gist, dominant object, attributes)
- semantic alignment losses (loose)

CANNOT (verified weak — do NOT rely on these):
- object/region localization or "where" reasoning
- edit-type prediction / protected-region reasoning
- reliable prompt rewriting / edit-plan generation
- candidate reranking that depends on judging the image
```

Treat the VLM hidden states as a **soft semantic prior**, not a perception/planning oracle. The U-Net + source latent carry the real spatial/appearance information.

### Important Rule

Do **not** use only the final pooled embedding.

Extract sequence hidden states from multiple VLM layers:

```yaml
vlm_tapped_layers:
  - 50_percent_depth
  - 75_percent_depth
  - final_layer
```

If your VLM has useful lower-level visual features, optionally tap:

```yaml
extra_tapped_layer:
  - 25_percent_depth
```

### VLM Input Templates

#### Text-to-Image

```text
[Generate]: {caption_or_prompt}
```

#### Image Editing

```text
[Edit]: A diptych with two side-by-side images of the same scene.
Compared to the right image, the left image has {edit_instruction}.
Preserve all unrelated objects, identity, text, logos, lighting, and background unless the instruction says otherwise.
```

#### Planner Prompt — DEFERRED (Section 0)

> The planner prompt below assumed reliable image grounding the VLM does not have (it
> misidentifies objects and can't localize). **Do not depend on planner JSON.** Kept only as a
> target for *after* a grounding upgrade or VLM swap.

```text
(deferred) Analyze the input image and instruction. Return:
1. edit type   2. target object/region   3. protected objects/regions
4. whether a mask is needed   5. likely failure risks   6. rewritten instruction
```

### VLM Output to Cache

For each training sample, cache:

```json
{
  "sample_id": "...",
  "vlm_text_tokens": "tensor path",
  "vlm_image_tokens": "tensor path or null",
  "vlm_hidden_states": {
    "layer_50": "tensor path",
    "layer_75": "tensor path",
    "layer_final": "tensor path"
  },
  "planner_json": "DEFERRED (Section 0) -- VLM cannot reliably produce this; do not cache or train on it"
}
```

Cache the **hidden states + tokens** (the usable signal). Caching VLM outputs is essential for the $500 compute budget.

---

## 5. VLM Token Refiner

The VLM token refiner is a trainable bridge between the frozen VLM and the image generator.

### Why It Is Needed

Your frozen VLM tokens are not naturally aligned to the diffusion U-Net’s cross-attention space. The refiner learns:

```text
VLM token space -> diffusion conditioning space
```

### Recommended Refiner Spec

```yaml
vlm_token_refiner:
  type: perceiver_qformer_style_resampler
  trainable: true
  layers: 3
  width: 1024
  heads: 16
  kv_heads: 2
  mlp_ratio: 2.0
  qk_norm: true
  rms_norm: true
  output_tokens: 256
  output_dim: 1024
  estimated_params: 30M-50M
```

### Inputs

```text
- VLM text hidden states
- VLM image hidden states
- optional VLM router/expert features
- task token: generate/edit/no-op/masked-edit
- optional edit-type token
```

### Outputs

```text
C_vlm: [batch, 256, 1024]
```

These tokens are used as cross-attention keys/values inside the U-Net.

---

## 6. Mask Strategy (Planner Head DROPPED — see Section 0)

> **The original VLM-driven Edit Planner / Mask Head is removed.** The grounding test
> (Section 0) showed the frozen 500M VLM cannot reliably identify or localize objects, so a
> head that predicts `target_mask` / `preserve_mask` / `edit_type` / `risk_flags` from its
> features has no dependable signal to learn from. Building it would burn compute on a module
> the backbone can't support. Masks come from the sources below instead — none need VLM grounding.

### Mask Sources (primary path)

**Training:**

```text
- Use dataset masks when available (MagicBrush, COCO/Open Images segmentation, GRIT regions).
- Otherwise derive masks from source<->target PIXEL DIFFERENCES (Section 14) -- this is the
  main source and is exactly what the foreground-weighted edit loss already consumes.
```

**Inference:**

```text
- Use the user-supplied mask if provided (primary for precise local edits).
- Else fall back to a soft, whole-image foreground emphasis (no region targeting).
- (No planner-predicted mask -- the head is dropped.)
```

### What we lose vs. the original plan

```text
- No automatic edit-type classification.
- No automatic target/preserve region prediction at inference (user mask or global edit only).
- No risk-flag / auto-retry gating from the VLM.
```

### Optional future re-introduction

Only revisit a learned planner/mask head AFTER the VLM clears grounding baselines
(continue-train on GRIT/RefCOCO-style referring data) OR if a stronger external VLM is swapped in
(Section 0 fork). Until then, the diff/user mask path is the contract.

---

## 7. VAE Design

> **DECISION (2026-06-13): DON'T train a VAE — use the pretrained FLUX.1-schnell VAE (Apache-2.0,
> f8c16).** Stage 0 (training an f8c16 VAE with L1+LPIPS+edge+OCR+GAN+KL) is deleted entirely. The
> FLUX VAE is exactly f8c16, Apache-licensed (commercial-clean), and reconstructs text/logos/detail
> far better than a $150 self-trained VAE — which removes the project's single biggest fidelity RISK
> and ~$150 + GAN-tuning dev time. Architecture is unchanged (16-ch latent, 64×128 two-panel canvas,
> the 400M U-Net). Load via `diffusers.AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell",
> subfolder="vae")`; apply FLUX latent normalization (shift≈0.1159, scale≈0.3611); keep it FROZEN —
> used only to cache latents (~$20 one-time) + decode at inference. Sections 8 (VAE architecture) and 9
> (VAE losses) are now MOOT (no VAE training). Alternatives: SD3.5 VAE (c16, higher fidelity, Stability
> Community License — commercial cap); SDXL VAE (MIT but only c4 → text/detail downgrade).

### Recommended VAE (original self-train spec — superseded by the note above; kept for reference)

Use:

```text
VAE-HQ-f8c16   (now satisfied by the pretrained FLUX.1-schnell VAE, not self-trained)
```

Meaning:

```text
spatial compression: 8x
latent channels: 16
```

### Why f8c16

DreamLite-style U-Net benefits from local spatial detail. f8 preserves more spatial information than f16.

At common resolutions:

| Image Resolution | VAE Latent | Two-Panel DreamLite Latent |
|---:|---:|---:|
| 512 x 512 | 64 x 64 x 16 | 64 x 128 x 16 |
| 768 x 768 | 96 x 96 x 16 | 96 x 192 x 16 |
| 1024 x 1024 | 128 x 128 x 16 | 128 x 256 x 16 |

For a cheaper first version:

```text
VAE-HQ-f8c8
```

For a transformer/MMDiT version, consider:

```text
VAE-HQ-f16c64
```

But for this DreamLite-style U-Net path, use **f8c16**.

---

## 8. VAE Architecture Specification

### VAE Encoder

```yaml
vae_encoder:
  input: RGB_image
  compression: 8x
  latent_channels: 16
  stem:
    conv: 3x3
    channels: 128
  down_blocks:
    - resolution: H/2
      channels: 128
      resblocks: 2
      attention: false
    - resolution: H/4
      channels: 256
      resblocks: 2
      attention: false
    - resolution: H/8
      channels: 512
      resblocks: 3
      attention: optional_bottleneck_only
  bottleneck:
    channels: 512
    resblocks: 2
    attention: optional
  latent_head:
    conv: 3x3
    output_channels: 32
    split:
      mu: 16
      logvar: 16
```

### VAE Decoder

```yaml
vae_decoder:
  input: latent_z
  latent_channels: 16
  latent_projection:
    conv: 3x3
    channels: 512
  up_blocks:
    - resolution: H/4
      channels: 512
      resblocks: 3
      attention: optional_bottleneck_only
    - resolution: H/2
      channels: 256
      resblocks: 3
      attention: false
    - resolution: H
      channels: 128
      resblocks: 3
      attention: false
  output:
    norm: groupnorm
    activation: silu
    conv: 3x3_to_RGB
```

### Optional Global Detail Path

Add a low-cost detail-preservation path:

```text
input image
  -> anti-aliased downsample to latent resolution
  -> small residual conv stack
  -> project to latent channels
  -> add to encoder mu
```

This helps preserve:

```text
- text strokes
- logos
- product labels
- fine edges
- UI elements
- small objects
```

### VAE Parameter Target

```yaml
vae_params:
  cheap_v0: 35M-55M
  recommended_v1: 60M-90M
  high_quality_v2: 100M-160M
```

For the first serious model:

```text
60M-90M f8c16 VAE
```

---

## 9. VAE Losses

Train the VAE separately first, then freeze it.

```text
L_vae =
  1.0  * L1_reconstruction
+ 0.5  * LPIPS_perceptual
+ 0.1  * edge_gradient_loss
+ 0.05 * OCR_text_preservation_loss
+ 0.05 * identity_or_product_embedding_loss
+ 0.1  * adversarial_decoder_loss
+ beta * KL_loss
```

Recommended KL weight:

```text
beta = 1e-6 to 1e-5
```

### VAE Evaluation Metrics

Track:

```text
- PSNR
- SSIM
- LPIPS
- OCR accuracy on text-heavy samples
- face identity similarity on face subset
- product/logo embedding similarity
- reconstruction sharpness on high-frequency crops
```

Minimum acceptable VAE behavior:

```text
- readable medium-size text at 768/1024
- stable logos/product labels
- no major face drift
- no excessive blur on object edges
- good color preservation
```

---

## 10. DreamLite-Style In-Context Latent U-Net

### Input Layout

#### Text-to-Image

```text
z_t      = noisy target latent
z_blank  = learned blank latent or zero latent

model_input = concat_width(z_t, z_blank)
```

#### Image Editing

```text
z_t      = noisy target latent
z_src    = VAE.encode(source_image)

model_input = concat_width(z_t, z_src)
```

#### Masked Editing

Add edit and preserve masks at latent resolution:

```text
edit_mask_latent:     H/8 x W/8 x 1
preserve_mask_latent: H/8 x W/8 x 1
```

Input channels:

```text
latent_channels + edit_mask + preserve_mask
```

For f8c16:

```text
16 + 2 = 18 channels per panel
```

Recommended layout:

```text
left target panel:
  noisy target latent + edit mask + preserve mask

right condition panel:
  source latent or blank latent + panel position embedding
```

### Output

The U-Net predicts velocity/noise for the full two-panel layout, but only the left target half is used:

```text
pred = U-Net([target | condition], t, C_vlm)
pred_target = pred[:, :, :, :target_width]
```

---

## 11. U-Net Architecture Specification

Recommended main model:

```yaml
unet:
  type: dreamlite_style_in_context_latent_unet
  prediction: velocity
  objective: flow_matching
  input_latent_channels: 16
  optional_mask_channels: 2
  input_channels_per_panel: 18
  output_channels: 16
  block_channels: [256, 512, 896]
  transformer_blocks_per_stage: [0, 2, 4]
  mid_blocks: 1
  high_res_self_attention: false
  conv_type: expanded_depthwise_separable
  attention:
    cross_attention_dim: 1024
    attention_type: GQA_or_MQA
    heads: 16
    kv_heads: 2
    qk_norm: true
    rms_norm: true
  conditioning:
    - timestep_embedding
    - task_embedding_generate_edit_noop
    - panel_position_embedding_left_right
    - VLM_refiner_cross_attention
    - optional_edit_type_embedding
    - optional_mask_embedding
  estimated_params: 420M-650M
```

### Block Layout

```text
Input two-panel latent
  -> stem conv
  -> Down Block 1: channels 256, no transformer
  -> Down Block 2: channels 512, 2 transformer/cross-attn blocks
  -> Down Block 3: channels 896, 4 transformer/cross-attn blocks
  -> Mid Block: residual + cross-attn
  -> Up Block 3: channels 896, 4 transformer/cross-attn blocks
  -> Up Block 2: channels 512, 2 transformer/cross-attn blocks
  -> Up Block 1: channels 256, no high-res self-attention
  -> output conv to latent velocity/noise
```

### Conditioning Inside Each Attention Block

```text
hidden image features
        |
        v
self-attention or local attention if resolution is low enough
        |
        v
cross-attention to VLM condition tokens C_vlm
        |
        v
MLP / expanded separable convolution
        |
        v
residual output
```

### Efficiency Choices

Use:

```text
- BF16 training
- FlashAttention where available
- gradient checkpointing
- GQA/MQA for cross-attention
- remove high-res self-attention
- cache VAE latents
- cache VLM tokens
- progressive resolution schedule
```

---

## 12. Model Variants

> **MEASURED COST REALITY (2026-06-13, `modal_dreamlite.py` probe).** A 400.8M in-context U-Net
> (`dreamlite/unet.py`, the ~0.39B DreamLite-class V0-512) was throughput-probed on H100 bf16:
> **78 img/s best single-H100** (B=48, peak 73GB; OOM at B=64) → 8×H100 ~564 img/s, 8×B200 ~1,130.
> Curriculum cost (sample-views ≈ passes × 850K data): **100M views = $1,231 (B200) / $1,555 (H100);
> 200M = $2,461 / $3,111; 300M = $3,692 / $4,666** — PLUS VAE ~$100-200, caching ~$20, distillation
> ~$100, and 768/1024 finetunes at 2-4× the 512 rate. **CONCLUSION: the "$500" target is off by
> ~3-8×; a real V0 is ~$1.5-4k.** Cheaper paths: 256px pilot (~$300-500, proves pipeline), the 215M
> U-Net variant, T2I-only single-panel first (~½ cost), or fewer views (100M floor, underfit risk).
> The two-panel 64×128 latent is the cost driver (~8,192 conv positions/sample, memory-bound).
>
> **OPTIMIZED (nanogpt-style, 2026-06-13).** channels_last (NHWC) + torch.compile + fused AdamW gave
> **1.6× throughput, free** — H100 78→126 img/s, B200 232 img/s (measured; fits B=128@107GB of 192GB).
> Optimized 8×B200 cost: **100M views = $830, 200M = $1,660, 300M = $2,489** (was $1,231/$2,461/$3,692).
> GB200 is NOT on Modal (B200 is the ceiling + best img/$). FP8 skipped (convs don't FP8 cleanly;
> linears are a minority of a conv U-Net). **Muon** stacks on top as a SAMPLE-EFFICIENCY win (cuts the
> view count ~1.2-1.6×, not throughput): route transformer linears→Muon, convs→reshape-Muon or AdamW,
> norms/embeds→AdamW (reuse `optim.py`); real gain needs an AdamW-vs-Muon A/B in the pilot. REVISED
> realistic V0-512: **~$830 (100M, underfit-risk) to ~$1.66k (200M)** + VAE ~$150 + distill ~$100.
> $500 is reachable only as a 256px pilot (1/4 spatial) + these opts + Muon.
>
> **$300 PATH FOUND — DC-AE f32 + DiT (measured 2026-06-13).** Swap the FLUX f8c16 VAE for **DC-AE
> f32c32** (32x spatial compression, MIT-Han-Lab efficientvit/SANA; license = verify Apache): 512px ->
> 16x16x32 latent, two-panel canvas = **512 tokens** (vs the conv U-Net's 8,192 positions). Replace the
> conv U-Net with a small **in-context DiT** (`dreamlite/dit.py`, 340M, AdaLN-Zero + cross-attn to VLM,
> flow-matching). Probed on B200: **887 img/s** (vs conv U-Net 232) -> 8xB200 cost **100M views = $217,
> 200M = $435, 300M = $652** + caching $20 + distill $100, NO VAE training. **A 512px V0 base ≈ $217;
> 256px pilot ~$50-100.** TRADE-OFF: f32 recon < f8c16 on fine TEXT/detail (acceptable for V0's global+
> simple-local scope; graduate to f8c16 conv-U-Net (~$1.66k) if text fidelity becomes critical). Both
> generators built: dreamlite/dit.py (DC-AE $300 path) + dreamlite/unet.py (f8c16 quality path).

### V0: Pipeline Proof

```yaml
model_variant: MoE-DreamLite-HQ-v0
resolution: 512
vae: f8c8
unet_channels: [192, 384, 768]
transformer_blocks: [0, 1, 3]
vlm_refiner_layers: 2
estimated_unet_params: 250M-350M
purpose: prove VLM conditioning and in-context editing
```

### V1: Recommended Main Model

```yaml
model_variant: MoE-DreamLite-HQ-v1
resolution: 512_then_768_then_1024
vae: f8c16
unet_channels: [256, 512, 896]
transformer_blocks: [0, 2, 4]
vlm_refiner_layers: 3
estimated_unet_params: 420M-650M
purpose: best quality/compute tradeoff
```

### V2: Larger Research Model

```yaml
model_variant: MoE-DreamLite-HQ-v2
resolution: 1024_native
vae: f8c16_or_f8c32
unet_channels: [320, 640, 1024]
transformer_blocks: [0, 3, 6]
vlm_refiner_layers: 4
estimated_unet_params: 700M-1.0B
purpose: stronger high-res generation/editing if compute allows
```

---

## 13. Flow Matching Objective

Use rectified-flow / flow-matching velocity prediction.

For target image latent:

```text
z_1 = VAE.encode(target_image)
z_0 = Gaussian noise
t   = sampled timestep in [0, 1]

z_t = (1 - t) * z_0 + t * z_1
v_target = z_1 - z_0
```

The model predicts:

```text
v_pred = U-Net(z_t, condition, t)
```

Base loss:

```text
L_flow = mean_squared_error(v_pred, v_target)
```

For editing:

```text
condition = {
  source_latent,
  frozen_VLM_tokens,
  edit_mask,
  preserve_mask,
  task_token
}
```

---

## 14. Foreground-Emphasis Edit Loss

For local edits, compute a changed-region mask from source and target images.

### Mask Construction

```python
# Pseudocode
source = load_source_image()
target = load_target_image()

# pixel-space difference
diff = mean(abs(source - target), channel="rgb")

# threshold changed pixels
mask = diff > threshold

# cleanup
mask = dilate(mask, kernel_size=7)
mask = remove_small_components(mask, min_area=64)
mask = max_pool_to_latent_resolution(mask, factor=8)

# soft mask optional
mask = gaussian_blur(mask)
```

### Loss

```text
L_edit = mean( w * (v_pred - v_target)^2 )
```

Where:

```text
w = 1 + alpha * foreground_mask
```

Recommended:

```yaml
foreground_loss:
  alpha_local_edit: 3.0_to_8.0
  alpha_medium_edit: 1.5_to_3.0
  alpha_global_edit: 0.5_to_1.0
```

Also include preservation loss outside the edited region:

```text
L_preserve = mean( (1 - mask) * abs(predicted_clean_latent - target_latent) )
```

---

## 15. Full Training Loss

```text
L_total =
  L_flow
+ lambda_edit      * L_foreground_weighted_edit
+ lambda_preserve  * L_unedited_region_preservation
+ lambda_noop      * L_noop_reconstruction
+ lambda_vlm_align * L_vlm_semantic_alignment   # loose semantic prior only
+ lambda_ocr       * L_text_logo_preservation
+ lambda_id        * L_identity_or_product_consistency
# L_planner REMOVED (Section 0: planner/mask head dropped)
```

Recommended initial weights:

```yaml
loss_weights:
  flow: 1.0
  foreground_edit: 0.5
  preserve: 0.2
  noop: 0.2
  vlm_alignment: 0.05
  ocr_text_logo: 0.05
  identity_product: 0.05
```

Start simple:

```text
Phase 1: L_flow only
Phase 2: L_flow + foreground_edit (diff-derived masks)
Phase 3: add preserve/no-op + ocr/identity preservation
Phase 4: (planner losses removed; optional external-ranker pass instead)
```

---

## 16. Training Data Overview

The complete dataset stack has seven buckets:

```text
1. VAE reconstruction data
2. Text-to-image prior data
3. Instruction image-editing pairs
4. Mask / grounding / segmentation data
5. Text/logo/OCR fidelity data
6. Preference / reward / reranking data
7. Domain-specific product/fashion/ad data
```

Use Hugging Face datasets where possible, but verify licenses before commercial use.

### Verification status (checked 2026-06-13 via HF Hub)

The DreamLite reference and all 10 primary dataset IDs were verified to **exist with the claimed
sizes**. DreamLite paper is real: arXiv 2603.28713 (Feng et al., 30 Mar 2026) — in-context spatial
concatenation, task-progressive joint pretraining, mobile U-Net, step distillation. **Licenses
(the part that gates commercial use):**

```text
CLEAN for commercial (permissive):
  ScaleEdit-12M ........ MIT        (largest edit source)
  OmniEdit-Filtered-1.2M MIT
  fine-t2i ............. Apache-2.0 (main T2I)
  HPDv2 ................ Apache-2.0 (preference)
  VINS-120K ............ Apache-2.0 (4K-res edits)
  CrispEdit-2M ......... CC-BY-4.0  (attribution)
  MagicBrush ........... CC-BY-4.0  (attribution, human triplets+masks)

CAVEATS:
  HQ-Edit ............. CC-BY-NC-4.0  -> NON-COMMERCIAL. Drop for commercial; research-only.
  GPT-Image-Edit-1.5M . CC-BY-4.0 BUT images regenerated by GPT-Image-1
                        -> OpenAI-ToS risk for training a commercial competitor. Treat as research
                        or replace with the MIT/Apache sources above.
  google/tecci ........ CC-BY-4.0 but BENCHMARK (test split only) + Gemini-generated subset
                        -> use for EVAL only, not training.
  Note: ScaleEdit / OmniEdit contain some model-distilled samples; for a strict commercial build,
        prefer their human/real-image subsets and document provenance.
```

**Clean commercial V1 core** = ScaleEdit (MIT) + OmniEdit (MIT) + fine-t2i (Apache) + CrispEdit +
MagicBrush (CC-BY) + VINS (Apache) + HPDv2 (Apache). That alone covers VAE, T2I prior, edit
pretraining, SFT, and preference — sufficient for the whole V1 curriculum without the NC/ToS-risk sets.

---

## 17. Main Hugging Face Dataset Stack

### Core Editing Datasets

| Dataset | HF ID | Use | Suggested First-Run Size |
|---|---|---|---:|
| ScaleEdit-12M | `InternVL-U/ScaleEdit-12M` | Large-scale edit pretraining across many edit types | 250K-500K |
| GPT-Image-Edit-1.5M | `UCSC-VLAA/GPT-Image-Edit-1.5M` | High-quality modern edit SFT/pretraining | 100K-200K |
| CrispEdit-2M | `WeiChow/CrispEdit-2M` | Balanced local/global edit operations | 100K-200K |
| OmniEdit Filtered | `TIGER-Lab/OmniEdit-Filtered-1.2M` | Additional edit diversity | 50K-100K |
| MagicBrush | `osunlp/MagicBrush` | Human edit triplets, masks, validation | all usable |
| HQ-Edit | `UCSC-VLAA/HQ-Edit` | Clean high-quality edit SFT | 30K-100K |

### Text-to-Image Prior Datasets

| Dataset | HF ID | Use | Suggested First-Run Size |
|---|---|---|---:|
| Fine-T2I | `ma-xu/fine-t2i` | Main T2I prior source | 150K-300K |
| JourneyDB | `JourneyDB/JourneyDB` | Aesthetic/style prior | 30K-100K |
| DiffusionDB | `poloclub/diffusiondb` | Prompt diversity after filtering | 20K-100K |
| COYO-700M | `kakaobrain/coyo-700m` | Optional large web-scale source | tiny filtered subset only |

### VAE Reconstruction Data

| Dataset | HF ID / Source | Use | Suggested First-Run Size |
|---|---|---|---:|
| Open Images V7 | multiple HF mirrors, e.g. `bitmind/open-images-v7` | Natural images, objects, masks | 50K-200K |
| MS COCO | multiple HF mirrors, e.g. `shunk031/MSCOCO` | Real images, captions, segmentation | 20K-80K |
| VINS-120K | `openvivo/VINS-120K` | High-res editing and reconstruction | 10K-30K |
| Fine-T2I filtered | `ma-xu/fine-t2i` | Aesthetic synthetic/curated images | 50K-100K |
| Fashion Products | `ashraq/fashion-product-images-small` | Product/fashion reconstruction | 20K-40K |
| CaptionedSynthText | `wendlerc/CaptionedSynthText` | OCR/text reconstruction | 30K-80K |

### Text, Logo, OCR, and UI Data

| Dataset | HF ID | Use | Suggested First-Run Size |
|---|---|---|---:|
| TECCI | `google/tecci` | Text-rich image editing benchmark/data | small eval/SFT subset |
| CaptionedSynthText | `wendlerc/CaptionedSynthText` | Text rendering and OCR reconstruction | 30K-80K |
| TextCaps | multiple HF mirrors | Text-in-image understanding | filtered subset |
| ScaleEdit text subsets | `InternVL-U/ScaleEdit-12M` | Poster, GUI, object-surface text edits | 10K-50K |

### Mask / Grounding / Preserve-Region Data

| Dataset | HF ID | Use |
|---|---|---|
| MS COCO | `shunk031/MSCOCO` or similar | Object masks, segmentation, captions |
| Open Images V7 | `bitmind/open-images-v7` or similar | Boxes, masks, objects, relationships |
| GRIT | `zzliang/GRIT` | Phrase-to-region grounding |
| MagicBrush | `osunlp/MagicBrush` | Edit masks where available |

### Preference / Ranking Data

| Dataset | HF ID | Use | Suggested First-Run Size |
|---|---|---|---:|
| HPDv2 | `ymhao/HPDv2` | Human preference for generated images | 50K-200K |
| ImageRewardDB | `zai-org/ImageRewardDB` | Expert T2I comparison pairs | all/filter |
| T2I DPO Human Preferences | `datapointai/text-2-image-dpo-human-preferences-full` | Pairwise prompt/image preferences | 50K-200K |
| Pico-Banana preference subset | check available HF mirror/paper page | Editing-specific preferences | license-cleared subset |
| Own candidates | generated locally | Best for your actual model failures | required |

### Domain-Specific Product/Fashion/Ad Data

| Dataset | HF ID | Use |
|---|---|---|
| Fashion Product Images | `ashraq/fashion-product-images-small` | Product/fashion VAE + SFT |
| DeepFashion Multimodal | `Marqo/deepfashion-multimodal` | Fashion captions and metadata |
| Amazon Reviews 2023 | `McAuley-Lab/Amazon-Reviews-2023` | Product metadata/captions; verify image availability |

---

## 18. First-Run Dataset Build

The first serious run should be compact and high-quality.

```yaml
first_run_dataset:
  vae:
    total: 100k
    mixture:
      open_images_or_coco_real: 40k
      fine_t2i_or_journeydb_filtered: 30k
      text_logo_ocr: 15k
      product_fashion: 10k
      high_res_vins: 5k

  t2i_prior:
    total: 200k
    mixture:
      fine_t2i: 150k
      journeydb_filtered: 30k
      diffusiondb_filtered: 20k

  edit_pretraining:
    total: 500k
    mixture:
      scaleedit_12m_balanced: 250k
      gpt_image_edit_1_5m_filtered: 125k
      crispedit_2m_balanced: 100k
      magicbrush_hqedit: 25k

  sft:
    total: 50k
    mixture:
      gpt_image_edit_best: 20k
      hq_edit_best: 15k
      magicbrush: 8k
      vins_highres: 5k
      text_logo: 2k

  reward_ranking:
    total: 100k
    mixture:
      hpdv2: 40k
      imagerewarddb: 20k
      t2i_dpo_preferences: 20k
      own_generated_candidates: 20k
```

Total first-run data:

```text
~850K image/generation/edit samples
~100K reward/preference samples
```

This is a realistic first build for verifying the architecture.

---

## 19. Full Dataset Build After V1 Works

```yaml
full_dataset_after_v1:
  vae:
    total: 500k-1M
  t2i_prior:
    total: 1M-2M
  edit_pretraining:
    total: 1M-3M
  sft:
    total: 100k-300k
  reward_ranking:
    total: 300k-1M
```

Use the full build only after:

```text
- VAE reconstruction is good
- in-context editing works at 512px
- masks improve local edit accuracy
- VLM refiner improves instruction following
- loss does not collapse during unified training
```

---

## 20. Unified Training Sample Format

Convert all datasets to a single schema.

### Editing Sample

```json
{
  "sample_id": "scaleedit_000001",
  "task": "edit",
  "edit_type": "object_replacement",
  "source_image": "images/source/scaleedit_000001.webp",
  "target_image": "images/target/scaleedit_000001.webp",
  "instruction": "replace the red backpack with a black leather bag",
  "source_caption": "a person wearing a red backpack on a city street",
  "target_caption": "a person wearing a black leather bag on a city street",
  "mask": "masks/scaleedit_000001.png",
  "quality": {
    "instruction_following": 3,
    "editing_consistency": 3,
    "generation_quality": 3
  },
  "source_dataset": "InternVL-U/ScaleEdit-12M",
  "license": "verify_before_commercial_use"
}
```

### Text-to-Image Sample

```json
{
  "sample_id": "fine_t2i_000001",
  "task": "generate",
  "source_image": null,
  "target_image": "images/target/fine_t2i_000001.webp",
  "instruction": "[Generate]: a studio product photo of a black smartwatch on a white background",
  "source_caption": null,
  "target_caption": "a studio product photo of a black smartwatch on a white background",
  "mask": null,
  "source_dataset": "ma-xu/fine-t2i",
  "license": "verify_before_commercial_use"
}
```

### No-Op Preservation Sample

```json
{
  "sample_id": "noop_000001",
  "task": "edit",
  "edit_type": "no_op_preservation",
  "source_image": "images/source/noop_000001.webp",
  "target_image": "images/source/noop_000001.webp",
  "instruction": "preserve the image exactly without changing anything",
  "mask": "masks/all_zero.png",
  "source_dataset": "derived",
  "license": "inherits_source_license"
}
```

---

## 21. Dataset Filtering Rules

Filtering is more important than raw dataset size.

### General Image Filters

```yaml
image_filters:
  min_short_side: 512
  preferred_short_side: 768
  remove:
    - broken_images
    - extreme_blur
    - heavy_jpeg_artifacts
    - watermarks_when_possible
    - near_duplicates
    - unsafe_or_unwanted_content
    - wrong_aspect_extremes
```

### Editing Pair Filters

Compute:

```text
change_ratio = changed_pixels / total_pixels
```

Suggested buckets:

```yaml
edit_change_buckets:
  local: 0.01_to_0.25
  medium: 0.25_to_0.60
  global: 0.60_to_0.95
```

Reject:

```text
- change_ratio < 0.005 unless it is a deliberate no-op
- change_ratio > 0.98 unless it is a style/global rewrite
- source-target pairs with obvious mismatched identity
- examples where instruction does not match target
- OCR text edits where expected text is absent
- target images with severe artifacts
```

### Task Balancing

Balance edit types:

```yaml
edit_type_balance:
  object_add: 10%
  object_remove: 10%
  object_replace: 12%
  color_material_change: 12%
  background_change: 10%
  relighting: 8%
  style_transfer: 10%
  text_logo_edit: 8%
  product_ad_edit: 8%
  no_op_preservation: 12%
```

No-op preservation is not optional. It teaches the editor not to drift.

---

## 22. Training Curriculum

### Stage 0: Load Pretrained VAE + Cache Latents (NO VAE training — see Section 7)

VAE training is REMOVED. Use the frozen FLUX.1-schnell f8c16 VAE; Stage 0 is just a one-time
encode pass to cache latents.

```yaml
stage_0_vae:
  vae: black-forest-labs/FLUX.1-schnell   # frozen, Apache-2.0, f8c16
  normalization: {shift: 0.1159, scale: 0.3611}   # FLUX latent shift/scale
  action: cache_latents_only              # encode all images once -> cached latent shards
  no_training: true                        # ~$20 one-time, not ~$150 + GAN tuning
  # (original self-train recipe — L1/LPIPS/edge/KL/GAN/OCR — is superseded; see Section 7 note)
```

### Stage 1: Text-to-Image Prior

```yaml
stage_1_t2i_prior:
  input_layout: "[noisy_target | blank]"
  prompt_template: "[Generate]: {caption}"
  resolution: 512
  data: fine_t2i_journeydb_diffusiondb_filtered
  objective: flow_matching_velocity
  trainable:
    - unet
    - vlm_refiner
  frozen:
    - vlm
    - vae
```

### Stage 2: Edit Alignment

```yaml
stage_2_edit_alignment:
  input_layout: "[noisy_target | source_latent]"
  prompt_template: "[Edit]: Compared to the right image, the left image has {instruction}"
  resolution: 512
  data: scaleedit_gpt_image_edit_crispedit_magicbrush_hqedit
  losses:
    - flow_matching_velocity
    - foreground_weighted_edit_loss   # masks from source<->target diff
    - preserve_loss
  trainable:
    - unet
    - vlm_refiner
    # edit_planner REMOVED (Section 0)
```

### Stage 3: Unified Generation + Editing

```yaml
stage_3_unified:
  input_layouts:
    generate: "[noisy_target | blank]"
    edit: "[noisy_target | source_latent]"
    noop: "[noisy_target | source_latent]"
  mixture:
    text_to_image: 35%
    editing: 45%
    no_op_preservation: 10%
    text_logo_ocr: 5%
    mask_control: 5%
  resolution_schedule:
    - 512
    - 768
    - 1024_short_finetune
```

### Stage 4: High-Quality SFT

```yaml
stage_4_sft:
  data:
    gpt_image_edit_best: 40%
    hq_edit: 25%
    magicbrush: 15%
    vins_high_res: 10%
    text_logo_ocr: 5%
    own_domain_data: 5%
  learning_rate: low
  goal:
    - improve aesthetics
    - improve instruction adherence
    - reduce artifacts
    - improve preservation
```

### Stage 5: Preference / Reward / Reranking (VLM ranker DEFERRED — Section 0)

> A ranker **on top of the frozen VLM** is unreliable here: scoring preservation/identity/artifacts
> needs the image understanding the VLM lacks (VQAv2 ~0.28). For V1, **prefer (a) generator-side
> preference loss (DPO-style on HPDv2 / ImageRewardDB / T2I-DPO) which doesn't need a VLM judge, or
> (b) an EXTERNAL scorer (e.g. an off-the-shelf reward model)**. Re-introduce a VLM ranker only after
> a grounding upgrade. The spec below is kept as the deferred target.

```yaml
stage_5_preference:   # deferred VLM-ranker variant; use generator-side DPO or external scorer for V1
  ranker_input:
    - source_image_optional
    - instruction
    - candidate_output
  ranker_outputs:
    - instruction_following_score
    - preservation_score
    - identity_score
    - text_logo_score
    - artifact_score
    - final_preference_score
  data:
    - HPDv2
    - ImageRewardDB
    - T2I_DPO_preferences
    - own_generated_candidates
```

At inference:

```text
1. generate 4 candidates
2. score candidates with an EXTERNAL scorer / human eval (VLM ranker deferred — Section 0)
3. return best
4. auto-retry if below threshold
```

### Stage 6: Few-Step Distillation

Only distill after the base model is strong.

```yaml
stage_6_distillation:
  teacher: MoE-DreamLite-HQ-v1_base
  student: same_arch_or_smaller
  target_steps:
    - 8
    - 4
  objective:
    - distribution_matching
    - teacher_student_latent_matching
    - preference_preservation
  warning: do_not_distill_a_weak_teacher
```

---

## 23. Compute Optimization Plan

To stay near the $500 compute budget:

```text
1. Cache VAE latents before U-Net training.
2. Cache frozen VLM tokens before U-Net training.
3. Use BF16.
4. Use FlashAttention.
5. Use gradient checkpointing.
6. Use streaming datasets and prefiltered local shards.
7. Train at 512 first.
8. Only do short 768/1024 finetunes after 512 works.
9. Do not train the VLM end-to-end.
10. Do not train on full datasets before validating architecture.
```

### Local Cache Structure

```text
data_cache/
  vae_latents/
    train_00000.safetensors
    train_00001.safetensors
  vlm_tokens/
    train_00000.safetensors
    train_00001.safetensors
  masks/
    train_00000.safetensors
  metadata/
    train_manifest.parquet
```

### Training Shards

Recommended shard size:

```text
200MB-500MB per shard
```

---

## 24. Inference Pipeline

### Text-to-Image

```text
Input prompt
  -> VLM encode prompt
  -> VLM refiner produces condition tokens
  -> initialize target noise latent
  -> concatenate [target_noise | blank_latent]
  -> flow sampling
  -> keep left target latent
  -> VAE decode
  -> output image
```

### Image Editing

```text
Input source image + instruction (+ optional user mask)
  -> VAE encode source image
  -> VLM encode source image + instruction (COARSE semantic tokens only)
  -> mask = user mask if provided, else soft whole-image foreground emphasis
     (no planner-predicted mask -- Section 0)
  -> initialize target noise latent
  -> concatenate [target_noise | source_latent]
  -> flow sampling with VLM condition + mask
  -> keep left target latent
  -> VAE decode
  -> generate multiple candidates
  -> select best via EXTERNAL scorer / human eval (VLM ranker deferred -- Section 0)
  -> output edited image
```

### Suggested Sampling Settings

```yaml
sampling:
  base_model_steps: 24-32
  distilled_model_steps: 4-8
  cfg_generation: 3.0-5.0
  cfg_editing: 2.0-4.0
  image_cfg_editing: 1.0-2.0
  candidates_per_prompt: 4
```

---

## 25. Evaluation Suite

### VAE Evaluation

```text
- reconstruction PSNR/SSIM/LPIPS
- OCR preservation
- face identity preservation
- product/logo preservation
- high-frequency crop sharpness
```

### Text-to-Image Evaluation

```text
- prompt adherence
- object count
- color binding
- spatial relations
- aesthetics
- artifact rate
```

### Editing Evaluation

```text
- edit success
- instruction following
- source preservation
- identity preservation
- text/logo preservation
- background drift
- over-edit rate
- under-edit rate
- artifact rate
```

### Internal Human Eval Rubric

Score each output 1-5:

```yaml
rubric:
  instruction_following: 1-5
  preservation: 1-5
  realism: 1-5
  text_logo_correctness: 1-5
  identity_product_consistency: 1-5
  artifacts: 1-5
  overall: 1-5
```

---

## 26. Implementation Checklist

### Phase A: Data

```text
[ ] Download/stream small filtered subsets.
[ ] Normalize all samples to unified JSON/Parquet schema.
[ ] Generate derived edit masks.
[ ] Deduplicate with pHash/CLIP similarity.
[ ] Filter by resolution and quality.
[ ] Build VAE/T2I/Edit/SFT/Reward shards.
```

### Phase B: VAE

```text
[ ] Train f8c8 VAE quick test.
[ ] Train f8c16 VAE serious version.
[ ] Evaluate text/logo/faces/product reconstruction.
[ ] Freeze VAE.
[ ] Cache latents.
```

### Phase C: VLM Cache

```text
[ ] Define Generate/Edit/No-op prompt templates.
[ ] Extract multi-layer VLM hidden states.
[ ] Cache VLM tokens.
[ ] (planner JSON dropped — Section 0; cache hidden states + tokens only)
```

### Phase D: U-Net Training

```text
[ ] Train 512px T2I prior.
[ ] Train 512px edit alignment.
[ ] Train unified model.
[ ] Add no-op preservation examples.
[ ] Add mask/preserve control.
[ ] Run high-quality SFT.
```

### Phase E: Quality Control

```text
[ ] Generate 4 candidates per eval prompt.
[ ] Wire external scorer / DPO selection (VLM ranker deferred — Section 0).
[ ] Add auto-retry loop.
[ ] Distill to 8/4 steps only after base quality is strong.
```

---

## 27. Recommended Config File

```yaml
model:
  name: MoE-DreamLite-HQ-v1

  frozen_vlm:
    params: 500M
    type: custom_moe_vlm
    frozen: true
    tapped_layers: [0.50, 0.75, 1.00]

  vlm_refiner:
    type: perceiver_resampler
    layers: 3
    width: 1024
    heads: 16
    kv_heads: 2
    output_tokens: 256
    qk_norm: true
    rms_norm: true

  vae:
    type: kl_vae
    compression: 8
    latent_channels: 16
    params: 60M-90M
    frozen_after_training: true

  unet:
    type: in_context_latent_unet
    prediction: velocity
    input_layout:
      generate: concat_width_target_blank
      edit: concat_width_target_source
    latent_channels: 16
    mask_channels: 2
    block_channels: [256, 512, 896]
    transformer_blocks: [0, 2, 4]
    mid_blocks: 1
    high_res_self_attention: false
    conv_type: expanded_depthwise_separable
    cross_attention_dim: 1024
    attention_type: gqa
    heads: 16
    kv_heads: 2
    qk_norm: true

training:
  precision: bf16
  objective: flow_matching
  timestep_sampling: logit_normal_or_uniform
  cache_vae_latents: true
  cache_vlm_tokens: true
  gradient_checkpointing: true
  flash_attention: true

  stages:
    - name: vae_training
      resolution: [256, 512, 768]
      train: vae_only

    - name: t2i_prior
      resolution: 512
      layout: target_blank
      train: [unet, vlm_refiner]

    - name: edit_alignment
      resolution: 512
      layout: target_source
      train: [unet, vlm_refiner]   # edit_planner removed (Section 0)

    - name: unified_training
      resolution: [512, 768]
      mixture:
        text_to_image: 0.35
        edit: 0.45
        noop: 0.10
        text_logo: 0.05
        mask_control: 0.05

    - name: sft
      resolution: [768, 1024]
      learning_rate: low

    - name: preference        # VLM ranker deferred (Section 0): use generator-side DPO or external scorer
      train: generator_dpo_or_external_scorer

    - name: distillation
      target_steps: [8, 4]
      only_after_base_is_good: true
```

---

## 28. References and Starting Points

### Architecture / Method

- DreamLite paper: https://arxiv.org/pdf/2603.28713
- DreamLite repo: https://github.com/ByteVisionLab/DreamLite
- IP-Adapter: https://arxiv.org/abs/2308.06721
- ControlNet: https://arxiv.org/abs/2302.05543
- Stable Diffusion 3 / MMDiT / rectified flow: https://stability.ai/news-updates/stable-diffusion-3-research-paper
- DiT: https://arxiv.org/abs/2212.09748
- REPA representation alignment: https://arxiv.org/abs/2410.06940

### Dataset Starting Points

- ScaleEdit-12M: https://huggingface.co/datasets/InternVL-U/ScaleEdit-12M
- GPT-Image-Edit-1.5M: https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M
- CrispEdit-2M: https://huggingface.co/datasets/WeiChow/CrispEdit-2M
- OmniEdit-Filtered-1.2M: https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M
- MagicBrush: https://huggingface.co/datasets/osunlp/MagicBrush
- HQ-Edit: https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit
- Fine-T2I: https://huggingface.co/datasets/ma-xu/fine-t2i
- JourneyDB: https://huggingface.co/datasets/JourneyDB/JourneyDB
- DiffusionDB: https://huggingface.co/datasets/poloclub/diffusiondb
- VINS-120K: https://huggingface.co/datasets/openvivo/VINS-120K
- TECCI: https://huggingface.co/datasets/google/tecci
- CaptionedSynthText: https://huggingface.co/datasets/wendlerc/CaptionedSynthText
- HPDv2: https://huggingface.co/datasets/ymhao/HPDv2
- ImageRewardDB: https://huggingface.co/datasets/zai-org/ImageRewardDB
- T2I DPO Human Preferences: https://huggingface.co/datasets/datapointai/text-2-image-dpo-human-preferences-full
- Fashion Product Images: https://huggingface.co/datasets/ashraq/fashion-product-images-small
- DeepFashion Multimodal: https://huggingface.co/datasets/Marqo/deepfashion-multimodal
- Amazon Reviews 2023: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

---

## 29. Final Recommended Build Order

Build in this order:

```text
1. Convert datasets into unified schema.
2. Train/evaluate f8c8 VAE quick version.
3. Train/evaluate f8c16 VAE serious version.
4. Cache VAE latents.
5. Cache frozen VLM tokens.
6. Train VLM token refiner only (planner/mask heads dropped — Section 0).
7. Train 512px T2I prior with [target | blank].
8. Train 512px edit alignment with [target | source].
9. Train unified generation/editing model.
10. Add no-op preservation and mask control.
11. Run high-quality SFT.
12. Candidate selection via external scorer / DPO (VLM ranker deferred — Section 0).
13. Distill to 8/4 steps if the base model is strong.
```

The most important first milestone:

```text
A 512px model that can perform local edits while preserving background, identity, text, and logos better than a plain small diffusion model.
```

That proves your frozen 500M MoE VLM is providing useful semantic control.
