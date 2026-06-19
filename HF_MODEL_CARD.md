---
license: apache-2.0
language:
- en
tags:
- mixture-of-experts
- multimodal
- any-to-text
- vision-language
- audio-language
- speech
library_name: pytorch
pipeline_tag: image-text-to-text
---

# MoE-Omni 500M — a unified any-modality → text Mixture-of-Experts model

A **500M-parameter Mixture-of-Experts decoder LLM** extended into a single **omni-modal** model that
takes **image · video · audio · speech · text** (in any combination) and produces text. Built from
scratch as a research project: the LLM was pretrained, context-extended to 2048, then connected to
frozen encoders via small MLP projectors (TinyLLaVA / Ultravox style) and co-trained.

> Research artifact at the 500M scale — capable but small. Expect hallucination on hard inputs. Not
> affiliated with any provider.

## Architecture

| Piece | Choice |
|---|---|
| LLM | custom **MoE transformer**, d_model 768, 16 layers, SwiGLU experts, aux-free bias balancing |
| Context | 2048 (continued-pretrain from 1024) |
| Vision (image+video) | `google/siglip2-so400m-patch14-384` (frozen) → `mm_projector` |
| Audio — sounds | `laion/clap-htsat-unfused` (frozen) → `audio_projector` |
| Audio — speech | `openai/whisper-small` encoder (frozen, stack-2 → 750 tok) → `speech_projector` |
| Connector | 2-layer GELU MLP per modality; features spliced at sentinel tokens (`<image>`,`<video>`,`<audio>`,`<speech>`) |

Encoders are **frozen in every stage**; only the projectors (+ the LLM, in SFT/joint stages) are trained.

## Modalities (one checkpoint: `weights/joint5_model.pt`)

- **Text** — language modeling.
- **Image** — captioning / VQA (LLaVA-Instruct + COCO).
- **Video** — sampled frames through the same vision path.
- **Audio (sounds)** — general audio captioning (Clotho): "crickets chirp as a car drives by".
- **Speech** — spoken-English understanding (LibriSpeech ASR + VoiceAssistant spoken-QA): hears a
  spoken question and answers it.
- **Image + voice (zero-shot)** — ask a spoken question *about* an image; the model grounds the
  question in the image ("how many people?" → "three girls").

## Evaluation (500M, via lmms-eval and our harness)

| Benchmark | Metric | Score | (reference: LLaVA-1.5-7B) |
|---|---|---|---|
| POPE (full) | accuracy / F1 | 50.0% / 66.7% | ~85% |
| GQA | exact-match | 27.85% | ~62% |
| VQAv2 (val) | VQA-acc | 37.29% | ~78% |
| LibriSpeech test-clean | WER | 28.54% | (Whisper-small alone ~3%) |

Numbers reflect the 500M scale and a frozen-LLM / small-projector setup; they are honest, not tuned for
leaderboard presentation.

### Text-only benchmarks (the base MoE LLM)

The backbone is a from-scratch MoE LLM. 0-shot via EleutherAI lm-evaluation-harness (7-task MicroLlama
convention), measured on our own checkpoints:

| Model | Params (total / active) | Tokens | HellaSwag | OBQA | WinoGrande | ARC-c | ARC-e | BoolQ | PIQA | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| **500M MoE** (this backbone) | 500M / 169M | 40B | 41.56 | 29.80 | 51.30 | 22.44 | 42.76 | 50.98 | 69.53 | **44.05** |
| 130M MoE (smaller sibling) | 140M / 62M | 10B | 32.54 | 28.20 | 52.17 | 23.46 | 37.71 | 61.31 | 65.40 | **42.97** |

Only ~169M of the 500M params are active per token (top-k MoE routing). These are the text scores of the
*base LLM* that the multimodal projectors are attached to.

## Tool use / function calling

The same LLM was also tuned as a **function caller** and folded into the unified model (`joint6`, the 6th
path). Trained on Nemotron-Agentic-v1 + BitAgent + interstellar (Needle-style single-shot: query+tools →
`[{"name","arguments"}]`), with a **grammar-constrained decoder** (schema-constrained names + arg keys →
0% parameter hallucination) and Needle-style weighted loss.

| Benchmark | Metric | Score |
|---|---|---|
| BitAgent (held-out) | exact-match / Name-F1 | **59% / 100%** |
| Nemotron (held-out) | exact-match | 37% |
| BFCL v3 Simple / Multiple | AST acc | 23% / 29% |
| BFCL v3 Relevance | call-when-needed | 78% |

Honest notes: strong on **single-call** on its own distributions; weak on **parallel** (trained
single-shot) and far below larger purpose-built callers on BFCL (different distribution + arg-value
precision at 500M). A multi-turn/agentic variant (`tools_traj`) reaches **95% on BFCL irrelevance**
(knowing when *not* to call) but trades away must-call — at 500M, rigid calling and abstention are in
tension. See `docs/` for the full study.

## Omni-agent (spoken + text tool use)

`joint7` is the flagship: an **8-path omni-agent** that runs the full loop in one checkpoint —
**hear/read a request → call the right tool → observe the result → summarize it in words** — on top
of image / video / audio / speech / text. Verified end-to-end:

```
🎙️ "What is the current temperature in San Francisco?"  (spoken → Whisper)
   → [{"name":"get_current_temperature","arguments":{"city":"San Francisco"}}]
   → tool result {"temperature":18,...}
   → "The current temperature in San Francisco is 18°C."
```

Use **`code/agent.py`** for reliable tool use — it bakes in a `tool_choice="required"` forced-call
policy (seed `[`, snap the tool name to the schema, repair JSON), so the model commits to a call when
tools are provided instead of chatting, then summarizes the result:

```python
from agent import agent_tool_call, agent_summarize
call   = agent_tool_call(vlm, tok, dev, tools_json, query="What's the weather in Tokyo?")  # or speech_features=...
answer = agent_summarize(vlm, tok, dev, tools_json, call, tool_result, query="...")
```

On its own the text path tends to answer conversationally (the shared LLM's captioning/QA "produce-text"
habit wins the first token at 500M), which is why the forced-call policy is the recommended path; the
speech path calls reliably since its data is exclusively call→summarize. Spoken proper nouns are bounded
by the Whisper encoder's ASR quality (~28% WER class).

## Repo layout

- `code/` — full training + inference code (model, MoE, projectors, **`agent.py`** forced-call policy, tool/BFCL harness, eval).
- `weights/joint12_model.pt` — **latest**: joint11 + **smoltalk** (chat), **google/mobile-actions** (mobile tool-calling) and **Aria-UI desktop grounding** (200k). Adds **UI grounding** — outputs a normalized click point `(cx, cy)` for an instruction like "Click the Sign in button" (the model now *points*, not just describes) — plus mobile actions + general chat. No regression. Same 8k / `patch16-512` architecture as joint11. (Partial run: best checkpoint at 1600 steps; the recurring DDP stall hit at ~step 2000.)
- `weights/joint11_model.pt` — joint10 lifted to **8k context** (rope_theta 1e6) + **high-res vision** (SigLIP2 `patch16-512`, 1024 tok). No regression; **sharper OCR** — accurate dates, preserved document structure, less hallucination vs joint10's 384px. **Load with the 8k config (rope_theta=1e6) + `siglip2-so400m-patch16-512` encoder** (the saved `config` carries rope_theta).
- `weights/joint10_model.pt` — joint9 + the **OCR** stream (Llama-Nemotron-VLM-v1 `ocr_4`, rendered-text→markdown). No regression; reads dense text (transcribes to markdown where joint9 only captioned). 384px / ctx2048.
- `weights/joint9_model.pt` — joint8 continued with streamed NVIDIA Nemotron-SFT-Agentic-v2 + Llama-Nemotron-VLM-v1 captioning (full, streamed). No regression; improved text tool-argument grounding. Use with `agent.py`.
- `weights/joint8_model.pt` — omni-agent with text must-call up-weighted (use with `agent.py`).
- `weights/joint7_model.pt` — **8-path omni-agent** (adds text-agent + speech-agent loops).
- `weights/joint6_model.pt` — 6-path unified: image / video / audio / speech / text / **tools**.
- `weights/joint5_model.pt` — 5-path unified (image / video / audio / speech / text).
- `weights/tools_v1_model.pt` — dedicated **function-caller** (BitAgent 59%).
- `weights/backbone_500M_ctx2048.pt` — the context-extended base LLM.
- `docs/` — architecture notes + the multimodal & tool-use build logs.

## Usage (sketch)

Inference needs the frozen encoders (pulled from HF) + this repo's code:

```python
# see code/multimodal.py (MoEVLM), code/speech.py, code/vision.py, code/audio.py
# load weights/joint5_model.pt into MoEVLM(llm, vision_dim, audio_dim, speech_dim),
# encode an input with the matching frozen encoder, splice at the sentinel token, generate.
```

The `code/modal_mm.py` harness has ready actions: `unified5` (all five paths), `image_voice`
(image + spoken question), `caption_speech`, `speech_wer`, etc.

## Limitations

- 500M scale → hallucinates on hard images and factual spoken-QA.
- POPE shows a strong object-presence "yes" bias (object-hallucination).
- ASR errors are homophone-ish (projector-only, no LLM finetune for ASR, no spelling rescore).
- Image+voice VQA works zero-shot but isn't yet a trained skill (spoken-VQA SFT would harden it).
