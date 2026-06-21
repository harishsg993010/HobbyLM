"""HobbyLM release pipeline: convert trained MoE checkpoints -> safetensors (per-model HF repos) + GGUF
(arch=hobbylm, one combined repo), with production READMEs. Runs on Modal (weights live on the volumes;
HF auth via the `huggingface` secret). Uploads to PRIVATE rootxhacker/HobbyLM-* repos.

  python -m modal run modal_hobbylm.py --action one --key chat      # smoke-test a single model
  python -m modal run modal_hobbylm.py --action all                 # all LLMs
  python -m modal run modal_hobbylm.py --action image               # HobbyLM-Image (safetensors only)
  python -m modal run modal_hobbylm.py --action gguf-readme         # push the combined GGUF repo README
"""
import modal

app = modal.App("hobbylm-release")

HF_USER = "rootxhacker"
GGUF_REPO = f"{HF_USER}/HobbyLM-gguf"
PRIVATE = False   # repos are public; flip to True to keep new repos private

_ASSETS = "hobby-chat/src-tauri/assets"
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch", "numpy", "safetensors", "gguf>=0.10.0", "huggingface_hub>=0.25.0")
       .add_local_file("export/to_gguf.py", "/root/moe-lab/to_gguf.py")
       .add_local_file(f"{_ASSETS}/vision_projector.safetensors", "/root/moe-lab/assets/vision_projector.safetensors")
       .add_local_file(f"{_ASSETS}/speech_projector.safetensors", "/root/moe-lab/assets/speech_projector.safetensors")
       .add_local_file(f"{_ASSETS}/melfilters.bytes", "/root/moe-lab/assets/melfilters.bytes")
       .add_local_file("flux4_scenes.png", "/root/moe-lab/assets/sample_scenes.png"))

runs_vol = modal.Volume.from_name("fineweb10B")
dream_vol = modal.Volume.from_name("dreamlite-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")

# key -> (checkpoint path on /data, repo suffix, human title, one-line summary, pipeline_tag)
LLM_MODELS = {
    "base":         ("/data/runs/500M_ctx8k/model.pt",          "HobbyLM-Base",
                     "HobbyLM-Base (500M sparse-MoE foundation LM)",
                     "A 500M-parameter sparse Mixture-of-Experts base language model pretrained on FineWeb.",
                     "text-generation"),
    "chat":         ("/data/runs/500M_chat_v2/model.pt",        "HobbyLM-Chat",
                     "HobbyLM-Chat (500M MoE, instruction-tuned)",
                     "Conversational/instruction-tuned variant of HobbyLM-Base (SmolTalk-style SFT).",
                     "text-generation"),
    "computer-use": ("/data/runs/500M_computer_use_v4/model.pt", "HobbyLM-Computer-Use",
                     "HobbyLM-Computer-Use (500M MoE, GUI agent / tool use)",
                     "Function-calling + accessibility-tree GUI-agent variant for computer-use tasks.",
                     "text-generation"),
    "omni":         ("/data/runs/500M_vlm_joint12/model.pt",    "HobbyLM-Omni",
                     "HobbyLM-Omni (500M MoE, text + image + audio)",
                     "Multimodal (omni) variant: a TinyLLaVA-style VLM over the HobbyLM MoE core, with vision and speech projectors.",
                     "image-text-to-text"),
    "diffusion":    ("/data/runs/500M_diff_chat/model.pt",      "HobbyLM-Diffusion",
                     "HobbyLM-Diffusion (500M MoE, instruction-tuned text diffusion / LLaDA-style)",
                     "Masked-diffusion (LLaDA-style) HobbyLM, chat-SFT'd on SmolTalk — bidirectional / parallel decoding.",
                     "text-generation"),
}

ARCH = """## Architecture

Every HobbyLM variant shares one core: a **sparse Mixture-of-Experts (MoE)** decoder in the modern
small-MoE style (DeepSeek-V3 / OLMoE lineage), where each design choice was picked by ablation rather
than by guesswork.

| Component | Value |
|---|---|
| Total parameters | ~500M (only a fraction is active per token) |
| Hidden size / layers | 768 / 16 (first FFN dense, the rest MoE) |
| Routed experts / active | 36 / top-6 (+ 1 always-on shared expert) |
| Attention | GQA, 12 query / 3 KV heads, decoupled head-dim 128, per-head QK-norm |
| Router | sigmoid gating, DeepSeek-V3 aux-loss-free load balancing, no top-k renorm |
| Positional | RoPE (θ up to 1e6 for the 8k-context checkpoints) |
| Tokenizer | GPT-2 byte-level BPE (50,304 vocab, sentinel-padded) |
| Optimizer | Muon on the 2-D + per-expert matrices, AdamW on everything else |

The full ablation log (QK-norm is the single biggest lever; aux-loss-free beats classic aux-loss;
≥32 experts and top-6 help; embedding-scaling hurt) lives in the project's architecture notes.
"""

# How the language-model numbers were produced — stated once, linked from every card.
METHOD = """> **How these were measured.** All language-model scores are **0-shot** through our own port of
> EleutherAI's `lm-evaluation-harness` (a custom `MoELMWrapper` that runs log-likelihood scoring over the
> HobbyLM MoE + GPT-2 tokenizer). Reference models in the comparison table were run through the **identical
> harness and task set**, so the numbers are apples-to-apples with ours — they are *not* copied from other
> model cards. We validated the harness against published cards (e.g. TinyLlama 52.75 vs card 52.99). These
> are small research models: read the numbers in context, not as leaderboard claims."""

# Per-model human-voice body + real benchmark tables (sourced from our own eval runs).
CARDS = {
    "base": {
        "intro": (
            "HobbyLM-Base is the foundation the whole family is built on: a 500M-parameter sparse "
            "Mixture-of-Experts decoder trained **from scratch** on FineWeb — no distillation, no borrowed "
            "weights. It exists to answer a simple question: how far can you get at the ~500M scale if you "
            "sweat the architecture and the training recipe instead of throwing tokens at the problem?"),
        "use": (
            "A pretrained base model for text completion, and the checkpoint you fine-tune for downstream "
            "tasks. It is **not** instruction-tuned — for chat, use [HobbyLM-Chat](https://huggingface.co/"
            f"{HF_USER}/HobbyLM-Chat)."),
        "bench": """## Benchmarks

0-shot, 7-task average through our harness (see note below). HobbyLM was trained on **40B tokens** — a tiny
budget next to the comparison models — so the right way to read this table is *per training token*.

| Model | Params | Pretrain tokens | Avg (7-task) |
|---|---|---|---|
| SmolLM2-360M | 360M | ~4T | 56.29 |
| Qwen3-0.6B | 600M | ~36T | 54.78 |
| gemma-3-270m | 270M | — | 48.09 |
| pythia-410m | 410M | 300B | 45.34 |
| **HobbyLM-Base (500M)** | **500M** | **40B** | **44.05** |
| opt-350m | 350M | 180B | 43.61 |
| HobbyLM-130M (sibling) | 130M | 10B | 42.97 |
| MicroLlama-300M | 300M | 50B | 42.23 |
| gpt2 | 124M | — | 40.62 |
| pythia-160m | 160M | 300B | 38.60 |

Per-task (0-shot): HellaSwag 41.5 · LAMBADA 40.0 · SciQ 70.3 · PIQA 69.6 · ARC-easy 42.7
(ARC-challenge / WinoGrande sit near chance, as expected at this scale). Validation loss: **3.03** at 1k
context, **2.94** after the 8k context-extension.

The ranking tracks **pretraining tokens**, not parameters: the top models see 50–900× more data than we do.
In the classic ≤300B-token regime, HobbyLM leads per token — the 130M (10B tokens) beats MicroLlama-300M
(50B), opt-350m (180B) and pythia-160m (300B). Token budget, not architecture, is the gap.

""" + METHOD,
        "train": (
            "Pretrained on ~40B unique FineWeb tokens (8×H100), then context-extended 1k→8k (RoPE θ 1e4→1e6). "
            "Muon on the hidden + per-expert matrices, AdamW on the router/embeddings/norms; fp32 router; "
            "chunked-checkpointed cross-entropy to fit a larger batch."),
        "limits": (
            "- It's a ~500M base model on a 40B-token budget: fluent and factually-okay on easy questions, "
            "but it hallucinates and can repeat without a repetition penalty at decode time.\n"
            "- Trained on English FineWeb; other languages and code are out of distribution.\n"
            "- Not aligned or safety-tuned."),
    },
    "chat": {
        "intro": (
            "HobbyLM-Chat is the instruction-tuned conversational model — HobbyLM-Base taken through SmolTalk "
            "supervised fine-tuning and a SmolLM2-style UltraFeedback DPO pass. The jump from base is large: "
            "it holds a coherent persona, follows instructions, and (with a repetition penalty) produces "
            "varied, flowing prose."),
        "use": (
            "General single- and multi-turn chat / instruction following. Prompt it with the trained "
            "`SYSTEM:` / `USER:` / `ASSISTANT:` turn format, and decode with a **repetition penalty ≈1.3** "
            "(this is what tames the small-model repetition tendency)."),
        "bench": """## Benchmarks

0-shot multiple-choice, our harness. Note that MC benchmarks measure *knowledge*, not *chat quality* — the
goal of this checkpoint is conversational fluency, which these tasks don't capture. The small dip vs the base
model is the usual **alignment tax**.

| Task | HobbyLM-Chat | HobbyLM-Base |
|---|---|---|
| ARC-challenge | 23.8 | 22.4 |
| ARC-easy | 42.2 | 42.8 |
| HellaSwag | 39.5 | 41.6 |
| PIQA | 67.1 | 69.5 |
| WinoGrande | 53.6 | 51.3 |
| OpenBookQA | 27.2 | 29.8 |
| BoolQ | 44.4 | 51.0 |
| **Average** | **42.5** | **44.0** |

Reasoning tasks (ARC, WinoGrande) held or improved; BoolQ dropped the most — chat phrasing fits the
log-likelihood format worse, not a capability loss. This is healthy for a ~500M chat model (SmolLM-360M range).

""" + METHOD,
        "train": (
            "SFT on ~1.5M chat trajectories (smol-smoltalk + the conversational smoltalk2 subsets), loss on "
            "assistant turns only; then UltraFeedback DPO (β=0.1) — the SmolLM2 recipe. SFT loss → ~1.50, DPO "
            "preference accuracy 0.50 → 0.64."),
        "limits": (
            "- Carries the 500M ceiling: factual hallucination, and weak adherence to strict output formats "
            "(e.g. exact syllable counts).\n"
            "- Use a repetition penalty at decode time; greedy decoding can loop.\n"
            "- Not safety-aligned — no RLHF safety tuning."),
    },
    "computer-use": {
        "intro": (
            "HobbyLM-Computer-Use is the agentic variant: function calling plus a **text-only GUI agent** that "
            "reads a serialized accessibility tree (no pixels, no screenshots) and emits a grounded UI action. "
            "It can also decompose a multi-step goal and drive it to completion, deciding when it's `finish`ed."),
        "use": (
            "Computer-use / GUI automation over a UI-Automation accessibility tree, and general tool / function "
            "calling. Serialize the screen as `SCREEN:\\n[ControlType] \"Name\" (state) …`, give it the 12-action "
            "schema, and it returns a grounded action as JSON. Powers the Computer panel in the hobby-chat app."),
        "bench": """## Benchmarks

Held-out evaluation of the v4 checkpoint (accessibility-tree grounding + multi-step planning). `param-hallucination`
is the rate of invented element names/arguments — strict tree-grounding in the data drives it to **0**.

| Split | JSON-parse | Name-F1 | Value-acc | Exact-match | Param-halluc |
|---|---|---|---|---|---|
| Planning (multi-step goals) | 96.5% | 94.7% | — | 82.6% | 0.0% |
| Grounding (real app trees) | ~96% | 95.5% | 91% | 78.4% | 0.0% |
| Grounding (synthetic screens) | 100% | 90.7% | 88.6% | 72.5% | 0.0% |

For general (non-GUI) function calling, the HobbyLM tool-use lineage scores **~24% average on BFCL v3**
(grammar-constrained) — strong relevance/abstention (relevance 77.8, beating the needle reference's 61.1),
weaker on parallel multi-call, which is the 500M ceiling. Exact-match understates real quality: many "misses"
are ambiguous numerics (e.g. *"give it a minute"* → `wait(60)` vs the reference `wait(7)`).

""" + METHOD,
        "train": (
            "Continue-SFT from the combined tool checkpoint on synthetic accessibility-tree data (Gemini-generated, "
            "strictly tree-validated) + real-app UI trees + planning trajectories, with a weighted loss. 13-action "
            "vocabulary (12 UI actions + `finish`)."),
        "limits": (
            "- Per-step grounding is ~80% accurate; on **long** goals those errors compound (short tasks usually "
            "complete, long ones can drift) and there is no per-step recovery.\n"
            "- Trained on trees capped at ~45 elements (2k-context era); very large raw UI trees should be filtered.\n"
            "- Near-identical controls (e.g. digit buttons) occasionally mis-ground."),
    },
    "omni": {
        "intro": (
            "HobbyLM-Omni is the multimodal core: **one** 500M MoE model that handles text, image, video, audio, "
            "and speech — plus tool use, OCR, and UI grounding — folded into a single checkpoint across 18 training "
            "paths (TinyLLaVA-style projectors over frozen SigLIP2 / Whisper / CLAP front-ends). The headline isn't "
            "any single score; it's the **breadth** in one small model."),
        "use": (
            "Vision-language and audio-language tasks: captioning, visual QA, OCR, sound/speech understanding, "
            "spoken-question answering, and tool calling. Image/audio/speech features are projected and spliced at "
            "the `[IMAGE]`/`[AUDIO]`/`[SPEECH]` sentinel tokens (ids 50257–50262)."),
        "bench": """## Benchmarks

Visual QA is scored with **containment** (the model is chat-trained and answers in full sentences, so strict
single-word exact-match badly under-scores it):

| Task | Score |
|---|---|
| VQAv2 (val) | 47.0 |
| GQA | 39.2 |
| POPE — accuracy / F1 | 50.0 / 66.7 |
| Tool calling — Needle (JSON-parse / Name-F1 / param-halluc) | 93.8 / 77.7 / 0.0 |
| BFCL (forced-call: simple / multiple) | 21.7 / 18.3 |
| Text — lm-eval 9-task avg | 0.432 |

POPE at 50/66.7 is a **real** ceiling — object-presence hallucination ("yes" to everything) is the known
small-VLM weakness, quantified. On function calling, Omni *can* call as well as the dedicated tool model
(forced-call simple 21.7 ≈ the specialist's 22.7); left to itself it prefers to abstain (irrelevance 86.7),
a safer agent failure mode. Speech does spoken-QA and commands rather than verbatim transcription.

""" + METHOD,
        "train": (
            "Built in stages on the context-extended (8k, θ 1e6) backbone with a 512px SigLIP2 vision tower: "
            "projector alignment → multimodal SFT → a joint 18-path co-training cycle (image / video / audio / "
            "speech / text / tools / OCR / UI-grounding) that keeps every modality from drifting."),
        "extra": (
            "\n## Multimodal use\n\nThis repo also ships the projector weights — `vision_projector.safetensors` "
            "(SigLIP2 → LLM) and `speech_projector.safetensors` (Whisper-mel → LLM), plus `melfilters.bytes`. "
            "The frozen front-ends encode the raw image/audio, the projectors map those features into the LLM "
            "embedding space, and they're spliced in at the modality sentinel tokens.\n"),
        "limits": (
            "- Breadth over depth: strong **in-distribution** (VQA, JSON tool calls with 0 hallucination, OCR, "
            "grounding) but below specialist sub-1B models on hard text reasoning (GSM8K, multi-hop QA).\n"
            "- Object-presence hallucination on POPE-style probes.\n"
            "- Verbose by default — ask for short answers explicitly, or score with containment, not exact-match."),
    },
    "diffusion": {
        "intro": (
            "HobbyLM-Diffusion is the family's experiment in a different decoding paradigm: a **masked-diffusion** "
            "language model (LLaDA-style). Instead of generating left-to-right, it attends bidirectionally and fills "
            "in `[MASK]` tokens over a few iterative denoising passes — so it can decode in parallel. This checkpoint "
            "is **instruction-tuned**: the diffusion base was chat-SFT'd on SmolTalk with a LLaDA-style objective "
            "(mask only the assistant response, denoise it conditioned on the clean prompt)."),
        "use": (
            "**Experimental** conversational generation via iterative denoising — it's a research artifact, not a "
            "reliable assistant. Prompt it with the trained `USER:` / `ASSISTANT:` turn format. It adopts the chat "
            "register and the question→answer shape, but at 500M with a pure-diffusion objective it hallucinates and "
            "follows instructions loosely. Decode knobs trade quality vs speed; good defaults: temp 0–0.3, steps ≈ 2× "
            "the generation length, repetition penalty 1.4–1.5."),
        "bench": """## Benchmarks

A masked-diffusion model can't be scored by the standard log-likelihood lm-eval harness, so the meaningful
numbers are training loss and **decoding throughput** — where the diffusion paradigm actually shows up:

| Metric | Value |
|---|---|
| Validation loss (≈21B tokens) | 3.52 |
| Throughput — H100, 128 tok, 32 steps | **117.7 tok/s** (~2.7× the AR model) |
| Throughput — H100, AR baseline | ~44 tok/s |
| Throughput — laptop CPU (q8, cached) | ~6.5 tok/s |

The throughput result reproduces the **Fast-dLLM** literature's 2–3× GPU range from a from-scratch
implementation: on memory-bound hardware (GPU) batching the whole canvas is nearly free, so fewer denoising
passes than tokens wins; on a compute-bound laptop the same code trails the AR engine. The knob is
steps-per-token (quality ↔ speed).

> A masked-diffusion LM at 500M trails an equal-scale autoregressive model on raw coherence — the method is
> fully validated end-to-end here; the limit is capacity and tokens, not the recipe.""",
        "train": (
            "Two stages. **Base:** converted from the autoregressive 500M base (weights transfer; same architecture, "
            "attention switched to bidirectional) and adapted on ~21B tokens with a masked-token objective reweighted "
            "by 1/p_mask (a DiffuGPT/DiffuLLaMA-style conversion, val loss 3.52). **Instruction tuning:** chat-SFT on "
            "SmolTalk trajectories — each assistant response is masked and denoised conditioned on the clean prompt."),
        "extra": (
            "\n## Decoding\n\nGeneration is **iterative bidirectional denoising** of `[MASK]` tokens, not "
            "left-to-right AR. The GGUF carries `diffusion.*` metadata (mask-token id, block size) for a "
            "diffusion-aware runtime; `hobby-rs` implements the cached semi-autoregressive denoiser.\n"),
        "limits": (
            "- **Hallucinates and follows instructions loosely** — the SFT shifts it into a conversational register "
            "and the Q→A shape, but it does not reliably produce correct or on-task answers. This is the expected "
            "ceiling for a 500M *pure-diffusion* model; the limit is capacity, not the recipe.\n"
            "- Decoding quality is very sensitive to the sampler settings (see above).\n"
            "- The CPU throughput win only materializes on memory-bound hardware; on a thermally-limited laptop "
            "the AR model is faster."),
    },
}


# Example prompt + an optional usage note per model (used by the runnable code block).
USAGE = {
    "base":         ("The capital of France is", ""),
    "chat":         ("USER: Give me three tips for better sleep.\\nASSISTANT:",
                     "Prompt it with the trained `USER:` / `ASSISTANT:` turn format (a leading "
                     "`SYSTEM:` turn is optional). A repetition penalty around **1.3** is recommended."),
    "computer-use": ("USER: What is 7 plus 2?\\nASSISTANT:",
                     "For GUI / tool use, the real prompt format is `TOOLS: [<schema>]\\nSCREEN:\\n"
                     "[ControlType] \"Name\" (state) …\\nUSER: <instruction>\\nASSISTANT:` and the model "
                     "replies with a JSON action. The end-to-end agent loop lives in `agents/` in the repo."),
    "omni":         ("USER: Explain a mixture-of-experts model in one sentence.\\nASSISTANT:",
                     "The snippet above is the **text** path. For image / audio / speech, encode the input "
                     "with the (frozen) SigLIP2 / Whisper / CLAP front-end, project it with the bundled "
                     "projectors, and splice it at the modality sentinel token — see `hobbylm/multimodal.py`, "
                     "or just pass `--image` / `--speech` to `hobby-rs`."),
    "diffusion":    ("The meaning of life is", ""),
}


def _usage_py(repo, key, prompt):
    if key == "diffusion":
        return f"""```python
# HobbyLM-Diffusion is a MASKED-DIFFUSION model: generation is iterative, bidirectional denoising
# — NOT autoregressive — so it uses the reference diffusion sampler (not transformers.generate).
# pip install torch safetensors tiktoken huggingface_hub
# git clone https://github.com/harishsg993010/HobbyLM && cd HobbyLM

import json, torch, tiktoken
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from hobbylm.config import ModelConfig
from hobbylm.model import MoETransformer
from hobbylm.diffusion import generate

repo = "{repo}"
cfg = ModelConfig(**{{k: v for k, v in json.load(open(hf_hub_download(repo, "config.json"))).items() if k != "preset"}})
cfg.expert_backend = "bmm"                          # "grouped" on CUDA
model = MoETransformer(cfg).eval()
model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))

enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode_ordinary("{prompt}")])
# iterative denoising: gen_len tokens over `steps` bidirectional passes (more steps + lower temp = better)
out = generate(model, ids, gen_len=96, steps=128, temperature=0.2, rep_penalty=1.5, remask_steps=2)
print(enc.decode(out[0].tolist()))
```"""
    return f"""```python
# HobbyLM is a CUSTOM sparse-MoE architecture, so load it with the reference implementation —
# NOT transformers.AutoModelForCausalLM (there is no AutoModel mapping for this arch).
# pip install torch safetensors tiktoken huggingface_hub
# git clone https://github.com/harishsg993010/HobbyLM && cd HobbyLM

import json, torch, tiktoken
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from hobbylm.config import ModelConfig
from hobbylm.model import MoETransformer
from hobbylm.generate import generate

repo = "{repo}"
cfg = ModelConfig(**{{k: v for k, v in json.load(open(hf_hub_download(repo, "config.json"))).items() if k != "preset"}})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.expert_backend = "grouped" if device.type == "cuda" else "bmm"

model = MoETransformer(cfg).to(device).eval()
model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))

enc = tiktoken.get_encoding("gpt2")
prompt = "{prompt}"
ids = torch.tensor([enc.encode_ordinary(prompt)], device=device)
out = generate(model, ids, max_new_tokens=64, temperature=0.7, top_k=0, device=device,
               repetition_penalty=1.3)               # temperature=0.0 for greedy
print(enc.decode(out[0].tolist()))
```"""


def _readme(title, summary, pipeline_tag, key):
    c = CARDS[key]
    extra = c.get("extra", "")
    gguf_name = title.split()[0]                       # e.g. "HobbyLM-Base"
    repo = f"{HF_USER}/{gguf_name}"
    prompt, note = USAGE[key]
    usage_code = _usage_py(repo, key, prompt)
    usage_note = f"\n> {note}\n" if note else ""
    return f"""---
license: apache-2.0
language: [en]
library_name: safetensors
pipeline_tag: {pipeline_tag}
tags: [hobbylm, mixture-of-experts, moe, sparse-moe]
---

# {title}

{c['intro']}

It's part of the **HobbyLM** family — a 500M sparse-MoE model (and its variants) built from scratch on a
hobby budget: FineWeb, a handful of Modal H100 hours, a lot of ablations, and a from-scratch Rust engine
([`hobby-rs`](https://github.com/harishsg993010/HobbyLM)) to run it on a laptop CPU.

## Intended use

{c['use']}

{ARCH}{extra}
{c['bench']}

## Usage

### Python (PyTorch reference implementation)

HobbyLM is a custom sparse-MoE architecture — there's no `transformers` `AutoModel` for it, so load it with
the small reference implementation from the [GitHub repo](https://github.com/harishsg993010/HobbyLM):

{usage_code}
{usage_note}
### GGUF + hobby-rs (CPU)

GGUF builds (architecture `hobbylm`) live in [`{GGUF_REPO}`](https://huggingface.co/{GGUF_REPO}). They load
directly in the from-scratch `hobby-rs` CPU engine — **stock llama.cpp won't load them** without registering
the `hobbylm` architecture first.

```bash
hobby-rs --model {gguf_name}.gguf --prompt "..." --n 64
```

## Training

{c['train']}

## Limitations

{c['limits']}

## License

Apache-2.0. Weights aren't a substitute for judgement — this is a research / hobby model at the 500M scale,
not a production system.
"""


def _save_safetensors(sd, path):
    import torch
    from safetensors.torch import save_file
    clean = {k: v.detach().to(torch.float32).cpu().contiguous().clone()
             for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(clean, path)


@app.function(image=img, volumes={"/data": runs_vol}, secrets=[HF_SECRET], timeout=60 * 60, memory=24000)
def export_llm(key: str):
    import os, sys, json, subprocess, torch
    from huggingface_hub import HfApi
    sys.path.insert(0, "/root/moe-lab"); os.chdir("/root/moe-lab")
    ckpt, suffix, title, summary, ptag = LLM_MODELS[key]
    repo = f"{HF_USER}/{suffix}"
    api = HfApi()
    print(f"[{key}] loading {ckpt}", flush=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd, cfg = ck["model"], ck["config"]

    os.makedirs("/tmp/out", exist_ok=True)
    _save_safetensors(sd, "/tmp/out/model.safetensors")
    json.dump(cfg, open("/tmp/out/config.json", "w"), indent=2, default=str)
    open("/tmp/out/README.md", "w").write(_readme(title, summary, ptag, key))

    # ---- GGUF (arch=hobbylm) ----
    gguf_name = f"{suffix}.gguf"
    subprocess.run([sys.executable, "to_gguf.py", "--ckpt", ckpt, "--out", f"/tmp/out/{gguf_name}",
                    "--arch", "hobbylm"], check=True)

    # ---- per-model safetensors repo ----
    api.create_repo(repo, private=PRIVATE, exist_ok=True, repo_type="model")
    for f in ["model.safetensors", "config.json", "README.md"]:
        api.upload_file(path_or_fileobj=f"/tmp/out/{f}", path_in_repo=f, repo_id=repo, repo_type="model")
    # ---- omni: ship projectors too ----
    if key == "omni":
        for pf in ["assets/vision_projector.safetensors", "assets/speech_projector.safetensors", "assets/melfilters.bytes"]:
            if os.path.exists(pf):
                api.upload_file(path_or_fileobj=pf, path_in_repo=os.path.basename(pf), repo_id=repo, repo_type="model")

    # ---- combined GGUF repo ----
    api.create_repo(GGUF_REPO, private=PRIVATE, exist_ok=True, repo_type="model")
    api.upload_file(path_or_fileobj=f"/tmp/out/{gguf_name}", path_in_repo=gguf_name, repo_id=GGUF_REPO, repo_type="model")
    sz = os.path.getsize(f"/tmp/out/{gguf_name}") / 1e9
    print(f"[{key}] DONE -> {repo} (safetensors) + {GGUF_REPO}/{gguf_name} ({sz:.2f} GB)", flush=True)
    return {"key": key, "repo": repo, "gguf": gguf_name}


@app.function(image=img, volumes={"/cache": dream_vol}, secrets=[HF_SECRET], timeout=30 * 60, memory=16000)
def export_image():
    import os, json, torch
    from huggingface_hub import HfApi
    repo = f"{HF_USER}/HobbyLM-Image"
    api = HfApi()
    ck = torch.load("/cache/model_1024flux4.pt", map_location="cpu", weights_only=False)
    sd, cfgd = ck["sd"], ck["cfg_dict"]
    os.makedirs("/tmp/img", exist_ok=True)
    _save_safetensors(sd, "/tmp/img/model.safetensors")
    json.dump({"dit_config": cfgd, "lat_std": float(ck["lat_std"]), "scaling_factor": float(ck["sf"]),
               "vae": "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
               "text_encoder": "openai/clip-vit-large-patch14", "resolution": 1024,
               "steps_trained": int(ck.get("steps", 0))}, open("/tmp/img/config.json", "w"), indent=2)
    open("/tmp/img/README.md", "w").write(f"""---
license: apache-2.0
pipeline_tag: text-to-image
library_name: safetensors
tags: [hobbylm, text-to-image, diffusion, dit, flow-matching]
---

# HobbyLM-Image — 1024px text-to-image DiT

The odd one out in the HobbyLM family: not a language model, but a **333M in-context flow-matching DiT** that
generates 1024×1024 images. It was built to see how good a text-to-image model you can train on a genuinely
small budget — the whole thing came together for roughly **$300 of Modal GPU time** by working in a heavily
compressed latent space instead of pixels.

It runs in the **DC-AE f32c32 (SANA-1.1)** latent (32× spatial compression → a 32×32×32 latent at 1024px) and
is conditioned on **CLIP-L** text features, with classifier-free guidance.

## Intended use

Text-to-image generation at 1024×1024. Strongest on single objects and cinematic scenes. A sibling 512px
checkpoint additionally does instruction-based image editing.

## How it works

```
CLIP-L(prompt) ─┐
                ├─►  DiT  ──(rectified-flow / CFG sampler, ~100 steps)──►  latent  ──►  DC-AE decode  ──►  1024² image
 Gaussian noise ─┘     (this repo)                                                       (frozen VAE)
```

The two frozen components are **not** included (download them from their own repos):
`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` (VAE) and `openai/clip-vit-large-patch14` (text encoder).
A full from-scratch CPU implementation of this pipeline (CLIP + DiT + DC-AE, in Rust) lives in
[`hobby-rs`](https://github.com/harishsg993010/HobbyLM).

## Samples

1024×1024, generated by this model (CFG ≈ 5, ~100 steps):

![HobbyLM-Image scene samples](sample_scenes.png)

## Results

This is a hobby-scale generator, so the honest "benchmark" is the training curve and qualitative behaviour
rather than FID / GenEval (which we did not compute):

| Property | Value |
|---|---|
| Flow-matching loss (final) | **0.76** (lowest of the model lineage — still decreasing) |
| Parameters | 333M (DiT only) |
| Resolution | 1024×1024 (32×32×32 latent) |
| VAE reconstruction | ~26 dB PSNR @512px; sharper at 1024px (32×32 latent) |

Qualitatively, the final checkpoint produces accurate objects and cinematic scenes. It is **soft on people,
hands, and multi-person scenes** — the real small-model / latent-resolution ceiling. Loss was still dropping
at the end of training, so the 333M DiT is not yet saturated.

## Files

- `model.safetensors` — the DiT weights.
- `config.json` — DiT config, `lat_std`, and the VAE `scaling_factor`.

There is no GGUF build: image-generation DiTs have no standard GGUF runtime.

## Limitations

- Hands and multi-person scenes are unreliable.
- Fine object crispness is capped by the 32× DC-AE latent; a less-compressed VAE would sharpen it at higher cost.
- Instruction-based **editing** is limited (the CLIP-L text encoder is a weak instruction follower); the real
  fix is a stronger conditioner, which is future work.

## License

Apache-2.0.
""")
    # sample showcase image referenced by the README
    sample_src = "/root/moe-lab/assets/sample_scenes.png"
    if os.path.exists(sample_src):
        import shutil
        shutil.copy(sample_src, "/tmp/img/sample_scenes.png")

    api.create_repo(repo, private=PRIVATE, exist_ok=True, repo_type="model")
    files = ["model.safetensors", "config.json", "README.md"]
    if os.path.exists("/tmp/img/sample_scenes.png"):
        files.append("sample_scenes.png")
    for f in files:
        api.upload_file(path_or_fileobj=f"/tmp/img/{f}", path_in_repo=f, repo_id=repo, repo_type="model")
    print(f"image DONE -> {repo}", flush=True)
    return {"repo": repo}


@app.function(image=img, secrets=[HF_SECRET], timeout=10 * 60)
def push_gguf_readme():
    from huggingface_hub import HfApi
    api = HfApi()
    body = f"""---
license: apache-2.0
tags: [hobbylm, gguf, mixture-of-experts, moe]
---

# HobbyLM-GGUF

GGUF builds of every **HobbyLM** language model — one file per variant, all sharing the same 500M sparse-MoE
core. These are the files you actually run on a laptop CPU.

| File | Model | What it's for | Headline number |
|---|---|---|---|
| `HobbyLM-Base.gguf` | [Base](https://huggingface.co/{HF_USER}/HobbyLM-Base) | pretrained foundation LM | 44.05 avg (0-shot, our harness) |
| `HobbyLM-Chat.gguf` | [Chat](https://huggingface.co/{HF_USER}/HobbyLM-Chat) | instruction / chat | 42.5 avg (alignment-tax dip from base) |
| `HobbyLM-Computer-Use.gguf` | [Computer-Use](https://huggingface.co/{HF_USER}/HobbyLM-Computer-Use) | GUI agent + tool calling | 95% name-F1, 0% param-hallucination |
| `HobbyLM-Omni.gguf` | [Omni](https://huggingface.co/{HF_USER}/HobbyLM-Omni) | multimodal core (text+image+audio) | VQAv2 47.0 / GQA 39.2 |
| `HobbyLM-Diffusion.gguf` | [Diffusion](https://huggingface.co/{HF_USER}/HobbyLM-Diffusion) | masked-diffusion LM | 117 tok/s on H100 (~2.7× AR) |

Full benchmark tables, methodology, and limitations are on each model's own card (linked above).

## Running them

```bash
# from https://github.com/harishsg993010/HobbyLM
hobby-rs --model HobbyLM-Chat.gguf --prompt "The capital of France is" --n 48
```

## ⚠️ These use a custom `hobbylm` architecture

Every GGUF sets `general.architecture = hobbylm` (all metadata keys are `hobbylm.*`). **Stock llama.cpp will
not load them** — they need the from-scratch [`hobby-rs`](https://github.com/harishsg993010/HobbyLM) engine,
or a llama.cpp patched to register the `hobbylm` arch (GQA + per-head QK-norm + sigmoid-gated MoE + aux-free
routing bias + 1 shared expert + a leading dense layer). `HobbyLM-Diffusion` additionally carries `diffusion.*`
metadata and needs the diffusion-aware (bidirectional, iterative-denoise) decoder.

## License
Apache-2.0.
"""
    api.create_repo(GGUF_REPO, private=PRIVATE, exist_ok=True, repo_type="model")
    open("/tmp/gr.md", "w").write(body)
    api.upload_file(path_or_fileobj="/tmp/gr.md", path_in_repo="README.md", repo_id=GGUF_REPO, repo_type="model")
    print(f"pushed README -> {GGUF_REPO}", flush=True)


@app.function(image=img, secrets=[HF_SECRET], timeout=10 * 60)
def push_llm_readme(key: str):
    """Re-render and upload ONLY the README.md for one LLM repo (no weights / GGUF re-upload)."""
    from huggingface_hub import HfApi
    _, suffix, title, summary, ptag = LLM_MODELS[key]
    repo = f"{HF_USER}/{suffix}"
    api = HfApi()
    api.create_repo(repo, private=PRIVATE, exist_ok=True, repo_type="model")
    open("/tmp/r.md", "w").write(_readme(title, summary, ptag, key))
    api.upload_file(path_or_fileobj="/tmp/r.md", path_in_repo="README.md", repo_id=repo, repo_type="model")
    print(f"pushed README -> {repo}", flush=True)
    return {"key": key, "repo": repo}


ALL_REPOS = [f"{HF_USER}/{m[1]}" for m in LLM_MODELS.values()] + [f"{HF_USER}/HobbyLM-Image", GGUF_REPO]


@app.function(image=img, secrets=[HF_SECRET], timeout=10 * 60)
def set_visibility(private: bool):
    """Flip every HobbyLM repo public (private=False) or private (private=True)."""
    from huggingface_hub import HfApi
    api = HfApi()
    for r in ALL_REPOS:
        try:
            try:
                api.update_repo_settings(repo_id=r, private=private, repo_type="model")
            except (AttributeError, TypeError):
                api.update_repo_visibility(repo_id=r, private=private, repo_type="model")
            print(f"{'PRIVATE' if private else 'PUBLIC '} -> {r}", flush=True)
        except Exception as e:
            print(f"{r} FAILED: {e}", flush=True)


@app.local_entrypoint()
def main(action: str = "one", key: str = "chat"):
    if action == "make-public":
        set_visibility.remote(False)
        return
    if action == "make-private":
        set_visibility.remote(True)
        return
    if action == "one":
        print(export_llm.remote(key))
    elif action == "all":
        handles = {k: export_llm.spawn(k) for k in LLM_MODELS}      # all LLMs in parallel
        handles["image"] = export_image.spawn()                     # + the image model
        for k, h in handles.items():
            try:
                print(h.get())
            except Exception as e:
                print(f"{k} FAILED: {e}")
        print(push_gguf_readme.remote())
    elif action == "readme":
        # README-only refresh: per-model cards (no weight re-upload) + image card + GGUF card.
        handles = {k: push_llm_readme.spawn(k) for k in LLM_MODELS}
        handles["image"] = export_image.spawn()                     # image: re-runs to refresh README + sample
        for k, h in handles.items():
            try:
                print(h.get())
            except Exception as e:
                print(f"{k} FAILED: {e}")
        print(push_gguf_readme.remote())
    elif action == "readme-llm":
        print(push_llm_readme.remote(key))
    elif action == "image":
        print(export_image.remote())
    elif action == "gguf-readme":
        print(push_gguf_readme.remote())
