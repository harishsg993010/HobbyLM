---
title: HobbyLM Playground
emoji: 🪶
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Chat, see & generate with the 500M HobbyLM MoE family
models:
  - rootxhacker/HobbyLM-Base
  - rootxhacker/HobbyLM-Chat
  - rootxhacker/HobbyLM-Computer-Use
  - rootxhacker/HobbyLM-Omni
  - rootxhacker/HobbyLM-Diffusion
  - rootxhacker/HobbyLM-Image
---

# 🪶 HobbyLM Playground

An interactive demo of **HobbyLM** — a from-scratch **500M sparse Mixture-of-Experts** language-model family
(plus a 333M text-to-image DiT), all trained on a hobby budget. One Space, three things to try:

- **💬 Chat** — talk to any variant: Base, Chat, Computer-Use, the multimodal Omni core, or the
  masked-diffusion model (which decodes by iterative denoising, not left-to-right).
- **🖼️ Ask about an image** — upload a picture and question the multimodal **Omni** model (SigLIP2 vision
  encoder → MoE LLM).
- **🎨 Generate an image** — text-to-image at 1024px with **HobbyLM-Image** (a flow-matching DiT in the
  DC-AE latent space, conditioned on CLIP-L).

The models use a custom `hobbylm` architecture, so this Space vendors the small reference implementation
(`hobbylm/`, `hobby_image/`) rather than going through `transformers.AutoModel`.

## Hardware

This Space is written for **ZeroGPU** (the heavy functions are decorated with `@spaces.GPU`). Enable
*ZeroGPU* in the Space's hardware settings for fast chat, image understanding, and 1024px generation. It
also runs on CPU (chat is slow; image generation is impractical there).

## Links

- Models: <https://huggingface.co/rootxhacker>
- Code + the from-scratch Rust CPU engine: <https://github.com/harishsg993010/HobbyLM>

These are tiny research models — genuinely fluent and fun, but with the capability ceiling of a 500M model
(hallucination, weak strict-format following, soft hands / multi-person in image generation). Apache-2.0.
