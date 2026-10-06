# 30M HobbyLM First-Run Spec

## Purpose

This document captures what we did to get a first small LLM training run working with the HobbyLM codebase, Hugging Face for dataset access, and Modal for GPU execution.

It is meant to be:

- a study note
- a handoff note
- a reproducible first-run spec

This was a **first proof run**, not a full Chinchilla-optimal training run.

---

## High-Level Summary

We used the HobbyLM repository as the training codebase, configured Hugging Face credentials so the code could download tokenized training data, configured Modal credentials and secrets so the code could run on an H100 GPU, added a small `30M` model preset for learning, downloaded a small portion of FineWeb data, and ran a short real pretraining job for 20 steps.

That means the project is already past setup. A real small model was trained and checkpoints were saved.

---

## What Each System Does

### GitHub Repository

The repository provides:

- model architecture
- MoE implementation
- training loop
- Modal orchestration code
- generation and evaluation helpers

Important files:

- [hobbylm/config.py](/private/tmp/HobbyLM/hobbylm/config.py:1)
- [hobbylm/model.py](/private/tmp/HobbyLM/hobbylm/model.py:1)
- [hobbylm/moe.py](/private/tmp/HobbyLM/hobbylm/moe.py:1)
- [training/train.py](/private/tmp/HobbyLM/training/train.py:1)
- [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:1)

### Hugging Face

Hugging Face was used for:

- storing the dataset
- authenticating dataset downloads

For this run, the dataset came from:

- `kjj0/fineweb10B-gpt2`

Declared in:

- [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:18)

### Modal

Modal was used for:

- cloud GPU execution
- persistent volumes for data and checkpoints
- secret injection for Hugging Face token access

In simple words:

- GitHub = code
- Hugging Face = data
- Modal = GPU machine

---

## Important Clarification About The Dataset

The dataset is **not stored inside the repo**.

The repo only contains code that knows how to download and use the dataset.

The actual training data was downloaded from Hugging Face by the `download` function in:

- [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:49)

The dataset files are tokenized binary shards such as:

- `fineweb_train_000001.bin`
- `fineweb_val_000000.bin`

These are stored in the Modal volume path:

- `/data/fineweb10B`

---

## Credential Setup

### Hugging Face Token

We created a Hugging Face token so Modal could access the dataset.

Conceptually:

- create token on Hugging Face
- give it read access suitable for dataset download
- store it in a Modal secret

The Modal secret name used was:

- `huggingface`

The environment variable stored inside that secret was:

- `HF_TOKEN`

### Modal Auth

We also configured local Modal authentication on the machine so CLI commands like `python -m modal run ...` could launch remote jobs.

Important security note:

- if a Modal token or secret was exposed in a local file or chat, it should be revoked and regenerated after the setup session

---

## What “30M Preset” Means

The `30M` preset is a custom small model configuration added for learning.

It lives in:

- [hobbylm/config.py](/private/tmp/HobbyLM/hobbylm/config.py:119)

Configuration:

- `d_model = 256`
- `n_layers = 13`
- `n_dense_layers = 1`
- `n_q_heads = 4`
- `n_kv_heads = 1`
- `head_dim = 64`
- `dense_ffn = 1024`
- `expert_ffn = 128`
- `n_experts = 12`
- `top_k = 2`
- `n_shared = 0`

Approximate size:

- total params: `29,995,392`
- active params: `18,198,912`

Important meaning:

- `30M` refers to **model size**
- it does **not** refer to dataset size

---

## What MoE Means Here

MoE means **Mixture of Experts**.

Instead of every token using one single large FFN in every block, an MoE block contains multiple smaller experts and a router.

For each token:

1. the router scores all experts
2. the top experts are selected
3. only those selected experts process the token
4. their outputs are combined

In this 30M preset:

- there are `12` experts in each MoE layer
- each token uses `top_k = 2` experts

This is implemented in:

- [hobbylm/moe.py](/private/tmp/HobbyLM/hobbylm/moe.py:47)

Simple analogy:

- dense model: every student goes to every teacher
- MoE model: each student is sent only to the best 2 teachers for that question

Why that matters:

- total params can be larger
- compute per token stays smaller than using everything every time

That is why the model has:

- about `30M` total parameters
- but only about `18.2M` active parameters per token

---

## What Architecture We Trained

The main model shell is in:

- [hobbylm/model.py](/private/tmp/HobbyLM/hobbylm/model.py:261)

The structure is:

1. token IDs go into `embed`
2. embeddings go through repeated transformer blocks
3. final normalization runs
4. `lm_head` produces next-token logits

The repeated transformer stack is the “looped architecture” idea.

For the 30M preset:

- total blocks: `13`
- first `1` block uses a dense FFN
- remaining `12` blocks use MoE

---

## Current Status

Right now, the project has completed a real **base pretraining starter path**.

What that means in simple words:

- the 30M model architecture exists
- the FineWeb token data was downloaded into Modal storage
- the model was trained on next-token prediction
- checkpoints were saved
- generation was tested from the saved checkpoint

Important reality:

- this is a **base model**
- it is learning language patterns
- it is **not yet** a polished chat assistant

That is why generations can look partially structured but still wrong.

Example intuition:

- pretraining teaches the model how text is shaped
- chat fine-tuning teaches the model how to answer in an assistant style

---

## What Success Looked Like So Far

### Step 1: Credentials Setup

Success looked like:

- Hugging Face token exists
- Modal secret exists
- Modal local auth works

Why it mattered:

- without this, the code cannot download data or start remote GPU jobs

### Step 2: GPU Smoke Test

Success looked like:

- the Modal job starts
- the H100 container boots
- loss prints are finite
- the log ends with `GPU SMOKE OK`

Why it mattered:

- this proves the GPU path works before spending money on real training

### Step 3: Data Download

Success looked like:

- FineWeb shards appear in the Modal volume
- the download log confirms train and val shards are ready

Why it mattered:

- the repo does not ship the dataset inside GitHub
- it downloads tokenized training data from Hugging Face into Modal storage

### Step 4: Short Training Run

Success looked like:

- loss starts high and then drops
- checkpoints get saved
- `model.pt` exists at the run path

Why it mattered:

- this proves the model is really learning, not just running

### Step 5: Longer Training Run

Success looked like:

- validation loss drops clearly below the short-run result
- later checkpoints save correctly
- generation can load the stronger checkpoint

Why it mattered:

- this moved the project from “setup proof” to “real small pretraining run”

### Step 6: Inference Test

Success looked like:

- a prompt can be sent into the saved model
- the model returns text from the checkpoint

Why it mattered:

- this closes the loop from code -> data -> GPU training -> saved weights -> usable inference

---

## What We Are Building Next

The next stage is **chat fine-tuning** on top of the pretrained 30M checkpoint.

Simple meaning:

- pretraining made the model a raw text predictor
- chat fine-tuning will push it toward assistant-style answers

We are reusing the repo’s existing chat-SFT pipeline in:

- [training/modal_tools.py](/private/tmp/HobbyLM/training/modal_tools.py:1)
- [training/train_tools.py](/private/tmp/HobbyLM/training/train_tools.py:1)
- [hobbylm/tool_data.py](/private/tmp/HobbyLM/hobbylm/tool_data.py:1)

The chat data comes from Hugging Face conversational datasets and is converted into:

- `USER: ...`
- `ASSISTANT: ...`

training segments.

---

## What Success Looks Like In The Next Stage

### Chat Data Prep

Success looks like:

- `chat_train.jsonl` and `chat_val.jsonl` are written to the Modal volume
- the prep log shows many usable conversations were kept

Meaning:

- we now have assistant-style supervised examples

### Chat SFT Training

Success looks like:

- the 30M pretrained checkpoint loads correctly
- chat training loss drops over steps
- a new run folder saves chat-tuned checkpoints

Meaning:

- the model is no longer only a raw base model
- it is starting to learn the shape of assistant responses

### Chat Test

Success looks like:

- prompts answered after chat-SFT look more like direct assistant replies
- fewer outputs are random or broken compared with the pure pretrained model

Meaning:

- we now have an early “assistant-like” model, even if still small and limited

---

## New Helper Command Path

To make the next stage easier, the Modal tool entrypoint was wired so it can now:

- prepare chat data with custom `max_len`
- train from a custom backbone run
- run a dedicated `train_chat` action for the 30M path

That means the intended 30M chat-SFT flow is now:

1. prepare chat data
2. initialize from `30M_study_1000`
3. train a chat-SFT run
4. test generations again

---

## Why A Model Can Say "France = Canada" Even When Training Is Working

This is one of the most important things to understand.

If the model answers:

- `The capital of France is Canada`

that does **not automatically mean** training is broken.

It usually means one or more of these things are true:

### 1. The Model Is Still Early In Training

Our 30M run is still small compared with the full rough Chinchilla-style target.

Simple meaning:

- the model has started learning language patterns
- but it has not seen enough tokens yet to stabilize many facts

So training can be working while factual recall is still weak.

### 2. Pretraining Learns Patterns, Not Truth Checking

The pretraining objective is:

- predict the next token

It is **not**:

- verify facts
- consult a database
- check a trusted encyclopedia

So the model learns what text sequences are likely, not what statements are guaranteed true.

### 3. The Dataset Is Real Web Data, Not Hand-Checked Knowledge

The FineWeb dataset is large-scale web text.

That means it contains a mixture of:

- correct facts
- repeated facts
- noisy text
- low-quality text
- outdated text
- random internet junk

So when your friend asked about the “genuineness” of the data, that is a valid concern.

Simple meaning:

- the data is useful for language learning
- but it is not perfectly fact-clean

### 4. Small Models Have Limited World Compression

A 30M model is very small compared with mainstream LLMs.

That means:

- it has less capacity
- it stores less stable factual structure
- it is more likely to confuse nearby patterns

So even if the training loop is healthy, a 30M base model can still answer basic facts incorrectly.

### 5. We Tested A Base Model, Not A Fully Tuned Assistant

The generations we checked came from a pretrained checkpoint.

That means:

- it learned raw language continuation
- it was not yet optimized to answer like a polished assistant

So it may continue text in a fluent-looking but incorrect way.

### Simple Summary

“France = Canada” can happen even when:

- loss is dropping
- checkpoints are saving
- training is real

because:

- the model is still undertrained
- the model is small
- the data is noisy web text
- pretraining predicts text, not truth

---

## How We Downloaded The Data

The dataset was **not** stored inside the GitHub repo.

The repo only contains the code that knows how to fetch it.

The actual download path was:

1. we ran the Modal entrypoint in [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:1)
2. that called the `download(...)` function
3. inside that function, `hf_hub_download(...)` pulled files from Hugging Face
4. those files were saved into the Modal volume under `/data/fineweb10B`

For this project:

- Hugging Face repo: `kjj0/fineweb10B-gpt2`
- training file names look like: `fineweb_train_000001.bin`
- validation file name looks like: `fineweb_val_000000.bin`

Simple system view:

- GitHub repo = code for training
- Hugging Face = where the tokenized dataset lives
- Modal volume = where downloaded data stays for remote jobs

### What We Downloaded

For the first pretraining run we downloaded:

- `1` train chunk
- `1` validation shard

In this setup, that means roughly:

- `~100M` train tokens available
- plus validation data

### Why We Downloaded It This Way

We used Modal download instead of local download because:

- training also runs on Modal
- Modal volumes keep data close to the GPU jobs
- future runs can reuse the same stored shards without redownloading

Each block does:

1. RMSNorm
2. attention
3. residual add
4. RMSNorm
5. dense FFN or MoE
6. residual add

Defined in:

- [hobbylm/model.py](/private/tmp/HobbyLM/hobbylm/model.py:246)

---

## What Attention Means In This Model

Attention is implemented in:

- [hobbylm/model.py](/private/tmp/HobbyLM/hobbylm/model.py:210)

This model uses:

- grouped-query attention
- RoPE positional encoding
- causal attention for next-token prediction

For the 30M preset:

- query heads: `4`
- key/value heads: `1`

That means multiple query heads share fewer key/value heads, which saves memory and compute.

---

## What Pretraining Means Here

This run was **autoregressive language-model pretraining**.

That means:

- feed token sequences into the model
- ask the model to predict the next token
- compare prediction with the correct next token
- compute loss
- update weights

The training loop is in:

- [training/train.py](/private/tmp/HobbyLM/training/train.py:53)

The core training section is:

- [training/train.py](/private/tmp/HobbyLM/training/train.py:185)

---

## What `x` and `y` Mean

Suppose a tokenized sequence is:

- `[101, 205, 17, 88, 999]`

Then next-token training conceptually uses:

- `x = [101, 205, 17, 88]`
- `y = [205, 17, 88, 999]`

Meaning:

- after token `101`, predict `205`
- after `101, 205`, predict `17`
- after `101, 205, 17`, predict `88`
- after `101, 205, 17, 88`, predict `999`

In the real training loop, batches are pulled like this:

- [training/train.py](/private/tmp/HobbyLM/training/train.py:198)

Specifically:

- `x, y = train_prefetch.next()`

So:

- `x` = input token IDs
- `y` = target next-token IDs

---

## What “Batch of Token IDs” Means

A language model does not train on raw text directly.

Text is first converted into tokens, and tokens are represented as integer IDs.

Example:

- raw text: `The capital of France is`
- token IDs: some integer list such as `[464, 3139, 286, 4881, ...]`

A **batch** means many token sequences grouped together so the GPU can process them in parallel.

So yes:

- a batch of token IDs = a chunk of text data after tokenization

---

## Exact Training Math For The First Run

The first real run used:

- `seq_len = 256`
- `micro_batch_seqs = 8`
- `batch_tokens = 32768`
- `gpus = 1`

From:

- [training/train.py](/private/tmp/HobbyLM/training/train.py:142)

Math:

- one sequence = `256` tokens
- one micro-batch = `8` sequences
- one micro-batch = `8 x 256 = 2048` tokens
- target tokens per optimizer step = `32768`
- accumulation count = `32768 / 2048 = 16`

Meaning:

- the code processes 16 micro-batches
- then performs 1 optimizer update

So in this run:

- `1 micro-batch = 2048 tokens`
- `1 training step = 32768 tokens`
- `20 steps = 655,360 tokens total`

This is why a “step” does not mean one sequence. It means one full learning update after enough micro-batches have been accumulated.

---

## Chinchilla-Style Intuition

A common rough rule is:

- optimal training tokens is around `20 x parameters`

So for a `30M` parameter model:

- `30M x 20 = 600M tokens`

This gives the rough intuition that a Chinchilla-style full run for a 30M model might be around:

- `600M` training tokens

Important:

- we did **not** train 600M tokens yet
- we only trained about `0.655M` tokens in the proof run

So this first run was only to prove the pipeline works, not to fully train the model.

---

## What “Proof Run” Means

A proof run is a short real training run used to verify that the entire stack works.

It checks:

- data access works
- Modal GPU execution works
- model forward pass works
- backward pass works
- optimizer step works
- checkpoints save correctly

So this was not fake or dry-run training.

It was real training, just very short.

---

## Commands We Used Conceptually

### 1. GPU Smoke Test

Purpose:

- verify the GPU path works before a real training run

Command:

```bash
/private/tmp/HobbyLM/.venv/bin/python -m modal run /private/tmp/HobbyLM/training/modal_train.py --action smoke --preset 30M
```

Outcome:

- successful
- confirmed forward/backward/optimizer path on H100

### 2. Download Training Data

Purpose:

- fetch tokenized FineWeb shards from Hugging Face into the Modal volume

Command:

```bash
/private/tmp/HobbyLM/.venv/bin/python -m modal run /private/tmp/HobbyLM/training/modal_train.py --action download --chunks 1 --data 10B
```

Outcome:

- downloaded one training chunk and validation shard
- roughly `100M` train tokens available from that one chunk

### 3. First Real 30M Training Run

Purpose:

- perform a short real pretraining run

Command:

```bash
/private/tmp/HobbyLM/.venv/bin/python -m modal run /private/tmp/HobbyLM/training/modal_train.py --action train --preset 30M --steps 20 --run-name 30M_demo_20 --micro 8 --seq-len 256 --batch-tokens 32768 --save-every 10 --data 10B
```

Outcome:

- real training completed
- checkpoints saved
- final validation loss written

---

## First Real Run Results

The first 30M run completed with:

- model: `30M`
- GPU: `1x H100`
- steps: `20`
- final validation loss: `10.5319`

Saved checkpoints:

- `/data/runs/30M_demo_20/ckpt_10.pt`
- `/data/runs/30M_demo_20/ckpt_20.pt`
- `/data/runs/30M_demo_20/model.pt`

Interpretation:

- the pipeline works end-to-end
- the model trained without crashing
- the loss decreased
- the project is ready for longer runs

---

## Second Stronger 30M Study Run

After the 20-step proof run, we launched a stronger study run to move from “pipeline verification” toward “small real pretraining.”

Command:

```bash
/private/tmp/HobbyLM/.venv/bin/python -m modal run /private/tmp/HobbyLM/training/modal_train.py --action train --preset 30M --steps 200 --run-name 30M_study_200 --micro 8 --seq-len 256 --batch-tokens 32768 --save-every 50 --data 10B
```

Outcome:

- run name: `30M_study_200`
- GPU: `1x H100`
- steps: `200`
- final validation loss: `7.7182`
- final model saved successfully

Saved checkpoints:

- `/data/runs/30M_study_200/ckpt_50.pt`
- `/data/runs/30M_study_200/ckpt_100.pt`
- `/data/runs/30M_study_200/ckpt_150.pt`
- `/data/runs/30M_study_200/ckpt_200.pt`
- `/data/runs/30M_study_200/model.pt`

Training progression from logs:

- step `0`: loss `11.0551`
- step `50`: loss `9.9315`
- step `100`: loss `8.9388`
- step `150`: loss `7.9511`
- step `190`: loss `7.7298`
- final val loss: `7.7182`

Interpretation:

- this is still a small run relative to a full Chinchilla-style budget
- but it is much more meaningful than the 20-step proof run
- it clearly demonstrates stable learning on real data
- it produced a much better checkpoint for later generation tests

---

## Third Long 30M Continuation Run

After `30M_study_200`, we launched a longer continuation run from that checkpoint.

Command:

```bash
/private/tmp/HobbyLM/.venv/bin/python -m modal run /private/tmp/HobbyLM/training/modal_train.py --action train --preset 30M --steps 1000 --run-name 30M_study_1000 --micro 8 --seq-len 256 --batch-tokens 32768 --save-every 200 --data 10B --init-from /data/runs/30M_study_200/model.pt
```

Outcome:

- run name: `30M_study_1000`
- resumed from: `/data/runs/30M_study_200/model.pt`
- final validation loss: `5.7572`
- final model saved successfully

Saved checkpoints:

- `/data/runs/30M_study_1000/ckpt_200.pt`
- `/data/runs/30M_study_1000/ckpt_400.pt`
- `/data/runs/30M_study_1000/ckpt_600.pt`
- `/data/runs/30M_study_1000/ckpt_800.pt`
- `/data/runs/30M_study_1000/ckpt_1000.pt`
- `/data/runs/30M_study_1000/model.pt`

Important validation milestones:

- step `250`: val loss `6.5828`
- step `500`: val loss `6.1658`
- step `750`: val loss `5.9073`
- step `1000`: final val loss `5.7572`

Interpretation:

- this is the strongest base-model checkpoint produced in this session
- the model improved substantially over both the 20-step and 200-step runs
- the continuation path from checkpoint works correctly
- this checkpoint is the correct one to use for the next generation test

---

## Local Code Changes Made During This Setup

### 1. Added a 30M preset

Added in:

- [hobbylm/config.py](/private/tmp/HobbyLM/hobbylm/config.py:119)

Purpose:

- make the project easier to study and cheaper to test

### 2. Added `--val_tokens` CLI support

Added in:

- [training/train.py](/private/tmp/HobbyLM/training/train.py:67)

Purpose:

- allow smaller validation runs during testing

### 3. Fixed `generate()` import path

Fixed in:

- [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:227)

Purpose:

- ensure `hobbylm.generate` can be imported inside the Modal container

### 4. Fixed Modal repo mount behavior

Fixed in:

- [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:26)

Purpose:

- ensure the actual HobbyLM repo is mounted into Modal regardless of the current shell directory when launching the command

---

## What Success Looks Like At This Stage

At this stage, success means:

- credentials are working
- data downloads correctly
- GPU training starts and finishes
- checkpoints are produced
- the user understands what model, data, tokens, steps, and batches mean

We achieved that.

---

## What Success Looks Like At Each Step

### 1. Credential Setup

Success looks like:

- Hugging Face token exists
- Modal auth works locally
- Modal secret can be referenced by the training script

Meaning:

- commands can launch
- dataset downloads can authenticate

### 2. Data Download

Success looks like:

- FineWeb shard files are downloaded into the Modal volume
- validation shard is present
- the training script can read them without crashing

Meaning:

- we have actual tokenized data available for pretraining

### 3. GPU Smoke Test

Success looks like:

- model forward pass works
- backward pass works
- optimizer step works
- the smoke run exits cleanly

Meaning:

- the code and GPU environment are compatible

### 4. First Proof Run

Success looks like:

- a real training run completes
- loss is finite
- checkpoints save
- final validation runs

Meaning:

- the full train/save/eval pipeline works end-to-end

### 5. Stronger Study Run

Success looks like:

- loss drops further than the proof run
- checkpoints save during training
- final validation is clearly better than before

Meaning:

- the model is not only runnable, it is genuinely learning

### 6. Long Continuation Run

Success looks like:

- resume from a checkpoint works with no missing or unexpected weights
- validation keeps improving over time
- final model is stronger than the previous checkpoint

Meaning:

- the project now has a meaningful base-model checkpoint, not just a demo artifact

### 7. Generation Test

Success looks like:

- the checkpoint loads
- the model produces text
- newer checkpoints feel more coherent than older checkpoints

Meaning:

- training improvements are showing up in actual inference behavior

### 8. Next Stage: Instruction Tuning

Success will look like:

- the model follows prompts more directly
- outputs look more like answers and less like raw continuation
- chat-style prompts behave better

Meaning:

- the base model starts turning into an assistant-style model

---

## Generation Comparison Milestone

We ran generation from both:

- `30M_study_200/model.pt`
- `30M_study_1000/model.pt`

using the prompt:

- `The capital of France is`

### 200-step checkpoint behavior

The `30M_study_200` output was mostly noisy and unstable, with weak sentence structure and many broken token fragments.

Interpretation:

- the model loaded and generated successfully
- but the checkpoint was still very early and low quality

### 1000-step checkpoint behavior

The `30M_study_1000` output was still factually wrong, but noticeably more sentence-like and more structurally coherent.

Sample start:

> The capital of France is in Canada, the boarding of B BR...

Interpretation:

- still not a good answer
- still not assistant quality
- but clearly more language-like than the 200-step model

This is an important success signal:

- better validation loss is now also showing up as better generation behavior

So the 1000-step checkpoint is the strongest base-model artifact produced in this session.

---

## Simple Interactive Interface

Because the trained checkpoints currently live on Modal rather than as local files on this machine, we added a tiny wrapper script that talks to the remote checkpoint through the existing Modal generation path.

Script:

- [scripts/modal_base_chat.py](/private/tmp/HobbyLM/scripts/modal_base_chat.py:1)

Purpose:

- keep a lightweight local conversation history
- format prompts as `User:` / `Assistant:`
- call the remote Modal generation path for each turn
- make the current base model feel more interactive without pretending it is a fully tuned chat model

Example usage:

```bash
/private/tmp/HobbyLM/.venv/bin/python scripts/modal_base_chat.py --run-name 30M_study_1000
```

Important caveat:

- this is still a base model
- it is not instruction-tuned yet
- so the interface can feel chat-like, but the model behavior will still be weaker than a real assistant

What success looks like here:

- the wrapper starts cleanly
- each user turn triggers remote generation
- conversation history is preserved locally across turns
- the user can experiment with the trained checkpoint interactively

---

## Current State

Current state of the project:

- setup is done
- first proof training run is done
- the repo is slightly improved for 30M experimentation
- next work should focus on either:
  - longer 30M training
  - generation from the trained checkpoint
  - study of one full forward/backward step

---

## Best Next Steps

Recommended order:

1. run generation successfully from the 30M checkpoint
2. run a longer 30M training job toward a more meaningful token budget
3. decide whether to stay near Chinchilla-style small-model scaling or move up to a larger preset

If following rough Chinchilla intuition:

- 30M model target is on the order of `600M` tokens
- the first run completed only about `655k` tokens

So the next meaningful scaling step is a longer training run, not a bigger model yet.

---

## One-Sentence Summary

We used the HobbyLM repo as the training system, Hugging Face as the dataset source, and Modal as the GPU runtime, added a 30M learning preset, and completed a first real short pretraining run that proved the full small-LLM pipeline works end-to-end.
