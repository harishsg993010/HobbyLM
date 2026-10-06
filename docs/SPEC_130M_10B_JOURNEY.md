# 130M FineWeb-10B Journey Spec

## Purpose

This document is the working runbook for our main `130M` HobbyLM run.

It is meant to be:

- a learning document
- a reproducible execution note
- a comparison record against the other parallel efforts
- a checkpoint and benchmark journey log

This run is being done in the existing HobbyLM repo, not in a brand-new repo.

---

## Main Mission

We are using the HobbyLM codebase to run the built-in `130M` preset on the FineWeb-10B dataset source, with a strong focus on:

- good training stability
- safe optimization choices
- regular checkpointing
- benchmarkability
- clear documentation

This is not an RL project.

This is:

- base pretraining
- checkpoint saving
- later benchmark comparison

---

## Budget Mode

We now have about `$13` of Modal credit left.

So the new rule is:

- do not spend large GPU count on uncertain runs

Simple order from now on:

1. run a cheap `check` profile
2. run a small `pilot` profile
3. if credits are very tight, run `tight_resume`
4. run a short `resume_block`
5. run a bigger `main_block` only after the earlier stages look healthy

What success looks like:

- we catch mistakes on cheaper runs first
- we only use `4xH100` for real learning blocks
- we avoid wasting credits on preventable failures

Simple rule:

- if we mainly want a learning signal, use `tight_resume`
- if we want a meaningful training chunk, use `resume_block`
- if we want a bigger real push and credits still look okay, use `main_block`

### Practical `$13` Workflow

Use this exact thinking order:

1. let the currently running block finish
2. record final validation loss and saved checkpoints
3. compare the new loss against the previous checkpoint
4. choose the next run shape only after that

Decision rule:

- if the block improves cleanly and credits still look okay, one more `tight_resume`
- if the block improves strongly and we still want faster progress, one `resume_block`
- if the block is flat or unstable, stop training and switch to testing / generation / evaluation

What success looks like:

- we do not guess the next run blindly
- every next run is justified by the latest loss result
- we keep enough credit for at least one test / eval step after training

---

## Very Simple Definitions

### Built-In Preset

A built-in preset means:

- the repo already has a ready-made model recipe

In this repo, `130M` is one such preset in:

- [hobbylm/config.py](/private/tmp/HobbyLM/hobbylm/config.py:125)

That preset already fixes:

- model width
- layer count
- number of experts
- attention heads
- MoE routing structure

So we are **not** inventing a fresh 135M architecture tonight.

We are using the repo's built-in `130M` preset because it is already part of HobbyLM and is safer to run quickly.

### Why 130M But Also 140M

The preset name is:

- `130M`

But the actual counted total parameters are about:

- `140M total`

So:

- `130M` = preset label in the repo
- `~140M` = actual counted total size

If someone says `135M`, that is usually just rough size language.

### Active Params Per Token

This model is MoE.

That means:

- not every expert is used for every token

So the model stores about:

- `140M total params`

but for one token it uses about:

- `62M active params`

Simple analogy:

- total teachers in the school = `140M`
- teachers actually helping one student right now = `62M`

---

## Existing Codebase We Are Reusing

We are not building a new training framework from scratch.

We are reusing HobbyLM:

- [hobbylm/config.py](/private/tmp/HobbyLM/hobbylm/config.py:1)
- [training/train.py](/private/tmp/HobbyLM/training/train.py:1)
- [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:1)

What we are doing is:

- choosing the preset
- choosing GPU count
- choosing checkpoint cadence
- choosing safe optimization settings
- documenting the journey carefully

---

## Dataset Plan

We are using:

- FineWeb-10B dataset source

In this setup:

- `data=10B` points at the FineWeb-10B corpus family
- train data is split into chunk files
- each train chunk is about `100M` tokens

So rough mapping is:

- `1 chunk` -> `~100M` tokens
- `10 chunks` -> `~1B` tokens
- `50 chunks` -> `~5B` tokens
- `100 chunks` -> `~10B` tokens

This means that if we want to move toward at least `5B` tokens available, we should eventually have about:

- `50` train chunks downloaded

Important distinction:

- available tokens = data stored in Modal volume
- consumed tokens = tokens the model has actually trained on

---

## GPU Plan

We are using the same HobbyLM repo and extending its launcher to support:

- `1xH100`
- `4xH100`
- `8xH100`

For this run, `130M` is locked and `4xH100` is the intended path.

That support now exists in:

- [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:324)

---

## Checkpoint Policy

Checkpoint interval is locked to:

- every `500` steps

Why:

- easier recovery
- easier benchmark snapshots
- easier comparison with other runs

So success looks like:

- `ckpt_500.pt`
- `ckpt_1000.pt`
- `ckpt_1500.pt`
- and so on

plus:

- final `model.pt`

---

## Safe Optimization Settings

We want safe optimization, not risky tricks.

In this repo, the recommended safe optimization bundle is:

- `opts=all_safe`

That means the run uses:

- `fused_ce=true`
- `polar` orthogonalizer

This comes from the throughput presets in:

- [training/modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:85)

Why we call it safe:

- it is already described in the repo as the shipping path
- it improves throughput without using the known broken experimental path

Things we should avoid for the main run:

- `fp8`
- `all_max`

because the repo notes those are experimental / broken for training reliability

---

## What Success Looks Like At Every Major Step

### Step 1: Lock The Run Definition

Success looks like:

- `130M` preset chosen
- `4xH100` chosen
- FineWeb-10B chosen
- checkpoint cadence fixed at `500`
- safe optimization mode fixed

### Step 2: Prepare Data

Success looks like:

- enough FineWeb chunks are downloaded into the Modal volume
- the run is not limited to just 1 chunk
- data is ready in `/data/fineweb10B`

### Step 3: Launch Training

Success looks like:

- Modal job starts
- `4xH100` path is used
- torchrun launches without error
- no missing-file or auth issue stops the run

### Step 4: Early Training Health Check

Success looks like:

- loss is finite
- no NaNs
- throughput stabilizes after warmup
- optimizer is stepping correctly

### Step 5: Stable Learning Curve

Success looks like:

- loss drops early
- then becomes smoother
- sometimes plateaus
- continues improving without instability

We are looking for:

- stable curve
- not chaos

### Step 6: Checkpoint Saving

Success looks like:

- checkpoints appear every `500` steps
- resume remains possible
- final model save works

### Step 7: Run Evaluation Later

Success looks like:

- a saved checkpoint can be benchmarked
- benchmark numbers are recorded clearly
- our run can be compared against the other code paths

---

## Main Training Shape We Want

We do not need perfect Chinchilla adherence.

We do want:

- a stable loss curve
- meaningful progress on the chosen dataset
- enough checkpoints to compare runs
- a benchmarkable result

So the main quality signal is:

- stable and improving training

not:

- theoretical perfection

---

## Journey Logging Rules

For this run, we should keep writing down:

- exact launch commands
- exact data chunk counts
- exact checkpoint names
- validation loss snapshots
- benchmark results
- any bug fixes or optimization changes

This document should be updated as the run progresses.

---

## Initial Execution Checklist

1. stay in this same HobbyLM repo
2. use built-in `130M` preset
3. use `1xH100` for checks and `4xH100` for real train blocks
4. use `opts=all_safe`
5. save every `500` steps
6. grow FineWeb-10B chunk availability
7. launch the main run
8. monitor stability
9. benchmark checkpoints later

---

## First Main Run Command

The first main launch shape for this journey is:

```bash
/private/tmp/HobbyLM/.venv/bin/python -m modal run /private/tmp/HobbyLM/training/modal_train.py \
  --action train \
  --preset 130M \
  --gpus 4 \
  --steps 4000 \
  --run-name 130M_10B_main \
  --save-every 500 \
  --data 10B \
  --opts all_safe
```

Simple meaning:

- use the built-in `130M` preset
- train on `4xH100`
- use FineWeb-10B source
- save checkpoints every `500` steps
- use the safe optimization bundle

### Success For This First Launch

Success looks like:

- Modal starts the `4xH100` job
- torchrun launches cleanly
- the first losses are finite
- checkpoints begin to appear every `500` steps

---

## Step-To-Token Math

For this 130M run shape:

- `batch_tokens = 262144` tokens per optimizer step

That means:

- `4000 steps`  -> about `1.05B` tokens consumed
- `8000 steps`  -> about `2.10B` tokens consumed
- `12000 steps` -> about `3.15B` tokens consumed
- `16000 steps` -> about `4.19B` tokens consumed
- `19073 steps` -> about `5.00B` tokens consumed

Important meaning:

- we already have about `5B` tokens available in storage
- but `4000` steps only uses about `1.05B` of them

So “5B data ready” and “5B tokens actually trained through” are different things.

---

## Resume Rule

We now support two different restart styles:

- `init_from` = weight-only warm start
- `resume_from` = true training resume

### `init_from` In Simple Words

This means:

- load the model brain
- but start a fresh training session around it

So it does **not** restore:

- optimizer momentum
- scheduler position
- exact saved step count

This is useful when we want:

- a fresh continuation experiment
- a new branch from an older checkpoint

### `resume_from` In Simple Words

This means:

- load the model weights
- load optimizer state
- load saved step count
- restore RNG state
- continue the training story properly

Simple analogy:

- `init_from` = same student, new notebook
- `resume_from` = same student, same notebook, same page

### Why Old Resumed Blocks Looked Like Step `0`

Earlier, our continuation runs were using the old weight-only path.

That is why a resumed block could say:

- `step 0`

even though the model had already learned a lot before.

The model memory was continuing.

The step counter and optimizer history were not.

### What Success Looks Like After This Patch

For future resumes, success means:

- checkpoint saves model + optimizer + RNG + step information
- `resume_from` starts from the real saved step
- learning-rate schedule stays continuous
- checkpoint history is easier to explain and publish later

So if we resume from a checkpoint saved at step `3000`, the next run should continue from there instead of pretending to be a brand-new step `0` block.

---

## Progress So Far

### Data

Current storage status:

- FineWeb-10B source
- `50` train chunks downloaded
- about `5B` tokens available

### Training History

Main run:

- run name: `130M_10B_main`
- trained to about `3000` steps before local-disconnect stop
- checkpoints saved through `ckpt_3000.pt`
- best reported validation at step `3000`: about `3.7778`

Resume run:

- run name: `130M_10B_resume_3000`
- loaded from `130M_10B_main/ckpt_3000.pt`
- trained `1000` additional steps
- saved `ckpt_500.pt` and final `model.pt`
- final validation loss: about `3.6362`

Effective total learning history so far:

- about `4000` steps
- about `1.05B` tokens consumed

Important note:

- those earlier continuation runs used the old weight-only restart style
- future continuation runs can use the new full `resume_from` path
- that will make total step accounting much cleaner

### Benchmark Preparation

Cached eval task groups:

- hellaswag
- openbookqa
- winogrande
- arc_challenge
- arc_easy
- boolq
- piqa
