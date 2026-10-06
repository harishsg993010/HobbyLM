# 130M Tonight Checklist

## Simple Meaning Of "Built-In Preset"

A **built-in preset** means:

- the repo already has a ready-made model setup
- we do not have to invent all the numbers ourselves

In this repo, the preset lives in:

- [hobbylm/config.py](/private/tmp/HobbyLM/hobbylm/config.py:125)

The preset name is:

- `130M`

That preset already defines things like:

- hidden size
- number of layers
- number of experts
- attention heads
- top-k routing

So a preset is basically:

- a saved model recipe

Simple analogy:

- preset = a ready recipe card
- custom model = creating a recipe from scratch

For tonight, using the preset is safer and faster.

---

## Why 130M, 135M, And 140M All Sound Different

This is just naming versus actual counted size.

### Preset Name

The repo calls the preset:

- `130M`

This is the label used in the code.

### Actual Counted Total Parameters

When the model is counted, it comes out around:

- `140M total parameters`

### Why This Happens

Because the preset name is:

- a simple category name

and the exact parameter count depends on:

- embeddings
- attention weights
- dense FFN weights
- MoE expert weights
- output weights

So:

- `130M` = preset name
- `~140M` = actual total parameter count after building the model

If someone says `135M`, that is usually just:

- a rough verbal size label

So for us:

- we are not building a brand-new exact `135M` model tonight
- we are using HobbyLM's built-in `130M` preset
- that preset actually lands around `140M total`

---

## What "62M Active Params Per Token" Means

The full model stores about:

- `140M total parameters`

But for **one token**, the MoE routing does not use every expert.

So only part of the model is active for that token.

That active part is about:

- `62M parameters`

Simple analogy:

- total teachers in the school = `140M`
- teachers actually helping one student right now = `62M`

That is why MoE can be:

- larger in total size
- cheaper per token than a dense model of the same total size

---

## What We Are Doing Tonight

We are **not** building a whole new Modal system from scratch.

We are:

- reusing Harish's HobbyLM repo
- reusing its built-in `130M` preset
- reusing its Modal training pipeline
- choosing better run settings
- documenting the run well
- saving checkpoints often
- benchmarking the model later

So this is:

- **reconfiguring HobbyLM well**

not:

- **writing a new training framework from zero**

---

## Budget Reality

We currently want to stay within about:

- `$25` Modal budget

So the realistic target is:

- train the `130M` preset
- use a good GPU path
- likely aim for around `5B tokens consumed` if time and budget allow

Important note:

- `5B tokens` is realistic as a training target
- `5B parameters` is not realistic for this budget

---

## Tonight Checklist

### Stage 1: Lock The Plan

Success looks like:

- we agree to use the built-in `130M` preset
- we agree this is a HobbyLM reconfiguration run
- we agree to target a strong benchmarkable run, not perfect theory

### Stage 2: Stop Small-Model Distraction

Success looks like:

- stop focusing on the 30M learning run
- switch attention fully to the 130M production run

### Stage 3: Prepare Enough Data

Success looks like:

- make sure enough FineWeb-10B chunks exist in Modal storage
- do not stay stuck at only 1 chunk

### Stage 4: Pick GPU Strategy

Success looks like:

- choose the strongest safe GPU option supported by the repo
- prefer the clean multi-GPU path already present in HobbyLM

### Stage 5: Launch 130M Training

Success looks like:

- run starts without errors
- loss is finite
- throughput is healthy
- checkpoints save regularly

### Stage 6: Watch Loss Shape

Success looks like:

- loss drops
- curve is stable
- no NaNs
- no exploding loss
- no obviously broken training

### Stage 7: Save Checkpoints Properly

Success looks like:

- checkpoints save every fixed interval
- we can resume if needed
- we can benchmark a checkpoint later

### Stage 8: Benchmark

Success looks like:

- run benchmark tasks on the trained model
- compare against the other versions
- decide whether this code path is the best one

---

## Immediate Next Actions

1. confirm tonight's target as:
   - built-in `130M` preset
   - FineWeb-10B source
   - regular checkpoints
   - benchmarkable run
2. prepare enough dataset chunks
3. launch the main 130M run on the strongest supported GPU setup
4. monitor loss and checkpoint saves
