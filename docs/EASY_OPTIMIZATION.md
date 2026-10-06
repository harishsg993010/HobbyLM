# Easy Optimization

## Purpose

This file is for simple, high-signal optimization lessons we discover while running HobbyLM.

It is meant to help with:

- learning
- sharing
- documenting practical engineering wins
- future writing or showcasing

This is not only about theory.

It is about:

- what we changed
- why it mattered
- what success looked like

---

## Optimization 1: Clear Step Accounting

### Problem

When we resumed training, the run step counter started again from `0`.

That created confusion between:

- current block step
- total learning history

### Why It Happened

The old code was only loading:

- model weights

It was not loading:

- optimizer state
- training step position
- scheduler progress

So the model remembered what it learned, but the training session did not fully remember where it was.

### Fix

We added proper resume support so future checkpoints can carry:

- model weights
- Muon optimizer state
- AdamW optimizer state
- saved step count
- RNG state

and future resume runs can continue from the real saved step.

### Why This Matters

This improves:

- cleaner experiment tracking
- better LR schedule continuity
- better momentum continuity
- easier resume after interruption

### Success Looks Like

A resumed run should:

- load model weights
- restore optimizer state
- continue from the saved step count
- keep the training story cleaner

---

## Optimization 2: Safe Throughput Path

### Choice

For the main 130M run we selected:

- `opts=all_safe`

which uses:

- `fused_ce=true`
- `polar` orthogonalizer

### Why This Matters

This gives a practical speed / memory improvement without relying on broken experimental paths.

### Success Looks Like

- training starts cleanly
- no NaNs
- throughput improves
- checkpoints still save correctly

---

## Optimization 3: Data Availability vs Data Consumption

### Problem

It is easy to confuse:

- data downloaded
- data actually trained through

### What We Learned

For the 130M run:

- `50` chunks downloaded means about `5B` tokens available
- but `4000` steps only consumes about `1.05B` tokens

### Why This Matters

This helps us avoid saying:

- “we used 5B tokens”

when what really happened was:

- “we made 5B tokens available”

### Success Looks Like

We always track both:

- available tokens
- consumed tokens

---

## Optimization 4: Reuse The Existing Repo

### Problem

There can be a temptation to rebuild everything from scratch.

### What We Chose

We reused the existing HobbyLM repo and improved it in place.

### Why This Matters

This saves:

- time
- debugging effort
- setup mistakes

and lets us focus on:

- better run strategy
- checkpoints
- benchmarks
- practical optimization

### Success Looks Like

- fewer unnecessary moving parts
- faster iteration
- cleaner comparison against other runs

---

## Optimization 5: Save Checkpoints Frequently

### Choice

For the 130M journey we chose:

- checkpoint every `500` steps

### Why This Matters

This gives:

- safer recovery
- milestone tracking
- benchmark snapshots
- easier continuation blocks

### Success Looks Like

- `ckpt_500.pt`
- `ckpt_1000.pt`
- `ckpt_1500.pt`
- and so on

---

## Optimization 6: Budget-Aware Run Profiles

### Problem

When credits get low, the biggest risk is not slow training.

The biggest risk is:

- launching the wrong size run
- using too many GPUs too early
- spending credits before checking the setup

### Fix

We added simple budget-aware launcher profiles in:

- [modal_train.py](/private/tmp/HobbyLM/training/modal_train.py:1)

These profiles help us pick safer default run shapes such as:

- `check`
- `pilot`
- `tight_resume`
- `resume_block`
- `main_block`

### Why This Matters

This makes the workflow simpler:

- first do a cheap check
- then do a small proof block
- then, if needed, do a very cheap continuation block
- then spend 4 GPUs only on a real continuation block

### Success Looks Like

Before a run starts, the launcher now prints:

- GPU count
- step count
- checkpoint count
- token math for the block
- whether we are doing a fresh start or a true resume

That means we can sanity-check the run **before** we burn credits.

### Simple Budget Rule

Use:

- `1 GPU` for smoke, pilot, generate, and eval
- `1 GPU` for `tight_resume` when credits are low
- `4 GPUs` for real training blocks
- avoid `8 GPUs` when credits are tight

### Very Simple Next-Step Rule

After every training block, ask only three questions:

1. Did validation loss improve?
2. Did checkpoints save correctly?
3. Do we still have enough credit for one more useful block?

Then decide:

- `yes + yes + yes` -> continue with the cheapest reasonable next block
- `yes + yes + maybe` -> prefer `tight_resume`
- `no` on loss or stability -> stop training and inspect / test

---

## Rule For Future Entries

Whenever we learn something useful, add:

1. the problem
2. the fix or idea
3. why it matters
4. what success looked like

This keeps the optimization story simple and publishable later.
