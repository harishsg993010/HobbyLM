//! hobby-rs as a library: a self-contained `Engine` (owns all weights, no GGUF borrow) that
//! loads the hobbylm GGUF and streams text generation. Used by the CLI and the Tauri app.

pub mod clip;
pub mod config;
pub mod dcae;
pub mod dit;
pub mod gguf;
pub mod imagegen;
pub mod imgops;
pub mod model;
pub mod ops;
pub mod png;
pub mod quant;
pub mod sample;
pub mod st;
pub mod tokenizer;

use anyhow::Result;
use std::path::Path;

pub use config::Config;
pub use tokenizer::Tokenizer;

/// Real GPT-2 vocab cutoff: ids >= this are user-defined sentinels we never emit.
pub const GPT2_VALID: usize = 50257;

#[derive(Clone)]
pub struct GenOpts {
    pub max_new: usize,
    pub temp: f32,
    pub top_p: f32,
    pub seed: u64,
    pub rep_penalty: f32, // 1.0 = off; >1 penalizes already-generated tokens (curbs repetition loops)
}

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts { max_new: 256, temp: 0.0, top_p: 0.95, seed: 1234, rep_penalty: 1.0 }
    }
}

/// Decode-time settings for the diffusion (iterative-denoising) path.
#[derive(Clone)]
pub struct DiffOpts {
    pub gen_len: usize,      // total tokens to generate (canvas length)
    pub block: usize,        // semi-autoregressive block length
    pub steps: usize,        // denoising steps spread across the whole canvas
    pub temp: f32,           // 0 = greedy; >0 = Gumbel sampling
    pub rep_penalty: f32,    // cross-canvas presence penalty (>1 curbs the repeat collapse)
    pub remask_steps: usize, // low-confidence remasking refinement passes per block
    pub remask_frac: f32,    // fraction of a block re-masked each refinement pass
    pub seed: u64,
    pub cached: bool,        // prefix KV-cache: each step re-runs only the active block (fast path)
    pub threshold: f32,      // Fast-dLLM confidence-threshold parallel decode: commit all conf>tau
                             // per step (0 = fixed num_transfer schedule using `steps`)
}

impl Default for DiffOpts {
    fn default() -> Self {
        // sweep-tuned defaults for the 500M_diff model (temp0 / strong rep-pen / a few remask passes)
        DiffOpts { gen_len: 128, block: 32, steps: 32, temp: 0.0, rep_penalty: 1.4,
                   remask_steps: 0, remask_frac: 0.3, seed: 1234, cached: true, threshold: 0.0 }
    }
}

/// Spread `n` unmask events across `steps` as evenly as possible (sums to n).
fn num_transfer(n: usize, steps: usize) -> Vec<usize> {
    let base = n / steps;
    let mut out = vec![base; steps];
    for o in out.iter_mut().take(n - base * steps) {
        *o += 1;
    }
    out
}

/// A loaded model + tokenizer. Send + Sync (owns all data), so it can live in shared state.
pub struct Engine {
    pub cfg: Config,
    model: model::Model,
    tok: Tokenizer,
}

impl Engine {
    pub fn load(path: &Path, quant: bool) -> Result<Engine> {
        let g = gguf::Gguf::open(path)?;
        let cfg = Config::from_gguf(&g)?;
        let tok = Tokenizer::from_gguf(&g)?;
        let model = model::Model::load(&g, cfg.clone(), quant)?;
        // `g` (the mmap) is dropped here — model + tok own everything.
        Ok(Engine { cfg, model, tok })
    }

    pub fn eos(&self) -> u32 {
        self.tok.eos
    }
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }
    pub fn weight_bytes(&self) -> usize {
        self.model.weight_bytes()
    }

    /// Generate a continuation of `prompt` (optionally prefixed by `embeds` modality vectors,
    /// each d_model long). `on_token` receives each new chunk of decoded text (UTF-8 safe deltas)
    /// and returns `true` to continue or `false` to stop (interruption). Returns the token count.
    pub fn generate(
        &self,
        prompt: &str,
        embeds: &[Vec<f32>],
        opts: &GenOpts,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> usize {
        let mut cache = model::KvCache::new(&self.cfg);
        let mut rng = sample::Rng::new(opts.seed);
        let ids = self.tok.encode(prompt);

        let mut inputs: Vec<Vec<f32>> = embeds.to_vec();
        for &id in &ids {
            inputs.push(self.model.token_embedding(id));
        }
        if inputs.is_empty() {
            return 0;
        }
        let mut pos = inputs.len();
        let mut logits = self.model.prefill(&inputs, &mut cache);
        let valid = GPT2_VALID.min(self.cfg.vocab_size);

        let mut out_ids: Vec<u32> = Vec::new();
        let mut emitted = 0usize; // bytes of decoded text already streamed
        for _ in 0..opts.max_new {
            for l in logits.iter_mut().take(self.cfg.vocab_size).skip(valid) {
                *l = f32::NEG_INFINITY;
            }
            // repetition penalty: discount logits of already-generated tokens (HF-style div/mul by penalty)
            if opts.rep_penalty != 1.0 {
                for &t in &out_ids {
                    let l = &mut logits[t as usize];
                    *l = if *l > 0.0 { *l / opts.rep_penalty } else { *l * opts.rep_penalty };
                }
            }
            let next = sample::sample(&logits, opts.temp, opts.top_p, &mut rng);
            if next == self.tok.eos {
                break;
            }
            out_ids.push(next);
            // emit the new UTF-8 delta (handles multibyte chars spanning tokens)
            let text = self.tok.decode(&out_ids);
            let delta = if text.len() > emitted { &text[emitted..] } else { "" };
            emitted = text.len();
            // called every token (even empty delta) so interruption is checked promptly
            if !on_token(delta) {
                break;
            }
            if pos + 1 >= self.cfg.context_length {
                break;
            }
            logits = self.model.forward(next, pos, &mut cache);
            pos += 1;
        }
        out_ids.len()
    }

    /// One bidirectional forward over `x[0..b1]`; for the block `[b0,b1)` return per-position
    /// (predicted id, prob of that prediction, prob of the CURRENT token) — sentinels banned,
    /// cross-canvas repetition penalty applied.
    fn diffusion_block(&self, x: &[u32], b0: usize, b1: usize, valid: usize, mask: u32,
                       rep_penalty: f32, temp: f32, rng: &mut sample::Rng)
                       -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let v = self.cfg.vocab_size;
        let blk_len = b1 - b0;
        let inputs: Vec<Vec<f32>> = (0..b1).map(|i| self.model.token_embedding(x[i])).collect();
        let all = self.model.forward_bidir(&inputs); // b1 * v

        // tokens already present (prompt + committed) -> penalized everywhere in the block
        let mut present = vec![false; valid];
        if rep_penalty != 1.0 {
            for &id in &x[..b1] {
                let t = id as usize;
                if id != mask && t < valid {
                    present[t] = true;
                }
            }
        }

        let (mut pred, mut pconf, mut cconf) =
            (vec![0u32; blk_len], vec![0.0f32; blk_len], vec![0.0f32; blk_len]);
        for i in 0..blk_len {
            let mut row = all[(b0 + i) * v..(b0 + i) * v + v].to_vec();
            for l in row.iter_mut().take(v).skip(valid) {
                *l = f32::NEG_INFINITY; // never emit mask/sentinel ids
            }
            if rep_penalty != 1.0 {
                for (t, p) in present.iter().enumerate() {
                    if *p {
                        let l = &mut row[t];
                        *l = if *l > 0.0 { *l / rep_penalty } else { *l * rep_penalty };
                    }
                }
            }
            let p = sample::gumbel_argmax(&row, temp, rng);
            let mut probs = row.clone();
            crate::ops::softmax(&mut probs);
            pred[i] = p;
            pconf[i] = probs[p as usize];
            cconf[i] = probs[x[b0 + i] as usize]; // 0 if current is mask (banned -> prob 0)
        }
        (pred, pconf, cconf)
    }

    /// Generate `opts.gen_len` tokens by iterative denoising. Dispatches to the prefix-KV-cached
    /// fast path (each denoising step re-runs only the active block) or the faithful uncached path
    /// (each step re-runs the whole canvas bidirectionally — exact, but O(canvas) per step).
    pub fn generate_diffusion(&self, prompt: &str, opts: &DiffOpts,
                              on_token: impl FnMut(&str) -> bool) -> usize {
        if opts.cached {
            self.generate_diffusion_cached(prompt, opts, on_token)
        } else {
            self.generate_diffusion_uncached(prompt, opts, on_token)
        }
    }

    /// Per-block scoring shared by both paths: given block logits (`all` = blk_len*vocab), the
    /// current block ids, and the committed-token set, return (predicted id, prob of prediction,
    /// prob of current token) per position — sentinels banned, cross-canvas rep-penalty applied.
    fn score_block(&self, all: &[f32], blk_ids: &[u32], committed: &[u32], valid: usize, mask: u32,
                   rep_penalty: f32, temp: f32, rng: &mut sample::Rng)
                   -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let v = self.cfg.vocab_size;
        let blk_len = blk_ids.len();
        let mut present = vec![false; valid];
        if rep_penalty != 1.0 {
            for &id in committed.iter().chain(blk_ids.iter()) {
                let t = id as usize;
                if id != mask && t < valid {
                    present[t] = true;
                }
            }
        }
        let (mut pred, mut pconf, mut cconf) =
            (vec![0u32; blk_len], vec![0.0f32; blk_len], vec![0.0f32; blk_len]);
        for i in 0..blk_len {
            let mut row = all[i * v..i * v + v].to_vec();
            for l in row.iter_mut().take(v).skip(valid) {
                *l = f32::NEG_INFINITY;
            }
            if rep_penalty != 1.0 {
                for (t, p) in present.iter().enumerate() {
                    if *p {
                        let l = &mut row[t];
                        *l = if *l > 0.0 { *l / rep_penalty } else { *l * rep_penalty };
                    }
                }
            }
            let p = sample::gumbel_argmax(&row, temp, rng);
            let mut probs = row.clone();
            crate::ops::softmax(&mut probs);
            pred[i] = p;
            pconf[i] = probs[p as usize];
            cconf[i] = probs[blk_ids[i] as usize];
        }
        (pred, pconf, cconf)
    }

    /// Prefix-KV-cached denoising: encode the prompt bidirectionally into a persistent cache, then
    /// fill each block over `steps` denoising passes that re-run ONLY the active block against the
    /// frozen prefix cache. ~O(block) per step instead of O(canvas). Approximation vs the faithful
    /// path: the prefix cache doesn't re-attend to the evolving block (standard dLLM caching).
    fn generate_diffusion_cached(&self, prompt: &str, opts: &DiffOpts,
                                 mut on_token: impl FnMut(&str) -> bool) -> usize {
        let c = &self.cfg;
        let mask = c.mask_token_id as u32;
        let valid = GPT2_VALID.min(c.vocab_size);
        let eos = self.tok.eos;
        let mut rng = sample::Rng::new(opts.seed);
        let prompt_ids = self.tok.encode(prompt);
        let p = prompt_ids.len();
        if p == 0 {
            return 0;
        }
        let gen_len = opts.gen_len.min(c.context_length.saturating_sub(p + opts.block).max(1));

        // bidirectional prompt encode -> seed the persistent cache
        let mut cache = model::KvCache::new(c);
        let pe: Vec<Vec<f32>> = prompt_ids.iter().map(|&id| self.model.token_embedding(id)).collect();
        let (_, pk, pv) = self.model.run_block(&pe, &cache, 0);
        for li in 0..c.n_layers {
            cache.k[li].extend_from_slice(&pk[li]);
            cache.v[li].extend_from_slice(&pv[li]);
        }
        cache.len = p;

        let embed = |ids: &[u32]| -> Vec<Vec<f32>> {
            ids.iter().map(|&id| self.model.token_embedding(id)).collect()
        };
        let mut committed: Vec<u32> = prompt_ids; // all finalized ids (rep-penalty + decode)
        let mut emitted = 0usize;
        let mut b0 = p;
        while b0 < p + gen_len {
            let b1 = (b0 + opts.block).min(p + gen_len);
            let blk_len = b1 - b0;
            let base = b0;
            let mut blk = vec![mask; blk_len];
            if opts.threshold > 0.0 {
                // Fast-dLLM confidence-threshold parallel decode: each step commit ALL still-masked
                // positions whose confidence exceeds tau; if none clear the bar, commit the single
                // most-confident one (guarantees progress). Adaptive: confident steps unmask many.
                for _ in 0..blk_len.max(1) {
                    if !blk.iter().any(|&id| id == mask) {
                        break;
                    }
                    let (all, _, _) = self.model.run_block(&embed(&blk), &cache, base);
                    let (pred, pconf, _) = self.score_block(
                        &all, &blk, &committed, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                    let masked: Vec<usize> = (0..blk_len).filter(|&i| blk[i] == mask).collect();
                    let mut any = false;
                    for &i in &masked {
                        if pconf[i] >= opts.threshold {
                            blk[i] = pred[i];
                            any = true;
                        }
                    }
                    if !any {
                        let best = *masked.iter()
                            .max_by(|&&a, &&b| pconf[a].partial_cmp(&pconf[b]).unwrap()).unwrap();
                        blk[best] = pred[best];
                    }
                }
            } else {
                // fixed num_transfer schedule (LLaDA): commit exactly k highest-confidence per step
                let sb = (((opts.steps * blk_len) as f32 / gen_len as f32).round() as usize).max(1);
                let sched = num_transfer(blk_len, sb);
                for &k_step in &sched {
                    let (all, _, _) = self.model.run_block(&embed(&blk), &cache, base);
                    let (pred, pconf, _) = self.score_block(
                        &all, &blk, &committed, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                    let mut cand: Vec<(usize, f32)> =
                        (0..blk_len).filter(|&i| blk[i] == mask).map(|i| (i, pconf[i])).collect();
                    if cand.is_empty() {
                        break;
                    }
                    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    for &(i, _) in cand.iter().take(k_step.min(cand.len())) {
                        blk[i] = pred[i];
                    }
                }
                if blk.iter().any(|&id| id == mask) {
                    let (all, _, _) = self.model.run_block(&embed(&blk), &cache, base);
                    let (pred, _, _) = self.score_block(
                        &all, &blk, &committed, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                    for i in 0..blk_len {
                        if blk[i] == mask {
                            blk[i] = pred[i];
                        }
                    }
                }
            }
            for _ in 0..opts.remask_steps {
                let (all, _, _) = self.model.run_block(&embed(&blk), &cache, base);
                let (_, _, cconf) =
                    self.score_block(&all, &blk, &committed, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                let r = ((blk_len as f32 * opts.remask_frac) as usize).max(1);
                let mut idx: Vec<(usize, f32)> = (0..blk_len).map(|i| (i, cconf[i])).collect();
                idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                for &(i, _) in idx.iter().take(r) {
                    blk[i] = mask;
                }
                let (all2, _, _) = self.model.run_block(&embed(&blk), &cache, base);
                let (pred, _, _) =
                    self.score_block(&all2, &blk, &committed, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                for i in 0..blk_len {
                    if blk[i] == mask {
                        blk[i] = pred[i];
                    }
                }
            }

            // finalize: append the block's K/V to the persistent cache
            let (_, bk, bv) = self.model.run_block(&embed(&blk), &cache, base);
            for li in 0..c.n_layers {
                cache.k[li].extend_from_slice(&bk[li]);
                cache.v[li].extend_from_slice(&bv[li]);
            }
            cache.len = b1;
            committed.extend_from_slice(&blk);

            // stream up to the first eos
            let mut endpos = committed.len();
            let mut hit = false;
            for (i, &id) in committed.iter().enumerate().skip(p) {
                if id == eos {
                    endpos = i;
                    hit = true;
                    break;
                }
            }
            let text = self.tok.decode(&committed[p..endpos]);
            let delta = if text.len() > emitted { text[emitted..].to_string() } else { String::new() };
            emitted = text.len();
            let cont = on_token(&delta);
            if hit || !cont {
                return endpos - p;
            }
            b0 = b1;
        }
        gen_len
    }

    /// Faithful (uncached) iterative denoising — each step re-runs the whole canvas bidirectionally.
    fn generate_diffusion_uncached(&self, prompt: &str, opts: &DiffOpts,
                              mut on_token: impl FnMut(&str) -> bool) -> usize {
        let c = &self.cfg;
        let mask = c.mask_token_id as u32;
        let valid = GPT2_VALID.min(c.vocab_size);
        let eos = self.tok.eos;
        let mut rng = sample::Rng::new(opts.seed);

        let prompt_ids = self.tok.encode(prompt);
        let p = prompt_ids.len();
        let gen_len = opts.gen_len.min(c.context_length.saturating_sub(p).max(1));
        let mut x: Vec<u32> = prompt_ids;
        x.extend(std::iter::repeat(mask).take(gen_len));

        let mut emitted = 0usize; // bytes of the generated region already streamed
        let mut b0 = p;
        while b0 < p + gen_len {
            let b1 = (b0 + opts.block).min(p + gen_len);
            let blk_len = b1 - b0;
            let sb = (((opts.steps * blk_len) as f32 / gen_len as f32).round() as usize).max(1);
            let sched = num_transfer(blk_len, sb);

            // fill: commit the most-confident still-masked positions each step
            for &k_step in &sched {
                let (pred, pconf, _) =
                    self.diffusion_block(&x, b0, b1, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                let mut cand: Vec<(usize, f32)> =
                    (0..blk_len).filter(|&i| x[b0 + i] == mask).map(|i| (i, pconf[i])).collect();
                if cand.is_empty() {
                    break;
                }
                cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                for &(i, _) in cand.iter().take(k_step.min(cand.len())) {
                    x[b0 + i] = pred[i];
                }
            }
            // commit any stragglers still masked
            if (0..blk_len).any(|i| x[b0 + i] == mask) {
                let (pred, _, _) =
                    self.diffusion_block(&x, b0, b1, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                for i in 0..blk_len {
                    if x[b0 + i] == mask {
                        x[b0 + i] = pred[i];
                    }
                }
            }
            // refine: re-mask the least-confident committed tokens, re-predict
            for _ in 0..opts.remask_steps {
                let (_, _, cconf) =
                    self.diffusion_block(&x, b0, b1, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                let r = ((blk_len as f32 * opts.remask_frac) as usize).max(1);
                let mut idx: Vec<(usize, f32)> = (0..blk_len).map(|i| (i, cconf[i])).collect();
                idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                for &(i, _) in idx.iter().take(r) {
                    x[b0 + i] = mask;
                }
                let (pred, _, _) =
                    self.diffusion_block(&x, b0, b1, valid, mask, opts.rep_penalty, opts.temp, &mut rng);
                for i in 0..blk_len {
                    if x[b0 + i] == mask {
                        x[b0 + i] = pred[i];
                    }
                }
            }

            // stream the generated region up to the first eos
            let mut endpos = b1;
            let mut hit_eos = false;
            for (i, &id) in x.iter().enumerate().take(b1).skip(p) {
                if id == eos {
                    endpos = i;
                    hit_eos = true;
                    break;
                }
            }
            let text = self.tok.decode(&x[p..endpos]);
            let delta = if text.len() > emitted { text[emitted..].to_string() } else { String::new() };
            emitted = text.len();
            let cont = on_token(&delta);
            if hit_eos || !cont {
                return endpos - p;
            }
            b0 = b1;
        }
        gen_len
    }
}
