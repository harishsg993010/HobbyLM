//! Weights + forward pass + KV cache for the hobbylm MoE. Each projection is a `Mat`
//! (F32 mmap view or owned Q8_0). The fp32 router and all norms stay F32.

use crate::config::Config;
use crate::gguf::Gguf;
use crate::ops::{dot, matvec, rmsnorm, silu, softmax, Rope};
use crate::quant::{Mat, Q8Tensor};
use anyhow::Result;
use rayon::prelude::*;

/// RMSNorm each of `tn` rows of `x` (row length `d`) into `out`, in parallel.
fn rmsnorm_rows(x: &[f32], w: &[f32], eps: f32, out: &mut [f32], d: usize) {
    crate::pool::par_chunks_mut(out, d, |t, o| rmsnorm(&x[t * d..t * d + d], w, eps, o));
}

/// Decode-time MoE over Q8 experts in two parallel passes instead of three matvecs per expert:
/// (1) gate and up rows of all selected experts against `h` quantized once, (2) the down
/// projection, where each output row sums its weighted contributions across experts.
fn moe_fused_q8(sel: &[(&Mat, &Mat, &Mat, f32)], h: &[f32], d: usize, f: usize) -> Vec<f32> {
    const CH: usize = crate::quant::ROWS_PER_TASK;
    let ns = sel.len();
    let qh = crate::quant::quantize_act(h);
    // gu row j*f + r: expert j/2, gate if j is even else up. A chunk may span two matrices.
    let mut gu = vec![0.0f32; ns * 2 * f];
    crate::pool::par_chunks_mut(&mut gu, CH, |c, yc| {
        let mut i = 0;
        while i < yc.len() {
            let row = c * CH + i;
            let (j, r) = (row / f, row % f);
            let len = (f - r).min(yc.len() - i);
            let m = if j % 2 == 0 { sel[j / 2].0 } else { sel[j / 2].1 };
            m.as_q8().unwrap().rows_dot(r, &qh, &mut yc[i..i + len]);
            i += len;
        }
    });
    let acts: Vec<crate::quant::QAct> = (0..ns)
        .map(|s| {
            let a: Vec<f32> = (0..f).map(|i| silu(gu[2 * s * f + i]) * gu[(2 * s + 1) * f + i]).collect();
            crate::quant::quantize_act(&a)
        })
        .collect();
    let mats: Vec<&crate::quant::Q8Tensor> = sel.iter().map(|s| s.2.as_q8().unwrap()).collect();
    let ar: Vec<&crate::quant::QAct> = acts.iter().collect();
    let ws: Vec<f32> = sel.iter().map(|s| s.3).collect();
    let mut out = vec![0.0f32; d];
    crate::pool::par_chunks_mut(&mut out, CH, |c, yc| {
        crate::quant::rows_dot_weighted(&mats, &ar, &ws, c * CH, yc)
    });
    out
}

/// Split a stacked (E, a, b) tensor into E per-expert Mats of (a, b). A Q8_0 tensor loaded for
/// the Q8 runtime is split block-for-block without dequantizing.
fn split_experts(g: &Gguf, name: &str, ne: usize, a: usize, b: usize, quant: bool) -> Result<Vec<Mat>> {
    let ab = a * b;
    if quant {
        if let Some((d, q)) = g.q8_0(name) {
            let nb = ab / 32;
            return Ok((0..ne)
                .into_par_iter()
                .map(|e| Mat::Q8(Q8Tensor::from_blocks(d[e * nb..(e + 1) * nb].to_vec(), q[e * ab..(e + 1) * ab].to_vec(), a, b)))
                .collect());
        }
    }
    let src = g.load(name)?;
    let s = src.as_slice();
    Ok((0..ne).map(|e| Mat::of(&s[e * ab..(e + 1) * ab], a, b, quant)).collect())
}

/// A weight matrix (out, in_), taking Q8_0 blocks straight from the file when the runtime is Q8.
fn load_mat(g: &Gguf, name: &str, out: usize, in_: usize, quant: bool) -> Result<Mat> {
    if quant {
        if let Some((d, q)) = g.q8_0(name) {
            return Ok(Mat::Q8(Q8Tensor::from_blocks(d, q, out, in_)));
        }
    }
    Ok(Mat::build(g.load(name)?, out, in_, quant))
}

enum Ffn {
    Dense {
        gate: Mat, // (dense_ffn, d)
        up: Mat,
        down: Mat, // (d, dense_ffn)
    },
    Moe {
        gate_inp: Mat,      // (n_exp, d) router, kept F32
        bias: Vec<f32>,     // (n_exp,)
        gate_exps: Vec<Mat>, // per expert (f, d)
        up_exps: Vec<Mat>,
        down_exps: Vec<Mat>, // per expert (d, f)
        gate_sh: Mat,       // (f, d)
        up_sh: Mat,
        down_sh: Mat,       // (d, f)
    },
}

struct Layer {
    attn_norm: Vec<f32>,
    attn_qkv: Mat,      // (q+2kv)*hd, d
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    attn_output: Mat,   // (d, q*hd)
    ffn_norm: Vec<f32>,
    ffn: Ffn,
}

pub struct Model {
    pub cfg: Config,
    rope: Rope,
    token_embd: Vec<f32>, // (vocab, d) lookup (dequant if needed)
    output_norm: Vec<f32>,
    output: Mat, // (vocab, d) lm_head
    layers: Vec<Layer>,
}

/// Per-layer K/V cache in f16, stored head-major in blocks of `KV_BLOCK` positions: within a block each
/// KV head's positions are contiguous, so attention streams one head's history sequentially
/// instead of striding across pages. Grows a block at a time (no full-context preallocation).
pub struct KvCache {
    k: Vec<Vec<u16>>,
    v: Vec<Vec<u16>>,
    n: Vec<usize>, // positions stored, per layer
    nkv: usize,
    hd: usize,
    pub len: usize,
}

const KV_BLOCK: usize = 256;

impl KvCache {
    pub fn new(cfg: &Config) -> Self {
        KvCache {
            k: vec![Vec::new(); cfg.n_layers],
            v: vec![Vec::new(); cfg.n_layers],
            n: vec![0; cfg.n_layers],
            nkv: cfg.n_kv_heads,
            hd: cfg.head_dim,
            len: 0,
        }
    }

    #[inline]
    fn at(&self, kvh: usize, t: usize) -> usize {
        ((t / KV_BLOCK * self.nkv + kvh) * KV_BLOCK + t % KV_BLOCK) * self.hd
    }

    /// Append positions to layer `li`; `k`/`v` are position-major rows of n_kv_heads * head_dim.
    pub fn push(&mut self, li: usize, k: &[f32], v: &[f32]) {
        let kvd = self.nkv * self.hd;
        for (kr, vr) in k.chunks(kvd).zip(v.chunks(kvd)) {
            let t = self.n[li];
            if t % KV_BLOCK == 0 {
                let grow = self.nkv * KV_BLOCK * self.hd;
                let len = self.k[li].len() + grow;
                self.k[li].resize(len, 0);
                self.v[li].resize(len, 0);
            }
            for h in 0..self.nkv {
                let o = self.at(h, t);
                crate::ops::to_f16(&kr[h * self.hd..h * self.hd + self.hd], &mut self.k[li][o..o + self.hd]);
                crate::ops::to_f16(&vr[h * self.hd..h * self.hd + self.hd], &mut self.v[li][o..o + self.hd]);
            }
            self.n[li] += 1;
        }
    }

    /// Key of KV head `kvh` at position `t` in layer `li`.
    #[inline]
    pub fn k(&self, li: usize, kvh: usize, t: usize) -> &[u16] {
        let o = self.at(kvh, t);
        &self.k[li][o..o + self.hd]
    }

    #[inline]
    pub fn v(&self, li: usize, kvh: usize, t: usize) -> &[u16] {
        let o = self.at(kvh, t);
        &self.v[li][o..o + self.hd]
    }
}

impl Model {
    /// `quant`: quantize big matmul weights to Q8_0 (router + norms + embeddings stay F32).
    pub fn load(g: &Gguf, cfg: Config, quant: bool) -> Result<Self> {
        let d = cfg.d_model;
        let f = cfg.expert_ffn;
        let rope = Rope::new(cfg.head_dim, cfg.context_length, cfg.rope_theta);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            // norms/bias are always F32 (owned copy); weight matrices go through dequant-aware load.
            let nt = |s: &str| -> Result<Vec<f32>> { Ok(g.f32(&format!("blk.{i}.{s}"))?.to_vec()) };
            let mt = |s: &str, out: usize, in_: usize| load_mat(g, &format!("blk.{i}.{s}"), out, in_, quant);
            let ld = |s: &str| g.load(&format!("blk.{i}.{s}"));
            let ffn = if cfg.is_moe(i) {
                Ffn::Moe {
                    // router kept F32 (fp32 routing precision)
                    gate_inp: Mat::build(ld("ffn_gate_inp.weight")?, cfg.n_experts, d, false),
                    bias: nt("exp_probs_b.bias")?,
                    gate_exps: split_experts(g, &format!("blk.{i}.ffn_gate_exps.weight"), cfg.n_experts, f, d, quant)?,
                    up_exps: split_experts(g, &format!("blk.{i}.ffn_up_exps.weight"), cfg.n_experts, f, d, quant)?,
                    down_exps: split_experts(g, &format!("blk.{i}.ffn_down_exps.weight"), cfg.n_experts, d, f, quant)?,
                    gate_sh: mt("ffn_gate_shexp.weight", f, d)?,
                    up_sh: mt("ffn_up_shexp.weight", f, d)?,
                    down_sh: mt("ffn_down_shexp.weight", d, f)?,
                }
            } else {
                let df = cfg.dense_ffn;
                Ffn::Dense {
                    gate: mt("ffn_gate.weight", df, d)?,
                    up: mt("ffn_up.weight", df, d)?,
                    down: mt("ffn_down.weight", d, df)?,
                }
            };
            layers.push(Layer {
                attn_norm: nt("attn_norm.weight")?,
                attn_qkv: mt("attn_qkv.weight", cfg.q_dim() + 2 * cfg.kv_dim(), d)?,
                attn_q_norm: nt("attn_q_norm.weight")?,
                attn_k_norm: nt("attn_k_norm.weight")?,
                attn_output: mt("attn_output.weight", d, cfg.q_dim())?,
                ffn_norm: nt("ffn_norm.weight")?,
                ffn,
            });
        }
        let output = load_mat(g, "output.weight", cfg.vocab_size, d, quant)?;
        let m = Model {
            token_embd: match g.load("token_embd.weight")? {
                crate::gguf::Src::Owned(v) => v,
                crate::gguf::Src::View(v) => v.to_vec(),
            },
            output_norm: g.f32("output_norm.weight")?.to_vec(),
            output,
            layers,
            rope,
            cfg,
        };
        Ok(m)
    }

    /// Total bytes held by all matmul weights (to report the quantized footprint).
    pub fn weight_bytes(&self) -> usize {
        let mut b = self.output.bytes();
        for l in &self.layers {
            b += l.attn_qkv.bytes() + l.attn_output.bytes();
            match &l.ffn {
                Ffn::Dense { gate, up, down } => b += gate.bytes() + up.bytes() + down.bytes(),
                Ffn::Moe { gate_exps, up_exps, down_exps, gate_sh, up_sh, down_sh, .. } => {
                    for e in gate_exps.iter().chain(up_exps).chain(down_exps) {
                        b += e.bytes();
                    }
                    b += gate_sh.bytes() + up_sh.bytes() + down_sh.bytes();
                }
            }
        }
        b
    }

    /// Embed a token id and run the blocks.
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let d = self.cfg.d_model;
        let x = self.token_embd[token as usize * d..token as usize * d + d].to_vec();
        self.forward_x(x, pos, cache)
    }

    /// Run the blocks on an externally-provided residual `x` (d_model) — used to splice
    /// precomputed image/audio/speech embeddings at marker positions (inputs_embeds).
    pub fn forward_x(&self, mut x: Vec<f32>, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model;
        debug_assert_eq!(x.len(), d);
        let mut h = vec![0.0f32; d];

        for (li, layer) in self.layers.iter().enumerate() {
            rmsnorm(&x, &layer.attn_norm, c.rms_eps, &mut h);
            let attn = self.attention(layer, &h, li, pos, cache);
            for i in 0..d {
                x[i] += attn[i];
            }
            rmsnorm(&x, &layer.ffn_norm, c.rms_eps, &mut h);
            let ff = self.ffn(layer, &h);
            for i in 0..d {
                x[i] += ff[i];
            }
        }
        cache.len = pos + 1;

        rmsnorm(&x, &self.output_norm, c.rms_eps, &mut h);
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output.matvec(&h, &mut logits);
        logits
    }

    fn attention(&self, layer: &Layer, h: &[f32], li: usize, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let c = &self.cfg;
        let hd = c.head_dim;
        let qd = c.q_dim();
        let kvd = c.kv_dim();
        let rep = c.n_heads / c.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut qkv = vec![0.0f32; qd + 2 * kvd];
        layer.attn_qkv.matvec(h, &mut qkv);
        let (q, kv) = qkv.split_at_mut(qd);
        let (k, v) = kv.split_at_mut(kvd);

        for hq in 0..c.n_heads {
            let qh = &mut q[hq * hd..hq * hd + hd];
            norm_head(qh, &layer.attn_q_norm, c.rms_eps);
            self.rope.apply(qh, pos);
        }
        for hk in 0..c.n_kv_heads {
            let kh = &mut k[hk * hd..hk * hd + hd];
            norm_head(kh, &layer.attn_k_norm, c.rms_eps);
            self.rope.apply(kh, pos);
        }

        cache.push(li, k, v);
        let cache = &*cache;
        let n = pos + 1;

        let mut out = vec![0.0f32; qd];
        let q = &*q;
        // Split-K ("flash-decoding"): one task per (KV head, block of ATT_CHUNK positions), so a long
        // context spreads over all threads. Each task reads its K/V rows once for the `rep` query
        // heads sharing them and leaves a partial (max, sum, unnormalized output) per query head;
        // the partials are then merged exactly.
        const ATT_CHUNK: usize = 64;
        assert!(rep <= 8, "score buffer holds up to 8 query heads per KV head");
        let nkv = c.n_kv_heads;
        let nch = n.div_ceil(ATT_CHUNK);
        let pw = hd + 2; // per (task, query head): [acc; hd], max, sum
        let mut part = vec![0.0f32; nkv * nch * rep * pw];
        crate::pool::par_chunks_mut(&mut part, rep * pw, |task, pt_| {
            let (kvh, ch) = (task / nch, task % nch);
            let (t0, t1) = (ch * ATT_CHUNK, ((ch + 1) * ATT_CHUNK).min(n));
            let mut sc = [0.0f32; ATT_CHUNK * 8];
            let len = t1 - t0;
            for t in t0..t1 {
                let kt = cache.k(li, kvh, t);
                for g in 0..rep {
                    let hq = kvh * rep + g;
                    sc[g * ATT_CHUNK + t - t0] = crate::ops::dot_f16(&q[hq * hd..hq * hd + hd], kt) * scale;
                }
            }
            for g in 0..rep {
                let s = &mut sc[g * ATT_CHUNK..g * ATT_CHUNK + len];
                let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0.0f32;
                for v in s.iter_mut() {
                    *v = (*v - m).exp();
                    l += *v;
                }
                pt_[g * pw + hd] = m;
                pt_[g * pw + hd + 1] = l;
            }
            for t in t0..t1 {
                let vt = cache.v(li, kvh, t);
                for g in 0..rep {
                    crate::ops::axpy_f16(sc[g * ATT_CHUNK + t - t0], vt, &mut pt_[g * pw..g * pw + hd]);
                }
            }
        });
        for hq in 0..c.n_heads {
            let (kvh, g) = (hq / rep, hq % rep);
            let pr = |ch: usize| &part[((kvh * nch + ch) * rep + g) * pw..((kvh * nch + ch) * rep + g + 1) * pw];
            let m = (0..nch).map(|ch| pr(ch)[hd]).fold(f32::NEG_INFINITY, f32::max);
            let oh = &mut out[hq * hd..hq * hd + hd];
            let mut l = 0.0f32;
            for ch in 0..nch {
                let p = pr(ch);
                let w = (p[hd] - m).exp();
                l += w * p[hd + 1];
                crate::ops::axpy(w, &p[..hd], oh);
            }
            let inv = 1.0 / l;
            oh.iter_mut().for_each(|v| *v *= inv);
        }

        let mut proj = vec![0.0f32; c.d_model];
        layer.attn_output.matvec(&out, &mut proj);
        proj
    }

    fn ffn(&self, layer: &Layer, h: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        match &layer.ffn {
            Ffn::Dense { gate, up, down } => {
                let f = c.dense_ffn;
                let mut g = vec![0.0f32; f];
                let mut u = vec![0.0f32; f];
                gate.matvec(h, &mut g);
                up.matvec(h, &mut u);
                for i in 0..f {
                    g[i] = silu(g[i]) * u[i];
                }
                let mut out = vec![0.0f32; c.d_model];
                down.matvec(&g, &mut out);
                out
            }
            Ffn::Moe {
                gate_inp,
                bias,
                gate_exps,
                up_exps,
                down_exps,
                gate_sh,
                up_sh,
                down_sh,
            } => {
                let d = c.d_model;
                let f = c.expert_ffn;
                let ne = c.n_experts;

                let mut logits = vec![0.0f32; ne];
                gate_inp.matvec_pool(h, &mut logits); // F32 router
                let mut scores = vec![0.0f32; ne];
                if c.gating_sigmoid {
                    for e in 0..ne {
                        scores[e] = crate::ops::sigmoid(logits[e]);
                    }
                } else {
                    scores.copy_from_slice(&logits);
                    softmax(&mut scores);
                }

                let mut order: Vec<(usize, f32)> =
                    (0..ne).map(|e| (e, scores[e] + bias[e])).collect();
                order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let topk: Vec<usize> = order[..c.top_k].iter().map(|&(e, _)| e).collect();

                let mut w: Vec<f32> = topk.iter().map(|&e| scores[e]).collect();
                if c.expert_weights_norm {
                    let s: f32 = w.iter().sum::<f32>() + 1e-9;
                    for wi in w.iter_mut() {
                        *wi /= s;
                    }
                }

                // (gate, up, down, weight) of every expert this token runs: routed top-k + shared
                let mut sel: Vec<(&Mat, &Mat, &Mat, f32)> = topk
                    .iter()
                    .enumerate()
                    .map(|(rank, &e)| (&gate_exps[e], &up_exps[e], &down_exps[e], w[rank] * c.expert_weights_scale))
                    .collect();
                if c.n_shared > 0 {
                    sel.push((gate_sh, up_sh, down_sh, 1.0));
                }
                if sel.iter().all(|s| s.0.as_q8().is_some() && s.1.as_q8().is_some() && s.2.as_q8().is_some()) {
                    return moe_fused_q8(&sel, h, d, f);
                }

                let mut out = vec![0.0f32; d];
                let mut gbuf = vec![0.0f32; f];
                let mut ubuf = vec![0.0f32; f];
                let mut ybuf = vec![0.0f32; d];

                for (rank, &e) in topk.iter().enumerate() {
                    gate_exps[e].matvec(h, &mut gbuf);
                    up_exps[e].matvec(h, &mut ubuf);
                    for i in 0..f {
                        gbuf[i] = silu(gbuf[i]) * ubuf[i];
                    }
                    down_exps[e].matvec(&gbuf, &mut ybuf);
                    let scale = w[rank] * c.expert_weights_scale;
                    for i in 0..d {
                        out[i] += scale * ybuf[i];
                    }
                }

                if c.n_shared > 0 {
                    gate_sh.matvec(h, &mut gbuf);
                    up_sh.matvec(h, &mut ubuf);
                    for i in 0..f {
                        gbuf[i] = silu(gbuf[i]) * ubuf[i];
                    }
                    down_sh.matvec(&gbuf, &mut ybuf);
                    for i in 0..d {
                        out[i] += ybuf[i];
                    }
                }
                out
            }
        }
    }
}

impl Model {
    /// Token-embedding row (for building the prefill input list).
    pub fn token_embedding(&self, id: u32) -> Vec<f32> {
        let d = self.cfg.d_model;
        self.token_embd[id as usize * d..id as usize * d + d].to_vec()
    }

    /// Process ALL prefill positions at once (matrix ops, weights read once), fill the cache,
    /// and return the logits of the LAST position. `inputs` = initial residual per position
    /// (token embeddings and/or spliced modality embeddings).
    pub fn prefill(&self, inputs: &[Vec<f32>], cache: &mut KvCache) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model;
        let tn = inputs.len();
        let base = cache.len;
        let mut x = vec![0.0f32; tn * d];
        for (t, row) in inputs.iter().enumerate() {
            x[t * d..t * d + d].copy_from_slice(row);
        }
        let mut h = vec![0.0f32; tn * d];

        for (li, layer) in self.layers.iter().enumerate() {
            rmsnorm_rows(&x, &layer.attn_norm, c.rms_eps, &mut h, d);
            let attn = self.attn_prefill(layer, &h, li, base, tn, cache);
            for i in 0..tn * d {
                x[i] += attn[i];
            }
            rmsnorm_rows(&x, &layer.ffn_norm, c.rms_eps, &mut h, d);
            let ff = self.ffn_prefill(layer, &h, tn);
            for i in 0..tn * d {
                x[i] += ff[i];
            }
        }
        cache.len = base + tn;

        let mut hl = vec![0.0f32; d];
        rmsnorm(&x[(tn - 1) * d..tn * d], &self.output_norm, c.rms_eps, &mut hl);
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output.matvec(&hl, &mut logits);
        logits
    }

    fn attn_prefill(&self, layer: &Layer, h: &[f32], li: usize, base: usize, tn: usize, cache: &mut KvCache) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model;
        let hd = c.head_dim;
        let qd = c.q_dim();
        let kvd = c.kv_dim();
        let qw = qd + 2 * kvd;
        let rep = c.n_heads / c.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut qkv = vec![0.0f32; tn * qw];
        layer.attn_qkv.matmul(h, &mut qkv, tn);

        // per-row QK-norm + RoPE (at position base+t), in place
        crate::pool::par_chunks_mut(&mut qkv, qw, |t, row| {
            let pos = base + t;
            let (q, kv) = row.split_at_mut(qd);
            let (k, _v) = kv.split_at_mut(kvd);
            for hq in 0..c.n_heads {
                let qh = &mut q[hq * hd..hq * hd + hd];
                norm_head(qh, &layer.attn_q_norm, c.rms_eps);
                self.rope.apply(qh, pos);
            }
            for hk in 0..c.n_kv_heads {
                let kh = &mut k[hk * hd..hk * hd + hd];
                norm_head(kh, &layer.attn_k_norm, c.rms_eps);
                self.rope.apply(kh, pos);
            }
        });

        // append all K/V to the cache (in position order)
        for t in 0..tn {
            let row = &qkv[t * qw..t * qw + qw];
            cache.push(li, &row[qd..qd + kvd], &row[qd + kvd..qw]);
        }
        let cache = &*cache;

        // causal attention, one task per (query row, KV head): the KV head's keys and values are
        // read once for the `rep` query heads that share them
        let nkv = c.n_kv_heads;
        let mut o = vec![0.0f32; tn * qd];
        crate::pool::par_chunks_mut(&mut o, rep * hd, |ci, og| {
            let (t, kvh) = (ci / nkv, ci % nkv);
            let n = base + t + 1;
            let qrow = &qkv[t * qw..t * qw + qd];
            let mut scores = vec![0.0f32; rep * n];
            for j in 0..n {
                let kt = cache.k(li, kvh, j);
                for g in 0..rep {
                    let hq = kvh * rep + g;
                    scores[g * n + j] = crate::ops::dot_f16(&qrow[hq * hd..hq * hd + hd], kt) * scale;
                }
            }
            for g in 0..rep {
                softmax(&mut scores[g * n..g * n + n]);
            }
            for j in 0..n {
                let vt = cache.v(li, kvh, j);
                for g in 0..rep {
                    crate::ops::axpy_f16(scores[g * n + j], vt, &mut og[g * hd..g * hd + hd]);
                }
            }
        });

        let mut proj = vec![0.0f32; tn * d];
        layer.attn_output.matmul(&o, &mut proj, tn);
        proj
    }

    /// Diffusion forward: run all `tn` positions with FULL BIDIRECTIONAL attention (no causal
    /// mask, no KV cache) and return logits for EVERY position (tn * vocab). Used by the
    /// iterative-denoising decoder — each step re-runs the whole prefix+block canvas.
    pub fn forward_bidir(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model;
        let tn = inputs.len();
        let mut x = vec![0.0f32; tn * d];
        for (t, row) in inputs.iter().enumerate() {
            x[t * d..t * d + d].copy_from_slice(row);
        }
        let mut h = vec![0.0f32; tn * d];
        for layer in self.layers.iter() {
            rmsnorm_rows(&x, &layer.attn_norm, c.rms_eps, &mut h, d);
            let attn = self.attn_bidir(layer, &h, tn);
            for i in 0..tn * d {
                x[i] += attn[i];
            }
            rmsnorm_rows(&x, &layer.ffn_norm, c.rms_eps, &mut h, d);
            let ff = self.ffn_prefill(layer, &h, tn); // FFN/MoE is position-independent — reuse
            for i in 0..tn * d {
                x[i] += ff[i];
            }
        }
        let mut hn = vec![0.0f32; tn * d];
        rmsnorm_rows(&x, &self.output_norm, c.rms_eps, &mut hn, d);
        let mut logits = vec![0.0f32; tn * c.vocab_size];
        self.output.matmul(&hn, &mut logits, tn);
        logits
    }

    /// Cached block forward for diffusion decode: run the `b` block positions (residual embeddings)
    /// attending to the FROZEN prefix K/V already in `cache` (positions 0..base) plus themselves
    /// (bidirectional within the block). Non-destructive. Returns (logits [b*vocab], per-layer block
    /// K, per-layer block V) — the K/V is appended to the cache by the caller when the block is
    /// finalized, so each denoising step only recomputes the active block, not the whole canvas.
    pub fn run_block(&self, block: &[Vec<f32>], cache: &KvCache, base: usize)
                     -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let c = &self.cfg;
        let d = c.d_model;
        let b = block.len();
        let mut x = vec![0.0f32; b * d];
        for (t, row) in block.iter().enumerate() {
            x[t * d..t * d + d].copy_from_slice(row);
        }
        let mut h = vec![0.0f32; b * d];
        let mut bk_all = Vec::with_capacity(c.n_layers);
        let mut bv_all = Vec::with_capacity(c.n_layers);
        for (li, layer) in self.layers.iter().enumerate() {
            rmsnorm_rows(&x, &layer.attn_norm, c.rms_eps, &mut h, d);
            let (attn, bk, bv) = self.attn_block_cached(layer, &h, li, cache, base, b);
            for i in 0..b * d {
                x[i] += attn[i];
            }
            rmsnorm_rows(&x, &layer.ffn_norm, c.rms_eps, &mut h, d);
            let ff = self.ffn_prefill(layer, &h, b);
            for i in 0..b * d {
                x[i] += ff[i];
            }
            bk_all.push(bk);
            bv_all.push(bv);
        }
        let mut hn = vec![0.0f32; b * d];
        rmsnorm_rows(&x, &self.output_norm, c.rms_eps, &mut hn, d);
        let mut logits = vec![0.0f32; b * c.vocab_size];
        self.output.matmul(&hn, &mut logits, b);
        (logits, bk_all, bv_all)
    }

    /// Attention for `run_block`: the `b` block queries attend to the cached prefix keys
    /// (0..base) and the block's own keys (bidirectional). Returns (proj b*d, block K, block V).
    fn attn_block_cached(&self, layer: &Layer, h: &[f32], li: usize, cache: &KvCache,
                         base: usize, b: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let d = c.d_model;
        let hd = c.head_dim;
        let qd = c.q_dim();
        let kvd = c.kv_dim();
        let qw = qd + 2 * kvd;
        let rep = c.n_heads / c.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut qkv = vec![0.0f32; b * qw];
        layer.attn_qkv.matmul(h, &mut qkv, b);
        qkv.par_chunks_mut(qw).enumerate().for_each(|(t, row)| {
            let pos = base + t;
            let (q, kv) = row.split_at_mut(qd);
            let (k, _v) = kv.split_at_mut(kvd);
            for hq in 0..c.n_heads {
                let qh = &mut q[hq * hd..hq * hd + hd];
                norm_head(qh, &layer.attn_q_norm, c.rms_eps);
                self.rope.apply(qh, pos);
            }
            for hk in 0..c.n_kv_heads {
                let kh = &mut k[hk * hd..hk * hd + hd];
                norm_head(kh, &layer.attn_k_norm, c.rms_eps);
                self.rope.apply(kh, pos);
            }
        });
        let mut bk = vec![0.0f32; b * kvd];
        let mut bv = vec![0.0f32; b * kvd];
        for t in 0..b {
            let row = &qkv[t * qw..t * qw + qw];
            bk[t * kvd..t * kvd + kvd].copy_from_slice(&row[qd..qd + kvd]);
            bv[t * kvd..t * kvd + kvd].copy_from_slice(&row[qd + kvd..qw]);
        }

        // prefix keys 0..base come from the cache (rope already applied at finalize time)
        let n = base + b;
        let mut o = vec![0.0f32; b * qd];
        o.par_chunks_mut(qd).enumerate().for_each(|(t, orow)| {
            let qrow = &qkv[t * qw..t * qw + qd];
            let mut scores = vec![0.0f32; n];
            for hq in 0..c.n_heads {
                let kvh = hq / rep;
                let qh = &qrow[hq * hd..hq * hd + hd];
                for (j, sc) in scores.iter_mut().enumerate().take(base) {
                    let kt = cache.k(li, kvh, j);
                    *sc = crate::ops::dot_f16(qh, kt) * scale;
                }
                for j in 0..b {
                    let kt = &bk[j * kvd + kvh * hd..j * kvd + kvh * hd + hd];
                    scores[base + j] = dot(qh, kt) * scale;
                }
                softmax(&mut scores);
                let oh = &mut orow[hq * hd..hq * hd + hd];
                for (j, &p) in scores.iter().enumerate().take(base) {
                    let vt = cache.v(li, kvh, j);
                    crate::ops::axpy_f16(p, vt, oh);
                }
                for j in 0..b {
                    let vt = &bv[j * kvd + kvh * hd..j * kvd + kvh * hd + hd];
                    let p = scores[base + j];
                    for di in 0..hd {
                        oh[di] += p * vt[di];
                    }
                }
            }
        });

        let mut proj = vec![0.0f32; b * d];
        layer.attn_output.matmul(&o, &mut proj, b);
        (proj, bk, bv)
    }

    /// Full bidirectional attention over `tn` positions (RoPE at absolute pos = t; every query
    /// attends to every key). No cache — K/V live in local buffers for this single pass.
    fn attn_bidir(&self, layer: &Layer, h: &[f32], tn: usize) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model;
        let hd = c.head_dim;
        let qd = c.q_dim();
        let kvd = c.kv_dim();
        let qw = qd + 2 * kvd;
        let rep = c.n_heads / c.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut qkv = vec![0.0f32; tn * qw];
        layer.attn_qkv.matmul(h, &mut qkv, tn);
        qkv.par_chunks_mut(qw).enumerate().for_each(|(t, row)| {
            let (q, kv) = row.split_at_mut(qd);
            let (k, _v) = kv.split_at_mut(kvd);
            for hq in 0..c.n_heads {
                let qh = &mut q[hq * hd..hq * hd + hd];
                norm_head(qh, &layer.attn_q_norm, c.rms_eps);
                self.rope.apply(qh, t);
            }
            for hk in 0..c.n_kv_heads {
                let kh = &mut k[hk * hd..hk * hd + hd];
                norm_head(kh, &layer.attn_k_norm, c.rms_eps);
                self.rope.apply(kh, t);
            }
        });

        let mut kbuf = vec![0.0f32; tn * kvd];
        let mut vbuf = vec![0.0f32; tn * kvd];
        for t in 0..tn {
            let row = &qkv[t * qw..t * qw + qw];
            kbuf[t * kvd..t * kvd + kvd].copy_from_slice(&row[qd..qd + kvd]);
            vbuf[t * kvd..t * kvd + kvd].copy_from_slice(&row[qd + kvd..qw]);
        }

        let mut o = vec![0.0f32; tn * qd];
        o.par_chunks_mut(qd).enumerate().for_each(|(t, orow)| {
            let qrow = &qkv[t * qw..t * qw + qd];
            let mut scores = vec![0.0f32; tn];
            for hq in 0..c.n_heads {
                let kvh = hq / rep;
                let qh = &qrow[hq * hd..hq * hd + hd];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kt = &kbuf[j * kvd + kvh * hd..j * kvd + kvh * hd + hd];
                    *sc = dot(qh, kt) * scale;
                }
                softmax(&mut scores);
                let oh = &mut orow[hq * hd..hq * hd + hd];
                for (j, &p) in scores.iter().enumerate() {
                    let vt = &vbuf[j * kvd + kvh * hd..j * kvd + kvh * hd + hd];
                    for di in 0..hd {
                        oh[di] += p * vt[di];
                    }
                }
            }
        });

        let mut proj = vec![0.0f32; tn * d];
        layer.attn_output.matmul(&o, &mut proj, tn);
        proj
    }

    fn ffn_prefill(&self, layer: &Layer, h: &[f32], tn: usize) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.d_model;
        let f = c.expert_ffn;
        match &layer.ffn {
            Ffn::Dense { gate, up, down } => {
                let df = c.dense_ffn;
                let mut g = vec![0.0f32; tn * df];
                let mut u = vec![0.0f32; tn * df];
                gate.matmul(h, &mut g, tn);
                up.matmul(h, &mut u, tn);
                for i in 0..tn * df {
                    g[i] = silu(g[i]) * u[i];
                }
                let mut out = vec![0.0f32; tn * d];
                down.matmul(&g, &mut out, tn);
                out
            }
            Ffn::Moe { gate_inp, bias, gate_exps, up_exps, down_exps, gate_sh, up_sh, down_sh } => {
                let ne = c.n_experts;
                let mut logits = vec![0.0f32; tn * ne];
                // F32 router, one task per token
                crate::pool::par_chunks_mut(&mut logits, ne, |t, row| gate_inp.matvec_serial(&h[t * d..t * d + d], row));

                // route each row; group token rows by selected expert
                let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
                for t in 0..tn {
                    let lg = &logits[t * ne..t * ne + ne];
                    let mut scores = vec![0.0f32; ne];
                    if c.gating_sigmoid {
                        for e in 0..ne {
                            scores[e] = crate::ops::sigmoid(lg[e]);
                        }
                    } else {
                        scores.copy_from_slice(lg);
                        softmax(&mut scores);
                    }
                    let mut order: Vec<(usize, f32)> =
                        (0..ne).map(|e| (e, scores[e] + bias[e])).collect();
                    order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let topk: Vec<usize> = order[..c.top_k].iter().map(|&(e, _)| e).collect();
                    let mut w: Vec<f32> = topk.iter().map(|&e| scores[e]).collect();
                    if c.expert_weights_norm {
                        let s: f32 = w.iter().sum::<f32>() + 1e-9;
                        for wi in w.iter_mut() {
                            *wi /= s;
                        }
                    }
                    for (r, &e) in topk.iter().enumerate() {
                        assign[e].push((t, w[r] * c.expert_weights_scale));
                    }
                }

                // expert groups: (gate, up, down, token rows with weights); the shared expert takes all rows
                let mut groups: Vec<ExpertRows> = assign
                    .into_iter()
                    .enumerate()
                    .filter(|(_, rows)| !rows.is_empty())
                    .map(|(e, rows)| (&gate_exps[e], &up_exps[e], &down_exps[e], rows))
                    .collect();
                if c.n_shared > 0 {
                    groups.push((gate_sh, up_sh, down_sh, (0..tn).map(|t| (t, 1.0)).collect()));
                }
                if groups.iter().all(|g| g.0.as_q8().is_some() && g.1.as_q8().is_some() && g.2.as_q8().is_some()) {
                    moe_prefill_q8(&groups, h, tn, d, f)
                } else {
                    moe_prefill_f32(&groups, h, tn, d, f)
                }
            }
        }
    }
}

/// One expert's (gate, up, down) and the token rows routed to it, with their gate weights.
type ExpertRows<'a> = (&'a Mat, &'a Mat, &'a Mat, Vec<(usize, f32)>);

/// Per-expert batched MoE, for F32 weights.
fn moe_prefill_f32(groups: &[ExpertRows], h: &[f32], tn: usize, d: usize, f: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tn * d];
    for (gate, up, down, rows) in groups {
        let m = rows.len();
        let mut xe = vec![0.0f32; m * d];
        for (i, &(t, _)) in rows.iter().enumerate() {
            xe[i * d..i * d + d].copy_from_slice(&h[t * d..t * d + d]);
        }
        let mut ge = vec![0.0f32; m * f];
        let mut ue = vec![0.0f32; m * f];
        gate.matmul(&xe, &mut ge, m);
        up.matmul(&xe, &mut ue, m);
        for i in 0..m * f {
            ge[i] = silu(ge[i]) * ue[i];
        }
        let mut ye = vec![0.0f32; m * d];
        down.matmul(&ge, &mut ye, m);
        for (i, &(t, wt)) in rows.iter().enumerate() {
            for k in 0..d {
                out[t * d + k] += wt * ye[i * d + k];
            }
        }
    }
    out
}

/// Batched MoE over Q8 experts in two parallel passes across ALL experts at once, instead of
/// three small matmuls per expert: (1) gate+up, one task per (expert, block of rows), with every
/// token row quantized once; (2) down, one task per (expert, block of output rows). Then each
/// token sums its experts' outputs.
fn moe_prefill_q8(groups: &[ExpertRows], h: &[f32], tn: usize, d: usize, f: usize) -> Vec<f32> {
    use crate::quant::{quantize_act, QAct};
    const CH: usize = crate::quant::ROWS_PER_TASK;
    let pool = crate::pool::pool();
    let mut qh: Vec<QAct> = (0..tn).map(|_| QAct::default()).collect();
    crate::pool::par_chunks_mut(&mut qh, 1, |t, q| q[0] = quantize_act(&h[t * d..t * d + d]));

    // pair p = (group, row in group), laid out group by group
    let mut first = Vec::with_capacity(groups.len());
    let mut npairs = 0;
    for g in groups {
        first.push(npairs);
        npairs += g.3.len();
    }

    // pass 1: gu[p] = [gate f | up f]
    let fch = f.div_ceil(CH);
    let mut gu = vec![0.0f32; npairs * 2 * f];
    let gp = gu.as_mut_ptr() as usize;
    pool.run(groups.len() * 2 * fch, &|task| {
        let (g, which, rc) = (task / (2 * fch), (task / fch) % 2, task % fch);
        let m = if which == 0 { groups[g].0 } else { groups[g].1 }.as_q8().unwrap();
        let (r0, n) = (rc * CH, CH.min(f - rc * CH));
        for (i, &(t, _)) in groups[g].3.iter().enumerate() {
            let off = (first[g] + i) * 2 * f + which * f + r0;
            // each (pair, matrix, row block) is written by exactly one task
            let y = unsafe { std::slice::from_raw_parts_mut((gp as *mut f32).add(off), n) };
            m.rows_dot(r0, &qh[t], y);
        }
    });

    let mut acts: Vec<QAct> = (0..npairs).map(|_| QAct::default()).collect();
    crate::pool::par_chunks_mut(&mut acts, 1, |p, a| {
        let row = &gu[p * 2 * f..p * 2 * f + 2 * f];
        let v: Vec<f32> = (0..f).map(|i| silu(row[i]) * row[f + i]).collect();
        a[0] = quantize_act(&v);
    });

    // pass 2: ye[p] = down(act[p])
    let dch = d.div_ceil(CH);
    let mut ye = vec![0.0f32; npairs * d];
    let yp = ye.as_mut_ptr() as usize;
    pool.run(groups.len() * dch, &|task| {
        let (g, rc) = (task / dch, task % dch);
        let m = groups[g].2.as_q8().unwrap();
        let (r0, n) = (rc * CH, CH.min(d - rc * CH));
        for i in 0..groups[g].3.len() {
            let p = first[g] + i;
            let y = unsafe { std::slice::from_raw_parts_mut((yp as *mut f32).add(p * d + r0), n) };
            m.rows_dot(r0, &acts[p], y);
        }
    });

    // combine: each token adds its experts' outputs in group order
    let mut contrib: Vec<Vec<(usize, f32)>> = vec![Vec::new(); tn];
    for (g, grp) in groups.iter().enumerate() {
        for (i, &(t, w)) in grp.3.iter().enumerate() {
            contrib[t].push((first[g] + i, w));
        }
    }
    let mut out = vec![0.0f32; tn * d];
    crate::pool::par_chunks_mut(&mut out, d, |t, o| {
        for &(p, w) in &contrib[t] {
            crate::ops::axpy(w, &ye[p * d..p * d + d], o);
        }
    });
    out
}

fn norm_head(v: &mut [f32], w: &[f32], eps: f32) {
    let n = v.len();
    let mut ss = 0.0f32;
    for &x in v.iter() {
        ss += x * x;
    }
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        v[i] = v[i] * scale * w[i];
    }
}

// silence unused import when matvec is only used via Mat
#[allow(dead_code)]
fn _touch(w: &[f32], x: &[f32], y: &mut [f32]) {
    matvec(w, x, y)
}
