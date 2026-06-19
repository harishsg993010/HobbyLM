//! Weights + forward pass + KV cache for the hobbylm MoE. Each projection is a `Mat`
//! (F32 mmap view or owned Q8_0). The fp32 router and all norms stay F32.

use crate::config::Config;
use crate::gguf::{Gguf, Src};
use crate::ops::{dot, matvec, rmsnorm, silu, softmax, Rope};
use crate::quant::Mat;
use anyhow::Result;
use rayon::prelude::*;

/// RMSNorm each of `tn` rows of `x` (row length `d`) into `out`, in parallel.
fn rmsnorm_rows(x: &[f32], w: &[f32], eps: f32, out: &mut [f32], d: usize) {
    out.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, xr)| rmsnorm(xr, w, eps, o));
}

/// Split a stacked (E, a, b) tensor into E per-expert Mats of (a, b).
fn split_experts(src: Src<'_>, ne: usize, a: usize, b: usize, quant: bool) -> Vec<Mat<'_>> {
    let ab = a * b;
    match src {
        Src::View(w) => (0..ne)
            .map(|e| Mat::from_view(&w[e * ab..(e + 1) * ab], a, b, quant))
            .collect(),
        Src::Owned(v) => (0..ne)
            .map(|e| Mat::from_owned(&v[e * ab..(e + 1) * ab], a, b, quant))
            .collect(),
    }
}

enum Ffn<'a> {
    Dense {
        gate: Mat<'a>, // (dense_ffn, d)
        up: Mat<'a>,
        down: Mat<'a>, // (d, dense_ffn)
    },
    Moe {
        gate_inp: Mat<'a>, // (n_exp, d) router, kept F32
        bias: &'a [f32],   // (n_exp,)
        gate_exps: Vec<Mat<'a>>, // per expert (f, d)
        up_exps: Vec<Mat<'a>>,
        down_exps: Vec<Mat<'a>>, // per expert (d, f)
        gate_sh: Mat<'a>,  // (f, d)
        up_sh: Mat<'a>,
        down_sh: Mat<'a>,  // (d, f)
    },
}

struct Layer<'a> {
    attn_norm: &'a [f32],
    attn_qkv: Mat<'a>,      // (q+2kv)*hd, d
    attn_q_norm: &'a [f32],
    attn_k_norm: &'a [f32],
    attn_output: Mat<'a>,   // (d, q*hd)
    ffn_norm: &'a [f32],
    ffn: Ffn<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    rope: Rope,
    token_embd: Src<'a>, // (vocab, d) lookup (dequant if needed)
    output_norm: &'a [f32],
    output: Mat<'a>, // (vocab, d) lm_head
    layers: Vec<Layer<'a>>,
}

pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
}

impl KvCache {
    pub fn new(cfg: &Config) -> Self {
        let cap = cfg.context_length * cfg.kv_dim();
        KvCache {
            k: (0..cfg.n_layers).map(|_| Vec::with_capacity(cap)).collect(),
            v: (0..cfg.n_layers).map(|_| Vec::with_capacity(cap)).collect(),
            len: 0,
        }
    }
}

impl<'a> Model<'a> {
    /// `quant`: quantize big matmul weights to Q8_0 (router + norms + embeddings stay F32).
    pub fn load(g: &'a Gguf, cfg: Config, quant: bool) -> Result<Self> {
        let d = cfg.d_model;
        let f = cfg.expert_ffn;
        let rope = Rope::new(cfg.head_dim, cfg.context_length, cfg.rope_theta);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            // norms/bias are always F32 (borrow); weight matrices go through dequant-aware load.
            let nt = |s: &str| g.f32(&format!("blk.{i}.{s}"));
            let mt = |s: &str, out: usize, in_: usize| -> Result<Mat<'a>> {
                Ok(Mat::build(g.load(&format!("blk.{i}.{s}"))?, out, in_, quant))
            };
            let ld = |s: &str| g.load(&format!("blk.{i}.{s}"));
            let ffn = if cfg.is_moe(i) {
                Ffn::Moe {
                    // router kept F32 (fp32 routing precision)
                    gate_inp: Mat::build(ld("ffn_gate_inp.weight")?, cfg.n_experts, d, false),
                    bias: nt("exp_probs_b.bias")?,
                    gate_exps: split_experts(ld("ffn_gate_exps.weight")?, cfg.n_experts, f, d, quant),
                    up_exps: split_experts(ld("ffn_up_exps.weight")?, cfg.n_experts, f, d, quant),
                    down_exps: split_experts(ld("ffn_down_exps.weight")?, cfg.n_experts, d, f, quant),
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
        let output = Mat::build(g.load("output.weight")?, cfg.vocab_size, d, quant);
        let m = Model {
            token_embd: g.load("token_embd.weight")?,
            output_norm: g.f32("output_norm.weight")?,
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
        let emb = self.token_embd.as_slice();
        let x = emb[token as usize * d..token as usize * d + d].to_vec();
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
            rmsnorm(&x, layer.attn_norm, c.rms_eps, &mut h);
            let attn = self.attention(layer, &h, li, pos, cache);
            for i in 0..d {
                x[i] += attn[i];
            }
            rmsnorm(&x, layer.ffn_norm, c.rms_eps, &mut h);
            let ff = self.ffn(layer, &h);
            for i in 0..d {
                x[i] += ff[i];
            }
        }
        cache.len = pos + 1;

        rmsnorm(&x, self.output_norm, c.rms_eps, &mut h);
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
            norm_head(qh, layer.attn_q_norm, c.rms_eps);
            self.rope.apply(qh, pos);
        }
        for hk in 0..c.n_kv_heads {
            let kh = &mut k[hk * hd..hk * hd + hd];
            norm_head(kh, layer.attn_k_norm, c.rms_eps);
            self.rope.apply(kh, pos);
        }

        cache.k[li].extend_from_slice(k);
        cache.v[li].extend_from_slice(v);
        let kc = &cache.k[li];
        let vc = &cache.v[li];
        let n = pos + 1;

        let mut out = vec![0.0f32; qd];
        let mut scores = vec![0.0f32; n];
        for hq in 0..c.n_heads {
            let kvh = hq / rep;
            let qh = &q[hq * hd..hq * hd + hd];
            for (t, sc) in scores.iter_mut().enumerate() {
                let kt = &kc[t * kvd + kvh * hd..t * kvd + kvh * hd + hd];
                *sc = dot(qh, kt) * scale;
            }
            softmax(&mut scores);
            let oh = &mut out[hq * hd..hq * hd + hd];
            for (t, &pscore) in scores.iter().enumerate() {
                let vt = &vc[t * kvd + kvh * hd..t * kvd + kvh * hd + hd];
                for di in 0..hd {
                    oh[di] += pscore * vt[di];
                }
            }
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
                gate_inp.matvec(h, &mut logits); // F32 router
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

impl<'a> Model<'a> {
    /// Token-embedding row (for building the prefill input list).
    pub fn token_embedding(&self, id: u32) -> Vec<f32> {
        let d = self.cfg.d_model;
        self.token_embd.as_slice()[id as usize * d..id as usize * d + d].to_vec()
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
            rmsnorm_rows(&x, layer.attn_norm, c.rms_eps, &mut h, d);
            let attn = self.attn_prefill(layer, &h, li, base, tn, cache);
            for i in 0..tn * d {
                x[i] += attn[i];
            }
            rmsnorm_rows(&x, layer.ffn_norm, c.rms_eps, &mut h, d);
            let ff = self.ffn_prefill(layer, &h, tn);
            for i in 0..tn * d {
                x[i] += ff[i];
            }
        }
        cache.len = base + tn;

        let mut hl = vec![0.0f32; d];
        rmsnorm(&x[(tn - 1) * d..tn * d], self.output_norm, c.rms_eps, &mut hl);
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
        qkv.par_chunks_mut(qw).enumerate().for_each(|(t, row)| {
            let pos = base + t;
            let (q, kv) = row.split_at_mut(qd);
            let (k, _v) = kv.split_at_mut(kvd);
            for hq in 0..c.n_heads {
                let qh = &mut q[hq * hd..hq * hd + hd];
                norm_head(qh, layer.attn_q_norm, c.rms_eps);
                self.rope.apply(qh, pos);
            }
            for hk in 0..c.n_kv_heads {
                let kh = &mut k[hk * hd..hk * hd + hd];
                norm_head(kh, layer.attn_k_norm, c.rms_eps);
                self.rope.apply(kh, pos);
            }
        });

        // append all K/V to the cache (in position order)
        for t in 0..tn {
            let row = &qkv[t * qw..t * qw + qw];
            cache.k[li].extend_from_slice(&row[qd..qd + kvd]);
            cache.v[li].extend_from_slice(&row[qd + kvd..qw]);
        }
        let kc = &cache.k[li];
        let vc = &cache.v[li];

        // causal attention, parallel over query rows
        let mut o = vec![0.0f32; tn * qd];
        o.par_chunks_mut(qd).enumerate().for_each(|(t, orow)| {
            let pos = base + t;
            let n = pos + 1;
            let qrow = &qkv[t * qw..t * qw + qd];
            let mut scores = vec![0.0f32; n];
            for hq in 0..c.n_heads {
                let kvh = hq / rep;
                let qh = &qrow[hq * hd..hq * hd + hd];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kt = &kc[j * kvd + kvh * hd..j * kvd + kvh * hd + hd];
                    *sc = dot(qh, kt) * scale;
                }
                softmax(&mut scores);
                let oh = &mut orow[hq * hd..hq * hd + hd];
                for (j, &p) in scores.iter().enumerate() {
                    let vt = &vc[j * kvd + kvh * hd..j * kvd + kvh * hd + hd];
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
                gate_inp.matmul(h, &mut logits, tn);

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

                let mut out = vec![0.0f32; tn * d];
                // routed experts: each processes all its rows in one batched matmul
                for e in 0..ne {
                    let rows = &assign[e];
                    let m = rows.len();
                    if m == 0 {
                        continue;
                    }
                    let mut xe = vec![0.0f32; m * d];
                    for (i, &(t, _)) in rows.iter().enumerate() {
                        xe[i * d..i * d + d].copy_from_slice(&h[t * d..t * d + d]);
                    }
                    let mut ge = vec![0.0f32; m * f];
                    let mut ue = vec![0.0f32; m * f];
                    gate_exps[e].matmul(&xe, &mut ge, m);
                    up_exps[e].matmul(&xe, &mut ue, m);
                    for i in 0..m * f {
                        ge[i] = silu(ge[i]) * ue[i];
                    }
                    let mut ye = vec![0.0f32; m * d];
                    down_exps[e].matmul(&ge, &mut ye, m);
                    for (i, &(t, wt)) in rows.iter().enumerate() {
                        for k in 0..d {
                            out[t * d + k] += wt * ye[i * d + k];
                        }
                    }
                }
                // shared expert: all rows
                if c.n_shared > 0 {
                    let mut gs = vec![0.0f32; tn * f];
                    let mut us = vec![0.0f32; tn * f];
                    gate_sh.matmul(h, &mut gs, tn);
                    up_sh.matmul(h, &mut us, tn);
                    for i in 0..tn * f {
                        gs[i] = silu(gs[i]) * us[i];
                    }
                    let mut ys = vec![0.0f32; tn * d];
                    down_sh.matmul(&gs, &mut ys, tn);
                    for i in 0..tn * d {
                        out[i] += ys[i];
                    }
                }
                out
            }
        }
    }
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
