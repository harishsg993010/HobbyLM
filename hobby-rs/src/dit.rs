//! HobbyImageDiT: the in-context flow-matching DiT (333M) that denoises the DC-AE latent, conditioned
//! on CLIP text tokens via cross-attention. Port of hobby_image/dit.py. Two-panel canvas (target|source)
//! with AdaLN-Zero self-attention + cross-attention + GELU-tanh MLP; outputs a velocity field.

use crate::imgops::{conv2d, gelu_tanh, linear, rmsnorm_rows, Map};
use crate::st::SafeTensors;
use rayon::prelude::*;
use std::path::Path;

pub struct DiTConfig {
    pub in_channels: usize, // 34
    pub out_channels: usize, // 32
    pub latent_h: usize,    // 32 -> 1024px
    pub panel_w: usize,     // 32
    pub d_model: usize,     // 1024
    pub depth: usize,       // 16
    pub heads: usize,       // 16
    pub ctx_dim: usize,     // 768
}

impl Default for DiTConfig {
    fn default() -> Self {
        DiTConfig { in_channels: 34, out_channels: 32, latent_h: 32, panel_w: 32,
                    d_model: 1024, depth: 16, heads: 16, ctx_dim: 768 }
    }
}

pub struct Dit {
    st: SafeTensors,
    pub cfg: DiTConfig,
}

/// LayerNorm with NO affine (ln1/ln2/ln_f): subtract mean, divide by std.
fn ln_noaffine(x: &[f32], n: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    out.par_chunks_mut(d).enumerate().for_each(|(i, o)| {
        let row = &x[i * d..(i + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            o[j] = (row[j] - mean) * inv;
        }
    });
    out
}

/// modulate: x * (1 + scale) + shift, broadcasting (d,) over n rows.
fn modulate(x: &[f32], n: usize, d: usize, shift: &[f32], scale: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            out[i * d + j] = x[i * d + j] * (1.0 + scale[j]) + shift[j];
        }
    }
    out
}

fn sinusoidal(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let a = t * f;
        out[i] = a.cos();
        out[half + i] = a.sin();
    }
    out
}

impl Dit {
    pub fn load(dir: &Path, cfg: DiTConfig) -> anyhow::Result<Dit> {
        let st = SafeTensors::open(&dir.join("dit.safetensors"))?;
        Ok(Dit { st, cfg })
    }

    /// Multi-head attention. q from `xq` (nq rows), k/v from `xkv` (nk rows). Per-head RMSNorm on
    /// q and k (weights qn/kn over head_dim) before the dot product. No causal mask (full attention).
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, prefix: &str, xq: &[f32], nq: usize, xkv: &[f32], nk: usize,
                 kv_in: usize) -> Vec<f32> {
        let d = self.cfg.d_model;
        let heads = self.cfg.heads;
        let hd = d / heads;
        let g = |n: &str| self.st.data(n);
        let q = linear(xq, nq, d, g(&format!("{prefix}.q.weight")), None, d);
        let kv = linear(xkv, nk, kv_in, g(&format!("{prefix}.kv.weight")), None, 2 * d);
        // split kv -> k (col 0..d), v (col d..2d), with the view (nk, 2, heads, hd)
        let mut k = vec![0.0f32; nk * d];
        let mut v = vec![0.0f32; nk * d];
        for i in 0..nk {
            k[i * d..(i + 1) * d].copy_from_slice(&kv[i * 2 * d..i * 2 * d + d]);
            v[i * d..(i + 1) * d].copy_from_slice(&kv[i * 2 * d + d..i * 2 * d + 2 * d]);
        }
        // per-head RMSNorm on q and k over head_dim (reshape rows to nq*heads x hd)
        let qn = g(&format!("{prefix}.qn.weight"));
        let kn = g(&format!("{prefix}.kn.weight"));
        let q = rmsnorm_rows(&q, nq * heads, hd, qn, 1e-6);
        let k = rmsnorm_rows(&k, nk * heads, hd, kn, 1e-6);
        let scale = 1.0 / (hd as f32).sqrt();
        // output (nq, d), parallel over (head, query row)
        let mut out = vec![0.0f32; nq * d];
        let cells: Vec<(usize, usize)> = (0..heads).flat_map(|h| (0..nq).map(move |i| (h, i))).collect();
        let results: Vec<(usize, usize, Vec<f32>)> = cells
            .par_iter()
            .map(|&(h, i)| {
                let off = h * hd;
                let qslice = &q[(i * heads + h) * hd..(i * heads + h) * hd + hd];
                let mut scores = vec![0.0f32; nk];
                for j in 0..nk {
                    let kslice = &k[(j * heads + h) * hd..(j * heads + h) * hd + hd];
                    scores[j] = crate::ops::dot(qslice, kslice) * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let mut o = vec![0.0f32; hd];
                // weighted sum of value rows (contiguous head slice) — vectorized axpy over hd.
                let nlanes = hd - hd % 8;
                for j in 0..nk {
                    let wj = scores[j] * inv;
                    let wv = wide::f32x8::splat(wj);
                    let vrow = &v[j * d + off..j * d + off + hd];
                    let mut t = 0;
                    while t < nlanes {
                        let ov = wide::f32x8::new(<[f32; 8]>::try_from(&o[t..t + 8]).unwrap());
                        let vv = wide::f32x8::new(<[f32; 8]>::try_from(&vrow[t..t + 8]).unwrap());
                        let r = wv.mul_add(vv, ov);
                        o[t..t + 8].copy_from_slice(r.as_array_ref());
                        t += 8;
                    }
                    while t < hd {
                        o[t] += wj * vrow[t];
                        t += 1;
                    }
                }
                (h, i, o)
            })
            .collect();
        for (h, i, o) in results {
            let off = h * hd;
            out[i * d + off..i * d + off + hd].copy_from_slice(&o);
        }
        linear(&out, nq, d, g(&format!("{prefix}.o.weight")), None, d)
    }

    /// Forward one denoising eval. `x` is the (34, H, 2*panel_w) two-panel input; `t` scalar in [0,1);
    /// `ctx` is the CLIP last_hidden_state (m x ctx_dim); `task` the task id. Returns the velocity
    /// field as a Map(out_channels, H, 2*panel_w) — caller slices the left panel.
    pub fn forward(&self, x: &Map, t: f32, ctx: &[f32], m: usize, task: usize) -> Map {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let h_lat = cfg.latent_h;
        let w = 2 * cfg.panel_w;
        let n = h_lat * w; // tokens
        let g = |nm: &str| self.st.data(nm);

        // patch embed (1x1 conv) -> (d, H, W); transpose to tokens (n, d) with t = y*W + x
        let pe = conv2d(x, g("patch_embed.weight"), Some(g("patch_embed.bias")), d, 1, 0, 1);
        let mut h = vec![0.0f32; n * d];
        for c in 0..d {
            for y in 0..h_lat {
                for xx in 0..w {
                    h[(y * w + xx) * d + c] = pe.d[(c * h_lat + y) * w + xx];
                }
            }
        }
        // + positional embedding (1, n, d)
        let pos = g("pos");
        for i in 0..n * d {
            h[i] += pos[i];
        }
        // + panel embedding: left half (col < panel_w) vs right
        let panel = g("panel"); // (2, d)
        for tkn in 0..n {
            let col = tkn % w;
            let p = if col < cfg.panel_w { 0 } else { 1 };
            for c in 0..d {
                h[tkn * d + c] += panel[p * d + c];
            }
        }
        // conditioning vector: t_mlp(sinusoidal(t*1000)) + task_emb(task)
        let sin = sinusoidal(t * 1000.0, d);
        let mut cond = linear(&sin, 1, d, g("t_mlp.0.weight"), Some(g("t_mlp.0.bias")), d);
        for v in cond.iter_mut() {
            *v = crate::imgops::silu(*v);
        }
        cond = linear(&cond, 1, d, g("t_mlp.2.weight"), Some(g("t_mlp.2.bias")), d);
        let task_emb = g("task_emb.weight");
        for c in 0..d {
            cond[c] += task_emb[task * d + c];
        }

        for l in 0..cfg.depth {
            let p = format!("blocks.{l}");
            // adaLN: SiLU(cond) then Linear -> 6d
            let mut cs = cond.clone();
            for v in cs.iter_mut() {
                *v = crate::imgops::silu(*v);
            }
            let mod6 = linear(&cs, 1, d, g(&format!("{p}.adaln.1.weight")),
                              Some(g(&format!("{p}.adaln.1.bias"))), 6 * d);
            let (sa_s, sa_sc, sa_g) = (&mod6[0..d], &mod6[d..2 * d], &mod6[2 * d..3 * d]);
            let (ml_s, ml_sc, ml_g) = (&mod6[3 * d..4 * d], &mod6[4 * d..5 * d], &mod6[5 * d..6 * d]);

            // self-attention with AdaLN-Zero gate
            let ln1 = ln_noaffine(&h, n, d, 1e-6);
            let modx = modulate(&ln1, n, d, sa_s, sa_sc);
            let sa = self.attention(&format!("{p}.sa"), &modx, n, &modx, n, d);
            for i in 0..n {
                for c in 0..d {
                    h[i * d + c] += sa_g[c] * sa[i * d + c];
                }
            }
            // cross-attention to ctx (lnc is affine LayerNorm), no gate
            let lnc = crate::imgops::layernorm(&h, n, d, g(&format!("{p}.lnc.weight")),
                                               g(&format!("{p}.lnc.bias")), 1e-6);
            let ca = self.attention(&format!("{p}.ca"), &lnc, n, ctx, m, cfg.ctx_dim);
            for i in 0..n * d {
                h[i] += ca[i];
            }
            // MLP with AdaLN-Zero gate
            let ln2 = ln_noaffine(&h, n, d, 1e-6);
            let modx = modulate(&ln2, n, d, ml_s, ml_sc);
            let hidden = g(&format!("{p}.mlp.0.weight")).len() / d;
            let mut f1 = linear(&modx, n, d, g(&format!("{p}.mlp.0.weight")),
                                Some(g(&format!("{p}.mlp.0.bias"))), hidden);
            for v in f1.iter_mut() {
                *v = gelu_tanh(*v);
            }
            let f2 = linear(&f1, n, hidden, g(&format!("{p}.mlp.2.weight")),
                            Some(g(&format!("{p}.mlp.2.bias"))), d);
            for i in 0..n {
                for c in 0..d {
                    h[i * d + c] += ml_g[c] * f2[i * d + c];
                }
            }
        }
        // final AdaLN (2d) + head
        let mut cs = cond.clone();
        for v in cs.iter_mut() {
            *v = crate::imgops::silu(*v);
        }
        let mod2 = linear(&cs, 1, d, g("adaln_f.1.weight"), Some(g("adaln_f.1.bias")), 2 * d);
        let lnf = ln_noaffine(&h, n, d, 1e-6);
        let hmod = modulate(&lnf, n, d, &mod2[0..d], &mod2[d..2 * d]);
        let out = linear(&hmod, n, d, g("head.weight"), Some(g("head.bias")), cfg.out_channels);
        // unpatchify (n, out_ch) -> Map(out_ch, H, W), token t = y*W + x
        let oc = cfg.out_channels;
        let mut map = Map::zeros(oc, h_lat, w);
        for y in 0..h_lat {
            for xx in 0..w {
                let tkn = y * w + xx;
                for c in 0..oc {
                    map.d[(c * h_lat + y) * w + xx] = out[tkn * oc + c];
                }
            }
        }
        map
    }
}
