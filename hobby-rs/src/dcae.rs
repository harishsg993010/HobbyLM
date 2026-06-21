//! DC-AE f32c32 (Sana 1.1) decoder: latent (32, 32, 32) -> RGB (3, 1024, 1024). Faithful port of
//! diffusers AutoencoderDC.Decoder. High-res stages are pure conv ResBlocks; low-res stages use
//! Sana's EfficientViT block (multiscale ReLU-kernel linear attention + GLU-MBConv). Upsampling is
//! "interpolate" (nearest x2 + 3x3 conv) with a pixel-shuffle channel shortcut.

use crate::imgops::{
    conv2d, pixel_shuffle, relu, repeat_interleave_ch, rmsnorm_ch, silu, upsample_nearest, Map,
};
use crate::st::SafeTensors;

const HEAD_DIM: usize = 32;
const EPS: f32 = 1e-5;

pub struct Dcae<'a> {
    st: &'a SafeTensors,
}

impl<'a> Dcae<'a> {
    pub fn new(st: &'a SafeTensors) -> Self {
        Dcae { st }
    }

    fn g(&self, name: &str) -> &[f32] {
        self.st.data(name)
    }

    /// 3x3 (or kxk) conv from `decoder.<p>.weight` (+ optional bias). `groups` for depthwise/grouped.
    fn conv(&self, p: &str, x: &Map, cout: usize, k: usize, pad: usize, groups: usize, bias: bool) -> Map {
        let w = self.g(&format!("decoder.{p}.weight"));
        let b = if bias { Some(self.g(&format!("decoder.{p}.bias"))) } else { None };
        conv2d(x, w, b, cout, k, pad, groups)
    }

    fn rmsn(&self, p: &str, x: &Map) -> Map {
        rmsnorm_ch(x, self.g(&format!("decoder.{p}.weight")), Some(self.g(&format!("decoder.{p}.bias"))), EPS)
    }

    /// ResBlock: residual + rms(conv2(silu(conv1(x)))). conv2 has no bias.
    fn resblock(&self, p: &str, x: &Map) -> Map {
        let c = x.c;
        let mut h = self.conv(&format!("{p}.conv1"), x, c, 3, 1, 1, true);
        for v in h.d.iter_mut() {
            *v = silu(*v);
        }
        let mut h = self.conv(&format!("{p}.conv2"), &h, c, 3, 1, 1, false);
        h = self.rmsn(&format!("{p}.norm"), &h);
        for i in 0..h.d.len() {
            h.d[i] += x.d[i];
        }
        h
    }

    /// DCUpBlock2d (interpolate=True, shortcut=True): x = conv(nearest_up(h)); y = pixel_shuffle(
    /// repeat_interleave(h, out*4/in)); return x + y.
    fn upblock(&self, p: &str, x: &Map, cout: usize) -> Map {
        let cin = x.c;
        let up = upsample_nearest(x, 2);
        let xc = self.conv(&format!("{p}.conv"), &up, cout, 3, 1, 1, true);
        let reps = cout * 4 / cin;
        let y = pixel_shuffle(&repeat_interleave_ch(x, reps), 2);
        debug_assert_eq!((xc.c, xc.h, xc.w), (y.c, y.h, y.w));
        let mut out = xc;
        for i in 0..out.d.len() {
            out.d[i] += y.d[i];
        }
        out
    }

    /// Sana multiscale linear attention (use_linear_attention path; H*W >> head_dim always here).
    fn sana_attn(&self, p: &str, x: &Map) -> Map {
        let (c, h, w) = (x.c, x.h, x.w);
        let hw = h * w;
        let nh = c / HEAD_DIM; // num heads
        let inner = c; // num_heads * head_dim
        // q,k,v via Linear over channel (per pixel). weights (inner, c).
        let to_tok = |m: &Map| -> Vec<f32> {
            // (c, h, w) -> (hw, c)
            let mut t = vec![0.0f32; hw * m.c];
            for ch in 0..m.c {
                for s in 0..hw {
                    t[s * m.c + ch] = m.d[ch * hw + s];
                }
            }
            t
        };
        let xt = to_tok(x);
        let q = crate::imgops::linear(&xt, hw, c, self.g(&format!("decoder.{p}.to_q.weight")), None, inner);
        let k = crate::imgops::linear(&xt, hw, c, self.g(&format!("decoder.{p}.to_k.weight")), None, inner);
        let v = crate::imgops::linear(&xt, hw, c, self.g(&format!("decoder.{p}.to_v.weight")), None, inner);
        // qkv0 channel-major map (3*inner, h, w): [q | k | v]
        let mut qkv0 = Map::zeros(3 * inner, h, w);
        for s in 0..hw {
            for ch in 0..inner {
                qkv0.d[ch * hw + s] = q[s * inner + ch];
                qkv0.d[(inner + ch) * hw + s] = k[s * inner + ch];
                qkv0.d[(2 * inner + ch) * hw + s] = v[s * inner + ch];
            }
        }
        // multiscale projection: depthwise 5x5 (groups=3*inner) then 1x1 grouped (groups=3*nh)
        let mp = format!("{p}.to_qkv_multiscale.0");
        let ms_in = self.conv(&format!("{mp}.proj_in"), &qkv0, 3 * inner, 5, 2, 3 * inner, false);
        let ms = self.conv(&format!("{mp}.proj_out"), &ms_in, 3 * inner, 1, 0, 3 * nh, false);
        // cat([qkv0, ms], channel) -> (6*inner, h, w)
        let mut cat = Map::zeros(6 * inner, h, w);
        cat.d[..3 * inner * hw].copy_from_slice(&qkv0.d);
        cat.d[3 * inner * hw..].copy_from_slice(&ms.d);
        // groups G = 2*nh; per group g, channels [g*96 .. g*96+96) = [q_hd | k_hd | v_hd]
        let g_groups = 2 * nh;
        // output (2*inner, h, w)
        let mut out = Map::zeros(2 * inner, h, w);
        for gi in 0..g_groups {
            let base = gi * 3 * HEAD_DIM;
            // gather q,k,v for this group: each (HEAD_DIM, hw); apply relu to q,k
            // linear attention: value_pad (hd+1, hw); scores = value_pad @ key^T (hd+1, hd);
            // o = scores @ query (hd+1, hw); normalize o[:hd]/(o[hd]+eps)
            let mut kk = vec![0.0f32; HEAD_DIM * hw];
            let mut qq = vec![0.0f32; HEAD_DIM * hw];
            let mut vv = vec![0.0f32; HEAD_DIM * hw];
            for t in 0..HEAD_DIM {
                let qrow = &cat.d[(base + t) * hw..(base + t + 1) * hw];
                let krow = &cat.d[(base + HEAD_DIM + t) * hw..(base + HEAD_DIM + t + 1) * hw];
                let vrow = &cat.d[(base + 2 * HEAD_DIM + t) * hw..(base + 2 * HEAD_DIM + t + 1) * hw];
                for s in 0..hw {
                    qq[t * hw + s] = relu(qrow[s]);
                    kk[t * hw + s] = relu(krow[s]);
                    vv[t * hw + s] = vrow[s];
                }
            }
            // scores[a in 0..hd+1][b in 0..hd] = sum_s value_pad[a][s]*key[b][s]
            // value_pad[hd] = ones.
            let hdp = HEAD_DIM + 1;
            let mut scores = vec![0.0f32; hdp * HEAD_DIM];
            for a in 0..hdp {
                for b in 0..HEAD_DIM {
                    let mut acc = 0.0f32;
                    if a < HEAD_DIM {
                        let vr = &vv[a * hw..(a + 1) * hw];
                        let kr = &kk[b * hw..(b + 1) * hw];
                        acc = crate::ops::dot(vr, kr);
                    } else {
                        // value_pad row of ones -> sum of key[b]
                        let kr = &kk[b * hw..(b + 1) * hw];
                        acc = kr.iter().sum();
                    }
                    scores[a * HEAD_DIM + b] = acc;
                }
            }
            // o[a][s] = sum_b scores[a][b] * query[b][s]
            let mut o = vec![0.0f32; hdp * hw];
            for a in 0..hdp {
                for b in 0..HEAD_DIM {
                    let sc = scores[a * HEAD_DIM + b];
                    if sc == 0.0 {
                        continue;
                    }
                    let qr = &qq[b * hw..(b + 1) * hw];
                    let orow = &mut o[a * hw..(a + 1) * hw];
                    for s in 0..hw {
                        orow[s] += sc * qr[s];
                    }
                }
            }
            // normalize and write to out channels [gi*hd .. gi*hd+hd)
            for t in 0..HEAD_DIM {
                for s in 0..hw {
                    let denom = o[HEAD_DIM * hw + s] + 1e-15;
                    out.d[(gi * HEAD_DIM + t) * hw + s] = o[t * hw + s] / denom;
                }
            }
        }
        // to_out: Linear(2*inner -> c) per pixel
        let ot = to_tok(&out);
        let proj = crate::imgops::linear(&ot, hw, 2 * inner, self.g(&format!("decoder.{p}.to_out.weight")), None, c);
        let mut res = Map::zeros(c, h, w);
        for s in 0..hw {
            for ch in 0..c {
                res.d[ch * hw + s] = proj[s * c + ch];
            }
        }
        let res = self.rmsn(&format!("{p}.norm_out"), &res);
        // residual
        let mut out2 = res;
        for i in 0..out2.d.len() {
            out2.d[i] += x.d[i];
        }
        out2
    }

    /// GLUMBConv: residual + rms(conv_point(h * silu(gate))), h/gate from conv_depth(silu(conv_inverted)).
    fn glumbconv(&self, p: &str, x: &Map) -> Map {
        let c = x.c;
        let hidden = 4 * c;
        let mut inv = self.conv(&format!("{p}.conv_inverted"), x, 2 * hidden, 1, 0, 1, true);
        for v in inv.d.iter_mut() {
            *v = silu(*v);
        }
        let dep = self.conv(&format!("{p}.conv_depth"), &inv, 2 * hidden, 3, 1, 2 * hidden, true);
        // chunk(2, dim=channel): h = dep[:hidden], gate = dep[hidden:]
        let (h, w) = (dep.h, dep.w);
        let hw = h * w;
        let mut gated = Map::zeros(hidden, h, w);
        for ch in 0..hidden {
            for s in 0..hw {
                let hv = dep.d[ch * hw + s];
                let gv = dep.d[(hidden + ch) * hw + s];
                gated.d[ch * hw + s] = hv * silu(gv);
            }
        }
        let point = self.conv(&format!("{p}.conv_point"), &gated, c, 1, 0, 1, false);
        let mut normed = self.rmsn(&format!("{p}.norm"), &point);
        for i in 0..normed.d.len() {
            normed.d[i] += x.d[i];
        }
        normed
    }

    fn efficientvit(&self, p: &str, x: &Map) -> Map {
        let a = self.sana_attn(&format!("{p}.attn"), x);
        self.glumbconv(&format!("{p}.conv_out"), &a)
    }

    /// Full decode. `latent` is the DC-AE latent Map(32, 32, 32) (already denormalized: z*lat_std/sf).
    pub fn decode(&self, latent: &Map) -> Map {
        // conv_in with in_shortcut: h = conv_in(latent) + repeat_interleave(latent, 1024/32)
        let cin0 = 1024usize;
        let mut h = self.conv("conv_in", latent, cin0, 3, 1, 1, true);
        let sc = repeat_interleave_ch(latent, cin0 / latent.c);
        for i in 0..h.d.len() {
            h.d[i] += sc.d[i];
        }
        // block out channels and layer counts per stage index i; forward order = reversed(5..0)
        // up_blocks[i].j keys: j=0 is DCUpBlock for i<5, then 3 type-blocks at j (1..3) or (0..2 for i=5).
        let out_ch = [128usize, 256, 512, 512, 1024, 1024];
        let is_vit = [false, false, false, true, true, true];
        for i in (0..6).rev() {
            let mut j0 = 0usize;
            if i < 5 {
                // upsample from out_ch[i+1] -> out_ch[i]
                h = self.upblock(&format!("up_blocks.{i}.0"), &h, out_ch[i]);
                j0 = 1;
            }
            for blk in 0..3 {
                let p = format!("up_blocks.{i}.{}", j0 + blk);
                h = if is_vit[i] {
                    self.efficientvit(&p, &h)
                } else {
                    self.resblock(&p, &h)
                };
            }
        }
        // norm_out (rms over channel) -> relu -> conv_out
        h = self.rmsn("norm_out", &h);
        for v in h.d.iter_mut() {
            *v = relu(*v);
        }
        self.conv("conv_out", &h, 3, 3, 1, 1, true)
    }
}
