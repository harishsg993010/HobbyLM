//! f32 image/diffusion kernels: grouped conv2d, channel rearrange (pixel-shuffle, nearest
//! upsample, repeat-interleave), channel-wise RMSNorm, LayerNorm, and activations. All tensors are
//! plain `Vec<f32>` in (C, H, W) row-major (channel-major) layout unless noted; transformer-style
//! ops act on (N, C) token layout. Correctness-first; conv parallelizes over output channels.

use rayon::prelude::*;

/// A CHW feature map.
#[derive(Clone)]
pub struct Map {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub d: Vec<f32>, // length c*h*w, index [(ch*h + y)*w + x]
}

impl Map {
    pub fn zeros(c: usize, h: usize, w: usize) -> Map {
        Map { c, h, w, d: vec![0.0; c * h * w] }
    }
    pub fn from(c: usize, h: usize, w: usize, d: Vec<f32>) -> Map {
        debug_assert_eq!(d.len(), c * h * w);
        Map { c, h, w, d }
    }
    #[inline]
    pub fn hw(&self) -> usize {
        self.h * self.w
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
#[inline]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}
/// tanh-approx GELU (DiT MLP uses approximate="tanh").
#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}
/// CLIP's quick_gelu = x * sigmoid(1.702 x).
#[inline]
pub fn quick_gelu(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

/// General grouped 2-D convolution. stride is always 1 here (the DC-AE decoder upsamples via
/// interpolate, not strided conv). weight is (cout, cin/groups, kh, kw) row-major; bias optional.
/// Output is (cout, oh, ow) with oh = h + 2*pad - kh + 1.
pub fn conv2d(x: &Map, w: &[f32], bias: Option<&[f32]>, cout: usize, k: usize, pad: usize,
              groups: usize) -> Map {
    use wide::f32x8;
    let (cin, h, ww) = (x.c, x.h, x.w);
    let oh = h + 2 * pad - k + 1;
    let ow = ww + 2 * pad - k + 1;
    let cin_g = cin / groups;
    let cout_g = cout / groups;
    let mut out = vec![0.0f32; cout * oh * ow];
    // Parallel over output channels. For each (input-channel, ky, kx) tap, add a vectorized,
    // contiguous axpy `out_row += w_val * in_row_shifted` over the columns where the (padded)
    // input column is in range — turning the conv into SIMD spatial axpys instead of per-pixel
    // scalar reductions.
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(oc, o)| {
        let g = oc / cout_g;
        let b = bias.map(|bb| bb[oc]).unwrap_or(0.0);
        for v in o.iter_mut() {
            *v = b;
        }
        let wbase = oc * cin_g * k * k;
        for ic in 0..cin_g {
            let xc = g * cin_g + ic;
            let xch = &x.d[xc * h * ww..(xc + 1) * h * ww];
            let wc = &w[wbase + ic * k * k..wbase + (ic + 1) * k * k];
            for ky in 0..k {
                for kx in 0..k {
                    let wv = wc[ky * k + kx];
                    if wv == 0.0 {
                        continue;
                    }
                    let wvec = f32x8::splat(wv);
                    // output rows oy with valid input row iy = oy + ky - pad in [0,h)
                    for oy in 0..oh {
                        let iy = oy + ky;
                        if iy < pad || iy >= h + pad {
                            continue;
                        }
                        let yy = iy - pad;
                        // output cols ox with ix = ox + kx - pad in [0, ww)
                        let ox0 = pad.saturating_sub(kx);
                        let ox1 = (ww + pad).saturating_sub(kx).min(ow);
                        if ox0 >= ox1 {
                            continue;
                        }
                        let orow = &mut o[oy * ow + ox0..oy * ow + ox1];
                        let irow = &xch[yy * ww + (ox0 + kx - pad)..yy * ww + (ox1 + kx - pad)];
                        let n = orow.len();
                        let mut i = 0;
                        while i + 8 <= n {
                            let ov = f32x8::new(<[f32; 8]>::try_from(&orow[i..i + 8]).unwrap());
                            let iv = f32x8::new(<[f32; 8]>::try_from(&irow[i..i + 8]).unwrap());
                            let r = wvec.mul_add(iv, ov);
                            orow[i..i + 8].copy_from_slice(r.as_array_ref());
                            i += 8;
                        }
                        while i < n {
                            orow[i] += wv * irow[i];
                            i += 1;
                        }
                    }
                }
            }
        }
    });
    Map::from(cout, oh, ow, out)
}

/// Nearest-neighbour upsample by integer factor `f` (matches F.interpolate mode="nearest").
pub fn upsample_nearest(x: &Map, f: usize) -> Map {
    let (c, h, w) = (x.c, x.h, x.w);
    let (oh, ow) = (h * f, w * f);
    let mut out = vec![0.0f32; c * oh * ow];
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(ch, o)| {
        let src = &x.d[ch * h * w..(ch + 1) * h * w];
        for oy in 0..oh {
            let sy = oy / f;
            for ox in 0..ow {
                o[oy * ow + ox] = src[sy * w + ox / f];
            }
        }
    });
    Map::from(c, oh, ow, out)
}

/// PixelShuffle by factor r: (C*r*r, H, W) -> (C, H*r, W*r). Matches torch.nn.functional.pixel_shuffle:
/// input channel index = c*r*r + (sy*r + sx); output[c, y*r+sy, x*r+sx] = input[c*r*r+sy*r+sx, y, x].
pub fn pixel_shuffle(x: &Map, r: usize) -> Map {
    let (c_in, h, w) = (x.c, x.h, x.w);
    let c = c_in / (r * r);
    let (oh, ow) = (h * r, w * r);
    let mut out = vec![0.0f32; c * oh * ow];
    for co in 0..c {
        for sy in 0..r {
            for sx in 0..r {
                let ci = co * r * r + sy * r + sx;
                let src = &x.d[ci * h * w..(ci + 1) * h * w];
                for y in 0..h {
                    for xx in 0..w {
                        out[(co * oh + (y * r + sy)) * ow + (xx * r + sx)] = src[y * w + xx];
                    }
                }
            }
        }
    }
    Map::from(c, oh, ow, out)
}

/// repeat_interleave along the channel dim: each channel repeated `reps` times consecutively.
pub fn repeat_interleave_ch(x: &Map, reps: usize) -> Map {
    let (c, h, w) = (x.c, x.h, x.w);
    let mut out = vec![0.0f32; c * reps * h * w];
    for ci in 0..c {
        let src = &x.d[ci * h * w..(ci + 1) * h * w];
        for r in 0..reps {
            let dst = (ci * reps + r) * h * w;
            out[dst..dst + h * w].copy_from_slice(src);
        }
    }
    Map::from(c * reps, h, w, out)
}

/// Channel-wise RMSNorm (normalize across C at each spatial location), with weight (+ optional bias).
/// Matches diffusers RMSNorm applied via movedim(1,-1): variance over the channel axis.
pub fn rmsnorm_ch(x: &Map, w: &[f32], bias: Option<&[f32]>, eps: f32) -> Map {
    let (c, h, ww) = (x.c, x.h, x.w);
    let hw = h * ww;
    // pass 1: per-pixel inverse rms across channels (parallel over pixel chunks; strided reads).
    let mut scale = vec![0.0f32; hw];
    let chunk = (hw / rayon::current_num_threads().max(1)).max(1);
    scale.par_chunks_mut(chunk).enumerate().for_each(|(ci, sl)| {
        let p0 = ci * chunk;
        for (i, s) in sl.iter_mut().enumerate() {
            let p = p0 + i;
            let mut ss = 0.0f32;
            for ch in 0..c {
                let v = x.d[ch * hw + p];
                ss += v * v;
            }
            *s = 1.0 / (ss / c as f32 + eps).sqrt();
        }
    });
    // pass 2: apply per channel (contiguous; parallel over channels).
    let mut out = vec![0.0f32; c * hw];
    out.par_chunks_mut(hw).enumerate().for_each(|(ch, o)| {
        let wc = w[ch];
        let b = bias.map(|bb| bb[ch]).unwrap_or(0.0);
        let xc = &x.d[ch * hw..(ch + 1) * hw];
        for p in 0..hw {
            o[p] = xc[p] * scale[p] * wc + b;
        }
    });
    Map::from(c, h, ww, out)
}

/// LayerNorm over the last dim of an (n, dim) token buffer (CLIP / DiT). weight+bias length dim.
pub fn layernorm(x: &[f32], n: usize, dim: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(i, o)| {
        let row = &x[i * dim..(i + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..dim {
            o[j] = (row[j] - mean) * inv * w[j] + b[j];
        }
    });
    out
}

/// RMSNorm over the last dim of an (n, dim) token buffer, weight only (DiT q/k norm style).
pub fn rmsnorm_rows(x: &[f32], n: usize, dim: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(i, o)| {
        let row = &x[i * dim..(i + 1) * dim];
        let ss = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for j in 0..dim {
            o[j] = row[j] * inv * w[j];
        }
    });
    out
}

/// y (n, out) = x (n, in) @ W^T (W is (out,in) row-major) + optional bias. Reuses the SIMD dot.
pub fn linear(x: &[f32], n: usize, in_: usize, w: &[f32], bias: Option<&[f32]>, out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * out];
    y.par_chunks_mut(out).enumerate().for_each(|(t, yr)| {
        let xr = &x[t * in_..(t + 1) * in_];
        for o in 0..out {
            let mut v = crate::ops::dot(&w[o * in_..(o + 1) * in_], xr);
            if let Some(b) = bias {
                v += b[o];
            }
            yr[o] = v;
        }
    });
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_identity_1x1() {
        // 2 in -> 1 out, 1x1 conv summing channels; on a 2x2 map.
        let x = Map::from(2, 2, 2, vec![1., 2., 3., 4., 10., 20., 30., 40.]);
        let w = vec![1.0, 1.0]; // (1,2,1,1)
        let y = conv2d(&x, &w, None, 1, 1, 0, 1);
        assert_eq!(y.d, vec![11., 22., 33., 44.]);
    }

    #[test]
    fn conv_3x3_pad1_box() {
        // single channel, 3x3 all-ones kernel, pad 1 -> box blur sums of neighbourhoods.
        let x = Map::from(1, 3, 3, vec![1., 1., 1., 1., 1., 1., 1., 1., 1.]);
        let w = vec![1.0; 9];
        let y = conv2d(&x, &w, None, 1, 3, 1, 1);
        // corner sees 4 ones, edge 6, center 9
        assert_eq!(y.d, vec![4., 6., 4., 6., 9., 6., 4., 6., 4.]);
    }

    #[test]
    fn pixel_shuffle_basic() {
        // C*r*r=4, r=2, 1x1 spatial -> 1ch 2x2. channels [a,b,c,d] -> [[a,b],[c,d]]
        let x = Map::from(4, 1, 1, vec![1., 2., 3., 4.]);
        let y = pixel_shuffle(&x, 2);
        assert_eq!((y.c, y.h, y.w), (1, 2, 2));
        assert_eq!(y.d, vec![1., 2., 3., 4.]);
    }

    #[test]
    fn upsample_nearest_2x() {
        let x = Map::from(1, 1, 2, vec![5., 9.]);
        let y = upsample_nearest(&x, 2);
        assert_eq!((y.h, y.w), (2, 4));
        assert_eq!(y.d, vec![5., 5., 9., 9., 5., 5., 9., 9.]);
    }

    #[test]
    fn repeat_interleave_basic() {
        let x = Map::from(2, 1, 1, vec![1., 2.]);
        let y = repeat_interleave_ch(&x, 2);
        assert_eq!(y.c, 4);
        assert_eq!(y.d, vec![1., 1., 2., 2.]);
    }

    #[test]
    fn rmsnorm_ch_unit() {
        // 2 channels, 1 pixel, weights 1 -> normalized so rms=1
        let x = Map::from(2, 1, 1, vec![3.0, 4.0]);
        let y = rmsnorm_ch(&x, &[1.0, 1.0], None, 0.0);
        // rms = sqrt((9+16)/2)=sqrt(12.5); 3/that, 4/that
        let s = (12.5f32).sqrt();
        assert!((y.d[0] - 3.0 / s).abs() < 1e-6);
        assert!((y.d[1] - 4.0 / s).abs() < 1e-6);
    }
}
