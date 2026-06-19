//! Q8_0 weight quantization (ggml-style: blocks of 32 -> one f32 scale + 32 int8) and a
//! `Mat` abstraction that lets every weight be either F32 (zero-copy mmap) or Q8 (owned),
//! dispatched by a single `matvec`. Q8_0 is near-lossless and ~3.8x smaller in RAM, so the
//! bandwidth-bound matmul runs faster.

use crate::ops::{self, dot};
use rayon::prelude::*;

const QK: usize = 32; // block size

/// Block-int8 dot of one weight row vs the quantized activation: returns sum_b scale_w[b]*scale_x[b]*
/// (sum_j qw[j]*qa[j]). On AVX2 this uses the ggml maddubs trick (abs(w)·(a·sign(w))) for a real
/// signed int8 dot; otherwise a 4-accumulator scalar fallback. `qw`/`qa` len = nb*QK.
#[inline]
fn qrow_dot(qw: &[i8], qa: &[i8], sw: &[f32], dx: &[f32], nb: usize) -> f32 {
    // When the crate is built with AVX2 (target-cpu=native / +avx2), use the inlinable intrinsic path.
    // No #[target_feature] attr -> the function INLINES into the rayon row-loop (no per-row call cost).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe { qrow_dot_avx2(qw, qa, sw, dx, nb) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
    let mut acc = 0.0f32;
    for b in 0..nb {
        let (w, a) = (&qw[b * QK..b * QK + QK], &qa[b * QK..b * QK + QK]);
        let (mut s0, mut s1, mut s2, mut s3) = (0i32, 0i32, 0i32, 0i32);
        let mut j = 0;
        while j < QK {
            s0 += w[j] as i32 * a[j] as i32;
            s1 += w[j + 1] as i32 * a[j + 1] as i32;
            s2 += w[j + 2] as i32 * a[j + 2] as i32;
            s3 += w[j + 3] as i32 * a[j + 3] as i32;
            j += 4;
        }
        acc += sw[b] * dx[b] * (s0 + s1 + s2 + s3) as f32;
    }
    acc
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn qrow_dot_avx2(qw: &[i8], qa: &[i8], sw: &[f32], dx: &[f32], nb: usize) -> f32 {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    // Keep the 8 per-block partial sums as FLOAT lanes; scale + FMA-accumulate across blocks; hsum ONCE
    // at the end (the ggml trick — avoids a per-block integer horizontal sum, which is the slow part).
    let mut accf = _mm256_setzero_ps();
    for b in 0..nb {
        let w = _mm256_loadu_si256(qw.as_ptr().add(b * QK) as *const __m256i);
        let a = _mm256_loadu_si256(qa.as_ptr().add(b * QK) as *const __m256i);
        let axw = _mm256_sign_epi8(w, w); // |w|
        let say = _mm256_sign_epi8(a, w); // a * sign(w)  -> signed*signed dot via maddubs
        let p16 = _mm256_maddubs_epi16(axw, say); // 16x i16
        let p32 = _mm256_madd_epi16(p16, ones); // 8x i32 partial sums of this block
        let d = _mm256_set1_ps(sw[b] * dx[b]);
        accf = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p32), accf);
    }
    // horizontal sum of 8 f32
    let lo = _mm256_castps256_ps128(accf);
    let hi = _mm256_extractf128_ps(accf, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0b01));
    _mm_cvtss_f32(s)
}

pub struct Q8Tensor {
    pub out: usize,
    pub in_: usize,
    nb: usize,         // blocks per row = in_/QK
    scales: Vec<f32>,  // [out * nb]
    qs: Vec<i8>,       // [out * in_]
}

impl Q8Tensor {
    pub fn quantize(w: &[f32], out: usize, in_: usize) -> Self {
        assert_eq!(in_ % QK, 0, "Q8_0 needs in_ divisible by 32 (got {in_})");
        assert_eq!(w.len(), out * in_);
        let nb = in_ / QK;
        let mut scales = vec![0.0f32; out * nb];
        let mut qs = vec![0i8; out * in_];
        // quantize row-blocks in parallel
        qs.par_chunks_mut(in_)
            .zip(scales.par_chunks_mut(nb))
            .enumerate()
            .for_each(|(o, (qrow, srow))| {
                let wrow = &w[o * in_..o * in_ + in_];
                for b in 0..nb {
                    let blk = &wrow[b * QK..b * QK + QK];
                    let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let d = amax / 127.0;
                    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                    srow[b] = d;
                    for j in 0..QK {
                        qrow[b * QK + j] = (blk[j] * id).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            });
        Q8Tensor { out, in_, nb, scales, qs }
    }

    /// int8 x int8 -> i32 dot, with weight & activation block scales applied at block end.
    /// The activation is quantized to int8 once (shared across all output rows).
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_);
        let in_ = self.in_;
        let nb = self.nb;

        // quantize activation x into per-block int8 + scale (once)
        let mut qx = vec![0i8; in_];
        let mut dx = vec![0.0f32; nb];
        for b in 0..nb {
            let blk = &x[b * QK..b * QK + QK];
            let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            dx[b] = d;
            for j in 0..QK {
                qx[b * QK + j] = (blk[j] * id).round().clamp(-127.0, 127.0) as i8;
            }
        }

        let work = |o: usize, yo: &mut f32| {
            *yo = qrow_dot(&self.qs[o * in_..o * in_ + in_], &qx,
                           &self.scales[o * nb..o * nb + nb], &dx, nb);
        };
        if y.len() >= 64 {
            y.par_iter_mut().enumerate().for_each(|(o, yo)| work(o, yo));
        } else {
            for (o, yo) in y.iter_mut().enumerate() {
                work(o, yo);
            }
        }
    }

    pub fn bytes(&self) -> usize {
        self.scales.len() * 4 + self.qs.len()
    }

    /// Batched int8 matmul: x is (tn, in), y is (tn, out). Each quantized weight row is read
    /// once and reused across all tn rows (which stay in cache) — the prefill speedup.
    pub fn matmul(&self, x: &[f32], y: &mut [f32], tn: usize) {
        let in_ = self.in_;
        let out = self.out;
        let nb = self.nb;
        // quantize all tn activation rows to int8 blocks
        let mut qx = vec![0i8; tn * in_];
        let mut dx = vec![0.0f32; tn * nb];
        qx.par_chunks_mut(in_)
            .zip(dx.par_chunks_mut(nb))
            .zip(x.par_chunks(in_))
            .for_each(|((qr, dr), xr)| {
                for b in 0..nb {
                    let blk = &xr[b * QK..b * QK + QK];
                    let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let d = amax / 127.0;
                    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                    dr[b] = d;
                    for j in 0..QK {
                        qr[b * QK + j] = (blk[j] * id).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            });
        // Yt (out, tn): parallel over output cols, weight row read once
        let mut yt = vec![0.0f32; out * tn];
        yt.par_chunks_mut(tn).enumerate().for_each(|(o, ytr)| {
            let qw = &self.qs[o * in_..o * in_ + in_];
            let sw = &self.scales[o * nb..o * nb + nb];
            for t in 0..tn {
                let qxr = &qx[t * in_..t * in_ + in_];
                let dxr = &dx[t * nb..t * nb + nb];
                let mut acc = 0.0f32;
                for b in 0..nb {
                    let qwb = &qw[b * QK..b * QK + QK];
                    let qxb = &qxr[b * QK..b * QK + QK];
                    let mut isum: i32 = 0;
                    for j in 0..QK {
                        isum += qwb[j] as i32 * qxb[j] as i32;
                    }
                    acc += sw[b] * dxr[b] * isum as f32;
                }
                ytr[t] = acc;
            }
        });
        for o in 0..out {
            let row = &yt[o * tn..o * tn + tn];
            for t in 0..tn {
                y[t * out + o] = row[t];
            }
        }
    }
}

/// A weight matrix: owned F32 or owned Q8_0 (no borrow — the Model owns all its data so the
/// source GGUF mmap can be dropped after load, making the engine a self-contained value).
pub enum Mat {
    Owned { w: Vec<f32>, out: usize, in_: usize },
    Q8(Q8Tensor),
}

impl Mat {
    /// From an f32 slice: quantize to Q8 or copy to an owned f32 buffer.
    pub fn of(w: &[f32], out: usize, in_: usize, quant: bool) -> Self {
        if quant {
            Mat::Q8(Q8Tensor::quantize(w, out, in_))
        } else {
            Mat::Owned { w: w.to_vec(), out, in_ }
        }
    }
    /// From a (possibly dequantized) tensor source, avoiding a copy when it's already owned f32.
    pub fn build(src: crate::gguf::Src<'_>, out: usize, in_: usize, quant: bool) -> Self {
        if quant {
            Mat::Q8(Q8Tensor::quantize(src.as_slice(), out, in_))
        } else {
            match src {
                crate::gguf::Src::Owned(v) => Mat::Owned { w: v, out, in_ },
                crate::gguf::Src::View(s) => Mat::Owned { w: s.to_vec(), out, in_ },
            }
        }
    }

    #[inline]
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Mat::Owned { w, .. } => ops::matvec(w, x, y),
            Mat::Q8(t) => t.matvec(x, y),
        }
    }

    /// Batched: x (tn, in) -> y (tn, out).
    #[inline]
    pub fn matmul(&self, x: &[f32], y: &mut [f32], tn: usize) {
        match self {
            Mat::Owned { w, out, in_ } => ops::matmul(w, x, y, tn, *in_, *out),
            Mat::Q8(t) => t.matmul(x, y, tn),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Mat::Owned { w, .. } => w.len() * 4,
            Mat::Q8(t) => t.bytes(),
        }
    }
}

// keep `dot` referenced so the import is used even if matvec inlines differently
#[allow(dead_code)]
fn _touch(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}
