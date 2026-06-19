//! f32 compute kernels. The hot path is `matvec` (rayon-parallel over output rows).

use rayon::prelude::*;
use wide::f32x8;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // 4 independent f32x8 accumulators: break the serial reduction dependency chain (which blocks
    // auto-vectorization of `s += a*b`) and saturate the FMA units. 32 floats/iter.
    let (mut a0, mut a1, mut a2, mut a3) = (f32x8::ZERO, f32x8::ZERO, f32x8::ZERO, f32x8::ZERO);
    let mut i = 0;
    let main = n - n % 32;
    while i < main {
        let la = |o: usize| f32x8::new(<[f32; 8]>::try_from(&a[o..o + 8]).unwrap());
        let lb = |o: usize| f32x8::new(<[f32; 8]>::try_from(&b[o..o + 8]).unwrap());
        a0 = la(i).mul_add(lb(i), a0);
        a1 = la(i + 8).mul_add(lb(i + 8), a1);
        a2 = la(i + 16).mul_add(lb(i + 16), a2);
        a3 = la(i + 24).mul_add(lb(i + 24), a3);
        i += 32;
    }
    while i + 8 <= n {
        let va = f32x8::new(<[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let vb = f32x8::new(<[f32; 8]>::try_from(&b[i..i + 8]).unwrap());
        a0 = va.mul_add(vb, a0);
        i += 8;
    }
    let mut s = ((a0 + a1) + (a2 + a3)).reduce_add();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// y[o] = sum_i w[o*in + i] * x[i]  for o in 0..out.  w is (out, in) row-major.
/// Parallel over output rows. Used for every Linear / per-expert GLU matmul.
pub fn matvec(w: &[f32], x: &[f32], y: &mut [f32]) {
    let in_ = x.len();
    debug_assert_eq!(w.len(), y.len() * in_);
    // Parallelize only when the work is worth the threading overhead.
    if y.len() >= 64 {
        y.par_iter_mut().enumerate().for_each(|(o, yo)| {
            *yo = dot(&w[o * in_..o * in_ + in_], x);
        });
    } else {
        for (o, yo) in y.iter_mut().enumerate() {
            *yo = dot(&w[o * in_..o * in_ + in_], x);
        }
    }
}

/// Batched matmul: X is (tn, in) row-major, Y is (tn, out). Computed parallel over output
/// columns (each weight row read ONCE from RAM, reused across all tn activation rows which
/// stay in cache) into a transposed buffer, then transposed back. The key prefill speedup.
pub fn matmul(w: &[f32], x: &[f32], y: &mut [f32], tn: usize, in_: usize, out: usize) {
    let mut yt = vec![0.0f32; out * tn]; // (out, tn)
    yt.par_chunks_mut(tn).enumerate().for_each(|(o, ytr)| {
        let wr = &w[o * in_..o * in_ + in_];
        for t in 0..tn {
            ytr[t] = dot(wr, &x[t * in_..t * in_ + in_]);
        }
    });
    for o in 0..out {
        let row = &yt[o * tn..o * tn + tn];
        for t in 0..tn {
            y[t * out + o] = row[t];
        }
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// out[i] = x[i] * rsqrt(mean(x^2)+eps) * w[i]   (matches torch F.rms_norm * weight)
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

/// In-place RMSNorm without a weight (used per-head before QK-norm weight is applied);
/// here we fold the weight in, so this variant is unused — kept for clarity.
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * scale * w[i];
    }
}

/// Numerically-stable softmax in place.
pub fn softmax(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Precomputed rotate-half (NeoX) RoPE cos/sin tables.
pub struct Rope {
    head_dim: usize,
    half: usize,
    cos: Vec<f32>, // [max_pos * half]
    sin: Vec<f32>,
}

impl Rope {
    pub fn new(head_dim: usize, max_pos: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_pos * half];
        let mut sin = vec![0.0f32; max_pos * half];
        for p in 0..max_pos {
            for i in 0..half {
                let inv_freq = (theta as f64).powf(-(2.0 * i as f64) / head_dim as f64);
                let ang = p as f64 * inv_freq;
                cos[p * half + i] = ang.cos() as f32;
                sin[p * half + i] = ang.sin() as f32;
            }
        }
        Rope { head_dim, half, cos, sin }
    }

    /// Apply RoPE to one head vector of length head_dim at the given position, in place.
    pub fn apply(&self, v: &mut [f32], pos: usize) {
        debug_assert_eq!(v.len(), self.head_dim);
        let base = pos * self.half;
        for i in 0..self.half {
            let c = self.cos[base + i];
            let s = self.sin[base + i];
            let x1 = v[i];
            let x2 = v[self.half + i];
            v[i] = x1 * c - x2 * s;
            v[self.half + i] = x2 * c + x1 * s;
        }
    }
}
