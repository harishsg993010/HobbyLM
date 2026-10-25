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

// ---- f16, the KV cache storage type: F16C where available, software otherwise ----

/// f32 -> IEEE half, round to nearest even.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1; // a carry into the exponent is still the correctly rounded value
    }
    sign | r as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: normalize
            let mut e: u32 = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// dst = f16(src)
pub fn to_f16(src: &[f32], dst: &mut [u16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    if src.len() % 8 == 0 {
        use std::arch::x86_64::*;
        for i in (0..src.len()).step_by(8) {
            unsafe {
                let v = _mm256_loadu_ps(src.as_ptr().add(i));
                let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
                _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
            }
        }
        return;
    }
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = f32_to_f16(x);
    }
}

/// a . b with b stored as f16
#[inline]
pub fn dot_f16(a: &[f32], b: &[u16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c", target_feature = "fma"))]
    if a.len() % 16 == 0 {
        use std::arch::x86_64::*;
        unsafe {
            let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            for i in (0..a.len()).step_by(16) {
                let b0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
                let b1 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + 8) as *const __m128i));
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), b0, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), b1, s1);
            }
            let s = _mm256_add_ps(s0, s1);
            let lo = _mm256_castps256_ps128(s);
            let hi = _mm256_extractf128_ps(s, 1);
            let t = _mm_add_ps(lo, hi);
            let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
            let t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0b01));
            return _mm_cvtss_f32(t);
        }
    }
    a.iter().zip(b).map(|(&x, &h)| x * f16_to_f32(h)).sum()
}

/// y += a * x with x stored as f16
#[inline]
pub fn axpy_f16(a: f32, x: &[u16], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c", target_feature = "fma"))]
    if x.len() % 8 == 0 {
        use std::arch::x86_64::*;
        unsafe {
            let av = _mm256_set1_ps(a);
            for i in (0..x.len()).step_by(8) {
                let xv = _mm256_cvtph_ps(_mm_loadu_si128(x.as_ptr().add(i) as *const __m128i));
                let yp = y.as_mut_ptr().add(i);
                _mm256_storeu_ps(yp, _mm256_fmadd_ps(av, xv, _mm256_loadu_ps(yp)));
            }
        }
        return;
    }
    for (yo, &h) in y.iter_mut().zip(x) {
        *yo += a * f16_to_f32(h);
    }
}

/// y += a * x
#[inline]
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let av = f32x8::splat(a);
    let mut i = 0;
    while i + 8 <= n {
        let xv = f32x8::new(<[f32; 8]>::try_from(&x[i..i + 8]).unwrap());
        let yv = f32x8::new(<[f32; 8]>::try_from(&y[i..i + 8]).unwrap());
        y[i..i + 8].copy_from_slice(&xv.mul_add(av, yv).to_array());
        i += 8;
    }
    while i < n {
        y[i] += a * x[i];
        i += 1;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The software f16 path (used without F16C) must round exactly like the hardware.
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    #[test]
    fn f16_software_matches_f16c() {
        let mut x = 0x1234_5678u64;
        let mut vals: Vec<f32> = vec![0.0, -0.0, 1.0, 65504.0, 65520.0, 1e-8, 6.1e-5, 5.9e-8, f32::INFINITY, -f32::INFINITY];
        for _ in 0..1_000_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = f32::from_bits(x as u32);
            if v.is_finite() && v.abs() < 1e6 {
                vals.push(v);
            }
        }
        vals.truncate(vals.len() / 8 * 8);
        let mut hw = vec![0u16; vals.len()];
        to_f16(&vals, &mut hw);
        for (&v, &h) in vals.iter().zip(&hw) {
            assert_eq!(f32_to_f16(v), h, "f32_to_f16({v:e})");
            let back = unsafe {
                use std::arch::x86_64::*;
                _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(h as i16)))
            };
            assert_eq!(f16_to_f32(h).to_bits(), back.to_bits(), "f16_to_f32({h:#x})");
        }
    }
}
