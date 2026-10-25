//! Q8_0 weight quantization (ggml-style: blocks of 32 -> one f32 scale + 32 int8) and a
//! `Mat` abstraction that lets every weight be either F32 (zero-copy mmap) or Q8 (owned),
//! dispatched by a single `matvec`. Q8_0 is near-lossless and ~3.8x smaller in RAM, so the
//! bandwidth-bound matmul runs faster.

use crate::ops::{self, dot};
use rayon::prelude::*;

const QK: usize = 32; // block size
/// Output rows per rayon task in the decode matvecs.
pub const ROWS_PER_TASK: usize = 32;

/// An activation quantized to per-block int8 + scale, the format the Q8 row kernels consume.
/// `nc` holds, per block, -128 x the sum of each group of 4 values: the exact correction that
/// lets the VNNI kernel treat the weights as unsigned (w + 128) in a single `vpdpbusd`.
#[derive(Default)]
pub struct QAct {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    nc: Vec<i32>,
}

pub fn quantize_act(x: &[f32]) -> QAct {
    let nb = x.len() / QK;
    let mut q = vec![0i8; x.len()];
    let mut d = vec![0.0f32; nb];
    let mut nc = vec![0i32; nb * 8];
    for b in 0..nb {
        let blk = &x[b * QK..b * QK + QK];
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let db = amax / 127.0;
        let id = if db > 0.0 { 1.0 / db } else { 0.0 };
        d[b] = db;
        for j in 0..QK {
            q[b * QK + j] = (blk[j] * id).round().clamp(-127.0, 127.0) as i8;
        }
        for l in 0..8 {
            let g = &q[b * QK + 4 * l..b * QK + 4 * l + 4];
            nc[b * 8 + l] = -128 * g.iter().map(|&v| v as i32).sum::<i32>();
        }
    }
    QAct { q, d, nc }
}

/// `R` consecutive weight rows (starting at `qw`/`sw`, row strides `in_` and `nb`) against one
/// activation, sharing each activation block load across the rows. Per-lane integer sums and the
/// accumulation order match `qrow_dot_avx2`, so results are bit-identical to it.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "avxvnni"))]
#[inline]
unsafe fn rows_dot_vnni<const R: usize>(qw: *const i8, sw: *const f32, in_: usize, nb: usize, a: &QAct, y: &mut [f32]) {
    use std::arch::x86_64::*;
    let flip = _mm256_set1_epi8(-128i8);
    let mut acc = [_mm256_setzero_ps(); R];
    for b in 0..nb {
        let av = _mm256_loadu_si256(a.q.as_ptr().add(b * QK) as *const __m256i);
        let nc = _mm256_loadu_si256(a.nc.as_ptr().add(b * 8) as *const __m256i);
        let dx = *a.d.get_unchecked(b);
        for r in 0..R {
            let w = _mm256_loadu_si256(qw.add(r * in_ + b * QK) as *const __m256i);
            let p32 = _mm256_dpbusd_avx_epi32(nc, _mm256_xor_si256(w, flip), av);
            let d = _mm256_set1_ps(*sw.add(r * nb + b) * dx);
            acc[r] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p32), acc[r]);
        }
    }
    for r in 0..R {
        let lo = _mm256_castps256_ps128(acc[r]);
        let hi = _mm256_extractf128_ps(acc[r], 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0b01));
        y[r] = _mm_cvtss_f32(s);
    }
}

/// Block-int8 dot of one weight row vs the quantized activation: returns sum_b scale_w[b]*scale_x[b]*
/// (sum_j qw[j]*qa[j]). On AVX2 this uses the ggml maddubs trick (abs(w)·(a·sign(w))) for a real
/// signed int8 dot; otherwise a 4-accumulator scalar fallback. `qw`/`qa` len = nb*QK.
#[allow(dead_code)]
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

/// y[r] = sum_s ws[s] * (row o0+r of mats[s]) . acts[s], for r in 0..y.len(): the MoE down
/// projection with every expert accumulated in registers and one horizontal sum per row.
pub fn rows_dot_weighted(mats: &[&Q8Tensor], acts: &[&QAct], ws: &[f32], o0: usize, y: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "avxvnni"))]
    unsafe {
        use std::arch::x86_64::*;
        let flip = _mm256_set1_epi8(-128i8);
        let mut r = 0;
        while r < y.len() {
            let rr = (y.len() - r).min(4);
            let mut acc = [_mm256_setzero_ps(); 4];
            for (s, m) in mats.iter().enumerate() {
                let (in_, nb, a) = (m.in_, m.nb, acts[s]);
                for b in 0..nb {
                    let av = _mm256_loadu_si256(a.q.as_ptr().add(b * QK) as *const __m256i);
                    let nc = _mm256_loadu_si256(a.nc.as_ptr().add(b * 8) as *const __m256i);
                    let wdx = ws[s] * *a.d.get_unchecked(b);
                    for k in 0..rr {
                        let o = o0 + r + k;
                        let w = _mm256_loadu_si256(m.qs.as_ptr().add(o * in_ + b * QK) as *const __m256i);
                        let p32 = _mm256_dpbusd_avx_epi32(nc, _mm256_xor_si256(w, flip), av);
                        let d = _mm256_set1_ps(*m.scales.get_unchecked(o * nb + b) * wdx);
                        acc[k] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p32), acc[k]);
                    }
                }
            }
            for k in 0..rr {
                let lo = _mm256_castps256_ps128(acc[k]);
                let hi = _mm256_extractf128_ps(acc[k], 1);
                let t = _mm_add_ps(lo, hi);
                let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
                let t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0b01));
                y[r + k] = _mm_cvtss_f32(t);
            }
            r += rr;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "avxvnni")))]
    {
        let mut tmp = vec![0.0f32; y.len()];
        y.iter_mut().for_each(|v| *v = 0.0);
        for (s, m) in mats.iter().enumerate() {
            m.rows_dot(o0, acts[s], &mut tmp);
            for (yo, &t) in y.iter_mut().zip(&tmp) {
                *yo += ws[s] * t;
            }
        }
    }
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

    /// From Q8_0 blocks already in this layout (scales [out * nb], values [out * in_]).
    pub fn from_blocks(scales: Vec<f32>, qs: Vec<i8>, out: usize, in_: usize) -> Self {
        assert_eq!(qs.len(), out * in_);
        Q8Tensor { out, in_, nb: in_ / QK, scales, qs }
    }

    /// Rows `o0..o0 + y.len()` against a pre-quantized activation (see `quantize_act`).
    #[inline]
    pub fn rows_dot(&self, o0: usize, a: &QAct, y: &mut [f32]) {
        let (in_, nb) = (self.in_, self.nb);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "avxvnni"))]
        {
            let n = y.len();
            let mut r = 0;
            while r + 4 <= n {
                let o = o0 + r;
                unsafe {
                    rows_dot_vnni::<4>(self.qs.as_ptr().add(o * in_), self.scales.as_ptr().add(o * nb),
                                       in_, nb, a, &mut y[r..r + 4])
                };
                r += 4;
            }
            while r < n {
                let o = o0 + r;
                unsafe {
                    rows_dot_vnni::<1>(self.qs.as_ptr().add(o * in_), self.scales.as_ptr().add(o * nb),
                                       in_, nb, a, &mut y[r..r + 1])
                };
                r += 1;
            }
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "avxvnni")))]
        for (r, yo) in y.iter_mut().enumerate() {
            let o = o0 + r;
            *yo = qrow_dot(&self.qs[o * in_..o * in_ + in_], &a.q, &self.scales[o * nb..o * nb + nb], &a.d, nb);
        }
    }

    /// int8 x int8 -> i32 dot, with weight & activation block scales applied at block end.
    /// The activation is quantized to int8 once (shared across all output rows).
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_);
        self.matvec_q(&quantize_act(x), y);
    }

    /// matvec with an already-quantized activation. Rows are handed to the pool in chunks so a
    /// small matrix is not split into hundreds of one-row tasks.
    pub fn matvec_q(&self, a: &QAct, y: &mut [f32]) {
        if y.len() >= 64 {
            crate::pool::par_chunks_mut(y, ROWS_PER_TASK, |c, yc| self.rows_dot(c * ROWS_PER_TASK, a, yc));
        } else {
            self.rows_dot(0, a, y);
        }
    }

    pub fn bytes(&self) -> usize {
        self.scales.len() * 4 + self.qs.len()
    }

    /// Batched int8 matmul: x is (tn, in), y is (tn, out). Each task takes a block of weight
    /// rows (which then stays in cache) and runs every activation row against it.
    pub fn matmul(&self, x: &[f32], y: &mut [f32], tn: usize) {
        let (in_, out) = (self.in_, self.out);
        let acts: Vec<QAct> = x.chunks(in_).take(tn).map(quantize_act).collect();
        let nchunks = out.div_ceil(ROWS_PER_TASK);
        let yp = y.as_mut_ptr() as usize;
        crate::pool::pool().run(nchunks, &|c| {
            let o0 = c * ROWS_PER_TASK;
            let n = ROWS_PER_TASK.min(out - o0);
            for (t, a) in acts.iter().enumerate() {
                // each (t, o0..o0+n) range is written by exactly one task
                let yr = unsafe { std::slice::from_raw_parts_mut((yp as *mut f32).add(t * out + o0), n) };
                self.rows_dot(o0, a, yr);
            }
        });
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

    /// F32 matvec on the spin pool in chunks of 8 rows (the router: small, but on the critical path).
    pub fn matvec_pool(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Mat::Owned { w, in_, .. } => crate::pool::par_chunks_mut(y, 8, |c, yc| {
                for (r, yo) in yc.iter_mut().enumerate() {
                    let o = c * 8 + r;
                    *yo = dot(&w[o * in_..o * in_ + in_], x);
                }
            }),
            Mat::Q8(t) => t.matvec(x, y),
        }
    }

    /// Single-threaded matvec, for matrices too small to be worth a parallel dispatch.
    pub fn matvec_serial(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Mat::Owned { w, in_, .. } => {
                for (o, yo) in y.iter_mut().enumerate() {
                    *yo = dot(&w[o * in_..o * in_ + in_], x);
                }
            }
            Mat::Q8(t) => t.rows_dot(0, &quantize_act(x), y),
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

    pub fn as_q8(&self) -> Option<&Q8Tensor> {
        match self {
            Mat::Q8(t) => Some(t),
            _ => None,
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Mat::Owned { w, .. } => w.len() * 4,
            Mat::Q8(t) => t.bytes(),
        }
    }
}


