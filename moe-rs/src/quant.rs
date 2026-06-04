//! Q8_0 weight quantization (ggml-style: blocks of 32 -> one f32 scale + 32 int8) and a
//! `Mat` abstraction that lets every weight be either F32 (zero-copy mmap) or Q8 (owned),
//! dispatched by a single `matvec`. Q8_0 is near-lossless and ~3.8x smaller in RAM, so the
//! bandwidth-bound matmul runs faster.

use crate::ops::{self, dot};
use rayon::prelude::*;

const QK: usize = 32; // block size

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
            let qrow = &self.qs[o * in_..o * in_ + in_];
            let srow = &self.scales[o * nb..o * nb + nb];
            let mut acc = 0.0f32;
            for b in 0..nb {
                let qw = &qrow[b * QK..b * QK + QK];
                let qa = &qx[b * QK..b * QK + QK];
                // i8*i8 -> i32 widening MAC (autovectorizes on AVX2)
                let mut isum: i32 = 0;
                for j in 0..QK {
                    isum += qw[j] as i32 * qa[j] as i32;
                }
                acc += srow[b] * dx[b] * isum as f32;
            }
            *yo = acc;
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

/// A weight matrix: zero-copy F32 mmap view, owned F32 (dequantized), or owned Q8_0.
pub enum Mat<'a> {
    View { w: &'a [f32], out: usize, in_: usize },
    Owned { w: Vec<f32>, out: usize, in_: usize },
    Q8(Q8Tensor),
}

impl<'a> Mat<'a> {
    /// From a zero-copy F32 mmap view: quantize to Q8 or keep the borrow.
    pub fn from_view(w: &'a [f32], out: usize, in_: usize, quant: bool) -> Self {
        if quant {
            Mat::Q8(Q8Tensor::quantize(w, out, in_))
        } else {
            Mat::View { w, out, in_ }
        }
    }
    /// From an owned (dequantized) buffer: quantize to Q8 or keep the owned f32.
    pub fn from_owned(w: &[f32], out: usize, in_: usize, quant: bool) -> Self {
        if quant {
            Mat::Q8(Q8Tensor::quantize(w, out, in_))
        } else {
            Mat::Owned { w: w.to_vec(), out, in_ }
        }
    }
    pub fn build(src: crate::gguf::Src<'a>, out: usize, in_: usize, quant: bool) -> Self {
        match src {
            crate::gguf::Src::View(w) => Self::from_view(w, out, in_, quant),
            crate::gguf::Src::Owned(v) => Self::from_owned(&v, out, in_, quant),
        }
    }

    #[inline]
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Mat::View { w, .. } => ops::matvec(w, x, y),
            Mat::Owned { w, .. } => ops::matvec(w, x, y),
            Mat::Q8(t) => t.matvec(x, y),
        }
    }

    /// Batched: x (tn, in) -> y (tn, out).
    #[inline]
    pub fn matmul(&self, x: &[f32], y: &mut [f32], tn: usize) {
        match self {
            Mat::View { w, out, in_ } => ops::matmul(w, x, y, tn, *in_, *out),
            Mat::Owned { w, out, in_ } => ops::matmul(w, x, y, tn, *in_, *out),
            Mat::Q8(t) => t.matmul(x, y, tn),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Mat::View { w, .. } => w.len() * 4,
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
