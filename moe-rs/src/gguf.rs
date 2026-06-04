//! Minimal GGUF v3 reader over an mmap. Parses the metadata KV store and the tensor
//! directory, and hands out zero-copy `&[f32]` slices for F32 tensors.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

// ggml tensor type ids we care about
pub const GGML_TYPE_F32: u32 = 0;

/// A parsed metadata value. Arrays are kept type-specialized for the few kinds we use.
#[derive(Debug, Clone)]
pub enum Meta {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
    Other, // value parsed and skipped (unused type)
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// dims as stored in GGUF (fastest-varying first, i.e. ne[0], ne[1], ...).
    /// For a Linear weight this is [in, out]; for stacked experts [in, out, n_expert].
    pub ne: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64, // relative to the data section start
}

impl TensorInfo {
    pub fn n_elems(&self) -> u64 {
        self.ne.iter().product()
    }
}

pub struct Gguf {
    mmap: Mmap,
    data_start: usize,
    pub meta: HashMap<String, Meta>,
    pub tensors: HashMap<String, TensorInfo>,
}

/// Sequential little-endian reader over a byte slice.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cur { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("GGUF: unexpected end of file");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn gstr(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

// GGUF metadata value type ids
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STR: u32 = 8;
const T_ARR: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

fn read_scalar(c: &mut Cur, ty: u32) -> Result<Meta> {
    Ok(match ty {
        T_U8 => Meta::U32(c.u8()? as u32),
        T_I8 => Meta::I32(c.u8()? as i8 as i32),
        T_U16 => Meta::U32(u16::from_le_bytes(c.take(2)?.try_into().unwrap()) as u32),
        T_I16 => Meta::I32(i16::from_le_bytes(c.take(2)?.try_into().unwrap()) as i32),
        T_U32 => Meta::U32(c.u32()?),
        T_I32 => Meta::I32(c.i32()?),
        T_F32 => Meta::F32(c.f32()?),
        T_BOOL => Meta::Bool(c.u8()? != 0),
        T_STR => Meta::Str(c.gstr()?),
        T_U64 => Meta::U64(c.u64()?),
        T_I64 => Meta::I32(c.u64()? as i32), // we never need >32-bit signed metadata
        T_F64 => Meta::F32(c.f64()? as f32),
        other => bail!("GGUF: unsupported metadata scalar type {other}"),
    })
}

fn read_value(c: &mut Cur, ty: u32) -> Result<Meta> {
    if ty != T_ARR {
        return read_scalar(c, ty);
    }
    let arr_ty = c.u32()?;
    let n = c.u64()? as usize;
    Ok(match arr_ty {
        T_STR => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(c.gstr()?);
            }
            Meta::ArrStr(v)
        }
        T_I32 | T_U32 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(c.i32()?);
            }
            Meta::ArrI32(v)
        }
        T_F32 => {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(c.f32()?);
            }
            Meta::ArrF32(v)
        }
        // arrays of other types: parse-and-discard to keep the cursor aligned
        _ => {
            for _ in 0..n {
                read_scalar(c, arr_ty)?;
            }
            Meta::Other
        }
    })
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let (meta, tensors, data_start) = {
            let mut c = Cur::new(&mmap);
            if c.u32()? != GGUF_MAGIC {
                bail!("not a GGUF file (bad magic)");
            }
            let version = c.u32()?;
            if version != 3 {
                bail!("unsupported GGUF version {version} (expected 3)");
            }
            let n_tensors = c.u64()? as usize;
            let n_meta = c.u64()? as usize;

            let mut meta = HashMap::with_capacity(n_meta);
            for _ in 0..n_meta {
                let key = c.gstr()?;
                let ty = c.u32()?;
                let val = read_value(&mut c, ty)?;
                meta.insert(key, val);
            }

            let mut tensors = HashMap::with_capacity(n_tensors);
            for _ in 0..n_tensors {
                let name = c.gstr()?;
                let nd = c.u32()? as usize;
                let mut ne = Vec::with_capacity(nd);
                for _ in 0..nd {
                    ne.push(c.u64()?);
                }
                let ggml_type = c.u32()?;
                let offset = c.u64()?;
                tensors.insert(name, TensorInfo { ne, ggml_type, offset });
            }

            // tensor data begins after the directory, aligned up to general.alignment (default 32)
            let align = match meta.get("general.alignment") {
                Some(Meta::U32(a)) => *a as usize,
                _ => 32,
            };
            let pos = c.p;
            let data_start = pos.div_ceil(align) * align;
            (meta, tensors, data_start)
        };
        Ok(Gguf { mmap, data_start, meta, tensors })
    }

    /// Zero-copy F32 view of a tensor by name. Bails on non-F32 (v1 is F32-only).
    pub fn f32(&self, name: &str) -> Result<&[f32]> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor `{name}` not found in GGUF"))?;
        if t.ggml_type != GGML_TYPE_F32 {
            bail!(
                "tensor `{name}` is ggml_type {} (expected F32=0); this engine is F32-only for now",
                t.ggml_type
            );
        }
        let n = t.n_elems() as usize;
        let byte_off = self.data_start + t.offset as usize;
        let bytes = &self.mmap[byte_off..byte_off + n * 4];
        // mmap is page-aligned, and F32 tensor offsets are 32-byte aligned -> safe to reinterpret.
        let (head, body, tail) = unsafe { bytes.align_to::<f32>() };
        if !head.is_empty() || !tail.is_empty() {
            bail!("tensor `{name}` is not 4-byte aligned");
        }
        Ok(body)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Raw tensor bytes from the mmap.
    fn raw(&self, t: &TensorInfo) -> Result<&[u8]> {
        let n = t.n_elems() as usize;
        let nbytes = type_nbytes(t.ggml_type, n)?;
        let off = self.data_start + t.offset as usize;
        Ok(&self.mmap[off..off + nbytes])
    }

    /// Load a tensor as f32: zero-copy view for F32, or owned (dequantized) for F16/quant types.
    pub fn load(&self, name: &str) -> Result<Src<'_>> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor `{name}` not found"))?;
        let n = t.n_elems() as usize;
        if t.ggml_type == GGML_TYPE_F32 {
            return Ok(Src::View(self.f32(name)?));
        }
        let b = self.raw(t)?;
        let v = match t.ggml_type {
            1 => dequant_f16(b, n),
            6 => dequant_q5_0(b, n),
            8 => dequant_q8_0(b, n),
            12 => dequant_q4_k(b, n),
            14 => dequant_q6_k(b, n),
            other => bail!("tensor `{name}`: unsupported ggml_type {other}"),
        };
        Ok(Src::Owned(v))
    }

    // ---- typed metadata accessors ----
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.meta.get(key)? {
            Meta::U32(v) => Some(*v),
            Meta::I32(v) => Some(*v as u32),
            Meta::U64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.meta.get(key)? {
            Meta::F32(v) => Some(*v),
            Meta::U32(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.meta.get(key)? {
            Meta::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.meta.get(key)? {
            Meta::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn get_str_arr(&self, key: &str) -> Option<&[String]> {
        match self.meta.get(key)? {
            Meta::ArrStr(v) => Some(v),
            _ => None,
        }
    }
    pub fn get_i32_arr(&self, key: &str) -> Option<&[i32]> {
        match self.meta.get(key)? {
            Meta::ArrI32(v) => Some(v),
            _ => None,
        }
    }
}

/// A tensor's f32 data: a zero-copy mmap view (F32) or an owned dequantized buffer.
pub enum Src<'a> {
    View(&'a [f32]),
    Owned(Vec<f32>),
}

impl Src<'_> {
    pub fn as_slice(&self) -> &[f32] {
        match self {
            Src::View(s) => s,
            Src::Owned(v) => v,
        }
    }
}

#[inline]
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    let val = if exp == 0 {
        mant * 2f32.powi(-24)
    } else if exp == 0x1f {
        if h & 0x3ff == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    };
    if sign == 1 { -val } else { val }
}

#[inline]
fn rd_half(b: &[u8], i: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// Byte size of a tensor of `n` elements stored as `ty`.
fn type_nbytes(ty: u32, n: usize) -> Result<usize> {
    Ok(match ty {
        0 => n * 4,            // F32
        1 => n * 2,            // F16
        6 => (n / 32) * 22,    // Q5_0: d(2)+qh(4)+qs(16)
        8 => (n / 32) * 34,    // Q8_0: d(2)+qs(32)
        12 => (n / 256) * 144, // Q4_K: d(2)+dmin(2)+scales(12)+qs(128)
        14 => (n / 256) * 210, // Q6_K: ql(128)+qh(64)+scales(16)+d(2)
        other => bail!("type_nbytes: unsupported ggml_type {other}"),
    })
}

fn dequant_f16(b: &[u8], n: usize) -> Vec<f32> {
    (0..n).map(|i| rd_half(b, i * 2)).collect()
}

fn dequant_q8_0(b: &[u8], n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for blk in 0..n / 32 {
        let o = blk * 34;
        let d = rd_half(b, o);
        for j in 0..32 {
            y[blk * 32 + j] = (b[o + 2 + j] as i8 as f32) * d;
        }
    }
    y
}

fn dequant_q5_0(b: &[u8], n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for blk in 0..n / 32 {
        let o = blk * 22;
        let d = rd_half(b, o);
        let qh = u32::from_le_bytes([b[o + 2], b[o + 3], b[o + 4], b[o + 5]]);
        let qs = &b[o + 6..o + 22];
        let base = blk * 32;
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0F) | xh0) as i32 - 16;
            let x1 = ((qs[j] >> 4) | xh1) as i32 - 16;
            y[base + j] = x0 as f32 * d;
            y[base + 16 + j] = x1 as f32 * d;
        }
    }
    y
}

#[inline]
fn scale_min_k4(j: usize, sc: &[u8]) -> (u8, u8) {
    if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        let d = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        let m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q4_k(b: &[u8], n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    let mut yi = 0usize;
    for blk in 0..n / 256 {
        let o = blk * 144;
        let d = rd_half(b, o);
        let dmin = rd_half(b, o + 2);
        let scales = &b[o + 4..o + 16];
        let qs = &b[o + 16..o + 144];
        let mut is = 0usize;
        let mut q = 0usize; // offset into qs
        for _ in 0..4 {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                y[yi] = d1 * (qs[q + l] & 0xF) as f32 - mm1;
                yi += 1;
            }
            for l in 0..32 {
                y[yi] = d2 * (qs[q + l] >> 4) as f32 - mm2;
                yi += 1;
            }
            q += 32;
            is += 2;
        }
    }
    y
}

fn dequant_q6_k(b: &[u8], n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for blk in 0..n / 256 {
        let o = blk * 210;
        let ql = &b[o..o + 128];
        let qh = &b[o + 128..o + 192];
        let sc = &b[o + 192..o + 208]; // int8
        let d = rd_half(b, o + 208);
        let ybase = blk * 256;
        for half in 0..2 {
            let yb = ybase + half * 128;
            let qlb = half * 64;
            let qhb = half * 32;
            let scb = half * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[qlb + l] & 0xF) | (((qh[qhb + l] >> 0) & 3) << 4)) as i32) - 32;
                let q2 = (((ql[qlb + l + 32] & 0xF) | (((qh[qhb + l] >> 2) & 3) << 4)) as i32) - 32;
                let q3 = (((ql[qlb + l] >> 4) | (((qh[qhb + l] >> 4) & 3) << 4)) as i32) - 32;
                let q4 = (((ql[qlb + l + 32] >> 4) | (((qh[qhb + l] >> 6) & 3) << 4)) as i32) - 32;
                let s = |k: usize| sc[scb + is + k] as i8 as f32;
                y[yb + l] = d * s(0) * q1 as f32;
                y[yb + l + 32] = d * s(2) * q2 as f32;
                y[yb + l + 64] = d * s(4) * q3 as f32;
                y[yb + l + 96] = d * s(6) * q4 as f32;
            }
        }
    }
    y
}
