//! Minimal safetensors reader (mmap, F32 only) — no external deps beyond memmap2.
//!
//! Format: [u64 LE header_len][JSON header][raw tensor bytes]. The JSON maps each tensor name to
//! `{dtype, shape, data_offsets:[start,end]}` where offsets are relative to the end of the header.
//! We only support F32 tensors (the image export writes everything as F32).

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct SafeTensors {
    _mmap: Mmap,
    base: usize,                         // byte offset where tensor data starts
    index: HashMap<String, TensorInfo>,
    ptr: *const u8,                      // start of the mmap (for building &[f32] slices)
}

#[derive(Clone)]
struct TensorInfo {
    shape: Vec<usize>,
    start: usize, // relative to base
    end: usize,
}

// SAFETY: the mmap is read-only and lives as long as the struct; slices we hand out borrow &self.
unsafe impl Send for SafeTensors {}
unsafe impl Sync for SafeTensors {}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < 8 {
            bail!("safetensors too small: {}", path.display());
        }
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let base = 8 + hlen;
        if base > mmap.len() {
            bail!("safetensors header overruns file: {}", path.display());
        }
        let header = std::str::from_utf8(&mmap[8..base])?;
        let mut index = HashMap::new();
        // tiny hand JSON walk: the header is a flat object of `"name":{...}` entries. We parse it
        // with a minimal tokenizer rather than pulling in serde.
        parse_header(header, &mut index)
            .with_context(|| format!("parse safetensors header of {}", path.display()))?;
        let ptr = mmap.as_ptr();
        Ok(SafeTensors { _mmap: mmap, base, index, ptr })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// Borrow a tensor's data as &[f32] plus its shape. Panics if the name is missing — callers
    /// know their key set; use `has` to probe optional tensors.
    pub fn get(&self, name: &str) -> (&[f32], &[usize]) {
        let info = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let n = (info.end - info.start) / 4;
        let off = self.base + info.start;
        let data = unsafe { std::slice::from_raw_parts(self.ptr.add(off) as *const f32, n) };
        (data, &info.shape)
    }

    /// Flat data only.
    pub fn data(&self, name: &str) -> &[f32] {
        self.get(name).0
    }
    /// Shape only.
    pub fn shape(&self, name: &str) -> &[usize] {
        self.get(name).1
    }
}

/// Parse the flat safetensors JSON header. Entries look like:
///   "blocks.0.sa.q.weight":{"dtype":"F32","shape":[1024,1024],"data_offsets":[0,4194304]}
/// plus an optional "__metadata__":{...} which we skip. No nested tensors, so a small scanner
/// suffices and avoids a serde dependency.
fn parse_header(s: &str, out: &mut HashMap<String, TensorInfo>) -> Result<()> {
    let b = s.as_bytes();
    let mut i = 0usize;
    // find opening brace
    while i < b.len() && b[i] != b'{' {
        i += 1;
    }
    i += 1;
    loop {
        // skip whitespace and commas
        while i < b.len() && (b[i] as char).is_whitespace() || (i < b.len() && b[i] == b',') {
            i += 1;
        }
        if i >= b.len() || b[i] == b'}' {
            break;
        }
        if b[i] != b'"' {
            break;
        }
        // read key string
        let (key, ni) = read_string(b, i)?;
        i = ni;
        skip_ws(b, &mut i);
        if i >= b.len() || b[i] != b':' {
            bail!("expected ':' after key {key}");
        }
        i += 1;
        skip_ws(b, &mut i);
        if i >= b.len() || b[i] != b'{' {
            bail!("expected object for {key}");
        }
        // read the object body up to the matching brace
        let obj_start = i;
        let mut depth = 0;
        while i < b.len() {
            if b[i] == b'{' {
                depth += 1;
            } else if b[i] == b'}' {
                depth -= 1;
                if depth == 0 {
                    i += 1;
                    break;
                }
            }
            i += 1;
        }
        let obj = &s[obj_start..i];
        if key == "__metadata__" {
            continue;
        }
        let dtype = json_str_field(obj, "dtype").unwrap_or_default();
        if dtype != "F32" {
            bail!("tensor {key} has dtype {dtype}, only F32 supported");
        }
        let shape = json_usize_array(obj, "shape");
        let offs = json_usize_array(obj, "data_offsets");
        if offs.len() != 2 {
            bail!("tensor {key} missing data_offsets");
        }
        out.insert(key, TensorInfo { shape, start: offs[0], end: offs[1] });
    }
    Ok(())
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && (b[*i] as char).is_whitespace() {
        *i += 1;
    }
}

fn read_string(b: &[u8], mut i: usize) -> Result<(String, usize)> {
    if b[i] != b'"' {
        bail!("expected string");
    }
    i += 1;
    let start = i;
    while i < b.len() && b[i] != b'"' {
        if b[i] == b'\\' {
            i += 1;
        }
        i += 1;
    }
    let s = String::from_utf8_lossy(&b[start..i]).into_owned();
    Ok((s, i + 1))
}

/// Extract `"field":"value"` from a small JSON object string.
fn json_str_field(obj: &str, field: &str) -> Option<String> {
    let pat = format!("\"{field}\"");
    let p = obj.find(&pat)? + pat.len();
    let rest = &obj[p..];
    let c = rest.find(':')? + 1;
    let rest = &rest[c..];
    let q = rest.find('"')? + 1;
    let rest = &rest[q..];
    let e = rest.find('"')?;
    Some(rest[..e].to_string())
}

/// Extract `"field":[a,b,c]` as Vec<usize>.
fn json_usize_array(obj: &str, field: &str) -> Vec<usize> {
    let pat = format!("\"{field}\"");
    let Some(p) = obj.find(&pat) else { return vec![] };
    let rest = &obj[p + pat.len()..];
    let Some(lb) = rest.find('[') else { return vec![] };
    let Some(rb) = rest[lb..].find(']') else { return vec![] };
    rest[lb + 1..lb + rb]
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect()
}
