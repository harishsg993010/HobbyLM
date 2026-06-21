//! CLIP ViT-L/14 text encoder (the conditioning tower for the image DiT) + its byte-level BPE
//! tokenizer, from scratch. Produces `last_hidden_state` (seq x 768) which the DiT cross-attends to.
//!
//! Tokenizer: lowercase + whitespace-clean, CLIP regex pre-tokenization, GPT-2 byte->unicode map,
//! word-end "</w>" marker, rank-based BPE merges, BOS/EOS wrap, pad/truncate to max_length.
//! Encoder: token+position embeddings, 12 pre-LN transformer layers (causal self-attn, quick-gelu
//! MLP), final layer norm.

use crate::imgops::{layernorm, linear, quick_gelu};
use crate::st::SafeTensors;
use anyhow::{Context, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::path::Path;

pub struct ClipConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub eps: f32,
    pub bos: u32,
    pub eos: u32,
    pub max_length: usize,
}

impl Default for ClipConfig {
    fn default() -> Self {
        ClipConfig { hidden: 768, layers: 12, heads: 12, eps: 1e-5, bos: 49406, eos: 49407, max_length: 64 }
    }
}

pub struct ClipText {
    st: SafeTensors,
    cfg: ClipConfig,
    tokens: Vec<String>,
    tok2id: HashMap<String, u32>,
    ranks: HashMap<String, usize>,
    byte_to_char: [char; 256],
    re: Regex,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    for c in 0x21..=0x7Eu32 {
        bs.push(c);
    }
    for c in 0xA1..=0xACu32 {
        bs.push(c);
    }
    for c in 0xAE..=0xFFu32 {
        bs.push(c);
    }
    let mut in_bs = [false; 256];
    for &b in &bs {
        in_bs[b as usize] = true;
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !in_bs[b as usize] {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut byte_to_char = ['\0'; 256];
    for i in 0..bs.len() {
        byte_to_char[bs[i] as usize] = char::from_u32(cs[i]).unwrap();
    }
    byte_to_char
}

impl ClipText {
    /// `dir` holds clip_text.safetensors, clip_tokens.txt, clip_merges.txt.
    pub fn load(dir: &Path, cfg: ClipConfig) -> Result<ClipText> {
        let st = SafeTensors::open(&dir.join("clip_text.safetensors"))?;
        let toks_raw = std::fs::read_to_string(dir.join("clip_tokens.txt"))
            .context("read clip_tokens.txt")?;
        let tokens: Vec<String> = toks_raw.lines().map(|l| l.replace("\\n", "\n")).collect();
        let mut tok2id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            tok2id.insert(t.clone(), i as u32);
        }
        let merges_raw = std::fs::read_to_string(dir.join("clip_merges.txt")).context("read clip_merges.txt")?;
        let mut ranks = HashMap::new();
        for (i, line) in merges_raw.lines().enumerate() {
            if i == 0 && line.starts_with('#') {
                continue;
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            ranks.insert(line.to_string(), ranks.len());
        }
        // CLIP pre-tokenization regex (case-insensitive); single digit per \p{N} token.
        let re = Regex::new(r"(?i)'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+")
            .context("compile CLIP regex")?;
        Ok(ClipText { st, cfg, tokens, tok2id, ranks, byte_to_char: bytes_to_unicode(), re })
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        // map utf-8 bytes -> unicode, append </w> to the last symbol
        let mapped: Vec<String> =
            piece.bytes().map(|b| self.byte_to_char[b as usize].to_string()).collect();
        if mapped.is_empty() {
            return;
        }
        let mut word: Vec<String> = mapped;
        let last = word.len() - 1;
        word[last] = format!("{}</w>", word[last]);
        loop {
            let mut best = usize::MAX;
            let mut best_i = None;
            for i in 0..word.len().saturating_sub(1) {
                let key = format!("{} {}", word[i], word[i + 1]);
                if let Some(&r) = self.ranks.get(&key) {
                    if r < best {
                        best = r;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else { break };
            word[i] = format!("{}{}", word[i], word[i + 1]);
            word.remove(i + 1);
        }
        for sym in &word {
            if let Some(&id) = self.tok2id.get(sym) {
                out.push(id);
            }
        }
    }

    /// Tokenize a prompt to exactly `max_length` ids: [BOS] tokens... [EOS] then EOS-padded.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let cleaned = {
            let lower = text.trim().to_lowercase();
            let mut s = String::with_capacity(lower.len());
            let mut prev_ws = false;
            for ch in lower.chars() {
                if ch.is_whitespace() {
                    if !prev_ws && !s.is_empty() {
                        s.push(' ');
                    }
                    prev_ws = true;
                } else {
                    s.push(ch);
                    prev_ws = false;
                }
            }
            s.trim_end().to_string()
        };
        let mut body = Vec::new();
        for m in self.re.find_iter(&cleaned) {
            if let Ok(mat) = m {
                self.bpe(mat.as_str(), &mut body);
            }
        }
        let ml = self.cfg.max_length;
        let mut ids = Vec::with_capacity(ml);
        ids.push(self.cfg.bos);
        ids.extend(body);
        ids.push(self.cfg.eos);
        if ids.len() > ml {
            ids.truncate(ml);
            ids[ml - 1] = self.cfg.eos;
        }
        while ids.len() < ml {
            ids.push(self.cfg.eos); // CLIP pad_token == eos
        }
        ids
    }

    /// Forward `ids` -> last_hidden_state (n x hidden). Causal self-attention; no padding mask
    /// (matches `clip(input_ids)` with only input_ids, the image pipeline's call).
    pub fn encode(&self, ids: &[u32]) -> Vec<f32> {
        let d = self.cfg.hidden;
        let n = ids.len();
        let heads = self.cfg.heads;
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let tok_emb = self.st.data("text_model.embeddings.token_embedding.weight");
        let pos_emb = self.st.data("text_model.embeddings.position_embedding.weight");
        // embeddings
        let mut h = vec![0.0f32; n * d];
        for (i, &id) in ids.iter().enumerate() {
            let te = &tok_emb[id as usize * d..(id as usize + 1) * d];
            let pe = &pos_emb[i * d..(i + 1) * d];
            for j in 0..d {
                h[i * d + j] = te[j] + pe[j];
            }
        }
        let g = |name: &str| self.st.data(name);
        for l in 0..self.cfg.layers {
            let p = format!("text_model.encoder.layers.{l}");
            // self-attention block (pre-LN)
            let ln1 = layernorm(&h, n, d, g(&format!("{p}.layer_norm1.weight")),
                                g(&format!("{p}.layer_norm1.bias")), self.cfg.eps);
            let q = linear(&ln1, n, d, g(&format!("{p}.self_attn.q_proj.weight")),
                           Some(g(&format!("{p}.self_attn.q_proj.bias"))), d);
            let k = linear(&ln1, n, d, g(&format!("{p}.self_attn.k_proj.weight")),
                           Some(g(&format!("{p}.self_attn.k_proj.bias"))), d);
            let v = linear(&ln1, n, d, g(&format!("{p}.self_attn.v_proj.weight")),
                           Some(g(&format!("{p}.self_attn.v_proj.bias"))), d);
            let mut attn = vec![0.0f32; n * d];
            for head in 0..heads {
                let off = head * hd;
                for i in 0..n {
                    // causal: attend j in 0..=i
                    let mut scores = vec![0.0f32; i + 1];
                    for j in 0..=i {
                        let mut s = 0.0f32;
                        for t in 0..hd {
                            s += q[i * d + off + t] * k[j * d + off + t];
                        }
                        scores[j] = s * scale;
                    }
                    // softmax
                    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    for t in 0..hd {
                        let mut acc = 0.0f32;
                        for j in 0..=i {
                            acc += scores[j] * inv * v[j * d + off + t];
                        }
                        attn[i * d + off + t] = acc;
                    }
                }
            }
            let ao = linear(&attn, n, d, g(&format!("{p}.self_attn.out_proj.weight")),
                            Some(g(&format!("{p}.self_attn.out_proj.bias"))), d);
            for idx in 0..n * d {
                h[idx] += ao[idx];
            }
            // MLP block (pre-LN, quick-gelu)
            let ln2 = layernorm(&h, n, d, g(&format!("{p}.layer_norm2.weight")),
                                g(&format!("{p}.layer_norm2.bias")), self.cfg.eps);
            let inter = g(&format!("{p}.mlp.fc1.weight")).len() / d;
            let mut fc1 = linear(&ln2, n, d, g(&format!("{p}.mlp.fc1.weight")),
                                 Some(g(&format!("{p}.mlp.fc1.bias"))), inter);
            for x in fc1.iter_mut() {
                *x = quick_gelu(*x);
            }
            let fc2 = linear(&fc1, n, inter, g(&format!("{p}.mlp.fc2.weight")),
                             Some(g(&format!("{p}.mlp.fc2.bias"))), d);
            for idx in 0..n * d {
                h[idx] += fc2[idx];
            }
        }
        layernorm(&h, n, d, g("text_model.final_layer_norm.weight"),
                  g("text_model.final_layer_norm.bias"), self.cfg.eps)
    }
}
