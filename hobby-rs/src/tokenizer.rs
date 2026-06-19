//! GPT-2 byte-level BPE, self-contained from the GGUF-embedded vocab + merges.

use crate::gguf::Gguf;
use anyhow::{Context, Result};
use fancy_regex::Regex;
use std::collections::HashMap;

pub struct Tokenizer {
    id2tok: Vec<String>,
    tok2id: HashMap<String, u32>,
    /// merge rank keyed by "left right" (space-joined, as stored in GGUF merges).
    ranks: HashMap<String, usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    re: Regex,
    pub eos: u32,
}

/// GPT-2 reversible byte<->unicode mapping (the `bytes_to_unicode` table).
fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
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
    let mut char_to_byte = HashMap::new();
    for i in 0..bs.len() {
        let ch = char::from_u32(cs[i]).unwrap();
        byte_to_char[bs[i] as usize] = ch;
        char_to_byte.insert(ch, bs[i] as u8);
    }
    (byte_to_char, char_to_byte)
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let tokens = g
            .get_str_arr("tokenizer.ggml.tokens")
            .context("missing tokenizer.ggml.tokens")?;
        let merges = g
            .get_str_arr("tokenizer.ggml.merges")
            .context("missing tokenizer.ggml.merges")?;
        let id2tok: Vec<String> = tokens.to_vec();
        let mut tok2id = HashMap::with_capacity(id2tok.len());
        for (i, t) in id2tok.iter().enumerate() {
            tok2id.insert(t.clone(), i as u32);
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            ranks.insert(m.clone(), i);
        }
        let (byte_to_char, char_to_byte) = bytes_to_unicode();
        let re = Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        )
        .context("compiling GPT-2 pretokenizer regex")?;
        let eos = g
            .get_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(50256);
        Ok(Tokenizer { id2tok, tok2id, ranks, byte_to_char, char_to_byte, re, eos })
    }

    pub fn vocab_len(&self) -> usize {
        self.id2tok.len()
    }

    /// BPE-merge one pre-token (already mapped to GPT-2 unicode chars), append ids.
    fn bpe(&self, word_str: &str, out: &mut Vec<u32>) {
        let mut word: Vec<String> = word_str.chars().map(|c| c.to_string()).collect();
        if word.is_empty() {
            return;
        }
        loop {
            // pick the adjacent pair with the lowest merge rank
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
            let merged = format!("{}{}", word[i], word[i + 1]);
            word[i] = merged;
            word.remove(i + 1);
        }
        for sym in &word {
            match self.tok2id.get(sym) {
                Some(&id) => out.push(id),
                None => {
                    // should not happen for GPT-2 vocab; fall back to per-char ids
                    for ch in sym.chars() {
                        if let Some(&id) = self.tok2id.get(&ch.to_string()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for m in self.re.find_iter(text) {
            let piece = match m {
                Ok(mat) => mat.as_str(),
                Err(_) => continue,
            };
            let mapped: String = piece.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            self.bpe(&mapped, &mut ids);
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(s) = self.id2tok.get(id as usize) {
                for ch in s.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single token id to a (possibly partial-utf8) string for streaming.
    pub fn decode_one(&self, id: u32) -> String {
        self.decode(&[id])
    }
}
