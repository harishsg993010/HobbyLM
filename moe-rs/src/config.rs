//! Model hyperparameters, read entirely from GGUF metadata (never hardcoded).
//! The published joint12 uses rope_theta=1e6 / 8k ctx, so reading from the file is mandatory.

use crate::gguf::Gguf;
use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: String,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_dense: usize, // leading dense blocks; blocks >= n_dense are MoE
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dense_ffn: usize,
    pub expert_ffn: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool, // renormalize top-k gate weights?
    pub gating_sigmoid: bool,      // true => sigmoid router, false => softmax
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g
            .get_str("general.architecture")
            .context("missing general.architecture")?
            .to_string();
        let k = |s: &str| format!("{arch}.{s}");
        let u = |s: &str| -> Result<usize> {
            g.get_u32(&k(s))
                .map(|v| v as usize)
                .with_context(|| format!("missing metadata `{}.{}`", arch, s))
        };

        // head_dim is DECOUPLED (128 != d_model/n_heads). Prefer attention.key_length.
        let head_dim = g
            .get_u32(&k("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                let d = g.get_u32(&k("embedding_length")).unwrap_or(0) as usize;
                let h = g.get_u32(&k("attention.head_count")).unwrap_or(1) as usize;
                d / h.max(1)
            });

        // vocab: prefer the token_embd tensor row count, fall back to metadata
        let vocab_size = g
            .info("token_embd.weight")
            .and_then(|t| t.ne.get(1).copied())
            .map(|v| v as usize)
            .or_else(|| g.get_u32(&k("vocab_size")).map(|v| v as usize))
            .context("cannot determine vocab_size")?;

        let gating_sigmoid = match g.get_u32(&k("expert_gating_func")) {
            Some(2) => true,  // SIGMOID
            Some(1) => false, // SOFTMAX
            _ => true,        // bailingmoe2 default in our export is sigmoid
        };

        Ok(Config {
            d_model: u("embedding_length")?,
            n_layers: u("block_count")?,
            n_dense: u("leading_dense_block_count").unwrap_or(1),
            n_heads: u("attention.head_count")?,
            n_kv_heads: u("attention.head_count_kv")?,
            head_dim,
            dense_ffn: u("feed_forward_length")?,
            expert_ffn: u("expert_feed_forward_length")?,
            n_experts: u("expert_count")?,
            top_k: u("expert_used_count")?,
            n_shared: u("expert_shared_count").unwrap_or(0),
            vocab_size,
            context_length: u("context_length").unwrap_or(8192),
            rms_eps: g
                .get_f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope_theta: g.get_f32(&k("rope.freq_base")).unwrap_or(1e6),
            expert_weights_scale: g.get_f32(&k("expert_weights_scale")).unwrap_or(1.0),
            expert_weights_norm: g.get_bool(&k("expert_weights_norm")).unwrap_or(false),
            gating_sigmoid,
            arch,
        })
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    pub fn is_moe(&self, layer: usize) -> bool {
        layer >= self.n_dense
    }
}
