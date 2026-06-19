//! hobby-rs — a from-scratch CPU inference engine for the 500M hobbylm MoE LLM.
//! Loads the F32 GGUF and generates text. No llama.cpp, no Python at runtime.

mod config;
mod gguf;
mod model;
mod ops;
mod quant;
mod sample;
mod tokenizer;

use anyhow::{bail, Result};
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Real GPT-2 vocab cutoff: ids >= this are user-defined sentinels (image/audio markers)
/// that we never want to emit during text generation.
const GPT2_VALID: usize = 50257;

#[derive(Parser)]
#[command(about = "CPU inference for the 500M hobbylm MoE (F32 GGUF)")]
struct Args {
    /// Path to the F32 hobbylm GGUF
    #[arg(short, long)]
    model: PathBuf,
    /// Prompt text
    #[arg(short, long, default_value = "The capital of France is")]
    prompt: String,
    /// Max new tokens
    #[arg(short, long, default_value_t = 64)]
    n: usize,
    /// Temperature (0 = greedy)
    #[arg(short, long, default_value_t = 0.0)]
    temp: f32,
    /// Top-p nucleus (only used when temp > 0)
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    /// RNG seed
    #[arg(long, default_value_t = 1234)]
    seed: u64,
    /// Number of threads (default: all cores)
    #[arg(long)]
    threads: Option<usize>,
    /// Weight precision: q8 (Q8_0, smaller+faster, near-lossless) or f32 (exact)
    #[arg(long, default_value = "q8")]
    quant: String,
    /// Print the GGUF's tensor-type histogram and exit
    #[arg(long)]
    info: bool,
    /// Precomputed image embeddings (raw little-endian f32, N*d_model) spliced before the prompt
    #[arg(long)]
    image: Option<PathBuf>,
    /// Precomputed audio (CLAP) embeddings
    #[arg(long)]
    audio: Option<PathBuf>,
    /// Precomputed speech (Whisper) embeddings
    #[arg(long)]
    speech: Option<PathBuf>,
}

/// Read a raw f32 embedding file into rows of length `d`.
fn load_embeds(path: &std::path::Path, d: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() || bytes.len() % (d * 4) != 0 {
        bail!("{}: size {} not a multiple of d_model*4 ({})", path.display(), bytes.len(), d * 4);
    }
    let n = bytes.len() / (d * 4);
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = vec![0.0f32; d];
        for (j, r) in row.iter_mut().enumerate() {
            let o = (i * d + j) * 4;
            *r = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
        8 => "Q8_0", 9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
        14 => "Q6_K", 15 => "Q8_K", _ => "?",
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok();
    }

    let t0 = Instant::now();
    let g = gguf::Gguf::open(&args.model)?;
    if args.info {
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<u32, (usize, String)> = BTreeMap::new();
        for (name, t) in &g.tensors {
            let e = counts.entry(t.ggml_type).or_insert((0, name.clone()));
            e.0 += 1;
        }
        for (ty, (n, ex)) in &counts {
            eprintln!("type {ty:>2} {:>5} : {n:>4} tensors   e.g. {ex}", ggml_type_name(*ty));
        }
        return Ok(());
    }
    let cfg = config::Config::from_gguf(&g)?;
    eprintln!(
        "loaded {} | d={} L={} heads={}/{} hd={} experts={}/top{} +{}shared expff={} vocab={} \
         ctx={} theta={} eps={} gating={}",
        cfg.arch, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
        cfg.n_experts, cfg.top_k, cfg.n_shared, cfg.expert_ffn, cfg.vocab_size,
        cfg.context_length, cfg.rope_theta, cfg.rms_eps,
        if cfg.gating_sigmoid { "sigmoid" } else { "softmax" },
    );
    let tok = tokenizer::Tokenizer::from_gguf(&g)?;
    let quant = match args.quant.as_str() {
        "q8" | "Q8" | "q8_0" => true,
        "f32" | "none" => false,
        other => bail!("unknown --quant `{other}` (use q8 or f32)"),
    };
    let model = model::Model::load(&g, cfg, quant)?;
    let c = &model.cfg;
    eprintln!(
        "init {:.2}s | quant={} | weights {:.2} GB in RAM",
        t0.elapsed().as_secs_f32(),
        if quant { "q8_0" } else { "f32" },
        model.weight_bytes() as f64 / 1e9,
    );

    // modality embeddings, spliced (in order) BEFORE the text prompt, matching training
    // ([IMAGE]/[AUDIO]/[SPEECH] markers come before "USER: ...").
    let mut mm_embeds: Vec<Vec<f32>> = Vec::new();
    for p in [&args.image, &args.audio, &args.speech].into_iter().flatten() {
        let e = load_embeds(p, c.d_model)?;
        eprintln!("spliced {} embeddings from {}", e.len(), p.display());
        mm_embeds.extend(e);
    }

    let ids = tok.encode(&args.prompt);
    if ids.is_empty() && mm_embeds.is_empty() {
        bail!("nothing to run: empty prompt and no embeddings");
    }
    let prefill_len = mm_embeds.len() + ids.len();
    if prefill_len + args.n >= c.context_length {
        bail!(
            "prefill ({}) + n ({}) exceeds context_length ({})",
            prefill_len,
            args.n,
            c.context_length
        );
    }

    let mut cache = model::KvCache::new(c);
    let mut rng = sample::Rng::new(args.seed);

    // ---- batched prefill: [modality embeds][text token embeds] in one pass ----
    let tp = Instant::now();
    let mut inputs: Vec<Vec<f32>> = mm_embeds;
    for &id in &ids {
        inputs.push(model.token_embedding(id));
    }
    let mut pos = inputs.len();
    let mut logits = model.prefill(&inputs, &mut cache);
    let prefill_s = tp.elapsed().as_secs_f32();

    // echo prompt, then stream generation
    print!("{}", args.prompt);
    std::io::stdout().flush().ok();

    let valid = GPT2_VALID.min(c.vocab_size);
    let tg = Instant::now();
    let mut generated = 0usize;
    for _ in 0..args.n {
        for l in logits.iter_mut().take(c.vocab_size).skip(valid) {
            *l = f32::NEG_INFINITY;
        }
        let next = sample::sample(&logits, args.temp, args.top_p, &mut rng);
        if next as usize == tok.eos as usize {
            break;
        }
        print!("{}", tok.decode_one(next));
        std::io::stdout().flush().ok();
        generated += 1;
        logits = model.forward(next, pos, &mut cache);
        pos += 1;
    }
    let gen_s = tg.elapsed().as_secs_f32();
    println!();
    eprintln!(
        "prefill {} pos in {:.2}s ({:.1} pos/s) | decode {} tok in {:.2}s ({:.1} tok/s)",
        prefill_len,
        prefill_s,
        prefill_len as f32 / prefill_s.max(1e-6),
        generated,
        gen_s,
        generated as f32 / gen_s.max(1e-6),
    );
    Ok(())
}
