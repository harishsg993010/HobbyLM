//! hobby-rs CLI — thin wrapper over the `hobby_rs` library Engine.

use anyhow::{bail, Result};
use clap::Parser;
use hobby_rs::{gguf, DiffOpts, Engine, GenOpts};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "CPU inference for the 500M hobbylm MoE")]
struct Args {
    /// Path to the GGUF (F32 or quantized: Q8_0/Q5_0/Q4_K/Q6_K)
    #[arg(short, long)]
    model: PathBuf,
    #[arg(short, long, default_value = "The capital of France is")]
    prompt: String,
    #[arg(short, long, default_value_t = 64)]
    n: usize,
    #[arg(short, long, default_value_t = 0.0)]
    temp: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 1234)]
    seed: u64,
    /// Repetition penalty (1.0 = off; ~1.3 curbs repetition loops in chat)
    #[arg(long, default_value_t = 1.0)]
    rep_penalty: f32,
    #[arg(long)]
    threads: Option<usize>,
    /// Weight precision: q8 (default, smaller+faster) or f32 (exact)
    #[arg(long, default_value = "q8")]
    quant: String,
    /// Print the GGUF tensor-type histogram and exit
    #[arg(long)]
    info: bool,
    /// (diffusion models) semi-AR block length; 0 = use the model's default
    #[arg(long, default_value_t = 0)]
    block: usize,
    /// (diffusion models) number of denoising steps across the canvas
    #[arg(long, default_value_t = 96)]
    steps: usize,
    /// (diffusion models) low-confidence remasking refinement passes per block
    #[arg(long, default_value_t = 0)]
    remask_steps: usize,
    /// (diffusion models) disable the prefix KV-cache (faithful but O(canvas)/step)
    #[arg(long)]
    no_cache: bool,
    /// (diffusion models) confidence threshold for Fast-dLLM parallel decode (0 = fixed schedule)
    #[arg(long, default_value_t = 0.0)]
    threshold: f32,
    /// Precomputed image / audio / speech embeddings (raw f32, N*d_model)
    #[arg(long)]
    image: Option<PathBuf>,
    #[arg(long)]
    audio: Option<PathBuf>,
    #[arg(long)]
    speech: Option<PathBuf>,
}

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
        8 => "Q8_0", 9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
        14 => "Q6_K", 15 => "Q8_K", _ => "?",
    }
}

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

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok();
    }

    if args.info {
        let g = gguf::Gguf::open(&args.model)?;
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

    let t0 = Instant::now();
    let quant = match args.quant.as_str() {
        "q8" | "Q8" | "q8_0" => true,
        "f32" | "none" => false,
        other => bail!("unknown --quant `{other}` (use q8 or f32)"),
    };
    let eng = Engine::load(&args.model, quant)?;
    let c = &eng.cfg;
    eprintln!(
        "loaded {} | d={} L={} experts={}/top{} vocab={} ctx={} | quant={} | weights {:.2} GB | init {:.2}s",
        c.arch, c.d_model, c.n_layers, c.n_experts, c.top_k, c.vocab_size, c.context_length,
        if quant { "q8_0" } else { "f32" }, eng.weight_bytes() as f64 / 1e9, t0.elapsed().as_secs_f32(),
    );

    let mut mm: Vec<Vec<f32>> = Vec::new();
    for p in [&args.image, &args.audio, &args.speech].into_iter().flatten() {
        let e = load_embeds(p, c.d_model)?;
        eprintln!("spliced {} embeddings from {}", e.len(), p.display());
        mm.extend(e);
    }

    print!("{}", args.prompt);
    std::io::stdout().flush().ok();
    let tg = Instant::now();
    let ntok = if c.diffusion {
        if !mm.is_empty() {
            eprintln!("note: diffusion decode is text-only; ignoring spliced embeddings");
        }
        let d = DiffOpts {
            gen_len: if args.n > 0 { args.n } else { 128 },
            block: if args.block > 0 { args.block } else { c.block_size.max(1) },
            steps: args.steps,
            temp: args.temp,
            rep_penalty: if args.rep_penalty != 1.0 { args.rep_penalty } else { 1.4 },
            remask_steps: args.remask_steps,
            remask_frac: 0.3,
            seed: args.seed,
            cached: !args.no_cache,
            threshold: args.threshold,
        };
        eprintln!("diffusion decode: gen_len={} block={} steps={} temp={} rep_penalty={} remask={} cached={} threshold={}",
                  d.gen_len, d.block, d.steps, d.temp, d.rep_penalty, d.remask_steps, d.cached, d.threshold);
        eng.generate_diffusion(&args.prompt, &d, |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
            true
        })
    } else {
        let opts = GenOpts { max_new: args.n, temp: args.temp, top_p: args.top_p, seed: args.seed, rep_penalty: args.rep_penalty };
        eng.generate(&args.prompt, &mm, &opts, |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
            true
        })
    };
    let gen_s = tg.elapsed().as_secs_f32();
    println!();
    eprintln!("decode {} tok in {:.2}s ({:.1} tok/s)", ntok, gen_s, ntok as f32 / gen_s.max(1e-6));
    Ok(())
}
