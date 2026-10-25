//! Accuracy of the Q8 runtime against exact F32 on a fixed text, teacher-forced:
//!
//!   cargo run --release --bin evalq -- <model.gguf> <text file> [prefill tokens]
//!
//! Reports top-1 agreement with F32, mean KL(F32 || Q8), and the perplexity of the text under each.
//! The first [prefill tokens] (default 32) go through the batched prefill path, the rest through decode.

use hobby_rs::Engine;
use std::path::PathBuf;

fn log_softmax(x: &[f32], valid: usize) -> Vec<f64> {
    let x = &x[..valid];
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + x.iter().map(|&v| ((v as f64) - m).exp()).sum::<f64>().ln();
    x.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(x: &[f64]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap()).unwrap()
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(&a[1]);
    let text = std::fs::read_to_string(&a[2])?;
    let split: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let valid = 50257; // GPT-2 vocab; the rest is padding / sentinels

    let reference = {
        let e = Engine::load(&model, false)?;
        let ids = e.encode(&text);
        (e.teacher_forced(&ids, split), ids)
    };
    let (ref_logits, ids) = reference;
    let e = Engine::load(&model, true)?;
    let q_logits = e.teacher_forced(&ids, split);

    let (mut agree, mut kl, mut nll_f, mut nll_q) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    let n = ref_logits.len() - 1; // the last logits have no next token to score
    for i in 0..n {
        let (lf, lq) = (log_softmax(&ref_logits[i], valid), log_softmax(&q_logits[i], valid));
        let next = ids[split + i] as usize;
        agree += (argmax(&lf) == argmax(&lq)) as usize;
        kl += lf.iter().zip(&lq).map(|(&p, &q)| p.exp() * (p - q)).sum::<f64>();
        nll_f -= lf[next];
        nll_q -= lq[next];
    }
    println!("{} tokens ({} prefill, {} scored)", ids.len(), split, n);
    println!("top-1 agreement with F32: {:.2}%", 100.0 * agree as f64 / n as f64);
    println!("mean KL(F32 || Q8): {:.5} nats", kl / n as f64);
    println!("perplexity: F32 {:.3} | Q8 {:.3}", (nll_f / n as f64).exp(), (nll_q / n as f64).exp());
    Ok(())
}
