//! Greedy / temperature / top-p sampling with a tiny self-contained RNG.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

impl Rng {
    /// Public uniform [0,1) draw (for the diffusion Gumbel sampler).
    pub fn unif(&mut self) -> f32 {
        self.next_f32()
    }
}

/// Gumbel-max sample from softmax(logits/temp): argmax_i (logits_i/temp + g_i), g = -ln(-ln(u)).
/// temp<=0 falls back to plain argmax. Skips non-finite logits (banned tokens).
pub fn gumbel_argmax(logits: &[f32], temp: f32, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        return argmax(logits);
    }
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if !l.is_finite() {
            continue;
        }
        let u = rng.next_f32().clamp(1e-9, 1.0 - 1e-9);
        let val = l / temp + -(-(u.ln())).ln();
        if val > bv {
            bv = val;
            best = i;
        }
    }
    best as u32
}

/// Pick the argmax token id.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature + optional top-p (nucleus) sampling. `temp<=0` => greedy.
pub fn sample(logits: &[f32], temp: f32, top_p: f32, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        return argmax(logits);
    }
    // softmax with temperature
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temp))
        .collect();
    let m = probs.iter().fold(f32::NEG_INFINITY, |a, &(_, v)| a.max(v));
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        p.1 = (p.1 - m).exp();
        sum += p.1;
    }
    for p in probs.iter_mut() {
        p.1 /= sum;
    }
    // top-p filter
    if top_p > 0.0 && top_p < 1.0 {
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0f32;
        let mut cut = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p.1;
            if cum >= top_p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
        let s: f32 = probs.iter().map(|p| p.1).sum();
        for p in probs.iter_mut() {
            p.1 /= s;
        }
    }
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for &(i, p) in &probs {
        cum += p;
        if r < cum {
            return i as u32;
        }
    }
    probs.last().map(|&(i, _)| i as u32).unwrap_or(0)
}
