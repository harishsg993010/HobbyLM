//! Microbenchmarks for the decode hot path: spin-pool dispatch latency vs rayon, and the
//! achievable memory read bandwidth when the pool streams a large buffer.
//!
//!   cargo run --release --bin poolbench -- [threads]
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let threads: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    let mut y = vec![0.0f32; 16 * 128];
    for (name, iters) in [("warm", 1000), ("pool", 20000)] {
        let t = Instant::now();
        for _ in 0..iters {
            hobby_rs::pool::par_chunks_mut(&mut y, 128, |c, yc| yc[0] += c as f32);
        }
        println!("{name}: {:.2} us/dispatch", t.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    let t = Instant::now();
    for _ in 0..20000 {
        y.par_chunks_mut(128).enumerate().for_each(|(c, yc)| yc[0] += c as f32);
    }
    println!("rayon: {:.2} us/dispatch", t.elapsed().as_secs_f64() * 1e6 / 20000.0);

    // stream 512 MB of i8 (well past the caches) in 32 KB chunks, as the matvecs do
    let buf: Vec<u64> = (0..64 << 20).map(|i| i as u64).collect();
    let ch = 4096; // u64s per chunk = 32 KB
    let mut sums = vec![0u64; buf.len() / ch];
    for rep in 0..4 {
        let t = Instant::now();
        hobby_rs::pool::par_chunks_mut(&mut sums, 1, |c, s| {
            s[0] = buf[c * ch..c * ch + ch].iter().fold(0u64, |a, &v| a ^ v);
        });
        let gbs = (buf.len() * 8) as f64 / t.elapsed().as_secs_f64() / 1e9;
        if rep > 0 {
            println!("read bandwidth: {gbs:.1} GB/s");
        }
    }
    std::hint::black_box(&sums);
}
