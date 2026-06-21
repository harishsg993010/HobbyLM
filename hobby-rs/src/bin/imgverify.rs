//! Stage-by-stage verification of the from-scratch Rust image pipeline against the PyTorch
//! reference bundle (image_weights/ref.safetensors), produced by export/to_image.py.
//!
//!   cargo run --release --bin imgverify -- [image_weights_dir]
//!
//! Reports max-abs-diff for each stage (CLIP ctx, DiT velocity, DC-AE pixels). Anything under a few
//! e-3 is a pass for F32-vs-F32 modulo reduction order.

use hobby_rs::clip::{ClipConfig, ClipText};
use hobby_rs::dcae::Dcae;
use hobby_rs::dit::{Dit, DiTConfig};
use hobby_rs::imgops::Map;
use hobby_rs::st::SafeTensors;
use std::path::PathBuf;
use std::time::Instant;

fn maxdiff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut md = 0.0f32;
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > md {
            md = d;
        }
        sum += d;
    }
    (md, sum / a.len() as f32)
}

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "../image_weights".into()));
    let refs = SafeTensors::open(&dir.join("ref.safetensors")).expect("open ref.safetensors");
    let prompt = "a red cube on a wooden table, studio lighting";
    let neg = "blurry, low quality, watermark, signature, text, jpeg artifacts, deformed, distorted";

    println!("== CLIP ==");
    let clip = ClipText::load(&dir, ClipConfig::default()).expect("load clip");
    let ids = clip.tokenize(prompt);
    let (ref_ids, sh) = refs.get("ids_cond");
    println!("  tokens (first 12): {:?}", &ids[..12.min(ids.len())]);
    let ref_ids_u: Vec<u32> = ref_ids.iter().map(|&f| f as u32).collect();
    let id_match = ids.len() == ref_ids_u.len() && ids.iter().zip(&ref_ids_u).all(|(a, b)| a == b);
    println!("  ids shape {:?} | exact id match: {}", sh, id_match);
    if !id_match {
        println!("  REF ids (first 12): {:?}", &ref_ids_u[..12.min(ref_ids_u.len())]);
    }
    let ctx = clip.encode(&ids);
    let (ref_ctx, csh) = refs.get("ctx");
    let (md, mean) = maxdiff(&ctx, ref_ctx);
    println!("  ctx {:?} cond: max|d|={:.2e} mean|d|={:.2e}", csh, md, mean);

    // uncond branch
    let uids = clip.tokenize(neg);
    let uctx = clip.encode(&uids);
    let (ref_u, _) = refs.get("uncond");
    let (umd, umean) = maxdiff(&uctx, ref_u);
    println!("  ctx uncond:    max|d|={:.2e} mean|d|={:.2e}", umd, umean);

    println!("== DiT (velocity @ step 0, t=0) ==");
    let dcfg = DiTConfig::default();
    let lat = dcfg.latent_h;
    let w = 2 * dcfg.panel_w;
    let dit = Dit::load(&dir, dcfg).expect("load dit");
    // build two-panel input: left = z0, right = 0, +2 zero mask channels
    let (z0, _) = refs.get("z0"); // (1,32,lat,lat)
    let mut x = Map::zeros(34, lat, w);
    for c in 0..32 {
        for y in 0..lat {
            for xx in 0..lat {
                x.d[(c * lat + y) * w + xx] = z0[(c * lat + y) * lat + xx];
            }
        }
    }
    let m = ctx.len() / 768;
    let left = |full: &Map| -> Vec<f32> {
        // slice left panel (cols 0..lat) -> (32, lat, lat)
        let mut o = vec![0.0f32; 32 * lat * lat];
        for c in 0..32 {
            for y in 0..lat {
                for xx in 0..lat {
                    o[(c * lat + y) * lat + xx] = full.d[(c * lat + y) * w + xx];
                }
            }
        }
        o
    };
    let vc = left(&dit.forward(&x, 0.0, &ctx, m, 0));
    let (ref_vc, vsh) = refs.get("vc0");
    let (vmd, vmean) = maxdiff(&vc, ref_vc);
    println!("  vc0 {:?}: max|d|={:.2e} mean|d|={:.2e}", vsh, vmd, vmean);
    let vu = left(&dit.forward(&x, 0.0, &uctx, m, 0));
    let (ref_vu, _) = refs.get("vu0");
    let (umd2, umean2) = maxdiff(&vu, ref_vu);
    println!("  vu0:        max|d|={:.2e} mean|d|={:.2e}", umd2, umean2);

    println!("== DC-AE decode ==");
    let dst = SafeTensors::open(&dir.join("dcae_decoder.safetensors")).expect("dcae");
    let dcae = Dcae::new(&dst);
    let (dec_in, _) = refs.get("dec_in"); // (1,32,32,32)
    let latent = Map::from(32, 32, 32, dec_in.to_vec());
    let t0 = Instant::now();
    let px = dcae.decode(&latent); // (3,1024,1024)
    let (ref_px, psh) = refs.get("pixels");
    let (pmd, pmean) = maxdiff(&px.d, ref_px);
    println!("  pixels {:?}: max|d|={:.2e} mean|d|={:.2e} | decode {:.1}s",
             psh, pmd, pmean, t0.elapsed().as_secs_f32());
}
