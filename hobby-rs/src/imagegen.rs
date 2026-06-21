//! End-to-end text-to-image: CLIP text encode -> HobbyImageDiT CFG flow-matching Euler sampler ->
//! DC-AE decode -> RGB. Pure CPU, F32. This is the image counterpart of the text `Engine`.

use crate::clip::{ClipConfig, ClipText};
use crate::dcae::Dcae;
use crate::dit::{DiTConfig, Dit};
use crate::imgops::Map;
use crate::sample::Rng;
use crate::st::SafeTensors;
use anyhow::{Context, Result};
use std::path::Path;

pub const NEG_DEFAULT: &str =
    "blurry, low quality, watermark, signature, text, jpeg artifacts, deformed, distorted";

pub struct ImageEngine {
    clip: ClipText,
    dit: Dit,
    dcae_st: SafeTensors,
    lat: usize,
    lat_std: f32,
    sf: f32,
}

/// Pull a float field out of a tiny flat JSON file (dit_meta.json).
fn json_f32(s: &str, field: &str) -> Option<f32> {
    let pat = format!("\"{field}\"");
    let p = s.find(&pat)? + pat.len();
    let rest = &s[p..];
    let c = rest.find(':')? + 1;
    let rest = rest[c..].trim_start();
    let end = rest.find([',', '\n', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse::<f32>().ok()
}

impl ImageEngine {
    /// `dir` holds the exported weights (dit/clip/dcae safetensors + metas + clip vocab/merges).
    pub fn load(dir: &Path) -> Result<ImageEngine> {
        let meta = std::fs::read_to_string(dir.join("dit_meta.json")).context("read dit_meta.json")?;
        let lat_std = json_f32(&meta, "lat_std").unwrap_or(0.98046875);
        let sf = json_f32(&meta, "sf").unwrap_or(0.41407);
        let dcfg = DiTConfig::default();
        let lat = dcfg.latent_h;
        let clip = ClipText::load(dir, ClipConfig::default())?;
        let dit = Dit::load(dir, dcfg)?;
        let dcae_st = SafeTensors::open(&dir.join("dcae_decoder.safetensors"))?;
        Ok(ImageEngine { clip, dit, dcae_st, lat, lat_std, sf })
    }

    /// Resolution of the produced image (px).
    pub fn resolution(&self) -> usize {
        self.lat * 32
    }

    /// Generate one image. `neg` empty -> zero-embedding uncond. `progress(step, total)` is called
    /// each sampler step (for UI/CLI feedback). Returns (rgb bytes h*w*3, w, h).
    pub fn generate(&self, prompt: &str, neg: &str, steps: usize, cfg: f32, seed: u64,
                    mut progress: impl FnMut(usize, usize)) -> (Vec<u8>, usize, usize) {
        let lat = self.lat;
        let w = 2 * lat; // two-panel canvas width (in latent cells)
        let d_ctx = 768;
        // text conditioning
        let cond_ids = self.clip.tokenize(prompt);
        let ctx = self.clip.encode(&cond_ids);
        let m = ctx.len() / d_ctx;
        let uncond = if neg.is_empty() {
            vec![0.0f32; ctx.len()]
        } else {
            let uids = self.clip.tokenize(neg);
            self.clip.encode(&uids)
        };

        // init latent noise z ~ N(0,1), source panel zeros
        let mut rng = Rng::new(seed);
        let mut z = vec![0.0f32; 32 * lat * lat];
        let mut i = 0;
        while i < z.len() {
            // Box-Muller: two normals per iteration
            let u1 = rng.unif().max(1e-9);
            let u2 = rng.unif();
            let r = (-2.0 * u1.ln()).sqrt();
            z[i] = r * (std::f32::consts::TAU * u2).cos();
            if i + 1 < z.len() {
                z[i + 1] = r * (std::f32::consts::TAU * u2).sin();
            }
            i += 2;
        }

        // CFG flow-matching Euler sampler
        let build_input = |z: &[f32]| -> Map {
            // (34, lat, 2*lat): left = z, right = 0, +2 zero mask channels
            let mut x = Map::zeros(34, lat, w);
            for c in 0..32 {
                for y in 0..lat {
                    for xx in 0..lat {
                        x.d[(c * lat + y) * w + xx] = z[(c * lat + y) * lat + xx];
                    }
                }
            }
            x
        };
        let left = |full: &Map| -> Vec<f32> {
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
        for step in 0..steps {
            let tt = step as f32 / steps as f32;
            let inp = build_input(&z);
            let vc = left(&self.dit.forward(&inp, tt, &ctx, m, 0));
            let vu = left(&self.dit.forward(&inp, tt, &uncond, m, 0));
            for k in 0..z.len() {
                z[k] += (vu[k] + cfg * (vc[k] - vu[k])) / steps as f32;
            }
            progress(step + 1, steps);
        }

        // denormalize and decode
        for v in z.iter_mut() {
            *v = *v * self.lat_std / self.sf;
        }
        let latent = Map::from(32, lat, lat, z);
        let dcae = Dcae::new(&self.dcae_st);
        let px = dcae.decode(&latent); // (3, res, res), values ~[-1,1]
        let res = px.h;
        let mut rgb = vec![0u8; res * res * 3];
        let hw = res * res;
        for y in 0..res {
            for x in 0..res {
                for c in 0..3 {
                    let v = px.d[c * hw + y * res + x];
                    rgb[(y * res + x) * 3 + c] = (((v + 1.0) * 127.5).clamp(0.0, 255.0)) as u8;
                }
            }
        }
        (rgb, res, res)
    }
}
