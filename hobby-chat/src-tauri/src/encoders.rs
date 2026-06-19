//! Pure-Rust local multimodal encoders via candle (no ONNX/DLL). SigLIP2 vision tower + Whisper-small
//! encoder run natively; weights auto-download from the HF hub on first use and cache. We then apply
//! the model's own trained projectors (mlp2x_gelu: Linear -> GELU -> Linear) to land in the 768-d MoE
//! embedding space, exactly like the precomputed *_embeds.bin — but for arbitrary images / live audio.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::{siglip, whisper};

const D_MODEL: usize = 768; // MoE embedding dim
const WHISPER_SAMPLES: usize = 16000 * 30; // 30 s window

// Projectors + mel filterbank are tiny — bake them into the binary.
const VISION_PROJ: &[u8] = include_bytes!("../assets/vision_projector.safetensors");
const SPEECH_PROJ: &[u8] = include_bytes!("../assets/speech_projector.safetensors");
const MEL_FILTERS: &[u8] = include_bytes!("../assets/melfilters.bytes");

/// mlp2x_gelu connector: Linear -> GELU(erf) -> Linear. Keys: net.0.{weight,bias}, net.2.{weight,bias}.
struct Projector {
    l0: Linear,
    l2: Linear,
}

impl Projector {
    fn load(bytes: &'static [u8], in_dim: usize, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(bytes.to_vec(), DType::F32, device)?;
        let l0 = candle_nn::linear(in_dim, D_MODEL, vb.pp("net.0"))?;
        let l2 = candle_nn::linear(D_MODEL, D_MODEL, vb.pp("net.2"))?;
        Ok(Projector { l0, l2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // nn.GELU() default is the exact (erf) variant — match it.
        let h = self.l0.forward(x)?.gelu_erf()?;
        Ok(self.l2.forward(&h)?)
    }
}

fn rows_of(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    // t: (1, N, D) -> Vec<N> of Vec<D>
    let t = t.squeeze(0)?.to_dtype(DType::F32)?;
    Ok(t.to_vec2::<f32>()?)
}

// ---------------- vision ----------------

pub struct VisionEncoder {
    model: siglip::VisionModel,
    proj: Projector,
    device: Device,
}

impl VisionEncoder {
    pub fn load(device: &Device) -> Result<Self> {
        let repo = "google/siglip2-so400m-patch16-512";
        let cfg_file = crate::hf::cached_file(repo, "config.json").context("download siglip config")?;
        let cfg: siglip::Config = serde_json::from_slice(&std::fs::read(cfg_file)?)?;
        let weights = crate::hf::cached_file(repo, "model.safetensors").context("download siglip weights")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
        let model = siglip::VisionModel::new(&cfg.vision_config, false, vb.pp("vision_model"))?;
        let proj = Projector::load(VISION_PROJ, cfg.vision_config.hidden_size, device)?;
        Ok(VisionEncoder { model, proj, device: device.clone() })
    }

    /// Load an image file, run the vision tower + projector, return ~1024 rows of 768.
    pub fn encode(&self, path: &str) -> Result<Vec<Vec<f32>>> {
        let img = image::open(path).context("open image")?.into_rgb8();
        self.encode_rgb(img)
    }

    /// Same, from in-memory image bytes (PNG/JPG/…) — used by the OpenAI API (data-URI / fetched images).
    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<Vec<f32>>> {
        let img = image::load_from_memory(bytes).context("decode image")?.into_rgb8();
        self.encode_rgb(img)
    }

    /// Preprocess (resize 512², normalize to [-1,1], CHW), run the vision tower + projector.
    fn encode_rgb(&self, img: image::RgbImage) -> Result<Vec<Vec<f32>>> {
        let img = image::imageops::resize(&img, 512, 512, image::imageops::FilterType::CatmullRom);
        let mut data = vec![0f32; 3 * 512 * 512];
        for y in 0..512usize {
            for x in 0..512usize {
                let p = img.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    data[c * 512 * 512 + y * 512 + x] = p[c] as f32 * 2.0 / 255.0 - 1.0;
                }
            }
        }
        let pixel_values = Tensor::from_vec(data, (1, 3, 512, 512), &self.device)?;
        let feats = self.model.forward(&pixel_values)?; // (1, 1024, 1152)
        let proj = self.proj.forward(&feats)?; // (1, 1024, 768)
        rows_of(&proj)
    }
}

// ---------------- speech ----------------

pub struct SpeechEncoder {
    encoder: whisper::model::AudioEncoder,
    config: whisper::Config,
    proj: Projector,
    mel_filters: Vec<f32>,
    device: Device,
}

impl SpeechEncoder {
    pub fn load(device: &Device) -> Result<Self> {
        let repo = "openai/whisper-small";
        let cfg_file = crate::hf::cached_file(repo, "config.json").context("download whisper config")?;
        let cfg: whisper::Config = serde_json::from_slice(&std::fs::read(cfg_file)?)?;
        let weights = crate::hf::cached_file(repo, "model.safetensors").context("download whisper weights")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], whisper::DTYPE, device)? };
        // load the full model but keep only the encoder
        let whole = whisper::model::Whisper::load(&vb, cfg.clone())?;
        let encoder = whole.encoder;

        let mut mel_filters = vec![0f32; MEL_FILTERS.len() / 4];
        for (i, c) in MEL_FILTERS.chunks_exact(4).enumerate() {
            mel_filters[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }

        let proj = Projector::load(SPEECH_PROJ, cfg.d_model * 2, device)?; // stack2 -> 1536
        Ok(SpeechEncoder { encoder, config: cfg, proj, mel_filters, device: device.clone() })
    }

    /// pcm: mono f32 @ 16 kHz. mel -> Whisper encoder -> stack adjacent frames (1500->750) -> projector.
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        let mut samples = vec![0f32; WHISPER_SAMPLES];
        let n = pcm.len().min(WHISPER_SAMPLES);
        samples[..n].copy_from_slice(&pcm[..n]);

        let mel = whisper::audio::pcm_to_mel(&self.config, &samples, &self.mel_filters);
        let n_mels = self.config.num_mel_bins;
        let frames = mel.len() / n_mels;
        // candle's pcm_to_mel appends a padding chunk (3000 -> 4500); the encoder wants exactly the
        // 30 s window (= max_source_positions*2 = 3000 mel frames -> 1500 encoder frames).
        let want = (self.config.max_source_positions * 2).min(frames);
        let mel_t = Tensor::from_vec(mel, (1, n_mels, frames), &self.device)?.narrow(2, 0, want)?;

        let h = self.encoder.forward(&mel_t, true)?; // (1, 1500, 768)
        let (b, t, c) = h.dims3()?;
        let t2 = t - (t % 2);
        let h = h.narrow(1, 0, t2)?.reshape((b, t2 / 2, c * 2))?; // (1, 750, 1536)
        let proj = self.proj.forward(&h)?; // (1, 750, 768)
        rows_of(&proj)
    }
}

// ---------------- lazy holder ----------------

/// Lazily-initialized encoders. The big encoder weights download from HF on first use and cache;
/// the small projectors are baked in. CPU device (the app is a local CPU build).
pub struct Encoders {
    device: Device,
    vision: Option<VisionEncoder>,
    speech: Option<SpeechEncoder>,
}

impl Encoders {
    pub fn new() -> Self {
        Encoders { device: Device::Cpu, vision: None, speech: None }
    }

    pub fn encode_image(&mut self, path: &str) -> Result<Vec<Vec<f32>>> {
        if self.vision.is_none() {
            self.vision = Some(VisionEncoder::load(&self.device)?);
        }
        self.vision.as_ref().unwrap().encode(path)
    }

    pub fn encode_image_bytes(&mut self, bytes: &[u8]) -> Result<Vec<Vec<f32>>> {
        if self.vision.is_none() {
            self.vision = Some(VisionEncoder::load(&self.device)?);
        }
        self.vision.as_ref().unwrap().encode_bytes(bytes)
    }

    pub fn encode_speech(&mut self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        if self.speech.is_none() {
            self.speech = Some(SpeechEncoder::load(&self.device)?);
        }
        self.speech.as_mut().unwrap().encode(pcm)
    }
}
