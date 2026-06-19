//! Minimal Hugging Face file fetcher with a local cache. We roll our own (instead of hf-hub) because
//! hf-hub's sync `ureq` path is hardwired to native-tls (Windows SChannel), whose handshake to HF's
//! Cloudflare CDN gets reset in some environments; ureq's default rustls works reliably.

use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::PathBuf;

fn cache_root() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    base.join("hobby-chat").join("models")
}

/// Return a local path to `{repo}/{file}` from the main revision, downloading + caching on first use.
/// Streams large files (encoder weights are ~0.5–1.6 GB). Follows HF's redirect to the LFS CDN.
pub fn cached_file(repo: &str, file: &str) -> Result<PathBuf> {
    let dir = cache_root().join(repo.replace('/', "--"));
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(file);
    if dest.exists() && std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0) > 0 {
        return Ok(dest);
    }

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .build();
    let resp = agent.get(&url).call().with_context(|| format!("GET {url}"))?;
    if resp.status() != 200 {
        bail!("HF {url} -> HTTP {}", resp.status());
    }

    let tmp = dir.join(format!("{file}.part"));
    {
        let mut reader = resp.into_reader();
        let mut out = std::fs::File::create(&tmp)?;
        // chunked copy so huge files stream without buffering in memory
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut out, &buf[..n])?;
        }
    }
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}
