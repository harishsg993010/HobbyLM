"""LibriSpeech ASR dataset for stage-1 speech alignment (read English audiobooks, 16 kHz, transcripts).

Same pyarrow + soundfile streaming as vlm_audio_data.ClothoAudio (the `datasets` Audio loader pulls
torchcodec / breaks across pyarrow versions, so we read the parquet directly and decode the embedded
flac bytes ourselves). Each sample -> logical [SPEECH_TOKEN] + " " + transcript + [EOT]; we train
next-token, so the projector learns to map Whisper speech features to the spoken words. LibriSpeech is
already 16 kHz (Whisper's native rate), so no resampling.
"""
from __future__ import annotations

import tiktoken
import torch
from torch.utils.data import Dataset

from multimodal import SPEECH_TOKEN
from vlm_audio_data import audio_collate  # modality-agnostic: pads ids/targets, returns waveforms

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256


class LibriSpeechASR(Dataset):
    """Reads LibriSpeech parquet shards (config clean / split train.100 by default) directly via pyarrow."""
    def __init__(self, repo: str = "openslr/librispeech_asr", match=("train", "clean", "100"),
                 max_shards: int = 0, sr: int = 16000, max_tok: int = 120):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
        sel = sorted(f for f in files if all(m in f for m in match))
        if not sel:
            raise RuntimeError(f"no parquet in {repo} matching {match}; sample files: {files[:5]}")
        if max_shards:
            sel = sel[:max_shards]
        tabs = [pq.read_table(hf_hub_download(repo, f, repo_type="dataset")) for f in sel]
        self.table = pa.concat_tables(tabs) if len(tabs) > 1 else tabs[0]
        names = self.table.column_names
        self.audio_col = next(c for c in names if "audio" in c.lower())
        self.text_col = next(c for c in names if c.lower() in ("text", "transcript", "sentence")
                             or "text" in c.lower())
        self.ac = self.table.column(self.audio_col)
        self.tc = self.table.column(self.text_col)
        self.sr = sr
        self.max_tok = max_tok
        print(f"[LibriSpeechASR] {repo} shards={len(sel)} n={self.table.num_rows} "
              f"audio_col={self.audio_col} text_col={self.text_col}", flush=True)

    def __len__(self):
        return self.table.num_rows

    def _wav(self, i):
        import io
        import numpy as np
        import soundfile as sf
        a = self.ac[i].as_py()
        b = a["bytes"] if isinstance(a, dict) else a
        wav, sr = sf.read(io.BytesIO(b), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        return np.ascontiguousarray(wav)

    def raw(self, i):
        """(waveform float32 @ 16k, transcript str) for inference/inspection. Transcript title-cased
        back from LibriSpeech's ALL-CAPS so it reads naturally and tokenizes like normal text."""
        txt = str(self.tc[i].as_py()).strip()
        if txt.isupper():
            txt = txt.capitalize()
        return self._wav(i), txt

    def __getitem__(self, i):
        import numpy as np
        try:
            wav = self._wav(i)
        except Exception:
            wav = np.zeros(self.sr, dtype=np.float32)
        txt = str(self.tc[i].as_py()).strip()
        if txt.isupper():
            txt = txt.capitalize()
        ids_txt = ENC.encode_ordinary(" " + txt)[:self.max_tok] + [EOT]
        logical = [SPEECH_TOKEN] + ids_txt
        ids = torch.tensor(logical[:-1], dtype=torch.long)
        tgt = torch.tensor(logical[1:], dtype=torch.long)
        return torch.as_tensor(wav, dtype=torch.float32), ids, tgt


speech_collate = audio_collate
