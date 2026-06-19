"""Spoken-QA dataset (VoiceAssistant-400K) for speech instruction tuning.

Each row: question_audio (the user's SPOKEN question) + answer (the assistant's text reply). We splice
the Whisper features of question_audio at SPEECH_TOKEN and train the model to generate the answer — i.e.
*answer about spoken content*, not transcribe it. Same pyarrow + soundfile streaming as the other audio
loaders; question audio is resampled to 16 kHz (Whisper's rate). The repetitive "identity" persona rows
are dropped so the model doesn't overfit "Hello! I'm Omni...".
"""
from __future__ import annotations

import tiktoken
import torch
from torch.utils.data import Dataset

from multimodal import SPEECH_TOKEN
from vlm_audio_data import audio_collate

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256


class VoiceAssistantQA(Dataset):
    def __init__(self, repo: str = "gpt-omni/VoiceAssistant-400K", max_shards: int = 4,
                 sr: int = 16000, max_ans: int = 96, drop_identity: bool = True):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
        files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
        if max_shards:
            files = files[:max_shards]
        if not files:
            raise RuntimeError(f"no parquet shards in {repo}")
        # keep only the columns we need (audio + answer + split) so we don't load the giant SNAC strings
        keep = None
        tabs = []
        for f in files:
            t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"))
            if keep is None:
                names = t.column_names
                acol = next(c for c in names if "audio" in c.lower())
                ans = next(c for c in names if c.lower() == "answer")
                qcol = next((c for c in names if c.lower() == "question"), None)
                scol = next((c for c in names if "split" in c.lower()), None)
                keep = [c for c in (acol, ans, qcol, scol) if c]
            tabs.append(t.select(keep))
        self.table = pa.concat_tables(tabs) if len(tabs) > 1 else tabs[0]
        self.audio_col, self.ans_col = keep[0], keep[1]
        self.q_col = next((c for c in keep[2:] if c.lower() == "question"), None)
        self.split_col = next((c for c in keep[2:] if "split" in c.lower()), None)
        self.qc = self.table.column(self.q_col) if self.q_col else None
        # row indices to use (skip persona/identity rows)
        self.idx = list(range(self.table.num_rows))
        if drop_identity and self.split_col is not None:
            sc = self.table.column(self.split_col)
            self.idx = [i for i in self.idx if str(sc[i].as_py()).lower() != "identity"]
        self.ac = self.table.column(self.audio_col)
        self.anc = self.table.column(self.ans_col)
        self.sr = sr
        self.max_ans = max_ans
        print(f"[VoiceAssistantQA] {repo} shards={len(files)} rows={self.table.num_rows} "
              f"used={len(self.idx)} audio_col={self.audio_col} ans_col={self.ans_col}", flush=True)

    def __len__(self):
        return len(self.idx)

    def _wav(self, row):
        import io
        import numpy as np
        import soundfile as sf
        a = self.ac[row].as_py()
        b = a["bytes"] if isinstance(a, dict) else a
        wav, sr = sf.read(io.BytesIO(b), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        return np.ascontiguousarray(wav)

    def raw(self, i):
        """(waveform @16k, answer str, question str) — question text is for display only (input is audio)."""
        row = self.idx[i]
        q = str(self.qc[row].as_py()).strip() if self.qc is not None else ""
        return self._wav(row), str(self.anc[row].as_py()).strip(), q

    def __getitem__(self, i):
        import numpy as np
        row = self.idx[i]
        try:
            wav = self._wav(row)
        except Exception:
            wav = np.zeros(self.sr, dtype=np.float32)
        ans = str(self.anc[row].as_py()).strip()
        ans_ids = ENC.encode_ordinary(" " + ans)[:self.max_ans] + [EOT]
        logical = [SPEECH_TOKEN] + ans_ids                 # speak the question -> generate the answer
        ids = torch.tensor(logical[:-1], dtype=torch.long)
        tgt = torch.tensor(logical[1:], dtype=torch.long)
        return torch.as_tensor(wav, dtype=torch.float32), ids, tgt


va_collate = audio_collate
