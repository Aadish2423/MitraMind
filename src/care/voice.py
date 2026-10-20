"""Voice check-ins: save the clip, convert to 16 kHz mono WAV, transcribe.

Speech-to-text backends (MITRAMIND_STT):
  whisper  local Hugging Face Whisper (default openai/whisper-small, multilingual, MIT licence;
           set MITRAMIND_STT_MODEL, e.g. an IndicWhisper / fine-tuned Hindi checkpoint). Runs on the GPU.
  sarvam   Sarvam AI "saarika" STT over HTTPS (needs SARVAM_API_KEY; data leaves the hospital, so
           use only with a DPA and an Indian region). Not tested here (no key).
  none     voice disabled.
The voice emotion models (m3 / m4) run on the saved WAV inside the Phase 1 orchestrator.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "var" / "audio"
SR = 16000
EXT = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/webm": ".webm", "audio/ogg": ".ogg",
       "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/aac": ".aac", "audio/amr": ".amr"}


class VoiceError(Exception):
    pass


def ffmpeg_path() -> str | None:
    return os.environ.get("MITRAMIND_FFMPEG") or shutil.which("ffmpeg")


def save_clip(patient_id: str, data: bytes, mime: str) -> Path:
    """Write the upload and return a 16 kHz mono WAV path (converted with ffmpeg if needed)."""
    if not data:
        raise VoiceError("empty audio")
    folder = AUDIO_DIR / patient_id
    folder.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    raw = folder / f"{stem}{EXT.get(mime.split(';')[0].strip(), '.bin')}"
    raw.write_bytes(data)
    wav = folder / f"{stem}.16k.wav"
    ff = ffmpeg_path()
    if ff:
        res = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-ac", "1", "-ar", str(SR), str(wav)],
                             capture_output=True, timeout=120)
        if res.returncode != 0:
            raise VoiceError(f"could not decode audio: {res.stderr.decode(errors='ignore')[:200]}")
        raw.unlink(missing_ok=True)
        return wav
    if raw.suffix != ".wav":
        raise VoiceError("ffmpeg is not installed, so only WAV uploads can be read")
    raw.rename(wav)
    return wav


def load_wav(path: Path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = resample_poly(wav, SR, sr).astype(np.float32)
    return wav


class WhisperSTT:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.environ.get("MITRAMIND_STT_MODEL", "openai/whisper-small")
        self._pipe = None

    @property
    def name(self) -> str:
        return f"whisper:{self.model_id}"

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            cuda = torch.cuda.is_available()
            self._pipe = pipeline("automatic-speech-recognition", model=self.model_id,
                                  device=0 if cuda else -1, torch_dtype=torch.float16 if cuda else torch.float32)
        return self._pipe

    def transcribe(self, wav_path: Path, language: str | None = None) -> dict:
        """language: None (auto-detect), 'hi' or 'en'. Hinglish speech is usually best left on auto."""
        audio = load_wav(wav_path)
        kwargs = {"task": "transcribe"}
        if language in ("hi", "en"):
            kwargs["language"] = {"hi": "hindi", "en": "english"}[language]
        out = self._load()({"raw": audio, "sampling_rate": SR}, generate_kwargs=kwargs,
                           return_timestamps=len(audio) > 30 * SR)
        return {"text": out["text"].strip(), "seconds": round(len(audio) / SR, 1), "engine": self.name}


class SarvamSTT:
    URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self):
        self.key = os.environ.get("SARVAM_API_KEY")
        self.model = os.environ.get("SARVAM_STT_MODEL", "saarika:v2.5")

    @property
    def name(self) -> str:
        return f"sarvam:{self.model}"

    def transcribe(self, wav_path: Path, language: str | None = None) -> dict:
        import requests

        if not self.key:
            raise VoiceError("SARVAM_API_KEY is not set")
        lang = {"hi": "hi-IN", "en": "en-IN"}.get(language or "", "unknown")
        with open(wav_path, "rb") as f:
            r = requests.post(self.URL, headers={"api-subscription-key": self.key},
                              files={"file": (wav_path.name, f, "audio/wav")},
                              data={"model": self.model, "language_code": lang}, timeout=120)
        r.raise_for_status()
        return {"text": r.json().get("transcript", "").strip(), "seconds": None, "engine": self.name}


def make_stt():
    kind = os.environ.get("MITRAMIND_STT", "whisper")
    if kind == "none":
        return None
    return SarvamSTT() if kind == "sarvam" else WhisperSTT()
