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
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "var" / "audio"
SR = 16000
_LOAD_LOCK = threading.Lock()
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


# ---------------------------------------------------------------- clean-up before speech-to-text
# Phone microphones, a hand or a scarf over the mouth, a quiet voice or a noisy room make speech harder to
# recognise. Each step is optional; src/eval_voice_robustness.py measures which ones help (MITRAMIND_AUDIO_ENHANCE).
TARGET_RMS_DB = -20.0


def highpass(wav, cutoff: float = 70.0):
    """Remove rumble and hum below speech (fans, traffic, handling noise)."""
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(4, cutoff, "highpass", fs=SR, output="sos"), wav).astype("float32")


def normalise(wav, target_db: float = TARGET_RMS_DB):
    """Bring quiet or over-loud recordings to a standard speech level, without clipping."""
    import numpy as np

    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    if rms < 1e-6:
        return wav
    out = wav * (10 ** (target_db / 20) / rms)
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.99 if peak > 0.99 else out).astype("float32")


def denoise(wav, strength: float = 1.0):
    """Light spectral gating: estimate the noise floor from the quietest frames and subtract it."""
    import numpy as np
    from scipy.signal import istft, stft

    if len(wav) < SR // 4:
        return wav
    _, _, z = stft(wav, fs=SR, nperseg=512)
    mag, phase = np.abs(z), np.angle(z)
    energy = mag.mean(axis=0)
    quiet = mag[:, energy <= np.percentile(energy, 15)]
    noise = quiet.mean(axis=1, keepdims=True) if quiet.size else 0
    clean = np.maximum(mag - strength * noise, 0.15 * mag)  # keep a floor so speech does not sound robotic
    _, out = istft(clean * np.exp(1j * phase), fs=SR, nperseg=512)
    return out[: len(wav)].astype("float32")


ENHANCERS = {"highpass": highpass, "normalise": normalise, "denoise": denoise}


def enhance(wav, steps: str | None = None):
    """Apply the steps named in `steps` / MITRAMIND_AUDIO_ENHANCE (comma-separated, in order), e.g.
    "highpass,normalise". "none" or empty: unchanged."""
    steps = os.environ.get("MITRAMIND_AUDIO_ENHANCE", DEFAULT_ENHANCE) if steps is None else steps
    for name in [s.strip() for s in steps.split(",") if s.strip() and s.strip() != "none"]:
        wav = ENHANCERS[name](wav)
    return wav


# From outputs/voice_robustness/report.md (5 noise draws per clip): Whisper alone makes no errors on muffled, quiet,
# phone or 10 dB-noise audio. Only muffled + quiet + noisy together hurts; there volume normalisation was best:
# crisis sentences still rated high/urgent 18/20 (vs 16/20 with nothing), and all 20 at least raised an alert with a
# helpline (vs 18/20). The noise gate did not help on average, so it is available but off. "none" switches off.
DEFAULT_ENHANCE = "normalise"


class WhisperSTT:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.environ.get("MITRAMIND_STT_MODEL", "openai/whisper-small")
        self._pipe = None

    @property
    def name(self) -> str:
        return f"whisper:{self.model_id}"

    def _load(self):
        # The server warms up the text models in a background thread at start-up; transformers' lazy
        # `from transformers import pipeline` is not thread-safe and failed with ImportError when the first voice
        # message arrived during that warm-up. Import the submodule directly, once, under a lock.
        with _LOAD_LOCK:
            if self._pipe is None:
                import torch
                from transformers.pipelines import pipeline

                cuda = torch.cuda.is_available()
                self._pipe = pipeline("automatic-speech-recognition", model=self.model_id,
                                      device=0 if cuda else -1, torch_dtype=torch.float16 if cuda else torch.float32)
        return self._pipe

    def transcribe(self, wav_path: Path, language: str | None = None) -> dict:
        """language: None (auto-detect), 'hi' or 'en'. Hinglish speech is usually best left on auto."""
        return self.transcribe_audio(enhance(load_wav(wav_path)), language)

    def transcribe_audio(self, audio, language: str | None = None) -> dict:
        """16 kHz mono float array -> text (no clean-up here; transcribe() applies enhance())."""
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
