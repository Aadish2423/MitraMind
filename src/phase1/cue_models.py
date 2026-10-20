"""Adapters that run the five cue models on one check-in, with language/modality routing.

  m1_hindi_text     Devanagari Hindi text       XLM-R (models/m1_hindi_text, from src/train_text.py)
  m2_hinglish_text  Hinglish + English text     XLM-R (models/m2_hinglish_text)
  m3_cremad_speech  voice clip                  WavLM-base-plus (models/m3_cremad_speech, src/train_cremad.py)
  m4_msp_speech     voice clip                  MSP-Podcast WavLM-large (downloaded from HF on first use)
  m5_lexicon_text   English / Hinglish text     rule + lexicon cues (src/lexicon_cues), always available

Neural models load lazily and only if their weights and libraries are present; a missing model
is reported in the card (`models_unavailable`) instead of failing the pipeline.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ROOT = SRC.parent
MODELS = ROOT / "models"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nahin", "nhi", "mujhe", "mera", "meri", "mere", "kya", "bahut", "bohot", "kuch",
    "aaj", "kal", "raha", "rahi", "rahe", "tha", "thi", "hoon", "hu", "kyun", "kyunki", "lekin", "aur",
    "yaar", "ghar", "kaam", "neend", "dil", "mann", "accha", "achha", "pata", "sab", "bhi", "toh", "ko",
}
EMOTION_CUES = ["anxious", "low_mood", "angry", "positive", "surprised", "neutral"]


def detect_language(text: str) -> str:
    if not text:
        return "en"
    letters = [c for c in text if c.isalpha()]
    if letters and sum(bool(DEVANAGARI.match(c)) for c in letters) / len(letters) > 0.3:
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(w in HINGLISH_MARKERS for w in words)
    return "hinglish" if words and (hits >= 3 or hits / len(words) > 0.15) else "en"


class Unavailable(Exception):
    pass


class LexiconModel:
    model_id = "m5_lexicon_text"

    def __init__(self):
        from lexicon_cues.extractor import CueExtractor

        self.extractor = CueExtractor()

    def run(self, text: str) -> dict:
        return self.extractor.extract(text)


class HFTextEmotionModel:
    """m1 / m2: XLM-R sequence classifiers saved by src/train_text.py (HinEmo-9 labels)."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        path = MODELS / model_id
        if not (path / "config.json").exists():
            raise Unavailable(f"no weights in models/{model_id} (train with src/train_text.py)")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise Unavailable(f"missing library: {e.name}") from e
        from common import CUE_MAP, device

        self.torch, self.cue_map, self.device = torch, CUE_MAP["hinemo9"], device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def run(self, text: str) -> dict:
        from common import cue_scores

        enc = self.tok([text], truncation=True, max_length=96, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**enc).logits.float(), 1)[0].cpu().numpy()
        scores, distress = cue_scores(self.labels, probs, self.cue_map)
        top = int(probs.argmax())
        return {"pred_label": self.labels[top], "pred_confidence": round(float(probs[top]), 4),
                "cue_probs": {c: round(scores[c], 4) for c in EMOTION_CUES}, "distress_score": round(distress, 4)}


class CremadSpeechModel:
    model_id = "m3_cremad_speech"

    def __init__(self):
        path = MODELS / self.model_id
        if not (path / "config.json").exists():
            raise Unavailable("no weights in models/m3_cremad_speech (train with src/train_cremad.py)")
        try:
            import torch
            from transformers import AutoFeatureExtractor, WavLMForSequenceClassification
        except ImportError as e:
            raise Unavailable(f"missing library: {e.name}") from e
        from common import CUE_MAP, device

        self.torch, self.cue_map, self.device = torch, CUE_MAP["cremad"], device()
        self.fe = AutoFeatureExtractor.from_pretrained(path)
        self.model = WavLMForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def run(self, audio_path: str) -> dict:
        from common import cue_scores
        from train_cremad import load_audio

        wav = load_audio(Path(audio_path))
        enc = self.fe(wav, sampling_rate=16000, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**enc).logits.float(), 1)[0].cpu().numpy()
        scores, distress = cue_scores(self.labels, probs, self.cue_map)
        top = int(probs.argmax())
        return {"pred_label": self.labels[top], "pred_confidence": round(float(probs[top]), 4),
                "cue_probs": {c: round(scores[c], 4) for c in EMOTION_CUES}, "distress_score": round(distress, 4)}


class MspSpeechModel:
    """m4: categorical + arousal/valence/dominance models; m4's arousal is its reliable signal."""
    model_id = "m4_msp_speech"

    def __init__(self, allow_download: bool = False):
        try:
            import torch  # noqa: F401
            from huggingface_hub import try_to_load_from_cache
            from run_msp import REPOS, load_ser
        except ImportError as e:
            raise Unavailable(f"missing library: {e.name}") from e
        cached = all(isinstance(try_to_load_from_cache(r, "model.safetensors"), str) for r in REPOS.values())
        if not cached and not allow_download:
            raise Unavailable("MSP models not in the HF cache (run src/run_msp.py once, or allow_download)")
        self.cat, self.cat_cfg = load_ser(REPOS["categorical"])
        self.dim, self.dim_cfg = load_ser(REPOS["dimensional"])
        self.labels = [self.cat_cfg["id2label"][str(i)] for i in range(self.cat_cfg["num_classes"])]

    def run(self, audio_path: str) -> dict:
        import torch
        from common import CUE_MAP, cue_scores
        from run_msp import run
        from train_cremad import load_audio

        wav = load_audio(Path(audio_path))
        probs = torch.softmax(torch.from_numpy(run(self.cat, self.cat_cfg, [wav])), 1)[0].numpy()
        aro, dom, val = run(self.dim, self.dim_cfg, [wav])[0]
        scores, distress = cue_scores(self.labels, probs, CUE_MAP["msp"])
        top = int(probs.argmax())
        return {"pred_label": self.labels[top], "pred_confidence": round(float(probs[top]), 4),
                "cue_probs": {c: round(scores[c], 4) for c in EMOTION_CUES}, "distress_score": round(distress, 4),
                "arousal": round(float(aro), 4), "valence": round(float(val), 4), "dominance": round(float(dom), 4)}


class CueModelRegistry:
    """Loads each model once, on first need, and remembers why a model is unavailable."""

    def __init__(self, enable_neural: bool = True, allow_download: bool = False):
        self.enable_neural, self.allow_download = enable_neural, allow_download
        self._models: dict[str, object] = {}
        self.unavailable: dict[str, str] = {}

    def get(self, model_id: str):
        if model_id in self._models:
            return self._models[model_id]
        if model_id in self.unavailable:
            return None
        if model_id != "m5_lexicon_text" and not self.enable_neural:
            self.unavailable[model_id] = "neural models disabled"
            return None
        try:
            model = {
                "m5_lexicon_text": LexiconModel,
                "m1_hindi_text": lambda: HFTextEmotionModel("m1_hindi_text"),
                "m2_hinglish_text": lambda: HFTextEmotionModel("m2_hinglish_text"),
                "m3_cremad_speech": CremadSpeechModel,
                "m4_msp_speech": lambda: MspSpeechModel(self.allow_download),
            }[model_id]()
        except Unavailable as e:
            self.unavailable[model_id] = str(e)
            return None
        self._models[model_id] = model
        return model

    @staticmethod
    def route(modality: str, language: str) -> list[str]:
        if modality == "audio":
            return ["m3_cremad_speech", "m4_msp_speech"]
        if language == "hi":
            return ["m1_hindi_text"]
        # m2 is trained on Hinglish + English rows; the lexicon reads English words in both.
        return ["m5_lexicon_text", "m2_hinglish_text"]
