"""Shared paths, the MitraMind cue schema, and CSV/metric helpers for the four cue models."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

# Every model's native labels are folded into these cues so the State Card sees one vocabulary.
CUES = ["anxious", "low_mood", "angry", "positive", "surprised", "neutral"]
DISTRESS_CUES = ("anxious", "low_mood", "angry")

# Native label -> MitraMind cue. Edit here to change how a dataset feeds the State Card.
CUE_MAP = {
    "hinemo9": {
        "Fear": "anxious", "Sadness": "low_mood", "Nostalgia": "low_mood", "Anger": "angry",
        "Joy": "positive", "Love": "positive", "Devotion": "positive",
        "Surprise": "surprised", "Neutral": "neutral",
    },
    "emoinhindi": {
        "apprehensive": "anxious", "fear": "anxious", "confused": "anxious",
        "sad": "low_mood", "guilty": "low_mood",
        "anger": "angry", "annoyed": "angry", "disgusted": "angry",
        "joy": "positive", "hopeful": "positive", "confident": "positive", "grateful": "positive",
        "compassion": "positive", "impressed": "positive",
        "surprised": "surprised", "anticipation": "neutral", "neutral": "neutral",
    },
    "cremad": {
        "ANG": "angry", "DIS": "angry", "FEA": "anxious",
        "HAP": "positive", "NEU": "neutral", "SAD": "low_mood",
    },
    "msp": {
        "Angry": "angry", "Disgust": "angry", "Contempt": "angry", "Fear": "anxious",
        "Sad": "low_mood", "Happy": "positive", "Surprise": "surprised", "Neutral": "neutral",
    },
}

CSV_COLUMNS = [
    "model_id", "train_data", "eval_data", "sample_id", "modality", "language_or_speaker", "input",
    "true_label", "true_cue", "pred_label", "pred_confidence", "top3", "pred_cue",
    *[f"p_{c}" for c in CUES], "distress_score", "arousal", "valence", "dominance", "correct",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_device(need_gb: float, env: str = "MITRAMIND_SPEECH_DEVICE"):
    """GPU only if it has `need_gb` free right now, else CPU. On an 8 GB card the chat LLM (kept loaded by Ollama),
    the text models and Whisper can leave too little room; filling the card made a model load fail with
    'CUDA error: illegal memory access', which then broke every later GPU call. `env` (cpu / cuda / auto) overrides."""
    import os

    choice = os.environ.get(env, "auto").lower()
    if choice in ("cpu", "cuda"):
        return torch.device(choice if choice == "cpu" or torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free, _ = torch.cuda.mem_get_info()
    return torch.device("cuda" if free / 1024 ** 3 >= need_gb else "cpu")


def autocast_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def cue_scores(labels, probs, cue_map, multilabel=False):
    """Fold per-label probabilities into cue probabilities and a 0-1 distress score.

    Single-label (softmax) models sum the probabilities of labels sharing a cue. Multi-label
    (sigmoid) models treat labels as independent, so a cue's score is P(at least one is present).
    """
    if multilabel:
        absent = {c: 1.0 for c in CUES}
        for lab, p in zip(labels, probs):
            absent[cue_map[lab]] *= 1.0 - float(p)
        scores = {c: 1.0 - absent[c] for c in CUES}
        distress = 1.0 - np.prod([1.0 - scores[c] for c in DISTRESS_CUES])
    else:
        scores = {c: 0.0 for c in CUES}
        for lab, p in zip(labels, probs):
            scores[cue_map[lab]] += float(p)
        distress = sum(scores[c] for c in DISTRESS_CUES)
    return scores, float(distress)


def cue_row(*, model_id, train_data, eval_data, sample_id, modality, group, inp, labels, probs,
            cue_map, true_label, true_cue, correct, multilabel=False, pred_label=None,
            arousal=None, valence=None, dominance=None):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs)
    scores, distress = cue_scores(labels, probs, cue_map, multilabel)
    if pred_label is None:
        pred_label = labels[order[0]]
    row = {
        "model_id": model_id, "train_data": train_data, "eval_data": eval_data,
        "sample_id": sample_id, "modality": modality, "language_or_speaker": group, "input": inp,
        "true_label": true_label, "true_cue": true_cue,
        "pred_label": pred_label, "pred_confidence": round(float(probs[order[0]]), 4),
        "top3": "; ".join(f"{labels[i]}:{probs[i]:.3f}" for i in order[:3]),
        "pred_cue": max(scores, key=scores.get),
        **{f"p_{c}": round(scores[c], 4) for c in CUES},
        "distress_score": round(distress, 4),
        "arousal": arousal, "valence": valence, "dominance": dominance,
        "correct": int(correct),
    }
    return row


def write_cues(rows, model_id):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / f"{model_id}_cues.csv"
    # utf-8-sig so Excel shows Devanagari text correctly.
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    return path.relative_to(ROOT)


def label_metrics(y_true, y_pred, labels):
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def write_metrics(metrics, model_id):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / f"{model_id}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path.relative_to(ROOT)
