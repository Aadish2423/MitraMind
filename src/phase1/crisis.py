"""Crisis-detection layer: keyword rules + a classifier, independent of the LLM.

Three signals, combined by fixed rules (never by the LLM):
  1. English crisis lexicon tiers from m5_lexicon_text (negation- and first-person-aware).
  2. Hindi / Hinglish crisis phrases (lexicons/crisis_indic.yaml).
  3. A TF-IDF + logistic-regression classifier trained on Reddit MH (r/SuicideWatch vs other
     mental-health and control subreddits), see src/train_crisis_classifier.py. English only.
Plus the PHQ-9 item 9 answer, which always overrides text signals upwards.

Levels: none < watch < high < urgent. `high` and `urgent` escalate to a human; `urgent` means
call the patient now and give Tele-MANAS. Thresholds are starting points for the pilot.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

from .schema import EntryRisk, QuestionnaireScore, RiskAssessment

ROOT = Path(__file__).resolve().parents[2]
INDIC_LEXICON = ROOT / "lexicons" / "crisis_indic.yaml"
CLASSIFIER_DIR = ROOT / "models" / "crisis_classifier"

HELPLINES = ["Tele-MANAS (24x7, Govt. of India): 14416 or 1-800-891-4416",
             "Emergency: 112"]
LEVELS = ["none", "watch", "high", "urgent"]
ACTIONS = {
    "none": "No crisis signal. Continue routine check-ins.",
    "watch": "Low-level risk language. Review at next visit; increase check-in frequency.",
    "high": "Possible suicidal ideation. Care team to contact the patient within 24 hours and "
            "complete a structured risk assessment (e.g. C-SSRS screen).",
    "urgent": "Possible suicidal behaviour or plan. Call the patient now; share Tele-MANAS 14416; "
              "follow the hospital's emergency protocol.",
}


@lru_cache(maxsize=1)
def _indic_patterns() -> list[tuple[int, str, re.Pattern]]:
    doc = yaml.safe_load(INDIC_LEXICON.read_text(encoding="utf-8"))
    return [(spec["tier"], name, re.compile(p)) for name, spec in doc["items"].items() for p in spec["patterns"]]


def indic_hits(text: str) -> list[tuple[int, str]]:
    # NFC turns precomposed nukta letters (e.g. U+0959 ख़) into letter + nukta, the form the lexicon uses.
    low = unicodedata.normalize("NFC", (text or "").lower())
    return [(tier, m.group(0)) for tier, _, pat in _indic_patterns() if (m := pat.search(low))]


class CrisisClassifier:
    """Loads models/crisis_classifier/{model.joblib,thresholds.json} if trained; otherwise inert."""

    def __init__(self, directory: Path = CLASSIFIER_DIR):
        self.model, self.thresholds = None, {}
        path = directory / "model.joblib"
        if path.exists():
            import joblib

            self.model = joblib.load(path)
            self.thresholds = json.loads((directory / "thresholds.json").read_text())

    @property
    def available(self) -> bool:
        return self.model is not None

    MIN_WORDS = 8  # trained on Reddit posts (median ~130 words); very short check-ins are out of range

    def prob(self, text: str) -> float | None:
        if not self.available or not text or len(text.split()) < self.MIN_WORDS:
            return None
        return float(self.model.predict_proba([text])[0, 1])


def _max_level(*levels: str) -> str:
    return max(levels, key=LEVELS.index)


def assess_entry(entry_id: str, text: str, lexicon_cues: dict | None, classifier: CrisisClassifier,
                 language: str) -> EntryRisk:
    tier = int(lexicon_cues.get("cue_risk_tier", 0)) if lexicon_cues else 0
    terms = [t for t in (lexicon_cues or {}).get("cue_risk_terms", "").split("; ") if t and not t.startswith("means:")]
    indic = indic_hits(text) if language in ("hi", "hinglish", "auto") else []
    tier = max([tier] + [t for t, _ in indic])
    prob = classifier.prob(text) if language in ("en", "hinglish") else None
    sentiment = (lexicon_cues or {}).get("cue_sent_compound")

    reasons, level = [], "none"
    if tier >= 3:
        level = "urgent"
        reasons.append("crisis phrases at behaviour/attempt tier")
    elif tier == 2:
        level = "high"
        reasons.append("suicidal-ideation phrases")
    elif tier == 1:
        level = "watch"
        reasons.append("hopelessness / burden phrases")
    if prob is not None and classifier.thresholds:
        high = classifier.thresholds.get("high")
        if high is not None and prob >= high:
            level = _max_level(level, "high")
            reasons.append(f"classifier p={prob:.2f} >= high threshold")
        elif prob >= classifier.thresholds["watch"] and (sentiment is None or sentiment < -0.05):
            # The classifier alone only raises "watch" when the check-in's tone is also negative;
            # on short, positive check-ins its word statistics are unreliable.
            level = _max_level(level, "watch")
            reasons.append(f"classifier p={prob:.2f} >= watch threshold, negative tone")
    return EntryRisk(entry_id=entry_id, level=level, lexicon_tier=tier, lexicon_terms=terms,
                     indic_terms=[t for _, t in indic], classifier_prob=None if prob is None else round(prob, 4),
                     reasons=reasons)


def combine(entry_risks: list[EntryRisk], phq9_history: list[QuestionnaireScore],
            phq9_change: int | None) -> RiskAssessment:
    level, reasons = "none", []
    flagged = [r for r in entry_risks if r.level != "none"]
    for r in flagged:
        level = _max_level(level, r.level)
    if flagged:
        worst = max(flagged, key=lambda r: LEVELS.index(r.level))
        reasons.append(f"{len(flagged)} check-in(s) with crisis language; highest '{worst.level}' "
                       f"({', '.join(worst.reasons)})")

    item9 = phq9_history[-1].item9 if phq9_history else None
    if item9 is not None and item9 >= 2:
        level = _max_level(level, "urgent")
        reasons.append(f"PHQ-9 item 9 = {item9} (thoughts of self-harm on more than half the days)")
    elif item9 == 1:
        level = _max_level(level, "high")
        reasons.append("PHQ-9 item 9 = 1 (thoughts of self-harm on several days)")
    if phq9_change is not None and phq9_change >= 5:
        level = _max_level(level, "watch")
        reasons.append(f"PHQ-9 worsened by {phq9_change} points")
    return RiskAssessment(
        level=level, escalate=level in ("high", "urgent"), reasons=reasons or ["no crisis signal"],
        recommended_action=ACTIONS[level], helplines=HELPLINES if level != "none" else [],
        flagged_entries=flagged, phq9_item9_latest=item9)
