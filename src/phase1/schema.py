"""User State Card schema (Pydantic v2).

Only `Narrative` is written by the LLM, under JSON-schema-constrained decoding. Everything
else (questionnaire scores, risk, cue aggregates) is computed in code, so a model can never
lower a risk level or invent a PHQ-9 score.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

StressorDomain = Literal["economic", "work", "school_exam", "family", "relationships", "health",
                         "isolation", "substance_use", "domestic", "covid19", "other"]
RiskLevel = Literal["none", "watch", "high", "urgent"]


# --------------------------------------------------------------------- inputs
class Entry(BaseModel):
    """One patient check-in between visits: a chat message, a journal note or a voice clip."""
    entry_id: str
    timestamp: datetime
    text: str | None = None
    audio_path: str | None = None
    transcript: str | None = Field(None, description="speech-to-text of audio_path, if available")
    language: Literal["auto", "en", "hinglish", "hi"] = "auto"
    shared_with_clinician: bool = Field(
        False, description="Reflection Share: only shared entries' text is shown to the LLM / clinician")


class QuestionnaireResponse(BaseModel):
    instrument: Literal["PHQ-9", "GAD-7"]
    date: date
    items: list[int] = Field(description="item scores 0-3, in instrument order")


class PatientBundle(BaseModel):
    patient_ref: str = Field(description="pseudonymous id, never a name or phone number")
    entries: list[Entry] = []
    questionnaires: list[QuestionnaireResponse] = []


# ------------------------------------------------------------- deterministic parts
class QuestionnaireScore(BaseModel):
    instrument: Literal["PHQ-9", "GAD-7"]
    date: date
    total: int
    severity: str
    item9: int | None = None


class QuestionnaireTrend(BaseModel):
    instrument: Literal["PHQ-9", "GAD-7"]
    n_administrations: int
    first_total: int
    latest_total: int
    change: int
    direction: Literal["improved", "worsened", "stable", "baseline_only"]
    clinically_meaningful_change: bool
    latest_severity: str
    history: list[QuestionnaireScore]


class EntryRisk(BaseModel):
    entry_id: str
    level: RiskLevel
    lexicon_tier: int
    lexicon_terms: list[str] = []
    indic_terms: list[str] = []
    classifier_prob: float | None = None
    reasons: list[str] = []


class RiskAssessment(BaseModel):
    level: RiskLevel
    escalate: bool
    reasons: list[str]
    recommended_action: str
    helplines: list[str]
    flagged_entries: list[EntryRisk] = []
    phq9_item9_latest: int | None = None


class CueAggregate(BaseModel):
    n_entries: int
    period_start: datetime | None
    period_end: datetime | None
    modalities: dict[str, int]
    languages: dict[str, int]
    models_used: list[str]
    models_unavailable: dict[str, str]
    mean_distress: float | None = Field(None, description="mean P(anxious|low_mood|angry) across emotion models")
    distress_trend: Literal["rising", "falling", "stable", "insufficient_data"]
    emotion_cue_share: dict[str, float] = Field(description="share of entries whose top emotion cue is X")
    sentiment_mean: float | None
    phq_domains: dict[str, int] = Field(description="PHQ-9 domain -> number of entries mentioning it")
    gad_domains: dict[str, int]
    sleep: dict[str, int]
    stressors: dict[str, int]
    stressor_terms: dict[str, list[str]]
    coping_helpful: dict[str, int]
    coping_mentioned: dict[str, int]
    coping_unhelpful: dict[str, int]


# --------------------------------------------------------------- LLM-written part
class Stressor(BaseModel):
    domain: StressorDomain
    detail: str = Field(max_length=120)


class Evidence(BaseModel):
    statement: str = Field(max_length=160)
    entry_ids: list[str] = Field(max_length=6)


class Narrative(BaseModel):
    clinician_summary: str = Field(
        max_length=700, description="3-5 sentence pre-visit summary built only from the given data")
    main_themes: list[str] = Field(max_length=5)
    stressors: list[Stressor] = Field(max_length=6)
    coping_that_helped: list[str] = Field(max_length=5)
    coping_concerns: list[str] = Field(max_length=4)
    sleep_note: str = Field(max_length=200)
    discussion_points: list[str] = Field(
        max_length=4, description="topics the clinician may want to explore; no diagnosis or treatment advice")
    evidence: list[Evidence] = Field(max_length=6)
    data_confidence: Literal["low", "medium", "high"]


# ---------------------------------------------------------------------- the card
class UserStateCard(BaseModel):
    schema_version: str = SCHEMA_VERSION
    patient_ref: str
    generated_at: datetime
    questionnaires: dict[str, QuestionnaireTrend]
    risk: RiskAssessment
    cues: CueAggregate
    narrative: Narrative
    narrative_source: Literal["llm", "rule_fallback"]
    llm_model: str | None = None
    llm_attempts: int = 0
    guardrail_notes: list[str] = []
    disclaimer: str = ("Wellness-support summary from patient check-ins. Not a diagnosis. Validated "
                       "questionnaires are the measure of symptom severity; risk flags need clinical review.")
