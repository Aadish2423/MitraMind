"""Phase 1 tests: questionnaires, crisis layer, orchestrator, LLM retry/guardrails (fake backend).

Run: python -m pytest tests   (or python tests/test_phase1.py)
"""
import json
import pathlib
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from phase1.crisis import indic_hits  # noqa: E402
from phase1.llm import LLMResult, inline_schema  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402
from phase1.questionnaires import score, trends  # noqa: E402
from phase1.schema import Entry, Narrative, PatientBundle, QuestionnaireResponse  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "patients"
ORCH = Phase1Orchestrator(enable_neural=False)


def bundle(name):
    return PatientBundle.model_validate_json((DEMO / f"{name}.json").read_text(encoding="utf-8"))


def entry(text, i="t1"):
    return Entry(entry_id=i, timestamp=datetime(2026, 10, 1, tzinfo=timezone.utc), text=text)


class FakeBackend:
    """Returns queued answers; records prompts."""
    name = "fake"

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def available(self):
        return True, ""

    def chat(self, system, user, schema):
        self.prompts.append(user)
        return LLMResult(self.answers.pop(0), 0.0, "fake")


def narrative(**over):
    base = dict(clinician_summary="PHQ-9 improved from 14 to 9 over the period; sleep problems were reported.",
                main_themes=["sleep"], stressors=[{"domain": "school_exam", "detail": "exams"}],
                coping_that_helped=["breathing"], coping_concerns=[], sleep_note="insomnia reported",
                discussion_points=["sleep"], evidence=[{"statement": "exam stress", "entry_ids": ["s01"]}],
                data_confidence="medium")
    base.update(over)
    return json.dumps(base)


def test_questionnaire_scoring():
    s = score(QuestionnaireResponse(instrument="PHQ-9", date=date(2026, 10, 1), items=[2, 2, 2, 2, 2, 2, 2, 1, 1]))
    assert (s.total, s.severity, s.item9) == (16, "moderately severe", 1)
    t = trends(bundle("demo_student_01").questionnaires)
    assert t["PHQ-9"].change == -5 and t["PHQ-9"].direction == "improved" and t["PHQ-9"].clinically_meaningful_change


def test_indic_crisis_phrases():
    assert max(t for t, _ in indic_hits("kabhi kabhi lagta hai marna hi better hai")) == 2
    assert max(t for t, _ in indic_hits("मैं मरना चाहता हूँ")) == 2
    assert indic_hits("marna nahi chahta, bas thak gaya hoon") == []


def test_indic_ideation_variants():
    # Missed by the lexicon in a manual run on 2026-10-20 (got the normal companion reply, no alert).
    for text in ("mujhe ab jeena nahi hai, sab khatam karna chahta hoon", "mujhe jeena nahi hai",
                 "sab kuch khatam karna chahti hoon", "jina nahi chahti", "mujhe nahi jeena ab",
                 "मुझे अब जीना नहीं है", "मुझे नहीं जीना", "मैं सब खत्म करना चाहता हूँ"):
        assert max((t for t, _ in indic_hits(text)), default=0) == 2, text


def test_indic_ideation_near_misses():
    for text in ("mujhe jeena hai, haar nahi maanunga", "exam khatam ho gaya, ab chill", "aaj sab kaam khatam karna hai",
                 "jeena seekh raha hoon", "मुझे जीना है"):
        assert indic_hits(text) == [], text


def test_devanagari_nukta_typed_either_way():
    # ख़ can arrive as one code point (U+0959) or as ख + nukta (U+0916 U+093C); both must match.
    for text in ("सब ख़त्म करना चाहती हूँ", "सब ख़त्म करना चाहती हूँ", "खुद को ख़त्म कर दूंगा"):
        assert max((t for t, _ in indic_hits(text)), default=0) == 2, text


def test_entry_triage_levels():
    assert ORCH.analyze_entry(entry("I have been saving up pills."))["risk"].level == "urgent"
    assert ORCH.analyze_entry(entry("I don't want to live anymore, I have the pills ready"))["risk"].level == "urgent"
    assert ORCH.analyze_entry(entry("I have the pills ready for tonight"))["risk"].level == "urgent"
    assert ORCH.analyze_entry(entry("mujhe ab jeena nahi hai, sab khatam karna chahta hoon"))["risk"].level == "high"
    for text in ("I have my BP pills ready for the trip", "I have the pills ready for my mother",
                 "The pharmacy texted that my pills are ready", "I don't have the pills ready yet"):
        assert ORCH.analyze_entry(entry(text))["risk"].level == "none", text
    assert ORCH.analyze_entry(entry("I want to die"))["risk"].level == "high"
    assert ORCH.analyze_entry(entry("I don't want to die, I just want the pain to stop."))["risk"].level in ("none", "watch")
    assert ORCH.analyze_entry(entry("Went for a walk, felt good today."))["risk"].level == "none"


def test_phq_item9_overrides_text():
    card, _ = ORCH.build_card(bundle("demo_professional_02"), use_llm=False)
    assert card.risk.level == "high" and card.risk.escalate and card.risk.phq9_item9_latest == 1


def test_cards_build_without_llm():
    for path in DEMO.glob("*.json"):
        card, _ = ORCH.build_card(PatientBundle.model_validate_json(path.read_text(encoding="utf-8")), use_llm=False)
        assert card.narrative_source == "rule_fallback" and card.narrative.clinician_summary


def test_coping_helped_detected():
    card, _ = ORCH.build_card(bundle("demo_student_01"), use_llm=False)
    assert "breathing" in card.cues.coping_helpful


def test_hindi_entries_get_cues():
    card, _ = ORCH.build_card(bundle("demo_caregiver_03"), use_llm=False)
    assert "health" in card.cues.stressors and card.cues.languages.get("hi") == 4


def test_shared_text_only_for_shared_entries():
    _, extra = ORCH.build_card(bundle("demo_student_01"), use_llm=False)
    shared = {c["entry_id"] for c in extra["digest"]["checkins"] if "shared_text" in c}
    assert shared == {"s01", "s03", "s07"}


def test_schema_is_inlined():
    assert "$ref" not in json.dumps(inline_schema(Narrative))


def test_llm_valid_answer_used():
    orch = Phase1Orchestrator(backend=FakeBackend([narrative()]), enable_neural=False)
    card, _ = orch.build_card(bundle("demo_student_01"))
    assert card.narrative_source == "llm" and card.llm_attempts == 1


def test_llm_guardrail_retry_then_fallback():
    bad = narrative(clinician_summary="The patient has major depression; increase the dose of sertraline.")
    fake = FakeBackend([bad, bad])
    orch = Phase1Orchestrator(backend=fake, enable_neural=False)
    card, _ = orch.build_card(bundle("demo_student_01"))
    assert card.narrative_source == "rule_fallback" and card.llm_attempts == 2
    assert "previous answer had these problems" in fake.prompts[1]


def test_llm_cannot_change_risk():
    calm = narrative(clinician_summary="Hopelessness was reported in recent check-ins.", evidence=[], stressors=[])
    orch = Phase1Orchestrator(backend=FakeBackend([calm]), enable_neural=False)
    card, _ = orch.build_card(bundle("demo_crisis_04"))
    assert card.risk.level == "urgent" and card.narrative_source == "llm"


def test_invented_numbers_rejected():
    fake = FakeBackend([narrative(clinician_summary="PHQ-9 went from 22 to 3."), narrative()])
    card, _ = Phase1Orchestrator(backend=fake, enable_neural=False).build_card(bundle("demo_student_01"))
    assert card.llm_attempts == 2 and card.narrative_source == "llm"


def test_wrong_checkin_counts_rejected():
    # The student has 8 check-ins and exams in 6 of them; "two of nine" and "5 of 8" are both wrong.
    for bad in ("Exams came up in two of nine check-ins.", "Sleep was a problem in 5 of 8 check-ins."):
        fake = FakeBackend([narrative(clinician_summary=bad), narrative()])
        card, _ = Phase1Orchestrator(backend=fake, enable_neural=False).build_card(bundle("demo_student_01"))
        assert card.llm_attempts == 2, bad
    ok = FakeBackend([narrative(clinician_summary="Exams came up in 6 of 8 check-ins, panic in none of the check-ins "
                                                  "and self-harm in any of 8 check-ins was not reported.")])
    card, _ = Phase1Orchestrator(backend=ok, enable_neural=False).build_card(bundle("demo_student_01"))
    assert card.llm_attempts == 1


def test_single_questionnaire_is_baseline():
    assert trends(bundle("demo_crisis_04").questionnaires)["PHQ-9"].direction == "baseline_only"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
