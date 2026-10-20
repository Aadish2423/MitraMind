"""Companion, consent, triage and privacy tests (no GPU, no LLM: fake backend, in-memory DB).

Run: python tests/test_care.py   (or python -m pytest tests)
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from care.service import CareService, ConsentRequired  # noqa: E402
from care.store import Store  # noqa: E402
from phase1.llm import LLMResult  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402

ORCH = Phase1Orchestrator(enable_neural=False)
ALL = {"companion": True, "care_team": True, "pilot_research": True, "research_text": False}


class FakeBackend:
    name = "fake"

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def available(self):
        return True, ""

    def chat(self, system, user, schema):
        self.prompts.append(user)
        return LLMResult(self.answers.pop(0), 0.0, "fake")


def reply(text, pause="none"):
    return json.dumps({"reply": text, "mind_pause": pause})


def service(answers=(), consents=ALL):
    fake = FakeBackend(answers)
    svc = CareService(Store(":memory:"), ORCH, fake)
    svc.enroll("p1", "Test patient", consents=consents)
    return svc, fake


def test_normal_message_gets_llm_reply():
    svc, fake = service([reply("That sounds stressful. Would a short breathing break help?", "box_breathing")])
    out = svc.chat("p1", "Exams are next week and I can't sleep.")
    assert out["reply_source"] == "llm" and out["mind_pause"] == "box_breathing" and out["risk"]["level"] == "none"
    assert "Exams are next week" in fake.prompts[0]
    assert len(svc.store.audit_log("p1")) == 1 and svc.store.alerts() == []


def test_crisis_message_never_reaches_llm():
    svc, fake = service([])
    out = svc.chat("p1", "I have been saving up pills.")
    assert out["reply_source"] == "crisis_protocol" and out["escalate"] and "14416" in out["reply"]
    assert fake.prompts == []
    alerts = svc.alerts()
    assert len(alerts) == 1 and alerts[0]["level"] == "urgent" and alerts[0]["source"] == "message"


def test_hinglish_crisis_reply_in_hinglish():
    svc, _ = service([])
    out = svc.chat("p1", "kabhi kabhi lagta hai marna hi better hai, sab ke liye bojh ban gaya hoon")
    assert out["risk"]["level"] == "high" and "Tele-MANAS ko 14416" in out["reply"]


def test_guardrails_retry_then_fallback():
    bad = reply("You have major depression; ask for sertraline 50 mg.")
    svc, fake = service([bad, bad])
    out = svc.chat("p1", "Feeling low all week.")
    assert out["reply_source"] == "fallback" and out["llm_attempts"] == 2
    assert "previous answer had these problems" in fake.prompts[1]


def test_role_and_secrecy_claims_rejected():
    for bad in ("I'm your therapist, so trust me.", "Don't worry, this stays between us."):
        svc, _ = service([reply(bad), reply("Thank you for telling me.")])
        out = svc.chat("p1", "Can I tell you something?")
        assert out["llm_attempts"] == 2 and out["reply_source"] == "llm", bad


def test_companion_needs_consent():
    svc, _ = service([], consents={"companion": False})
    try:
        svc.chat("p1", "hello")
    except ConsentRequired:
        return
    raise AssertionError("chat without consent was allowed")


def test_clinician_sees_only_shared_text():
    svc, _ = service([reply("Thanks for sharing."), reply("I hear you.")])
    svc.chat("p1", "Private thought about my family.")
    svc.chat("p1", "Work has been very stressful.", share=True)
    texts = [m["text"] for m in svc.clinician_messages("p1")]
    assert texts == ["Work has been very stressful."]
    bundle = svc.bundle("p1")
    assert [e.shared_with_clinician for e in bundle.entries] == [False, True]


def test_no_care_team_consent_blocks_card():
    svc, _ = service([], consents={"companion": True, "care_team": False})
    try:
        svc.state_card("p1", use_llm=False)
    except ConsentRequired:
        return
    raise AssertionError("card built without care_team consent")


def test_questionnaire_item9_opens_alert():
    svc, _ = service([])
    out = svc.submit_questionnaire("p1", "PHQ-9", [2, 2, 2, 2, 1, 2, 1, 1, 1])
    assert out["risk"]["level"] == "high" and out["alert_id"] and out["helplines"]
    assert svc.alerts()[0]["source"] == "questionnaire"


def test_state_card_from_stored_chats():
    svc, _ = service([reply("That sounds hard."), reply("Glad it helped.")])
    svc.submit_questionnaire("p1", "PHQ-9", [2, 2, 2, 2, 1, 2, 1, 1, 0])
    svc.chat("p1", "Exams are stressing me out and I can't sleep.", share=True)
    svc.chat("p1", "Did some breathing exercises, it helped a bit.")
    res = svc.state_card("p1", use_llm=False)
    card = res["card"]
    assert card.cues.n_entries == 2 and "school_exam" in card.cues.stressors
    assert "breathing" in card.cues.coping_helpful and res["markdown"].startswith("#")


def test_dashboard_hides_scores_without_consent():
    svc, _ = service([])
    svc.enroll("p2", "Other", consents={"companion": True, "care_team": False})
    rows = {r["id"]: r for r in svc.dashboard()}
    assert "usage" in rows["p1"] and "usage" not in rows["p2"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
