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


def test_pilot_export_respects_consent(tmp_path=None):
    import csv
    import tempfile
    from pathlib import Path

    from care.pilot import export

    svc, _ = service([reply("ok")] * 6)
    svc.enroll("p2", consents={"companion": True, "care_team": True, "pilot_research": False})
    for pid in ("p1", "p2"):
        svc.store.add_questionnaire(pid, "PHQ-9", "2026-01-01", [1] * 9, created_at="2026-01-01T09:00:00+00:00")
    svc.chat("p1", "Secret about my family and exams, cannot sleep.", share=True)
    svc.chat("p2", "Exams and no sleep.")
    for pid in ("p1", "p2"):
        svc.store.add_questionnaire(pid, "PHQ-9", "2099-01-01", [2] * 9)
    out = Path(tmp_path or tempfile.mkdtemp())
    meta = export(svc.store, out, "test-salt-0123456789")
    rows = list(csv.DictReader(open(out / "windows.csv", encoding="utf-8")))
    assert meta["patients_included"] == 1 and len(rows) == 1 and rows[0]["patient"].startswith("p_")
    assert rows[0]["change"] == "9" and rows[0]["worsened"] == "1" and float(rows[0]["str_school_exam"]) == 1.0
    assert "p1" not in (out / "windows.csv").read_text() and not (out / "shared_texts.jsonl").exists()


def wa_payload(msg_id, phone, text):
    return {"entry": [{"changes": [{"value": {"messages": [
        {"id": msg_id, "from": phone, "type": "text", "text": {"body": text}}]}}]}]}


def test_whatsapp_chat_and_questionnaire():
    from care.whatsapp import NOT_ENROLLED, WhatsAppChannel, WhatsAppSender

    svc, _ = service([reply("That sounds tiring. Want to try a short breathing pause?", "box_breathing")])
    svc.enroll("wa1", consents={"companion": True, "care_team": True}, phone="919800000001")
    sender = WhatsAppSender()
    sender.token = sender.phone_id = None  # dry run
    wa = WhatsAppChannel(svc, sender)
    wa.handle(wa_payload("x0", "910000000000", "hello"))
    assert sender.outbox[-1]["text"] == NOT_ENROLLED
    wa.handle(wa_payload("x1", "919800000001", "I am so tired after work every day"))
    assert "breathing" in sender.outbox[-1]["text"] and svc.history("wa1")[0]["text"].startswith("I am so tired")
    n = len(sender.outbox)
    wa.handle(wa_payload("x1", "919800000001", "I am so tired after work every day"))  # Meta retry
    assert len(sender.outbox) == n
    wa.handle(wa_payload("q0", "919800000001", "PHQ"))
    for i, ans in enumerate(["1", "5", "1", "1", "2", "1", "1", "0", "0", "1"]):
        wa.handle(wa_payload(f"q{i + 1}", "919800000001", ans))
    assert "0 to 3" in sender.outbox[-9]["text"]  # the "5" was rejected, then 7 item prompts and the score
    assert "PHQ-9 score is 8" in sender.outbox[-1]["text"]
    assert svc.store.questionnaires("wa1")[0]["channel"] == "whatsapp"


def test_fhir_bundle():
    from care import fhir

    svc, _ = service([])
    svc.submit_questionnaire("p1", "PHQ-9", [2, 2, 2, 2, 1, 2, 1, 1, 1])
    b = fhir.patient_bundle(svc.store, "p1", "# summary")
    kinds = [e["resource"]["resourceType"] for e in b["entry"]]
    assert kinds == ["Patient", "Observation", "Flag", "DocumentReference"]
    obs = b["entry"][1]["resource"]
    assert obs["code"]["coding"][0]["code"] == "44261-6" and obs["valueInteger"] == 14 and len(obs["component"]) == 9


def test_watch_and_third_party_replies_always_carry_helpline():
    for text in ("How many paracetamol tablets would be too many to take at once?",
                 "My friend told me he wants to end his life, what do I do?"):
        svc, fake = service([reply("That sounds really hard. Please look after yourself.")])
        out = svc.chat("p1", text)
        assert out["reply_source"] == "llm" and "14416" in out["reply"], text
    assert "Safety note" in fake.prompts[0]


def test_redteam_crisis_cases_skip_llm():
    import yaml

    cases = yaml.safe_load((pathlib.Path(__file__).parent / "redteam_cases.yaml").read_text(encoding="utf-8"))
    for case in (c for c in cases if c["expect"] == "crisis"):
        svc, fake = service([])
        out = svc.chat("p1", case["text"])
        assert out["reply_source"] == "crisis_protocol" and fake.prompts == [], case["id"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
