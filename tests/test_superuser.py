"""Superuser, clinician-patient links, the patient's own summary card and the clinician report.

Run: python -m pytest tests
"""
import pathlib
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from care.routes import build_router  # noqa: E402
from care.service import CareService  # noqa: E402
from care.store import Store  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402

ORCH = Phase1Orchestrator(enable_neural=False)
SHARE = {"companion": True, "care_team": True}


def setup(monkeypatch):
    for var in ("MITRAMIND_SMTP_HOST", "MITRAMIND_SMTP_USER", "MITRAMIND_CLINICIAN_DOMAINS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MITRAMIND_ENV", "dev")
    svc = CareService(Store(":memory:"), ORCH)
    app = FastAPI()
    app.include_router(build_router(svc))
    return TestClient(app), svc


def clinician(c, email, name):
    code = c.post("/v1/auth/clinician/signup", json={"email": email, "password": "long-password-1",
                                                     "name": name}).json()["dev_code"]
    s = c.post("/v1/auth/clinician/verify", json={"email": email, "code": code}).json()
    return {"Authorization": f"Bearer {s['token']}"}, s["clinician"]


def superuser(c, svc):
    svc.create_superuser("boss@example.com", "superuser-password-1", "Boss")
    s = c.post("/v1/auth/clinician/login", json={"email": "boss@example.com", "password": "superuser-password-1"})
    assert s.status_code == 200 and s.json()["role"] == "superuser"
    return {"Authorization": f"Bearer {s.json()['token']}"}


def test_clinician_sees_only_linked_patients_superuser_sees_all(monkeypatch):
    c, svc = setup(monkeypatch)
    h_a, doc_a = clinician(c, "dr.a@gmail.com", "Dr A")
    h_b, _ = clinician(c, "dr.b@gmail.com", "Dr B")
    assert doc_a["care_code"].startswith("MM-")
    # A patient signs up with Dr A's code; another patient links nobody.
    s = c.post("/v1/auth/signup", json={"email": "pat@example.com", "password": "patient-pass", "consents": SHARE,
                                        "clinician_code": doc_a["care_code"]}).json()
    pid = s["profile"]["id"]
    assert s["profile"]["clinician_name"] == "Dr A"
    svc.enroll("lonely", consents=SHARE)
    ids = lambda h: {r["id"] for r in c.get("/v1/clinician/dashboard", headers=h).json()}  # noqa: E731
    assert ids(h_a) == {pid} and ids(h_b) == set()
    assert c.get(f"/v1/clinician/patients/{pid}/state-card?use_llm=false", headers=h_b).status_code == 403
    assert c.get(f"/v1/clinician/patients/{pid}/state-card?use_llm=false", headers=h_a).status_code == 200
    h_su = superuser(c, svc)
    assert ids(h_su) == {pid, "lonely"}
    # Superuser-only routes stay closed to clinicians.
    for path in ("/v1/admin/overview", "/v1/admin/clinicians", "/v1/whatsapp/outbox", "/v1/audit"):
        assert c.get(path, headers=h_a).status_code == 403, path
        assert c.get(path, headers=h_su).status_code == 200, path


def test_wrong_clinician_code_is_refused(monkeypatch):
    c, _ = setup(monkeypatch)
    r = c.post("/v1/auth/signup", json={"email": "pat@example.com", "password": "patient-pass", "consents": SHARE,
                                        "clinician_code": "MM-NOPE00"})
    assert r.status_code == 422


def test_alerts_follow_the_link(monkeypatch):
    c, svc = setup(monkeypatch)
    h_a, doc_a = clinician(c, "dr.a@gmail.com", "Dr A")
    h_b, _ = clinician(c, "dr.b@gmail.com", "Dr B")
    svc.enroll("p1", consents=SHARE)
    svc.store.set_patient_clinician("p1", doc_a["id"])
    aid = svc.store.add_alert("p1", "high", ["test"], "message")
    assert [a["id"] for a in c.get("/v1/clinician/alerts", headers=h_a).json()] == [aid]
    assert c.get("/v1/clinician/alerts", headers=h_b).json() == []
    assert c.post(f"/v1/clinician/alerts/{aid}", headers=h_b, json={"status": "acknowledged"}).status_code == 403


def test_superuser_assigns_patients(monkeypatch):
    c, svc = setup(monkeypatch)
    h_a, doc_a = clinician(c, "dr.a@gmail.com", "Dr A")
    h_su = superuser(c, svc)
    svc.enroll("p1", consents=SHARE)
    r = c.post("/v1/admin/patients/p1/clinician", headers=h_su, json={"clinician_id": doc_a["id"]})
    assert r.status_code == 200 and r.json()["clinician_name"] == "Dr A"
    assert [p["id"] for p in c.get("/v1/clinician/dashboard", headers=h_a).json()] == ["p1"]
    assert c.post("/v1/admin/patients/p1/clinician", headers=h_a, json={"clinician_id": None}).status_code == 403


def test_patient_sees_own_summary_card(monkeypatch):
    c, svc = setup(monkeypatch)
    s = c.post("/v1/auth/signup", json={"email": "pat@example.com", "password": "patient-pass",
                                        "consents": SHARE}).json()
    pid, h = s["profile"]["id"], {"Authorization": f"Bearer {s['token']}"}
    svc.store.add_questionnaire(pid, "PHQ-9", "2026-10-01", [2] * 9)
    svc.store.add_questionnaire(pid, "PHQ-9", "2026-10-15", [1] * 9)
    svc.store.add_message(pid, "user", "Exams are stressing me, the breathing exercise helped a bit.")
    card = c.get(f"/v1/patients/{pid}/summary", headers=h).json()
    phq = card["scores"][0]
    assert phq["latest"] == 9 and phq["change"] == -9 and phq["direction"] == "improved"
    assert "studies / exams" in card["came_up"] and card["helplines"]
    assert c.get(f"/v1/patients/{pid}/summary").status_code == 401  # another person can't read it


def test_clinician_report_flags_slow_alerts_and_counts_outcomes(monkeypatch):
    c, svc = setup(monkeypatch)
    h_a, doc_a = clinician(c, "dr.a@gmail.com", "Dr A")
    h_b, doc_b = clinician(c, "dr.b@gmail.com", "Dr B")
    h_su = superuser(c, svc)
    for pid, doc, scores in (("p1", doc_a, ([3] * 9, [1] * 9)), ("p2", doc_b, ([1] * 9, [1] * 9))):
        svc.enroll(pid, consents=SHARE)
        svc.store.set_patient_clinician(pid, doc["id"])
        svc.store.add_questionnaire(pid, "PHQ-9", "2026-10-01", scores[0])
        svc.store.add_questionnaire(pid, "PHQ-9", "2026-10-15", scores[1])
        svc.store.add_message(pid, "user", "had an okay day")  # active in the last 14 days
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(timespec="seconds")
    svc.store.add_alert("p2", "urgent", ["test"], "message", created_at=old)  # Dr B has not picked it up for 5 h
    report = {r["clinician"]["name"]: r for r in c.get("/v1/admin/clinicians", headers=h_su).json()}
    assert report["Dr B"]["status"] == "needs attention" and report["Dr B"]["alerts"]["late"] == 1
    assert report["Dr A"]["outcomes"]["PHQ-9"]["improved"] == 1 and report["Dr A"]["status"] == "on track"
    ov = c.get("/v1/admin/overview", headers=h_su).json()
    assert ov["clinicians"] == 2 and ov["open_alerts"]["urgent"] == 1
