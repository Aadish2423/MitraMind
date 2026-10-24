"""No card leaks: who can see which User State Card, and that unshared text never appears in one.

Run: python -m pytest tests
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from care.routes import build_router  # noqa: E402
from care.service import CareService  # noqa: E402
from care.store import Store  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402

ORCH = Phase1Orchestrator(enable_neural=False)
SHARE = {"companion": True, "care_team": True}
SECRET = "my private secret about the divorce"
SHARED = "work has been really stressful this week"


def world(monkeypatch):
    """Two clinicians, a superuser, and three patients: A (Dr One, shares with care team), B (Dr Two),
    C (Dr One, does not share with the care team)."""
    for var in ("MITRAMIND_SMTP_HOST", "MITRAMIND_SMTP_USER", "MITRAMIND_CLINICIAN_DOMAINS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MITRAMIND_ENV", "dev")
    monkeypatch.setenv("MITRAMIND_PATIENT_EMAIL_VERIFY", "0")
    svc = CareService(Store(":memory:"), ORCH)
    app = FastAPI()
    app.include_router(build_router(svc))
    c = TestClient(app)
    docs = {}
    for name in ("one", "two"):
        email = f"dr.{name}@gmail.com"
        code = c.post("/v1/auth/clinician/signup", json={"email": email, "password": "clinician-pass",
                                                         "name": f"Dr {name}"}).json()["dev_code"]
        s = c.post("/v1/auth/clinician/verify", json={"email": email, "code": code}).json()
        docs[name] = {"h": {"Authorization": f"Bearer {s['token']}"}, "code": s["clinician"]["care_code"]}
    svc.create_superuser("boss@example.com", "superuser-password-1", "Boss")
    su = c.post("/v1/auth/admin/login", json={"email": "boss@example.com", "password": "superuser-password-1"}).json()
    pats = {}
    for name, doc, consents in (("A", "one", SHARE), ("B", "two", SHARE), ("C", "one", {"companion": True})):
        s = c.post("/v1/auth/signup", json={"email": f"{name.lower()}@example.com", "password": "patient-pass",
                                            "consents": consents, "clinician_code": docs[doc]["code"]}).json()
        pid = s["profile"]["id"]
        svc.store.add_message(pid, "user", SECRET, shared=False)
        svc.store.add_message(pid, "user", SHARED, shared=True)
        svc.store.add_questionnaire(pid, "PHQ-9", "2026-10-01", [2] * 9)
        pats[name] = {"id": pid, "h": {"Authorization": f"Bearer {s['token']}"}}
    return c, svc, docs, {"Authorization": f"Bearer {su['token']}"}, pats


def card(c, pid, h):
    return c.get(f"/v1/clinician/patients/{pid}/state-card?use_llm=false", headers=h)


def test_patients_see_only_their_own_summary(monkeypatch):
    c, _, _, _, p = world(monkeypatch)
    assert c.get(f"/v1/patients/{p['A']['id']}/summary", headers=p["A"]["h"]).status_code == 200
    assert c.get(f"/v1/patients/{p['B']['id']}/summary", headers=p["A"]["h"]).status_code == 403
    assert c.get(f"/v1/patients/{p['A']['id']}/summary").status_code == 401
    # A patient's token never opens a clinician card.
    assert card(c, p["A"]["id"], p["A"]["h"]).status_code == 401


def test_clinicians_see_only_their_patients_cards(monkeypatch):
    c, _, d, _, p = world(monkeypatch)
    assert card(c, p["A"]["id"], d["one"]["h"]).status_code == 200
    for path in ("state-card?use_llm=false", "shared", "fhir"):
        assert c.get(f"/v1/clinician/patients/{p['A']['id']}/{path}", headers=d["two"]["h"]).status_code == 403, path
    assert {r["id"] for r in c.get("/v1/clinician/dashboard", headers=d["one"]["h"]).json()} == {p["A"]["id"], p["C"]["id"]}
    assert {r["id"] for r in c.get("/v1/clinician/dashboard", headers=d["two"]["h"]).json()} == {p["B"]["id"]}


def test_superuser_sees_every_consenting_card(monkeypatch):
    c, _, _, su, p = world(monkeypatch)
    assert card(c, p["A"]["id"], su).status_code == 200 and card(c, p["B"]["id"], su).status_code == 200
    assert len(c.get("/v1/clinician/dashboard", headers=su).json()) == 3


def test_no_card_without_care_team_consent_even_for_superuser(monkeypatch):
    c, _, d, su, p = world(monkeypatch)
    assert card(c, p["C"]["id"], d["one"]["h"]).status_code == 403
    assert card(c, p["C"]["id"], su).status_code == 403
    row = next(r for r in c.get("/v1/clinician/dashboard", headers=d["one"]["h"]).json() if r["id"] == p["C"]["id"])
    assert "PHQ-9" not in row and "usage" not in row  # alerts count only


def test_unshared_text_never_appears_in_any_card(monkeypatch):
    c, _, d, su, p = world(monkeypatch)
    pid = p["A"]["id"]
    for body in (card(c, pid, d["one"]["h"]).json(), card(c, pid, su).json(),
                 c.get(f"/v1/patients/{pid}/summary", headers=p["A"]["h"]).json(),
                 c.get(f"/v1/clinician/patients/{pid}/fhir?include_card=true", headers=d["one"]["h"]).json()):
        assert SECRET not in json.dumps(body)
    shared = c.get(f"/v1/clinician/patients/{pid}/shared", headers=d["one"]["h"]).json()
    assert [m["text"] for m in shared] == [SHARED]


def test_card_content_is_the_phase1_card(monkeypatch):
    """The dashboard card is exactly the Phase 1 State Card for the stored check-ins (nothing added or dropped)."""
    c, svc, d, _, p = world(monkeypatch)
    pid = p["A"]["id"]
    got = card(c, pid, d["one"]["h"]).json()["card"]
    want = json.loads(ORCH.build_card(svc.bundle(pid), use_llm=False)[0].model_dump_json())
    for k in ("generated_at",):
        got.pop(k), want.pop(k)
    assert got == want
