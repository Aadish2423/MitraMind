"""Clinician accounts: Gmail-only sign-up, emailed 6-digit code, and the lock on every clinician route.

Run: python -m pytest tests
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import care.mailer as mailer  # noqa: E402
from care.routes import build_router  # noqa: E402
from care.service import CareService  # noqa: E402
from care.store import Store  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402

ORCH = Phase1Orchestrator(enable_neural=False)
GMAIL, PASSWORD = "dr.mehta@gmail.com", "safe-password-1"
CLINICIAN_ROUTES = ["/v1/clinician/dashboard", "/v1/clinician/alerts", "/v1/audit", "/v1/whatsapp/outbox"]


def client(monkeypatch):
    for var in ("MITRAMIND_SMTP_HOST", "MITRAMIND_SMTP_USER", "MITRAMIND_CLINICIAN_DOMAINS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MITRAMIND_ENV", "dev")
    svc = CareService(Store(":memory:"), ORCH)
    app = FastAPI()
    app.include_router(build_router(svc))
    return TestClient(app), svc


def signed_up(c):
    r = c.post("/v1/auth/clinician/signup", json={"email": GMAIL, "password": PASSWORD, "name": "Dr. Mehta"})
    assert r.status_code == 200, r.text
    return r.json()


def verified(c):
    code = signed_up(c)["dev_code"]
    r = c.post("/v1/auth/clinician/verify", json={"email": GMAIL, "code": code})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def test_only_gmail_can_sign_up(monkeypatch):
    c, _ = client(monkeypatch)
    for email in ("doctor@hospital.org", "someone@yahoo.com", "x@gmail.co"):
        r = c.post("/v1/auth/clinician/signup", json={"email": email, "password": PASSWORD})
        assert r.status_code == 422 and "Gmail" in r.json()["detail"], email
    assert c.post("/v1/auth/clinician/signup", json={"email": GMAIL, "password": "short"}).status_code == 422


def test_clinician_routes_need_a_verified_clinician(monkeypatch):
    c, svc = client(monkeypatch)
    for path in CLINICIAN_ROUTES:
        assert c.get(path).status_code == 401, path
    signup = signed_up(c)
    assert signup["status"] == "verification_sent" and len(signup["dev_code"]) == 6 and not signup["emailed"]
    # Not verified yet: login asks for the code instead of giving a token.
    r = c.post("/v1/auth/clinician/login", json={"email": GMAIL, "password": PASSWORD})
    assert r.status_code == 403 and r.json()["verification_required"]
    assert c.post("/v1/auth/clinician/verify", json={"email": GMAIL, "code": "000000"}).status_code == 401
    r = c.post("/v1/auth/clinician/verify", json={"email": GMAIL, "code": signup["dev_code"]})
    assert r.status_code == 200 and r.json()["role"] == "clinician"
    token = r.json()["token"]
    for path in CLINICIAN_ROUTES:
        assert c.get(path, headers={"Authorization": f"Bearer {token}"}).status_code == 200, path
    me = c.get("/v1/auth/clinician/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert me["clinician"]["email"] == GMAIL and me["clinician"]["name"] == "Dr. Mehta"
    # After verifying, a normal login works, and the code cannot be used twice.
    assert c.post("/v1/auth/clinician/login", json={"email": GMAIL, "password": PASSWORD}).status_code == 200
    assert c.post("/v1/auth/clinician/verify", json={"email": GMAIL, "code": signup["dev_code"]}).status_code == 401


def test_patient_token_does_not_open_clinician_routes(monkeypatch):
    c, _ = client(monkeypatch)
    s = c.post("/v1/auth/signup", json={"email": "patient@example.com", "password": "patient-pass",
                                        "consents": {"companion": True}}).json()
    assert c.get("/v1/clinician/dashboard", headers={"Authorization": f"Bearer {s['token']}"}).status_code == 401


def test_logout_ends_the_clinician_session(monkeypatch):
    c, _ = client(monkeypatch)
    h = {"Authorization": f"Bearer {verified(c)}"}
    assert c.post("/v1/auth/clinician/logout", headers=h).status_code == 200
    assert c.get("/v1/clinician/dashboard", headers=h).status_code == 401


def test_alert_handler_is_the_logged_in_clinician(monkeypatch):
    c, svc = client(monkeypatch)
    svc.enroll("p1", consents={"companion": True, "care_team": True})
    aid = svc.store.add_alert("p1", "high", ["test"], "message")
    h = {"Authorization": f"Bearer {verified(c)}"}
    r = c.post(f"/v1/clinician/alerts/{aid}", headers=h, json={"status": "acknowledged", "handled_by": "someone else"})
    assert r.json()["handled_by"] == "Dr. Mehta"
    assert svc.store.alerts("acknowledged")[0]["handled_by"] == "Dr. Mehta"


def test_wrong_codes_lock_the_code(monkeypatch):
    c, _ = client(monkeypatch)
    code = signed_up(c)["dev_code"]
    wrong = "111111" if code != "111111" else "222222"
    for _ in range(5):
        c.post("/v1/auth/clinician/verify", json={"email": GMAIL, "code": wrong})
    r = c.post("/v1/auth/clinician/verify", json={"email": GMAIL, "code": code})
    assert r.status_code in (401, 429)  # the right code no longer works after 5 wrong tries


def test_resend_waits_a_minute_and_does_not_reveal_accounts(monkeypatch):
    c, _ = client(monkeypatch)
    signed_up(c)
    assert c.post("/v1/auth/clinician/resend", json={"email": GMAIL}).status_code == 422  # too soon
    r = c.post("/v1/auth/clinician/resend", json={"email": "nobody@gmail.com"})
    assert r.status_code == 200 and r.json()["status"] == "verification_sent"


def test_outside_dev_mode_email_must_be_configured(monkeypatch):
    c, _ = client(monkeypatch)
    monkeypatch.setenv("MITRAMIND_ENV", "production")
    r = c.post("/v1/auth/clinician/signup", json={"email": GMAIL, "password": PASSWORD})
    assert r.status_code == 503


def test_code_goes_by_gmail_smtp_when_configured(monkeypatch):
    c, _ = client(monkeypatch)
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent.append({"host": host, "port": port})

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context):
            sent[-1]["tls"] = True

        def login(self, user, password):
            sent[-1]["user"] = user

        def send_message(self, msg):
            sent[-1]["to"], sent[-1]["body"] = msg["To"], msg.get_content()

    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("MITRAMIND_SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("MITRAMIND_SMTP_USER", "mitramind.sender@gmail.com")
    monkeypatch.setenv("MITRAMIND_SMTP_PASSWORD", "app-password")
    body = signed_up(c)
    assert body["emailed"] and "dev_code" not in body  # the code is only in the email
    assert sent[0]["host"] == "smtp.gmail.com" and sent[0]["tls"] and sent[0]["to"] == GMAIL
    code = next(w for w in sent[0]["body"].split() if w.isdigit() and len(w) == 6)
    assert c.post("/v1/auth/clinician/verify", json={"email": GMAIL, "code": code}).status_code == 200
