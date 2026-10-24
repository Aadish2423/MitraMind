"""Forgot password for all three roles, and patient email verification.

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
SHARE = {"companion": True, "care_team": True}


def setup(monkeypatch, verify_patients=False):
    for var in ("MITRAMIND_SMTP_HOST", "MITRAMIND_SMTP_USER", "MITRAMIND_CLINICIAN_DOMAINS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MITRAMIND_ENV", "dev")
    monkeypatch.setenv("MITRAMIND_PATIENT_EMAIL_VERIFY", "1" if verify_patients else "0")
    svc = CareService(Store(":memory:"), ORCH)
    app = FastAPI()
    app.include_router(build_router(svc))
    return TestClient(app), svc


def patient(c, email="pat@gmail.com", password="patient-pass"):
    return c.post("/v1/auth/signup", json={"email": email, "password": password, "consents": SHARE}).json()


def test_patient_resets_password_and_old_sessions_end(monkeypatch):
    c, _ = setup(monkeypatch)
    old = patient(c)
    sent = c.post("/v1/auth/forgot", json={"email": "pat@gmail.com"}).json()
    assert sent["status"] == "reset_sent" and len(sent["dev_code"]) == 6
    r = c.post("/v1/auth/reset", json={"email": "pat@gmail.com", "code": sent["dev_code"], "password": "brand-new-pass"})
    assert r.status_code == 200
    assert c.post("/v1/auth/login", json={"email": "pat@gmail.com", "password": "patient-pass"}).status_code == 401
    assert c.post("/v1/auth/login", json={"email": "pat@gmail.com", "password": "brand-new-pass"}).status_code == 200
    assert c.get("/v1/auth/me", headers={"Authorization": f"Bearer {old['token']}"}).status_code == 401
    # The code works once.
    again = c.post("/v1/auth/reset", json={"email": "pat@gmail.com", "code": sent["dev_code"], "password": "another-pass"})
    assert again.status_code == 401


def test_forgot_does_not_reveal_accounts(monkeypatch):
    c, _ = setup(monkeypatch)
    r = c.post("/v1/auth/forgot", json={"email": "nobody@gmail.com"})
    assert r.status_code == 200 and r.json()["status"] == "reset_sent" and "dev_code" not in r.json()
    assert c.post("/v1/auth/reset", json={"email": "nobody@gmail.com", "code": "123456",
                                          "password": "whatever-pass"}).status_code == 401


def test_reset_codes_are_per_role(monkeypatch):
    c, svc = setup(monkeypatch)
    patient(c, "same@gmail.com")
    code = c.post("/v1/auth/clinician/signup", json={"email": "same@gmail.com", "password": "clinician-pass",
                                                     "name": "Dr Same"}).json()["dev_code"]
    c.post("/v1/auth/clinician/verify", json={"email": "same@gmail.com", "code": code})
    clin = c.post("/v1/auth/clinician/forgot", json={"email": "same@gmail.com"}).json()["dev_code"]
    # A clinician reset code does not change the patient account with the same address.
    assert c.post("/v1/auth/reset", json={"email": "same@gmail.com", "code": clin,
                                          "password": "hacked-pass"}).status_code == 401
    assert c.post("/v1/auth/clinician/reset", json={"email": "same@gmail.com", "code": clin,
                                                    "password": "new-clinician-pass"}).status_code == 200
    assert c.post("/v1/auth/login", json={"email": "same@gmail.com", "password": "patient-pass"}).status_code == 200


def test_superuser_reset_only_through_admin_route(monkeypatch):
    c, svc = setup(monkeypatch)
    svc.create_superuser("boss@example.com", "superuser-password-1", "Boss")
    assert "dev_code" not in c.post("/v1/auth/clinician/forgot", json={"email": "boss@example.com"}).json()
    code = c.post("/v1/auth/admin/forgot", json={"email": "boss@example.com"}).json()["dev_code"]
    short = c.post("/v1/auth/admin/reset", json={"email": "boss@example.com", "code": code, "password": "short-pass"})
    assert short.status_code == 422  # superusers need 12+ characters
    ok = c.post("/v1/auth/admin/reset", json={"email": "boss@example.com", "code": code,
                                              "password": "a-much-longer-password"})
    assert ok.status_code == 200
    assert c.post("/v1/auth/admin/login", json={"email": "boss@example.com",
                                                "password": "a-much-longer-password"}).status_code == 200


def test_patient_email_verification_when_switched_on(monkeypatch):
    c, _ = setup(monkeypatch, verify_patients=True)
    s = patient(c)
    assert s["status"] == "verification_sent" and "token" not in s
    r = c.post("/v1/auth/login", json={"email": "pat@gmail.com", "password": "patient-pass"})
    assert r.status_code == 403 and r.json()["verification_required"]
    v = c.post("/v1/auth/verify", json={"email": "pat@gmail.com", "code": s["dev_code"]})
    assert v.status_code == 200 and v.json()["token"]
    assert c.post("/v1/auth/login", json={"email": "pat@gmail.com", "password": "patient-pass"}).status_code == 200


def test_reset_email_goes_by_smtp(monkeypatch):
    c, _ = setup(monkeypatch)
    patient(c)
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent.append({"to": msg["To"], "subject": msg["Subject"], "body": msg.get_content()})

    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("MITRAMIND_SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("MITRAMIND_SMTP_USER", "mitramind.sender@gmail.com")
    r = c.post("/v1/auth/forgot", json={"email": "pat@gmail.com"}).json()
    assert r["emailed"] and "dev_code" not in r
    assert sent[0]["to"] == "pat@gmail.com" and sent[0]["subject"].startswith("Reset your MitraMind password")
