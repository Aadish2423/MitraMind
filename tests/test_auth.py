"""Patient accounts (email + password), the per-patient route guard, and photo messages.

Run: python -m pytest tests   (needs httpx for FastAPI's TestClient)
"""
import base64
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import care.photos as photos  # noqa: E402
from care.routes import build_router  # noqa: E402
from care.service import CareService  # noqa: E402
from care.store import Store  # noqa: E402
from phase1.llm import LLMResult  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402

ORCH = Phase1Orchestrator(enable_neural=False)
CONSENTS = {"companion": True, "care_team": True}
PNG = b"\x89PNG\r\n\x1a\n" + bytes(32)
photos.PHOTO_DIR = pathlib.Path(tempfile.mkdtemp(prefix="mm_photos_"))


class FakeBackend:
    """Accepts images like the real backends; records what each call received."""
    name = "fake"

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def available(self):
        return True, ""

    def chat(self, system, user, schema, images=None):
        self.calls.append({"user": user, "schema": schema, "images": images})
        return LLMResult(self.answers.pop(0), 0.0, "fake")


def reply(text, **extra):
    return json.dumps({"reply": text, "mind_pause": "none", **extra})


def client(answers=()):
    fake = FakeBackend(answers)
    svc = CareService(Store(":memory:"), ORCH, fake)
    app = FastAPI()
    app.include_router(build_router(svc))
    return TestClient(app), svc, fake


def signup(c, email="asha@example.com", password="calm-river-42", **extra):
    r = c.post("/v1/auth/signup", json={"email": email, "password": password, "consents": CONSENTS, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_signup_login_logout():
    c, _, _ = client()
    s = signup(c)
    assert s["profile"]["id"].startswith("p_") and s["profile"]["consents"]["companion"]
    login = c.post("/v1/auth/login", json={"email": "  Asha@Example.com ", "password": "calm-river-42"})
    assert login.status_code == 200 and login.json()["profile"]["id"] == s["profile"]["id"]
    assert c.post("/v1/auth/login", json={"email": "asha@example.com", "password": "wrong-pass"}).status_code == 401
    assert c.post("/v1/auth/login", json={"email": "nobody@example.com", "password": "x" * 10}).status_code == 401
    token = login.json()["token"]
    assert c.get("/v1/auth/me", headers=auth(token)).json()["email"] == "asha@example.com"
    c.post("/v1/auth/logout", headers=auth(token))
    assert c.get("/v1/auth/me", headers=auth(token)).status_code == 401


def test_signup_rejects_bad_input():
    c, svc, _ = client()
    signup(c)
    svc.enroll("pilot_017", consents=CONSENTS)
    for body in ({"email": "not-an-email", "password": "long-enough-1"},
                 {"email": "b@example.com", "password": "short"},
                 {"email": "ASHA@example.com", "password": "another-pass-1"},  # duplicate, any case
                 {"email": "c@example.com", "password": "long-enough-1", "patient_code": "pilot_017"}):
        assert c.post("/v1/auth/signup", json={**body, "consents": CONSENTS}).status_code == 422, body


def test_signup_with_care_team_code():
    c, _, _ = client()
    assert signup(c, patient_code="pilot_042")["profile"]["id"] == "pilot_042"


def test_secrets_are_stored_hashed():
    c, svc, _ = client()
    s = signup(c)
    row = svc.store.account_by_email("asha@example.com")
    assert row["password_hash"].startswith("scrypt$") and "calm-river-42" not in row["password_hash"]
    stored = svc.store._all("SELECT token_hash FROM sessions")
    assert stored and all(r["token_hash"] != s["token"] for r in stored)


def test_patient_routes_need_the_owners_token():
    c, svc, _ = client([reply("Thank you for telling me.")])
    a, b = signup(c), signup(c, email="ravi@example.com")
    pid = a["profile"]["id"]
    assert c.get(f"/v1/patients/{pid}/messages").status_code == 401
    assert c.get(f"/v1/patients/{pid}/messages", headers=auth(b["token"])).status_code == 403
    assert c.post(f"/v1/patients/{pid}/chat", json={"text": "hi"}, headers=auth(b["token"])).status_code == 403
    assert c.post(f"/v1/patients/{pid}/chat", json={"text": "hi"}, headers=auth(a["token"])).status_code == 200
    assert c.get(f"/v1/patients/{pid}", headers=auth("made-up-token")).status_code == 401
    # Code-only patients (demo data, WhatsApp, older pilot links) keep working without a login.
    svc.enroll("pilot_001", consents=CONSENTS)
    assert c.get("/v1/patients/pilot_001/messages").status_code == 200


def test_login_is_throttled_after_repeated_failures():
    c, _, _ = client()
    signup(c)
    for _ in range(5):
        assert c.post("/v1/auth/login", json={"email": "asha@example.com", "password": "nope-nope"}).status_code == 401
    r = c.post("/v1/auth/login", json={"email": "asha@example.com", "password": "calm-river-42"})
    assert r.status_code == 429


def test_photo_message_reaches_the_llm_and_can_be_fetched():
    c, _, fake = client([reply("What a peaceful view of the sea. Does it remind you of a calm day?")])
    s = signup(c)
    pid, h = s["profile"]["id"], auth(s["token"])
    r = c.post(f"/v1/patients/{pid}/chat", headers=h,
               json={"text": "this place makes me feel calm", "photo_base64": base64.b64encode(PNG).decode()})
    assert r.status_code == 200 and r.json()["has_image"] and r.json()["reply_source"] == "llm"
    call = fake.calls[0]
    assert call["images"] == [("image/png", base64.b64encode(PNG).decode())]
    assert "image_concern" in json.dumps(call["schema"]) and "Photo:" in call["user"]
    hist = c.get(f"/v1/patients/{pid}/messages", headers=h).json()
    mid = next(m["id"] for m in hist if m["role"] == "user")
    assert next(m for m in hist if m["id"] == mid)["has_image"]
    got = c.get(f"/v1/patients/{pid}/messages/{mid}/photo", headers=h)
    assert got.status_code == 200 and got.content == PNG
    assert c.get(f"/v1/patients/{pid}/messages/{mid}/photo").status_code == 401


def test_photo_only_and_bad_uploads():
    c, _, fake = client([reply("Thank you for sharing this photo with me.")])
    s = signup(c)
    pid, h = s["profile"]["id"], auth(s["token"])
    ok = c.post(f"/v1/patients/{pid}/chat", headers=h, json={"photo_base64": base64.b64encode(PNG).decode()})
    assert ok.status_code == 200 and "[shared a photo]" in fake.calls[0]["user"]
    assert c.post(f"/v1/patients/{pid}/chat", headers=h,
                  json={"photo_base64": base64.b64encode(b"%PDF-1.7 not an image").decode()}).status_code == 422
    assert c.post(f"/v1/patients/{pid}/chat", headers=h, json={"photo_base64": "@@not base64@@"}).status_code == 422
    assert c.post(f"/v1/patients/{pid}/chat", headers=h, json={"text": "  "}).status_code == 422


def test_worrying_photo_opens_a_watch_alert_with_helpline():
    c, svc, _ = client([reply("I'm really glad you showed me. I'm worried about you.", image_concern=True)])
    s = signup(c)
    pid = s["profile"]["id"]
    out = c.post(f"/v1/patients/{pid}/chat", headers=auth(s["token"]),
                 json={"text": "", "photo_base64": base64.b64encode(PNG).decode()}).json()
    assert out["image_concern"] and "14416" in out["reply"]
    alerts = svc.store.alerts(patient_id=pid)
    assert len(alerts) == 1 and alerts[0]["level"] == "watch"


def test_text_messages_keep_the_text_only_schema():
    c, _, fake = client([reply("That sounds tiring.")])
    s = signup(c)
    c.post(f"/v1/patients/{s['profile']['id']}/chat", headers=auth(s["token"]), json={"text": "long day at work"})
    assert fake.calls[0]["images"] is None and "image_concern" not in json.dumps(fake.calls[0]["schema"])
