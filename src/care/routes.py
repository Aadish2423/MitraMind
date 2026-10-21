"""HTTP routes for the patient app (companion, questionnaires, consent) and the clinician dashboard.

Patients sign up and log in with email + password (/v1/auth/*) and send the session token as
`Authorization: Bearer <token>`. Every /v1/patients/{pid}/... route checks that token when that patient has an
account; code-only patients (demo data, tests, WhatsApp) keep the old open behaviour.

Clinicians have their own accounts (/v1/auth/clinician/*): Gmail addresses only, confirmed with a 6-digit code
sent to that Gmail, 12-hour sessions. Every /v1/clinician/... route and /v1/audit needs a verified clinician's
token. Hospital SSO can replace this later (see task.md, "Remaining").
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import time
from collections import defaultdict, deque
from datetime import date as Date
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import fhir, instruments, mind_pause
from .mailer import MailNotConfigured
from .photos import MAX_BYTES as MAX_PHOTO_BYTES
from .service import AuthError, CareService, ConsentRequired, NotFound, VerificationRequired, normalise_email
from .store import CONSENT_PURPOSES, CONSENT_VERSION
from .whatsapp import WhatsAppChannel, verify_signature


class Enrollment(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{2,40}$", description="pseudonymous id, never a name or phone")
    display_name: str | None = None
    preferred_language: Literal["auto", "en", "hinglish", "hi"] = "auto"
    clinician_id: str | None = None
    phone: str | None = Field(None, description="WhatsApp number with country code, digits only (stored hashed)")
    consents: dict[str, bool] = {}


class ConsentChange(BaseModel):
    purpose: str
    granted: bool


class SignupIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=200)
    preferred_language: Literal["auto", "en", "hinglish", "hi"] = "auto"
    consents: dict[str, bool] = {}
    patient_code: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{2,40}$",
                                     description="code from the care team, if they gave one")


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=200)


class ClinicianSignupIn(BaseModel):
    email: str = Field(max_length=254, description="a Gmail address")
    password: str = Field(max_length=200)
    name: str | None = Field(None, max_length=80, description="shown to colleagues, e.g. 'Dr. A. Sharma'")


class CodeIn(BaseModel):
    email: str = Field(max_length=254)
    code: str = Field(pattern=r"^\s*\d{6}\s*$")


class EmailIn(BaseModel):
    email: str = Field(max_length=254)


class ChatIn(BaseModel):
    text: str = Field("", max_length=4000)
    share: bool = Field(False, description="Reflection Share: let the care team read this message")
    photo_base64: str | None = Field(None, description="optional JPEG / PNG / WebP photo, base64-encoded (max 5 MB)")


class VoiceIn(BaseModel):
    audio_base64: str = Field(description="the recording, base64-encoded")
    mime: str = "audio/wav"
    share: bool = False
    language: Literal["hi", "en"] | None = Field(None, description="speech language hint; omit to auto-detect")


class ShareIn(BaseModel):
    shared: bool


class MindPauseIn(BaseModel):
    exercise: str
    completed: bool = True


class QuestionnaireIn(BaseModel):
    instrument: Literal["PHQ-9", "GAD-7"]
    items: list[int]
    date: Date | None = None


class AlertUpdate(BaseModel):
    status: Literal["acknowledged", "resolved"]
    handled_by: str | None = None
    note: str | None = None
    outcome: Literal["confirmed_risk", "false_alarm", "unclear"] | None = Field(
        None, description="when resolving: was the risk real? Labels recalibrate the crisis layer (Phase 3)")


class FeedbackIn(BaseModel):
    clinician_id: str | None = None
    rating: int | None = Field(None, ge=1, le=5)
    accurate: bool | None = None
    minutes_saved: float | None = None
    comment: str | None = None


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotFound as e:
        raise HTTPException(404, str(e)) from e
    except ConsentRequired as e:
        raise HTTPException(403, str(e)) from e
    except AuthError as e:
        raise HTTPException(401, str(e)) from e
    except MailNotConfigured as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _bearer(request: Request) -> str | None:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def ndjson_response(events) -> StreamingResponse:
    """Stream service events as newline-delimited JSON."""
    def lines():
        try:
            for ev in events:
                yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"
        except Exception as e:  # mid-stream failure: tell the app instead of silently cutting the connection
            yield json.dumps({"type": "error", "detail": f"the reply could not be finished ({e.__class__.__name__})"}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


LOGIN_WINDOW_S, LOGIN_MAX_FAILURES = 15 * 60, 5


def build_router(svc: CareService) -> APIRouter:
    r = APIRouter()
    failures: dict[str, deque] = defaultdict(deque)  # email -> times of recent failed logins (per process)

    def guard(request: Request, pid: str) -> None:
        """A patient with an account can only be read or changed with that patient's own session token."""
        if svc.store.account_for_patient(pid) is None:
            return
        acc = svc.account(_bearer(request))
        if acc is None:
            raise HTTPException(401, "please log in again")
        if acc["patient_id"] != pid:
            raise HTTPException(403, "this is not your account")

    # ---------------------------------------------------------------- patient accounts
    @r.post("/v1/auth/signup", tags=["auth"])
    def signup(body: SignupIn):
        return _call(svc.signup, body.email, body.password, consents=body.consents,
                     preferred_language=body.preferred_language, patient_code=body.patient_code)

    @r.post("/v1/auth/login", tags=["auth"])
    def login(body: LoginIn):
        email, t = normalise_email(body.email), time.monotonic()
        recent = failures[email]
        while recent and t - recent[0] > LOGIN_WINDOW_S:
            recent.popleft()
        if len(recent) >= LOGIN_MAX_FAILURES:
            raise HTTPException(429, "too many attempts; please wait 15 minutes and try again")
        try:
            out = svc.login(email, body.password)
        except AuthError as e:
            recent.append(t)
            raise HTTPException(401, str(e)) from e
        failures.pop(email, None)
        return out

    @r.post("/v1/auth/logout", tags=["auth"])
    def logout(request: Request):
        if token := _bearer(request):
            svc.logout(token)
        return {"ok": True}

    @r.get("/v1/auth/me", tags=["auth"])
    def me(request: Request):
        acc = svc.account(_bearer(request))
        if acc is None:
            raise HTTPException(401, "please log in again")
        return {"email": acc["email"], "profile": _call(svc.profile, acc["patient_id"])}

    # ---------------------------------------------------------------- clinician accounts
    def too_many(key: str) -> deque:
        t, recent = time.monotonic(), failures[key]
        while recent and t - recent[0] > LOGIN_WINDOW_S:
            recent.popleft()
        if len(recent) >= LOGIN_MAX_FAILURES:
            raise HTTPException(429, "too many attempts; please wait 15 minutes and try again")
        return recent

    @r.post("/v1/auth/clinician/signup", tags=["auth"])
    def clinician_signup(body: ClinicianSignupIn):
        return _call(svc.clinician_signup, body.email, body.password, body.name)

    @r.post("/v1/auth/clinician/verify", tags=["auth"])
    def clinician_verify(body: CodeIn):
        key = "verify:" + normalise_email(body.email)
        recent = too_many(key)
        try:
            out = svc.verify_clinician(body.email, body.code)
        except AuthError as e:
            recent.append(time.monotonic())
            raise HTTPException(401, str(e)) from e
        failures.pop(key, None)
        return out

    @r.post("/v1/auth/clinician/resend", tags=["auth"])
    def clinician_resend(body: EmailIn):
        return _call(svc.resend_clinician_code, body.email)

    @r.post("/v1/auth/clinician/login", tags=["auth"])
    def clinician_login(body: LoginIn):
        key = "clinician:" + normalise_email(body.email)
        recent = too_many(key)
        try:
            out = svc.clinician_login(body.email, body.password)
        except AuthError as e:
            recent.append(time.monotonic())
            raise HTTPException(401, str(e)) from e
        except VerificationRequired as e:
            detail = {"detail": str(e), "verification_required": True}
            if e.dev_code:
                detail["dev_code"] = e.dev_code
            return JSONResponse(detail, status_code=403)
        except MailNotConfigured as e:
            raise HTTPException(503, str(e)) from e
        failures.pop(key, None)
        return out

    @r.post("/v1/auth/clinician/logout", tags=["auth"])
    def clinician_logout(request: Request):
        if token := _bearer(request):
            svc.clinician_logout(token)
        return {"ok": True}

    def clinician_only(request: Request) -> dict:
        acc = svc.clinician(_bearer(request))
        if acc is None:
            raise HTTPException(401, "clinician login required")
        return acc

    @r.get("/v1/auth/clinician/me", tags=["auth"])
    def clinician_me(acc: dict = Depends(clinician_only)):
        return {"role": "clinician", "clinician": svc.clinician_profile(acc)}

    # ---------------------------------------------------------------- patient app
    @r.get("/v1/consent-purposes", tags=["patient"])
    def consent_purposes():
        return {"version": CONSENT_VERSION, "purposes": CONSENT_PURPOSES}

    @r.post("/v1/patients", tags=["patient"])
    def enroll(body: Enrollment):
        return _call(svc.enroll, body.id, body.display_name, body.preferred_language, body.clinician_id,
                     phone=body.phone, consents=body.consents)

    @r.get("/v1/patients/{pid}", tags=["patient"])
    def profile(pid: str, request: Request):
        guard(request, pid)
        return _call(svc.profile, pid)

    @r.post("/v1/patients/{pid}/consents", tags=["patient"])
    def change_consent(pid: str, body: ConsentChange, request: Request):
        guard(request, pid)
        _call(svc.profile, pid)
        _call(svc.store.record_consent, pid, body.purpose, body.granted)
        return _call(svc.profile, pid)

    def decode_photo(body: ChatIn) -> bytes | None:
        if not body.photo_base64:
            return None
        try:
            photo = base64.b64decode(body.photo_base64, validate=True)
        except binascii.Error as e:
            raise HTTPException(422, "photo_base64 is not valid base64") from e
        if len(photo) > MAX_PHOTO_BYTES:
            raise HTTPException(413, "photos can be at most 5 MB")
        return photo

    @r.post("/v1/patients/{pid}/chat", tags=["patient"])
    def chat(pid: str, body: ChatIn, request: Request):
        guard(request, pid)
        return _call(svc.chat, pid, body.text, share=body.share, photo=decode_photo(body))

    @r.post("/v1/patients/{pid}/chat/stream", tags=["patient"])
    def chat_stream(pid: str, body: ChatIn, request: Request):
        """Same as /chat, but the reply arrives sentence by sentence as newline-delimited JSON events:
        {"type": "sentence", "text"} ..., optionally {"type": "replace", "text"}, then {"type": "final", ...}."""
        guard(request, pid)
        return ndjson_response(_call(svc.chat_stream, pid, body.text, share=body.share, photo=decode_photo(body)))

    @r.get("/v1/patients/{pid}/messages/{mid}/photo", tags=["patient"])
    def message_photo(pid: str, mid: int, request: Request):
        guard(request, pid)
        return FileResponse(_call(svc.photo, pid, mid), headers={"Cache-Control": "private, max-age=3600"})

    def decode_audio(body: VoiceIn) -> bytes:
        try:
            audio = base64.b64decode(body.audio_base64, validate=True)
        except binascii.Error as e:
            raise HTTPException(422, "audio_base64 is not valid base64") from e
        if len(audio) > 15 * 1024 * 1024:
            raise HTTPException(413, "voice message too long (max 15 MB)")
        return audio

    @r.post("/v1/patients/{pid}/voice", tags=["patient"])
    def voice(pid: str, body: VoiceIn, request: Request):
        guard(request, pid)
        return _call(svc.voice, pid, decode_audio(body), body.mime, share=body.share, language=body.language)

    @r.post("/v1/patients/{pid}/voice/stream", tags=["patient"])
    def voice_stream(pid: str, body: VoiceIn, request: Request):
        """Same as /voice, but streams NDJSON: {"type": "transcript"} first, then the /chat/stream events."""
        guard(request, pid)
        return ndjson_response(_call(svc.voice_stream, pid, decode_audio(body), body.mime, share=body.share,
                                     language=body.language))

    @r.get("/v1/patients/{pid}/messages", tags=["patient"])
    def messages(pid: str, request: Request, limit: int = 50):
        guard(request, pid)
        return _call(svc.history, pid, limit)

    @r.post("/v1/patients/{pid}/messages/{mid}/share", tags=["patient"])
    def share(pid: str, mid: int, body: ShareIn, request: Request):
        guard(request, pid)
        if not svc.share(pid, mid, body.shared):
            raise HTTPException(404, "no such patient message")
        return {"message_id": mid, "shared": body.shared}

    @r.get("/v1/mind-pause", tags=["patient"])
    def mind_pause_list(language: str = "en"):
        return mind_pause.catalogue(language)

    @r.post("/v1/patients/{pid}/mind-pause", tags=["patient"])
    def mind_pause_log(pid: str, body: MindPauseIn, request: Request):
        guard(request, pid)
        if body.exercise not in mind_pause.EXERCISES:
            raise HTTPException(422, f"unknown exercise {body.exercise}")
        return {"id": _call(svc.mind_pause_done, pid, body.exercise, body.completed)}

    @r.get("/v1/questionnaires/{instrument}/form", tags=["patient"])
    def questionnaire_form(instrument: Literal["PHQ-9", "GAD-7"]):
        return instruments.form(instrument)

    @r.post("/v1/patients/{pid}/questionnaires", tags=["patient"])
    def submit_questionnaire(pid: str, body: QuestionnaireIn, request: Request):
        guard(request, pid)
        return _call(svc.submit_questionnaire, pid, body.instrument, body.items, body.date)

    # ---------------------------------------------------------------- clinician dashboard (verified clinicians only)
    def who(acc: dict) -> str:
        return acc["name"] or acc["email"]

    @r.get("/v1/clinician/dashboard", tags=["clinician"])
    def dashboard(acc: dict = Depends(clinician_only)):
        return svc.dashboard()

    @r.get("/v1/clinician/alerts", tags=["clinician"])
    def alerts(status: Literal["open", "acknowledged", "resolved", "all"] = "open",
               acc: dict = Depends(clinician_only)):
        return svc.alerts(None if status == "all" else status)

    @r.post("/v1/clinician/alerts/{aid}", tags=["clinician"])
    def update_alert(aid: int, body: AlertUpdate, acc: dict = Depends(clinician_only)):
        # The person who handled the alert is the logged-in clinician, not whatever the page sends.
        if not svc.store.update_alert(aid, body.status, who(acc), body.note, body.outcome):
            raise HTTPException(404, "no such alert")
        return {"id": aid, "status": body.status, "handled_by": who(acc)}

    @r.get("/v1/clinician/patients/{pid}/state-card", tags=["clinician"])
    def clinician_card(pid: str, use_llm: bool = True, acc: dict = Depends(clinician_only)):
        return _call(svc.state_card, pid, use_llm)

    @r.get("/v1/clinician/patients/{pid}/shared", tags=["clinician"])
    def shared_reflections(pid: str, acc: dict = Depends(clinician_only)):
        return _call(svc.clinician_messages, pid)

    @r.post("/v1/clinician/patients/{pid}/feedback", tags=["clinician"])
    def feedback(pid: str, body: FeedbackIn, acc: dict = Depends(clinician_only)):
        _call(svc.profile, pid)
        return {"id": svc.store.add_feedback(pid, f"clinician:{acc['id']}", body.rating, body.accurate,
                                             body.minutes_saved, body.comment, None)}

    @r.get("/v1/clinician/patients/{pid}/fhir", tags=["clinician"])
    def fhir_export(pid: str, include_card: bool = False, acc: dict = Depends(clinician_only)):
        _call(svc._require_care_team, pid)
        md = _call(svc.state_card, pid, False)["markdown"] if include_card else None
        return fhir.patient_bundle(svc.store, pid, md)

    # ---------------------------------------------------------------- WhatsApp Business webhook
    wa = WhatsAppChannel(svc)

    @r.get("/webhooks/whatsapp", tags=["whatsapp"])
    def wa_verify(request: Request):
        q = request.query_params
        token = os.environ.get("WHATSAPP_VERIFY_TOKEN")
        if q.get("hub.mode") == "subscribe" and token and q.get("hub.verify_token") == token:
            return PlainTextResponse(q.get("hub.challenge", ""))
        raise HTTPException(403, "verification failed")

    @r.post("/webhooks/whatsapp", tags=["whatsapp"])
    async def wa_inbound(request: Request, background: BackgroundTasks):
        body = await request.body()
        if not verify_signature(body, request.headers.get("X-Hub-Signature-256")):
            raise HTTPException(403, "bad signature")
        # Answer Meta at once; the companion reply (LLM, speech-to-text) can take tens of seconds.
        background.add_task(wa.handle, json.loads(body or b"{}"))
        return {"status": "accepted"}

    @r.get("/v1/whatsapp/outbox", tags=["whatsapp"])
    def wa_outbox(limit: int = 20, acc: dict = Depends(clinician_only)):
        """Replies the WhatsApp channel sent or, without credentials, would have sent (testing)."""
        return wa.sender.outbox[-limit:]

    @r.get("/v1/audit", tags=["clinician"])
    def audit(patient_id: str | None = None, limit: int = 100, acc: dict = Depends(clinician_only)):
        return svc.store.audit_log(patient_id, limit)

    return r
