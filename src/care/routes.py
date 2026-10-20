"""HTTP routes for the patient app (companion, questionnaires, consent) and the clinician dashboard.

No authentication yet: this is a pilot prototype. Before any real patient uses it, put these behind
hospital SSO / OTP login with patient and clinician roles (see task.md, "Remaining").
"""
from __future__ import annotations

import base64
import binascii
from datetime import date as Date
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import instruments, mind_pause
from .service import CareService, ConsentRequired, NotFound
from .store import CONSENT_PURPOSES, CONSENT_VERSION


class Enrollment(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{2,40}$", description="pseudonymous id, never a name or phone")
    display_name: str | None = None
    preferred_language: Literal["auto", "en", "hinglish", "hi"] = "auto"
    clinician_id: str | None = None
    consents: dict[str, bool] = {}


class ConsentChange(BaseModel):
    purpose: str
    granted: bool


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    share: bool = Field(False, description="Reflection Share: let the care team read this message")


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
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def build_router(svc: CareService) -> APIRouter:
    r = APIRouter()

    # ---------------------------------------------------------------- patient app
    @r.get("/v1/consent-purposes", tags=["patient"])
    def consent_purposes():
        return {"version": CONSENT_VERSION, "purposes": CONSENT_PURPOSES}

    @r.post("/v1/patients", tags=["patient"])
    def enroll(body: Enrollment):
        return _call(svc.enroll, body.id, body.display_name, body.preferred_language, body.clinician_id,
                     consents=body.consents)

    @r.get("/v1/patients/{pid}", tags=["patient"])
    def profile(pid: str):
        return _call(svc.profile, pid)

    @r.post("/v1/patients/{pid}/consents", tags=["patient"])
    def change_consent(pid: str, body: ConsentChange):
        _call(svc.profile, pid)
        _call(svc.store.record_consent, pid, body.purpose, body.granted)
        return _call(svc.profile, pid)

    @r.post("/v1/patients/{pid}/chat", tags=["patient"])
    def chat(pid: str, body: ChatIn):
        return _call(svc.chat, pid, body.text, share=body.share)

    @r.post("/v1/patients/{pid}/voice", tags=["patient"])
    def voice(pid: str, body: VoiceIn):
        try:
            audio = base64.b64decode(body.audio_base64, validate=True)
        except binascii.Error as e:
            raise HTTPException(422, "audio_base64 is not valid base64") from e
        if len(audio) > 15 * 1024 * 1024:
            raise HTTPException(413, "voice message too long (max 15 MB)")
        return _call(svc.voice, pid, audio, body.mime, share=body.share, language=body.language)

    @r.get("/v1/patients/{pid}/messages", tags=["patient"])
    def messages(pid: str, limit: int = 50):
        return _call(svc.history, pid, limit)

    @r.post("/v1/patients/{pid}/messages/{mid}/share", tags=["patient"])
    def share(pid: str, mid: int, body: ShareIn):
        if not svc.share(pid, mid, body.shared):
            raise HTTPException(404, "no such patient message")
        return {"message_id": mid, "shared": body.shared}

    @r.get("/v1/mind-pause", tags=["patient"])
    def mind_pause_list(language: str = "en"):
        return mind_pause.catalogue(language)

    @r.post("/v1/patients/{pid}/mind-pause", tags=["patient"])
    def mind_pause_log(pid: str, body: MindPauseIn):
        if body.exercise not in mind_pause.EXERCISES:
            raise HTTPException(422, f"unknown exercise {body.exercise}")
        return {"id": _call(svc.mind_pause_done, pid, body.exercise, body.completed)}

    @r.get("/v1/questionnaires/{instrument}/form", tags=["patient"])
    def questionnaire_form(instrument: Literal["PHQ-9", "GAD-7"]):
        return instruments.form(instrument)

    @r.post("/v1/patients/{pid}/questionnaires", tags=["patient"])
    def submit_questionnaire(pid: str, body: QuestionnaireIn):
        return _call(svc.submit_questionnaire, pid, body.instrument, body.items, body.date)

    # ---------------------------------------------------------------- clinician dashboard
    @r.get("/v1/clinician/dashboard", tags=["clinician"])
    def dashboard():
        return svc.dashboard()

    @r.get("/v1/clinician/alerts", tags=["clinician"])
    def alerts(status: Literal["open", "acknowledged", "resolved", "all"] = "open"):
        return svc.alerts(None if status == "all" else status)

    @r.post("/v1/clinician/alerts/{aid}", tags=["clinician"])
    def update_alert(aid: int, body: AlertUpdate):
        if not svc.store.update_alert(aid, body.status, body.handled_by, body.note):
            raise HTTPException(404, "no such alert")
        return {"id": aid, "status": body.status}

    @r.get("/v1/clinician/patients/{pid}/state-card", tags=["clinician"])
    def clinician_card(pid: str, use_llm: bool = True):
        return _call(svc.state_card, pid, use_llm)

    @r.get("/v1/clinician/patients/{pid}/shared", tags=["clinician"])
    def shared_reflections(pid: str):
        return _call(svc.clinician_messages, pid)

    @r.post("/v1/clinician/patients/{pid}/feedback", tags=["clinician"])
    def feedback(pid: str, body: FeedbackIn):
        _call(svc.profile, pid)
        return {"id": svc.store.add_feedback(pid, body.clinician_id, body.rating, body.accurate, body.minutes_saved,
                                             body.comment, None)}

    @r.get("/v1/audit", tags=["clinician"])
    def audit(patient_id: str | None = None, limit: int = 100):
        return svc.store.audit_log(patient_id, limit)

    return r
