"""CareService: the one place that ties storage, Phase 1 analysis, the companion and triage together.

Used by the API, the WhatsApp webhook and the tests, so all channels follow the same rules:
- no companion use without `companion` consent;
- every patient message runs through the Phase 1 safety layer before any reply;
- any risk above "none" (message or questionnaire) opens an alert for the care team;
- clinicians see usage counts, scores, alerts and only the messages the patient shared,
  and nothing at all without `care_team` consent (safety alerts excepted, see store.CONSENT_PURPOSES).
"""
from __future__ import annotations

import os
import re
import secrets
from datetime import date as Date
from datetime import datetime, timedelta, timezone

from phase1.crisis import ACTIONS, HELPLINES, combine
from phase1.orchestrator import COPING_LABELS, STRESSOR_LABELS
from phase1.questionnaires import MEANINGFUL_CHANGE, score, trends
from phase1.render import card_to_markdown
from phase1.schema import Entry, PatientBundle, QuestionnaireResponse

from . import mailer
from .companion import Companion, history_offset
from .photos import MIME, as_base64, save_photo
from .store import CODE_MINUTES, LEVEL_ORDER, Store

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8
MIN_CLINICIAN_PASSWORD = 10
RESEND_SECONDS = 60
# How fast an alert should be picked up, from the crisis layer's recommended actions ("urgent: call now",
# "high: within 24 hours"). Confirm with the hospital's protocol.
ACK_TARGET_HOURS = {"urgent": 1, "high": 24, "watch": 72}


CODE_ERRORS = {"wrong": "that code is not right; please check it or ask for a new one",
               "expired": "that code has expired; please ask for a new one",
               "too_many": "too many wrong codes; please ask for a new one",
               "missing": "please ask for a new code"}


def patient_email_verification() -> bool:
    """MITRAMIND_PATIENT_EMAIL_VERIFY=1: new patients confirm their email with a code before their first login.
    Off by default so code-only, demo and WhatsApp patients, and older tests, keep working."""
    return os.environ.get("MITRAMIND_PATIENT_EMAIL_VERIFY", "0") == "1"


def clinician_domains() -> set[str]:
    """Email domains a clinician may sign up with (Gmail only by default)."""
    raw = os.environ.get("MITRAMIND_CLINICIAN_DOMAINS", "gmail.com,googlemail.com")
    return {d.strip().lower() for d in raw.split(",") if d.strip()}
PHOTO_ALERT_REASON = "photo may show self-harm or danger (flagged by the companion's image check; please review)"


class ConsentRequired(Exception):
    pass


class NotFound(Exception):
    pass


class AuthError(Exception):
    pass


class Forbidden(Exception):
    """A clinician asked for a patient who is not theirs."""


class VerificationRequired(Exception):
    """The clinician has not confirmed their Gmail yet; a code was (re)sent."""

    def __init__(self, message: str, dev_code: str | None = None):
        super().__init__(message)
        self.dev_code = dev_code


def normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def _jsonable_analysis(a: dict) -> dict:
    return {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in a.items() if k not in ("text",)}


class CareService:
    def __init__(self, store: Store, orchestrator, chat_backend=None, stt=None):
        self.store, self.orch, self.stt = store, orchestrator, stt
        self.companion = Companion(chat_backend)

    # ------------------------------------------------------------------ patients
    def enroll(self, patient_id: str, display_name: str | None = None, preferred_language: str = "auto",
               clinician_id: str | None = None, phone: str | None = None, consents: dict[str, bool] | None = None,
               channel: str = "app") -> dict:
        if self.store.patient(patient_id):
            raise ValueError(f"patient {patient_id} already exists")
        self.store.create_patient(patient_id, display_name, preferred_language, clinician_id, phone)
        for purpose, granted in (consents or {}).items():
            self.store.record_consent(patient_id, purpose, granted, channel)
        return self.profile(patient_id)

    def profile(self, patient_id: str) -> dict:
        p = self._patient(patient_id)
        doc = self._linked_clinician(p)
        return {**{k: p[k] for k in ("id", "display_name", "preferred_language", "clinician_id", "created_at")},
                "clinician_name": (doc["name"] or doc["email"]) if doc else None,
                "consents": self.store.consents(patient_id)}

    def _linked_clinician(self, patient: dict) -> dict | None:
        cid = patient.get("clinician_id")
        return self.store.clinician_by_id(int(cid)) if cid and str(cid).isdigit() else None

    def link_clinician(self, patient_id: str, care_code: str) -> dict:
        """The patient enters the code their clinician gave them; from then on that clinician sees them."""
        self._patient(patient_id)
        doc = self.store.clinician_by_code(care_code)
        if doc is None:
            raise ValueError("we couldn't find a clinician with that code; please check it with your clinic")
        self.store.set_patient_clinician(patient_id, doc["id"])
        return self.profile(patient_id)

    def _patient(self, patient_id: str) -> dict:
        p = self.store.patient(patient_id)
        if p is None:
            raise NotFound(f"no patient {patient_id}")
        return p

    # ------------------------------------------------------------------ patient accounts
    def signup(self, email: str, password: str, *, consents: dict[str, bool] | None = None,
               preferred_language: str = "auto", patient_code: str | None = None,
               clinician_code: str | None = None) -> dict:
        """New account + new patient record. `patient_code` is the code the care team gave, if any; it must be new,
        so nobody can sign up onto an existing patient's history by guessing their code."""
        email = normalise_email(email)
        if not EMAIL_RE.match(email):
            raise ValueError("please enter a valid email address")
        if len(password or "") < MIN_PASSWORD:
            raise ValueError(f"the password needs at least {MIN_PASSWORD} characters")
        if self.store.account_by_email(email):
            raise ValueError("an account with this email already exists; please log in")
        if patient_code and self.store.patient(patient_code):
            raise ValueError("this patient code is already in use; please check it with your care team")
        doc = None
        if clinician_code and clinician_code.strip():
            doc = self.store.clinician_by_code(clinician_code)
            if doc is None:
                raise ValueError("we couldn't find a clinician with that code; please check it or leave it empty")
        pid = patient_code or self._new_patient_id()
        self.enroll(pid, preferred_language=preferred_language, consents=consents)
        if doc:
            self.store.set_patient_clinician(pid, doc["id"])
        verify = patient_email_verification()
        acc = self.store.create_account(email, password, pid, verified=not verify)
        if verify:  # no session until the patient types the code we emailed
            return self._send_code(email, "verify-patient", force=True)
        return self._session(acc)

    def login(self, email: str, password: str) -> dict:
        email = normalise_email(email)
        acc = self.store.check_login(email, password or "")
        if acc is None:
            raise AuthError("wrong email or password")
        if not acc["email_verified"]:
            raise VerificationRequired("please confirm your email first; we sent you a 6-digit code",
                                       self._send_code_quietly(email, "verify-patient"))
        return self._session(acc)

    def verify_patient(self, email: str, code: str) -> dict:
        email = normalise_email(email)
        if self.store.account_by_email(email) is None:
            raise AuthError(CODE_ERRORS["wrong"])
        self._check_code(email, "verify-patient", code)
        self.store.mark_account_verified(email)
        return self._session(self.store.account_by_email(email))

    def resend_patient_code(self, email: str) -> dict:
        email = normalise_email(email)
        acc = self.store.account_by_email(email)
        if acc is None or acc["email_verified"]:
            return {"status": "verification_sent", "email": email, "emailed": mailer.configured()}
        return self._send_code(email, "verify-patient")

    # ------------------------------------------------------------------ forgot password (all three roles)
    def _reset_account(self, role: str, email: str) -> dict | None:
        if role == "patient":
            return self.store.account_by_email(email)
        acc = self.store.clinician_by_email(email)
        return acc if acc and bool(acc["is_superuser"]) == (role == "superuser") else None

    def request_password_reset(self, role: str, email: str) -> dict:
        """Email a reset code. The answer is the same whether or not the account exists."""
        email = normalise_email(email)
        generic = {"status": "reset_sent", "email": email, "emailed": mailer.configured()}
        if self._reset_account(role, email) is None:
            return generic
        try:
            return {**self._send_code(email, f"reset-{role}"), "status": "reset_sent"}
        except ValueError:  # asked again within a minute: the earlier code still works
            return generic

    def reset_password(self, role: str, email: str, code: str, password: str) -> dict:
        email = normalise_email(email)
        need = {"patient": MIN_PASSWORD, "clinician": MIN_CLINICIAN_PASSWORD, "superuser": 12}[role]
        if len(password or "") < need:
            raise ValueError(f"the new password needs at least {need} characters")
        if self._reset_account(role, email) is None:
            raise AuthError(CODE_ERRORS["wrong"])
        self._check_code(email, f"reset-{role}", code)
        if role == "patient":
            self.store.set_account_password(email, password)
            self.store.mark_account_verified(email)  # the code proved they own the address
        else:
            self.store.set_clinician_password(email, password)
            self.store.mark_clinician_verified(email)
        return {"status": "password_reset", "email": email}

    def account(self, token: str | None) -> dict | None:
        return self.store.session_account(token) if token else None

    def logout(self, token: str) -> None:
        self.store.end_session(token)

    def _session(self, acc: dict) -> dict:
        return {"token": self.store.new_session(acc["id"]), "email": acc["email"],
                "profile": self.profile(acc["patient_id"])}

    # ------------------------------------------------------------------ clinician accounts (Gmail-verified)
    def clinician_signup(self, email: str, password: str, name: str | None = None) -> dict:
        """Create (or, if never verified, restart) a clinician account and email a 6-digit code to the Gmail."""
        email = normalise_email(email)
        if not EMAIL_RE.match(email):
            raise ValueError("please enter a valid email address")
        domains = clinician_domains()
        if email.rsplit("@", 1)[1] not in domains:
            raise ValueError("clinician accounts need a Gmail address (" + ", ".join(sorted(domains)) + ")")
        if len(password or "") < MIN_CLINICIAN_PASSWORD:
            raise ValueError(f"clinician passwords need at least {MIN_CLINICIAN_PASSWORD} characters")
        name = (name or "").strip()[:80] or None
        existing = self.store.clinician_by_email(email)
        if existing and existing["verified_at"]:
            raise ValueError("a clinician account with this email already exists; please log in")
        if existing:
            self.store.reset_unverified_clinician(email, password, name)
        else:
            self.store.create_clinician(email, password, name)
        return self._send_code(email, "verify-clinician", force=True)

    def resend_clinician_code(self, email: str) -> dict:
        email = normalise_email(email)
        acc = self.store.clinician_by_email(email)
        if acc is None or acc["verified_at"]:
            # Same answer either way, so this endpoint does not reveal which emails have accounts.
            return {"status": "verification_sent", "email": email, "emailed": mailer.configured()}
        return self._send_code(email, "verify-clinician")

    def _send_code(self, email: str, purpose: str, force: bool = False) -> dict:
        """Email a 6-digit code for `purpose` (verify-patient, verify-clinician, reset-<role>)."""
        key = f"{purpose}:{email}"
        sent = self.store.code_sent_at(key)
        if not force and sent and datetime.fromisoformat(sent) > datetime.now(timezone.utc) - timedelta(
                seconds=RESEND_SECONDS):
            raise ValueError("please wait a minute before asking for another code")
        code = f"{secrets.randbelow(10 ** 6):06d}"
        self.store.set_email_code(key, code)
        emailed = mailer.send_code(email, code, CODE_MINUTES, purpose)
        out = {"status": "verification_sent", "email": email, "emailed": emailed}
        if not emailed:
            out["dev_code"] = code  # dev mode without SMTP only (see mailer.py)
        return out

    def _send_code_quietly(self, email: str, purpose: str) -> str | None:
        """(Re)send on login; within a minute of the last code keep that one. Returns the dev code, if any."""
        try:
            return self._send_code(email, purpose).get("dev_code")
        except ValueError:
            return None

    def _check_code(self, email: str, purpose: str, code: str) -> None:
        result = self.store.check_email_code(f"{purpose}:{email}", (code or "").strip())
        if result != "ok":
            raise AuthError(CODE_ERRORS[result])

    def verify_clinician(self, email: str, code: str) -> dict:
        email = normalise_email(email)
        acc = self.store.clinician_by_email(email)
        if acc is None:
            raise AuthError(CODE_ERRORS["wrong"])
        self._check_code(email, "verify-clinician", code)
        self.store.mark_clinician_verified(email)
        return self._clinician_session(self.ensure_care_code(self.store.clinician_by_email(email)))

    def _new_care_code(self) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I, so it is easy to read out
        while True:
            code = "MM-" + "".join(secrets.choice(alphabet) for _ in range(6))
            if not self.store.clinician_by_code(code):
                return code

    def clinician_login(self, email: str, password: str, superuser: bool = False) -> dict:
        """Clinician login, or superuser login with superuser=True. Each form only accepts its own kind of
        account, with the same message as a wrong password, so neither form reveals the other's accounts."""
        email = normalise_email(email)
        acc = self.store.check_clinician_login(email, password or "")
        if acc is None or bool(acc["is_superuser"]) != superuser:
            raise AuthError("wrong email or password")
        if not acc["verified_at"]:
            raise VerificationRequired("please confirm your Gmail first; we sent you a 6-digit code",
                                       self._send_code_quietly(email, "verify-clinician"))
        return self._clinician_session(self.ensure_care_code(acc))

    def ensure_care_code(self, acc: dict) -> dict:
        """Clinician accounts verified before care codes existed get one the next time they are used."""
        if acc["care_code"] or acc["is_superuser"]:
            return acc
        self.store.set_care_code(acc["id"], self._new_care_code())
        return self.store.clinician_by_id(acc["id"])

    def clinician(self, token: str | None) -> dict | None:
        return self.store.clinician_session(token) if token else None

    def clinician_logout(self, token: str) -> None:
        self.store.end_clinician_session(token)

    def _clinician_session(self, acc: dict) -> dict:
        return {"token": self.store.new_clinician_session(acc["id"]), "role": self.role(acc),
                "clinician": self.clinician_profile(acc)}

    @staticmethod
    def role(acc: dict) -> str:
        return "superuser" if acc.get("is_superuser") else "clinician"

    @classmethod
    def clinician_profile(cls, acc: dict) -> dict:
        return {"id": acc["id"], "email": acc["email"], "name": acc["name"], "verified_at": acc["verified_at"],
                "role": cls.role(acc), "care_code": acc.get("care_code")}

    def create_superuser(self, email: str, password: str, name: str | None = None) -> dict:
        email = normalise_email(email)
        if not EMAIL_RE.match(email):
            raise ValueError("please enter a valid email address")
        if len(password or "") < 12:
            raise ValueError("superuser passwords need at least 12 characters")
        return self.clinician_profile(self.store.upsert_superuser(email, password, name))

    # ------------------------------------------------------------------ who may see which patient
    def can_see(self, acc: dict, patient: dict) -> bool:
        return bool(acc.get("is_superuser")) or str(patient.get("clinician_id") or "") == str(acc["id"])

    def require_access(self, acc: dict, patient_id: str) -> dict:
        p = self._patient(patient_id)
        if not self.can_see(acc, p):
            raise Forbidden("this patient is not linked to you")
        return p

    def visible_patients(self, acc: dict) -> list[dict]:
        return [p for p in self.store.patients() if self.can_see(acc, p)]

    def assign_clinician(self, patient_id: str, clinician_id: int | None) -> dict:
        self._patient(patient_id)
        if clinician_id is not None:
            doc = self.store.clinician_by_id(clinician_id)
            if doc is None or not doc["verified_at"] or doc["is_superuser"]:
                raise ValueError("no verified clinician with that id")
        self.store.set_patient_clinician(patient_id, clinician_id)
        return self.profile(patient_id)

    def _new_patient_id(self) -> str:
        while True:
            pid = "p_" + secrets.token_hex(4)
            if not self.store.patient(pid):
                return pid

    # ------------------------------------------------------------------ companion
    def chat(self, patient_id: str, text: str, *, share: bool = False, modality: str = "text", channel: str = "app",
             audio_path: str | None = None, language: str = "auto", photo: bytes | None = None) -> dict:
        turn = self._start_turn(patient_id, text, share=share, modality=modality, channel=channel,
                                audio_path=audio_path, language=language, photo=photo)
        out = self.companion.respond(turn["history"], turn["text"], turn["analysis"], images=turn["images"])
        return self._finish_turn(turn, out)

    def chat_stream(self, patient_id: str, text: str, *, share: bool = False, photo: bytes | None = None,
                    modality: str = "text", channel: str = "app", audio_path: str | None = None):
        """Same rules as chat(), but returns an iterator of companion events ("sentence" / "replace") ending in
        {"type": "final", **chat()-style result}. Input problems raise here, before anything is streamed."""
        turn = self._start_turn(patient_id, text, share=share, photo=photo, modality=modality, channel=channel,
                                audio_path=audio_path)

        def events():
            final = None
            for ev in self.companion.respond_stream(turn["history"], turn["text"], turn["analysis"],
                                                    images=turn["images"]):
                if ev["type"] == "final":
                    final = {k: v for k, v in ev.items() if k != "type"}
                else:
                    yield ev
            yield {"type": "final", **self._finish_turn(turn, final)}

        return events()

    def _start_turn(self, patient_id: str, text: str, *, share: bool = False, modality: str = "text",
                    channel: str = "app", audio_path: str | None = None, language: str = "auto",
                    photo: bytes | None = None) -> dict:
        """Checks, photo storage, Phase 1 analysis (safety layer) and storing the patient's message."""
        self._patient(patient_id)
        if not self.store.has_consent(patient_id, "companion"):
            raise ConsentRequired("the patient has not consented to the companion")
        text = (text or "").strip()
        if not text and not photo:
            raise ValueError("write a message or attach a photo")
        photo_path = save_photo(patient_id, photo) if photo else None
        # Language is detected per message (patients switch scripts mid-conversation), not taken from the profile.
        entry = Entry(entry_id="live", timestamp=datetime.now(timezone.utc), text=text,
                      audio_path=audio_path if modality == "voice" else None, language=language,
                      shared_with_clinician=share)
        analysis = self.orch.analyze_entry(entry)
        history = self.store.messages_from(patient_id, history_offset(self.store.message_count(patient_id)))
        msg_id = self.store.add_message(patient_id, "user", text, modality=modality, channel=channel,
                                        language=analysis["language"], shared=share, audio_path=audio_path,
                                        risk_level=analysis["risk"].level, analysis=_jsonable_analysis(analysis),
                                        image_path=str(photo_path) if photo_path else None)
        return {"patient_id": patient_id, "text": text, "analysis": analysis, "history": history, "msg_id": msg_id,
                "modality": modality, "channel": channel, "photo_path": photo_path,
                "images": [(MIME[photo_path.suffix], as_base64(photo_path))] if photo_path else None}

    def _finish_turn(self, turn: dict, out: dict) -> dict:
        """Store the reply, write the audit log, open alerts, and build the API result."""
        patient_id, analysis, msg_id = turn["patient_id"], turn["analysis"], turn["msg_id"]
        risk = analysis["risk"]
        reply_id = self.store.add_message(patient_id, "assistant", out["reply"], modality=turn["modality"],
                                          channel=turn["channel"], language=analysis["language"],
                                          reply_source=out["source"])
        self.store.audit("crisis_reply" if out["source"] == "crisis_protocol" else "chat_reply", patient_id=patient_id,
                         model=out["model"], prompt=out["prompt"], output=out["reply"], guardrails=out["log"],
                         attempts=out["attempts"], risk_level=risk.level)
        alert_id = None
        if risk.level != "none":
            alert_id = self.store.add_alert(patient_id, risk.level, risk.reasons or ["crisis language"], "message", msg_id)
        elif out["image_concern"]:
            alert_id = self.store.add_alert(patient_id, "watch", [PHOTO_ALERT_REASON], "message", msg_id)
        return {"message_id": msg_id, "reply_id": reply_id, "reply": out["reply"], "reply_source": out["source"],
                "mind_pause": out["mind_pause"], "language": analysis["language"], "risk": risk.model_dump(),
                "escalate": risk.level in ("high", "urgent"), "helplines": HELPLINES if risk.level != "none" else [],
                "alert_id": alert_id, "models_used": analysis["models_used"], "emotion": analysis["emotion"],
                "llm_attempts": out["attempts"], "has_image": turn["photo_path"] is not None,
                "image_concern": out["image_concern"]}

    def voice(self, patient_id: str, audio: bytes, mime: str = "audio/wav", *, share: bool = False,
              channel: str = "app", language: str | None = None) -> dict:
        """Voice message: transcribe, then the same path as text, plus voice emotion models on the clip."""
        wav, stt = self._transcribe(patient_id, audio, mime, language)
        out = self.chat(patient_id, stt["text"], share=share, modality="voice", channel=channel, audio_path=str(wav))
        return {**out, **self._stt_fields(stt)}

    def voice_stream(self, patient_id: str, audio: bytes, mime: str = "audio/wav", *, share: bool = False,
                     language: str | None = None):
        """Voice message with a streamed reply: first {"type": "transcript", ...}, then the chat_stream() events.
        Transcription and input checks happen here, before anything is streamed."""
        wav, stt = self._transcribe(patient_id, audio, mime, language)

        def gen():
            # The transcript goes out before the safety and emotion analysis, so the patient sees their words at once.
            yield {"type": "transcript", "text": stt["text"], **self._stt_fields(stt)}
            for ev in self.chat_stream(patient_id, stt["text"], share=share, modality="voice", audio_path=str(wav)):
                yield {**ev, **self._stt_fields(stt)} if ev["type"] == "final" else ev

        return gen()

    def _transcribe(self, patient_id: str, audio: bytes, mime: str, language: str | None):
        from .voice import VoiceError, save_clip

        self._patient(patient_id)
        if not self.store.has_consent(patient_id, "companion"):
            raise ConsentRequired("the patient has not consented to the companion")
        if self.stt is None:
            raise ValueError("speech-to-text is disabled on this server (MITRAMIND_STT=none)")
        try:
            wav = save_clip(patient_id, audio, mime)
            stt = self.stt.transcribe(wav, language)
        except VoiceError as e:
            raise ValueError(str(e)) from e
        if not stt["text"]:
            raise ValueError("no speech was recognised in the recording")
        return wav, stt

    @staticmethod
    def _stt_fields(stt: dict) -> dict:
        return {"transcript": stt["text"], "stt_engine": stt["engine"], "audio_seconds": stt["seconds"]}

    def history(self, patient_id: str, limit: int = 50) -> list[dict]:
        self._patient(patient_id)
        return [{**{k: m[k] for k in ("id", "role", "text", "modality", "language", "shared", "risk_level",
                                      "reply_source", "created_at")}, "has_image": bool(m.get("image_path"))}
                for m in self.store.messages(patient_id, limit=limit)]

    def photo(self, patient_id: str, message_id: int) -> str:
        m = self.store.message(patient_id, message_id)
        if not m or not m.get("image_path"):
            raise NotFound("no photo on this message")
        return m["image_path"]

    def share(self, patient_id: str, message_id: int, shared: bool) -> bool:
        return self.store.set_shared(patient_id, message_id, shared)

    def mind_pause_done(self, patient_id: str, exercise: str, completed: bool) -> int:
        self._patient(patient_id)
        return self.store.add_mind_pause(patient_id, exercise, completed)

    # ------------------------------------------------------------------ questionnaires
    def submit_questionnaire(self, patient_id: str, instrument: str, items: list[int], on: Date | None = None,
                             channel: str = "app") -> dict:
        self._patient(patient_id)
        on = on or datetime.now(timezone.utc).date()
        s = score(QuestionnaireResponse(instrument=instrument, date=on, items=items))  # validates items
        qid = self.store.add_questionnaire(patient_id, instrument, on.isoformat(), items, channel)
        tr = trends(self._responses(patient_id))
        # Only a new PHQ-9 is new evidence for these PHQ-9 rules (item 9, worsening). Judging the stored PHQ-9 again on
        # a GAD-7 would re-open the same alert on every later check-in.
        phq = tr.get("PHQ-9") if instrument == "PHQ-9" else None
        risk = combine([], phq.history if phq else [], phq.change if phq and phq.n_administrations > 1 else None)
        alert_id = None
        if risk.level != "none":
            alert_id = self.store.add_alert(patient_id, risk.level, risk.reasons, "questionnaire", qid)
        return {"questionnaire_id": qid, "score": s.model_dump(), "trend": tr[instrument].model_dump(),
                "risk": {"level": risk.level, "reasons": risk.reasons}, "alert_id": alert_id,
                "helplines": HELPLINES if risk.level in ("high", "urgent") else []}

    def _responses(self, patient_id: str) -> list[QuestionnaireResponse]:
        return [QuestionnaireResponse(instrument=q["instrument"], date=q["date"], items=q["items"])
                for q in self.store.questionnaires(patient_id)]

    # ------------------------------------------------------------------ clinician side
    def _require_care_team(self, patient_id: str):
        self._patient(patient_id)
        if not self.store.has_consent(patient_id, "care_team"):
            raise ConsentRequired("the patient has not consented to share with the care team")

    def bundle(self, patient_id: str) -> PatientBundle:
        """The patient's stored check-ins as a Phase 1 bundle. Text is marked shared only where the
        patient shared it; the State Card shows the LLM only those texts."""
        entries = [Entry(entry_id=f"m{m['id']}", timestamp=m["created_at"], text=m["text"],
                         audio_path=m["audio_path"] if m["modality"] == "voice" else None,
                         shared_with_clinician=bool(m["shared"]))
                   for m in self.store.messages(patient_id, role="user")]
        return PatientBundle(patient_ref=patient_id, entries=entries, questionnaires=self._responses(patient_id))

    def state_card(self, patient_id: str, use_llm: bool = True) -> dict:
        self._require_care_team(patient_id)
        bundle = self.bundle(patient_id)
        card, extra = self.orch.build_card(bundle, use_llm=use_llm)
        self.store.audit("state_card", patient_id=patient_id, model=card.llm_model,
                         prompt=str(extra.get("digest")), output=card.narrative.model_dump_json(),
                         guardrails=card.guardrail_notes, attempts=card.llm_attempts, risk_level=card.risk.level)
        return {"card": card, "markdown": card_to_markdown(card)}

    def clinician_messages(self, patient_id: str) -> list[dict]:
        self._require_care_team(patient_id)
        return [{k: m[k] for k in ("id", "text", "modality", "language", "risk_level", "created_at")}
                for m in self.store.messages(patient_id, role="user", for_clinician=True) if m["shared"]]

    def dashboard(self, acc: dict | None = None) -> list[dict]:
        """Patients the clinician may see (all of them for a superuser, or with acc=None inside the server)."""
        rows = []
        for p in (self.visible_patients(acc) if acc else self.store.patients()):
            pid = p["id"]
            consents = self.store.consents(pid)
            alerts = self.store.alerts("open", pid)
            doc = self._linked_clinician(p)
            row = {"id": pid, "display_name": p["display_name"], "consents": consents, "open_alerts": len(alerts),
                   "highest_open_alert": alerts[0]["level"] if alerts else "none",
                   "clinician": {"id": doc["id"], "name": doc["name"] or doc["email"]} if doc else None}
            if consents["care_team"]:
                tr = trends(self._responses(pid))
                row.update({k: {"latest": t.latest_total, "severity": t.latest_severity, "change": t.change,
                                "direction": t.direction} for k, t in tr.items()})
                row["usage"] = self.store.usage(pid)
            rows.append(row)
        return sorted(rows, key=lambda r: (-LEVEL_ORDER.get(r["highest_open_alert"], 0), r["id"]))

    def alerts(self, status: str | None = "open", acc: dict | None = None) -> list[dict]:
        visible = {p["id"] for p in self.visible_patients(acc)} if acc else None
        return [{**a, "recommended_action": ACTIONS.get(a["level"], "")} for a in self.store.alerts(status)
                if visible is None or a["patient_id"] in visible]

    def alert_patient(self, alert_id: int) -> str | None:
        rows = [a for a in self.store.alerts(None) if a["id"] == alert_id]
        return rows[0]["patient_id"] if rows else None

    # ------------------------------------------------------------------ the patient's own summary card
    def patient_summary(self, patient_id: str) -> dict:
        """A friendly version of the State Card for the patient: no risk labels or clinical wording,
        just their scores over time, what came up, what helped, and where to get help."""
        self._patient(patient_id)
        card, _ = self.orch.build_card(self.bundle(patient_id), use_llm=False)
        cues = card.cues
        scores = []
        for ins, t in card.questionnaires.items():
            scores.append({
                "instrument": ins, "label": {"PHQ-9": "Mood (PHQ-9)", "GAD-7": "Worry (GAD-7)"}[ins],
                "latest": t.latest_total, "first": t.first_total, "change": t.change, "band": t.latest_severity,
                "max": 27 if ins == "PHQ-9" else 21, "direction": t.direction,
                "history": [{"date": h.date.isoformat(), "total": h.total} for h in t.history],
                "note": {"improved": "Lower is better: your score has come down. Well done for keeping at it.",
                         "worsened": "Your score has gone up. It may help to talk about this at your next visit.",
                         "stable": "Your score has stayed about the same.",
                         "baseline_only": "This is your first check-in; we'll show the trend after the next one."}
                        .get(t.direction, "")})
        top = lambda d, labels, n=4: [labels.get(k, k.replace("_", " ")) for k, _ in  # noqa: E731
                                      sorted(d.items(), key=lambda kv: -kv[1])[:n]]
        pauses = self.store.mind_pauses(patient_id)
        tough = card.risk.level in ("high", "urgent")
        return {
            "period": {"start": cues.period_start, "end": cues.period_end, "checkins": cues.n_entries},
            "scores": scores,
            "came_up": top(cues.stressors, STRESSOR_LABELS),
            "helped": top(cues.coping_helpful, COPING_LABELS),
            "sleep_mentions": cues.phq_domains.get("sleep", 0),  # check-ins that mentioned sleep problems
            "mind_pauses": {"started": len(pauses), "completed": sum(1 for p in pauses if p["completed"])},
            "clinician": self.profile(patient_id)["clinician_name"],
            "support": ("Some of what you shared recently sounded very hard. Your care team has been told and will "
                        "reach out. You can call Tele-MANAS 14416 (free, 24x7) any time, or 112 in an emergency."
                        if tough else None),
            "helplines": HELPLINES,
        }

    # ------------------------------------------------------------------ superuser: how is each clinician doing?
    def clinician_report(self) -> list[dict]:
        """Per clinician: caseload, engagement, score change of their patients, and how fast alerts were picked up.

        Outcome numbers depend on how unwell each caseload is to begin with, so they start a conversation; they
        are not a verdict on a clinician."""
        now_ = datetime.now(timezone.utc)
        patients = self.store.patients()
        all_alerts = self.store.alerts(None)
        out = []
        for doc in self.store.clinicians():
            if doc["is_superuser"] or not doc["verified_at"]:
                continue
            mine = [p for p in patients if str(p["clinician_id"] or "") == str(doc["id"])]
            ids = {p["id"] for p in mine}
            consenting = [p for p in mine if self.store.has_consent(p["id"], "care_team")]
            active = 0
            for p in mine:
                last = self.store.last_message_at(p["id"])
                if last and datetime.fromisoformat(last) > now_ - timedelta(days=14):
                    active += 1
            outcomes = {}
            for ins in ("PHQ-9", "GAD-7"):
                changes = []
                for p in consenting:
                    t = trends(self._responses(p["id"])).get(ins)
                    if t and t.n_administrations > 1:
                        changes.append(t.change)
                thr = MEANINGFUL_CHANGE[ins]
                outcomes[ins] = {"patients_with_2_scores": len(changes),
                                 "mean_change": round(sum(changes) / len(changes), 1) if changes else None,
                                 "improved": sum(c <= -thr for c in changes), "worsened": sum(c >= thr for c in changes)}
            alerts = [a for a in all_alerts if a["patient_id"] in ids]
            ack_hours, late, open_serious = [], [], 0
            for a in alerts:
                created = datetime.fromisoformat(a["created_at"])
                acked = datetime.fromisoformat(a["acknowledged_at"]) if a.get("acknowledged_at") else None
                target = ACK_TARGET_HOURS.get(a["level"])
                waited = ((acked or now_) - created).total_seconds() / 3600
                if acked:
                    ack_hours.append(waited)
                if a["status"] == "open" and a["level"] in ("high", "urgent"):
                    open_serious += 1
                if target is not None and waited > target:
                    late.append(a["level"])
            ack_hours.sort()
            median = round(ack_hours[len(ack_hours) // 2], 1) if ack_hours else None
            reasons = []
            if late.count("urgent"):
                reasons.append(f"{late.count('urgent')} urgent alert(s) not picked up within {ACK_TARGET_HOURS['urgent']} h")
            if late.count("high"):
                reasons.append(f"{late.count('high')} high alert(s) not picked up within {ACK_TARGET_HOURS['high']} h")
            phq = outcomes["PHQ-9"]
            if phq["patients_with_2_scores"] >= 3 and phq["worsened"] > phq["improved"]:
                reasons.append("more patients' PHQ-9 got clearly worse than clearly better")
            if mine and active == 0:
                reasons.append("no patient used MitraMind in the last 14 days")
            if not mine:
                status = "no patients yet"
            elif late.count("urgent") or late.count("high") or open_serious:
                status = "needs attention"
            elif reasons:
                status = "watch"
            else:
                status = "on track"
            out.append({
                "clinician": self.clinician_profile(doc), "last_login_at": doc["last_login_at"], "status": status,
                "reasons": reasons or (["alerts picked up on time and no warning signs"] if mine else []),
                "patients": len(mine), "sharing_with_care_team": len(consenting), "active_last_14_days": active,
                "outcomes": outcomes,
                "alerts": {"total": len(alerts), "open": sum(a["status"] == "open" for a in alerts),
                           "open_high_or_urgent": open_serious, "median_hours_to_acknowledge": median,
                           "late": len(late)},
                "summaries_rated": self.store.feedback_count(f"clinician:{doc['id']}"),
            })
        order = {"needs attention": 0, "watch": 1, "on track": 2, "no patients yet": 3}
        return sorted(out, key=lambda r: (order[r["status"]], -(r["patients"])))

    def overview(self) -> dict:
        patients = self.store.patients()
        open_alerts = self.store.alerts("open")
        docs = [d for d in self.store.clinicians() if d["verified_at"] and not d["is_superuser"]]
        return {"patients": len(patients), "clinicians": len(docs),
                "unassigned_patients": sum(1 for p in patients if not self._linked_clinician(p)),
                "open_alerts": {lvl: sum(a["level"] == lvl for a in open_alerts) for lvl in ("urgent", "high", "watch")},
                "clinician_choices": [{"id": d["id"], "name": d["name"] or d["email"]} for d in docs]}
