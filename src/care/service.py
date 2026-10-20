"""CareService: the one place that ties storage, Phase 1 analysis, the companion and triage together.

Used by the API, the WhatsApp webhook and the tests, so all channels follow the same rules:
- no companion use without `companion` consent;
- every patient message runs through the Phase 1 safety layer before any reply;
- any risk above "none" (message or questionnaire) opens an alert for the care team;
- clinicians see usage counts, scores, alerts and only the messages the patient shared,
  and nothing at all without `care_team` consent (safety alerts excepted, see store.CONSENT_PURPOSES).
"""
from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timezone

from phase1.crisis import ACTIONS, HELPLINES, combine
from phase1.questionnaires import score, trends
from phase1.render import card_to_markdown
from phase1.schema import Entry, PatientBundle, QuestionnaireResponse

from .companion import Companion
from .store import LEVEL_ORDER, Store


class ConsentRequired(Exception):
    pass


class NotFound(Exception):
    pass


def _jsonable_analysis(a: dict) -> dict:
    return {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in a.items() if k not in ("text",)}


class CareService:
    def __init__(self, store: Store, orchestrator, chat_backend=None):
        self.store, self.orch = store, orchestrator
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
        return {**{k: p[k] for k in ("id", "display_name", "preferred_language", "clinician_id", "created_at")},
                "consents": self.store.consents(patient_id)}

    def _patient(self, patient_id: str) -> dict:
        p = self.store.patient(patient_id)
        if p is None:
            raise NotFound(f"no patient {patient_id}")
        return p

    # ------------------------------------------------------------------ companion
    def chat(self, patient_id: str, text: str, *, share: bool = False, modality: str = "text", channel: str = "app",
             audio_path: str | None = None, language: str = "auto") -> dict:
        self._patient(patient_id)
        if not self.store.has_consent(patient_id, "companion"):
            raise ConsentRequired("the patient has not consented to the companion")
        # Language is detected per message (patients switch scripts mid-conversation), not taken from the profile.
        entry = Entry(entry_id="live", timestamp=datetime.now(timezone.utc), text=text,
                      audio_path=audio_path if modality == "voice" else None, language=language,
                      shared_with_clinician=share)
        analysis = self.orch.analyze_entry(entry)
        risk = analysis["risk"]
        history = self.store.messages(patient_id, limit=16)
        msg_id = self.store.add_message(patient_id, "user", text, modality=modality, channel=channel,
                                        language=analysis["language"], shared=share, audio_path=audio_path,
                                        risk_level=risk.level, analysis=_jsonable_analysis(analysis))
        out = self.companion.respond(history, text, analysis)
        reply_id = self.store.add_message(patient_id, "assistant", out["reply"], modality=modality, channel=channel,
                                          language=analysis["language"], reply_source=out["source"])
        self.store.audit("crisis_reply" if out["source"] == "crisis_protocol" else "chat_reply", patient_id=patient_id,
                         model=out["model"], prompt=out["prompt"], output=out["reply"], guardrails=out["log"],
                         attempts=out["attempts"], risk_level=risk.level)
        alert_id = None
        if risk.level != "none":
            alert_id = self.store.add_alert(patient_id, risk.level, risk.reasons or ["crisis language"], "message", msg_id)
        return {"message_id": msg_id, "reply_id": reply_id, "reply": out["reply"], "reply_source": out["source"],
                "mind_pause": out["mind_pause"], "language": analysis["language"], "risk": risk.model_dump(),
                "escalate": risk.level in ("high", "urgent"), "helplines": HELPLINES if risk.level != "none" else [],
                "alert_id": alert_id, "models_used": analysis["models_used"], "emotion": analysis["emotion"],
                "llm_attempts": out["attempts"]}

    def history(self, patient_id: str, limit: int = 50) -> list[dict]:
        self._patient(patient_id)
        return [{k: m[k] for k in ("id", "role", "text", "modality", "language", "shared", "risk_level",
                                   "reply_source", "created_at")} for m in self.store.messages(patient_id, limit=limit)]

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
        phq = tr.get("PHQ-9")
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

    def dashboard(self) -> list[dict]:
        rows = []
        for p in self.store.patients():
            pid = p["id"]
            consents = self.store.consents(pid)
            alerts = self.store.alerts("open", pid)
            row = {"id": pid, "display_name": p["display_name"], "consents": consents, "open_alerts": len(alerts),
                   "highest_open_alert": alerts[0]["level"] if alerts else "none"}
            if consents["care_team"]:
                tr = trends(self._responses(pid))
                row.update({k: {"latest": t.latest_total, "severity": t.latest_severity, "change": t.change,
                                "direction": t.direction} for k, t in tr.items()})
                row["usage"] = self.store.usage(pid)
            rows.append(row)
        return sorted(rows, key=lambda r: (-LEVEL_ORDER.get(r["highest_open_alert"], 0), r["id"]))

    def alerts(self, status: str | None = "open") -> list[dict]:
        out = []
        for a in self.store.alerts(status):
            out.append({**a, "recommended_action": ACTIONS.get(a["level"], "")})
        return out
