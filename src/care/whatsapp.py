"""WhatsApp Business (Meta Cloud API) channel: the same companion and check-ins over WhatsApp.

Setup (Meta developer console -> WhatsApp -> Configuration):
  callback URL   https://<your-host>/webhooks/whatsapp
  env            WHATSAPP_VERIFY_TOKEN (any secret you also type in the console),
                 WHATSAPP_APP_SECRET (to verify X-Hub-Signature-256; required outside local testing),
                 WHATSAPP_TOKEN + WHATSAPP_PHONE_NUMBER_ID (to send replies; without them replies are
                 only logged: "dry run").
Patients are matched by phone number, stored only as a SHA-256 hash (clinic enrols them with the
number). Unknown numbers get a fixed "ask your clinic" reply and nothing is stored.

Commands a patient can send: "PHQ" / "GAD" (start a check-in, answer each item with 0-3), "stop"
(cancel a check-in), "share" (share your previous message with your care team). Anything else is a
companion message; voice notes are transcribed like app voice messages.
Written against the Cloud API message format; not yet tested with a real Meta account.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone

import requests

from . import instruments
from .service import CareService

log = logging.getLogger("mitramind.whatsapp")
GRAPH = f"https://graph.facebook.com/{os.environ.get('WHATSAPP_GRAPH_VERSION', 'v21.0')}"
NOT_ENROLLED = ("Hi! This number is not registered with MitraMind. Please ask your clinic to enrol you. "
                "If you are in crisis, call Tele-MANAS 14416 (free, 24x7) or 112.")
OPTIONS_TEXT = "Reply 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day."


def verify_signature(body: bytes, header: str | None) -> bool:
    secret = os.environ.get("WHATSAPP_APP_SECRET")
    if not secret:
        return os.environ.get("MITRAMIND_ENV", "dev") == "dev"  # unsigned webhooks only in local development
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1])


class WhatsAppSender:
    def __init__(self):
        self.token = os.environ.get("WHATSAPP_TOKEN")
        self.phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
        self.outbox: list[dict] = []  # dry-run record (also handy in tests)

    def send(self, to: str, text: str) -> None:
        self.outbox.append({"to": to, "text": text, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        if not (self.token and self.phone_id):
            log.info("dry run, WhatsApp reply not sent: %s", text[:80])
            return
        r = requests.post(f"{GRAPH}/{self.phone_id}/messages", timeout=30,
                          headers={"Authorization": f"Bearer {self.token}"},
                          json={"messaging_product": "whatsapp", "to": to, "type": "text",
                                "text": {"body": text[:4096]}})
        if r.status_code >= 300:
            log.error("WhatsApp send failed %s: %s", r.status_code, r.text[:300])

    def download_media(self, media_id: str) -> tuple[bytes, str]:
        if not self.token:
            raise RuntimeError("WHATSAPP_TOKEN is needed to download voice notes")
        h = {"Authorization": f"Bearer {self.token}"}
        meta = requests.get(f"{GRAPH}/{media_id}", headers=h, timeout=30)
        meta.raise_for_status()
        info = meta.json()
        audio = requests.get(info["url"], headers=h, timeout=60)
        audio.raise_for_status()
        return audio.content, info.get("mime_type", "audio/ogg")


class WhatsAppChannel:
    def __init__(self, svc: CareService, sender: WhatsAppSender | None = None):
        self.svc, self.sender = svc, sender or WhatsAppSender()

    @staticmethod
    def messages(payload: dict) -> list[dict]:
        out = []
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                out += change.get("value", {}).get("messages", []) or []
        return out

    def handle(self, payload: dict) -> int:
        n = 0
        for msg in self.messages(payload):
            if not self.svc.store.first_time(msg.get("id", ""), "whatsapp"):
                continue
            try:
                self._one(msg)
                n += 1
            except Exception:  # never let one bad message stop the batch; Meta would retry forever
                log.exception("WhatsApp message %s failed", msg.get("id"))
                self.sender.send(msg.get("from", ""), "Sorry, something went wrong. Please try again. "
                                 "If you are in crisis, call Tele-MANAS 14416 or 112.")
        return n

    def _one(self, msg: dict) -> None:
        phone = msg["from"]
        patient = self.svc.store.patient_by_phone(phone)
        if patient is None:
            self.sender.send(phone, NOT_ENROLLED)
            return
        pid = patient["id"]
        if not self.svc.store.has_consent(pid, "companion"):
            self.sender.send(phone, "Your MitraMind companion is switched off. You can turn it on in the app or "
                                    "ask your clinic. In a crisis call Tele-MANAS 14416 or 112.")
            return
        if msg.get("type") == "audio":
            data, mime = self.sender.download_media(msg["audio"]["id"])
            res = self.svc.voice(pid, data, mime, channel="whatsapp")
            self.sender.send(phone, res["reply"])
            return
        text = (msg.get("text") or {}).get("body", "").strip()
        if not text:
            self.sender.send(phone, "I can read text and listen to voice notes.")
            return
        if self._questionnaire(pid, phone, text):
            return
        if text.lower() == "share":
            last = self.svc.store.messages(pid, role="user", limit=1)
            if last:
                self.svc.share(pid, last[0]["id"], True)
            self.sender.send(phone, "Done, your care team can now read your previous message.")
            return
        res = self.svc.chat(pid, text, channel="whatsapp")
        self.sender.send(phone, res["reply"])

    def _questionnaire(self, pid: str, phone: str, text: str) -> bool:
        """PHQ-9 / GAD-7 one item at a time. Returns True if the message was part of a check-in."""
        low = text.lower()
        session = self.svc.store.q_session(pid)
        start = {"phq": "PHQ-9", "phq9": "PHQ-9", "phq-9": "PHQ-9", "gad": "GAD-7", "gad7": "GAD-7",
                 "gad-7": "GAD-7"}.get(low)
        if start:
            self.svc.store.set_q_session(pid, start, [], "whatsapp")
            self.sender.send(phone, f"{start} check-in. {instruments.STEM}\n{OPTIONS_TEXT}\n\n"
                                    f"1. {instruments.ITEMS[start][0]}")
            return True
        if session is None:
            return False
        if low in ("stop", "cancel"):
            self.svc.store.end_q_session(pid)
            self.sender.send(phone, "Check-in cancelled. You can start again any time with PHQ or GAD.")
            return True
        if low not in ("0", "1", "2", "3"):
            self.sender.send(phone, f"Please reply with a number from 0 to 3 (or 'stop'). {OPTIONS_TEXT}")
            return True
        answers, inst = session["answers"] + [int(low)], session["instrument"]
        items = instruments.ITEMS[inst]
        if len(answers) < len(items):
            self.svc.store.set_q_session(pid, inst, answers, "whatsapp")
            self.sender.send(phone, f"{len(answers) + 1}. {items[len(answers)]}")
            return True
        self.svc.store.end_q_session(pid)
        res = self.svc.submit_questionnaire(pid, inst, answers, channel="whatsapp")
        msg = f"Thank you. Your {inst} score is {res['score']['total']} ({res['score']['severity']} range)."
        if res["helplines"]:
            msg += (" Some answers suggest you are going through a very hard time. Your care team has been told. "
                    "You can call Tele-MANAS 14416 (free, 24x7) any time, or 112 in an emergency.")
        self.sender.send(phone, msg)
        return True
