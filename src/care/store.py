"""SQLite storage for the companion, clinician dashboard and pilot data (stdlib only).

One file (MITRAMIND_DB, default var/mitramind.db). The schema maps 1:1 to Postgres tables for
production; only `_connect()` would change.

Privacy rules enforced here, not in the UI:
- message text is returned to clinician-facing callers only if the patient shared it
  (`messages.shared = 1`); see `messages(..., for_clinician=True)`.
- consent is append-only: the latest row per (patient, purpose) is the current state, and
  withdrawals are kept for the audit trail (DPDP Act 2023).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "var" / "mitramind.db"

# What each consent purpose allows. The companion needs `companion`; the care team sees usage counts
# and shared reflections with `care_team`; `pilot_research` puts de-identified cue features and scores
# in the Phase 2 dataset; `research_text` additionally allows the patient's shared text in it.
CONSENT_PURPOSES = {
    "companion": "Use the MitraMind chat and voice companion. If I write or say something that suggests I may "
                 "be in danger, MitraMind will alert my care team",
    "care_team": "Share questionnaire scores, alerts, usage and shared reflections with my care team",
    "pilot_research": "Use my de-identified scores and cue features (no text) to improve MitraMind",
    "research_text": "Also use the text of reflections I chose to share, de-identified, for research",
}
CONSENT_VERSION = "2026-10-v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY, display_name TEXT, preferred_language TEXT DEFAULT 'auto',
    clinician_id TEXT, phone_hash TEXT UNIQUE, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS consents (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL REFERENCES patients(id),
    purpose TEXT NOT NULL, granted INTEGER NOT NULL, version TEXT NOT NULL, channel TEXT,
    recorded_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL REFERENCES patients(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')), text TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT 'text', channel TEXT NOT NULL DEFAULT 'app', language TEXT,
    shared INTEGER NOT NULL DEFAULT 0, audio_path TEXT, risk_level TEXT, analysis_json TEXT,
    reply_source TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS questionnaires (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL REFERENCES patients(id),
    instrument TEXT NOT NULL, date TEXT NOT NULL, items_json TEXT NOT NULL, total INTEGER NOT NULL,
    channel TEXT NOT NULL DEFAULT 'app', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL REFERENCES patients(id),
    level TEXT NOT NULL, reasons_json TEXT NOT NULL, source TEXT NOT NULL, source_id INTEGER,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acknowledged', 'resolved')),
    handled_by TEXT, handled_at TEXT, note TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, patient_id TEXT, kind TEXT NOT NULL,
    model TEXT, input_sha256 TEXT, output TEXT, guardrail_json TEXT, attempts INTEGER, risk_level TEXT);
CREATE TABLE IF NOT EXISTS mind_pause (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL REFERENCES patients(id),
    exercise TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clinician_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, clinician_id TEXT,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5), accurate INTEGER, minutes_saved REAL, comment TEXT,
    card_json TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_messages_patient ON messages(patient_id, created_at);
CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts(status, created_at);
"""

# Columns added after the first release; applied to existing databases on startup.
MIGRATIONS = [
    ("alerts", "outcome TEXT"),  # clinician's label when resolving: confirmed_risk / false_alarm / unclear
]
ALERT_OUTCOMES = ("confirmed_risk", "false_alarm", "unclear")
LEVEL_ORDER = {"none": 0, "watch": 1, "high": 2, "urgent": 3}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("MITRAMIND_DB") or DEFAULT_DB)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        for table, column in MIGRATIONS:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass  # column already there

    # ------------------------------------------------------------------ helpers
    def _exec(self, sql: str, args=()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _all(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def _one(self, sql: str, args=()) -> dict | None:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ patients and consent
    def create_patient(self, patient_id: str, display_name: str | None = None, preferred_language: str = "auto",
                       clinician_id: str | None = None, phone: str | None = None) -> dict:
        self._exec("INSERT INTO patients (id, display_name, preferred_language, clinician_id, phone_hash, created_at) "
                   "VALUES (?, ?, ?, ?, ?, ?)",
                   (patient_id, display_name, preferred_language, clinician_id, sha256(phone) if phone else None, now()))
        return self.patient(patient_id)

    def patient(self, patient_id: str) -> dict | None:
        return self._one("SELECT * FROM patients WHERE id = ?", (patient_id,))

    def patient_by_phone(self, phone: str) -> dict | None:
        return self._one("SELECT * FROM patients WHERE phone_hash = ?", (sha256(phone),))

    def patients(self) -> list[dict]:
        return self._all("SELECT * FROM patients ORDER BY created_at")

    def record_consent(self, patient_id: str, purpose: str, granted: bool, channel: str = "app") -> None:
        if purpose not in CONSENT_PURPOSES:
            raise ValueError(f"unknown consent purpose {purpose}")
        self._exec("INSERT INTO consents (patient_id, purpose, granted, version, channel, recorded_at) "
                   "VALUES (?, ?, ?, ?, ?, ?)", (patient_id, purpose, int(granted), CONSENT_VERSION, channel, now()))

    def consents(self, patient_id: str) -> dict[str, bool]:
        rows = self._all("SELECT purpose, granted FROM consents WHERE patient_id = ? ORDER BY id", (patient_id,))
        current = {p: False for p in CONSENT_PURPOSES}
        for r in rows:
            current[r["purpose"]] = bool(r["granted"])
        return current

    def has_consent(self, patient_id: str, purpose: str) -> bool:
        return self.consents(patient_id).get(purpose, False)

    # ------------------------------------------------------------------ messages
    def add_message(self, patient_id: str, role: str, text: str, *, modality: str = "text", channel: str = "app",
                    language: str | None = None, shared: bool = False, audio_path: str | None = None,
                    risk_level: str | None = None, analysis: dict | None = None,
                    reply_source: str | None = None, created_at: str | None = None) -> int:
        cur = self._exec(
            "INSERT INTO messages (patient_id, role, text, modality, channel, language, shared, audio_path, "
            "risk_level, analysis_json, reply_source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (patient_id, role, text, modality, channel, language, int(shared), audio_path, risk_level,
             json.dumps(analysis, default=str) if analysis is not None else None, reply_source, created_at or now()))
        return cur.lastrowid

    def set_shared(self, patient_id: str, message_id: int, shared: bool) -> bool:
        cur = self._exec("UPDATE messages SET shared = ? WHERE id = ? AND patient_id = ? AND role = 'user'",
                         (int(shared), message_id, patient_id))
        return cur.rowcount == 1

    def messages(self, patient_id: str, *, role: str | None = None, limit: int | None = None,
                 for_clinician: bool = False) -> list[dict]:
        sql, args = "SELECT * FROM messages WHERE patient_id = ?", [patient_id]
        if role:
            sql += " AND role = ?"
            args.append(role)
        sql += " ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        rows = list(reversed(self._all(sql, args)))
        for r in rows:
            r["analysis"] = json.loads(r.pop("analysis_json")) if r.get("analysis_json") else None
            if for_clinician and not r["shared"]:
                r["text"] = None
                r["audio_path"] = None
        return rows

    # ------------------------------------------------------------------ questionnaires
    def add_questionnaire(self, patient_id: str, instrument: str, date: str, items: list[int],
                          channel: str = "app", created_at: str | None = None) -> int:
        cur = self._exec("INSERT INTO questionnaires (patient_id, instrument, date, items_json, total, channel, "
                         "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (patient_id, instrument, date, json.dumps(items), sum(items), channel, created_at or now()))
        return cur.lastrowid

    def questionnaires(self, patient_id: str) -> list[dict]:
        rows = self._all("SELECT * FROM questionnaires WHERE patient_id = ? ORDER BY date, id", (patient_id,))
        for r in rows:
            r["items"] = json.loads(r.pop("items_json"))
        return rows

    # ------------------------------------------------------------------ alerts
    def add_alert(self, patient_id: str, level: str, reasons: list[str], source: str,
                  source_id: int | None = None, created_at: str | None = None) -> int:
        cur = self._exec("INSERT INTO alerts (patient_id, level, reasons_json, source, source_id, created_at) "
                         "VALUES (?, ?, ?, ?, ?, ?)",
                         (patient_id, level, json.dumps(reasons), source, source_id, created_at or now()))
        return cur.lastrowid

    def alerts(self, status: str | None = "open", patient_id: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM alerts WHERE 1 = 1", []
        if status:
            sql += " AND status = ?"
            args.append(status)
        if patient_id:
            sql += " AND patient_id = ?"
            args.append(patient_id)
        rows = self._all(sql + " ORDER BY id DESC", args)
        for r in rows:
            r["reasons"] = json.loads(r.pop("reasons_json"))
        return sorted(rows, key=lambda r: -LEVEL_ORDER.get(r["level"], 0))

    def update_alert(self, alert_id: int, status: str, handled_by: str | None = None, note: str | None = None,
                     outcome: str | None = None) -> bool:
        if outcome is not None and outcome not in ALERT_OUTCOMES:
            raise ValueError(f"outcome must be one of {ALERT_OUTCOMES}")
        cur = self._exec("UPDATE alerts SET status = ?, handled_by = ?, handled_at = ?, note = ?, "
                         "outcome = COALESCE(?, outcome) WHERE id = ?",
                         (status, handled_by, now(), note, outcome, alert_id))
        return cur.rowcount == 1

    # ------------------------------------------------------------------ audit, Mind Pause, feedback
    def audit(self, kind: str, *, patient_id: str | None = None, model: str | None = None, prompt: str = "",
              output: str = "", guardrails=None, attempts: int | None = None, risk_level: str | None = None) -> None:
        """Every AI-generated (or fixed safety) reply shown to anyone is logged here."""
        self._exec("INSERT INTO audit_log (created_at, patient_id, kind, model, input_sha256, output, guardrail_json, "
                   "attempts, risk_level) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (now(), patient_id, kind, model, sha256(prompt), output, json.dumps(guardrails, default=str),
                    attempts, risk_level))

    def audit_log(self, patient_id: str | None = None, limit: int = 200) -> list[dict]:
        if patient_id:
            return self._all("SELECT * FROM audit_log WHERE patient_id = ? ORDER BY id DESC LIMIT ?", (patient_id, limit))
        return self._all("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))

    def add_mind_pause(self, patient_id: str, exercise: str, completed: bool, created_at: str | None = None) -> int:
        return self._exec("INSERT INTO mind_pause (patient_id, exercise, completed, created_at) VALUES (?, ?, ?, ?)",
                          (patient_id, exercise, int(completed), created_at or now())).lastrowid

    def mind_pauses(self, patient_id: str) -> list[dict]:
        return self._all("SELECT * FROM mind_pause WHERE patient_id = ? ORDER BY id", (patient_id,))

    def add_feedback(self, patient_id: str, clinician_id: str | None, rating: int | None, accurate: bool | None,
                     minutes_saved: float | None, comment: str | None, card: dict | None) -> int:
        return self._exec("INSERT INTO clinician_feedback (patient_id, clinician_id, rating, accurate, minutes_saved, "
                          "comment, card_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (patient_id, clinician_id, rating, None if accurate is None else int(accurate), minutes_saved,
                           comment, json.dumps(card, default=str) if card else None, now())).lastrowid

    def feedback(self) -> list[dict]:
        return self._all("SELECT * FROM clinician_feedback ORDER BY id")

    # ------------------------------------------------------------------ usage (what the clinician may see)
    def usage(self, patient_id: str) -> dict:
        row = self._one(
            "SELECT COUNT(*) AS user_messages, SUM(shared) AS shared, SUM(modality = 'voice') AS voice, "
            "MAX(created_at) AS last_message FROM messages WHERE patient_id = ? AND role = 'user'", (patient_id,))
        pauses = self._one("SELECT COUNT(*) AS n, SUM(completed) AS done FROM mind_pause WHERE patient_id = ?",
                           (patient_id,))
        return {"user_messages": row["user_messages"] or 0, "shared_reflections": row["shared"] or 0,
                "voice_messages": row["voice"] or 0, "last_message": row["last_message"],
                "mind_pause_started": pauses["n"] or 0, "mind_pause_completed": pauses["done"] or 0}
