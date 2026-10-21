"""SQLite storage for the companion, clinician dashboard and pilot data (stdlib only).

One file (MITRAMIND_DB, default var/mitramind.db). The schema maps 1:1 to Postgres tables for
production; only `_connect()` would change.

Privacy rules enforced here, not in the UI:
- message text is returned to clinician-facing callers only if the patient shared it
  (`messages.shared = 1`); see `messages(..., for_clinician=True)`.
- consent is append-only: the latest row per (patient, purpose) is the current state, and
  withdrawals are kept for the audit trail (DPDP Act 2023).
- a patient's login email lives only in `accounts`; it is never copied into messages, prompts, alerts or
  exports, which all key on the pseudonymous patient id. Passwords are stored as scrypt hashes and
  session tokens as SHA-256 hashes, so a copied database file cannot be used to log in.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache
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
CREATE TABLE IF NOT EXISTS questionnaire_sessions (
    patient_id TEXT PRIMARY KEY REFERENCES patients(id), instrument TEXT NOT NULL, answers_json TEXT NOT NULL,
    channel TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS inbound_events (
    event_id TEXT PRIMARY KEY, channel TEXT NOT NULL, received_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
    patient_id TEXT NOT NULL UNIQUE REFERENCES patients(id), created_at TEXT NOT NULL, last_login_at TEXT);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, account_id INTEGER NOT NULL REFERENCES accounts(id),
    created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clinicians (
    id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, name TEXT,
    verified_at TEXT, created_at TEXT NOT NULL, last_login_at TEXT);
CREATE TABLE IF NOT EXISTS clinician_sessions (
    token_hash TEXT PRIMARY KEY, clinician_id INTEGER NOT NULL REFERENCES clinicians(id),
    created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS email_codes (
    email TEXT PRIMARY KEY, code_hash TEXT NOT NULL, expires_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, sent_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_messages_patient ON messages(patient_id, created_at);
CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts(status, created_at);
"""

# Columns added after the first release; applied to existing databases on startup.
MIGRATIONS = [
    ("alerts", "outcome TEXT"),  # clinician's label when resolving: confirmed_risk / false_alarm / unclear
    ("messages", "image_path TEXT"),  # photo the patient attached to a chat message
    ("clinicians", "is_superuser INTEGER NOT NULL DEFAULT 0"),  # sees every patient and every clinician's report
    ("clinicians", "care_code TEXT"),  # short code a patient enters to link to this clinician
    ("alerts", "acknowledged_at TEXT"),  # first time anyone acknowledged or resolved it (response-time metric)
]
POST_MIGRATION = ["CREATE UNIQUE INDEX IF NOT EXISTS ix_clinicians_code ON clinicians(care_code)"]
ALERT_OUTCOMES = ("confirmed_risk", "false_alarm", "unclear")
LEVEL_ORDER = {"none": 0, "watch": 1, "high": 2, "urgent": 3}
SESSION_DAYS = 30
CLINICIAN_SESSION_HOURS = 12  # one working day; clinicians see many patients' data
CODE_MINUTES, CODE_MAX_ATTEMPTS = 10, 5
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(digest) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), digest)


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_password(secrets.token_hex(8))


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
        for sql in POST_MIGRATION:
            self._conn.execute(sql)
        self._conn.commit()

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

    # ------------------------------------------------------------------ patient accounts (email + password)
    def create_account(self, email: str, password: str, patient_id: str) -> dict:
        self._exec("INSERT INTO accounts (email, password_hash, patient_id, created_at) VALUES (?, ?, ?, ?)",
                   (email, hash_password(password), patient_id, now()))
        return self.account_by_email(email)

    def account_by_email(self, email: str) -> dict | None:
        return self._one("SELECT * FROM accounts WHERE email = ?", (email,))

    def account_for_patient(self, patient_id: str) -> dict | None:
        return self._one("SELECT * FROM accounts WHERE patient_id = ?", (patient_id,))

    def check_login(self, email: str, password: str) -> dict | None:
        acc = self.account_by_email(email)
        if acc is None:
            verify_password(password, _dummy_hash())  # same work as a real check, so timing does not reveal accounts
            return None
        if not verify_password(password, acc["password_hash"]):
            return None
        self._exec("UPDATE accounts SET last_login_at = ? WHERE id = ?", (now(), acc["id"]))
        return acc

    def new_session(self, account_id: int, days: int = SESSION_DAYS) -> str:
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(timespec="seconds")
        self._exec("INSERT INTO sessions (token_hash, account_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                   (sha256(token), account_id, now(), expires))
        return token

    def session_account(self, token: str) -> dict | None:
        return self._one("SELECT a.* FROM sessions s JOIN accounts a ON a.id = s.account_id "
                         "WHERE s.token_hash = ? AND s.expires_at > ?", (sha256(token), now()))

    def end_session(self, token: str) -> None:
        self._exec("DELETE FROM sessions WHERE token_hash = ?", (sha256(token),))

    # ------------------------------------------------------------------ clinician accounts (Gmail-verified)
    def create_clinician(self, email: str, password: str, name: str | None) -> dict:
        self._exec("INSERT INTO clinicians (email, password_hash, name, created_at) VALUES (?, ?, ?, ?)",
                   (email, hash_password(password), name, now()))
        return self.clinician_by_email(email)

    def reset_unverified_clinician(self, email: str, password: str, name: str | None) -> dict:
        """Signing up again before verifying replaces the password and name (the old code is replaced too)."""
        self._exec("UPDATE clinicians SET password_hash = ?, name = ? WHERE email = ? AND verified_at IS NULL",
                   (hash_password(password), name, email))
        return self.clinician_by_email(email)

    def clinician_by_email(self, email: str) -> dict | None:
        return self._one("SELECT * FROM clinicians WHERE email = ?", (email,))

    def check_clinician_login(self, email: str, password: str) -> dict | None:
        acc = self.clinician_by_email(email)
        if acc is None:
            verify_password(password, _dummy_hash())
            return None
        if not verify_password(password, acc["password_hash"]):
            return None
        return acc

    def mark_clinician_verified(self, email: str) -> None:
        self._exec("UPDATE clinicians SET verified_at = COALESCE(verified_at, ?) WHERE email = ?", (now(), email))

    def new_clinician_session(self, clinician_id: int, hours: int = CLINICIAN_SESSION_HOURS) -> str:
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat(timespec="seconds")
        self._exec("INSERT INTO clinician_sessions (token_hash, clinician_id, created_at, expires_at) "
                   "VALUES (?, ?, ?, ?)", (sha256(token), clinician_id, now(), expires))
        self._exec("UPDATE clinicians SET last_login_at = ? WHERE id = ?", (now(), clinician_id))
        return token

    def clinician_session(self, token: str) -> dict | None:
        return self._one("SELECT c.* FROM clinician_sessions s JOIN clinicians c ON c.id = s.clinician_id "
                         "WHERE s.token_hash = ? AND s.expires_at > ? AND c.verified_at IS NOT NULL",
                         (sha256(token), now()))

    def end_clinician_session(self, token: str) -> None:
        self._exec("DELETE FROM clinician_sessions WHERE token_hash = ?", (sha256(token),))

    def clinician_by_id(self, clinician_id: int) -> dict | None:
        return self._one("SELECT * FROM clinicians WHERE id = ?", (clinician_id,))

    def clinician_by_code(self, care_code: str) -> dict | None:
        return self._one("SELECT * FROM clinicians WHERE care_code = ? AND verified_at IS NOT NULL "
                         "AND is_superuser = 0", ((care_code or "").strip().upper(),))

    def clinicians(self) -> list[dict]:
        return self._all("SELECT * FROM clinicians ORDER BY created_at")

    def set_care_code(self, clinician_id: int, care_code: str) -> None:
        self._exec("UPDATE clinicians SET care_code = ? WHERE id = ?", (care_code, clinician_id))

    def upsert_superuser(self, email: str, password: str, name: str | None) -> dict:
        """Create a superuser, or turn an existing clinician account into one with this password."""
        if self.clinician_by_email(email):
            self._exec("UPDATE clinicians SET password_hash = ?, name = COALESCE(?, name), is_superuser = 1, "
                       "verified_at = COALESCE(verified_at, ?) WHERE email = ?", (hash_password(password), name, now(), email))
        else:
            self._exec("INSERT INTO clinicians (email, password_hash, name, verified_at, created_at, is_superuser) "
                       "VALUES (?, ?, ?, ?, ?, 1)", (email, hash_password(password), name, now(), now()))
        return self.clinician_by_email(email)

    def set_patient_clinician(self, patient_id: str, clinician_id: int | None) -> None:
        self._exec("UPDATE patients SET clinician_id = ? WHERE id = ?",
                   (None if clinician_id is None else str(clinician_id), patient_id))

    def feedback_count(self, clinician_ref: str) -> int:
        return self._one("SELECT COUNT(*) AS n FROM clinician_feedback WHERE clinician_id = ?", (clinician_ref,))["n"]

    def last_message_at(self, patient_id: str) -> str | None:
        row = self._one("SELECT MAX(created_at) AS t FROM messages WHERE patient_id = ? AND role = 'user'", (patient_id,))
        return row["t"] if row else None

    def set_email_code(self, email: str, code: str, minutes: int = CODE_MINUTES) -> None:
        expires = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat(timespec="seconds")
        self._exec("INSERT INTO email_codes (email, code_hash, expires_at, attempts, sent_at) VALUES (?, ?, ?, 0, ?) "
                   "ON CONFLICT(email) DO UPDATE SET code_hash = excluded.code_hash, expires_at = excluded.expires_at, "
                   "attempts = 0, sent_at = excluded.sent_at", (email, sha256(f"{email}:{code}"), expires, now()))

    def code_sent_at(self, email: str) -> str | None:
        row = self._one("SELECT sent_at FROM email_codes WHERE email = ?", (email,))
        return row["sent_at"] if row else None

    def check_email_code(self, email: str, code: str) -> str:
        """'ok' (and the code is used up), 'wrong', 'expired', 'too_many' or 'missing'."""
        row = self._one("SELECT * FROM email_codes WHERE email = ?", (email,))
        if row is None:
            return "missing"
        if row["expires_at"] <= now():
            return "expired"
        if row["attempts"] >= CODE_MAX_ATTEMPTS:
            return "too_many"
        if not hmac.compare_digest(row["code_hash"], sha256(f"{email}:{code}")):
            self._exec("UPDATE email_codes SET attempts = attempts + 1 WHERE email = ?", (email,))
            return "wrong"
        self._exec("DELETE FROM email_codes WHERE email = ?", (email,))
        return "ok"

    # ------------------------------------------------------------------ messages
    def add_message(self, patient_id: str, role: str, text: str, *, modality: str = "text", channel: str = "app",
                    language: str | None = None, shared: bool = False, audio_path: str | None = None,
                    risk_level: str | None = None, analysis: dict | None = None,
                    reply_source: str | None = None, created_at: str | None = None,
                    image_path: str | None = None) -> int:
        cur = self._exec(
            "INSERT INTO messages (patient_id, role, text, modality, channel, language, shared, audio_path, "
            "risk_level, analysis_json, reply_source, created_at, image_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (patient_id, role, text, modality, channel, language, int(shared), audio_path, risk_level,
             json.dumps(analysis, default=str) if analysis is not None else None, reply_source, created_at or now(),
             image_path))
        return cur.lastrowid

    def message_count(self, patient_id: str) -> int:
        return self._one("SELECT COUNT(*) AS n FROM messages WHERE patient_id = ?", (patient_id,))["n"]

    def messages_from(self, patient_id: str, offset: int) -> list[dict]:
        """The patient's messages in order, skipping the first `offset` (the companion's history window)."""
        rows = self._all("SELECT * FROM messages WHERE patient_id = ? ORDER BY id LIMIT -1 OFFSET ?",
                         (patient_id, offset))
        for r in rows:
            r["analysis"] = json.loads(r.pop("analysis_json")) if r.get("analysis_json") else None
        return rows

    def message(self, patient_id: str, message_id: int) -> dict | None:
        return self._one("SELECT * FROM messages WHERE id = ? AND patient_id = ?", (message_id, patient_id))

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
                r["image_path"] = None
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
                         "outcome = COALESCE(?, outcome), acknowledged_at = COALESCE(acknowledged_at, ?) WHERE id = ?",
                         (status, handled_by, now(), note, outcome, now(), alert_id))
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

    # ------------------------------------------------------------------ chat-channel questionnaires, dedup
    def q_session(self, patient_id: str) -> dict | None:
        row = self._one("SELECT * FROM questionnaire_sessions WHERE patient_id = ?", (patient_id,))
        if row:
            row["answers"] = json.loads(row.pop("answers_json"))
        return row

    def set_q_session(self, patient_id: str, instrument: str, answers: list[int], channel: str) -> None:
        self._exec("INSERT INTO questionnaire_sessions (patient_id, instrument, answers_json, channel, updated_at) "
                   "VALUES (?, ?, ?, ?, ?) ON CONFLICT(patient_id) DO UPDATE SET instrument = excluded.instrument, "
                   "answers_json = excluded.answers_json, channel = excluded.channel, updated_at = excluded.updated_at",
                   (patient_id, instrument, json.dumps(answers), channel, now()))

    def end_q_session(self, patient_id: str) -> None:
        self._exec("DELETE FROM questionnaire_sessions WHERE patient_id = ?", (patient_id,))

    def first_time(self, event_id: str, channel: str) -> bool:
        """True the first time an inbound event id is seen (webhooks are retried)."""
        try:
            self._exec("INSERT INTO inbound_events (event_id, channel, received_at) VALUES (?, ?, ?)",
                       (event_id, channel, now()))
            return True
        except sqlite3.IntegrityError:
            return False

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
