"""Email codes: clinician and patient sign-up verification, and password resets (stdlib smtplib only).

Settings can live in a `.env` file in the repository root (never committed; see .env.example).

Gmail setup (the sending account needs 2-step verification and an app password):
  MITRAMIND_SMTP_HOST=smtp.gmail.com  MITRAMIND_SMTP_PORT=587
  MITRAMIND_SMTP_USER=<sender>@gmail.com  MITRAMIND_SMTP_PASSWORD=<16-character app password>
  MITRAMIND_SMTP_FROM="MitraMind <sender@gmail.com>"   (optional, defaults to the user)
Without SMTP settings, and only while MITRAMIND_ENV is "dev" (the default on a developer machine), the code
is written to the server log and returned to the page so the flow can be tested locally. Any other
MITRAMIND_ENV refuses to sign clinicians up until SMTP is configured.
"""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

log = logging.getLogger("mitramind.mailer")


class MailNotConfigured(Exception):
    pass


def configured() -> bool:
    return bool(os.environ.get("MITRAMIND_SMTP_HOST") and os.environ.get("MITRAMIND_SMTP_USER"))


def dev_mode() -> bool:
    return os.environ.get("MITRAMIND_ENV", "dev") == "dev"


MESSAGES = {  # purpose -> (subject, what the code is for, what to do if it was not you)
    "verify-clinician": ("Your MitraMind clinician code", "confirm your clinician account",
                         "If you did not try to create a clinician account, ignore this email."),
    "verify-patient": ("Your MitraMind sign-up code", "confirm your MitraMind account",
                       "If you did not sign up for MitraMind, ignore this email."),
    "reset-patient": ("Reset your MitraMind password", "choose a new password",
                      "If you did not ask to reset your password, ignore this email; your password stays the same."),
    "reset-clinician": ("Reset your MitraMind clinician password", "choose a new password",
                        "If you did not ask to reset your password, ignore this email; your password stays the same."),
    "reset-superuser": ("Reset your MitraMind superuser password", "choose a new password",
                        "If you did not ask for this, tell your MitraMind administrator straight away."),
}


def send_verification_code(to: str, code: str, minutes: int) -> bool:
    return send_code(to, code, minutes, "verify-clinician")


def send_code(to: str, code: str, minutes: int, purpose: str) -> bool:
    """Send the code. Returns True if it went by email, False if it was only logged (dev without SMTP)."""
    subject, what, not_you = MESSAGES[purpose]
    if not configured():
        if not dev_mode():
            raise MailNotConfigured("email is not set up on this server (MITRAMIND_SMTP_*); ask your administrator")
        log.warning("DEV MODE, no SMTP: %s code for %s is %s", purpose, to, code)
        print(f"[MitraMind DEV] {purpose} code for {to}: {code}", flush=True)
        return False
    msg = EmailMessage()
    user = os.environ["MITRAMIND_SMTP_USER"]
    msg["From"] = os.environ.get("MITRAMIND_SMTP_FROM", user)
    msg["To"] = to
    msg["Subject"] = f"{subject}: {code}"
    msg.set_content(
        f"Your MitraMind code is {code}\n\nUse it to {what}. It expires in {minutes} minutes and works once.\n\n"
        f"{not_you}\n\nIn a crisis, call Tele-MANAS 14416 (free, 24x7) or 112.\n")
    host, port = os.environ["MITRAMIND_SMTP_HOST"], int(os.environ.get("MITRAMIND_SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=20) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(user, os.environ.get("MITRAMIND_SMTP_PASSWORD", ""))
        s.send_message(msg)
    return True
