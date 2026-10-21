"""Email for clinician verification codes (stdlib smtplib only).

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


def send_verification_code(to: str, code: str, minutes: int) -> bool:
    """Send the code. Returns True if it went by email, False if it was only logged (dev without SMTP)."""
    if not configured():
        if not dev_mode():
            raise MailNotConfigured("email is not set up on this server (MITRAMIND_SMTP_*); ask your administrator")
        log.warning("DEV MODE, no SMTP: verification code for %s is %s", to, code)
        print(f"[MitraMind DEV] verification code for {to}: {code}", flush=True)
        return False
    msg = EmailMessage()
    user = os.environ["MITRAMIND_SMTP_USER"]
    msg["From"] = os.environ.get("MITRAMIND_SMTP_FROM", user)
    msg["To"] = to
    msg["Subject"] = f"Your MitraMind clinician code: {code}"
    msg.set_content(
        f"Your MitraMind verification code is {code}\n\n"
        f"It expires in {minutes} minutes. If you did not try to create a clinician account, ignore this email.\n")
    host, port = os.environ["MITRAMIND_SMTP_HOST"], int(os.environ.get("MITRAMIND_SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=20) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(user, os.environ.get("MITRAMIND_SMTP_PASSWORD", ""))
        s.send_message(msg)
    return True
