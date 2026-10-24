"""Create (or reset) a MitraMind superuser: sees every patient, every clinician and the clinician report.

Superusers cannot sign up on the website; they are made here, on the server.

    python src/create_superuser.py --email admin@example.com --name "Admin"          # asks for a password
    python src/create_superuser.py --email admin@example.com --generate --save var/superuser.txt

--generate makes a random password; --save writes the email and password to a file (keep it out of git:
var/ is ignored) instead of printing the password.
"""
import argparse
import getpass
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from care.service import CareService  # noqa: E402
from care.store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True)
    ap.add_argument("--name")
    ap.add_argument("--db", help="SQLite file (default: MITRAMIND_DB or var/mitramind.db)")
    ap.add_argument("--generate", action="store_true", help="make a random password")
    ap.add_argument("--save", help="write the email and generated password to this file")
    args = ap.parse_args()

    if args.generate:
        password = secrets.token_urlsafe(15)
    else:
        password = getpass.getpass("Superuser password (12+ characters): ")
        if password != getpass.getpass("Again: "):
            ap.error("the passwords do not match")
    svc = CareService(Store(args.db), orchestrator=None)
    try:
        prof = svc.create_superuser(args.email, password, args.name)
    except ValueError as e:
        ap.error(str(e))
    print(f"superuser ready: {prof['email']} (id {prof['id']})")
    if args.generate:
        if args.save:
            path = Path(args.save)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"email: {prof['email']}\npassword: {password}\n", encoding="utf-8")
            print(f"password written to {path}")
        else:
            print(f"password: {password}")


if __name__ == "__main__":
    main()
