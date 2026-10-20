"""Phase 2: export the de-identified pilot dataset (see src/care/pilot.py for the rules).

    set MITRAMIND_EXPORT_SALT=<secret, 16+ chars, kept at the hospital>
    python src/export_pilot_dataset.py --db var/mitramind.db --out outputs/phase2_pilot
    python src/export_pilot_dataset.py --db var/pilot_sim.db --out outputs/phase2_synthetic --source synthetic
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from care.pilot import export  # noqa: E402
from care.store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=["pilot", "synthetic"], default="pilot")
    args = ap.parse_args()
    salt = os.environ.get("MITRAMIND_EXPORT_SALT")
    if args.source == "synthetic" and not salt:
        salt = "synthetic-data-not-secret"  # pseudonyms of fictional patients need no secrecy
    if not salt:
        ap.error("set MITRAMIND_EXPORT_SALT (16+ characters, kept at the hospital) before exporting pilot data")
    meta = export(Store(args.db), Path(args.out), salt, args.source)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
