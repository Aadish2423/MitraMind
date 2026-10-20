"""Download the trained cue models (m1-m3) from the GitHub Release into models/.

The weights are too large for Git, so they are published as release assets (tag models-v1). Each zip is checked
against the SHA-256 below before anything is unpacked, then extracted to models/<name>/. Standard library only.

    python src/download_models.py                    # all three, about 2.6 GB
    python src/download_models.py m3_cremad_speech   # only the ones you name
    python src/download_models.py --force            # replace models that are already there
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RELEASE = "https://github.com/Aadish2423/mitramind-cue-models/releases/download/models-v1"
ASSETS = {  # name: (sha256 of the zip, size in bytes)
    "m1_hindi_text": ("60dd7ccdb5fed7b6b9aa488a99e325314f2ecd1b43d8b19186959fa4c3f14a8a", 1116199289),
    "m2_hinglish_text": ("2eedf435cc61eda29939905bb67badb712bf71063fce11ddfe849454c250f242", 1116199313),
    "m3_cremad_speech": ("db4a20ed3915728a902a36dcb4a331439a60fd20b8dbe61f3d0f752e32fc1572", 378352865),
}


def download(url: str, dest: Path, size: int) -> str:
    """Stream `url` to `dest` with a progress line; return the file's SHA-256."""
    h, done = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:,.0f} / {size / 1e6:,.0f} MB", end="", flush=True)
    print()
    return h.hexdigest()


def extract(zpath: Path, name: str) -> None:
    target = (MODELS / name).resolve()
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            if not (MODELS / member).resolve().is_relative_to(target):
                raise SystemExit(f"refusing to unpack {zpath.name}: unexpected path {member!r}")
        if target.exists():
            shutil.rmtree(target)
        z.extractall(MODELS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"models to fetch (default: all of {', '.join(ASSETS)})")
    ap.add_argument("--force", action="store_true", help="download again even if the model is already there")
    args = ap.parse_args()
    unknown = set(args.names) - set(ASSETS)
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(sorted(unknown))}; choose from {', '.join(ASSETS)}")

    MODELS.mkdir(exist_ok=True)
    for name in args.names or ASSETS:
        sha, size = ASSETS[name]
        if (MODELS / name / "config.json").exists() and not args.force:
            print(f"{name}: already in models/, skipping (use --force to replace)")
            continue
        print(f"{name}: downloading {size / 1e9:.2f} GB")
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / f"{name}.zip"
            got = download(f"{RELEASE}/{name}.zip", zpath, size)
            if got != sha:
                raise SystemExit(f"{name}: checksum mismatch (expected {sha}, got {got}); nothing was unpacked")
            extract(zpath, name)
        print(f"{name}: ready in models/{name}/")


if __name__ == "__main__":
    sys.exit(main())
