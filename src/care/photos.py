"""Photos patients attach to chat messages (e.g. something that comforts them): type and size checks, storage.

The file type is read from the first bytes, never from the name or the browser's claim, and only JPEG, PNG
and WebP are kept. Files go to var/photos/<patient>/ (git-ignored); the database stores the path.
"""
from __future__ import annotations

import base64
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHOTO_DIR = ROOT / "var" / "photos"
MAX_BYTES = 5 * 1024 * 1024
MIME = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


class PhotoError(ValueError):
    pass


def sniff(data: bytes) -> str:
    """Return the file extension for a supported image, or raise PhotoError."""
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise PhotoError("only JPEG, PNG or WebP photos can be shared")


def save_photo(patient_id: str, data: bytes) -> Path:
    if not data:
        raise PhotoError("the photo is empty")
    if len(data) > MAX_BYTES:
        raise PhotoError(f"photos can be at most {MAX_BYTES // (1024 * 1024)} MB")
    ext = sniff(data)
    folder = PHOTO_DIR / patient_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    return path


def as_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")
