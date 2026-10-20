"""Hindi / Hinglish cue phrases (lexicons/indic_symptoms.yaml) on the same domains as m5."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

LEXICON = Path(__file__).resolve().parents[2] / "lexicons" / "indic_symptoms.yaml"


@lru_cache(maxsize=1)
def _patterns() -> dict[str, dict[str, re.Pattern]]:
    doc = yaml.safe_load(LEXICON.read_text(encoding="utf-8"))
    return {fam: {item: re.compile("|".join(f"(?:{p})" for p in pats)) for item, pats in items.items()}
            for fam, items in doc.items() if isinstance(items, dict)}


def indic_cues(text: str) -> dict[str, dict[str, list[str]]]:
    """family -> item -> matched phrases, e.g. {"sleep": {"insomnia": ["neend nahi aati"]}}."""
    low = (text or "").lower()
    out: dict[str, dict[str, list[str]]] = {}
    for fam, items in _patterns().items():
        for item, pat in items.items():
            hits = sorted({m.group(0).strip() for m in pat.finditer(low)})
            if hits:
                out.setdefault(fam, {})[item] = hits
    return out
