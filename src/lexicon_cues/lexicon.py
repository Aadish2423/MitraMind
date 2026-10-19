"""Text normalisation and YAML lexicon compilation.

Lexicon terms and input text go through the same `normalize()` + `tokenize()`, so a
term written as "can't sleep" matches "cant sleep", "can not sleep" and "cannot sleep".
Matching runs on the space-joined token string with whole-token boundaries, which fixes
the substring bug in Low et al.'s original counter ("rent" inside "parent").

Term syntax:
  plain phrase      whole-token match after normalisation
  word*             prefix wildcard on one token          (unemploy* -> unemployment)
  *                 any single token                       (lost * pounds)
  {a|b c}           alternatives, expanded before compile  ({can|could} not sleep)
  re:<regex>        raw regex on the normalised token string
"""
from __future__ import annotations

import html
import itertools
import pathlib
import re
from dataclasses import dataclass, field

import yaml

LEXICON_DIR = pathlib.Path(__file__).resolve().parents[2] / "lexicons"

# ------------------------------------------------------------------- normalisation
_REPLACEMENTS = [
    (re.compile(r"https?://\S+|www\.\S+"), " url "),
    (re.compile(r"[’‘`´]"), "'"),
    (re.compile(r"\b([ap])\.m\.?"), r"\1m"),
    (re.compile(r"\b(?:can't|cannot|cant)\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"\b(do|does|did|is|was|were|are|could|would|should|have|has|had|must)nt\b"), r"\1 not"),
    (re.compile(r"\bi'm\b|\bim\b"), "i am"),
    (re.compile(r"\bive\b"), "i have"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"\b(it|that|there|what|he|she|who|where|here|everything|nothing)'s\b"), r"\1 is"),
    (re.compile(r"'s\b"), ""),
    (re.compile(r"\bwanna\b"), "want to"),
    (re.compile(r"\bgonna\b"), "going to"),
    (re.compile(r"\bgotta\b"), "got to"),
    (re.compile(r"\bidk\b"), "i do not know"),
    (re.compile(r"\n+"), " . "),
]
_TOKEN = re.compile(r"\d+(?::\d+)?(?:am|pm)?|[a-z0-9]+(?:-[a-z0-9]+)*|[.!?;,]")

WILD_TOKEN = "zzwildtok"
WILD_SUFFIX = "zzwildsfx"


def normalize(text: str) -> str:
    text = html.unescape(text or "").lower()
    for pattern, repl in _REPLACEMENTS:
        text = pattern.sub(repl, text)
    return text


def tokenize(normalized: str) -> list[str]:
    return _TOKEN.findall(normalized)


# ----------------------------------------------------------------- term compiling
_BRACES = re.compile(r"\{([^{}]*)\}")


def expand_braces(term: str) -> list[str]:
    parts = _BRACES.split(term)  # literal, options, literal, options, ...
    choices = [[p] if i % 2 == 0 else p.split("|") for i, p in enumerate(parts)]
    return [re.sub(r"\s+", " ", "".join(c)).strip() for c in itertools.product(*choices)]


def term_to_regex(term: str) -> str | None:
    if term.startswith("re:"):
        return term[3:]
    marked = re.sub(r"(?<!\S)\*(?!\S)", WILD_TOKEN, term)
    marked = re.sub(r"\*", WILD_SUFFIX, marked)
    tokens = tokenize(normalize(marked))
    if not tokens:
        return None
    out = []
    for tok in tokens:
        if tok == WILD_TOKEN:
            out.append(r"\S+")
        elif tok.endswith(WILD_SUFFIX):
            out.append(re.escape(tok[: -len(WILD_SUFFIX)]) + r"[a-z0-9-]*")
        else:
            out.append(re.escape(tok))
    return " ".join(out)


def compile_terms(terms: list[str]) -> re.Pattern | None:
    regexes = {r for t in terms for e in expand_braces(t) if (r := term_to_regex(e))}
    if not regexes:
        return None
    # Longest alternative first so "kill myself" beats "kill" style prefixes.
    alt = "|".join(sorted(regexes, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9-])(?:{alt})(?![a-z0-9-])")


# --------------------------------------------------------------------- lexicons
@dataclass
class Item:
    name: str
    label: str
    pattern: re.Pattern | None
    requires_first_person: bool
    negatable: bool
    tier: int | None = None
    in_total: bool = True


@dataclass
class Family:
    name: str
    description: str
    source: str
    item_rates: bool
    items: list[Item] = field(default_factory=list)

    @property
    def tiered(self) -> bool:
        return any(i.tier for i in self.items)


def load_lexicons(directory: pathlib.Path = LEXICON_DIR) -> dict[str, Family]:
    docs = []
    for path in sorted(directory.glob("*.yaml")):
        # Regex-based Hindi/Hinglish files (crisis_indic, indic_symptoms) belong to src/phase1.
        docs.extend(d for d in yaml.safe_load_all(path.read_text(encoding="utf-8"))
                    if d and "family" in d and all("terms" in s or "include" in s for s in d["items"].values()))

    raw_terms = {
        (d["family"], name): list(spec.get("terms") or [])
        for d in docs
        for name, spec in d["items"].items()
    }

    families: dict[str, Family] = {}
    for d in docs:
        defaults = d.get("defaults", {})
        fam = Family(d["family"], d.get("description", ""), d.get("source", ""),
                     bool(d.get("item_rates", False)))
        for name, spec in d["items"].items():
            terms = list(spec.get("terms") or [])
            for ref in spec.get("include", []):
                ref_fam, ref_item = ref.split(".")
                if (ref_fam, ref_item) not in raw_terms:
                    raise KeyError(f"{d['family']}.{name} includes unknown item {ref}")
                terms += raw_terms[(ref_fam, ref_item)]
            fam.items.append(Item(
                name=name,
                label=spec.get("label", name),
                pattern=compile_terms(terms),
                requires_first_person=spec.get("requires_first_person",
                                               defaults.get("requires_first_person", False)),
                negatable=spec.get("negatable", defaults.get("negatable", False)),
                tier=spec.get("tier"),
                in_total=spec.get("tier", 1) != 0,
            ))
        families[fam.name] = fam
    return families
