"""Rule/lexicon-based cue extractor (Phase 1, no LLM).

`CueExtractor().extract(text)` returns one flat dict of interpretable cues:

  cue_self_*   pronoun focus rates                       (open word lists)
  cue_sent_*   VADER neg/neu/pos/compound                 (vaderSentiment, MIT)
  cue_emo_*    Empath emotion categories per 100 tokens   (empath analyze(), MIT, offline)
  cue_ling_*   negation, absolutist, certainty, tentative, past/future rates
  cue_cd_*     12 cognitive-distortion types              (Bathina et al. n-grams)
  cue_phq_*    PHQ-9 symptom domains                      (own seeds, clinician review pending)
  cue_gad_*    GAD-7 symptom domains                      (own seeds, clinician review pending)
  cue_sleep_*  sleep disturbance
  cue_risk_*   crisis cues + risk_tier 0-4                (Low et al. suicidality + own, C-SSRS tiers)
  cue_str_*    stressor domains                           (Low et al. lexicons + SAD categories)
  cue_cope_*   coping strategies mentioned                (own seeds)
  cue_style_*  length, readability, caps, punctuation     (textstat, MIT)

Clinical families (phq, gad, sleep, risk) only count a hit when it is not negated in the
3 tokens before it, and when its sentence is about the writer (first-person pronoun, or
no other-person pronoun at all). Hits removed by negation are counted in *_negated.
"""
from __future__ import annotations

import bisect
import os
from multiprocessing import Pool

from .lexicon import Family, load_lexicons, normalize, tokenize

FAMILY_ORDER = ["self", "ling", "cd", "phq", "gad", "sleep", "risk", "str", "cope"]
TIER_NAMES = {0: "none", 1: "indicator", 2: "ideation", 3: "behavior", 4: "attempt"}

EMPATH_CATS = [
    "sadness", "fear", "nervousness", "anger", "rage", "hate", "disgust", "shame",
    "suffering", "pain", "torment", "death", "weakness", "irritability", "disappointment",
    "negative_emotion", "positive_emotion", "joy", "optimism", "cheerfulness", "love",
    "affection", "contentment", "surprise", "sleep", "emotional",
]

NEGATORS = {"not", "no", "never", "without", "nor", "neither", "hardly", "barely"}
CLAUSE_BREAK = {".", "!", "?", ";", ",", "but", "however", "although", "though"}
SENT_BREAK = {".", "!", "?"}
# Subject-like first person only: "my" is left out so "my friend can't sleep" is not self.
FIRST_PERSON = {"i", "me", "myself"}
OTHER_PERSON = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they",
                "them", "their", "theirs", "themselves", "you", "your", "yours", "yourself"}
PUNCT = SENT_BREAK | {";", ","}
NEG_WINDOW = 3
MAX_TERMS = 15


class CueExtractor:
    def __init__(self, use_empath: bool = True):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.families: dict[str, Family] = load_lexicons()
        self.vader = SentimentIntensityAnalyzer()
        self.empath = None
        if use_empath:
            from empath import Empath

            self.empath = Empath()  # analyze() only: create_category() calls a remote server

    # ----------------------------------------------------------------- public API
    def extract(self, text: str) -> dict:
        text = text if isinstance(text, str) else ""
        tokens = tokenize(normalize(text))
        joined = " ".join(tokens)
        starts, pos = [], 0
        for tok in tokens:
            starts.append(pos)
            pos += len(tok) + 1

        words = [t for t in tokens if t not in PUNCT]
        n_words = len(words)
        per100 = 100.0 / n_words if n_words else 0.0
        sent_of, sent_fp, sent_other = self._sentences(tokens)

        out: dict = {"n_words": n_words}
        out.update(self._sentiment(text))
        out.update(self._emotions(text, n_words))

        found: dict[str, list[str]] = {}
        for fam_name in FAMILY_ORDER:
            fam = self.families[fam_name]
            total, negated, terms, present = 0, 0, [], []
            for item in fam.items:
                count = 0
                for m in (item.pattern.finditer(joined) if item.pattern else ()):
                    t = bisect.bisect_right(starts, m.start()) - 1
                    if item.negatable and self._negated(tokens, t):
                        negated += 1
                        continue
                    s = sent_of[t]
                    if item.requires_first_person and not sent_fp[s] and sent_other[s]:
                        continue
                    count += 1
                    if len(terms) < MAX_TERMS:
                        terms.append(f"{item.name}:{m.group(0)}")
                key = f"cue_{fam.name}_{item.name}"
                if fam.item_rates:
                    out[f"{key}_rate"] = round(count * per100, 3)
                else:
                    out[key] = count
                if count and item.in_total:
                    total += count
                    present.append(item.name)
            if fam.item_rates:
                continue
            out[f"cue_{fam.name}_total"] = total
            out[f"cue_{fam.name}_rate"] = round(total * per100, 3)
            out[f"cue_{fam.name}_n_items"] = len(present)
            if any(i.negatable for i in fam.items):
                out[f"cue_{fam.name}_negated"] = negated
            if fam.tiered:
                tier = max((i.tier for i in fam.items if i.tier and out[f"cue_{fam.name}_{i.name}"]),
                           default=0)
                out[f"cue_{fam.name}_tier"] = tier
                out[f"cue_{fam.name}_tier_name"] = TIER_NAMES[tier]
            out[f"cue_{fam.name}_terms"] = "; ".join(dict.fromkeys(terms))
            found[fam.name] = present

        out.update(self._style(text, tokens, words, sent_of))
        out["cue_summary"] = self._summary(found, out)
        return out

    def extract_many(self, texts, workers: int = 1, chunksize: int = 64) -> list[dict]:
        texts = list(texts)
        if workers <= 1 or len(texts) < 2 * chunksize:
            return [self.extract(t) for t in texts]
        with Pool(workers, initializer=_init_worker, initargs=(self.empath is not None,)) as pool:
            return pool.map(_extract_in_worker, texts, chunksize=chunksize)

    # -------------------------------------------------------------------- helpers
    @staticmethod
    def _sentences(tokens):
        sent_of, sent_fp, sent_other, sid = [], [False], [False], 0
        for tok in tokens:
            sent_of.append(sid)
            if tok in FIRST_PERSON:
                sent_fp[sid] = True
            elif tok in OTHER_PERSON:
                sent_other[sid] = True
            if tok in SENT_BREAK:
                sid += 1
                sent_fp.append(False)
                sent_other.append(False)
        return sent_of, sent_fp, sent_other

    @staticmethod
    def _negated(tokens, t: int) -> bool:
        for j in range(t - 1, max(t - 1 - NEG_WINDOW, -1), -1):
            if tokens[j] in CLAUSE_BREAK:
                return False
            if tokens[j] in NEGATORS:
                return True
        return False

    def _sentiment(self, text: str) -> dict:
        s = self.vader.polarity_scores(text)
        return {f"cue_sent_{k}": s[k] for k in ("neg", "neu", "pos", "compound")}

    def _emotions(self, text: str, n_words: int) -> dict:
        if self.empath is None:
            return {}
        scores = self.empath.analyze(text, categories=EMPATH_CATS, normalize=True) if n_words else None
        scores = scores or {}
        return {f"cue_emo_{c}": round(100 * (scores.get(c) or 0.0), 3) for c in EMPATH_CATS}

    @staticmethod
    def _style(text, tokens, words, sent_of) -> dict:
        import textstat

        n_words = len(words)
        n_sents = len({sent_of[i] for i, t in enumerate(tokens) if t not in PUNCT}) or (1 if n_words else 0)
        raw_words = [w for w in text.split() if sum(ch.isalpha() for ch in w) >= 2]
        caps = sum(1 for w in raw_words if w.isupper())
        out = {
            "cue_style_n_sents": n_sents,
            "cue_style_words_per_sent": round(n_words / n_sents, 2) if n_sents else 0.0,
            "cue_style_ttr": round(len(set(words)) / n_words, 3) if n_words else 0.0,
            "cue_style_caps_ratio": round(caps / len(raw_words), 3) if raw_words else 0.0,
            "cue_style_n_question": text.count("?"),
            "cue_style_n_exclaim": text.count("!"),
        }
        for name, fn in (("fk_grade", textstat.flesch_kincaid_grade),
                         ("ari", textstat.automated_readability_index),
                         ("gunning_fog", textstat.gunning_fog)):
            try:
                out[f"cue_style_{name}"] = round(float(fn(text)), 2) if n_words else 0.0
            except Exception:
                out[f"cue_style_{name}"] = float("nan")
        return out

    @staticmethod
    def _summary(found: dict[str, list[str]], out: dict) -> str:
        parts = []
        if out.get("cue_risk_tier"):
            parts.append(f"RISK:{out['cue_risk_tier_name']}")
        for fam in ("phq", "gad", "sleep", "str"):
            if found.get(fam):
                parts.append(f"{fam}:{','.join(found[fam])}")
        parts.append(f"sent:{out['cue_sent_compound']:+.2f}")
        return " | ".join(parts)


# ---------------------------------------------------------------- multiprocessing
_WORKER: CueExtractor | None = None


def _init_worker(use_empath: bool) -> None:
    global _WORKER
    _WORKER = CueExtractor(use_empath=use_empath)


def _extract_in_worker(text: str) -> dict:
    return _WORKER.extract(text)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)
