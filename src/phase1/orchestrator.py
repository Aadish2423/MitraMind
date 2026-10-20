"""Phase 1 orchestrator: check-ins -> cue models -> safety layer -> LLM -> User State Card.

    orch = Phase1Orchestrator(backend=OllamaBackend("qwen3.5"))
    analysis = orch.analyze_entry(entry)          # real-time: cues + crisis triage for one check-in
    card = orch.build_card(bundle)                # pre-visit: the full User State Card

Privacy: the LLM sees cue counts, matched cue words and questionnaire scores for every check-in,
but the text of a check-in only if the patient shared it (Entry.shared_with_clinician).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

from .crisis import CrisisClassifier, assess_entry, combine
from .cue_models import EMOTION_CUES, CueModelRegistry, detect_language
from .indic import indic_cues
from .llm import constrain_narrative_schema, generate, guardrail_issues, inline_schema
from .questionnaires import trends
from .schema import (CueAggregate, Entry, EntryRisk, Evidence, Narrative, PatientBundle, Stressor,
                     UserStateCard)

STRESSOR_LABELS = {
    "economic": "money / housing costs", "work": "work", "school_exam": "studies / exams", "family": "family",
    "relationships": "relationships", "health": "physical health", "isolation": "loneliness / isolation",
    "substance_use": "alcohol / substances", "domestic": "conflict at home", "covid19": "COVID-19",
}
PHQ_LABELS = {
    "anhedonia": "loss of interest", "depressed_mood": "low mood", "sleep": "sleep problems", "fatigue": "tiredness",
    "appetite": "appetite changes", "worthlessness_guilt": "guilt / worthlessness", "concentration": "poor concentration",
    "psychomotor": "restlessness / slowing", "self_harm_thoughts": "thoughts of self-harm",
}
GAD_LABELS = {
    "nervous": "nervousness", "uncontrollable_worry": "uncontrollable worry", "excessive_worry": "excessive worry",
    "trouble_relaxing": "trouble relaxing", "restless": "restlessness", "irritable": "irritability",
    "dread": "sense of dread", "panic_somatic": "panic / bodily anxiety",
}
SLEEP_LABELS = {"insomnia": "trouble falling/staying asleep", "hypersomnia": "sleeping too much",
                "small_hours": "awake at 1-4 am", "nightmares": "nightmares", "sleep_aid": "sleep aids"}
COPING_LABELS = {
    "breathing": "breathing exercises", "mindfulness": "meditation / yoga / prayer", "physical_activity": "exercise / walking",
    "social_support": "talking to someone", "journaling": "journaling", "creative_music": "music / hobbies",
    "routine_selfcare": "rest / self-care", "professional_help": "therapy / professional help",
    "substance_coping": "alcohol or substances to cope", "avoidance": "avoidance / withdrawal",
}
UNHELPFUL_COPING = {"substance_coping", "avoidance"}
_HELPED = re.compile(r"\b(help|helped|helps|helping|helpful|better|calm|calmer|calmed|relax|relaxed|relaxing|"
                     r"worked|works|eased|lighter|good|nice|peaceful|fresh|faster|easier|sukoon|accha|achha)\b"
                     r"|शांत|अच्छा लगा|हल्का|सुकून")
_NOT_HELPED = re.compile(r"\b(not|never|no) (help|helped|helps|helping|work|worked|use)\b|\buseless\b|\bwaste\b"
                         r"|\bfayda nahi|कोई फायदा नहीं")
_SENTENCE = re.compile(r"(?<=[.!?\n।])\s*")

SYSTEM_PROMPT = """You are MitraMind's clinical documentation assistant. You write a short pre-visit summary \
for the patient's psychiatrist or psychologist from structured check-in data collected between visits.

Rules:
- Use only facts present in DATA. Do not guess. If the data is thin, say so and set data_confidence to "low".
- PHQ-9 and GAD-7 totals are the measure of severity. Quote them exactly, with the direction of change.
- Describe what the patient reported, with counts taken from DATA (for example "mentioned trouble sleeping \
in N of M check-ins", where N and M are the real counts). Never diagnose, never name disorders, never \
suggest medication or treatment changes.
- Do not assess or comment on suicide or self-harm risk; a separate safety system reports it.
- Check-in text appears only where the patient chose to share it. Quote at most a few words.
- evidence: statements about what the patient said in check-ins, each citing the check-in entry_ids \
(e.g. "s03") where it was reported. Questionnaire scores need no evidence item.
- stressors[].domain must be one of the stressor domains present in DATA, or "other".
- coping_that_helped: only strategies DATA marks as reported helpful. coping_concerns: unhelpful coping \
(e.g. alcohol, avoidance) or strategies reported as not helping.
- discussion_points: 2-4 neutral topics the clinician may want to explore in the consultation.
- clinician_summary: 3-5 sentences, at most 90 words, plain clinical English. Only use numbers that appear in DATA."""


class Phase1Orchestrator:
    def __init__(self, backend=None, enable_neural: bool = True, allow_download: bool = False):
        self.backend = backend
        self.registry = CueModelRegistry(enable_neural=enable_neural, allow_download=allow_download)
        self.classifier = CrisisClassifier()

    # ------------------------------------------------------------- per check-in
    def analyze_entry(self, entry: Entry) -> dict:
        text = entry.text or entry.transcript or ""
        modality = "audio" if entry.audio_path else "text"
        language = entry.language if entry.language != "auto" else detect_language(text)

        model_ids = self.registry.route(modality, language)
        if modality == "audio" and text:
            model_ids += self.registry.route("text", language)
        outputs, used = {}, []
        for mid in dict.fromkeys(model_ids):
            model = self.registry.get(mid)
            if model is None:
                continue
            inp = entry.audio_path if mid in ("m3_cremad_speech", "m4_msp_speech") else text
            if inp:
                outputs[mid] = model.run(inp)
                used.append(mid)

        lex = outputs.get("m5_lexicon_text")
        emotion = [o for m, o in outputs.items() if m != "m5_lexicon_text"]
        cue_probs = ({c: float(np.mean([o["cue_probs"][c] for o in emotion])) for c in EMOTION_CUES}
                     if emotion else None)
        risk = assess_entry(entry.entry_id, text, lex, self.classifier, language)
        digest = self._lexicon_digest(lex) if lex else None
        coping = self._coping(text) if lex else {}
        if text and language in ("hi", "hinglish"):
            digest, coping = self._merge_indic(text, digest, coping)
            used.append("indic_lexicon")
        return {
            "entry_id": entry.entry_id, "timestamp": entry.timestamp, "modality": modality, "language": language,
            "shared": entry.shared_with_clinician, "text": text, "models_used": used,
            "emotion": None if cue_probs is None else {
                "cue_probs": {c: round(p, 4) for c, p in cue_probs.items()},
                "top_cue": max(cue_probs, key=cue_probs.get),
                "distress": round(float(np.mean([o["distress_score"] for o in emotion])), 4),
                "arousal": next((o["arousal"] for o in emotion if "arousal" in o), None)},
            "lexicon": digest,
            "coping": coping,
            "risk": risk,
        }

    @staticmethod
    def _merge_indic(text: str, digest: dict | None, coping: dict) -> tuple[dict, dict]:
        hits = indic_cues(text)
        digest = digest or {"phq": [], "gad": [], "sleep": [], "stressors": {}, "sentiment": None,
                            "risk_tier": 0, "self_focus": None, "n_words": len(text.split())}
        for fam in ("phq", "gad", "sleep"):
            digest[fam] = sorted(set(digest[fam]) | set(hits.get(fam, {})))
        if "insomnia" in hits.get("sleep", {}) and "sleep" not in digest["phq"]:
            digest["phq"] = sorted(digest["phq"] + ["sleep"])
        for dom, terms in hits.get("stressors", {}).items():
            digest["stressors"][dom] = sorted(set(digest["stressors"].get(dom, [])) | set(terms))
        coping = dict(coping)
        for sent in _SENTENCE.split(text):
            low = sent.lower()
            for item in indic_cues(sent).get("coping", {}):
                verdict = ("not_helpful" if _NOT_HELPED.search(low)
                           else "helpful" if _HELPED.search(low) else "mentioned")
                if coping.get(item) != "helpful":
                    coping[item] = verdict
        return digest, coping

    @staticmethod
    def _items(cues: dict, family: str, names) -> list[str]:
        return [n for n in names if cues.get(f"cue_{family}_{n}", 0) > 0]

    def _lexicon_digest(self, cues: dict) -> dict:
        terms = defaultdict(list)
        for pair in (cues.get("cue_str_terms") or "").split("; "):
            if ":" in pair:
                item, term = pair.split(":", 1)
                terms[item].append(term)
        return {
            "phq": self._items(cues, "phq", PHQ_LABELS), "gad": self._items(cues, "gad", GAD_LABELS),
            "sleep": self._items(cues, "sleep", ["insomnia", "hypersomnia", "small_hours", "nightmares", "sleep_aid"]),
            "stressors": {k: sorted(set(v)) for k, v in terms.items()},
            "sentiment": cues["cue_sent_compound"], "risk_tier": cues["cue_risk_tier"],
            "self_focus": cues["cue_self_fps_rate"], "n_words": cues["n_words"],
        }

    def _coping(self, text: str) -> dict[str, str]:
        """coping item -> 'helpful' | 'not_helpful' | 'mentioned', judged per sentence."""
        from lexicon_cues.lexicon import normalize, tokenize

        fam = self.registry.get("m5_lexicon_text").extractor.families["cope"]
        found = {}
        sents = [" ".join(tokenize(normalize(s))) for s in _SENTENCE.split(text)]
        for i, joined in enumerate(sents):
            # "Tried the breathing exercise. It actually helped." -> judge with the next sentence too.
            window = " ".join(sents[i:i + 2])
            for item in fam.items:
                if item.pattern and item.pattern.search(joined):
                    if _NOT_HELPED.search(window) or item.name in UNHELPFUL_COPING:
                        verdict = "not_helpful"
                    elif _HELPED.search(window):
                        verdict = "helpful"
                    else:
                        verdict = "mentioned"
                    if found.get(item.name) != "helpful":
                        found[item.name] = verdict
        return found

    # --------------------------------------------------------------- per patient
    def build_card(self, bundle: PatientBundle, use_llm: bool = True) -> tuple[UserStateCard, dict]:
        entries = sorted(bundle.entries, key=lambda e: e.timestamp)
        analyses = [self.analyze_entry(e) for e in entries]
        q = trends(bundle.questionnaires)
        phq = q.get("PHQ-9")
        risk = combine([a["risk"] for a in analyses], phq.history if phq else [],
                       phq.change if phq and phq.change > 0 else None)
        agg = self._aggregate(analyses)
        digest = self._digest(analyses, q, agg)

        notes, narrative, source, attempts, model_name = [], None, "rule_fallback", 0, None
        if use_llm and self.backend is not None:
            ok, why = self.backend.available()
            if not ok:
                notes.append(f"LLM unavailable ({why}); rule-based narrative used")
            else:
                allowed_ids = {a["entry_id"] for a in analyses}
                allowed_str = set(agg.stressors)
                allowed_nums = set(re.findall(r"\b\d+\b", json.dumps(digest, default=str)))
                user = "DATA:\n" + json.dumps(digest, ensure_ascii=False, default=str, indent=1)
                schema = constrain_narrative_schema(inline_schema(Narrative), sorted(allowed_ids), sorted(allowed_str))
                counts = checkin_counts(digest)
                narrative, log = generate(
                    self.backend, SYSTEM_PROMPT, user, Narrative,
                    lambda n: guardrail_issues(n, allowed_ids, allowed_str, allowed_nums, len(analyses), counts),
                    schema=schema)
                attempts, model_name = len(log), self.backend.name
                for entry in log:
                    for issue in entry.get("guardrail_issues", []) + ([entry["error"]] if "error" in entry else []):
                        notes.append(f"attempt {entry['attempt']}: {issue}")
                if narrative is not None:
                    source = "llm"
                else:
                    notes.append("LLM output rejected after retries; rule-based narrative used")
        if narrative is None:
            narrative = self._rule_narrative(analyses, q, agg)

        card = UserStateCard(
            patient_ref=bundle.patient_ref, generated_at=datetime.now(timezone.utc), questionnaires=q, risk=risk,
            cues=agg, narrative=narrative, narrative_source=source, llm_model=model_name, llm_attempts=attempts,
            guardrail_notes=notes)
        return card, {"digest": digest, "analyses": analyses}

    def _aggregate(self, analyses: list[dict]) -> CueAggregate:
        def count(key) -> dict[str, int]:
            c = Counter(x for a in analyses if a["lexicon"] for x in a["lexicon"][key])
            return dict(c.most_common())

        stress_terms = defaultdict(set)
        for a in analyses:
            for dom, terms in (a["lexicon"] or {}).get("stressors", {}).items():
                stress_terms[dom].update(terms)
        coping = defaultdict(Counter)
        for a in analyses:
            for item, verdict in a["coping"].items():
                coping[verdict][item] += 1
        distress = [a["emotion"]["distress"] for a in analyses if a["emotion"]]
        tops = Counter(a["emotion"]["top_cue"] for a in analyses if a["emotion"])
        sent = [a["lexicon"]["sentiment"] for a in analyses if a["lexicon"] and a["lexicon"]["sentiment"] is not None]
        used = sorted({m for a in analyses for m in a["models_used"]})
        return CueAggregate(
            n_entries=len(analyses),
            period_start=analyses[0]["timestamp"] if analyses else None,
            period_end=analyses[-1]["timestamp"] if analyses else None,
            modalities=dict(Counter(a["modality"] for a in analyses)),
            languages=dict(Counter(a["language"] for a in analyses)),
            models_used=used, models_unavailable=dict(self.registry.unavailable),
            mean_distress=round(float(np.mean(distress)), 3) if distress else None,
            distress_trend=_trend(distress, 0.1),
            emotion_cue_share={c: round(n / sum(tops.values()), 3) for c, n in tops.most_common()},
            sentiment_mean=round(float(np.mean(sent)), 3) if sent else None,
            phq_domains=count("phq"), gad_domains=count("gad"), sleep=count("sleep"),
            stressors=dict(Counter(d for a in analyses if a["lexicon"] for d in a["lexicon"]["stressors"]).most_common()),
            stressor_terms={k: sorted(v)[:8] for k, v in stress_terms.items()},
            coping_helpful=dict(coping["helpful"]), coping_mentioned=dict(coping["mentioned"]),
            coping_unhelpful=dict(coping["not_helpful"]))

    def _digest(self, analyses, q, agg: CueAggregate) -> dict:
        """What the LLM sees. Cue-level for every check-in; text only for shared check-ins."""
        days = (agg.period_end - agg.period_start).days + 1 if analyses else 0
        weekly = defaultdict(list)
        for a in analyses:
            weekly[(a["timestamp"] - timedelta(days=a["timestamp"].weekday())).date()].append(a)
        return {
            "period": {"start": agg.period_start, "end": agg.period_end, "days": days,
                       "weeks": max(1, round(days / 7)) if days else 0, "n_checkins": agg.n_entries},
            "questionnaires": {
                name: {"history": [{"date": s.date, "total": s.total, "severity": s.severity} for s in t.history],
                       "change": t.change, "direction": t.direction}
                for name, t in q.items()},
            "reported_in_checkins": {
                "phq9_domains": {PHQ_LABELS[k]: v for k, v in agg.phq_domains.items()},
                "gad7_domains": {GAD_LABELS[k]: v for k, v in agg.gad_domains.items()},
                "sleep": {"checkins_with_sleep_problems": sum(1 for a in analyses if a["lexicon"] and a["lexicon"]["sleep"]),
                          "types_reported": {SLEEP_LABELS.get(k, k): v for k, v in agg.sleep.items()}},
                "stressor_domains": {k: {"label": STRESSOR_LABELS.get(k, k), "checkins": v,
                                         "words_used": agg.stressor_terms.get(k, [])} for k, v in agg.stressors.items()},
                "coping_reported_helpful": {COPING_LABELS.get(k, k): v for k, v in agg.coping_helpful.items()},
                "coping_mentioned": {COPING_LABELS.get(k, k): v for k, v in agg.coping_mentioned.items()},
                "coping_unhelpful_or_not_working": {COPING_LABELS.get(k, k): v for k, v in agg.coping_unhelpful.items()},
            },
            # Lexicon sentiment is left out on purpose: VADER reads "everyone would be better off without
            # me" as positive, and the LLM then reports "mood improved". Emotion-model distress stays.
            "mood_signals": {"emotion_model_distress_0_to_1": agg.mean_distress, "distress_trend": agg.distress_trend,
                             "top_emotion_cue_share": agg.emotion_cue_share},
            "weekly": [{"week_of": wk, "checkins": len(items),
                        "domains": sorted({PHQ_LABELS[d] for a in items if a["lexicon"] for d in a["lexicon"]["phq"]}
                                          | {STRESSOR_LABELS.get(d, d) for a in items if a["lexicon"] for d in a["lexicon"]["stressors"]})}
                       for wk, items in sorted(weekly.items())],
            "checkins": [{"entry_id": a["entry_id"], "date": a["timestamp"].date(), "language": a["language"],
                          "modality": a["modality"],
                          **({"phq9": a["lexicon"]["phq"], "gad7": a["lexicon"]["gad"], "sleep": a["lexicon"]["sleep"],
                              "stressors": list(a["lexicon"]["stressors"])} if a["lexicon"] else {}),
                          **({"top_emotion": a["emotion"]["top_cue"]} if a["emotion"] else {}),
                          "coping": a["coping"],
                          **({"shared_text": a["text"][:400]} if a["shared"] and a["text"] else {})}
                         for a in analyses],
        }

    def _rule_narrative(self, analyses, q, agg: CueAggregate) -> Narrative:
        n = agg.n_entries
        parts = []
        if agg.period_start:
            days = (agg.period_end - agg.period_start).days + 1
            parts.append(f"{n} check-ins over {days} days.")
        for name, t in q.items():
            scores = " -> ".join(str(s.total) for s in t.history)
            parts.append(f"{name} {scores} ({t.direction}; now {t.latest_severity}).")
        top_sym = [f"{PHQ_LABELS[k]} ({v}/{n})" for k, v in list(agg.phq_domains.items())[:3]]
        top_anx = [f"{GAD_LABELS[k]} ({v}/{n})" for k, v in list(agg.gad_domains.items())[:2]]
        if top_sym or top_anx:
            parts.append("Most often reported: " + ", ".join(top_sym + top_anx) + ".")
        top_str = [STRESSOR_LABELS.get(k, k) for k in list(agg.stressors)[:3]]
        if top_str:
            parts.append("Main stressors mentioned: " + ", ".join(top_str) + ".")
        helped = [k.replace("_", " ") for k in agg.coping_helpful]
        if helped:
            parts.append("Reported as helpful: " + ", ".join(helped) + ".")
        sleep = ", ".join(f"{k.replace('_', ' ')} in {v} check-ins" for k, v in agg.sleep.items()) or "no sleep problems mentioned"
        evidence = []
        for dom in list(agg.stressors)[:2]:
            ids = [a["entry_id"] for a in analyses if a["lexicon"] and dom in a["lexicon"]["stressors"]][:6]
            evidence.append(Evidence(statement=f"{STRESSOR_LABELS.get(dom, dom)} mentioned in {len(ids)} check-in(s)",
                                     entry_ids=ids))
        return Narrative(
            clinician_summary=" ".join(parts)[:700] or "No check-in data in this period.",
            main_themes=[PHQ_LABELS[k] for k in list(agg.phq_domains)[:3]] + [STRESSOR_LABELS.get(k, k) for k in list(agg.stressors)[:2]],
            stressors=[Stressor(domain=k if k in STRESSOR_LABELS else "other",
                                detail=("words used: " + ", ".join(agg.stressor_terms.get(k, [])[:4]))[:120])
                       for k in list(agg.stressors)[:6]],
            coping_that_helped=helped[:5],
            coping_concerns=[k.replace("_", " ") for k in agg.coping_unhelpful][:4],
            sleep_note=sleep[:200],
            discussion_points=[f"Explore {PHQ_LABELS[k]}" for k in list(agg.phq_domains)[:2]]
            + [f"Discuss {STRESSOR_LABELS.get(k, k)}" for k in list(agg.stressors)[:2]],
            evidence=evidence,
            data_confidence="low" if n < 3 else "medium" if n < 8 else "high")


def checkin_counts(digest: dict) -> set[int]:
    """Every per-domain check-in count in the digest (valid N in 'N of M check-ins'), plus M itself."""
    rep = digest["reported_in_checkins"]
    counts = {digest["period"]["n_checkins"], rep["sleep"]["checkins_with_sleep_problems"]}
    for key in ("phq9_domains", "gad7_domains", "coping_reported_helpful", "coping_mentioned",
                "coping_unhelpful_or_not_working"):
        counts |= set(rep[key].values())
    counts |= {v["checkins"] for v in rep["stressor_domains"].values()}
    return counts


def _trend(values: list[float], delta: float) -> str:
    if len(values) < 4:
        return "insufficient_data"
    half = len(values) // 2
    diff = float(np.mean(values[half:]) - np.mean(values[:half]))
    return "rising" if diff > delta else "falling" if diff < -delta else "stable"


def entry_risk_summary(r: EntryRisk) -> str:
    return f"{r.level} (tier {r.lexicon_tier}{', p=' + str(r.classifier_prob) if r.classifier_prob is not None else ''})"
