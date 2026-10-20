"""Phase 2: the de-identified pilot dataset that links what patients said to how their scores changed.

One row per *window* = the time between two consecutive administrations of the same questionnaire
(PHQ-9 or GAD-7) for one patient. Features summarise the companion messages inside the window
(cue counts, emotion, risk, usage); labels are the score change at the end of the window.

Rules:
- only patients whose current `pilot_research` consent is granted;
- patient ids become salted SHA-256 pseudonyms (the salt stays with the hospital, never in the export);
- no message text at all, except shared reflections of patients who also granted `research_text`
  (written to a separate file);
- timestamps become day offsets from the patient's first questionnaire (no calendar dates).
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from phase1.cue_models import EMOTION_CUES
from phase1.orchestrator import COPING_LABELS, GAD_LABELS, PHQ_LABELS, STRESSOR_LABELS
from phase1.questionnaires import MEANINGFUL_CHANGE

from .store import ALERT_OUTCOMES, CONSENT_VERSION, LEVEL_ORDER, Store

SLEEP_ITEMS = ["insomnia", "hypersomnia", "small_hours", "nightmares", "sleep_aid"]


def pseudonym(patient_id: str, salt: str) -> str:
    return "p_" + hashlib.sha256(f"{salt}:{patient_id}".encode()).hexdigest()[:12]


def _ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _day(value: str) -> datetime:
    return _ts(value if "T" in value else value + "T23:59:59+00:00")


def feature_names() -> list[str]:
    names = ["days", "n_messages", "n_voice", "n_shared", "messages_per_week", "mind_pause_started",
             "mind_pause_completed", "max_risk", "n_risk_watch_plus", "n_alerts", "mean_distress"]
    names += [f"emo_{c}" for c in EMOTION_CUES]
    names += [f"phq_{d}" for d in PHQ_LABELS] + [f"gad_{d}" for d in GAD_LABELS]
    names += [f"sleep_{s}" for s in SLEEP_ITEMS] + [f"str_{s}" for s in STRESSOR_LABELS]
    names += [f"cope_{c}" for c in COPING_LABELS]
    return names


def window_features(msgs: list[dict], pauses: list[dict], alerts: list[dict], days: float) -> dict:
    """Share of messages mentioning each cue (0-1) plus usage and emotion means for one window."""
    n = len(msgs)
    f = {"days": round(days, 1), "n_messages": n, "n_voice": sum(m["modality"] == "voice" for m in msgs),
         "n_shared": sum(bool(m["shared"]) for m in msgs),
         "messages_per_week": round(n / max(days, 1) * 7, 2),
         "mind_pause_started": len(pauses), "mind_pause_completed": sum(bool(p["completed"]) for p in pauses),
         "max_risk": max((LEVEL_ORDER.get(m["risk_level"] or "none", 0) for m in msgs), default=0),
         "n_risk_watch_plus": sum(LEVEL_ORDER.get(m["risk_level"] or "none", 0) >= 1 for m in msgs),
         "n_alerts": len(alerts)}
    analyses = [m["analysis"] or {} for m in msgs]
    emos = [a["emotion"] for a in analyses if a.get("emotion")]
    f["mean_distress"] = round(statistics.mean(e["distress"] for e in emos), 4) if emos else None
    for c in EMOTION_CUES:
        f[f"emo_{c}"] = round(statistics.mean(e["cue_probs"][c] for e in emos), 4) if emos else None
    counts = Counter()
    for a in analyses:
        lex = a.get("lexicon") or {}
        counts.update(f"phq_{d}" for d in lex.get("phq", []))
        counts.update(f"gad_{d}" for d in lex.get("gad", []))
        counts.update(f"sleep_{s}" for s in lex.get("sleep", []))
        counts.update(f"str_{s}" for s in lex.get("stressors", {}))
        counts.update(f"cope_{c}" for c in (a.get("coping") or {}))
    for name in feature_names():
        if name.split("_")[0] in ("phq", "gad", "sleep", "str", "cope"):
            f[name] = round(counts[name] / n, 4) if n else 0.0
    return f


def export(store: Store, out_dir: Path, salt: str, source: str = "pilot") -> dict:
    if not salt or len(salt) < 16:
        raise ValueError("use a secret salt of at least 16 characters (keep it at the hospital)")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, texts, alert_rows, n_patients, skipped = [], [], [], 0, 0
    for p in store.patients():
        pid = p["id"]
        consents = store.consents(pid)
        if not consents["pilot_research"]:
            skipped += 1
            continue
        n_patients += 1
        pseud = pseudonym(pid, salt)
        qs = store.questionnaires(pid)
        if not qs:
            continue
        t_ref = _day(qs[0]["date"])
        msgs = store.messages(pid, role="user")
        pauses = store.mind_pauses(pid)
        alerts = store.alerts(None, pid)
        for a in alerts:
            if a.get("outcome") in ALERT_OUTCOMES:
                alert_rows.append({"patient": pseud, "day": round((_ts(a["created_at"]) - t_ref).days, 1),
                                   "level": a["level"], "source": a["source"], "outcome": a["outcome"]})
        for instrument in ("PHQ-9", "GAD-7"):
            hist = [q for q in qs if q["instrument"] == instrument]
            for w, (q0, q1) in enumerate(zip(hist, hist[1:])):
                t0, t1 = _day(q0["date"]), _day(q1["date"])
                inside = lambda ts: t0 < _ts(ts) <= t1  # noqa: E731
                w_msgs = [m for m in msgs if inside(m["created_at"])]
                feats = window_features(w_msgs, [x for x in pauses if inside(x["created_at"])],
                                        [x for x in alerts if inside(x["created_at"])], (t1 - t0).total_seconds() / 86400)
                change = q1["total"] - q0["total"]
                rows.append({"patient": pseud, "instrument": instrument, "window": w,
                             "start_day": (t0 - t_ref).days, "baseline_total": q0["total"],
                             "baseline_item9": q0["items"][8] if instrument == "PHQ-9" else None,
                             **feats, "next_total": q1["total"], "change": change,
                             "worsened": int(change >= MEANINGFUL_CHANGE[instrument]),
                             "improved": int(change <= -MEANINGFUL_CHANGE[instrument])})
                if consents["research_text"]:
                    texts += [{"patient": pseud, "instrument": instrument, "window": w, "text": m["text"]}
                              for m in w_msgs if m["shared"]]

    cols = (["patient", "instrument", "window", "start_day", "baseline_total", "baseline_item9"] + feature_names()
            + ["next_total", "change", "worsened", "improved"])
    with open(out_dir / "windows.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    with open(out_dir / "alert_outcomes.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["patient", "day", "level", "source", "outcome"])
        wr.writeheader()
        wr.writerows(alert_rows)
    if texts:
        with open(out_dir / "shared_texts.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in texts)
    fb = [{"patient": pseudonym(r["patient_id"], salt), "rating": r["rating"], "accurate": r["accurate"],
           "minutes_saved": r["minutes_saved"]} for r in store.feedback()
          if store.has_consent(r["patient_id"], "pilot_research")]
    with open(out_dir / "summary_ratings.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["patient", "rating", "accurate", "minutes_saved"])
        wr.writeheader()
        wr.writerows(fb)

    meta = {"source": source, "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "consent_version": CONSENT_VERSION, "patients_included": n_patients,
            "patients_without_consent": skipped, "windows": len(rows),
            "windows_by_instrument": dict(Counter(r["instrument"] for r in rows)),
            "worsened_rate": round(statistics.mean(r["worsened"] for r in rows), 3) if rows else None,
            "shared_texts": len(texts), "labelled_alerts": len(alert_rows), "summary_ratings": len(fb)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "DATASHEET.md").write_text(_datasheet(meta), encoding="utf-8")
    return meta


def _datasheet(meta: dict) -> str:
    synthetic = meta["source"] == "synthetic"
    return f"""# MitraMind pilot dataset ({meta['source']})

{"**SYNTHETIC DATA** made by `src/simulate_pilot.py` to test the pipeline. Never use it to judge a model." if synthetic else
"De-identified pilot data. Keep under the ethics approval and data-sharing agreement; do not publish raw rows."}

- Exported: {meta['exported_at']}, consent version {meta['consent_version']}
- Patients: {meta['patients_included']} with `pilot_research` consent ({meta['patients_without_consent']} excluded)
- Windows: {meta['windows']} {meta['windows_by_instrument']}; worsened rate {meta['worsened_rate']}
- Shared texts (research_text consent): {meta['shared_texts']}; labelled alerts: {meta['labelled_alerts']};
  clinician summary ratings: {meta['summary_ratings']}

## Files

| File | One row per | Contents |
|---|---|---|
| `windows.csv` | questionnaire window | features of the messages between two administrations, and the score change |
| `alert_outcomes.csv` | resolved alert with an outcome label | alert level, source, clinician label (confirmed_risk / false_alarm / unclear) |
| `summary_ratings.csv` | clinician rating of a State Card | usefulness 1-5, accurate, minutes saved |
| `shared_texts.jsonl` | shared reflection (only with research_text consent) | pseudonym, window, text |

## Columns of windows.csv

- `patient`: salted SHA-256 pseudonym. `start_day`: days since the patient's first questionnaire.
- `baseline_total`, `baseline_item9`: score at the start of the window.
- Usage: `days`, `n_messages`, `n_voice`, `n_shared`, `messages_per_week`, `mind_pause_*`.
- Safety: `max_risk` (0 none … 3 urgent), `n_risk_watch_plus`, `n_alerts`.
- Emotion (m1-m4 mean): `mean_distress`, `emo_*`. Empty when no neural model ran.
- Cues: `phq_*`, `gad_*`, `sleep_*`, `str_*`, `cope_*` = share of the window's messages mentioning the cue.
- Labels: `next_total`, `change`, `worsened` (change ≥ 5 PHQ-9 / ≥ 4 GAD-7), `improved`.
"""
