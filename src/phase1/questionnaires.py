"""PHQ-9 and GAD-7 scoring: the validated measure the State Card is anchored to.

Severity bands follow Kroenke et al. 2001 (PHQ-9) and Spitzer et al. 2006 (GAD-7). The
"clinically meaningful change" thresholds (5 points PHQ-9, 4 points GAD-7) are commonly used
conventions; confirm them with the clinical champion.
"""
from __future__ import annotations

from .schema import QuestionnaireResponse, QuestionnaireScore, QuestionnaireTrend

N_ITEMS = {"PHQ-9": 9, "GAD-7": 7}
BANDS = {
    "PHQ-9": [(4, "minimal"), (9, "mild"), (14, "moderate"), (19, "moderately severe"), (27, "severe")],
    "GAD-7": [(4, "minimal"), (9, "mild"), (14, "moderate"), (21, "severe")],
}
MEANINGFUL_CHANGE = {"PHQ-9": 5, "GAD-7": 4}


def score(resp: QuestionnaireResponse) -> QuestionnaireScore:
    n = N_ITEMS[resp.instrument]
    if len(resp.items) != n or any(i not in (0, 1, 2, 3) for i in resp.items):
        raise ValueError(f"{resp.instrument} needs {n} item scores in 0-3, got {resp.items}")
    total = sum(resp.items)
    severity = next(label for upper, label in BANDS[resp.instrument] if total <= upper)
    return QuestionnaireScore(instrument=resp.instrument, date=resp.date, total=total, severity=severity,
                              item9=resp.items[8] if resp.instrument == "PHQ-9" else None)


def trends(responses: list[QuestionnaireResponse]) -> dict[str, QuestionnaireTrend]:
    out = {}
    for instrument in ("PHQ-9", "GAD-7"):
        hist = sorted((score(r) for r in responses if r.instrument == instrument), key=lambda s: s.date)
        if not hist:
            continue
        change = hist[-1].total - hist[0].total
        meaningful = abs(change) >= MEANINGFUL_CHANGE[instrument]
        if len(hist) == 1:
            direction = "baseline_only"
        else:
            direction = "stable" if not meaningful and abs(change) < 2 else ("improved" if change < 0 else "worsened")
        out[instrument] = QuestionnaireTrend(
            instrument=instrument, n_administrations=len(hist), first_total=hist[0].total,
            latest_total=hist[-1].total, change=change, direction=direction,
            clinically_meaningful_change=meaningful, latest_severity=hist[-1].severity, history=hist)
    return out
