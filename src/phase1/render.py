"""Render a User State Card as the clinician's pre-visit summary (Markdown)."""
from __future__ import annotations

from .schema import UserStateCard

BANNER = {"urgent": "URGENT - contact patient now", "high": "HIGH - review within 24 h",
          "watch": "WATCH - review at next visit", "none": "No crisis signal"}


def card_to_markdown(card: UserStateCard) -> str:
    c, n, r = card.cues, card.narrative, card.risk
    period = (f"{c.period_start:%d %b} - {c.period_end:%d %b %Y}" if c.period_start else "no check-ins")
    lines = [f"# Pre-visit summary: {card.patient_ref}", "",
             f"**Period:** {period} | {c.n_entries} check-ins | generated {card.generated_at:%Y-%m-%d %H:%M} UTC", "",
             f"## Safety: {BANNER[r.level]}", ""]
    for reason in r.reasons:
        lines.append(f"- {reason}")
    if r.level != "none":
        lines += [f"- **Action:** {r.recommended_action}", *(f"- {h}" for h in r.helplines)]
        for e in r.flagged_entries:
            terms = ", ".join(e.lexicon_terms + e.indic_terms) or "-"
            lines.append(f"  - check-in `{e.entry_id}`: {e.level}; phrases: {terms}"
                         + (f"; classifier p={e.classifier_prob}" if e.classifier_prob is not None else ""))
    lines += ["", "## Questionnaires", ""]
    if card.questionnaires:
        lines += ["| Scale | Scores (oldest -> latest) | Change | Now |", "|---|---|---|---|"]
        for name, t in card.questionnaires.items():
            hist = " -> ".join(f"{s.total} ({s.date:%d %b})" for s in t.history)
            flag = " (clinically meaningful)" if t.clinically_meaningful_change else ""
            change = "baseline (1 score)" if t.direction == "baseline_only" else f"{t.change:+d} {t.direction}{flag}"
            lines.append(f"| {name} | {hist} | {change} | {t.latest_severity} |")
    else:
        lines.append("No PHQ-9 / GAD-7 completed in this period.")
    lines += ["", "## Summary", "", n.clinician_summary, ""]
    if n.main_themes:
        lines += ["**Themes:** " + "; ".join(n.main_themes), ""]
    if n.stressors:
        lines += ["**Stressors:**", *(f"- {s.domain.replace('_', ' ')}: {s.detail}" for s in n.stressors), ""]
    lines += [f"**Sleep:** {n.sleep_note}", ""]
    if n.coping_that_helped:
        lines.append("**Helped:** " + "; ".join(n.coping_that_helped))
    if n.coping_concerns:
        lines.append("**Coping concerns:** " + "; ".join(n.coping_concerns))
    if n.discussion_points:
        lines += ["", "**Possible discussion points:**", *(f"- {d}" for d in n.discussion_points)]
    if n.evidence:
        lines += ["", "**Evidence:**", *(f"- {e.statement} ({', '.join(e.entry_ids)})" for e in n.evidence)]
    lines += ["", "---",
              f"Narrative: {card.narrative_source}" + (f" ({card.llm_model}, {card.llm_attempts} attempt(s))" if card.llm_model else "")
              + f" | data confidence {n.data_confidence} | cue models: {', '.join(c.models_used) or 'none'}"]
    if c.models_unavailable:
        lines.append("Unavailable models: " + "; ".join(f"{k} ({v})" for k, v in c.models_unavailable.items()))
    if card.guardrail_notes:
        lines.append("Guardrail notes: " + " | ".join(card.guardrail_notes))
    lines += ["", f"_{card.disclaimer}_", ""]
    return "\n".join(lines)
