# MitraMind pilot dataset (synthetic)

**SYNTHETIC DATA** made by `src/simulate_pilot.py` to test the pipeline. Never use it to judge a model.

- Exported: 2026-10-20T04:18:22+00:00, consent version 2026-10-v1
- Patients: 172 with `pilot_research` consent (28 excluded)
- Windows: 2064 {'PHQ-9': 1032, 'GAD-7': 1032}; worsened rate 0.089
- Shared texts (research_text consent): 820; labelled alerts: 188;
  clinician summary ratings: 53

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
