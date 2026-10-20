# MitraMind Care: build log and remaining work

This file tracks the build from Phase 1 (State Card) to the companion, the clinician side and the
Phase 2/3 data and model pipeline. It is updated and pushed after every step, so whoever picks the work
up can see exactly what is done and what is left.

Branch: `phase1-orchestrator` on github.com/Aadish2423/mitramind-cue-models.

## Status at a glance

| # | Step | Status |
|---|------|--------|
| 0 | Phase 1: cue models m1-m5, safety layer, PHQ-9/GAD-7, LLM State Card, API, test page | done |
| 1 | Storage: patients, DPDP consent, messages, questionnaires, alerts, audit log of every AI reply | todo |
| 2 | Text chat companion: safety first, LLM reply with guardrails, Mind Pause, Reflection Share | todo |
| 3 | Voice: speech-to-text, voice emotion (m3/m4), spoken replies | todo |
| 4 | Clinician side: alert queue (risk triage), dashboard, State Card from stored data | todo |
| 5 | Phase 2: de-identified pilot dataset export (conversations linked to PHQ/GAD change) | todo |
| 6 | Phase 3: train small models on the pilot dataset | todo |
| 7 | Integrations: WhatsApp Business webhook, FHIR export | todo |
| 8 | Red-team suite, end-to-end test, docs | todo |

## How the strategy maps to code

| Strategy item | Where it lives |
|---|---|
| Automated check-ins (PHQ-9 / GAD-7, scores over time) | `src/phase1/questionnaires.py`; app forms (step 1/2) |
| Between-visit companion: chat and voice, Mind Pause, Reflection Share | step 2 and 3 |
| Risk triage: worsening scores or crisis cues, alert, staff call, Tele-MANAS | `src/phase1/crisis.py` + alert queue (step 4) |
| Pre-visit summary | `src/phase1/orchestrator.py` (User State Card) |
| Phase 2: consented pilot data linking conversations to outcome change | step 5 |
| Phase 3: small, cheap models on that data | step 6 |
| WhatsApp channel, FHIR / ABDM | step 7 |
| DPDP consent, audit log of every AI reply, red-team suite | steps 1, 2, 8 |

## Log

Entries are added as each step finishes.
