# MitraMind Care: companion, clinician side, and the Phase 2/3 pipeline

Phase 1 ([PHASE1.md](PHASE1.md)) turns check-ins into a User State Card. This layer turns that into the
product from the strategy: a between-visit companion for patients, a triage and summary tool for the care
team, and the data loop that later trains MitraMind's own small models.

```
 patient app (/)  ─┐                                         ┌─► alert queue ─► clinician dashboard (/clinician)
 WhatsApp webhook ─┼─► CareService ─► Phase 1 analysis ─► risk ┤
 voice (Whisper) ──┘     (consent)    (cues + crisis layer)   └─► reply: high/urgent → fixed crisis reply + helplines
                              │                                         otherwise → LLM (schema, guardrails, retry, fallback)
                              ▼
                 SQLite store: messages, questionnaires, consents, alerts, audit log, Mind Pause, feedback
                              │
               Phase 2 export (consented, pseudonymised) ─► Phase 3 small models (deterioration, alert precision)
```

## Rules that hold on every channel (`src/care/service.py`)

| Rule | How |
|---|---|
| No companion without consent | `companion` consent; it also tells the patient that danger signs alert the care team |
| Safety before any reply | `Phase1Orchestrator.analyze_entry()` runs first; high/urgent never reaches the LLM |
| Every AI reply is logged | `audit_log`: kind, model, prompt SHA-256, reply, guardrail log, attempts, risk |
| Chats are private | Clinicians see counts, scores and alerts; message text only if the patient shared it |
| Care-team view needs consent | `care_team` consent gates the dashboard row, State Card, shared texts and FHIR |
| Consent is append-only | latest row per purpose is current; withdrawals are kept (DPDP Act 2023) |

## Companion (`src/care/companion.py`)

- System prompt: warm companion, not a clinician; same language and script as the patient (English,
  Hinglish, Hindi); 2-5 sentences; suggestions limited to Mind Pause, sleep routine, reaching out to someone
  they trust, journaling, or raising the topic at the next visit.
- Output schema `{reply, mind_pause}` (constrained decoding, as for the State Card).
- Guardrails: diagnosis words, medicine or dose advice, claims of being a therapist or doctor, promises of
  secrecy, over 110 words. One retry with the problems listed, then a fixed fallback reply.
- Risk `watch` (hopelessness or burden phrases, classifier): the reply is still from the LLM, with a safety
  note in the prompt, and a low-priority alert is opened.
- Crisis replies (English / Hinglish / Hindi) are fixed text, reviewed once, never generated.
  **They need review by the clinical champion and a native speaker before the pilot.**
- Mind Pause (`mind_pause.py`): box breathing, 5-4-3-2-1 grounding, bedtime body scan.
- Reflection Share: per message, at send time or later (`/messages/{id}/share`, or "share" on WhatsApp).

## Voice (`src/care/voice.py`)

Any upload format is converted with ffmpeg to 16 kHz mono WAV and saved under `var/audio/`. STT is local
Whisper (`openai/whisper-small` by default; `MITRAMIND_STT_MODEL` for an Indic fine-tune) or Sarvam
(`MITRAMIND_STT=sarvam`). The transcript takes the text path; the WAV also goes to m3 / m4 for voice emotion.
In the patient app, replies are read aloud with the browser's speech synthesis.

## Clinician side

`/v1/clinician/alerts` (acknowledge, resolve with an outcome label), `/v1/clinician/dashboard`,
`/v1/clinician/patients/{id}/state-card`, `/shared`, `/feedback`, `/fhir`. Page: `/clinician`.

## Phase 2 and 3

| Step | Script | Output |
|---|---|---|
| Synthetic pilot (pipeline test only) | `src/simulate_pilot.py` | `var/pilot_sim.db` |
| De-identified export | `src/export_pilot_dataset.py` | `windows.csv`, `alert_outcomes.csv`, `summary_ratings.csv`, `DATASHEET.md` |
| Small models | `src/train_outcome_model.py` | `models/outcome_model/*.joblib`, `outputs/phase3/*.md` |

The synthetic export and report are committed (`outputs/phase2_synthetic/`, `outputs/phase3/`) so the
pipeline can be inspected without running it. They say nothing about real patients.

## Red-team suite

`tests/redteam_cases.yaml` (direct and indirect self-harm in three languages, medication and diagnosis
requests, jailbreaks, secrecy, abuse disclosure, off-topic) run by `src/redteam_companion.py` against any
Ollama model. Results: `outputs/redteam/companion_redteam.md`. Add every real pilot incident as a case
before changing a prompt or model.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MITRAMIND_LLM` | `ollama:qwen3.5` | State Card narrator (`openai:<model>@<url>` or `none`) |
| `MITRAMIND_CHAT_LLM` | = `MITRAMIND_LLM` | companion replies |
| `MITRAMIND_NEURAL` | `1` | load m1-m4 |
| `MITRAMIND_DB` | `var/mitramind.db` | SQLite file |
| `MITRAMIND_STT`, `MITRAMIND_STT_MODEL` | `whisper`, `openai/whisper-small` | speech-to-text |
| `SARVAM_API_KEY`, `SARVAM_STT_MODEL` | | Sarvam STT |
| `MITRAMIND_FFMPEG` | `ffmpeg` on PATH | audio conversion |
| `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_APP_SECRET`, `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID` | | WhatsApp Cloud API |
| `MITRAMIND_ENV` | `dev` | anything else requires `WHATSAPP_APP_SECRET` |
| `MITRAMIND_EXPORT_SALT` | | secret salt for pilot pseudonyms (16+ chars) |
