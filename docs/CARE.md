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

Login pages: `/login` and `/signup` (patients), `/clinician/login` and `/clinician/signup`, `/admin/login`.
Clinicians use a Gmail address and confirm it with a 6-digit code sent to it (`/v1/auth/clinician/signup`,
`/verify`, `/resend`, `/login`, `/logout`, `/me`); superusers log in with `/v1/auth/admin/login`. Each login
accepts only its own kind of account. Every clinician route needs a verified clinician's token. Forgot password:
`POST <prefix>/forgot` then `<prefix>/reset` with prefix `/v1/auth`, `/v1/auth/clinician` or `/v1/auth/admin`.
Codes go by email through SMTP settings in `.env` (see `.env.example`; Gmail needs an app password).

Three views of the User State Card:
- **Patient** (`/#my-summary`, `GET /v1/patients/{id}/summary`): their own friendly card: PHQ-9 / GAD-7 over
  time, what came up, what helped, their clinician. No risk labels or clinical wording.
- **Clinician** (`/clinician`): a card for every patient linked to them. Each clinician has a care code
  (`MM-XXXXXX`); patients enter it at sign-up or under Privacy, or a superuser links them.
- **Superuser** (`/admin`): every patient's card, assignment of patients to clinicians, and a report per clinician
  (caseload, active patients, PHQ-9 / GAD-7 change of their patients, alert pick-up time against targets of
  urgent 1 h, high 24 h, watch 72 h, summaries rated) with a status: needs attention / watch / on track.
  Superusers log in from the same page (role "Superuser") but are created only on the server:
  `python src/create_superuser.py --email ... [--generate --save var/superuser.txt]`.

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
| `MITRAMIND_CHAT_LLM` | `ollama:gemma4:e4b` | companion replies (19/19 red-team cases, 7.8 s per reply) |
| `MITRAMIND_NEURAL` | `1` | load m1-m4 |
| `MITRAMIND_DB` | `var/mitramind.db` | SQLite file |
| `MITRAMIND_STT`, `MITRAMIND_STT_MODEL` | `whisper`, `openai/whisper-small` | speech-to-text |
| `SARVAM_API_KEY`, `SARVAM_STT_MODEL` | | Sarvam STT |
| `MITRAMIND_FFMPEG` | `ffmpeg` on PATH | audio conversion |
| `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_APP_SECRET`, `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID` | | WhatsApp Cloud API |
| `MITRAMIND_ENV` | `dev` | anything else requires `WHATSAPP_APP_SECRET` |
| `MITRAMIND_EXPORT_SALT` | | secret salt for pilot pseudonyms (16+ chars) |
| `MITRAMIND_CLINICIAN_DOMAINS` | `gmail.com,googlemail.com` | email domains allowed for clinician sign-up |
| `MITRAMIND_PATIENT_EMAIL_VERIFY` | `0` | `1`: new patients confirm their email with a code before logging in |
| `MITRAMIND_AUDIO_ENHANCE` | `normalise` | audio clean-up before speech-to-text: any of `highpass`, `denoise`, `normalise`, or `none` |
| `MITRAMIND_SMTP_HOST`, `_PORT`, `_USER`, `_PASSWORD`, `_FROM` | | sends clinician codes (Gmail: `smtp.gmail.com`, `587`, an app password); without them codes show on screen in dev mode only |
