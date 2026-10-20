# MitraMind Care: build log and remaining work

This file tracks the build from Phase 1 (State Card) to the companion, the clinician side and the
Phase 2/3 data and model pipeline. It is updated and pushed after every step, so whoever picks the work
up can see exactly what is done and what is left.

Branch: `phase1-orchestrator` on github.com/Aadish2423/mitramind-cue-models.

## Status at a glance

| # | Step | Status |
|---|------|--------|
| 0 | Phase 1: cue models m1-m5, safety layer, PHQ-9/GAD-7, LLM State Card, API, test page | done |
| 1 | Storage: patients, DPDP consent, messages, questionnaires, alerts, audit log of every AI reply | done |
| 2 | Text chat companion: safety first, LLM reply with guardrails, Mind Pause, Reflection Share | done |
| 3 | Voice: speech-to-text, voice emotion (m3/m4), spoken replies | done (m3 needs training, see Remaining) |
| 4 | Clinician side: alert queue (risk triage), dashboard, State Card from stored data | done |
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

### Steps 1-2: storage and the text companion (done)

- `src/care/store.py`: SQLite (`var/mitramind.db`, gitignored; set `MITRAMIND_DB`). Tables: patients,
  consents (append-only, latest row wins), messages, questionnaires, alerts, audit_log, mind_pause,
  clinician_feedback. Consent purposes: `companion` (includes "safety alerts go to my care team"),
  `care_team`, `pilot_research`, `research_text`.
- `src/care/companion.py`: every message is analysed by Phase 1 first. Risk high/urgent → a fixed crisis
  reply in English / Hinglish / Hindi with Tele-MANAS 14416 and 112; the LLM is never called. Otherwise
  the LLM replies under a JSON schema (`reply`, `mind_pause`). Guardrails reject diagnosis words,
  medicine/dose advice, "I'm your therapist" and secrecy promises; one retry, then a fixed fallback.
  Risk "watch" adds a safety note to the prompt and opens a low-priority alert.
- `src/care/service.py`: the rules for every channel (consent checks, alerts, audit, clinician privacy).
- `src/care/mind_pause.py`: box breathing, 5-4-3-2-1 grounding, bedtime body scan (en + hi).
- `src/care/instruments.py`: PHQ-9 / GAD-7 item text (English only, see Remaining).
- `src/care/routes.py` + `src/api.py`: patient routes `/v1/patients/...` (enrol, consent, chat, messages,
  share, mind-pause, questionnaires) and clinician routes `/v1/clinician/...`; `/v1/audit`.
- `src/web/app.html` at `/`: patient app with Chat, Check-in, Mind Pause and Privacy tabs.
- `tests/test_care.py`: 11 tests (crisis never reaches the LLM, guardrail retry/fallback, consent,
  clinician sees only shared text, questionnaire alerts, State Card from stored chats).
- Checked live with Ollama qwen3.5: a Hinglish message got a Hinglish reply with a Mind Pause offer
  (31 s); "I want to die" got the crisis reply, an alert and an audit row without calling the LLM.

### Step 3: voice (done)

- `src/care/voice.py`: the clip is saved under `var/audio/<patient>/`, converted to 16 kHz mono WAV with
  ffmpeg (any browser/WhatsApp format), and transcribed. STT backends: local Whisper
  (`MITRAMIND_STT=whisper`, default `openai/whisper-small`, multilingual, runs on the GPU; set
  `MITRAMIND_STT_MODEL` for an IndicWhisper / Hindi fine-tune) or Sarvam saarika over HTTPS
  (`MITRAMIND_STT=sarvam` + `SARVAM_API_KEY`, written but not tested: no key).
- `POST /v1/patients/{id}/voice` (`audio_base64`, `mime`, `share`, optional `language` hint): transcript ?
  the same safety-first companion path as text, with `modality=voice`. The saved WAV goes to the voice
  emotion models m3 / m4 through the orchestrator.
- Patient app: a Voice button records 16 kHz WAV in the browser (no ffmpeg needed for app uploads) and
  "Read replies aloud" uses the browser's speech synthesis (hi-IN / en-IN voices).
- Checked: a Windows-TTS English clip was transcribed exactly by Whisper-small and got a companion reply
  (59 s on the first call, which includes loading Whisper).
- m4 (MSP-Podcast WavLM) runs once its two MIT-licensed models are in the Hugging Face cache:
  `python src/run_msp.py` once, or the `snapshot_download` of `3loi/SER-Odyssey-Baseline-WavLM-Categorical`
  and `...-Multi-Attributes`. m3 needs `src/prepare_cremad.py` + `src/train_cremad.py` (see README).

### Step 4: clinician side (done)

- Alerts: any message or questionnaire with risk above "none" opens an alert (watch / high / urgent),
  with the recommended action from the crisis layer. `GET /v1/clinician/alerts`,
  `POST /v1/clinician/alerts/{id}` (acknowledged / resolved, who, outcome note).
- `GET /v1/clinician/dashboard`: patients sorted by open-alert level, PHQ-9 / GAD-7 latest, severity and
  change, companion usage (messages, voice, shared, Mind Pauses). Nothing beyond the alert count is
  shown without the patient's `care_team` consent.
- `GET /v1/clinician/patients/{id}/state-card`: the Phase 1 User State Card built from the stored chats and
  questionnaires (only shared texts reach the LLM). `.../shared`: the reflections the patient shared.
  `.../feedback`: clinician rating, accuracy, minutes saved (pilot outcome data).
- `src/web/clinician.html` at `/clinician`. Checked live: alert acknowledge, summary from stored chats.
