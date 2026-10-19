# Phase 1: from check-ins to a User State Card

Phase 1 turns a patient's between-visit check-ins (chat text, journal notes, voice clips) and their
PHQ-9 / GAD-7 answers into a **User State Card**: a structured JSON record plus a one-page
pre-visit summary for the clinician. Each layer has a fixed job:

```
check-in (text / voice) ──► language + modality routing
                              │
          ┌───────────────────┼─────────────────────────────┐
          ▼                   ▼                             ▼
   m1 Hindi text       m2 Hinglish/English text     m3 + m4 speech        m5 lexicon (English/Hinglish)
   (XLM-R)             (XLM-R)                      (WavLM)               + Hindi/Hinglish phrase lexicon
          └──────────── emotion cues, distress ─────────────┘           PHQ/GAD domains, sleep, stressors,
                                                                          coping, crisis tiers
                              │
                              ▼
   SAFETY LAYER (no LLM): crisis lexicon tiers + Hindi/Hinglish crisis phrases
                          + TF-IDF/LR crisis classifier + PHQ-9 item 9  ──► none / watch / high / urgent
                              │
   QUESTIONNAIRES (no LLM): PHQ-9 / GAD-7 totals, severity bands, change over time
                              │
                              ▼
   cue digest (counts, matched words, scores; check-in text ONLY if the patient shared it)
                              │
                              ▼
   LLM (open weights, JSON-schema-constrained decoding) writes ONLY the narrative
     └─► Pydantic validation ─► guardrails (no diagnosis, no treatment advice, no risk talk,
         only real entry ids / stressor domains / numbers) ─► 1 retry ─► rule-based fallback
                              │
                              ▼
   User State Card (JSON)  +  pre-visit summary (Markdown)
```

## Design rules

- **The LLM cannot change safety or scores.** Risk level, PHQ-9 / GAD-7 totals and cue counts are
  computed in code and merged after the LLM call. A model that hallucinates or refuses still produces
  a usable card because the rule-based narrative takes over.
- **Validated questionnaires are the measure.** Cues describe what the patient talked about; the
  severity shown is always the PHQ-9 / GAD-7 total.
- **Privacy by default.** The LLM sees, for every check-in, only the cue labels and matched cue words
  ("exams", "neend nahi aati"). It sees a check-in's text only if the patient turned on Reflection Share
  (`shared_with_clinician`), and only the first 400 characters.
- **Fail soft, never silent.** Missing neural models are listed in `cues.models_unavailable`. LLM
  failures and guardrail rejections are listed in `guardrail_notes`.

## Safety layer

| Signal | Source | Can raise to |
|---|---|---|
| English crisis lexicon (C-SSRS-style tiers, negation and first-person aware) | `lexicons/crisis.yaml` via m5 | indicator→watch, ideation→high, behaviour/attempt→urgent |
| Hindi / Hinglish crisis phrases | `lexicons/crisis_indic.yaml` | same tiers |
| Crisis classifier (TF-IDF + logistic regression, r/SuicideWatch vs 20 other subreddits) | `src/train_crisis_classifier.py` | watch (needs negative tone). Never "high" on its own: no threshold reached 75% precision |
| PHQ-9 item 9 | questionnaire | 1 → high, ≥2 → urgent |
| PHQ-9 worsened ≥ 5 points | questionnaire | watch |

`high` and `urgent` set `escalate=true` and show Tele-MANAS (14416 / 1-800-891-4416) and 112.
Classifier: test ROC-AUC 0.92. At the watch threshold it flags 64% of r/SuicideWatch posts, 19% of
r/depression, 5% of r/lonely and 0.2% of r/fitness (`outputs/phase1/crisis_classifier_metrics.json`).

## Files

| Path | What |
|---|---|
| `src/phase1/schema.py` | Pydantic models: Entry, PatientBundle, UserStateCard, Narrative (LLM part) |
| `src/phase1/cue_models.py` | m1-m5 adapters, language detection, routing, lazy loading |
| `src/phase1/indic.py`, `lexicons/indic_symptoms.yaml` | Hindi / Hinglish symptom, stressor and coping phrases |
| `src/phase1/crisis.py`, `lexicons/crisis_indic.yaml` | safety layer |
| `src/phase1/questionnaires.py` | PHQ-9 / GAD-7 scoring and trends |
| `src/phase1/llm.py` | Ollama and OpenAI-compatible backends, schema inlining, guardrails, retry |
| `src/phase1/orchestrator.py` | the pipeline: `analyze_entry()`, `build_card()` |
| `src/phase1/render.py` | Markdown pre-visit summary |
| `src/phase1/bench.py` | LLM structured-output benchmark |
| `src/run_phase1.py` | CLI: `analyze`, `card`, `demo`, `bench` |
| `src/api.py` | FastAPI service |
| `examples/patients/*.json` | four fictional demo patients (English, Hinglish, Hindi, crisis) |
| `outputs/phase1/cards/` | demo cards: `.json`, `.md`, and `.llm_input.json` (exactly what the LLM saw) |

## Run

```bash
pip install -r requirements-phase1.txt
python src/train_crisis_classifier.py                 # ~2 min, needs the Reddit MH cache from m5
ollama pull qwen3.5                                    # or another model, see docs/LLM_SELECTION.md
python src/run_phase1.py demo                          # cards for the 4 demo patients
python src/run_phase1.py analyze --text "neend nahi aati, exams ka tension"
python src/run_phase1.py card my_patient.json --backend openai --model Qwen/Qwen3.5-9B --base-url http://gpu-box:8000/v1
uvicorn api:app --app-dir src --port 8080              # POST /v1/checkins/analyze, /v1/state-card
python -m pytest tests                                 # 24 tests, no GPU or LLM needed
```

Use the GPU venv from the main README for m1-m4. Without it, the pipeline runs on m5 plus the
Hindi/Hinglish lexicon and lists the neural models as unavailable.
