# MitraMind

**A calm AI companion for Indian patients between hospital visits.** Patients talk or type to Mitra, a friendly
2D companion, in English, Hinglish or हिन्दी. A safety layer reads every message before any AI model does, and a
dashboard keeps the patient's care team in the loop. Hackathon prototype: MitraMind is a wellness support tool,
not a doctor and not an emergency service.

## The product: three sites, one server

One local server (`uvicorn`, port 8080) serves three web apps:

| Site | For | Log in | After login |
|---|---|---|---|
| Patient app | Patients | <http://127.0.0.1:8080/login> (sign up: `/signup`) | <http://127.0.0.1:8080/> |
| Clinician dashboard | Doctors and care teams | <http://127.0.0.1:8080/clinician/login> (sign up: `/clinician/signup`) | <http://127.0.0.1:8080/clinician> |
| Superuser view | Admins | <http://127.0.0.1:8080/admin/login> (no sign-up page) | <http://127.0.0.1:8080/admin> |

Developer pages: `/docs` (every API endpoint, with try-it-out) and `/tester` (Phase 1 pipeline tester).

**Patient app**
- **Home:** what MitraMind is and why, the four features, the team and privacy settings. Mitra's eyes follow the
  cursor and a small Mitra hops onto the card you point at.
- **Chat with Mitra** (`#chat`): text, voice messages (transcribed on the device by Whisper) and photos. Replies
  stream sentence by sentence. In voice mode Mitra speaks the reply, with moving mouth and live captions.
- **Check-in** (`#check-in`): PHQ-9 (mood) and GAD-7 (worry), with the change since the first check-in.
- **Mind Pause** (`#mind-pause`): box breathing, 5-4-3-2-1 grounding and a body scan, led by Mitra.
- **My summary** (`#my-summary`): the patient's own progress card.
- **Privacy:** consent switches (India's DPDP Act), and per-message "Shared with care team" or "Private".
- Email and password accounts, forgot password, optional email verification.

**Safety first.** Every message goes through the crisis layer before the LLM. High or urgent risk gets a fixed
crisis reply with Tele-MANAS 14416 and 112, never an AI-written one, and opens an alert for the care team.

**Clinician dashboard**
- Gmail sign-up confirmed by an emailed code. Each clinician gets a care-team code; patients enter it in the app
  to link to them.
- Overview tiles, the alert queue (acknowledge, or resolve with a note and "was the risk real?"), one User State
  Card per patient, the patient table, and a pre-visit summary (rule-based in seconds, or AI-written in 20-60 s)
  that the clinician can rate. An open dashboard refreshes every 30 seconds.
- Clinicians see scores and usage only with the patient's consent, and only the messages the patient shared.
  Safety alerts always reach the care team.

What opens an alert:

| Source | Rule | Level |
|---|---|---|
| Chat message (text or voice) | Crisis language (English, Hinglish, Hindi lexicons and phrases) | watch / high / urgent |
| Photo | The model judges the photo worrying | watch |
| PHQ-9 | Item 9 (thoughts of self-harm) = 1 | high |
| PHQ-9 | Item 9 ≥ 2 | urgent |
| PHQ-9 | Total worse by 5+ points since the previous PHQ-9 | watch |

A GAD-7, or a high PHQ-9 total with item 9 = 0, does not open an alert; it shows in the scores and cards.

**Superuser view:** every clinician and patient, linking patients to clinicians, and the clinician report.
Create a superuser with `python src/create_superuser.py --email you@example.com --generate`.

## Quick start

Needs Python 3.11 and [Ollama](https://ollama.com). An NVIDIA GPU speeds up voice transcription and the cue
models but is optional.

```bash
git clone https://github.com/Aadish2423/mitramind-cue-models.git
cd mitramind-cue-models
python -m venv /c/venvs/mitramind_cues          # a short path matters on Windows (see "How to run")
PY=/c/venvs/mitramind_cues/Scripts/python.exe
$PY -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install -r requirements.txt -r requirements-phase1.txt
$PY src/download_models.py                      # trained models m1-m3 (optional, about 2.6 GB)
ollama pull gemma4:e4b                          # companion chat
ollama pull qwen3.5                             # pre-visit summaries
cp .env.example .env                            # optional: Gmail SMTP for sign-up and reset codes
$PY -m uvicorn api:app --app-dir src --port 8080
```

Then open <http://127.0.0.1:8080>. Without email set up (dev mode), verification and password-reset codes are
shown on screen. Run the tests with `$PY -m pytest tests`; they need no GPU, models or LLM.

Settings (environment variables or `.env`):

| Variable | Default | What it does |
|---|---|---|
| `MITRAMIND_CHAT_LLM` | `ollama:gemma4:e4b` | Model for companion replies |
| `MITRAMIND_LLM` | `ollama:qwen3.5` | Model for the pre-visit summary (`none` = rule-based only) |
| `MITRAMIND_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama address |
| `MITRAMIND_STT` / `MITRAMIND_STT_MODEL` | `whisper` / `openai/whisper-small` | Speech-to-text (`none` turns voice off) |
| `MITRAMIND_ENV` | `dev` | `production` refuses sign-up and password reset without SMTP |
| `MITRAMIND_PATIENT_EMAIL_VERIFY` | `0` | `1` = patients confirm their email before the first login |
| `MITRAMIND_SMTP_*` | unset | Gmail SMTP for codes (see `.env.example`) |
| `MITRAMIND_CLINICIAN_DOMAINS` | `gmail.com,googlemail.com` | Email domains allowed for clinician accounts |

Replies are much faster with a GPU build of Ollama. On CPU only, the first sentence of a reply takes about 15-20 s.

## Team

| | Name | GitHub | LinkedIn |
|---|---|---|---|
| <img src="src/web/team/aadish.jpg" width="64" alt="Aadish Bane"> | **Aadish Bane** | [Aadish2423](https://github.com/Aadish2423) | [aadish-bane](https://www.linkedin.com/in/aadish-bane-292a221ba/) |
| <img src="src/web/team/shishir.jpg" width="64" alt="Shishir Pusalkar"> | **Shishir Pusalkar** | [DEADPOOL-152006](https://github.com/DEADPOOL-152006) | [shishir-pusalkar](https://www.linkedin.com/in/shishir-pusalkar-515191285/) |

---

## Research: the cue models behind MitraMind

Five cue models that turn text or a voice clip into MitraMind cues, and the **Phase 1 pipeline** that
combines them with a crisis-safety layer, PHQ-9 / GAD-7 scoring and an open-weight LLM into a
**User State Card** for the clinician (see [Phase 1 pipeline](#phase-1-pipeline-user-state-card) below
and [docs/PHASE1.md](docs/PHASE1.md)).

Models m1–m4 output emotion cues (`anxious`, `low_mood`, `angry`, `positive`, `surprised`, `neutral`)
plus a 0–1 `distress_score`; every held-out prediction is stored in `outputs/cues_all.csv`. Model m5
outputs interpretable clinical-domain cues (PHQ-9 / GAD-7 domains, sleep, stressors, coping, crisis tiers).

| Model | Input | Trained on | Tested on | Base |
|---|---|---|---|---|
| `m1_hindi_text` | Hindi text (Devanagari) | HinEmo-9 Hindi + English rows (**stand-in for EmoInHindi**) | HinEmo-9 human-verified Hindi test rows | XLM-R base |
| `m2_hinglish_text` | Hinglish text (romanized) | HinEmo-9 Hinglish + English rows | HinEmo-9 human-verified Hinglish test rows | XLM-R base |
| `m3_cremad_speech` | Voice clip | CREMA-D, 63 actors (9 more for model selection) | CREMA-D, 19 actors never heard in training | WavLM-base-plus |
| `m4_msp_speech` | Voice clip | MSP-Podcast (MSP lab's released model, not retrained) | Same unseen CREMA-D actors as m3 | WavLM-large |
| `m5_lexicon_text` | English (and English words in Hinglish) | none: rule + lexicon based | Dreaddit, GoEmotions, Reddit Mental Health (Low et al. 2020) | VADER, Empath, textstat + own YAML lexicons |

## Results (held-out test sets, run on 2026-10-18)

| Model | Test rows | Emotion accuracy | Emotion macro-F1 | Cue accuracy | Cue macro-F1 |
|---|---:|---:|---:|---:|---:|
| m1_hindi_text | 347 | 0.585 | 0.594 | 0.637 | 0.578 |
| m2_hinglish_text | 393 | 0.629 | 0.630 | 0.667 | 0.630 |
| m3_cremad_speech | 1,558 | 0.732 | 0.737 | 0.757 | 0.748 |
| m4_msp_speech | 1,558 | — (different labels) | — | 0.411 | 0.287 |

For reference, the HinEmo-9 paper reports 0.630 macro-F1 for XLM-R on all three scripts together.
Per-class scores and confusion matrices are in `outputs/<model>_metrics.json`.

How well each model catches the `anxious` cue, the one that matters most for MitraMind:

| Model | Recall | Precision |
|---|---:|---:|
| m1_hindi_text | 0.37 | 0.88 |
| m2_hinglish_text | 0.67 | 0.46 |
| m3_cremad_speech (FEA) | 0.70 | 0.64 |
| m4_msp_speech | 0.05 | 0.37 |

## m5_lexicon_text: interpretable cues (no training)

Rule- and lexicon-based extraction of 151 cue columns per text, written as one CSV per benchmark
dataset in `outputs/m5_lexicon_text/` (Reddit MH gzipped to stay under GitHub's 100 MB limit):

| Prefix | Cues |
|---|---|
| `cue_self_` `cue_ling_` | pronoun focus, negation, absolutist words, certainty, tentative, past/future focus |
| `cue_sent_` `cue_emo_` | VADER sentiment; 26 Empath emotion categories |
| `cue_cd_` | 12 cognitive-distortion types (Bathina et al. 2021) |
| `cue_phq_` `cue_gad_` `cue_sleep_` | PHQ-9 and GAD-7 symptom domains, sleep disturbance (own seed lexicons) |
| `cue_risk_` | crisis cues in C-SSRS-style tiers 0–4 (Low et al. suicidality list, fixed, + own terms) |
| `cue_str_` `cue_cope_` | stressor domains incl. Indian exam terms; coping strategies |
| `cue_style_` | readability, length, caps, punctuation |

Matching is whole-word with contraction expansion; clinical families drop negated hits ("I don't want
to die") and hits not about the writer ("my friend can't sleep"). Every family has a `_terms`
evidence column. Results (`outputs/m5_lexicon_text/validation_report.md`):

- Open word lists vs proprietary LIWC (Dreaddit): pronouns and negation r ≈ 0.99, tentative 0.87, certainty 0.74.
- Dreaddit stress: first-person rate 10.4 vs 7.0 (AUC 0.72), as in the original paper; VADER negative AUC 0.76.
- Reddit MH: the expected community ranks first on 13 of 14 cues (r/suicidewatch on crisis cues, r/depression
  on PHQ, r/anxiety on GAD, r/lonely on isolation, r/EDAnonymous on appetite, r/adhd on concentration).

```bash
python src/extract_lexicon_cues.py        # downloads Dreaddit, GoEmotions, Low et al. (Zenodo, ~460 MB, slow)
python src/validate_lexicon_cues.py
```

The PHQ-9, GAD-7, sleep, coping and crisis lexicons are developer-authored seeds and need clinician
review. Third-party word lists are credited in [NOTICE](NOTICE).

## Phase 1 pipeline: User State Card

`src/phase1/` wires the five models into the Phase 1 product loop (architecture in [docs/PHASE1.md](docs/PHASE1.md),
LLM choice in [docs/LLM_SELECTION.md](docs/LLM_SELECTION.md)):

1. **Routing:** Devanagari → m1, Hinglish / English → m2 + m5, voice → m3 + m4 (+ text models on a transcript),
   plus a Hindi/Hinglish phrase lexicon for symptoms, stressors and coping.
2. **Safety layer (no LLM):** crisis lexicon tiers, Hindi/Hinglish crisis phrases, a TF-IDF crisis classifier
   (test AUC 0.92, `src/train_crisis_classifier.py`) and PHQ-9 item 9 → `none / watch / high / urgent`,
   with Tele-MANAS 14416 on escalation.
3. **Questionnaires (no LLM):** PHQ-9 / GAD-7 totals, severity bands, change over time.
4. **LLM (open weights, JSON-schema-constrained):** writes only the narrative of the card from a cue digest.
   Output is Pydantic-validated and guardrailed (no diagnosis, treatment advice or risk talk; only real entry
   ids, stressor domains and numbers), retried once, with a rule-based fallback.

```bash
pip install -r requirements-phase1.txt
python src/train_crisis_classifier.py
ollama pull qwen3.5
python src/run_phase1.py demo                       # outputs/phase1/cards/*.md and *.json
python src/run_phase1.py analyze --text "neend nahi aati, exams ka tension"
uvicorn api:app --app-dir src --port 8080           # REST API
python -m pytest tests
```

## MitraMind Care: companion, clinician dashboard, Phases 2-3

`src/care/` builds the product around Phase 1 (details in [docs/CARE.md](docs/CARE.md), build log and open
work in [task.md](task.md)):

- **Companion (chat + voice):** every patient message goes through the Phase 1 safety layer first. High or
  urgent risk gets a fixed crisis reply with Tele-MANAS 14416 and 112 and an alert, and never reaches the LLM.
  Other messages get an LLM reply with guardrails, plus Mind Pause exercises and Reflection Share. Voice
  messages are transcribed by local Whisper and also scored by the voice emotion models.
- **Clinician side:** alert queue, dashboard (scores, usage, open alerts), pre-visit summary from the stored
  chats, summary ratings. All of it is gated by the patient's consent (DPDP), and every AI reply is audit-logged.
- **Phase 2 / 3:** de-identified pilot export linking message cues to PHQ-9 / GAD-7 change, a synthetic
  pilot simulator, and a small deterioration model trainer with patient-grouped CV.
- **Integrations:** WhatsApp Business webhook (chat, voice notes, PHQ-9 / GAD-7) and a FHIR R4 export.

```bash
uvicorn api:app --app-dir src --port 8080   # / patient app, /clinician dashboard, /admin, /tester, /docs
python src/redteam_companion.py --models qwen3.5
python src/simulate_pilot.py && python src/export_pilot_dataset.py --db var/pilot_sim.db --out outputs/phase2_synthetic --source synthetic
python src/train_outcome_model.py --data outputs/phase2_synthetic
```

## Datasets

| Dataset | Type | Licence / access | Used as |
|---|---|---|---|
| EmoInHindi (IIT Patna, LREC 2022) | Hindi dialogue CSV, ~44k utterances, multi-label (16 emotions) + intensity 0–3 | Agreement form: <https://github.com/priyanshu-profile/EmoInHindi> | Loader and trainer ready (`--emoinhindi_csv`); not yet available |
| HinEmo-9 (2026) | 30,436 YouTube comments, JSONL, 9 emotions, English / Devanagari Hindi / romanized Hinglish | CC BY 4.0 (annotations); comment text stays under YouTube's terms | m1 stand-in, m2 |
| CREMA-D | 7,442 WAV clips, 16 kHz, 91 actors, 6 emotions | ODbL / DbCL | m3 training, m3 + m4 evaluation |
| MSP-Podcast | Natural podcast speech, 8 emotions + arousal/valence/dominance | UT Dallas licence (academic; commercial is paid) | Not downloaded; m4 uses the lab's MIT-licensed models trained on it |
| Dreaddit (Turcan & McKeown 2019) | 3,553 Reddit segments, stress label, LIWC/DAL columns | No licence stated: research use | m5 validation |
| GoEmotions (Demszky et al. 2020) | 54,263 Reddit comments, 28 emotions | Apache-2.0 | m5 validation |
| Reddit Mental Health Dataset (Low et al. 2020) | 161,125 Jan–Apr 2020 posts, 21 subreddits, LIWC/lexicon columns | ODC-PDDL, subject to Reddit API terms | m5 validation; crisis classifier training |

Data checks:
- HinEmo-9 files match the authors' SHA-256 checksums (the authors computed them with Windows line endings).
- Every CREMA-D WAV comes from the Hugging Face mirror `MahiA/CREMA-D` and was checked against the
  SHA-256 in the official GitHub repo's LFS pointer. Clips that fail are listed in `data/crema_d/excluded_clips.csv`.

## How to run

This repo holds code and results only. Datasets and trained weights are not in Git. To run the app you can
download the trained weights (below); to reproduce them, re-download the data and retrain (steps 1-3, about
1.5 hours on a 4 GB RTX 3050).

**Download the trained models (m1-m3)** instead of training them. They are published as a GitHub Release
([models-v1](https://github.com/Aadish2423/mitramind-cue-models/releases/tag/models-v1), about 2.6 GB). The script
uses only Python's standard library, checks each file's SHA-256 before unpacking, and fills `models/`:

```bash
python src/download_models.py                    # all three
python src/download_models.py m3_cremad_speech   # or only the ones you need
```

Using them needs PyTorch (step 1). Without the weights or PyTorch the app still runs: the Phase 1 pipeline falls
back to the lexicon cues (m5) and lists the missing models under `models_unavailable` in `/health`.

**1. Environment** (Python 3.11). On Windows, keep the virtual environment at a short path such as
`C:\venvs\mitramind_cues`: Windows cannot load DLLs from paths longer than 260 characters.

```bash
python -m venv /c/venvs/mitramind_cues
PY=/c/venvs/mitramind_cues/Scripts/python.exe
$PY -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install -r requirements.txt
```

**2. Data** (HinEmo-9 text, and the official CREMA-D repo zip that holds each WAV's SHA-256):

```bash
mkdir -p data/hinemo9 data/crema_d_official
for f in train dev test gold; do
  curl -L -o data/hinemo9/$f.jsonl https://raw.githubusercontent.com/im-rakeshsahani/hinemo9/main/data/$f.jsonl
done
curl -L -o data/crema_d_official/crema_master.zip https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip
```

**3. Train, run and merge:**

```bash
$PY -I src/prepare_cremad.py              # download + verify CREMA-D audio, build actor split
$PY -I -X utf8 src/train_text.py --model m2
$PY -I -X utf8 src/train_text.py --model m1
$PY -I src/train_cremad.py
$PY -I src/run_msp.py
$PY -I src/merge_cues.py                  # writes outputs/cues_all.csv and outputs/model_scores.csv
```

When EmoInHindi access arrives, retrain model 1 on it (dialogue-level split, multi-label + intensity):

```bash
$PY -I -X utf8 src/train_text.py --model m1 --emoinhindi_csv path/to/LREC_EmoInHindi.csv
```

The loader expects the columns `dialogueId, utterance_no, authorRole, utterance, emotions, emoIntensity`
(comma-separated emotions and intensities), as seen in public notebooks that use the dataset. This path has
been tested on a fake file in that format, not on the real data.

## Output: `outputs/cues_all.csv`

One row per held-out sample per model (Excel-friendly UTF-8).

| Column | Meaning |
|---|---|
| `model_id`, `train_data`, `eval_data` | Which model, what it learned from, where the sample came from |
| `sample_id`, `modality`, `language_or_speaker`, `input` | The sample: text itself, or the WAV path |
| `true_label`, `true_cue` | Ground truth in the dataset's own labels, and as a MitraMind cue |
| `pred_label`, `pred_confidence`, `top3` | The model's native prediction |
| `pred_cue`, `p_anxious` … `p_neutral` | Prediction folded into MitraMind cues |
| `distress_score` | P(anxious or low_mood or angry), 0–1 |
| `arousal`, `valence`, `dominance` | m4 only, roughly 0–1 |
| `correct` | 1 if the prediction matches; for m4 (different label set) the cue must match |

Per-model files (`outputs/<model>_cues.csv`, `outputs/<model>_metrics.json`) hold the same rows plus
per-class scores, confusion matrices and training history. The label → cue mapping is `CUE_MAP` in `src/common.py`.

## Limits to keep in mind

- These are cue extractors for a wellness companion, not diagnostic tools. Validated scales (PHQ-9, GAD-7) stay the measure.
- Model 1 is a stand-in: HinEmo-9 is YouTube comments, not help-seeking dialogue. Retrain on EmoInHindi when you get it.
- CREMA-D is acted English speech. Expect lower accuracy on real patients, Indian accents and phone microphones.
- m1 misses most Devanagari fear: HinEmo-9 has only 105 Devanagari Fear comments in its training split.
  EmoInHindi has far more `fear`/`apprehensive` examples in Hindi, so retraining on it is the first fix.
- m4's categories barely transfer to acted speech: it labels most clips Sad, Neutral or Angry. Its arousal score does
  transfer (mean arousal ANG 0.66 > HAP 0.48 > FEA 0.41 > DIS 0.34 > NEU 0.29 > SAD 0.21), so use m4 for arousal and
  m3 for the emotion category. Valence barely separates emotions here (0.34–0.45).
- `Nostalgia` maps to `low_mood`, which gives fond "old is gold" comments a high distress score. Change it in
  `CUE_MAP` if you would rather treat nostalgia as neutral or positive.
- Some HinEmo-9 rows labelled "Hindi" contain romanized text, so m1's test set is not purely Devanagari.
- Licences: HinEmo-9's text and the MSP-Podcast-trained weights carry conditions from their sources; check them before commercial use.

## Credits

The `outputs/*_cues.csv` files reproduce HinEmo-9 test comments (PII-masked by its authors) under CC BY 4.0.

- HinEmo-9: R. Sahani and F. A, *HinEmo-9: An Emotion Dataset for Code-Mixed Hindi–English Across Two Scripts*, 2026 (under review). <https://github.com/im-rakeshsahani/hinemo9>, CC BY 4.0.
- CREMA-D: H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, R. Verma, *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*, IEEE Trans. Affective Computing 5(4), 2014. ODbL.
- MSP-Podcast baselines: L. Goncalves et al., *Odyssey 2024 – Speech Emotion Recognition Challenge: Dataset, Baseline Framework, and Results*, Odyssey 2024. Models: `3loi/SER-Odyssey-Baseline-WavLM-Categorical`, `3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes` (MIT).
- EmoInHindi: G. V. Singh, P. Priya, M. Firdaus, A. Ekbal, P. Bhattacharyya, *EmoInHindi*, LREC 2022.
- Base models: XLM-RoBERTa (`xlm-roberta-base`), WavLM (`microsoft/wavlm-base-plus`, `microsoft/wavlm-large`).

## Licence

The code is released under the [MIT licence](LICENSE). Data reproduced in `outputs/` keeps its source licence:
HinEmo-9 comment text is CC BY 4.0, and CREMA-D-derived files follow ODbL / DbCL. Lexicon portions adapted
from Low et al. 2020 (Apache-2.0) and Bathina et al. 2021 are credited in [NOTICE](NOTICE). The m5 output
CSVs contain only derived cues and hashed row ids, no Reddit text or usernames. The demo patients in
`examples/patients/` are fictional.
