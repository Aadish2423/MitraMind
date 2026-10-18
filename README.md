# MitraMind phase-1 cue models

Four models, one per dataset, that turn text or a voice clip into MitraMind cues
(`anxious`, `low_mood`, `angry`, `positive`, `surprised`, `neutral`) plus a 0–1 `distress_score`.
Every held-out prediction is stored in `outputs/cues_all.csv`.

| Model | Input | Trained on | Tested on | Base |
|---|---|---|---|---|
| `m1_hindi_text` | Hindi text (Devanagari) | HinEmo-9 Hindi + English rows (**stand-in for EmoInHindi**) | HinEmo-9 human-verified Hindi test rows | XLM-R base |
| `m2_hinglish_text` | Hinglish text (romanized) | HinEmo-9 Hinglish + English rows | HinEmo-9 human-verified Hinglish test rows | XLM-R base |
| `m3_cremad_speech` | Voice clip | CREMA-D, 63 actors (9 more for model selection) | CREMA-D, 19 actors never heard in training | WavLM-base-plus |
| `m4_msp_speech` | Voice clip | MSP-Podcast (MSP lab's released model, not retrained) | Same unseen CREMA-D actors as m3 | WavLM-large |

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

## Datasets

| Dataset | Type | Licence / access | Used as |
|---|---|---|---|
| EmoInHindi (IIT Patna, LREC 2022) | Hindi dialogue CSV, ~44k utterances, multi-label (16 emotions) + intensity 0–3 | Agreement form: <https://github.com/priyanshu-profile/EmoInHindi> | Loader and trainer ready (`--emoinhindi_csv`); not yet available |
| HinEmo-9 (2026) | 30,436 YouTube comments, JSONL, 9 emotions, English / Devanagari Hindi / romanized Hinglish | CC BY 4.0 (annotations); comment text stays under YouTube's terms | m1 stand-in, m2 |
| CREMA-D | 7,442 WAV clips, 16 kHz, 91 actors, 6 emotions | ODbL / DbCL | m3 training, m3 + m4 evaluation |
| MSP-Podcast | Natural podcast speech, 8 emotions + arousal/valence/dominance | UT Dallas licence (academic; commercial is paid) | Not downloaded; m4 uses the lab's MIT-licensed models trained on it |

Data checks:
- HinEmo-9 files match the authors' SHA-256 checksums (the authors computed them with Windows line endings).
- Every CREMA-D WAV comes from the Hugging Face mirror `MahiA/CREMA-D` and was checked against the
  SHA-256 in the official GitHub repo's LFS pointer. Clips that fail are listed in `data/crema_d/excluded_clips.csv`.

## How to run

This repo holds code and results only. Datasets and trained weights are not included; the steps below
re-download the data and retrain (about 1.5 hours on a 4 GB RTX 3050).

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
