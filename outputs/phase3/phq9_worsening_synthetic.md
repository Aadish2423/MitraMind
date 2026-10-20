# Phase 3: PHQ-9 deterioration model (synthetic data)

**PIPELINE TEST ONLY: trained on synthetic data.**

1032 windows from 172 patients, 80 worsened (prevalence 0.078).
Patient-grouped 5-fold cross-validation (no patient in both train and test).

| Model | AUROC | AUPRC | Brier |
|---|---|---|---|
| baseline_score_only | 0.516 | 0.075 | 0.0717 |
| logistic_all | 0.736 | 0.214 | 0.1867 |
| boosted_trees_all | 0.769 | 0.187 | 0.1136 |

AUPRC of a random guess equals the prevalence (0.078). A model is only worth using if it beats
`baseline_score_only` clearly, on real pilot data.

Score-change regression (MAE, points): ridge 2.303 vs.
"no change" 2.436.

## Strongest logistic-regression features (standardised coefficients)

| Feature | Coefficient |
|---|---|
| baseline_total | -1.656 |
| cope_breathing | -0.925 |
| cope_social_support | -0.617 |
| gad_nervous | +0.597 |
| mind_pause_started | +0.386 |
| str_family | -0.373 |
| gad_irritable | +0.336 |
| phq_concentration | +0.321 |
| mind_pause_completed | -0.308 |
| phq_worthlessness_guilt | +0.285 |
| str_school_exam | -0.271 |
| phq_fatigue | +0.268 |

## Alert precision from clinician labels

| Level / source | Labelled alerts | Share that were real risk |
|---|---|---|
| watch/message | 154 | 0.851 |

Use this table with the clinical champion to move the crisis-layer thresholds: a level whose alerts are
rarely real wastes staff time; never lower sensitivity for `high` / `urgent` on precision alone.
