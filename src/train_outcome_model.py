"""Phase 3: train small, cheap models on the Phase 2 pilot dataset.

1. Deterioration model: will the next PHQ-9 (or GAD-7) be clinically worse (>= 5 / >= 4 points), given
   the score at the start of the window and what the patient's companion messages showed during it?
   Patient-grouped 5-fold cross-validation; compared with a baseline that only knows the starting score.
2. Alert precision: of the alerts clinicians resolved with an outcome label, how many were real, per
   level and source. This is the evidence for moving the crisis-layer thresholds.

The models are tiny (logistic regression / shallow boosted trees on ~70 numbers), run on any CPU and
can be hosted inside the hospital. They are research outputs: nothing here is shown to clinicians
until a model trained on real pilot data has been validated and approved.

    python src/train_outcome_model.py --data outputs/phase2_synthetic
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["next_total", "change", "worsened", "improved"]
IDS = ["patient", "instrument", "window", "start_day"]


def candidates():
    return {
        "baseline_score_only": (["baseline_total"], make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))),
        "logistic_all": (None, make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"))),
        "boosted_trees_all": (None, make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                           class_weight="balanced", random_state=0))),
    }


def oof(model, X, y, groups, k=5):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred[te] = model.predict_proba(X.iloc[te])[:, 1]
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder written by src/export_pilot_dataset.py")
    ap.add_argument("--instrument", choices=["PHQ-9", "GAD-7"], default="PHQ-9")
    ap.add_argument("--models-dir", default=str(ROOT / "models" / "outcome_model"))
    ap.add_argument("--report-dir", default=str(ROOT / "outputs" / "phase3"))
    args = ap.parse_args()

    data = Path(args.data)
    meta = json.loads((data / "meta.json").read_text())
    source = meta["source"]
    df = pd.read_csv(data / "windows.csv")
    df = df[df["instrument"] == args.instrument].reset_index(drop=True)
    feats = [c for c in df.columns if c not in IDS + LABELS and df[c].notna().any()]
    X, y, groups = df[feats], df["worsened"], df["patient"]
    n_pos, n_pat = int(y.sum()), groups.nunique()
    if n_pos < 10 or n_pat < 10:
        raise SystemExit(f"too little data to train: {n_pos} worsened windows, {n_pat} patients")

    results, preds = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (cols, model) in candidates().items():
            p = oof(model, X[cols] if cols else X, y, groups)
            preds[name] = p
            results[name] = {"auroc": round(roc_auc_score(y, p), 3), "auprc": round(average_precision_score(y, p), 3),
                             "brier": round(brier_score_loss(y, p), 4)}
        ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
        change_pred = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            ridge.fit(X.iloc[tr], df["change"].iloc[tr])
            change_pred[te] = ridge.predict(X.iloc[te])
    change_res = {"mae_ridge": round(mean_absolute_error(df["change"], change_pred), 3),
                  "mae_predict_no_change": round(mean_absolute_error(df["change"], np.zeros(len(df))), 3)}

    best = max((n for n in results if n != "baseline_score_only"), key=lambda n: results[n]["auprc"])
    cols, model = candidates()[best]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    coefs = {}
    if best == "logistic_all":
        lr = model[-1]
        coefs = dict(sorted(zip(feats, lr.coef_[0].round(3)), key=lambda kv: -abs(kv[1]))[:12])

    alerts = pd.read_csv(data / "alert_outcomes.csv") if (data / "alert_outcomes.csv").exists() else pd.DataFrame()
    alert_precision = {}
    if len(alerts):
        lab = alerts[alerts["outcome"] != "unclear"]
        for (level, src), g in lab.groupby(["level", "source"]):
            alert_precision[f"{level}/{src}"] = {"labelled": len(g),
                                                 "precision": round((g["outcome"] == "confirmed_risk").mean(), 3)}

    out_models = Path(args.models_dir)
    out_models.mkdir(parents=True, exist_ok=True)
    tag = f"{args.instrument.lower().replace('-', '')}_worsening"
    joblib.dump({"model": model, "features": feats}, out_models / f"{tag}.joblib")
    info = {"task": f"{args.instrument} worsens by >= the meaningful-change threshold by the next administration",
            "trained_on": source, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "windows": len(df), "patients": int(n_pat), "worsened": n_pos, "best_model": best,
            "cv": results, "change_regression": change_res, "top_logistic_coefficients": coefs,
            "alert_precision": alert_precision,
            "use": ("PIPELINE TEST ONLY: trained on synthetic data" if source == "synthetic" else
                    "research only until externally validated and approved by the clinical team")}
    (out_models / f"{tag}.json").write_text(json.dumps(info, indent=2, default=str))
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{tag}_{source}.json").write_text(json.dumps(info, indent=2, default=str))
    (report_dir / f"{tag}_{source}.md").write_text(report(info, args.instrument), encoding="utf-8")
    print(json.dumps(info, indent=2, default=str))


def report(info: dict, instrument: str) -> str:
    rows = "\n".join(f"| {n} | {r['auroc']} | {r['auprc']} | {r['brier']} |" for n, r in info["cv"].items())
    coefs = "\n".join(f"| {k} | {v:+.3f} |" for k, v in info["top_logistic_coefficients"].items()) or "| - | - |"
    alerts = "\n".join(f"| {k} | {v['labelled']} | {v['precision']} |" for k, v in info["alert_precision"].items()) \
        or "| - | - | - |"
    prevalence = info["worsened"] / info["windows"]
    return f"""# Phase 3: {instrument} deterioration model ({info['trained_on']} data)

**{info['use']}.**

{info['windows']} windows from {info['patients']} patients, {info['worsened']} worsened (prevalence {prevalence:.3f}).
Patient-grouped 5-fold cross-validation (no patient in both train and test).

| Model | AUROC | AUPRC | Brier |
|---|---|---|---|
{rows}

AUPRC of a random guess equals the prevalence ({prevalence:.3f}). A model is only worth using if it beats
`baseline_score_only` clearly, on real pilot data.

Score-change regression (MAE, points): ridge {info['change_regression']['mae_ridge']} vs.
"no change" {info['change_regression']['mae_predict_no_change']}.

## Strongest logistic-regression features (standardised coefficients)

| Feature | Coefficient |
|---|---|
{coefs}

## Alert precision from clinician labels

| Level / source | Labelled alerts | Share that were real risk |
|---|---|---|
{alerts}

Use this table with the clinical champion to move the crisis-layer thresholds: a level whose alerts are
rarely real wastes staff time; never lower sensitivity for `high` / `urgent` on precision alone.
"""


if __name__ == "__main__":
    main()
