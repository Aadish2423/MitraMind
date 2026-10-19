"""Train the crisis-language classifier used by the safety layer (src/phase1/crisis.py).

Data: Low et al. 2020 Reddit Mental Health Dataset (Jan-Apr 2020 posts, cached by
src/extract_lexicon_cues.py in data/lexicon_benchmarks/reddit_mh.parquet).
Positive = r/SuicideWatch, negative = every other mental-health, broad and control subreddit,
so the model must separate suicidal language from general distress (r/depression etc.).
Model: TF-IDF word 1-2-grams + logistic regression (CPU, ~2 min). No class re-weighting, so near-empty
input scores near the 13% base rate instead of 0.5.

Thresholds: `watch` is set for 65% recall on the validation split (lexicon tiers and PHQ-9 item 9
carry the main load; the classifier is a second net, so it is kept fairly precise). `high` (classifier alone
escalating) is only set if a threshold reaches 75% precision at >= 20% recall; otherwise the
classifier can raise "watch" but never "high" by itself. Test-split metrics are reported in
outputs/phase1/crisis_classifier_metrics.json.
Caveat: subreddit membership is a weak proxy for risk, and posts are split at random (the
cached data has no author ids), so one author can appear in train and test.

    python src/train_crisis_classifier.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from lexicon_cues.datasets import load_reddit_mh

ROOT = Path(__file__).resolve().parents[1]
OUT_MODEL = ROOT / "models" / "crisis_classifier"
OUT_METRICS = ROOT / "outputs" / "phase1" / "crisis_classifier_metrics.json"


def at_threshold(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    return {"threshold": round(float(t), 4), "precision": round(tp / max(pred.sum(), 1), 4),
            "recall": round(tp / max((y == 1).sum(), 1), 4), "flag_rate": round(float(pred.mean()), 4)}


def main():
    df = load_reddit_mh()
    y = (df["subreddit"] == "suicidewatch").astype(int).to_numpy()
    X = df["text"].to_numpy()
    X_tr, X_rest, y_tr, y_rest, sub_tr, sub_rest = train_test_split(
        X, y, df["subreddit"].to_numpy(), test_size=0.3, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te, _, sub_te = train_test_split(
        X_rest, y_rest, sub_rest, test_size=0.5, stratify=y_rest, random_state=42)
    print(f"train {len(y_tr):,}  val {len(y_val):,}  test {len(y_te):,}  positive rate {y.mean():.3f}")

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True,
                        strip_accents="unicode"),
        LogisticRegression(C=4.0, max_iter=2000))
    model.fit(X_tr, y_tr)

    p_val = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    watch = float(thr[np.where(rec[:-1] >= 0.65)[0][-1]])
    # r/depression posts are often suicidal too, so precision saturates; only allow the classifier
    # to escalate to "high" on its own if some threshold reaches 75% precision at >= 20% recall.
    ok = np.where((prec[:-1] >= 0.75) & (rec[:-1] >= 0.20))[0]
    high = float(thr[ok[0]]) if len(ok) else None
    pr_table = [at_threshold(y_val, p_val, t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]

    p_te = model.predict_proba(X_te)[:, 1]
    pred_watch = p_te >= watch
    by_sub = {s: round(float(pred_watch[sub_te == s].mean()), 4) for s in sorted(set(sub_te))}
    metrics = {
        "data": "Low et al. 2020 Reddit MH, Jan-Apr 2020; positive = r/suicidewatch",
        "rows": {"train": len(y_tr), "val": len(y_val), "test": len(y_te)},
        "test_roc_auc": round(roc_auc_score(y_te, p_te), 4),
        "test_average_precision": round(average_precision_score(y_te, p_te), 4),
        "thresholds_from_validation": {"watch (65% recall)": round(watch, 4),
                                       "high (75% precision, >=20% recall)": None if high is None else round(high, 4)},
        "test_at_watch": at_threshold(y_te, p_te, watch),
        "test_at_high": None if high is None else at_threshold(y_te, p_te, high),
        "validation_precision_recall_table": pr_table,
        "test_share_flagged_watch_by_subreddit": by_sub,
    }
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_MODEL / "model.joblib", compress=3)
    (OUT_MODEL / "thresholds.json").write_text(json.dumps({"watch": watch, "high": high}))
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "validation_precision_recall_table"}, indent=2))


if __name__ == "__main__":
    main()
