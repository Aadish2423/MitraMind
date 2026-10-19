"""Sanity-check the cue CSVs against each dataset's labels and its precomputed features.

    python src/validate_lexicon_cues.py    # writes outputs/m5_lexicon_text/validation_report.md

Dreaddit    cue vs stress label (AUC) + agreement with the shipped LIWC2015 / DAL columns
GoEmotions  VADER vs sentiment groups, Empath vs Ekman groups, clinical cues vs labels
Reddit MH   agreement with Low et al.'s shipped lexicon/LIWC/VADER columns, and
            subreddit profiles (e.g. does r/suicidewatch top the risk cues?)

These are face-validity checks of the lexicons on public data, not clinical validation.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from extract_lexicon_cues import output_path
from lexicon_cues.datasets import LOADERS

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "m5_lexicon_text"


# ----------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_pair(name: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path = output_path(name)
    if not path.exists():
        return None
    cues = pd.read_csv(path, low_memory=False)
    raw = LOADERS[name]().loc[cues["row_idx"]].reset_index(drop=True)
    return cues, raw


def numeric_cues(cues: pd.DataFrame) -> list[str]:
    return [c for c in cues.columns
            if (c.startswith("cue_") or c == "n_words") and pd.api.types.is_numeric_dtype(cues[c])]


def auc(y, x) -> float:
    x = pd.Series(x).fillna(0)
    if len(set(y)) < 2 or x.nunique() < 2:
        return float("nan")
    return roc_auc_score(y, x)


def agreement(cues, raw, pairs) -> pd.DataFrame:
    rows = []
    for ours, theirs, note in pairs:
        a = cues[ours] if isinstance(ours, str) else cues[list(ours)].sum(axis=1)
        b = raw[theirs] if isinstance(theirs, str) else raw[list(theirs)].sum(axis=1)
        ok = a.notna() & b.notna()
        rows.append({
            "our cue": ours if isinstance(ours, str) else " + ".join(ours),
            "reference": theirs if isinstance(theirs, str) else " + ".join(theirs),
            "pearson": a[ok].corr(b[ok]),
            "spearman": a[ok].corr(b[ok], method="spearman"),
            "note": note,
        })
    return pd.DataFrame(rows)


def separation(cues, y, top=20) -> pd.DataFrame:
    rows = []
    for c in numeric_cues(cues):
        rows.append({"cue": c, "mean_pos": cues.loc[y == 1, c].mean(),
                     "mean_neg": cues.loc[y == 0, c].mean(), "AUC": auc(y, cues[c])})
    df = pd.DataFrame(rows).dropna(subset=["AUC"])
    df["strength"] = (df["AUC"] - 0.5).abs()
    return df.sort_values("strength", ascending=False).head(top).drop(columns="strength")


# ---------------------------------------------------------------------- Dreaddit
def dreaddit_section() -> str:
    pair = load_pair("dreaddit")
    if pair is None:
        return "## Dreaddit\n\n_cues_dreaddit.csv not found._\n"
    cues, raw = pair
    y = cues["label"].astype(int)
    pairs = [
        ("cue_self_fps_rate", "lex_liwc_i", "first-person singular"),
        ("cue_self_fpp_rate", "lex_liwc_we", "first-person plural"),
        ("cue_self_second_rate", "lex_liwc_you", "second person"),
        ("cue_self_third_rate", ("lex_liwc_shehe", "lex_liwc_they"), "third person"),
        ("cue_ling_negation_rate", "lex_liwc_negate", "negation"),
        ("cue_ling_certainty_rate", "lex_liwc_certain", "certainty"),
        ("cue_ling_tentative_rate", "lex_liwc_tentat", "tentative"),
        ("cue_ling_past_focus_rate", "lex_liwc_focuspast", "past focus"),
        ("cue_ling_future_focus_rate", "lex_liwc_focusfuture", "future focus"),
        ("cue_sent_compound", "sentiment", "Pattern sentiment shipped with Dreaddit"),
        ("cue_sent_compound", "lex_liwc_Tone", "LIWC emotional tone"),
        ("cue_sent_neg", "lex_liwc_negemo", "negative emotion"),
        ("cue_sent_pos", "lex_liwc_posemo", "positive emotion"),
        ("cue_emo_sadness", "lex_liwc_sad", "sadness"),
        ("cue_emo_nervousness", "lex_liwc_anx", "anxiety (Empath)"),
        ("cue_gad_rate", "lex_liwc_anx", "anxiety (GAD-7 lexicon)"),
        ("cue_emo_anger", "lex_liwc_anger", "anger"),
        ("cue_str_family", "lex_liwc_family", "family"),
        ("cue_str_economic", "lex_liwc_money", "money"),
        ("cue_str_work", "lex_liwc_work", "work"),
        ("cue_str_health", "lex_liwc_health", "health"),
        ("cue_risk_total", "lex_liwc_death", "death / crisis"),
        ("cue_style_fk_grade", "syntax_fk_grade", "Flesch-Kincaid grade"),
        ("cue_style_ari", "syntax_ari", "ARI"),
        ("n_words", "lex_liwc_WC", "word count"),
    ]
    pairs = [p for p in pairs if all(c in raw.columns for c in ([p[1]] if isinstance(p[1], str) else p[1]))]
    fps = cues.groupby(y)["cue_self_fps_rate"].mean()
    liwc_i = raw.groupby(y)["lex_liwc_i"].mean()
    return "\n\n".join([
        "## Dreaddit (stress vs non-stress, n = %d)" % len(cues),
        "**Agreement with the LIWC2015 / syntax columns shipped in the dataset.** "
        "Our open word lists stand in for proprietary LIWC; high correlation means the "
        "open proxy captures the same construct.",
        md_table(agreement(cues, raw, pairs)),
        f"**Paper check.** Turcan & McKeown report LIWC 'I' at 9.81% (stress) vs 6.53% (non-stress). "
        f"Shipped LIWC column here: {liwc_i.get(1, np.nan):.2f} vs {liwc_i.get(0, np.nan):.2f}; "
        f"our `cue_self_fps_rate`: {fps.get(1, np.nan):.2f} vs {fps.get(0, np.nan):.2f}.",
        "**Cues that best separate stress (label 1) from non-stress (label 0)**, by AUC "
        "(0.5 = no signal; below 0.5 = higher in non-stress).",
        md_table(separation(cues, y)),
    ])


# -------------------------------------------------------------------- GoEmotions
EKMAN_CUES = {
    "anger": ["cue_emo_anger", "cue_emo_rage", "cue_emo_hate"],
    "disgust": ["cue_emo_disgust"],
    "fear": ["cue_emo_fear", "cue_emo_nervousness"],
    "joy": ["cue_emo_joy", "cue_emo_positive_emotion", "cue_emo_optimism", "cue_emo_love"],
    "sadness": ["cue_emo_sadness", "cue_emo_suffering", "cue_emo_disappointment"],
    "surprise": ["cue_emo_surprise"],
}


def goemotions_section() -> str:
    pair = load_pair("goemotions")
    if pair is None:
        return "## GoEmotions\n\n_cues_goemotions.csv not found._\n"
    cues, _ = pair
    groups = cues["sentiment_group"].fillna("")
    single = groups.isin(["positive", "negative"])
    y_sent = (groups[single] == "positive").astype(int)
    sent_rows = [{"cue": c, "AUC (positive vs negative)": auc(y_sent, cues.loc[single, c])}
                 for c in ("cue_sent_compound", "cue_sent_pos", "cue_sent_neg", "cue_emo_positive_emotion",
                           "cue_emo_negative_emotion")]

    ekman_rows = []
    for g, cols in EKMAN_CUES.items():
        y = cues["ekman"].fillna("").str.split("|").map(lambda s: g in s).astype(int)
        score = cues[cols].sum(axis=1)
        ekman_rows.append({"Ekman group": g, "rows with label": int(y.sum()),
                           "cue": " + ".join(c.replace("cue_emo_", "") for c in cols),
                           "AUC": auc(y, score),
                           "fires on labelled rows": (score[y == 1] > 0).mean(),
                           "fires on other rows": (score[y == 0] > 0).mean()})

    labels = cues["label_names"].fillna("").str.split("|")
    clinical = [
        ("cue_gad_total", {"nervousness", "fear"}),
        ("cue_phq_depressed_mood", {"sadness", "grief", "disappointment"}),
        ("cue_phq_worthlessness_guilt", {"remorse", "embarrassment"}),
        ("cue_cd_total", {"disapproval", "annoyance"}),
    ]
    clin_rows = []
    for c, target in clinical:
        y = labels.map(lambda s: bool(target & set(s))).astype(int)
        clin_rows.append({"cue": c, "target labels": ", ".join(sorted(target)), "AUC": auc(y, cues[c]),
                          "fires on target": (cues.loc[y == 1, c] > 0).mean(),
                          "fires on others": (cues.loc[y == 0, c] > 0).mean()})
    risk = cues["cue_risk_tier_name"].value_counts().rename_axis("risk tier").reset_index(name="rows")
    return "\n\n".join([
        "## GoEmotions (n = %d short Reddit comments, 28 labels)" % len(cues),
        "**Affect polarity vs the official sentiment mapping** (rows whose labels are all positive or all negative).",
        md_table(pd.DataFrame(sent_rows)),
        "**Empath emotion categories vs the official Ekman mapping.** Comments average about 13 words, so "
        "lexicon hit rates are low; AUC is the fairer measure.",
        md_table(pd.DataFrame(ekman_rows)),
        "**Clinical lexicons vs related emotion labels** (weak proxy: GoEmotions is not clinical).",
        md_table(pd.DataFrame(clin_rows)),
        "**Risk-tier distribution** (expected to be rare on general comments).",
        md_table(risk),
    ])


# --------------------------------------------------------------------- Reddit MH
EXPECTED_TOP = [
    ("cue_risk_tier", "suicidewatch"),
    ("cue_risk_rate", "suicidewatch"),
    ("cue_phq_rate", "depression"),
    ("cue_gad_rate", "anxiety"),
    ("cue_gad_panic_somatic", "healthanxiety"),
    ("cue_sleep_rate", "bipolarreddit"),
    ("cue_str_substance_use", "alcoholism"),
    ("cue_str_isolation", "lonely"),
    ("cue_str_covid19", "COVID19_support"),
    ("cue_str_domestic", "ptsd"),
    ("cue_phq_appetite", "EDAnonymous"),
    ("cue_phq_concentration", "adhd"),
    ("cue_self_fps_rate", "depression"),
    ("cue_ling_absolutist_rate", "suicidewatch"),
]


def reddit_section() -> str:
    pair = load_pair("reddit_mh")
    if pair is None:
        return "## Reddit Mental Health (Low et al.)\n\n_cues_reddit_mh.csv not found._\n"
    cues, raw = pair
    # Low et al.'s liwc_* columns are raw counts (not % of words), and their "1st_pers" is
    # LIWC2007 first-person plural. Compare counts with counts.
    for c in ("cue_self_fps_rate", "cue_self_fpp_rate", "cue_ling_negation_rate", "cue_ling_certainty_rate"):
        cues[c.replace("_rate", "_count")] = cues[c] * cues["n_words"] / 100
    pairs = [
        ("cue_str_economic", "economic_stress_total", "Low lexicon, now whole-word"),
        ("cue_str_isolation", "isolation_total", "Low lexicon, concat bug fixed"),
        ("cue_str_substance_use", "substance_use_total", "Low lexicon + extras"),
        ("cue_str_domestic", "domestic_stress_total", "Low lexicon + extras"),
        ("cue_risk_means", "guns_total", "guns folded into means vocabulary"),
        ("cue_risk_total", "suicidality_total", "tiered, negation + first-person scoped"),
        ("cue_sent_compound", "sent_compound", "same VADER lexicon, different tokenisation"),
        ("cue_sent_neg", "sent_neg", ""),
        ("cue_self_fpp_count", "liwc_1st_pers", "LIWC2007 'we' (named 1st_pers), counts"),
        ("cue_self_fps_count", "liwc_personal_pronouns", "LIWC personal pronouns, counts"),
        ("cue_ling_negation_count", "liwc_negations", "LIWC negations, counts"),
        ("cue_ling_certainty_count", "liwc_certainty", "LIWC certainty, counts"),
        ("cue_gad_total", "liwc_anxiety", "LIWC anxiety, counts"),
        ("cue_str_family", "liwc_family", "LIWC family, counts"),
        ("cue_str_economic", "liwc_money", "LIWC money, counts"),
        ("cue_str_health", "liwc_health", "LIWC health, counts"),
        ("cue_style_fk_grade", "flesch_kincaid_grade_level", "textacy F-K"),
        ("cue_style_gunning_fog", "gunning_fog_index", "textacy fog"),
        ("n_words", "n_words", "word count"),
    ]
    pairs = [p for p in pairs if p[1] in raw.columns]

    profile_cols = ["cue_risk_rate", "cue_phq_rate", "cue_gad_rate", "cue_sleep_rate", "cue_cd_rate",
                    "cue_self_fps_rate", "cue_ling_absolutist_rate", "cue_sent_compound"]
    prof = cues.groupby("subreddit")[profile_cols].mean()
    prof.insert(0, "% risk>=ideation", cues.assign(r=cues["cue_risk_tier"] >= 2).groupby("subreddit")["r"].mean() * 100)
    prof.insert(0, "posts", cues.groupby("subreddit").size())
    prof = prof.sort_values("cue_risk_rate", ascending=False).reset_index()

    rank_rows = []
    means_all = cues.groupby("subreddit")[[c for c, _ in EXPECTED_TOP]].mean()
    for c, expected in EXPECTED_TOP:
        order = means_all[c].sort_values(ascending=False)
        rank = list(order.index).index(expected) + 1 if expected in order.index else None
        rank_rows.append({"cue": c, "top subreddit": order.index[0], "expected": expected,
                          "expected rank": f"{rank}/{len(order)}" if rank else "n/a"})

    y = (cues["group"] == "mental_health").astype(int)
    sep = separation(cues.loc[cues["group"] != "broad"], y[cues["group"] != "broad"], top=15)
    return "\n\n".join([
        "## Reddit Mental Health Dataset (Low et al. 2020, Jan-Apr 2020, n = %d posts)" % len(cues),
        "**Agreement with features Low et al. shipped in the files.** The `*_total` lexicon columns were "
        "computed with raw substring counting, so divergence is expected where that bug bites.",
        md_table(agreement(cues, raw, pairs)),
        "**Does the expected community rank top for each cue?** (mean per subreddit)",
        md_table(pd.DataFrame(rank_rows)),
        "**Subreddit cue profiles**, sorted by crisis-cue rate.",
        md_table(prof, floatfmt="{:.2f}"),
        "**Mental-health subreddits vs control subreddits** (fitness, meditation, parenting, teaching): "
        "strongest separating cues by AUC.",
        md_table(sep),
    ])


def main() -> None:
    sections = [
        "# Phase 1 cue validation report",
        "Generated by `validate_cues.py`. Face-validity checks of the open lexicons on public data. "
        "None of this is clinical validation; labels are stress annotations, emotion annotations and "
        "subreddit membership, not diagnoses.",
        dreaddit_section(), goemotions_section(), reddit_section(),
    ]
    OUT.mkdir(exist_ok=True)
    path = OUT / "validation_report.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
