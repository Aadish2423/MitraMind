"""Loaders for the three Phase 1 benchmark datasets.

Every loader returns a DataFrame with a `text` column, a `split` column and the
dataset's own gold labels. Raw downloads are cached under data/lexicon_benchmarks/ as parquet so
repeated runs are offline.

Sources (see reports/Mental health cue extraction tooling.md for licences):
  Dreaddit    HF andreagasparini/dreaddit                    (no licence stated: research use)
  GoEmotions  HF google-research-datasets/go_emotions        (Apache-2.0)
  Reddit MH   Zenodo record 3941387, Low et al. 2020         (ODC-PDDL, subject to Reddit API terms)
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "lexicon_benchmarks"

GOEMO_MAPPING_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/goemotions/data/{name}_mapping.json"
)
ZENODO_URL = "https://zenodo.org/api/records/3941387/files/{name}/content"

# Low et al. 2020 subreddits: 15 support groups + 2 broad, plus a few small controls.
RMHD_MENTAL_HEALTH = [
    "EDAnonymous", "addiction", "alcoholism", "adhd", "anxiety", "autism",
    "bipolarreddit", "bpd", "depression", "healthanxiety", "lonely", "ptsd",
    "schizophrenia", "socialanxiety", "suicidewatch",
]
RMHD_BROAD = ["mentalhealth", "COVID19_support"]
RMHD_CONTROL = ["fitness", "meditation", "parenting", "teaching"]


def _cached(name: str, build) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    return df


# --------------------------------------------------------------------------- Dreaddit
def load_dreaddit() -> pd.DataFrame:
    def build():
        from datasets import load_dataset

        ds = load_dataset("andreagasparini/dreaddit")
        parts = []
        for split in ds:
            part = ds[split].to_pandas()
            part.insert(0, "split", split)
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    return _cached("dreaddit", build)


# ------------------------------------------------------------------------- GoEmotions
def _goemo_mapping(name: str) -> dict[str, list[str]]:
    path = RAW / f"goemotions_{name}_mapping.json"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(GOEMO_MAPPING_URL.format(name=name), timeout=30)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def load_goemotions() -> pd.DataFrame:
    def build():
        from datasets import load_dataset

        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
        names = ds["train"].features["labels"].feature.names
        parts = []
        for split in ds:
            part = ds[split].to_pandas()
            part.insert(0, "split", split)
            part["label_names"] = part["labels"].map(lambda ix: "|".join(names[i] for i in ix))
            parts.append(part.drop(columns=["labels"]))
        return pd.concat(parts, ignore_index=True)

    df = _cached("goemotions", build)
    ekman = {e: g for g, emos in _goemo_mapping("ekman").items() for e in emos}
    senti = {e: g for g, emos in _goemo_mapping("sentiment").items() for e in emos}
    ekman["neutral"] = senti["neutral"] = "neutral"

    def groups(labels: str, table: dict[str, str]) -> str:
        return "|".join(sorted({table[l] for l in labels.split("|")}))

    df["ekman"] = df["label_names"].map(lambda s: groups(s, ekman))
    df["sentiment_group"] = df["label_names"].map(lambda s: groups(s, senti))
    # GoEmotions masks names and religions; give the tokenizer a neutral word instead.
    df["text"] = df["text"].str.replace(r"\[(NAME|RELIGION)\]", "someone", regex=True)
    return df


# -------------------------------------------------------------------------- Reddit MH
def _download(name: str, dest: pathlib.Path) -> None:
    tmp = dest.with_suffix(".part")
    with requests.get(ZENODO_URL.format(name=name), stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def load_reddit_mh(subreddits: list[str] | None = None) -> pd.DataFrame:
    subs = subreddits or RMHD_MENTAL_HEALTH + RMHD_BROAD + RMHD_CONTROL

    def build():
        folder = RAW / "rmhd"
        folder.mkdir(parents=True, exist_ok=True)
        parts = []
        for sub in subs:
            name = f"{sub}_post_features_tfidf_256.csv"
            dest = folder / name
            if not dest.exists():
                print(f"  downloading {name}")
                _download(name, dest)
            # tf-idf columns are 3/4 of the file and unused here.
            part = pd.read_csv(dest, usecols=lambda c: not c.startswith("tfidf_"))
            part = part.drop(columns=[c for c in ("author",) if c in part.columns])
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        df = df.rename(columns={"post": "text"})
        df["subreddit"] = df["subreddit"].astype(str)
        lower = {s.lower(): s for s in subs}
        df["subreddit"] = df["subreddit"].map(lambda s: lower.get(s.lower(), s))
        df["group"] = df["subreddit"].map(
            lambda s: "mental_health" if s in RMHD_MENTAL_HEALTH
            else "broad" if s in RMHD_BROAD else "control"
        )
        df.insert(0, "split", "post_2020")
        df["text"] = df["text"].fillna("").astype(str)
        return df[~df["text"].str.fullmatch(r"\s*(\[removed\]|\[deleted\])?\s*")].reset_index(drop=True)

    return _cached("reddit_mh", build)


LOADERS = {
    "dreaddit": load_dreaddit,
    "goemotions": load_goemotions,
    "reddit_mh": load_reddit_mh,
}

# Gold-label / metadata columns copied into each output CSV (never author fields).
LABEL_COLUMNS = {
    "dreaddit": ["subreddit", "label", "confidence"],
    "goemotions": ["label_names", "ekman", "sentiment_group"],
    "reddit_mh": ["subreddit", "group", "date"],
}
ID_COLUMNS = {"dreaddit": "id", "goemotions": "id", "reddit_mh": None}
