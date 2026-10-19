"""Model 5 (m5_lexicon_text): extract lexicon cues from the English benchmarks, one CSV per dataset.

    python src/extract_lexicon_cues.py                       # all three datasets
    python src/extract_lexicon_cues.py dreaddit goemotions   # a subset
    python src/extract_lexicon_cues.py reddit_mh --limit 2000 --include-text

Writes outputs/m5_lexicon_text/cues_<dataset>.csv (UTF-8; Reddit MH as .csv.gz because the
plain CSV is over GitHub's 100 MB limit). Rows keep the dataset's gold labels and a hashed
row id; author fields are never written and raw text only with --include-text.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd

from lexicon_cues.datasets import ID_COLUMNS, LABEL_COLUMNS, LOADERS
from lexicon_cues.extractor import CueExtractor, default_workers

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "m5_lexicon_text"
COMPRESSED = {"reddit_mh"}


def output_path(dataset: str) -> pathlib.Path:
    return OUT / f"cues_{dataset}.csv{'.gz' if dataset in COMPRESSED else ''}"


def row_ids(df: pd.DataFrame, dataset: str) -> pd.Series:
    col = ID_COLUMNS[dataset]
    keys = df[col].astype(str) if col else df.index.astype(str) + "|" + df["text"].str[:200]
    return keys.map(lambda k: hashlib.sha1(f"{dataset}|{k}".encode()).hexdigest()[:16])


def run(dataset: str, extractor: CueExtractor, workers: int, limit: int | None,
        include_text: bool) -> pathlib.Path:
    t0 = time.time()
    df = LOADERS[dataset]()
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0).sort_index()
    print(f"[{dataset}] {len(df):,} rows, extracting with {workers} worker(s)...")
    cues = pd.DataFrame(extractor.extract_many(df["text"].tolist(), workers=workers), index=df.index)

    meta = pd.DataFrame({
        "dataset": dataset,
        "split": df["split"],
        "row_idx": df.index,
        "row_id": row_ids(df, dataset),
    })
    labels = df[[c for c in LABEL_COLUMNS[dataset] if c in df.columns]]
    parts = [meta, labels] + ([df[["text"]]] if include_text else []) + [cues]
    out = pd.concat(parts, axis=1)

    OUT.mkdir(parents=True, exist_ok=True)
    path = output_path(dataset)
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"[{dataset}] wrote {path.name}: {out.shape[0]:,} rows x {out.shape[1]} cols "
          f"in {time.time() - t0:.0f}s")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"any of {list(LOADERS)} (default: all)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--limit", type=int, help="random sample of N rows per dataset (for quick runs)")
    ap.add_argument("--include-text", action="store_true", help="also write the raw text column")
    ap.add_argument("--no-empath", action="store_true", help="skip Empath emotion categories")
    args = ap.parse_args()
    if unknown := set(args.datasets) - set(LOADERS):
        ap.error(f"unknown dataset(s) {sorted(unknown)}; choose from {list(LOADERS)}")

    extractor = CueExtractor(use_empath=not args.no_empath)
    for name in args.datasets or list(LOADERS):
        run(name, extractor, args.workers, args.limit, args.include_text)


if __name__ == "__main__":
    main()
