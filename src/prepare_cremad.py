"""Download the CREMA-D WAVs, verify each against the official repo, and build a speaker-independent split.

The official GitHub repo stores audio in Git LFS, so its zip (data/crema_d_official/crema_master.zip)
holds a pointer per WAV with the file's SHA-256. The audio itself comes from the Hugging Face mirror
MahiA/CREMA-D; any file whose hash differs from the official pointer is dropped.

Output: data/crema_d/metadata.csv with one row per verified clip and a train/val/test split by actor,
so the test set contains only voices the model never heard during training.
"""
import argparse
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import snapshot_download

from common import DATA

OFFICIAL_ZIP = DATA / "crema_d_official" / "crema_master.zip"
WAV_DIR = DATA / "crema_d_wav"
META_DIR = DATA / "crema_d"
NAME_RE = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})\.wav$")


def official_index():
    z = zipfile.ZipFile(OFFICIAL_ZIP)
    hashes = {}
    for name in z.namelist():
        if "/AudioWAV/" in name and name.endswith(".wav"):
            pointer = z.read(name).decode()
            oid = re.search(r"oid sha256:([0-9a-f]{64})", pointer).group(1)
            size = int(re.search(r"size (\d+)", pointer).group(1))
            hashes[Path(name).name] = (oid, size)
    demo_name = next(n for n in z.namelist() if n.endswith("VideoDemographics.csv"))
    demographics = pd.read_csv(io.BytesIO(z.read(demo_name)))
    return hashes, demographics


def speaker_split(demographics, seed, test_frac=0.2, val_frac=0.1):
    """Assign whole actors to splits, stratified by sex so each split has a similar gender balance."""
    rng = np.random.default_rng(seed)
    split = {}
    for _, group in demographics.groupby("Sex"):
        actors = group["ActorID"].to_numpy().copy()
        rng.shuffle(actors)
        n_test, n_val = round(len(actors) * test_frac), round(len(actors) * val_frac)
        for i, actor in enumerate(actors):
            split[int(actor)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hashes, demographics = official_index()
    print(f"official index: {len(hashes)} WAV pointers, {len(demographics)} actors")

    snapshot_download(repo_id="MahiA/CREMA-D", repo_type="dataset", allow_patterns=["audios/*.wav"],
                      local_dir=WAV_DIR, max_workers=8)

    split = speaker_split(demographics, args.seed)
    demo = demographics.set_index("ActorID")
    rows, problems = [], []
    for name, (oid, size) in sorted(hashes.items()):
        path = WAV_DIR / "audios" / name
        if not path.exists():
            problems.append((name, "missing from mirror"))
            continue
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != oid:
            problems.append((name, "hash mismatch with official repo"))
            continue
        info = sf.info(io.BytesIO(data))
        if info.duration < 0.3:
            problems.append((name, f"too short ({info.duration:.2f}s)"))
            continue
        actor, sentence, emotion, level = NAME_RE.match(name).groups()
        actor = int(actor)
        rows.append({
            "file": str(path.relative_to(DATA.parent)).replace("\\", "/"),
            "clip": name[:-4], "actor": actor, "sentence": sentence, "emotion": emotion, "level": level,
            "sex": demo.loc[actor, "Sex"], "age": int(demo.loc[actor, "Age"]),
            "duration_s": round(info.duration, 3), "sample_rate": info.samplerate,
            "split": split[actor],
        })

    META_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(META_DIR / "metadata.csv", index=False)
    pd.DataFrame(problems, columns=["clip", "problem"]).to_csv(META_DIR / "excluded_clips.csv", index=False)

    print(f"verified clips: {len(meta)}   excluded: {len(problems)}")
    for name, why in problems:
        print(f"  excluded {name}: {why}")
    print("sample rates:", meta["sample_rate"].value_counts().to_dict())
    print(meta.groupby("split").agg(clips=("clip", "size"), actors=("actor", "nunique")).to_string())
    print(pd.crosstab(meta["split"], meta["emotion"]).to_string())


if __name__ == "__main__":
    main()
