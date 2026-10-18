"""Model 3: WavLM-base-plus fine-tuned on CREMA-D for 6-class speech emotion.

Uses the speaker-independent split from prepare_cremad.py: test actors are never heard in training,
which is the situation MitraMind is in with every new patient.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, WavLMForSequenceClassification, get_linear_schedule_with_warmup

from common import (CUE_MAP, DATA, MODELS, ROOT, autocast_dtype, cue_row, device, label_metrics, set_seed,
                    write_cues, write_metrics)

MODEL_ID = "m3_cremad_speech"
SR = 16000
LABELS = sorted(CUE_MAP["cremad"])  # ANG DIS FEA HAP NEU SAD


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = resample_poly(wav, SR, sr).astype(np.float32)
    return wav


def load_split(meta, split, limit=None):
    df = meta[meta["split"] == split].reset_index(drop=True)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=0).reset_index(drop=True)
    return df, [load_audio(ROOT / f) for f in df["file"]]


def make_loader(extractor, wavs, labels, batch, shuffle, max_s):
    max_len = int(max_s * SR)

    def collate(idx):
        clips = []
        for i in idx:
            w = wavs[i]
            if len(w) > max_len:  # random crop in training, centre crop otherwise
                start = np.random.randint(0, len(w) - max_len + 1) if shuffle else (len(w) - max_len) // 2
                w = w[start:start + max_len]
            clips.append(w)
        enc = extractor(clips, sampling_rate=SR, padding=True, return_attention_mask=True, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in idx])
        return enc

    return DataLoader(list(range(len(wavs))), batch_size=batch, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    dev_, out = device(), []
    for b in loader:
        with torch.autocast(dev_.type, dtype=autocast_dtype()):
            out.append(model(input_values=b["input_values"].to(dev_),
                             attention_mask=b["attention_mask"].to(dev_)).logits.float().cpu())
    return torch.softmax(torch.cat(out), 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="microsoft/wavlm-base-plus")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_s", type=float, default=4.0, help="crop length for training clips")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use only N clips per split (smoke test)")
    args = ap.parse_args()
    set_seed(args.seed)
    dev_ = device()

    meta = pd.read_csv(DATA / "crema_d" / "metadata.csv")
    lab2id = {l: i for i, l in enumerate(LABELS)}
    (train, train_w), (val, val_w), (test, test_w) = (load_split(meta, s, args.limit) for s in ("train", "val", "test"))
    print(f"train {len(train)} clips / {train.actor.nunique()} actors, val {len(val)} / {val.actor.nunique()}, "
          f"test {len(test)} / {test.actor.nunique()}", flush=True)

    extractor = AutoFeatureExtractor.from_pretrained(args.base)
    model = WavLMForSequenceClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=lab2id).to(dev_)
    model.freeze_feature_encoder()

    ys = lambda df: [lab2id[e] for e in df["emotion"]]
    # WavLM-base-plus group-normalises over the whole input, so zero padding shifts its features;
    # evaluating one clip at a time keeps val/test scores free of that effect.
    train_loader = make_loader(extractor, train_w, ys(train), args.batch, True, args.max_s)
    val_loader = make_loader(extractor, val_w, ys(val), 1, False, 8.0)
    test_loader = make_loader(extractor, test_w, ys(test), 1, False, 8.0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_loader) // args.accum
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    out_dir = MODELS / MODEL_ID
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for i, b in enumerate(train_loader, 1):
            with torch.autocast(dev_.type, dtype=autocast_dtype()):
                loss = model(input_values=b["input_values"].to(dev_), attention_mask=b["attention_mask"].to(dev_),
                             labels=b["labels"].to(dev_)).loss / args.accum
            loss.backward()
            losses.append(loss.item() * args.accum)
            if i % args.accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        val_f1 = f1_score(ys(val), predict(model, val_loader).argmax(1), average="macro")
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                        "val_macro_f1": round(val_f1, 4), "seconds": round(time.time() - t0)})
        print(history[-1], flush=True)
        if val_f1 > best:
            best = val_f1
            model.save_pretrained(out_dir); extractor.save_pretrained(out_dir)

    model = WavLMForSequenceClassification.from_pretrained(out_dir).to(dev_)
    probs = predict(model, test_loader)
    rows = []
    for r, p in zip(test.itertuples(index=False), probs):
        pred = LABELS[int(p.argmax())]
        rows.append(cue_row(
            model_id=MODEL_ID, train_data="CREMA-D train actors", eval_data="CREMA-D unseen test actors",
            sample_id=r.clip, modality="audio", group=f"actor {r.actor} ({r.sex}, {r.age})", inp=r.file,
            labels=LABELS, probs=p, cue_map=CUE_MAP["cremad"], true_label=r.emotion,
            true_cue=CUE_MAP["cremad"][r.emotion], correct=pred == r.emotion))
    preds = [LABELS[i] for i in probs.argmax(1)]
    test_hi = test["level"].eq("HI").to_numpy()
    metrics = {
        "model_id": MODEL_ID, "base_model": args.base, "train_clips": len(train), "history": history,
        "test_actors": sorted(test["actor"].unique().tolist()),
        "test_emotion": label_metrics(test["emotion"].tolist(), preds, LABELS),
        "test_cue": label_metrics([CUE_MAP["cremad"][e] for e in test["emotion"]], [r["pred_cue"] for r in rows],
                                  sorted(set(CUE_MAP["cremad"].values()))),
        "test_accuracy_high_intensity_clips": round(float(np.mean(np.array(preds)[test_hi] == test["emotion"][test_hi])), 4)
        if test_hi.any() else None,
    }
    print("wrote", write_cues(rows, MODEL_ID))
    print("wrote", write_metrics(metrics, MODEL_ID))
    print(json.dumps({k: {m: metrics[k][m] for m in ("n", "accuracy", "macro_f1")}
                      for k in ("test_emotion", "test_cue")}, indent=2))


if __name__ == "__main__":
    main()
