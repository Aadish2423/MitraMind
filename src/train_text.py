"""Models 1 and 2: XLM-R emotion classifiers for Hindi (Devanagari) and Hinglish (romanized) text.

  m1_hindi_text     HinEmo-9 Devanagari Hindi + English rows, tested on Devanagari Hindi.
                    Stand-in until EmoInHindi access arrives: pass --emoinhindi_csv to train on
                    EmoInHindi instead (multi-label emotions + 0-3 intensity, split by dialogue).
  m2_hinglish_text  HinEmo-9 romanized Hinglish + English rows, tested on Hinglish.

English rows are added to both because HinEmo-9 has very few Devanagari Fear/Surprise examples;
XLM-R transfers what it learns from them across scripts. Each model is selected on its own
script's human-verified dev rows and scored on its own script's human-verified test rows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import (CUE_MAP, DATA, MODELS, autocast_dtype, cue_row, device, label_metrics, set_seed,
                    write_cues, write_metrics)

HINEMO = DATA / "hinemo9"
SETUPS = {
    "m1": {"model_id": "m1_hindi_text", "target": "Hindi", "train_langs": {"Hindi", "English"}},
    "m2": {"model_id": "m2_hinglish_text", "target": "Hinglish", "train_langs": {"Hinglish", "English"}},
}
EMOINHINDI_COLUMNS = ["dialogueId", "utterance_no", "authorRole", "utterance", "emotions", "emoIntensity"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def load_hinemo(setup, limit=None):
    train = read_jsonl(HINEMO / "train.jsonl")
    dev = read_jsonl(HINEMO / "dev.jsonl")
    test = read_jsonl(HINEMO / "test.jsonl")
    train = train[train["language_label"].isin(setup["train_langs"])]
    dev = dev[dev["language_label"] == setup["target"]]
    test = test[test["language_label"] == setup["target"]]
    if limit:
        train, dev, test = train.sample(min(limit, len(train)), random_state=0), dev.head(limit), test.head(limit)
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


def load_emoinhindi(csv_path, seed, limit=None):
    """Read the EmoInHindi CSV (format seen in public notebooks) and split by dialogue."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if list(df.columns) != EMOINHINDI_COLUMNS:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, header=None, names=EMOINHINDI_COLUMNS)
        df = df[df["emotions"] != "emotions"]  # drop a header row read as data
    rows = []
    for r in df.itertuples(index=False):
        emotions = [e.strip() for e in r.emotions.split(",") if e.strip()]
        levels = [int(x) if x.strip().isdigit() else 0 for x in r.emoIntensity.split(",")]
        levels += [0] * (len(emotions) - len(levels))
        intensity = {}
        for e, lvl in zip(emotions, levels):
            intensity[e] = max(intensity.get(e, 0), lvl)  # the same emotion can be listed twice
        rows.append({"id": f"d{r.dialogueId}_u{r.utterance_no}", "dialogue": r.dialogueId,
                     "role": r.authorRole, "text": r.utterance, "intensity": intensity})
    data = pd.DataFrame(rows)
    if limit:
        data = data.head(limit)
    unknown = {e for d in data["intensity"] for e in d} - set(CUE_MAP["emoinhindi"])
    if unknown:
        raise SystemExit(f"EmoInHindi labels missing from CUE_MAP['emoinhindi']: {sorted(unknown)}")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, rest_idx = next(gss.split(data, groups=data["dialogue"]))
    rest = data.iloc[rest_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    dev_idx, test_idx = next(gss.split(rest, groups=rest["dialogue"]))
    return (data.iloc[train_idx].reset_index(drop=True), rest.iloc[dev_idx].reset_index(drop=True),
            rest.iloc[test_idx].reset_index(drop=True))


class MultiLabelIntensity(nn.Module):
    """Encoder with two heads: which emotions are present (sigmoid) and how intense each is (0-3)."""

    def __init__(self, base, n_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.emotion = nn.Linear(hidden, n_labels)
        self.intensity = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.dropout(h)
        return self.emotion(h), 3 * torch.sigmoid(self.intensity(h))


def batches(tokenizer, texts, targets, batch_size, max_len, shuffle):
    order = list(range(len(texts)))

    def collate(idx):
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt")
        enc["targets"] = torch.as_tensor(np.stack([targets[i] for i in idx]))
        return enc

    return DataLoader(order, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def freeze_word_embeddings(encoder):
    # XLM-R's 250k-token embedding matrix is 70% of its weights; freezing it lets training fit in 4 GB VRAM.
    encoder.get_input_embeddings().weight.requires_grad_(False)


def run_epochs(model, step_fn, train_loader, eval_fn, epochs, lr, save_fn):
    dev_ = device()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total = epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in train_loader:
            batch = {k: v.to(dev_) for k, v in batch.items()}
            with torch.autocast(dev_.type, dtype=autocast_dtype()):
                loss = step_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        score = eval_fn()
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                        "dev_macro_f1": round(score, 4), "seconds": round(time.time() - t0)})
        print(history[-1], flush=True)
        if score > best:
            best = score
            save_fn()
    return history


@torch.no_grad()
def predict(model, loader, forward):
    model.eval()
    dev_, outs = device(), []
    for batch in loader:
        batch = {k: v.to(dev_) for k, v in batch.items() if k != "targets"}
        with torch.autocast(dev_.type, dtype=autocast_dtype()):
            outs.append(forward(model, batch))
    return [torch.cat(parts).float().cpu().numpy() for parts in zip(*outs)]


def train_hinemo(args, setup):
    model_id = setup["model_id"]
    train, dev, test = load_hinemo(setup, args.limit)
    labels = sorted(CUE_MAP["hinemo9"])
    lab2id = {l: i for i, l in enumerate(labels)}
    print(f"{model_id}: train {len(train)} ({train['language_label'].value_counts().to_dict()}), "
          f"dev {len(dev)}, test {len(test)}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=lab2id).to(device())
    freeze_word_embeddings(model)

    def loader(df, shuffle):
        y = [np.int64(lab2id[l]) for l in df["label"]]
        return batches(tok, df["text"].tolist(), y, args.batch, args.max_len, shuffle)

    train_loader, dev_loader, test_loader = loader(train, True), loader(dev, False), loader(test, False)
    out_dir = MODELS / model_id
    forward = lambda m, b: (m(**b).logits,)

    def step(m, b):
        return nn.functional.cross_entropy(m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits,
                                           b["targets"])

    def dev_f1():
        (logits,) = predict(model, dev_loader, forward)
        return f1_score(dev["label"].map(lab2id), logits.argmax(1), average="macro")

    def save():
        model.save_pretrained(out_dir); tok.save_pretrained(out_dir)

    history = run_epochs(model, step, train_loader, dev_f1, args.epochs, args.lr, save)

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device())
    (logits,) = predict(model, test_loader, forward)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    rows = []
    for r, p in zip(test.itertuples(index=False), probs):
        pred = labels[int(p.argmax())]
        rows.append(cue_row(
            model_id=model_id, train_data=f"HinEmo-9 {'+'.join(sorted(setup['train_langs']))}",
            eval_data=f"HinEmo-9 test (human gold, {setup['target']})", sample_id=r.id, modality="text",
            group=r.language_label, inp=r.text, labels=labels, probs=p, cue_map=CUE_MAP["hinemo9"],
            true_label=r.label, true_cue=CUE_MAP["hinemo9"][r.label], correct=pred == r.label))
    preds = [labels[i] for i in probs.argmax(1)]
    cue_true = [CUE_MAP["hinemo9"][l] for l in test["label"]]
    metrics = {
        "model_id": model_id, "base_model": args.base, "train_rows": len(train), "history": history,
        "test_emotion": label_metrics(test["label"].tolist(), preds, labels),
        "test_cue": label_metrics(cue_true, [r["pred_cue"] for r in rows], sorted(set(CUE_MAP["hinemo9"].values()))),
    }
    return rows, metrics


def train_emoinhindi(args):
    model_id = "m1_hindi_text"
    train, dev, test = load_emoinhindi(args.emoinhindi_csv, args.seed, args.limit)
    labels = sorted(CUE_MAP["emoinhindi"])
    lab2id = {l: i for i, l in enumerate(labels)}
    print(f"{model_id} (EmoInHindi): train {len(train)}, dev {len(dev)}, test {len(test)} utterances")

    def targets(df):
        out = []
        for inten in df["intensity"]:
            t = np.zeros(2 * len(labels), dtype=np.float32)  # [present flags | intensities]
            for e, lvl in inten.items():
                t[lab2id[e]] = 1.0
                t[len(labels) + lab2id[e]] = lvl
            out.append(t)
        return out

    tok = AutoTokenizer.from_pretrained(args.base)
    model = MultiLabelIntensity(args.base, len(labels)).to(device())
    freeze_word_embeddings(model.encoder)
    loader = lambda df, sh: batches(tok, df["text"].tolist(), targets(df), args.batch, args.max_len, sh)
    train_loader, dev_loader, test_loader = loader(train, True), loader(dev, False), loader(test, False)
    out_dir = MODELS / model_id
    n = len(labels)
    forward = lambda m, b: m(b["input_ids"], b["attention_mask"])

    def step(m, b):
        logits, inten = forward(m, b)
        present, level = b["targets"][:, :n], b["targets"][:, n:]
        bce = nn.functional.binary_cross_entropy_with_logits(logits.float(), present)
        mse = ((inten.float() - level) ** 2 * present).sum() / present.sum().clamp(min=1)
        return bce + 0.5 * mse

    def dev_f1():
        logits, _ = predict(model, dev_loader, forward)
        y = np.stack(targets(dev))[:, :n]
        return f1_score(y, (logits > 0).astype(int), average="macro", zero_division=0)

    def save():
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "model.pt")
        (out_dir / "labels.json").write_text(json.dumps({"base": args.base, "labels": labels}))
        tok.save_pretrained(out_dir)

    history = run_epochs(model, step, train_loader, dev_f1, args.epochs, args.lr, save)

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device()))
    logits, inten = predict(model, test_loader, forward)
    probs = 1 / (1 + np.exp(-logits))
    rows, y_true = [], np.stack(targets(test))[:, :n]
    for r, p, lv, yt in zip(test.itertuples(index=False), probs, inten, y_true):
        present = [labels[i] for i in np.flatnonzero(p >= 0.5)] or [labels[int(p.argmax())]]
        pred = "|".join(f"{l}({lv[lab2id[l]]:.1f})" for l in present)
        true = "|".join(f"{e}({lvl})" for e, lvl in sorted(r.intensity.items()))
        true_cues = sorted({CUE_MAP["emoinhindi"][e] for e in r.intensity})
        rows.append(cue_row(
            model_id=model_id, train_data="EmoInHindi (dialogue-level split)",
            eval_data="EmoInHindi test dialogues", sample_id=r.id, modality="text", group=f"Hindi/{r.role}",
            inp=r.text, labels=labels, probs=p, cue_map=CUE_MAP["emoinhindi"], multilabel=True,
            pred_label=pred, true_label=true, true_cue="|".join(true_cues),
            correct=set(present) == set(r.intensity)))
    pred_bin = (probs >= 0.5).astype(int)
    metrics = {
        "model_id": model_id, "base_model": args.base, "train_rows": len(train), "history": history,
        "test_micro_f1": round(f1_score(y_true, pred_bin, average="micro", zero_division=0), 4),
        "test_macro_f1": round(f1_score(y_true, pred_bin, average="macro", zero_division=0), 4),
        "test_exact_match": round(float(np.mean([r["correct"] for r in rows])), 4),
    }
    return rows, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SETUPS), required=True)
    ap.add_argument("--emoinhindi_csv", help="train m1 on EmoInHindi instead of the HinEmo-9 stand-in")
    ap.add_argument("--base", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use only N rows per split (smoke test)")
    args = ap.parse_args()
    set_seed(args.seed)

    if args.emoinhindi_csv:
        if args.model != "m1":
            raise SystemExit("--emoinhindi_csv only applies to --model m1")
        rows, metrics = train_emoinhindi(args)
    else:
        rows, metrics = train_hinemo(args, SETUPS[args.model])
    model_id = metrics["model_id"]
    print("wrote", write_cues(rows, model_id))
    print("wrote", write_metrics(metrics, model_id))
    summary = {k: v for k, v in metrics.items() if k.startswith("test") and not isinstance(v, dict)}
    for k in ("test_emotion", "test_cue"):
        if k in metrics:
            summary[k] = {m: metrics[k][m] for m in ("n", "accuracy", "macro_f1")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
