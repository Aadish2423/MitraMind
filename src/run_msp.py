"""Model 4: the MSP lab's own MSP-Podcast models (Odyssey 2024 baselines, MIT licence), used as released.

MSP-Podcast audio needs a licence from UT Dallas, but the lab published two WavLM-large models trained on
it: an 8-emotion classifier and an arousal/dominance/valence regressor. The architecture below is a copy
of their pipeline_utils.py, rebuilt locally so no code from the model repo is executed and the
WavLM-large backbone is not downloaded twice.

They are run on the same unseen CREMA-D test actors as model 3. MSP-Podcast is natural podcast speech
and CREMA-D is acted, so this is a cross-corpus check and the cue-level accuracy is the fair comparison.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

from common import (CUE_MAP, DATA, ROOT, autocast_dtype, cue_row, cue_scores, device, label_metrics, write_cues,
                    write_metrics)
from train_cremad import load_audio

MODEL_ID = "m4_msp_speech"
REPOS = {"categorical": "3loi/SER-Odyssey-Baseline-WavLM-Categorical",
         "dimensional": "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"}
SR = 16000


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.zeros(dim, 1))

    def forward(self, xs, mask):
        feat_lens = (torch.div(mask.sum(1) - 1, SR * 0.02, rounding_mode="floor") + 1).int().tolist()
        pooled = []
        for x, n in zip(xs, feat_lens):
            x = x[:n].unsqueeze(0)
            w = F.softmax(torch.matmul(torch.tanh(self.sap_linear(x)), self.attention).squeeze(2), dim=1)
            w = w.view(1, -1, 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum(x ** 2 * w, dim=1) - mu ** 2).clamp(min=1e-5))
            pooled.append(torch.cat((mu, rh), 1).squeeze(0))
        return torch.stack(pooled)


class EmotionRegression(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, out_dim, dropout):
        super().__init__()
        self.fc = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))])
        for _ in range(n_layers - 1):
            self.fc.append(nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)))
        self.out = nn.Sequential(nn.Linear(hidden, out_dim))
        self.inp_drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.inp_drop(x)
        for fc in self.fc:
            h = fc(h)
        return self.out(h)


class SERModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ssl_model = AutoModel.from_config(AutoConfig.from_pretrained(cfg["ssl_type"]))
        self.pool_model = AttentiveStatisticsPooling(cfg["hidden_size"])
        self.ser_model = EmotionRegression(cfg["hidden_size"] * 2, cfg["hidden_size"], cfg["classifier_hidden_layers"],
                                           cfg["num_classes"], cfg["classifier_dropout_prob"])

    def forward(self, x, mask):
        h = self.ssl_model(x, attention_mask=mask).last_hidden_state
        return self.ser_model(self.pool_model(h, mask))


def load_ser(repo, dev=None):
    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    model = SERModel(cfg)
    state = load_file(hf_hub_download(repo, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The WavLM checkpoint may carry buffers the fresh config does not create; real weights must all load.
    if missing or any(not k.endswith(("masked_spec_embed",)) for k in unexpected):
        raise SystemExit(f"{repo}: weights do not match architecture.\nmissing={missing}\nunexpected={unexpected}")
    return model.to(dev or device()).eval(), cfg


@torch.no_grad()
def run(model, cfg, wavs):
    dev_, outs = next(model.parameters()).device, []  # wherever load_ser put the model
    for w in wavs:  # one clip at a time: no padding, exactly as in the model card
        x = torch.from_numpy((w - cfg["mean"]) / (cfg["std"] + 1e-6)).unsqueeze(0).to(dev_)
        mask = torch.ones_like(x)
        with torch.autocast(dev_.type, dtype=autocast_dtype()):
            outs.append(model(x, mask).float().cpu())
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="CREMA-D split to run on (test = same clips as model 3)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    meta = pd.read_csv(DATA / "crema_d" / "metadata.csv")
    clips = meta[meta["split"] == args.split].reset_index(drop=True)
    if args.limit:
        clips = clips.head(args.limit)
    wavs = [load_audio(ROOT / f) for f in clips["file"]]
    print(f"running on {len(clips)} CREMA-D {args.split} clips", flush=True)

    cat_model, cat_cfg = load_ser(REPOS["categorical"])
    labels = [cat_cfg["id2label"][str(i)] for i in range(cat_cfg["num_classes"])]
    probs = torch.softmax(torch.from_numpy(run(cat_model, cat_cfg, wavs)), 1).numpy()
    del cat_model; torch.cuda.empty_cache()

    dim_model, dim_cfg = load_ser(REPOS["dimensional"])
    dims = run(dim_model, dim_cfg, wavs)  # columns: arousal, dominance, valence (about 0..1)
    del dim_model; torch.cuda.empty_cache()

    rows = []
    for r, p, (aro, dom, val) in zip(clips.itertuples(index=False), probs, dims):
        scores, _ = cue_scores(labels, p, CUE_MAP["msp"])
        true_cue = CUE_MAP["cremad"][r.emotion]
        rows.append(cue_row(
            model_id=MODEL_ID, train_data="MSP-Podcast (pretrained, Odyssey 2024 baseline)",
            eval_data=f"CREMA-D unseen {args.split} actors (cross-corpus)", sample_id=r.clip, modality="audio",
            group=f"actor {r.actor} ({r.sex}, {r.age})", inp=r.file, labels=labels, probs=p,
            cue_map=CUE_MAP["msp"], true_label=r.emotion, true_cue=true_cue,
            correct=max(scores, key=scores.get) == true_cue,
            arousal=round(float(aro), 4), valence=round(float(val), 4), dominance=round(float(dom), 4)))

    out = pd.DataFrame(rows)
    true_cues = out["true_cue"].tolist()
    cue_labels = sorted(set(CUE_MAP["msp"].values()))
    by_emotion = (out.assign(emotion=clips["emotion"]).groupby("emotion")[["arousal", "valence", "dominance"]]
                  .mean().round(3))
    metrics = {
        "model_id": MODEL_ID, "repos": REPOS, "eval_clips": len(out),
        "note": "Cross-corpus: label sets differ, so 'correct' compares MitraMind cues.",
        "test_cue": label_metrics(true_cues, out["pred_cue"].tolist(), cue_labels),
        "predicted_label_counts": out["pred_label"].value_counts().to_dict(),
        "mean_dimensions_by_true_emotion": by_emotion.to_dict(orient="index"),
    }
    print("wrote", write_cues(rows, MODEL_ID))
    print("wrote", write_metrics(metrics, MODEL_ID))
    print(json.dumps({m: metrics["test_cue"][m] for m in ("n", "accuracy", "macro_f1")}, indent=2))
    print(by_emotion.to_string())


if __name__ == "__main__":
    main()
