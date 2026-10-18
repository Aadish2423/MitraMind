"""Combine each model's cue CSV into outputs/cues_all.csv and print a one-line score per model."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import CSV_COLUMNS, OUTPUTS

MODEL_IDS = ["m1_hindi_text", "m2_hinglish_text", "m3_cremad_speech", "m4_msp_speech"]


def main():
    frames, summary = [], []
    for model_id in MODEL_IDS:
        csv = OUTPUTS / f"{model_id}_cues.csv"
        if not csv.exists():
            print(f"missing {csv.name}; run its script first")
            continue
        frames.append(pd.read_csv(csv, encoding="utf-8-sig", dtype={"sample_id": str}))
        m = json.loads((OUTPUTS / f"{model_id}_metrics.json").read_text(encoding="utf-8"))
        native = m.get("test_emotion", {})
        summary.append({
            "model_id": model_id, "rows": len(frames[-1]),
            "emotion_accuracy": native.get("accuracy", m.get("test_exact_match")),
            "emotion_macro_f1": native.get("macro_f1", m.get("test_macro_f1")),
            "cue_accuracy": m.get("test_cue", {}).get("accuracy"),
            "cue_macro_f1": m.get("test_cue", {}).get("macro_f1"),
        })
    allc = pd.concat(frames, ignore_index=True)[CSV_COLUMNS]
    allc.to_csv(OUTPUTS / "cues_all.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(summary)
    table.to_csv(OUTPUTS / "model_scores.csv", index=False)
    print(f"wrote {OUTPUTS / 'cues_all.csv'} ({len(allc)} rows)")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
