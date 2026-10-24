"""Does voice chat still work when the audio is muffled, quiet, noisy or phone-quality?

Takes clean recordings (var/audio_eval/clean/<sentence>__<voice>.wav and var/audio_eval/sentences.json with
the words actually said), degrades each one in several ways, runs Whisper with each clean-up option from
care/voice.py, and measures:
  - word error rate (WER): share of words wrong, missing or added (0 = perfect);
  - crisis detection: does the transcript get the same Phase 1 risk level as the words actually said?

    python src/eval_voice_robustness.py            # writes outputs/voice_robustness/{report.md,results.json}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from care.voice import SR, WhisperSTT, enhance, load_wav  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402
from phase1.schema import Entry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "var" / "audio_eval"


def lowpass(wav, cutoff):
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(6, cutoff, "lowpass", fs=SR, output="sos"), wav).astype("float32")


def add_noise(wav, snr_db, rng):
    # Mostly pink-ish (more low-frequency energy, like a room or a fan) plus some white noise.
    white = rng.standard_normal(len(wav))
    pink = np.cumsum(rng.standard_normal(len(wav)))
    pink = pink - np.convolve(pink, np.ones(400) / 400, mode="same")  # remove the drift
    noise = 0.6 * pink / (np.std(pink) + 1e-9) + 0.4 * white / (np.std(white) + 1e-9)
    p_sig = np.mean(wav ** 2)
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10))) / (np.std(noise) + 1e-9)
    return (wav + noise).astype("float32")


def phone(wav):
    from scipy.signal import butter, resample_poly, sosfilt

    narrow = sosfilt(butter(4, [300, 3400], "bandpass", fs=SR, output="sos"), wav)
    return resample_poly(resample_poly(narrow, 1, 2), 2, 1).astype("float32")  # through 8 kHz and back


def gain(wav, db):
    return (wav * 10 ** (db / 20)).astype("float32")


CONDITIONS = {
    "clean": lambda w, r: w,
    "muffled (cloth over mic, <1.5 kHz)": lambda w, r: lowpass(w, 1500),
    "very muffled (<700 Hz)": lambda w, r: lowpass(w, 700),
    "very quiet (-30 dB)": lambda w, r: gain(w, -30),
    "noisy room (SNR 10 dB)": lambda w, r: add_noise(w, 10, r),
    "very noisy (SNR 0 dB)": lambda w, r: add_noise(w, 0, r),
    "phone call (8 kHz)": lambda w, r: phone(w),
    "muffled + quiet + noisy": lambda w, r: add_noise(gain(lowpass(w, 900), -24), 5, r),
}
VARIANTS = ["none", "normalise", "highpass,normalise", "highpass,denoise,normalise"]


def words(text):
    return re.findall(r"[a-z']+", text.lower().replace("can't", "cannot").replace("can not", "cannot"))


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(len(r), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=str(EVAL / "clean"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "voice_robustness"))
    ap.add_argument("--repeats", type=int, default=5,
                    help="noise draws per clip for the noisy conditions (results vary with the exact noise)")
    args = ap.parse_args()

    sentences = json.loads((EVAL / "sentences.json").read_text(encoding="utf-8-sig"))
    clips = sorted(Path(args.clips).glob("*.wav"))
    stt, orch = WhisperSTT(), Phase1Orchestrator(enable_neural=False)
    risk = lambda text: orch.analyze_entry(Entry(entry_id="x", timestamp=datetime.now(timezone.utc),  # noqa: E731
                                                 text=text))["risk"].level
    truth = {k: risk(s) for k, s in sentences.items()}
    rng = np.random.default_rng(7)
    rows = []
    for clip in clips:
        key = clip.stem.split("__")[0]
        clean = load_wav(clip)
        for cond, degrade in CONDITIONS.items():
            for draw in range(args.repeats if "noisy" in cond else 1):
                audio = degrade(clean, rng)  # every clean-up option gets the same noisy audio
                for variant in VARIANTS:
                    text = stt.transcribe_audio(enhance(audio, variant), None)["text"]  # auto language, as in the app
                    rows.append({"clip": clip.stem, "sentence": key, "condition": cond, "draw": draw,
                                 "variant": variant, "transcript": text, "wer": round(wer(sentences[key], text), 3),
                                 "risk_expected": truth[key], "risk_got": risk(text)})
        print(f"done {clip.name}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def summary(cond, variant):
        sel = [r for r in rows if r["condition"] == cond and r["variant"] == variant]
        crisis = [r for r in sel if r["risk_expected"] in ("high", "urgent")]
        flagged = [r for r in sel if r["risk_expected"] != "none"]
        return {"wer": round(float(np.mean([r["wer"] for r in sel])), 3),
                "crisis_caught": f"{sum(r['risk_got'] in ('high', 'urgent') for r in crisis)}/{len(crisis)}",
                "crisis_any_alert": f"{sum(r['risk_got'] != 'none' for r in crisis)}/{len(crisis)}",
                "same_risk_level": f"{sum(r['risk_got'] == r['risk_expected'] for r in flagged)}/{len(flagged)}",
                "false_alarms": sum(r["risk_expected"] == "none" and r["risk_got"] != "none" for r in sel)}

    table = {c: {v: summary(c, v) for v in VARIANTS} for c in CONDITIONS}
    (out / "summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    lines = ["# Voice chat with muffled, quiet, noisy and phone audio", "",
             f"{len(clips)} clean English recordings (Windows text-to-speech, 2 voices, {len(sentences)} sentences "
             "including 2 crisis sentences and 1 'burden' sentence), each degraded 8 ways and transcribed by "
             f"{stt.name} (language auto-detected, as in the app) with 4 clean-up options. Noisy conditions use "
             f"{args.repeats} different noise draws per clip. WER = share of words wrong (lower is better). "
             "'Crisis caught' = high/urgent sentences that still got high/urgent from the transcript; 'any alert' "
             "= they at least got watch or above (an alert for the care team and a helpline in the reply).", "",
             "| Condition | Clean-up | WER | Crisis caught | Crisis, any alert | Risk level kept | False alarms |",
             "|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        for v in VARIANTS:
            s = table[c][v]
            lines.append(f"| {c} | {v} | {s['wer']:.3f} | {s['crisis_caught']} | {s['crisis_any_alert']} | "
                         f"{s['same_risk_level']} | {s['false_alarms']} |")
    overall = {v: round(float(np.mean([r["wer"] for r in rows if r["variant"] == v])), 3) for v in VARIANTS}
    lines += ["", "Mean WER over all conditions: " + ", ".join(f"{v} {w:.3f}" for v, w in overall.items()), "",
              "Worst transcripts (no clean-up):", ""]
    worst = sorted((r for r in rows if r["variant"] == "none"), key=lambda r: -r["wer"])[:8]
    lines += [f"- {r['condition']}, {r['clip']} (WER {r['wer']}): \"{r['transcript']}\"" for r in worst]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6 + len(CONDITIONS) * len(VARIANTS) + 3]))


if __name__ == "__main__":
    main()
