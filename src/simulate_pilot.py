"""Make a SYNTHETIC pilot database to test the Phase 2 export and Phase 3 training end to end.

Each fictional patient has a hidden severity that improves, stays flat or worsens over 12 weeks.
They chat on about half the days (English, Hinglish or Hindi phrases whose symptom content follows the
hidden severity) and fill PHQ-9 and GAD-7 every 14 days. Every message goes through the real Phase 1
analysis, so the export sees the same cue JSON as live data.

This data only checks that the pipeline runs. A model trained on it learns the simulator, not patients.

    python src/simulate_pilot.py --patients 120 --db var/pilot_sim.db
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from care.store import Store  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402
from phase1.schema import Entry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SYMPTOMS = {
    "en": ["I couldn't sleep last night", "lying awake at 3 am again", "felt so tired all day",
           "nothing feels enjoyable anymore", "feeling really low today", "I feel like a failure",
           "I can't concentrate on anything", "I'm not hungry at all", "I kept worrying about everything",
           "I felt nervous and on edge", "I couldn't relax at all", "I get irritated so easily these days"],
    "hinglish": ["raat ko neend nahi aayi", "bahut thakaan ho rahi hai", "kuch bhi accha nahi lag raha",
                 "aaj mood bahut low hai", "bahut tension ho rahi hai", "dimaag shaant nahi ho raha",
                 "kisi kaam mein mann nahi lag raha", "chhoti chhoti baat pe gussa aa jata hai"],
    "hi": ["रात को नींद नहीं आई", "बहुत थकान लग रही है", "कुछ भी अच्छा नहीं लग रहा", "आज मन बहुत उदास है",
           "बहुत चिंता हो रही है", "किसी काम में मन नहीं लग रहा"],
}
GOOD = {
    "en": ["had a decent day", "went for a walk in the evening and it helped", "talked to my friend and felt lighter",
           "did the breathing exercise, it helped", "watched a movie with my family", "slept better last night"],
    "hinglish": ["aaj walk pe gaya, accha laga", "dost se baat ki, halka mehsoos hua", "aaj din theek tha",
                 "breathing exercise ki, thoda help hua", "kal raat neend acchi aayi"],
    "hi": ["आज टहलने गया, अच्छा लगा", "दोस्त से बात की, हल्का महसूस हुआ", "आज दिन ठीक था", "कल रात नींद अच्छी आई"],
}
STRESSORS = {
    "work": {"en": "my boss gave me another deadline", "hinglish": "office mein boss ne phir deadline de di",
             "hi": "ऑफिस में काम का बहुत दबाव है"},
    "school_exam": {"en": "exams are coming up and I'm behind", "hinglish": "exams ka bahut pressure hai",
                    "hi": "परीक्षा की बहुत टेंशन है"},
    "economic": {"en": "worried about rent and the EMI", "hinglish": "EMI aur rent ki tension hai",
                 "hi": "पैसों की चिंता है"},
    "family": {"en": "had a fight with my parents", "hinglish": "ghar pe papa se fight ho gayi",
               "hi": "घर में झगड़ा हो गया"},
    "relationships": {"en": "my partner and I argued again", "hinglish": "girlfriend se phir jhagda ho gaya",
                      "hi": "पति से फिर बहस हो गई"},
    "health": {"en": "my back pain is bad again", "hinglish": "body pain phir se badh gaya",
               "hi": "माँ की तबीयत ठीक नहीं है"},
}
BURDEN = {"en": "sometimes I feel like a burden to everyone", "hinglish": "lagta hai sab ke liye bojh ban gaya hoon",
          "hi": "लगता है मैं सब पर बोझ हूँ"}


def items_for(total: float, n: int, rng: random.Random, item9_ok: bool) -> list[int]:
    total = max(0, min(3 * n, round(total)))
    items = [0] * n
    order = list(range(n if item9_ok else n - 1)) if n == 9 else list(range(n))
    for _ in range(total):
        free = [i for i in order if items[i] < 3]
        if not free:
            break
        items[rng.choice(free)] += 1
    return items


def message(sev: float, lang: str, stressors: list[str], rng: random.Random, signal: float = 1.0) -> str:
    parts = []
    # How strongly symptom talk tracks hidden severity. signal=1: from ~5% of phrases at severity 4 to
    # ~95% at 20; signal=0: symptom talk unrelated to severity (the trainer should then find nothing).
    p_sym = min(0.95, max(0.05, 0.5 + signal * (sev - 12) / 18))
    for _ in range(2):
        parts.append(rng.choice(SYMPTOMS[lang]) if rng.random() < p_sym else rng.choice(GOOD[lang]))
    if rng.random() < 0.55:
        parts.append(STRESSORS[rng.choice(stressors)][lang])
    if sev > 17 and rng.random() < 0.08:
        parts.append(BURDEN[lang])
    rng.shuffle(parts)
    sep = "। " if lang == "hi" else ". "
    return sep.join(parts) + ("।" if lang == "hi" else ".")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", type=int, default=120)
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--db", default=str(ROOT / "var" / "pilot_sim.db"))
    ap.add_argument("--neural", action="store_true", help="also run m1/m2 (GPU); default lexicon only")
    ap.add_argument("--signal", type=float, default=1.0,
                    help="how strongly symptom talk follows severity (0 = none, a negative control)")
    args = ap.parse_args()

    db = Path(args.db)
    if db.exists():
        db.unlink()
    rng = random.Random(args.seed)
    store = Store(db)
    orch = Phase1Orchestrator(enable_neural=args.neural)
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    n_msgs = 0
    for i in range(args.patients):
        pid = f"sim_{i:03d}"
        lang = rng.choices(["en", "hinglish", "hi"], [0.5, 0.35, 0.15])[0]
        store.create_patient(pid, None, lang)
        for purpose, p in (("companion", 1), ("care_team", 1), ("pilot_research", 0.85), ("research_text", 0.3)):
            store.record_consent(pid, purpose, rng.random() < p, "simulated")
        kind = rng.choices(["improving", "stable", "worsening"], [0.4, 0.35, 0.25])[0]
        sev0 = rng.uniform(5, 20)
        slope = {"improving": -rng.uniform(1.5, 4.5), "stable": rng.uniform(-1, 1),
                 "worsening": rng.uniform(1.5, 4.5)}[kind] / 14  # points per day
        gad_ratio = rng.uniform(0.55, 0.85)
        stressors = rng.sample(list(STRESSORS), 2)
        chat_p = rng.uniform(0.25, 0.75)

        def sev(day):
            return max(0.0, min(27.0, sev0 + slope * day + rng.gauss(0, 1.2)))

        for day in range(args.weeks * 7 + 1):
            t = start + timedelta(days=day, hours=rng.randint(8, 22), minutes=rng.randint(0, 59))
            if day % 14 == 0:
                s = sev(day)
                for inst, n, total in (("PHQ-9", 9, s), ("GAD-7", 7, s * gad_ratio)):
                    store.add_questionnaire(pid, inst, t.date().isoformat(),
                                            items_for(total + rng.gauss(0, 1.5), n, rng, s > 15 and rng.random() < 0.25),
                                            "simulated", t.isoformat())
            if day % 14 != 0 and rng.random() < chat_p:
                text = message(sev(day), lang, stressors, rng, args.signal)
                a = orch.analyze_entry(Entry(entry_id="sim", timestamp=t, text=text))
                analysis = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in a.items() if k != "text"}
                mid = store.add_message(pid, "user", text, language=a["language"], shared=rng.random() < 0.2,
                                        risk_level=a["risk"].level, analysis=analysis, created_at=t.isoformat())
                n_msgs += 1
                if a["risk"].level != "none":
                    aid = store.add_alert(pid, a["risk"].level, a["risk"].reasons, "message", mid, t.isoformat())
                    store.update_alert(aid, "resolved", "sim-clinician", "simulated",
                                       "confirmed_risk" if sev(day) > 17 else rng.choice(["false_alarm", "unclear"]))
                if rng.random() < 0.15:
                    store.add_mind_pause(pid, rng.choice(["box_breathing", "grounding_54321", "body_scan"]),
                                         rng.random() < 0.7, t.isoformat())
        if rng.random() < 0.3:
            store.add_feedback(pid, "sim-clinician", rng.randint(3, 5), rng.random() < 0.8,
                               round(rng.uniform(3, 12), 1), None, None)
    print(f"synthetic pilot: {args.patients} patients, {n_msgs} messages -> {db}")


if __name__ == "__main__":
    main()
