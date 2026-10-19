"""MitraMind Phase 1 command line.

    python src/run_phase1.py analyze --text "Can't sleep, exams next week"     # one check-in: cues + triage
    python src/run_phase1.py card examples/patients/demo_student_01.json       # one User State Card
    python src/run_phase1.py demo                                              # cards for all demo patients
    python src/run_phase1.py bench --models qwen3.5 mistral                    # LLM structured-output benchmark

LLM options (card / demo): --backend ollama|openai|none, --model, --think, --base-url, --api-key.
Cards are written to outputs/phase1/cards/<patient_ref>.json and .md.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1.llm import OllamaBackend, OpenAICompatBackend
from phase1.orchestrator import Phase1Orchestrator
from phase1.render import card_to_markdown
from phase1.schema import Entry, PatientBundle

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "outputs" / "phase1" / "cards"
DEMOS = sorted((ROOT / "examples" / "patients").glob("*.json"))


def make_backend(args):
    if args.backend == "none":
        return None
    if args.backend == "openai":
        return OpenAICompatBackend(args.model, args.base_url, args.api_key)
    return OllamaBackend(args.model, think=args.think)


def load_bundle(path: Path) -> PatientBundle:
    return PatientBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_card(card, extra) -> Path:
    CARDS.mkdir(parents=True, exist_ok=True)
    base = CARDS / card.patient_ref
    base.with_suffix(".json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(card_to_markdown(card), encoding="utf-8")
    (CARDS / f"{card.patient_ref}.llm_input.json").write_text(
        json.dumps(extra["digest"], indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return base


def cmd_analyze(args):
    orch = Phase1Orchestrator(enable_neural=not args.no_neural)
    entry = Entry(entry_id="cli", timestamp=datetime.now(timezone.utc), text=args.text, language=args.language)
    a = orch.analyze_entry(entry)
    a["risk"] = a["risk"].model_dump()
    a.pop("text")
    a["models_unavailable"] = orch.registry.unavailable
    print(json.dumps(a, indent=2, ensure_ascii=False, default=str))


def cmd_card(args, paths):
    orch = Phase1Orchestrator(backend=make_backend(args), enable_neural=not args.no_neural)
    for path in paths:
        t0 = time.time()
        card, extra = orch.build_card(load_bundle(path), use_llm=args.backend != "none")
        base = write_card(card, extra)
        print(f"{card.patient_ref}: risk={card.risk.level} narrative={card.narrative_source} "
              f"attempts={card.llm_attempts} {time.time() - t0:.0f}s -> {base.relative_to(ROOT)}.md")
        for note in card.guardrail_notes:
            print("   note:", note)


def cmd_bench(args):
    from phase1.bench import run_benchmark

    run_benchmark(args.models, DEMOS, think_variants=args.think_variants, enable_neural=not args.no_neural,
                  append=args.append)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--text", required=True)
    a.add_argument("--language", default="auto", choices=["auto", "en", "hinglish", "hi"])
    c = sub.add_parser("card")
    c.add_argument("patients", nargs="+", type=Path)
    d = sub.add_parser("demo")
    b = sub.add_parser("bench")
    b.add_argument("--models", nargs="+", default=["qwen3.5", "mistral"])
    b.add_argument("--think-variants", action="store_true", help="also run thinking models with think=true")
    b.add_argument("--append", action="store_true", help="merge into the existing benchmark files")
    for p in (c, d):
        p.add_argument("--backend", default="ollama", choices=["ollama", "openai", "none"])
        p.add_argument("--model", default="qwen3.5")
        p.add_argument("--think", action="store_true", help="enable reasoning for thinking models (slower)")
        p.add_argument("--base-url", default="http://localhost:8000/v1")
        p.add_argument("--api-key")
    for p in (a, c, d, b):
        p.add_argument("--no-neural", action="store_true", help="lexicon + classifier only (no m1-m4)")
    args = ap.parse_args()
    if args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "card":
        cmd_card(args, args.patients)
    elif args.cmd == "demo":
        cmd_card(args, DEMOS)
    else:
        cmd_bench(args)


if __name__ == "__main__":
    main()
