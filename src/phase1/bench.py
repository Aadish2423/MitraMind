"""Benchmark local open-weight LLMs as the State Card narrator (structured output).

For every model x demo patient it records: did the first answer parse against the Narrative
JSON schema, did it pass the guardrails (no diagnosis / treatment advice / risk talk, only
real entry ids, stressor domains and numbers), how many attempts were needed, and latency.
Writes outputs/phase1/llm_benchmark.{json,md}.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from pydantic import ValidationError

from .llm import OllamaBackend, constrain_narrative_schema, guardrail_issues, inline_schema
from .orchestrator import SYSTEM_PROMPT, Phase1Orchestrator, checkin_counts
from .schema import Narrative, PatientBundle

OUT = Path(__file__).resolve().parents[2] / "outputs" / "phase1"


def _case(orch: Phase1Orchestrator, path: Path):
    bundle = PatientBundle.model_validate_json(path.read_text(encoding="utf-8"))
    _, extra = orch.build_card(bundle, use_llm=False)
    digest, analyses = extra["digest"], extra["analyses"]
    allowed = ({a["entry_id"] for a in analyses},
               set(digest["reported_in_checkins"]["stressor_domains"]),
               set(re.findall(r"\b\d+\b", json.dumps(digest, default=str))),
               len(analyses), checkin_counts(digest))
    return bundle.patient_ref, "DATA:\n" + json.dumps(digest, ensure_ascii=False, default=str, indent=1), allowed


def run_benchmark(models: list[str], patient_files: list[Path], think_variants: bool = False,
                  enable_neural: bool = True, append: bool = False) -> list[dict]:
    orch = Phase1Orchestrator(enable_neural=enable_neural)
    cases = [_case(orch, p) for p in patient_files]
    schema = inline_schema(Narrative)
    variants = []
    for m in models:
        variants.append(OllamaBackend(m, think=False if m.startswith(("qwen3", "gemma4")) else None))
        if think_variants and m.startswith("qwen3"):
            variants.append(OllamaBackend(m, think=True))

    rows = []
    for backend in variants:
        ok, why = backend.available()
        if not ok:
            print(f"skip {backend.name}: {why}")
            continue
        backend.chat("Reply with JSON.", "Say hi.", {"type": "object", "properties": {"hi": {"type": "string"}},
                                                     "required": ["hi"]})  # warm-up: load weights
        for ref, user, (ids, doms, nums, n_checkins, counts) in cases:
            t0 = time.time()
            try:
                res = backend.chat(SYSTEM_PROMPT, user, constrain_narrative_schema(schema, sorted(ids), sorted(doms)))
                raw, err = res.raw, None
            except Exception as e:  # noqa: BLE001 - record any backend failure in the table
                raw, err = "", f"{e.__class__.__name__}: {e}"
            secs = time.time() - t0
            row = {"model": backend.name, "patient": ref, "seconds": round(secs, 1), "error": err}
            try:
                n = Narrative.model_validate_json(raw)
                issues = guardrail_issues(n, ids, doms, nums, n_checkins, counts)
                row.update(schema_valid=True, guardrail_issues=issues, passed=not issues,
                           summary_words=len(n.clinician_summary.split()), summary=n.clinician_summary)
            except ValidationError as e:
                row.update(schema_valid=False, guardrail_issues=[f"schema: {e.errors()[0]['msg']}"], passed=False,
                           summary_words=None, summary=raw[:300])
            rows.append(row)
            print(f"{backend.name:32s} {ref:22s} {secs:6.1f}s valid={row['schema_valid']} pass={row['passed']} "
                  f"{'; '.join(row['guardrail_issues'])[:120]}")

    OUT.mkdir(parents=True, exist_ok=True)
    if append and (OUT / "llm_benchmark.json").exists():
        new_models = {r["model"] for r in rows}
        old = json.loads((OUT / "llm_benchmark.json").read_text(encoding="utf-8"))
        rows = [r for r in old if r["model"] not in new_models] + rows
    (OUT / "llm_benchmark.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "llm_benchmark.md").write_text(_markdown(rows), encoding="utf-8")
    print(f"wrote {OUT / 'llm_benchmark.md'}")
    return rows


def _markdown(rows: list[dict]) -> str:
    lines = ["# LLM benchmark: State Card narrative (structured output)", "",
             "First-attempt results per demo patient (no retry). `pass` = schema-valid and no guardrail issue.", "",
             "| Model | Schema-valid | Guardrail pass | Mean latency (s) | Mean summary words |", "|---|---|---|---|---|"]
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model]
        words = [r["summary_words"] for r in rs if r["summary_words"]]
        lines.append(f"| {model} | {sum(r['schema_valid'] for r in rs)}/{len(rs)} | {sum(r['passed'] for r in rs)}/{len(rs)} "
                     f"| {sum(r['seconds'] for r in rs) / len(rs):.1f} | {sum(words) / len(words) if words else 0:.0f} |")
    lines += ["", "## Per patient", "", "| Model | Patient | s | Valid | Pass | Issues |", "|---|---|---|---|---|---|"]
    for r in rows:
        issues = "; ".join(r["guardrail_issues"]).replace("|", "/") or "-"
        lines.append(f"| {r['model']} | {r['patient']} | {r['seconds']} | {r['schema_valid']} | {r['passed']} | {issues} |")
    lines += ["", "## Summaries", ""]
    for r in rows:
        lines += [f"**{r['model']} / {r['patient']}:** {r['summary']}", ""]
    return "\n".join(lines)
