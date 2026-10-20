"""LLM backends with JSON-schema-constrained decoding, validation, retry and guardrails.

Backends (no vendor SDK needed, plain HTTP):
  ollama   local Ollama >= 0.5: /api/chat with `format=<JSON schema>` (grammar-constrained).
  openai   any OpenAI-compatible server: vLLM, llama.cpp server, LM Studio, SGLang, or a hosted
           endpoint in an Indian region, with `response_format={"type": "json_schema", ...}`.

The LLM only writes the `Narrative` part of the card. Its output is parsed with Pydantic and
checked by `guardrail_issues()`; on failure we retry once with the problems listed, then fall
back to the rule-based narrative.
"""
from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass

import requests
from pydantic import BaseModel, ValidationError


def inline_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema with $refs inlined and titles dropped (some grammar converters reject $ref)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    out = walk(schema)
    _close_objects(out)
    return out


def constrain_narrative_schema(schema: dict, entry_ids: list[str], stressor_domains: list[str]) -> dict:
    """Per-request schema: evidence ids and stressor domains become enums of what is in the data,
    so constrained decoding cannot produce an unknown check-in id or an absent stressor domain."""
    schema = copy.deepcopy(schema)
    props = schema["properties"]
    if entry_ids:
        props["evidence"]["items"]["properties"]["entry_ids"]["items"] = {"type": "string", "enum": sorted(entry_ids)}
    else:
        props["evidence"]["maxItems"] = 0
    props["stressors"]["items"]["properties"]["domain"]["enum"] = sorted(set(stressor_domains) | {"other"})
    return schema


def _close_objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node.setdefault("additionalProperties", False)
            node["required"] = list(node["properties"])
        for v in node.values():
            _close_objects(v)
    elif isinstance(node, list):
        for v in node:
            _close_objects(v)


@dataclass
class LLMResult:
    raw: str
    seconds: float
    model: str


class OllamaBackend:
    def __init__(self, model: str = "qwen3.5", host: str = "http://localhost:11434", think: bool | None = False,
                 num_ctx: int = 8192, timeout: int = 600):
        self.model, self.host, self.think, self.num_ctx, self.timeout = model, host.rstrip("/"), think, num_ctx, timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}" + ("" if self.think is None else f"(think={self.think})")

    def available(self) -> tuple[bool, str]:
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=5).json()
        except requests.RequestException as e:
            return False, f"Ollama not reachable at {self.host}: {e.__class__.__name__}"
        names = {m["name"] for m in tags.get("models", [])} | {m["name"].split(":")[0] for m in tags.get("models", [])}
        return (self.model in names, "" if self.model in names else f"model {self.model} not pulled")

    def chat(self, system: str, user: str, schema: dict, images: list[tuple[str, str]] | None = None) -> LLMResult:
        """`images`: (mime, base64) pairs attached to the user turn; needs a vision model (e.g. gemma4)."""
        user_msg = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [b64 for _, b64 in images]
        body = {"model": self.model, "stream": False, "format": schema,
                "messages": [{"role": "system", "content": system}, user_msg],
                "options": {"temperature": 0, "num_ctx": self.num_ctx, "seed": 7}}
        if self.think is not None:
            body["think"] = self.think
        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return LLMResult(r.json()["message"]["content"], time.time() - t0, self.name)


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 timeout: int = 600):
        self.model, self.base_url, self.api_key, self.timeout = model, base_url.rstrip("/"), api_key, timeout

    @property
    def name(self) -> str:
        return f"openai:{self.model}@{self.base_url}"

    def available(self) -> tuple[bool, str]:
        try:
            requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5).raise_for_status()
            return True, ""
        except requests.RequestException as e:
            return False, f"server not reachable: {e.__class__.__name__}"

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def chat(self, system: str, user: str, schema: dict, images: list[tuple[str, str]] | None = None) -> LLMResult:
        content = user if not images else [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}} for mime, b64 in images]
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "narrative", "strict": True, "schema": schema}}}
        t0 = time.time()
        r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return LLMResult(r.json()["choices"][0]["message"]["content"], time.time() - t0, self.name)


# ------------------------------------------------------------------ guardrails
_DIAGNOSIS = re.compile(
    r"\b(diagnos\w*|has (major )?depression|is depressed|suffers from|disorder|bipolar|schizophreni\w*|"
    r"ptsd|ocd|adhd|psychosis|psychotic|"
    r"(minimal|mild|moderate|moderately severe|severe|level of|major|clinical) (depression|anxiety))\b", re.I)
MAX_SUMMARY_WORDS = 110
_TREATMENT = re.compile(
    r"\b(prescrib\w*|\d+\s?mg|dosage|dose|increase (the )?(medication|dose)|start (an? )?(ssri|antidepressant)|"
    r"ssri|sertraline|fluoxetine|escitalopram|benzodiazepine)\b", re.I)
_RISK_DISMISS = re.compile(r"\b(no (suicid\w*|self-harm) risk|not at risk|low risk|risk is (low|minimal))\b", re.I)


def guardrail_issues(narrative, allowed_entry_ids: set[str], allowed_stressors: set[str],
                     allowed_numbers: set[str], n_checkins: int | None = None,
                     checkin_counts: set[int] | None = None) -> list[str]:
    """Problems that make an LLM narrative unsafe or ungrounded. Empty list = OK.

    `checkin_counts` are the per-domain check-in counts in the digest; a claim like "in 3 of 7
    check-ins" must use one of them as N and the real number of check-ins as M.
    """
    issues = []
    texts = [narrative.clinician_summary, narrative.sleep_note, *narrative.main_themes,
             *narrative.discussion_points, *narrative.coping_that_helped, *narrative.coping_concerns,
             *(s.detail for s in narrative.stressors), *(e.statement for e in narrative.evidence)]
    blob = " ".join(texts)
    if m := _DIAGNOSIS.search(blob):
        issues.append(f"diagnostic language ('{m.group(0)}'); report the score and severity band "
                      "(e.g. 'PHQ-9 17, moderately severe range') and describe reported experiences instead")
    if m := _TREATMENT.search(blob):
        issues.append(f"treatment/medication advice ('{m.group(0)}') is not allowed")
    if (words := len(narrative.clinician_summary.split())) > MAX_SUMMARY_WORDS:
        issues.append(f"clinician_summary has {words} words; keep it under 90")
    if m := _RISK_DISMISS.search(blob):
        issues.append(f"risk judgement ('{m.group(0)}'); risk is assessed separately, do not comment on it")
    bad_ids = {i for e in narrative.evidence for i in e.entry_ids} - allowed_entry_ids
    if bad_ids:
        issues.append(f"evidence cites unknown entry ids {sorted(bad_ids)}")
    bad_dom = {s.domain for s in narrative.stressors} - allowed_stressors - {"other"}
    if bad_dom:
        issues.append(f"stressor domains {sorted(bad_dom)} are not in the cue data")
    summary = narrative.clinician_summary.lower()
    nums = re.findall(r"\b\d+\b", summary) + [str(_NUMBER_WORDS[w]) for w in re.findall(r"\b[a-z]+\b", summary)
                                              if w in _NUMBER_WORDS]
    for num in nums:
        if num not in allowed_numbers:
            issues.append(f"number {num} in the summary does not appear in the data")
            break
    if n_checkins is not None and checkin_counts is not None:
        for n_txt, m_txt in _N_OF_M.findall(blob.lower()):
            n, m = _to_int(n_txt), _to_int(m_txt)
            if n is None or m is None:  # "any of 4 check-ins", "none of the check-ins"
                continue
            if m != n_checkins or n not in checkin_counts:
                issues.append(f"'{n_txt} of {m_txt} check-ins' does not match the data "
                              f"({n_checkins} check-ins; per-domain counts {sorted(checkin_counts)})")
                break
    return issues


def _to_int(token: str) -> int | None:
    return int(token) if token.isdigit() else {"one": 1, **_NUMBER_WORDS}.get(token)


_N_OF_M = re.compile(r"\b(\d+|[a-z]+)\s+(?:of|out of)\s+(?:the\s+|all\s+)?(\d+|[a-z]+)\s+(?:check-?ins?|entries)\b")


_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split()) if i >= 2}  # "one" is too common ("one of the stressors")


def generate(backend, system: str, user: str, model: type[BaseModel], check, max_attempts: int = 2,
             schema: dict | None = None, images: list[tuple[str, str]] | None = None):
    """Call the backend until the output parses and passes `check` (returns issue list)."""
    schema = schema or inline_schema(model)
    log, prompt = [], user
    for attempt in range(1, max_attempts + 1):
        try:
            res = backend.chat(system, prompt, schema, images=images) if images else backend.chat(system, prompt, schema)
        except requests.RequestException as e:
            log.append({"attempt": attempt, "error": f"request failed: {e.__class__.__name__}: {e}"})
            break
        entry = {"attempt": attempt, "seconds": round(res.seconds, 1)}
        try:
            obj = model.model_validate_json(_strip_fences(res.raw))
        except ValidationError as e:
            entry["error"] = f"schema validation failed: {e.error_count()} error(s): {e.errors()[0]['msg']}"
            issues = [entry["error"]]
        else:
            issues = check(obj)
            if not issues:
                log.append(entry)
                return obj, log
            entry["guardrail_issues"] = issues
        log.append(entry)
        prompt = (user + "\n\nYour previous answer had these problems; fix them and answer again:\n- "
                  + "\n- ".join(issues))
    return None, log


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    return raw
