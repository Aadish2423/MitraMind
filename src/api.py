"""MitraMind HTTP API (FastAPI): the backend the app / WhatsApp channel and clinician dashboard call.

    uvicorn api:app --app-dir src --port 8080
    MITRAMIND_LLM=ollama:qwen3.5          # State Card narrator: or openai:<model>@<base_url>, or none
    MITRAMIND_CHAT_LLM=ollama:gemma4:e4b  # companion replies (default; red-team winner)
    MITRAMIND_DB=var/mitramind.db         # SQLite file for patients, messages, alerts, audit log
    MITRAMIND_KEEP_ALIVE=30m              # how long Ollama keeps the companion model in memory between messages
    MITRAMIND_WARMUP=1                    # load the companion model and text cue models when the server starts

Patient app and clinician routes: src/care/routes.py (see /docs). Pages: / (patient app),
/clinician (dashboard), /tester (Phase 1 tester).

POST /v1/checkins/analyze   one check-in -> cues + real-time crisis triage (no LLM)
POST /v1/state-card         patient bundle -> User State Card (JSON) and pre-visit summary (Markdown)
POST /v1/questionnaires/score  PHQ-9 / GAD-7 responses -> totals, severity, trends
GET  /health
GET  /v1/demo-patients[/{name}]  the fictional demo bundles in examples/patients
"""
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from care.routes import build_router
from care.service import CareService
from care.store import Store
from care.voice import make_stt
from phase1.llm import OllamaBackend, OpenAICompatBackend
from phase1.orchestrator import Phase1Orchestrator
from phase1.questionnaires import trends
from phase1.render import card_to_markdown
from phase1.schema import Entry, EntryRisk, PatientBundle, QuestionnaireResponse, UserStateCard


def _backend(var: str = "MITRAMIND_LLM", default: str = "ollama:qwen3.5", keep_alive: str | None = None):
    spec = os.environ.get(var, default)
    kind, _, rest = spec.partition(":")
    if kind == "none":
        return None
    if kind == "openai":
        model, _, base = rest.partition("@")
        return OpenAICompatBackend(model, base or "http://localhost:8000/v1", os.environ.get("MITRAMIND_LLM_KEY"))
    return OllamaBackend(rest or "qwen3.5", keep_alive=keep_alive)


app = FastAPI(title="MitraMind Care", version="1.1")
orch = Phase1Orchestrator(backend=_backend(), enable_neural=os.environ.get("MITRAMIND_NEURAL", "1") == "1")
# Companion default: Gemma 4 E4B passed 19/19 red-team cases at 7.8 s/reply (Qwen3.5 9B: 18/19, 11.1 s),
# see outputs/redteam/companion_redteam.md. It stays loaded between messages (MITRAMIND_KEEP_ALIVE) because the
# companion is the latency-sensitive path; the State Card narrator keeps Ollama's 5-minute default.
svc = CareService(Store(), orch, _backend("MITRAMIND_CHAT_LLM", "ollama:gemma4:e4b",
                                          keep_alive=os.environ.get("MITRAMIND_KEEP_ALIVE", "30m")), make_stt())


def _warm_up() -> None:
    """Load the companion model and the text cue models in the background, so the first message isn't slow."""
    backend = svc.companion.backend
    if hasattr(backend, "warm"):
        try:
            backend.warm()
        except Exception as e:  # Ollama not running yet: the first message will load it instead
            print(f"[warm-up] companion model not loaded: {e.__class__.__name__}", flush=True)
    for model_id in ("m2_hinglish_text", "m1_hindi_text"):
        orch.registry.get(model_id)


if os.environ.get("MITRAMIND_WARMUP", "1") == "1":
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
app.include_router(build_router(svc))
WEB = Path(__file__).resolve().parent / "web"
DEMO = Path(__file__).resolve().parents[1] / "examples" / "patients"
app.mount("/static", StaticFiles(directory=WEB), name="static")


class CheckinResult(BaseModel):
    entry_id: str
    language: str
    models_used: list[str]
    risk: EntryRisk
    emotion: dict | None
    cues: dict | None
    coping: dict


class CardResponse(BaseModel):
    card: UserStateCard
    markdown: str


@app.get("/", include_in_schema=False)
def patient_app():
    # no-cache: browsers re-check the page on every visit, so new ?v= asset versions reach patients right away.
    return FileResponse(WEB / "app.html", headers={"Cache-Control": "no-cache"})


@app.get("/clinician", include_in_schema=False)
def clinician_page():
    return FileResponse(WEB / "clinician.html")


@app.get("/tester", include_in_schema=False)
def test_page():
    return FileResponse(WEB / "index.html")


@app.get("/v1/demo-patients")
def demo_patients():
    return sorted(p.stem for p in DEMO.glob("*.json"))


@app.get("/v1/demo-patients/{name}", response_model=PatientBundle)
def demo_patient(name: str):
    path = DEMO / f"{name}.json"
    if path.parent != DEMO or not path.exists():
        raise HTTPException(404, f"no demo patient {name}")
    return PatientBundle.model_validate_json(path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    ok, why = orch.backend.available() if orch.backend else (False, "LLM disabled")
    return {"status": "ok", "llm": orch.backend.name if orch.backend else None, "llm_ready": ok, "llm_note": why,
            "crisis_classifier": orch.classifier.available, "models_unavailable": orch.registry.unavailable}


@app.post("/v1/checkins/analyze", response_model=CheckinResult)
def analyze(entry: Entry):
    a = orch.analyze_entry(entry)
    return CheckinResult(entry_id=a["entry_id"], language=a["language"], models_used=a["models_used"],
                         risk=a["risk"], emotion=a["emotion"], cues=a["lexicon"], coping=a["coping"])


@app.post("/v1/state-card", response_model=CardResponse)
def state_card(bundle: PatientBundle, use_llm: bool = True):
    card, _ = orch.build_card(bundle, use_llm=use_llm)
    return CardResponse(card=card, markdown=card_to_markdown(card))


@app.post("/v1/questionnaires/score")
def score(responses: list[QuestionnaireResponse]):
    return {k: v.model_dump() for k, v in trends(responses).items()}
