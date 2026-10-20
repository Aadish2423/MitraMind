"""Streaming companion replies: sentence events, per-sentence guardrails, replacement, crisis path, NDJSON route.

Run: python -m pytest tests
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from care.routes import build_router  # noqa: E402
from care.service import CareService  # noqa: E402
from care.store import Store  # noqa: E402
from phase1.llm import LLMResult, partial_json_string, take_sentences  # noqa: E402
from phase1.orchestrator import Phase1Orchestrator  # noqa: E402

ORCH = Phase1Orchestrator(enable_neural=False)
CONSENTS = {"companion": True, "care_team": True}


class StreamBackend:
    """Streams `raw` in small pieces; `answers` feed the non-streaming retry/fallback path."""
    name = "fake-stream"

    def __init__(self, raw, answers=()):
        self.raw, self.answers, self.calls = raw, list(answers), []

    def available(self):
        return True, ""

    def chat_stream(self, system, user, schema, images=None):
        self.calls.append("stream")
        for i in range(0, len(self.raw), 7):
            yield self.raw[i:i + 7]

    def chat(self, system, user, schema, images=None):
        self.calls.append("chat")
        return LLMResult(self.answers.pop(0), 0.0, "fake")


def reply(text, pause="none"):
    return json.dumps({"reply": text, "mind_pause": pause})


def service(backend):
    svc = CareService(Store(":memory:"), ORCH, backend)
    svc.enroll("p1", consents=CONSENTS)
    return svc


def run(svc, text):
    events = list(svc.chat_stream("p1", text))
    return [e for e in events if e["type"] != "final"], events[-1]


def test_partial_json_string():
    assert partial_json_string('{"reply": "He said \\"hi\\"\\nok', "reply") == ('He said "hi"\nok', False)
    assert partial_json_string('{"reply": "caf\\u00e9.", "mind_pause": "none"}', "reply") == ("café.", True)
    assert partial_json_string('{"reply": "half \\', "reply") == ("half ", False)
    assert partial_json_string('{"rep', "reply") == ("", False)


def test_take_sentences():
    assert take_sentences("That sounds hard. Would you like a pause? Main samajh") == (
        ["That sounds hard.", "Would you like a pause?"], "Main samajh")
    assert take_sentences("बहुत मुश्किल है। क्या आप") == (["बहुत मुश्किल है।"], "क्या आप")
    assert take_sentences("Last bit", final=True) == (["Last bit"], "")


def test_sentences_stream_then_final_is_stored():
    text = "That sounds really tiring. Work can pile up. Would you like a short Mind Pause?"
    svc = service(StreamBackend(reply(text, "box_breathing")))
    events, final = run(svc, "work was so stressful today")
    assert [e["text"] for e in events] == ["That sounds really tiring.", "Work can pile up.",
                                           "Would you like a short Mind Pause?"]
    assert final["reply"] == text and final["reply_source"] == "llm" and final["mind_pause"] == "box_breathing"
    assert svc.history("p1")[-1]["text"] == text


def test_unsafe_sentence_is_never_sent():
    backend = StreamBackend(reply("That sounds hard. It sounds like an anxiety disorder. Talk to me."),
                            answers=[reply("That sounds really hard. Would it help to note this for your visit?")])
    svc = service(backend)
    events, final = run(svc, "i feel low every day")
    sent = " ".join(e["text"] for e in events if e["type"] == "sentence")
    assert sent == "That sounds hard." and "disorder" not in sent
    assert events[-1] == {"type": "replace", "text": final["reply"]}
    assert "disorder" not in final["reply"] and backend.calls == ["stream", "chat"]
    assert svc.history("p1")[-1]["text"] == final["reply"]


def test_broken_output_is_replaced():
    backend = StreamBackend('{"reply": "Okay. I hear you', answers=[reply("I hear you. Take your time.")])
    events, final = run(service(backend), "rough day")
    assert events[0] == {"type": "sentence", "text": "Okay."}
    assert events[-1]["type"] == "replace" and final["reply"] == "I hear you. Take your time."


def test_crisis_message_skips_the_llm():
    backend = StreamBackend(reply("should never be used"))
    svc = service(backend)
    events, final = run(svc, "main jeena nahi chahta")
    assert backend.calls == [] and final["reply_source"] == "crisis_protocol" and final["escalate"]
    assert "14416" in " ".join(e["text"] for e in events)
    assert len(svc.store.alerts(patient_id="p1")) == 1


def test_history_window_moves_in_steps():
    from care.companion import history_offset
    assert [history_offset(n) for n in (0, 6, 9, 10, 16, 21, 22)] == [0, 0, 0, 6, 12, 12, 18]


def test_consecutive_prompts_share_their_beginning():
    """So Ollama can reuse its cached reading of the conversation instead of re-reading it every turn."""
    answers = [reply("I hear you."), reply("That makes sense."), reply("Thank you for sharing.")]

    class Recorder(StreamBackend):
        prompts = []

        def chat_stream(self, system, user, schema, images=None):
            self.prompts.append(user)
            yield from super().chat_stream(system, user, schema, images)

    backend = Recorder("")
    svc = service(backend)
    for text, ans in zip(("long day", "work was hard", "feeling a bit better"), answers):
        backend.raw = ans
        list(svc.chat_stream("p1", text))
    p = Recorder.prompts
    for earlier, later in zip(p, p[1:]):
        assert later.startswith(earlier.split("Patient (latest):")[0])


def test_route_streams_ndjson_and_is_guarded():
    svc = CareService(Store(":memory:"), ORCH, StreamBackend(reply("I'm glad you wrote. What helped today?")))
    app = FastAPI()
    app.include_router(build_router(svc))
    c = TestClient(app)
    s = c.post("/v1/auth/signup", json={"email": "s@example.com", "password": "stream-test-1", "consents": CONSENTS}).json()
    pid = s["profile"]["id"]
    assert c.post(f"/v1/patients/{pid}/chat/stream", json={"text": "hi"}).status_code == 401
    r = c.post(f"/v1/patients/{pid}/chat/stream", json={"text": "hi"}, headers={"Authorization": f"Bearer {s['token']}"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in r.text.splitlines() if line]
    assert [e["type"] for e in lines] == ["sentence", "sentence", "final"] and lines[-1]["message_id"]


class FakeSTT:
    name = "fake-stt"

    def __init__(self, text):
        self.text = text

    def transcribe(self, wav_path, language=None):
        return {"text": self.text, "seconds": 2.0, "engine": self.name}


def voice_service(backend, heard, tmp_dir):
    import care.voice as voice
    voice.AUDIO_DIR = pathlib.Path(tmp_dir)
    svc = CareService(Store(":memory:"), ORCH, backend, stt=FakeSTT(heard))
    svc.enroll("p1", consents=CONSENTS)
    return svc


def test_voice_stream_sends_transcript_then_sentences(tmp_path):
    svc = voice_service(StreamBackend(reply("That sounds exhausting. Want to try a body scan?", "body_scan")),
                        "I couldn't sleep, exams tomorrow", tmp_path)
    events = list(svc.voice_stream("p1", b"RIFF fake wav"))
    assert events[0] == {"type": "transcript", "text": "I couldn't sleep, exams tomorrow", "transcript":
                         "I couldn't sleep, exams tomorrow", "stt_engine": "fake-stt", "audio_seconds": 2.0}
    assert [e["text"] for e in events[1:-1]] == ["That sounds exhausting.", "Want to try a body scan?"]
    final = events[-1]
    assert final["type"] == "final" and final["transcript"] == "I couldn't sleep, exams tomorrow"
    user = [m for m in svc.history("p1") if m["role"] == "user"][-1]
    assert user["modality"] == "voice" and user["text"] == "I couldn't sleep, exams tomorrow"


def test_spoken_crisis_still_skips_the_llm(tmp_path):
    backend = StreamBackend(reply("should never be used"))
    svc = voice_service(backend, "I don't want to live anymore", tmp_path)
    final = list(svc.voice_stream("p1", b"RIFF fake wav"))[-1]
    assert backend.calls == [] and final["reply_source"] == "crisis_protocol" and final["alert_id"]


def test_empty_speech_is_rejected_before_streaming(tmp_path):
    import pytest
    svc = voice_service(StreamBackend(reply("x")), "", tmp_path)
    with pytest.raises(ValueError, match="no speech"):
        svc.voice_stream("p1", b"RIFF fake wav")


def test_voice_stream_route_is_guarded(tmp_path):
    import base64
    import care.voice as voice
    voice.AUDIO_DIR = pathlib.Path(tmp_path)
    svc = CareService(Store(":memory:"), ORCH, StreamBackend(reply("I hear you. Thank you for telling me.")),
                      stt=FakeSTT("long day today"))
    app = FastAPI()
    app.include_router(build_router(svc))
    c = TestClient(app)
    s = c.post("/v1/auth/signup", json={"email": "v@example.com", "password": "voice-test-1", "consents": CONSENTS}).json()
    pid, body = s["profile"]["id"], {"audio_base64": base64.b64encode(b"RIFF fake wav").decode()}
    assert c.post(f"/v1/patients/{pid}/voice/stream", json=body).status_code == 401
    r = c.post(f"/v1/patients/{pid}/voice/stream", json=body, headers={"Authorization": f"Bearer {s['token']}"})
    types = [json.loads(line)["type"] for line in r.text.splitlines() if line]
    assert r.status_code == 200 and types == ["transcript", "sentence", "sentence", "final"]
