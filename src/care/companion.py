"""The between-visit companion: one reply per patient message, safety layer first.

Order of operations for every message (text or voice transcript):
  1. Phase 1 analysis (cue models + crisis layer), done by the caller (CareService).
  2. risk high / urgent  -> fixed, reviewed crisis reply with helplines; the LLM is never called.
  3. otherwise           -> LLM reply (JSON-schema constrained) -> guardrails -> 1 retry -> fixed fallback.
The LLM sees the recent conversation and a few cue hints, never the patient's identity.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from typing import Literal

import requests
from pydantic import BaseModel, Field, ValidationError

from phase1.llm import (_DIAGNOSIS, _TREATMENT, _strip_fences, generate, inline_schema, partial_json_string,
                        take_sentences)

MAX_REPLY_WORDS = 110
# The history window starts at a multiple of HISTORY_STEP messages instead of sliding every turn, so consecutive
# prompts begin with the same text and Ollama can sometimes reuse its cached reading of it. Measured on CPU
# (Gemma 4 E4B, ~36 prompt tokens/s): a full re-read of ~580 tokens takes ~15 s, a cache hit ~3 s. Gemma's
# sliding-window attention only lets llama.cpp resume from saved checkpoints, so hits are not guaranteed; a short
# window (4 to 9 messages) keeps the misses cheaper. On a GPU the re-read takes about a second either way.
HISTORY_MIN, HISTORY_STEP = 4, 6
HISTORY_MAX = HISTORY_MIN + HISTORY_STEP


def history_offset(n_messages: int) -> int:
    """Index of the first stored message the companion's history window includes."""
    return max(0, (n_messages - HISTORY_MIN) // HISTORY_STEP * HISTORY_STEP)


class CompanionReply(BaseModel):
    reply: str = Field(max_length=900, description="what MitraMind says to the patient")
    mind_pause: Literal["none", "box_breathing", "grounding_54321", "body_scan"] = Field(
        description="offer a Mind Pause exercise only if the patient sounds anxious, overwhelmed or unable to sleep")


class CompanionPhotoReply(CompanionReply):
    """Used only when the patient attached a photo, so text-only replies keep their tested schema."""
    image_concern: bool = Field(
        False, description="true only if the photo shows an injury, self-harm, a weapon, a large amount of medicine "
                           "or anything else suggesting the patient may be in danger")


SYSTEM_PROMPT = """You are MitraMind, a warm and calm companion app. The patient's hospital care team prescribed you \
for the time between clinic visits. You are not a therapist or a doctor and never say you are.

How to reply:
- Use the same language and script as the patient's latest message: English, Hinglish (Hindi in Latin letters) or \
Hindi (Devanagari).
- 2 to 5 short sentences, under 90 words. First reflect what they said and validate the feeling. Then give at most \
one gentle question or one small suggestion.
- Suggestions may only be: a Mind Pause exercise, simple sleep routine steps, reaching out to someone they trust, \
writing a reflection, or bringing the topic to their next clinic visit.
- Never diagnose and never name disorders. Never give medicine, dose or treatment advice and never tell them to \
start, stop or change treatment; say their doctor is the right person and suggest noting the question for the visit.
- Do not promise secrecy. If asked, say chats stay private unless they choose to share them, but anything that \
suggests they may be in danger is passed to their care team.
- If they mention wanting to die or hurting themselves, tell them Tele-MANAS 14416 (free, 24x7) and 112 are there \
right now.
- Set mind_pause to an exercise only when they sound anxious or overwhelmed (box_breathing), panicky \
(grounding_54321) or unable to sleep (body_scan); otherwise "none"."""

_ROLE_CLAIM = re.compile(r"\b(i am|i'm|main|mai) (a |an |your |aapka |aapki )?"
                         r"(therapist|doctor|psychiatrist|psychologist|counsell?or)\b", re.I)
_SECRECY = re.compile(r"\b(i won't tell|i will not tell|no one will (ever )?know|stays? between us|"
                      r"kisi ko nahi bataunga|kisi ko nahi bataungi)\b", re.I)

CRISIS_REPLY = {
    "en": ("Thank you for telling me. I'm really sorry you're feeling this way, and you don't have to face it alone. "
           "Please call Tele-MANAS now on 14416 (free, 24x7) to talk to a trained counsellor. If you are in immediate "
           "danger, call 112. I have let your care team know so they can reach out to you."),
    "hinglish": ("Mujhe batane ke liye shukriya. Mujhe bahut afsos hai ki aap aisa mehsoos kar rahe ho, aur aapko "
                 "yeh akele nahi jhelna hai. Please abhi Tele-MANAS ko 14416 par call karo (free, 24x7), wahan trained "
                 "counsellor se baat hogi. Agar aap turant khatre mein ho toh 112 par call karo. Maine aapki care team "
                 "ko bata diya hai, woh aapse contact karenge."),
    "hi": ("मुझे बताने के लिए धन्यवाद। मुझे बहुत दुख है कि आप ऐसा महसूस कर रहे हैं, और आपको यह अकेले नहीं सहना है। "
           "कृपया अभी Tele-MANAS को 14416 पर कॉल करें (मुफ़्त, 24x7), वहाँ प्रशिक्षित काउंसलर से बात होगी। अगर आप "
           "तुरंत खतरे में हैं तो 112 पर कॉल करें। मैंने आपकी केयर टीम को सूचित कर दिया है ताकि वे आपसे संपर्क कर सकें।"),
}
URGENT_EXTRA = {
    "en": " If you can, move away from anything you could use to hurt yourself and stay near someone you trust.",
    "hinglish": " Agar ho sake, toh jis cheez se aap khud ko nuksaan pahuncha sakte ho usse door ho jao, aur kisi "
                "bharose wale insaan ke paas raho.",
    "hi": " अगर हो सके, तो जिस चीज़ से आप खुद को नुकसान पहुँचा सकते हैं उससे दूर हो जाएँ, और किसी भरोसेमंद व्यक्ति के "
          "पास रहें।",
}
FALLBACK_REPLY = {
    "en": "Thank you for sharing this with me. It sounds like a lot to carry. Would you like to try a short Mind "
          "Pause, or tell me a little more about what has been hardest today?",
    "hinglish": "Share karne ke liye shukriya. Lagta hai aap bahut kuch jhel rahe ho. Kya aap ek chhota sa Mind "
                "Pause try karna chahoge, ya mujhe thoda aur bataoge ki aaj sabse mushkil kya laga?",
    "hi": "मुझसे यह साझा करने के लिए धन्यवाद। लगता है आप बहुत कुछ सह रहे हैं। क्या आप एक छोटा सा Mind Pause "
          "करना चाहेंगे, या मुझे थोड़ा और बताएँगे कि आज सबसे मुश्किल क्या लगा?",
}
# Someone else at risk: the patient is not flagged, but the reply must give the helplines.
THIRD_PARTY = re.compile(
    r"\b(friend|brother|sister|mother|father|mom|mum|dad|son|daughter|husband|wife|partner|cousin|someone|"
    r"dost|bhai|behen|didi)\b.{0,80}\b(kill (him|her|them)sel(f|ves)|end (his|her|their) li(fe|ves)|suicid\w*|"
    r"wants? to die|khudkushi|aatmahatya|marna chahta|marna chahti)", re.I)
THIRD_PARTY_NOTE = ("Safety note: the patient is worried that someone else may be at risk of suicide. Acknowledge "
                    "how frightening that is, suggest they stay with the person if it is safe, and give Tele-MANAS "
                    "14416 (free, 24x7) and 112 for an emergency. Do not ask for details about the method.")
HELPLINE_LINE = {
    "en": "If things ever feel too heavy, Tele-MANAS is free on 14416, any time, and 112 is there in an emergency.",
    "hinglish": "Agar kabhi sab bahut bhaari lage, Tele-MANAS 14416 par free mein kabhi bhi baat kar sakte ho, "
                "aur emergency mein 112.",
    "hi": "अगर कभी सब बहुत भारी लगे, तो Tele-MANAS 14416 पर कभी भी मुफ़्त बात कर सकते हैं, और आपात स्थिति में 112।",
}
_HELPLINE = re.compile(r"14416")
WATCH_NOTE = ("Safety note: the patient's latest message contains hopelessness or burden phrases. Respond with extra "
              "care, gently ask how they are coping right now, and mention that Tele-MANAS 14416 is free and "
              "available any time if things feel too heavy.")
PHOTO_NOTE = ("Photo: the patient attached a photo to their latest message (you can see it). Say something warm and "
              "specific about what it shows and how it might comfort or support them; if its meaning is unclear, "
              "gently ask what it means to them. Do not comment on their body, weight or looks, and do not guess who "
              "people in it are. Set image_concern to true only if the photo shows an injury, self-harm, a weapon, a "
              "large amount of medicine or anything else suggesting they may be in danger, and then also mention "
              "Tele-MANAS 14416 (free, 24x7) and 112.")
PHOTO_ONLY = "[shared a photo]"


def crisis_reply(level: str, language: str) -> str:
    lang = language if language in CRISIS_REPLY else "en"
    return CRISIS_REPLY[lang] + (URGENT_EXTRA[lang] if level == "urgent" else "")


def fallback_reply(language: str) -> str:
    return FALLBACK_REPLY.get(language, FALLBACK_REPLY["en"])


def text_issues(text: str) -> list[str]:
    """Content guardrails that can be checked on any piece of a reply, including a single streamed sentence."""
    issues = []
    if m := _DIAGNOSIS.search(text):
        issues.append(f"diagnostic language ('{m.group(0)}'); describe feelings in everyday words instead")
    if m := _TREATMENT.search(text):
        issues.append(f"medication/treatment advice ('{m.group(0)}'); say their doctor is the right person")
    if m := _ROLE_CLAIM.search(text):
        issues.append(f"claims a clinical role ('{m.group(0)}'); you are a companion app")
    if m := _SECRECY.search(text):
        issues.append(f"promises secrecy ('{m.group(0)}'); safety concerns go to the care team")
    return issues


def reply_issues(r: CompanionReply) -> list[str]:
    issues = text_issues(r.reply)
    if (words := len(r.reply.split())) > MAX_REPLY_WORDS:
        issues.append(f"reply has {words} words; keep it under 90")
    if not r.reply.strip():
        issues.append("empty reply")
    return issues


def build_prompt(history: list[dict], message: str, language: str, analysis: dict, photo: bool = False) -> str:
    lines = ["Recent conversation (oldest first):"]
    for m in history[-HISTORY_MAX:]:
        text = m["text"] or (PHOTO_ONLY if m.get("image_path") else "")
        lines.append(f"{'Patient' if m['role'] == 'user' else 'MitraMind'}: {text}")
    lines.append(f"Patient (latest): {message or PHOTO_ONLY}")
    hints = [f"language: {language}"]
    if analysis.get("emotion"):
        hints.append(f"emotion cue: {analysis['emotion']['top_cue']}")
    lex = analysis.get("lexicon") or {}
    if lex.get("stressors"):
        hints.append("stressors mentioned: " + ", ".join(lex["stressors"]))
    if lex.get("sleep"):
        hints.append("sleep problems mentioned")
    lines.append("\nCue hints (from the cue models, may be wrong): " + "; ".join(hints))
    if analysis["risk"].level == "watch":
        lines.append("\n" + WATCH_NOTE)
    if THIRD_PARTY.search(message):
        lines.append("\n" + THIRD_PARTY_NOTE)
    if photo:
        lines.append("\n" + PHOTO_NOTE)
    return "\n".join(lines)


def needs_helpline(message: str, analysis: dict) -> bool:
    return analysis["risk"].level == "watch" or bool(THIRD_PARTY.search(message))


def with_helpline(reply: str, language: str) -> str:
    """Deterministic: a watch-level or someone-else-at-risk reply always carries the helpline."""
    if _HELPLINE.search(reply):
        return reply
    return reply.rstrip() + " " + HELPLINE_LINE.get(language, HELPLINE_LINE["en"])


class Companion:
    def __init__(self, backend=None):
        self.backend = backend
        self.schema = inline_schema(CompanionReply)
        self.photo_schema = inline_schema(CompanionPhotoReply)

    def respond(self, history: list[dict], message: str, analysis: dict,
                images: list[tuple[str, str]] | None = None) -> dict:
        """Returns {"reply", "mind_pause", "image_concern", "source", "model", "attempts", "log", "prompt"}.

        `images`: (mime, base64) photos attached to this message. The text safety layer still runs first; the
        LLM's image_concern flag is an extra, weaker signal for photos, never a replacement for it.
        """
        risk, language = analysis["risk"], analysis["language"]
        if risk.level in ("high", "urgent"):
            return {"reply": crisis_reply(risk.level, language), "mind_pause": "none", "image_concern": False,
                    "source": "crisis_protocol", "model": None, "attempts": 0, "log": [], "prompt": message}
        prompt = build_prompt(history, message, language, analysis, photo=bool(images))
        out, log = None, []
        if self.backend is not None:
            ok, why = self.backend.available()
            if ok:
                model, schema = (CompanionPhotoReply, self.photo_schema) if images else (CompanionReply, self.schema)
                out, log = generate(self.backend, SYSTEM_PROMPT, prompt, model, reply_issues, schema=schema,
                                    images=images)
            else:
                log = [{"error": why}]
        image_concern = bool(getattr(out, "image_concern", False))
        helpline = needs_helpline(message, analysis) or image_concern
        if out is None:
            reply = fallback_reply(language)
            return {"reply": with_helpline(reply, language) if helpline else reply, "mind_pause": "box_breathing",
                    "image_concern": False, "source": "fallback", "model": getattr(self.backend, "name", None),
                    "attempts": len(log), "log": log, "prompt": prompt}
        reply = out.reply.strip()
        return {"reply": with_helpline(reply, language) if helpline else reply, "mind_pause": out.mind_pause,
                "image_concern": image_concern, "source": "llm", "model": self.backend.name, "attempts": len(log),
                "log": log, "prompt": prompt}

    def respond_stream(self, history: list[dict], message: str, analysis: dict,
                       images: list[tuple[str, str]] | None = None) -> Iterator[dict]:
        """Like respond(), but yields the reply as it is written, so the app can show and speak it sooner.

        Events: {"type": "sentence", "text"} for each sentence, optionally {"type": "replace", "text"}, and last
        {"type": "final", **respond()-style result}. Every sentence passes the content guardrails before it is sent.
        If one fails, or the finished reply fails validation, streaming stops and the reply from the normal
        retry / fallback path replaces what was shown. Crisis replies and missing backends go through respond().
        """
        risk, language = analysis["risk"], analysis["language"]
        backend = self.backend
        if risk.level in ("high", "urgent") or backend is None or not hasattr(backend, "chat_stream"):
            yield from self._whole(self.respond(history, message, analysis, images), shown=False)
            return
        ok, why = backend.available()
        if not ok:
            yield from self._whole(self.respond(history, message, analysis, images), shown=False)
            return
        prompt = build_prompt(history, message, language, analysis, photo=bool(images))
        model, schema = (CompanionPhotoReply, self.photo_schema) if images else (CompanionReply, self.schema)
        raw, done_upto, shown, problem, t0 = "", 0, [], None, time.time()
        try:
            for chunk in backend.chat_stream(SYSTEM_PROMPT, prompt, schema, images=images):
                raw += chunk
                text, closed = partial_json_string(raw, "reply")
                sentences, rest = take_sentences(text[done_upto:], final=closed)
                done_upto = len(text) - len(rest)
                for s in sentences:
                    if issues := text_issues(s):
                        problem = issues
                        break
                    shown.append(s)
                    yield {"type": "sentence", "text": s}
                if problem:
                    break
        except requests.RequestException as e:
            problem = [f"stream failed: {e.__class__.__name__}"]
        obj = None
        if not problem:
            try:
                obj = model.model_validate_json(_strip_fences(raw))
                problem = [i for i in reply_issues(obj) if "words; keep it under" not in i] or None
            except ValidationError as e:
                problem = [f"schema validation failed: {e.error_count()} error(s)"]
        stream_log = {"attempt": 1, "seconds": round(time.time() - t0, 1), "streamed": True}
        if problem:
            out = self.respond(history, message, analysis, images)
            out["log"] = [{**stream_log, "guardrail_issues": problem}] + out["log"]
            yield from self._whole(out, shown=bool(shown))
            return
        image_concern = bool(getattr(obj, "image_concern", False))
        reply = obj.reply.strip()
        if (needs_helpline(message, analysis) or image_concern) and not _HELPLINE.search(reply):
            extra = HELPLINE_LINE.get(language, HELPLINE_LINE["en"])
            yield {"type": "sentence", "text": extra}
            reply = f"{reply} {extra}"
        yield {"type": "final", "reply": reply, "mind_pause": obj.mind_pause, "image_concern": image_concern,
               "source": "llm", "model": backend.name, "attempts": 1, "log": [stream_log], "prompt": prompt}

    @staticmethod
    def _whole(out: dict, shown: bool) -> Iterator[dict]:
        """Send a finished reply: as sentences if nothing was shown yet, else as one replacement."""
        if shown:
            yield {"type": "replace", "text": out["reply"]}
        else:
            for s in take_sentences(out["reply"], final=True)[0]:
                yield {"type": "sentence", "text": s}
        yield {"type": "final", **out}
