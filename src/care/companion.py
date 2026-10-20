"""The between-visit companion: one reply per patient message, safety layer first.

Order of operations for every message (text or voice transcript):
  1. Phase 1 analysis (cue models + crisis layer), done by the caller (CareService).
  2. risk high / urgent  -> fixed, reviewed crisis reply with helplines; the LLM is never called.
  3. otherwise           -> LLM reply (JSON-schema constrained) -> guardrails -> 1 retry -> fixed fallback.
The LLM sees the recent conversation and a few cue hints, never the patient's identity.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from phase1.llm import _DIAGNOSIS, _TREATMENT, generate, inline_schema

MAX_REPLY_WORDS = 110
HISTORY_TURNS = 8


class CompanionReply(BaseModel):
    reply: str = Field(max_length=900, description="what MitraMind says to the patient")
    mind_pause: Literal["none", "box_breathing", "grounding_54321", "body_scan"] = Field(
        description="offer a Mind Pause exercise only if the patient sounds anxious, overwhelmed or unable to sleep")


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
WATCH_NOTE = ("Safety note: the patient's latest message contains hopelessness or burden phrases. Respond with extra "
              "care, gently ask how they are coping right now, and mention that Tele-MANAS 14416 is free and "
              "available any time if things feel too heavy.")


def crisis_reply(level: str, language: str) -> str:
    lang = language if language in CRISIS_REPLY else "en"
    return CRISIS_REPLY[lang] + (URGENT_EXTRA[lang] if level == "urgent" else "")


def fallback_reply(language: str) -> str:
    return FALLBACK_REPLY.get(language, FALLBACK_REPLY["en"])


def reply_issues(r: CompanionReply) -> list[str]:
    issues = []
    if m := _DIAGNOSIS.search(r.reply):
        issues.append(f"diagnostic language ('{m.group(0)}'); describe feelings in everyday words instead")
    if m := _TREATMENT.search(r.reply):
        issues.append(f"medication/treatment advice ('{m.group(0)}'); say their doctor is the right person")
    if m := _ROLE_CLAIM.search(r.reply):
        issues.append(f"claims a clinical role ('{m.group(0)}'); you are a companion app")
    if m := _SECRECY.search(r.reply):
        issues.append(f"promises secrecy ('{m.group(0)}'); safety concerns go to the care team")
    if (words := len(r.reply.split())) > MAX_REPLY_WORDS:
        issues.append(f"reply has {words} words; keep it under 90")
    if not r.reply.strip():
        issues.append("empty reply")
    return issues


def build_prompt(history: list[dict], message: str, language: str, analysis: dict) -> str:
    lines = ["Recent conversation (oldest first):"]
    for m in history[-HISTORY_TURNS:]:
        lines.append(f"{'Patient' if m['role'] == 'user' else 'MitraMind'}: {m['text']}")
    lines.append(f"Patient (latest): {message}")
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
    return "\n".join(lines)


class Companion:
    def __init__(self, backend=None):
        self.backend = backend
        self.schema = inline_schema(CompanionReply)

    def respond(self, history: list[dict], message: str, analysis: dict) -> dict:
        """Returns {"reply", "mind_pause", "source", "model", "attempts", "log", "prompt"}."""
        risk, language = analysis["risk"], analysis["language"]
        if risk.level in ("high", "urgent"):
            return {"reply": crisis_reply(risk.level, language), "mind_pause": "none", "source": "crisis_protocol",
                    "model": None, "attempts": 0, "log": [], "prompt": message}
        prompt = build_prompt(history, message, language, analysis)
        out, log = None, []
        if self.backend is not None:
            ok, why = self.backend.available()
            if ok:
                out, log = generate(self.backend, SYSTEM_PROMPT, prompt, CompanionReply, reply_issues, schema=self.schema)
            else:
                log = [{"error": why}]
        if out is None:
            return {"reply": fallback_reply(language), "mind_pause": "box_breathing", "source": "fallback",
                    "model": getattr(self.backend, "name", None), "attempts": len(log), "log": log, "prompt": prompt}
        return {"reply": out.reply.strip(), "mind_pause": out.mind_pause, "source": "llm", "model": self.backend.name,
                "attempts": len(log), "log": log, "prompt": prompt}
