"""Mind Pause: short guided exercises the companion can offer (scripted, no LLM).

Each step is (seconds, English, Hindi). The app shows/speaks one step at a time with a timer.
The Hindi lines need a native-speaker and clinician review before the pilot.
"""

EXERCISES = {
    "box_breathing": {
        "title": {"en": "Box breathing (2 min)", "hi": "बॉक्स ब्रीदिंग (2 मिनट)"},
        "for": "anxious, overwhelmed",
        "steps": [(4, "Breathe in slowly through your nose", "नाक से धीरे-धीरे साँस अंदर लें"),
                  (4, "Hold your breath gently", "साँस को आराम से रोकें"),
                  (4, "Breathe out slowly through your mouth", "मुँह से धीरे-धीरे साँस बाहर छोड़ें"),
                  (4, "Hold, and relax your shoulders", "रुकें, और कंधों को ढीला छोड़ें")] * 4,
    },
    "grounding_54321": {
        "title": {"en": "5-4-3-2-1 grounding (3 min)", "hi": "5-4-3-2-1 ग्राउंडिंग (3 मिनट)"},
        "for": "panicky, racing thoughts",
        "steps": [(30, "Name 5 things you can see around you", "अपने आसपास की 5 चीज़ें देखें और उनके नाम लें"),
                  (30, "Notice 4 things you can touch", "4 चीज़ों को छूकर महसूस करें"),
                  (30, "Listen for 3 sounds", "3 आवाज़ें सुनें"),
                  (20, "Notice 2 things you can smell", "2 गंध पहचानें"),
                  (20, "Notice 1 thing you can taste", "1 स्वाद पर ध्यान दें"),
                  (15, "Take one slow breath. You are here, right now.",
                   "एक धीमी साँस लें। आप यहीं हैं, इसी पल में।")],
    },
    "body_scan": {
        "title": {"en": "Bedtime body scan (3 min)", "hi": "सोने से पहले बॉडी स्कैन (3 मिनट)"},
        "for": "can't sleep, tense",
        "steps": [(20, "Lie down and let your eyes close", "लेट जाएँ और आँखें बंद कर लें"),
                  (30, "Notice your feet and legs, and let them get heavy", "पैरों पर ध्यान दें, उन्हें भारी होने दें"),
                  (30, "Let your stomach and chest soften with each breath", "हर साँस के साथ पेट और छाती को ढीला छोड़ें"),
                  (30, "Relax your hands, arms and shoulders", "हाथों, बाँहों और कंधों को आराम दें"),
                  (30, "Soften your jaw, face and forehead", "जबड़े, चेहरे और माथे को ढीला छोड़ें"),
                  (30, "Breathe slowly and let your whole body rest", "धीरे-धीरे साँस लें और पूरे शरीर को आराम करने दें")],
    },
}


def exercise(name: str, language: str = "en") -> dict:
    ex = EXERCISES[name]
    hi = language == "hi"
    return {"id": name, "title": ex["title"]["hi" if hi else "en"], "for": ex["for"],
            "steps": [{"seconds": s, "text": h if hi else e} for s, e, h in ex["steps"]],
            "total_seconds": sum(s for s, _, _ in ex["steps"])}


def catalogue(language: str = "en") -> list[dict]:
    return [exercise(n, language) for n in EXERCISES]
