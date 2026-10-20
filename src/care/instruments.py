"""PHQ-9 and GAD-7 item text for the app and WhatsApp check-ins.

English wording from the public-domain instruments (Pfizer; no permission needed to reproduce).
Use the official validated Hindi and regional translations (phqscreeners.com) before the pilot;
they are not reproduced here, so non-English patients currently see English item text.
"""

STEM = "Over the last 2 weeks, how often have you been bothered by any of the following problems?"
OPTIONS = [(0, "Not at all"), (1, "Several days"), (2, "More than half the days"), (3, "Nearly every day")]

ITEMS = {
    "PHQ-9": [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself, or that you are a failure or have let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
        "Moving or speaking so slowly that other people could have noticed, or the opposite: being so fidgety "
        "or restless that you have been moving around a lot more than usual",
        "Thoughts that you would be better off dead, or of hurting yourself in some way",
    ],
    "GAD-7": [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it is hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid, as if something awful might happen",
    ],
}


def form(instrument: str) -> dict:
    return {"instrument": instrument, "stem": STEM, "items": ITEMS[instrument],
            "options": [{"value": v, "label": label} for v, label in OPTIONS]}
