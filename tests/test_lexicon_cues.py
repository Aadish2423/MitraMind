"""Behavioural tests for the matching rules. Run: python -m pytest tests  (or python tests/test_extractor.py)"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lexicon_cues.extractor import CueExtractor  # noqa: E402
from lexicon_cues.lexicon import expand_braces  # noqa: E402

X = CueExtractor(use_empath=False)


def cue(text):
    return X.extract(text)


def test_brace_expansion():
    assert set(expand_braces("{can|could} not {sleep|rest}")) == {
        "can not sleep", "can not rest", "could not sleep", "could not rest"}


def test_contraction_variants_match_same_term():
    for t in ("I can't sleep", "I cant sleep", "I cannot sleep", "I can not sleep"):
        assert cue(t)["cue_sleep_insomnia"] == 1, t


def test_whole_word_matching_fixes_low_substring_bug():
    r = cue("My parent is visiting.")
    assert r["cue_str_economic"] == 0  # "rent" inside "parent" must not match
    assert r["cue_str_family"] == 1


def test_negation_window():
    r = cue("I don't want to die, I just want the pain to stop.")
    assert r["cue_risk_tier"] == 0 and r["cue_risk_negated"] >= 1
    assert cue("I'm not hopeless anymore")["cue_risk_indicator"] == 0


def test_negation_inside_phrase_is_not_negation():
    assert cue("I can't take it anymore")["cue_risk_indicator"] == 1


def test_first_person_scoping():
    assert cue("My friend can't sleep and he feels worthless.")["cue_phq_total"] == 0
    assert cue("Can't sleep again. Exhausted.")["cue_phq_total"] == 2  # subject-dropped self talk


def test_risk_tiers():
    assert cue("I want to die")["cue_risk_tier_name"] == "ideation"
    assert cue("I tried to kill myself last year")["cue_risk_tier_name"] == "attempt"
    assert cue("No point in getting rid of it")["cue_risk_tier"] == 0
    assert cue("This team is hopeless")["cue_risk_tier"] == 0


def test_small_hours_and_exams():
    r = cue("It's 3am and my board exams are tomorrow")
    assert r["cue_sleep_small_hours"] == 1 and r["cue_str_school_exam"] >= 1


def test_empty_text():
    r = cue("")
    assert r["n_words"] == 0 and r["cue_risk_tier"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
