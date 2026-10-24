"""Audio clean-up before speech-to-text (care/voice.py). No Whisper needed.

Run: python -m pytest tests
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from care.voice import SR, enhance, normalise  # noqa: E402


def tone(seconds=1.0, freq=440.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)))


def test_normalise_lifts_a_quiet_recording_to_speech_level():
    quiet = tone(amp=0.003)  # about -50 dB
    assert abs(rms_db(normalise(quiet)) - (-20)) < 0.5
    assert np.max(np.abs(normalise(tone(amp=0.99) * 3))) <= 0.99  # never clips


def test_enhance_keeps_length_and_can_be_switched_off():
    rng = np.random.default_rng(0)
    noisy = tone() + 0.05 * rng.standard_normal(SR).astype("float32")
    for steps in ("highpass,denoise,normalise", "normalise"):
        assert len(enhance(noisy, steps)) == len(noisy)
    assert np.array_equal(enhance(noisy, "none"), noisy)


def test_highpass_removes_hum_but_keeps_speech_band():
    hum, voice = tone(freq=40, amp=0.5), tone(freq=500, amp=0.5)
    assert rms_db(enhance(hum, "highpass")) < rms_db(hum) - 15
    assert abs(rms_db(enhance(voice, "highpass")) - rms_db(voice)) < 1


def test_silence_is_left_alone():
    silent = np.zeros(SR, dtype="float32")
    assert np.array_equal(enhance(silent, "highpass,denoise,normalise"), silent)
