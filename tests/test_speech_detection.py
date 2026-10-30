import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("speech_detection", ROOT / "src-onnx/speech_detection.py")
speech = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speech)


def test_classifier_carries_audio_context_and_recurrent_state():
    class Session:
        def __init__(self): self.calls = []
        def run(self, _outputs, inputs):
            self.calls.append(inputs)
            return np.array([[0.8]], np.float32), inputs["state"] + 1

    session = Session()
    detector = speech.SpeechDetector(session)
    assert detector.predict((np.ones(512, dtype="<i2") * 16384).tobytes()) == pytest.approx(0.8)
    detector.predict((np.ones(512, dtype="<i2") * 8192).tobytes())
    assert session.calls[0]["input"].shape == (1, 576)
    np.testing.assert_array_equal(session.calls[0]["input"][0, :64], 0)
    np.testing.assert_array_equal(session.calls[1]["input"][0, :64], 0.5)
    np.testing.assert_array_equal(session.calls[1]["state"], 1)


def test_classifier_rejects_wrong_packet_size():
    with pytest.raises(ValueError):
        speech.SpeechDetector(None).predict(b"bad")


def test_event_detector_waits_for_full_window_and_uses_recent_audio():
    class Session:
        def __init__(self): self.inputs = []
        def run(self, _outputs, inputs):
            self.inputs.append(inputs["waveform"].copy())
            scores = np.zeros((1, 521), np.float32)
            scores[0, 0] = 0.75
            return [scores]
    session = Session()
    detector = speech.EventDetector(session)
    for _ in range(29):
        assert detector.predict(b"\0\0" * 512) is None
    assert detector.predict(b"\0\0" * 512) == pytest.approx(0.75)
    assert session.inputs[0].shape == (15360,)
