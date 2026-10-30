"""Small local, stateful speech detector for microphone turn-taking."""

import hashlib
import logging
import urllib.request
from collections import deque

import numpy as np

from .paths import PROJECT_ROOT as ROOT

MODEL = ROOT / "resources/models/silero-vad/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
MODEL_URL = ("https://raw.githubusercontent.com/snakers4/silero-vad/"
             "066409f3d9db45a30d21dfcbf9147826ffdd7052/src/silero_vad/data/silero_vad.onnx")
EVENT_MODEL = ROOT / "resources/models/yamnet/yamnet.onnx"
EVENT_SHA256 = "d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e"
EVENT_URL = ("https://huggingface.co/audiomagic/yamnet-onnx/resolve/"
             "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd/yamnet.onnx")
LOG = logging.getLogger("lfm3")


def load_session(path, url, checksum, download=False):
    if not path.exists() and download:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(20_000_000)
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError("Speech detector download failed its checksum.")
        path.write_bytes(data)
    if not path.exists():
        LOG.warning("Speech detection is unavailable; start with --download to enable hands-free speech detection.")
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("Speech detector file failed its checksum.")
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def load_speech_session(download=False):
    return load_session(MODEL, MODEL_URL, MODEL_SHA256, download)


def load_event_session(download=False):
    return load_session(EVENT_MODEL, EVENT_URL, EVENT_SHA256, download)


class SpeechDetector:
    def __init__(self, session):
        self.session = session
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def predict(self, pcm):
        if len(pcm) != 1024:
            raise ValueError("Speech detection requires 512 PCM16 samples.")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(1, 512) / 32768
        result, self.state = self.session.run(None, {
            "input": np.concatenate((self.context, samples), axis=1),
            "state": self.state,
            "sr": np.array(16000, dtype=np.int64),
        })
        self.context = samples[:, -64:].copy()
        return float(result[0, 0])


class EventDetector:
    """Classify one recent 0.96-second window every 160 ms."""

    def __init__(self, session):
        self.session = session
        self.frames = deque(maxlen=30)
        self.count = 0
        self.score = None

    def predict(self, pcm):
        if len(pcm) != 1024:
            raise ValueError("Event detection requires 512 PCM16 samples.")
        self.frames.append(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)
        self.count += 1
        if len(self.frames) == 30 and (self.count - 30) % 5 == 0:
            scores = self.session.run(["output_0"], {"waveform": np.concatenate(self.frames)})[0]
            self.score = float(scores[0, 0])  # AudioSet class 0: speech.
        return self.score
