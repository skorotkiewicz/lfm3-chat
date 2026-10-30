# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi", "uvicorn[standard]", "numpy", "pillow", "httpx", "pytest"]
# ///
"""Synthetic UI preview. No LFM3 weights, speech synthesis, or Smart Turn inference.

Run: uv run --python 3.12 tests/live_fixture.py --port 8001
Normal server options, including HTTPS certificate/key flags, are supported.
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-pytorch"))
sys.path.insert(0, str(ROOT / "tests"))

from test_serve import Tensor, streaming_engine

import serve


class ScriptedTurnDetector:
    def __init__(self):
        self.count = 0

    def predict(self, pcm):
        self.count += 1
        return 0.9 if self.count % 2 == 0 else 0.1


def create_preview_engine(args):
    engine, _ = streaming_engine()
    args.ui_only = True
    args.model = "Preview fixture: synthetic responses"
    engine.args = args
    engine.context_length = 32768
    engine.preview = True
    engine.turn_detector = ScriptedTurnDetector()

    def generate(**kwargs):
        text_stream, audio_stream = kwargs["streamer"], kwargs["audio_streamer"]
        text_stream.put(kwargs["input_ids"])
        for index in range(24):
            time.sleep(0.05)
            text_stream.put(np.array([[index]]))
            audio_stream.put(np.full(12, (index % 9) + 1, dtype=int).view(Tensor))
        text_stream.end()
        audio_stream.end()
        return SimpleNamespace(
            sequence_lengths=[kwargs["input_ids"].shape[1] + 3],
            sequences=np.zeros((1, kwargs["input_ids"].shape[1] + 3), dtype=int),
            audio_lengths=[24],
            audio_codes=None,
        )

    engine.model.generate = generate
    return engine


if __name__ == "__main__":
    args = serve.parse_args()
    engine = create_preview_engine(args)
    print("PREVIEW ONLY: fixed text, synthetic audio, and scripted turn decisions.")
    uvicorn.run(
        serve.create_app(args, engine),
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
