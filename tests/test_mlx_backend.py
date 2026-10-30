"""Contract checks for the Apple Silicon backend without loading model weights."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_mlx", ROOT / "src-mlx/serve_mlx.py")
mlx_server = importlib.util.module_from_spec(spec)
original_path = sys.path[:]
with patch.dict(sys.modules):
    sys.modules.pop("serve", None)
    sys.modules.pop("turn_detection", None)
    spec.loader.exec_module(mlx_server)
sys.path[:] = original_path


def test_collect_media_preserves_turn_and_frame_order():
    conversation = [
        {"role": "system", "content": "Prompt"},
        {"role": "user", "content": [
            {"type": "text", "text": "Camera at 0.1s:"},
            {"type": "image", "image": "one.jpg"},
            {"type": "audio", "audio": "one.wav"},
        ]},
        {"role": "assistant", "content": "First reply"},
        {"role": "user", "content": [
            {"type": "image", "image": "two.jpg"},
            {"type": "audio", "audio": "two.wav"},
        ]},
    ]
    messages, images, audio = mlx_server.collect_media(conversation)
    assert images == ["one.jpg", "two.jpg"]
    assert audio == ["one.wav", "two.wav"]
    assert [part["type"] for part in messages[1]["content"]] == ["text", "image", "audio"]
    assert messages[2]["content"] == "First reply"
    assert [part["type"] for part in messages[3]["content"]] == ["image", "audio"]


def test_status_reports_actual_backend():
    engine = SimpleNamespace(ready=True, device="gpu", precision="8bit", backend="mlx",
                             turn_detector=None, preview=False, close=lambda: None)
    args = SimpleNamespace(ui_only=True, model="local", precision="auto")
    from fastapi.testclient import TestClient
    with TestClient(mlx_server.base.create_app(args, engine)) as client:
        assert client.get("/api/status").json()["backend"] == "mlx"


def test_reply_streams_audio_and_checks_cancellation():
    import mlx.core as mx

    class FakeModel:
        config = SimpleNamespace(audio_eos_code=2016, audio_out_token_id=99)

        def codec(self):
            return SimpleNamespace(config=SimpleNamespace(stft_n_fft=1280),
                                   decode=lambda codes: mx.zeros((codes.shape[-1] * 1920,)))

        def stream_slots(self, *args, **kwargs):
            yield 10, None
            for _ in range(5):
                yield 99, mx.zeros((12,), dtype=mx.int32)
            yield 99, mx.full((12,), 2016, dtype=mx.int32)

    processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda tokens, **kwargs: "hello"))
    backend = mlx_server.MLXModel(FakeModel(), processor)
    events = []
    text, pcm = backend.reply({"input_ids": [1]}, 20, events.append, lambda: None)
    assert text == "hello"
    assert len(pcm) == 5 * 1920 * 2
    assert events[0] == {"type": "text", "text": "hello"}
    assert b"".join(e["pcm"] for e in events if e["type"] == "audio") == pcm
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise mlx_server.base.GenerationCancelled

    with pytest.raises(mlx_server.base.GenerationCancelled):
        backend.reply({"input_ids": [1]}, 20, events.append, cancel)
