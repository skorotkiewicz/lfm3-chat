import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src-pytorch"))
from test_serve import streaming_engine

import serve
import turn_detection


def test_default_model_directory_stays_in_project_resources():
    project_root = Path(__file__).resolve().parents[1]
    assert (
        turn_detection.MODEL_DIR
        == project_root / "resources" / "models" / "smart-turn-v3"
    )


def begin(ws, turn_id=1):
    ws.send_json({"type": "start", "id": turn_id, "session_id": "smart-session"})
    for _ in range(16):
        ws.send_bytes(b"\0\0" * 1024)


def test_incomplete_pause_keeps_the_same_utterance_until_completion():
    engine, _ = streaming_engine()
    engine.turn_detector = SimpleNamespace(predict=Mock(side_effect=[0.2, 0.9]))
    with (
        TestClient(serve.create_app(engine.args, engine)) as client,
        client.websocket_connect("/api/live") as ws,
    ):
        assert ws.receive_json()["turn_detector"] == "smart-turn-v3"
        begin(ws)
        ws.send_json({"type": "pause", "check": 1})
        first = ws.receive_json()
        assert first["type"] == "turn_decision" and first["complete"] is False
        assert not engine.sessions
        ws.send_bytes(b"\0\0" * 1024)
        ws.send_json({"type": "pause", "check": 2})
        second = ws.receive_json()
        assert second["complete"] is True
        assert second["id"] == 1
        ws.send_json({"type": "commit"})
        assert ws.receive_json()["type"] == "generating"
        ws.send_json({"type": "cancel"})


def test_cancelled_utterance_does_not_deliver_its_late_turn_decision():
    engine, _ = streaming_engine()
    entered, release = threading.Event(), threading.Event()

    def predict(pcm):
        if not entered.is_set():
            entered.set()
            assert release.wait(3)
        return 0.95

    engine.turn_detector = SimpleNamespace(predict=predict)
    try:
        with (
            TestClient(serve.create_app(engine.args, engine)) as client,
            client.websocket_connect("/api/live") as ws,
        ):
            ws.receive_json()
            begin(ws)
            ws.send_json({"type": "pause", "check": 1})
            assert entered.wait(2)
            begin(ws, turn_id=2)
            ws.send_json({"type": "pause", "check": 1})
            release.set()
            decision = ws.receive_json()
            assert decision["type"] == "turn_decision" and decision["id"] == 2
    finally:
        release.set()


def test_classifier_failure_returns_a_hold_instead_of_losing_audio():
    engine, _ = streaming_engine()
    engine.turn_detector = SimpleNamespace(
        predict=Mock(side_effect=RuntimeError("failed"))
    )
    with (
        TestClient(serve.create_app(engine.args, engine)) as client,
        client.websocket_connect("/api/live") as ws,
    ):
        ws.receive_json()
        begin(ws)
        ws.send_json({"type": "pause", "check": 1})
        decision = ws.receive_json()
        assert decision["complete"] is False and decision["probability"] is None
        ws.send_json({"type": "commit"})
        assert ws.receive_json()["type"] == "generating"
        ws.send_json({"type": "cancel"})


def test_smart_turn_uses_normalized_whisper_features_with_left_padding():
    detector = turn_detection.SmartTurn.__new__(turn_detection.SmartTurn)
    detector.lock = threading.Lock()
    detector.features = Mock(
        return_value=SimpleNamespace(
            input_features=np.zeros((1, 80, 800), dtype=np.float32)
        )
    )
    detector.session = Mock()
    detector.session.run.return_value = [np.array([[0.8]], dtype=np.float32)]
    pcm = np.full(16000, 1000, dtype="<i2").tobytes()
    assert detector.predict(pcm) == pytest.approx(0.8)
    call = detector.features.call_args
    assert call.args[0].shape == (128000,)
    assert not call.args[0][:112000].any()
    assert call.args[0][-1] == pytest.approx(1000 / 32768)
    assert call.kwargs["do_normalize"] is True
    assert detector.session.run.call_args.args[1]["input_features"].shape == (
        1,
        80,
        800,
    )
    detector.session.run.return_value = [np.array([[float("nan")]])]
    with pytest.raises(ValueError, match="invalid probability"):
        detector.predict(pcm)


def test_auto_mode_stays_offline_and_falls_back_when_model_is_not_cached(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(turn_detection, "MODEL_DIR", tmp_path / "models")
    download = Mock(side_effect=OSError("not cached"))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    args = serve.parse_args([])
    assert turn_detection.load_turn_detector(args) is None
    assert download.call_args.kwargs["local_files_only"] is True
    args.turn_detector = "smart"
    with pytest.raises(RuntimeError, match="--download"):
        turn_detection.load_turn_detector(args)


def test_explicit_download_and_local_model_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(turn_detection, "MODEL_DIR", tmp_path / "models")
    model = tmp_path / "turn.onnx"
    model.touch()
    ready = object()
    download = Mock(side_effect=[OSError("not cached"), str(model)])
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    monkeypatch.setattr(turn_detection, "SmartTurn", lambda path: ready)
    assert turn_detection.load_turn_detector(serve.parse_args(["--download"])) is ready
    assert download.call_args.kwargs["local_dir"] == tmp_path / "models"
    download.reset_mock()
    assert (
        turn_detection.load_turn_detector(
            serve.parse_args(["--turn-model", str(model)])
        )
        is ready
    )
    download.assert_not_called()
    assert (
        turn_detection.load_turn_detector(
            serve.parse_args(["--turn-detector", "silence"])
        )
        is None
    )


def test_cached_model_is_copied_to_resources_and_reused_without_network(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "resources" / "models" / "smart-turn-v3"
    monkeypatch.setattr(turn_detection, "MODEL_DIR", model_dir)
    cached = tmp_path / "cached.onnx"
    cached.write_bytes(b"cached model")
    download = Mock(return_value=str(cached))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    monkeypatch.setattr(turn_detection, "SmartTurn", lambda path: path)
    args = serve.parse_args([])
    local = model_dir / turn_detection.MODEL_FILE
    assert turn_detection.load_turn_detector(args) == str(local)
    assert local.read_bytes() == b"cached model"
    download.assert_called_once_with(
        turn_detection.MODEL_ID, turn_detection.MODEL_FILE, local_files_only=True
    )
    download.reset_mock()
    assert turn_detection.load_turn_detector(args) == str(local)
    download.assert_not_called()
