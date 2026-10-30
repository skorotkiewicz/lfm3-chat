import base64
import io
import struct
import sys
import threading
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src-pytorch"))
import serve


def test_default_model_download_stays_in_resources(tmp_path, monkeypatch):
    assert serve.parse_args([]).model == str(
        Path(__file__).resolve().parents[1] / "resources/models/lfm3-omni-2b-pytorch"
    )
    folder = tmp_path / "resources/models/lfm3-omni-2b-pytorch"
    monkeypatch.setattr(serve, "MODEL", folder)
    monkeypatch.chdir(tmp_path)
    download = Mock(side_effect=lambda *args, **kwargs: folder.mkdir(parents=True, exist_ok=True))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    with pytest.raises(FileNotFoundError, match="--download"):
        serve.resolve_model(str(folder))
    download.assert_not_called()
    # Resume into the same folder even when a partial download already exists.
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    assert serve.resolve_model("resources/models/lfm3-omni-2b-pytorch", download=True) == str(folder)
    download.assert_called_once_with(serve.MODEL_ID, local_dir=folder)
    download.reset_mock()
    assert serve.resolve_model(str(folder)) == str(folder)
    assert serve.resolve_model("existing-model", download=True) == "existing-model"
    assert serve.resolve_model(serve.MODEL_ID) == serve.MODEL_ID
    download.assert_not_called()


def test_console_entry_point_passes_model_and_https_options_to_server(monkeypatch):
    run = Mock()
    monkeypatch.setattr(serve.uvicorn, "run", run)
    serve.main(
        [
            "--ui-only",
            "--model",
            "/existing/model",
            "--host",
            "0.0.0.0",
            "--port",
            "8888",
            "--ssl-certfile",
            "cert.pem",
            "--ssl-keyfile",
            "key.pem",
        ]
    )
    assert run.call_count == 1
    assert run.call_args.kwargs == {
        "host": "0.0.0.0",
        "port": 8888,
        "ssl_certfile": "cert.pem",
        "ssl_keyfile": "key.pem",
    }
    assert run.call_args.args[0].state.engine.args.model == "/existing/model"


def recording(seconds=1, rate=16000):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\0\0" * int(seconds * rate))
    return stream.getvalue()


def payload():
    image = io.BytesIO()
    Image.new("RGB", (384, 288), "green").save(image, format="JPEG")
    return {
        "session_id": "test-session",
        "audio": base64.b64encode(recording()).decode(),
        "frames": [{"time": 0, "image": base64.b64encode(image.getvalue()).decode()}],
    }


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def __init__(self):
        self.conversations = []

    def apply_chat_template(self, conversation, **kwargs):
        self.conversations.append(conversation)
        for message in conversation:
            if message["role"] != "user":
                continue
            for item in message["content"]:
                if item["type"] in ("audio", "image"):
                    assert Path(item[item["type"]]).is_file()
        return Inputs(input_ids=np.zeros((1, len(conversation) * 10), dtype=int))

    def decode(self, *args, **kwargs):
        return "Hello, I can see you."


class Waveform:
    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros((1, 24000), dtype=np.float32)


def engine():
    instance = serve.Engine(
        serve.parse_args(
            ["--ui-only", "--history-turns", "2", "--max-new-tokens", "16"]
        )
    )
    instance.ready = True
    instance.device = "cpu"
    instance.context_length = 200
    instance.processor = Processor()
    instance.torch = SimpleNamespace(inference_mode=nullcontext)

    def generate(**kwargs):
        assert kwargs["generation_mode"] == "interleaved"
        assert kwargs["return_dict_in_generate"] is True
        return SimpleNamespace(
            sequence_lengths=[kwargs["input_ids"].shape[1] + 3],
            sequences=np.zeros((1, kwargs["input_ids"].shape[1] + 3), dtype=int),
            audio_lengths=[12],
            audio_codes=np.zeros((1, 12, 12), dtype=int),
        )

    instance.model = SimpleNamespace(generate=generate)
    instance.codec = SimpleNamespace(
        decode=lambda codes: SimpleNamespace(audio_values=[Waveform()])
    )
    return instance


def test_routes_and_valid_media():
    instance = engine()
    with TestClient(serve.create_app(instance.args, instance)) as client:
        assert "/static/app.js" in client.get("/").text
        assert "AudioWorkletProcessor" in client.get("/static/app.js").text
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/turn-gate.js").status_code == 200
        assert client.get("/api/status").json()["ready"] is True
        assert client.get("/api/status").json()["capabilities"] == {"speech_detection": "volume"}
        response = client.post("/api/turn", json=payload())
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["text"] == "Hello, I can see you."
        assert result["frames"] == 1
        with wave.open(io.BytesIO(base64.b64decode(result["audio"])), "rb") as wav:
            assert wav.getframerate() == 24000
            assert wav.getnframes() == 24000
        assert client.delete("/api/session/test-session").status_code == 200
        assert not instance.sessions


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(session_id="../bad"),
        lambda p: p.update(audio="garbage"),
        lambda p: p.update(audio=base64.b64encode(recording(rate=48000)).decode()),
        lambda p: p.update(audio=base64.b64encode(recording(seconds=31)).decode()),
        lambda p: p["frames"][0].update(time=-0.1),
        lambda p: p["frames"][0].update(time=float("nan")),
        lambda p: p["frames"][0].update(time=10),
        lambda p: p["frames"][0].update(image="eA=="),
        lambda p: p.update(frames=p["frames"] * 7),
    ],
)
def test_bad_media(change):
    data = payload()
    change(data)
    with pytest.raises(ValueError):
        serve.validate_turn(data)


def test_history_and_context_trimming():
    instance = engine()
    args = serve.validate_turn(payload())
    try:
        instance.respond(*args)
        old_root = instance.sessions["test-session"].turns[0].directory.name
        instance.respond(*args)
        instance.respond(*args)
        assert len(instance.sessions["test-session"].turns) == 2
        assert not Path(old_root).exists()
        assert len(instance.processor.conversations[-1]) == 6
        instance.context_length = 45
        instance.respond(*args)
        assert len(instance.processor.conversations[-1]) == 2
        assert len(instance.sessions["test-session"].turns) == 1
    finally:
        instance.close()


def test_error_does_not_commit_turn_or_hold_lock():
    instance = engine()
    instance.model.generate = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("test failure")
    )
    with pytest.raises(serve.HTTPException) as exc:
        instance.respond(*serve.validate_turn(payload()))
    assert exc.value.status_code == 500
    assert not instance.sessions["test-session"].turns
    assert not instance.lock.locked()
    instance.close()


def test_busy_ui_only_and_body_limits():
    instance = engine()
    with TestClient(serve.create_app(instance.args, instance)) as client:
        instance.lock.acquire()
        try:
            assert client.post("/api/turn", json=payload()).status_code == 409
            assert client.delete("/api/session/test-session").status_code == 409
        finally:
            instance.lock.release()
        instance.ready = False
        assert client.post("/api/turn", json=payload()).status_code == 503
        assert (
            client.post("/api/turn", content="x" * (serve.MAX_BODY + 1)).status_code
            == 413
        )
        assert client.post("/api/turn", content="[invalid").status_code == 400
        assert client.post("/api/turn", json=[]).status_code == 400


def test_sessions_are_isolated_and_evicted():
    instance = engine()
    _session_id, audio, frames = serve.validate_turn(payload())
    roots = []
    try:
        for index in range(10):
            key = f"chat-{index}"
            instance.respond(key, audio, frames)
            roots.append(instance.sessions[key].turns[0].directory.name)
            assert len(instance.processor.conversations[-1]) == 2
        assert len(instance.sessions) == 8
        assert not Path(roots[0]).exists()
        assert not Path(roots[1]).exists()
    finally:
        instance.close()
    assert not any(Path(root).exists() for root in roots)


class Tensor(np.ndarray):
    def unsqueeze(self, dimension):
        return np.expand_dims(self, dimension).view(Tensor)

    def numel(self):
        return self.size

    def new_empty(self, shape):
        return np.empty(shape, dtype=self.dtype).view(Tensor)


class AudioValues:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class CodecCache:
    def __init__(self, config):
        self.tail = np.array([], dtype=np.float32)
        self.finished = False


def streaming_engine():
    instance = engine()
    instance.codec_cache_class = CodecCache
    instance.torch.stack = lambda values, dim: np.stack(values, axis=dim).view(Tensor)
    calls = []

    def decode(codes, streaming_cache, finalize):
        assert not streaming_cache.finished
        calls.append((codes.shape[-1], streaming_cache, finalize))
        values = np.repeat(np.asarray(codes[0, 0], dtype=np.float32) / 20, 1920)
        values = np.concatenate((streaming_cache.tail, values))
        if finalize:
            streaming_cache.finished = True
        else:
            values, streaming_cache.tail = values[:-4], values[-4:]
        return SimpleNamespace(audio_values=[AudioValues(values)])

    instance.codec = SimpleNamespace(config=SimpleNamespace(), decode=decode)

    def generate(**kwargs):
        text_stream, audio_stream = kwargs["streamer"], kwargs["audio_streamer"]
        text_stream.put(kwargs["input_ids"])
        for index in range(9):
            text_stream.put(np.array([[index]]))
            audio_stream.put(np.full(12, index + 1, dtype=int).view(Tensor))
        text_stream.end()
        audio_stream.end()
        assert calls, "codec must run during generation"
        return SimpleNamespace(
            sequence_lengths=[kwargs["input_ids"].shape[1] + 3],
            sequences=np.zeros((1, kwargs["input_ids"].shape[1] + 3), dtype=int),
            audio_lengths=[9],
            audio_codes=None,
        )

    instance.model.generate = generate
    return instance, calls


def test_streaming_audio_keeps_codec_state_and_flushes_tail():
    instance, calls = streaming_engine()
    events = []
    try:
        result = instance.respond(*serve.validate_turn(payload()), emit=events.append)
        assert [call[0] for call in calls] == [4, 4, 1]
        assert [call[2] for call in calls] == [False, False, True]
        assert len({id(call[1]) for call in calls}) == 1
        chunks = [event["pcm"] for event in events if event["type"] == "audio"]
        assert len(chunks) == 3
        expected = (
            (np.repeat(np.arange(1, 10, dtype=np.float32) / 20, 1920) * 32767)
            .astype("<i2")
            .tobytes()
        )
        assert b"".join(chunks) == expected
        with wave.open(io.BytesIO(base64.b64decode(result["audio"])), "rb") as wav:
            assert wav.readframes(wav.getnframes()) == expected
        assert events[0]["type"] == "text"
    finally:
        instance.close()


def test_stream_cancellation_does_not_store_an_incomplete_reply():
    instance, _ = streaming_engine()
    cancel = threading.Event()

    def emit(event):
        if event["type"] == "audio":
            cancel.set()

    try:
        with pytest.raises(serve.GenerationCancelled):
            instance.respond(*serve.validate_turn(payload()), emit=emit, cancel=cancel)
        assert not instance.sessions["test-session"].turns
        assert not instance.lock.locked()
    finally:
        instance.close()


def ws_start(ws, turn_id=1, samples=16000):
    ws.send_json({"type": "start", "id": turn_id, "session_id": "live-session"})
    pcm = np.zeros(samples, dtype="<i2").tobytes()
    for offset in range(0, len(pcm), 2048):
        ws.send_bytes(pcm[offset : offset + 2048])
    ws.send_json({"type": "commit"})


def test_websocket_streams_text_and_binary_audio_before_done():
    instance, _ = streaming_engine()
    with (
        TestClient(serve.create_app(instance.args, instance)) as client,
        client.websocket_connect("/api/live") as ws,
    ):
        assert ws.receive_json()["type"] == "ready"
        ws_start(ws)
        audio, events = [], []
        while True:
            packet = ws.receive()
            if "bytes" in packet:
                turn_id = struct.unpack("<I", packet["bytes"][:4])[0]
                assert turn_id == 1
                audio.append(packet["bytes"][4:])
            else:
                import json

                event = json.loads(packet["text"])
                events.append(event["type"])
                if event["type"] == "done":
                    assert audio, "audio arrives before done"
                    assert "audio" not in event
                    break
        assert "text" in events
        assert len(audio) == 3
        ws.send_json({"type": "reset", "id": 2, "session_id": "live-session"})
        assert ws.receive_json()["type"] == "reset"
        assert not instance.sessions


def test_websocket_interrupts_a_running_generation():
    instance = engine()
    began, stopped = threading.Event(), threading.Event()
    calls = []

    def respond(session_id, audio, frames, emit, cancel):
        calls.append(session_id)
        if len(calls) == 1:
            began.set()
            assert cancel.wait(3), "interrupt reaches the inference worker"
            stopped.set()
            raise serve.GenerationCancelled
        emit({"type": "text", "text": "New answer"})
        emit({"type": "audio", "pcm": b"\0\0" * 240})
        return {"text": "New answer", "seconds": 0.1, "sample_rate": 24000, "frames": 0}

    instance.respond = respond
    with (
        TestClient(serve.create_app(instance.args, instance)) as client,
        client.websocket_connect("/api/live") as ws,
    ):
        ws.receive_json()
        ws_start(ws)
        assert ws.receive_json()["type"] == "generating"
        assert began.wait(2)
        ws.send_json({"type": "cancel"})
        ws_start(ws, turn_id=2)
        assert ws.receive_json() == {"type": "generating", "id": 2}
        assert ws.receive_json()["text"] == "New answer"
        assert struct.unpack("<I", ws.receive_bytes()[:4])[0] == 2
        assert ws.receive_json()["type"] == "done"
        assert stopped.is_set()


def test_websocket_disconnect_cancels_the_worker():
    instance = engine()
    began, stopped = threading.Event(), threading.Event()

    def respond(session_id, audio, frames, emit, cancel):
        began.set()
        assert cancel.wait(3)
        stopped.set()
        raise serve.GenerationCancelled

    instance.respond = respond
    with TestClient(serve.create_app(instance.args, instance)) as client:
        with client.websocket_connect("/api/live") as ws:
            ws.receive_json()
            ws_start(ws)
            assert began.wait(2)
        assert stopped.wait(2)


def test_websocket_rejects_bad_packets_and_recovers():
    instance, _ = streaming_engine()
    with (
        TestClient(serve.create_app(instance.args, instance)) as client,
        client.websocket_connect("/api/live") as ws,
    ):
        ws.receive_json()
        ws.send_json({"type": "start", "id": 1, "session_id": "live-session"})
        ws.send_bytes(b"\0")
        assert ws.receive_json()["type"] == "error"
        ws_start(ws, turn_id=2)
        assert ws.receive_json()["type"] == "generating"
        ws.send_json({"type": "cancel"})


def test_websocket_rejects_cross_origin_connection():
    instance = engine()
    with (
        TestClient(serve.create_app(instance.args, instance)) as client,
        pytest.raises(serve.WebSocketDisconnect),
        client.websocket_connect(
            "/api/live", headers={"origin": "http://unrelated.example"}
        ),
    ):
        pass
