"""Exercise the ONNX fork without installing runtimes or downloading weights."""

import base64
import fnmatch
import importlib.util
import io
import json
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src-onnx"))

from lfm3_onnx.config import MelConfig
from lfm3_onnx.generation import AudioSampling, log_mel
from lfm3_onnx.model import (
    BUNDLES, CodecStream, OmniModel, bundle_patterns, choose_device_precision, resolve_model,
)
from lfm3_onnx.ops import CachedDecoder
from lfm3_onnx.preprocessing import Tokenizer, image_patches, stft_to_wave, whisper_features


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


onnx_turns = load_module("onnx_turns", ROOT / "src-onnx/turn_detection.py")
with patch.dict(sys.modules, {"turn_detection": onnx_turns}):
    serve = load_module("onnx_serve", ROOT / "src-onnx/serve.py")


@pytest.mark.parametrize("device,precision,providers,expected", [
    ("auto", "auto", [], ("cpu", "q4")),
    ("auto", "auto", ["CUDAExecutionProvider"], ("cuda", "q4f16")),
    ("auto", "q4", ["CUDAExecutionProvider"], ("cpu", "q4")),
    ("cuda", "auto", ["CUDAExecutionProvider"], ("cuda", "q4f16")),
])
def test_device_selection(device, precision, providers, expected):
    assert choose_device_precision(device, precision, providers) == expected


def test_incompatible_devices_fail_without_fallback():
    with pytest.raises(ValueError, match="q4f16"):
        choose_device_precision("cuda", "q4", ["CUDAExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        choose_device_precision("cuda", "q4f16", [])
    with pytest.raises(SystemExit):
        serve.parse_args(["--device", "cuda", "--precision", "q4"])


def test_downloads_only_requested_bundle_and_reuses_local_path(tmp_path, monkeypatch):
    from lfm3_onnx import model

    download = Mock(return_value=str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(model, "MODEL_DIR", tmp_path / "models")
    for precision, files in BUNDLES.items():
        download.reset_mock()
        resolve_model(model.MODEL_ID, precision, download=True)
        patterns = download.call_args.kwargs["allow_patterns"]
        assert download.call_args.kwargs["local_dir"] == tmp_path / "models"
        for graph in files.values():
            assert any(fnmatch.fnmatch(f"onnx/{graph}_data", pattern) for pattern in patterns)
        for graph in ("decoder_fp16.onnx", "decoder_q8.onnx", "codec_encoder.onnx", "model.safetensors"):
            assert not any(fnmatch.fnmatch(f"onnx/{graph}", pattern) for pattern in patterns)
    assert not any(fnmatch.fnmatch("onnx/decoder_q4f16.onnx_data", p) for p in bundle_patterns("q4"))
    download.reset_mock()
    assert resolve_model(str(tmp_path), "q4", download=True) == tmp_path
    download.assert_not_called()
    with pytest.raises(FileNotFoundError):
        resolve_model(str(tmp_path / "missing"), "q4", download=True)
    download.assert_not_called()


def test_mel_silence_and_smart_turn_features():
    mel = log_mel(np.zeros(16000, np.float32), MelConfig())
    assert mel.shape == (128, 99)
    np.testing.assert_array_equal(mel, -2)
    features = whisper_features(np.zeros(128000, np.float32))
    assert features.shape == (1, 80, 800)
    np.testing.assert_allclose(features, -1.5, atol=1e-6)
    t = np.arange(128000) / 16000
    tone = whisper_features(np.sin(t * 2 * np.pi * 440).astype(np.float32))
    assert tone.dtype == np.float32 and np.isfinite(tone).all()
    assert float(tone.max() - tone.min()) == pytest.approx(2, abs=1e-4)


def test_image_patch_order_preserves_channels_temporal_duplicates_and_spatial_blocks():
    # Already on the resize grid, so no interpolation obscures patch ordering.
    pixels = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    config = dict(patch_size=16, merge_size=2, temporal_patch_size=2,
                  rescale_factor=1 / 255, image_mean=[0.5] * 3, image_std=[0.5] * 3,
                  size={"shortest_edge": 4096, "longest_edge": 4096})
    blocks, count = image_patches(Image.fromarray(pixels), config)
    assert count == 4 and blocks.shape == (2, 2, 2, 2, 1536)
    expected = (pixels.astype(np.float32) / 255 - 0.5) / 0.5
    # block row 1, col 0, patch row 0, col 1 is pixels[y=32:48, x=16:32].
    patch_pixels = blocks[1, 0, 0, 1].reshape(3, 2, 16, 16)
    np.testing.assert_allclose(patch_pixels[:, 0], expected[32:48, 16:32].transpose(2, 0, 1), atol=2e-7)
    np.testing.assert_array_equal(patch_pixels[:, 0], patch_pixels[:, 1])


def test_inverse_stft_recovers_known_signal():
    n_fft, hop, rows = 1280, 320, 24
    t = np.arange((rows - 1) * hop + n_fft)
    source = np.sin(2 * np.pi * t * 440 / 24000) * 0.2
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    spectrum = np.fft.rfft(np.lib.stride_tricks.sliding_window_view(source, n_fft)[::hop] * window)
    features = np.concatenate([np.log(np.maximum(abs(spectrum), 1e-20)), np.angle(spectrum)], axis=-1)
    recovered = stft_to_wave(features, n_fft, hop)
    assert recovered.shape == (rows * hop,)
    np.testing.assert_allclose(recovered, source[480:-480], atol=1e-6)


def test_tokenizer_template_retains_generation_body_and_multimodal_markers(tmp_path, monkeypatch):
    backend = SimpleNamespace(encode=lambda *a, **k: SimpleNamespace(ids=[3]), decode=lambda *a, **k: "hello")
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: backend)))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>"}))
    (tmp_path / "chat_template.jinja").write_text(
        '{{ bos_token }}{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}'
        '{% generation %}assistant:<think></think>{% endgeneration %}'
    )
    tokenizer = Tokenizer(tmp_path)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "<|audio|><|image|>"}])
    assert prompt == "<bos>user:<|audio|><|image|>assistant:<think></think>"
    assert tokenizer(prompt) == {"input_ids": [3]}


def generation_model():
    model = object.__new__(OmniModel)
    tokens = SimpleNamespace(audio=1, audio_out=2, audio_start=3, mixed_start=4, text_end=5, eos=6)
    model.hidden = 2
    model.config = SimpleNamespace(tokens=tokens, num_codebooks=12, codebook_size=4, audio_eos_code=3,
                                   maskgit_schedule=[4, 4, 4], interleaved_n_text=6, interleaved_n_audio=12)
    model.feature_tokens = {"audio_features": 1, "audio_out_codes": 2, "image_features": 7, "video_features": 8}
    model.placeholders = tuple(model.feature_tokens.values())
    steps, depth_calls = [], []

    def step(ids, **side):
        steps.append((ids, side))
        logits = np.zeros((1, 1, 16), np.float32)
        eos_frame = np.any(side["audio_out_codes"] == 3)
        logits[0, 0, 6 if eos_frame else 10] = 10
        return logits

    decoder = SimpleNamespace(step=step, last_outputs={"hidden_states": SimpleNamespace(numpy=lambda: np.zeros((1, 1, 2)))})
    model.cached_decoder = lambda capacity: decoder

    def depth(names, feed):
        # Three passes produce speech; the next frame ends the audio segment.
        depth_calls.append(feed)
        logits = np.zeros((1, 12, 4), np.float32)
        logits[:, :, 2 if len(depth_calls) <= 3 else 3] = 20
        return [logits]

    model.depth_decoder = SimpleNamespace(run=depth)
    return model, steps, depth_calls


def test_prepare_encodes_media_and_expands_each_placeholder(tmp_path):
    model, _, _ = generation_model()
    audio_path, image_path = tmp_path / "input.wav", tmp_path / "frame.jpg"
    audio_path.write_bytes(serve.pcm_wav(b"\0\0" * 16000, 16000))
    Image.new("RGB", (64, 64), "green").save(image_path)
    model.image_token = 7
    model.image_config = dict(patch_size=16, merge_size=2, temporal_patch_size=2,
                              rescale_factor=1 / 255, image_mean=[0.5] * 3, image_std=[0.5] * 3,
                              size={"shortest_edge": 4096, "longest_edge": 4096})
    model.audio_encoder = SimpleNamespace(run=lambda names, feed: [np.ones((13, 2), np.float32)])
    model.config.mel = MelConfig()
    from lfm3_onnx.preprocessing import mel_filters
    model.mel_filters = mel_filters(model.config.mel)
    model.vision_encoder = SimpleNamespace(run=lambda names, feed: [np.full((4, 2), 2, np.float32)])
    # The real prompt_ids expands <|audio|> according to the encoder's row count.
    class Tokens:
        apply_chat_template = staticmethod(lambda *args, **kwargs: "rendered")

        def __call__(self, *args, **kwargs):
            return {"input_ids": [9, 7, 1]}

    model.tokenizer = Tokens()
    inputs = model.prepare([{"role": "user", "content": [
        {"type": "image", "image": str(image_path)}, {"type": "audio", "audio": str(audio_path)},
    ]}])
    assert inputs["input_ids"] == [9] + [7] * 4 + [1] * 13
    np.testing.assert_array_equal(inputs["audio_features"], np.ones((13, 2)))
    np.testing.assert_array_equal(inputs["image_features"], np.full((4, 2), 2))
    model.check_inputs(inputs["input_ids"], model.side_inputs(inputs["audio_features"], image_features=inputs["image_features"]))


def test_interleaved_generation_uses_three_maskgit_passes_and_replays_eos_codes():
    model, steps, depth = generation_model()
    texts, frames = [], []
    result = model.generate_ids([9], generation_mode="interleaved", max_new_tokens=32,
                                audio_sampling=AudioSampling(temperature=0), on_token=texts.append, on_frame=frames.append)
    assert texts == [10] * 6 + [6]
    assert len(frames) == 1 and result.audio_codes.shape == (1, 12)
    assert steps[0][0] == [9, 4]
    np.testing.assert_array_equal(steps[-1][1]["audio_out_codes"], np.full((1, 12), 3))
    assert [int(x["masked"].sum()) for x in depth] == [12, 8, 4, 12, 8, 4]


def test_generation_cancellation_propagates_at_first_audio_callback():
    model, steps, _ = generation_model()

    def cancel(frame):
        raise serve.GenerationCancelled

    with pytest.raises(serve.GenerationCancelled):
        model.generate_ids([9], generation_mode="interleaved", on_frame=cancel)
    assert len(steps) == 7  # Prefill plus the six text tokens.


def test_codec_stream_emits_complete_audio_once_and_can_cancel():
    def decode(codes):
        # A future frame changes the trailing overlap of the previous decode.
        samples = np.sin(np.arange(len(codes) * 1920) / 37) * 0.2
        samples[-480:] = 0
        return samples

    model = SimpleNamespace(config=SimpleNamespace(codec_n_fft=1280), decode_audio=decode)
    events = []
    stream = CodecStream(model, events.append, lambda: None)
    for _ in range(9):
        stream.put(np.zeros(12, np.int64))
    pcm = stream.end()
    from lfm3_onnx.model import waveform_pcm
    assert pcm == waveform_pcm(decode(np.zeros((9, 12))))
    assert len(events) == 3 and pcm == b"".join(event["pcm"] for event in events)


def test_gpu_caches_remain_on_gpu_between_decoder_steps(monkeypatch):
    class Value:
        def __init__(self, data, device):
            self.data, self.device = data, device

        def numpy(self):
            return self.data

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        OrtValue=SimpleNamespace(ortvalue_from_numpy=lambda array, device, index: Value(array, device))))

    class Binding:
        def __init__(self):
            self.inputs, self.destinations = {}, {}

        def bind_cpu_input(self, name, value):
            self.inputs[name] = value

        def bind_ortvalue_input(self, name, value):
            self.inputs[name] = value

        def bind_output(self, name, device):
            self.destinations[name] = device

        def get_outputs(self):
            return self.values

    inputs = [SimpleNamespace(name="past_key_values.0.key", shape=[1, 4, "past_sequence_length", 8], type="tensor(float16)"),
              SimpleNamespace(name="past_conv.0", shape=[1, 2, 7], type="tensor(float)")]
    output_names = ["logits", "hidden_states", "present.0.key", "present_conv.0"]

    def run(binding):
        sequence = binding.inputs["attention_mask"].shape[1]
        binding.values = [Value(np.zeros((1, 1, 16)), "cpu"), Value(np.zeros((1, 1, 2)), "cpu"),
                          Value(np.zeros((1, 4, sequence, 8), np.float16), "cuda"),
                          Value(np.zeros((1, 2, 7), np.float32), "cuda")]

    session = SimpleNamespace(get_inputs=lambda: inputs, get_outputs=lambda: [SimpleNamespace(name=n) for n in output_names],
                              get_providers=lambda: ["CUDAExecutionProvider"], io_binding=Binding, run_with_iobinding=run)
    decoder = CachedDecoder(session)
    decoder.step([9, 4])
    first_cache = decoder.cache["past_key_values.0.key"][0]
    decoder.step([10])
    assert decoder.past == 3
    assert first_cache.device == decoder.cache["past_key_values.0.key"][0].device == "cuda"
    assert decoder.cache["past_key_values.0.key"][0].data.shape == (1, 4, 3, 8)


def test_onnx_http_websocket_https_and_cancelled_session(tmp_path, monkeypatch):
    from lfm3_onnx import model

    monkeypatch.setattr(model, "PROJECT_ROOT", tmp_path)
    args = serve.parse_args(["--ui-only", "--device", "cpu", "--precision", "q4", "--max-new-tokens", "16"])
    engine = serve.Engine(args)
    engine.ready, engine.device, engine.context_length = True, "cpu", 128

    def reply(inputs, count, emit, check):
        check()
        if emit:
            emit({"type": "text", "text": "Hello"})
            emit({"type": "audio", "pcm": b"\0\0" * 2400})
        return "Hello", b"\0\0" * 2400

    engine.model = SimpleNamespace(prepare=lambda messages, check: {"input_ids": [9]}, reply=reply)
    wav = serve.pcm_wav(b"\0\0" * 16000, 16000)
    payload = {"session_id": "onnx-test", "audio": base64.b64encode(wav).decode(), "frames": []}
    with TestClient(serve.create_app(args, engine)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/turn-gate.js").status_code == 200
        assert client.get("/api/status").json()["precision"] == "q4"
        response = client.post("/api/turn", json=payload)
        assert response.status_code == 200, response.text
        with wave.open(io.BytesIO(base64.b64decode(response.json()["audio"])), "rb") as output:
            assert output.getframerate() == 24000
        assert str(tmp_path / "resources/turns") in engine.sessions["onnx-test"].turns[0].directory.name
        with client.websocket_connect("/api/live") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_json({"type": "start", "id": 7, "session_id": "onnx-live"})
            for _ in range(8):
                websocket.send_bytes(b"\0\0" * 2000)
            websocket.send_json({"type": "commit", "use_camera": False})
            messages = []
            for _ in range(4):
                messages.append(websocket.receive())
            json_messages = [json.loads(m["text"]) for m in messages if "text" in m]
            assert [m["type"] for m in json_messages] == ["generating", "text", "done"]
            binary = next(m["bytes"] for m in messages if "bytes" in m)
            assert int.from_bytes(binary[:4], "little") == 7 and len(binary) == 4804

        cancelled = threading.Event()

        def cancel_reply(*unused):
            cancelled.set()
            return "Interrupted", b"\0\0"

        engine.model.reply = cancel_reply
        with pytest.raises(serve.GenerationCancelled):
            engine.respond("cancelled", wav, [], cancel=cancelled)
        assert not engine.sessions["cancelled"].turns
        run = Mock()
        monkeypatch.setattr(serve.uvicorn, "run", run)
        serve.main(["--ui-only", "--ssl-certfile", "cert.pem", "--ssl-keyfile", "key.pem"])
        assert run.call_args.kwargs["ssl_certfile"] == "cert.pem"
    assert not list((tmp_path / "resources/turns").iterdir())


def test_speech_websocket_keeps_classifier_state_per_connection():
    args = serve.parse_args(["--ui-only"])
    engine = serve.Engine(args)
    app = serve.create_app(args, engine)

    class SpeechSession:
        def run(self, _outputs, inputs):
            # Each connection starts with an independent all-zero state.
            return np.array([[inputs["state"].sum()]], np.float32), inputs["state"] + 1

    class EventSession:
        def run(self, _outputs, inputs):
            scores = np.zeros((1, 521), np.float32)
            scores[0, 0] = 0.75
            return [scores]

    with TestClient(app) as client:
        assert client.get("/api/status").json()["capabilities"] == {"speech_detection": "manual"}
        app.state.speech_session, app.state.event_session = SpeechSession(), EventSession()
        assert client.get("/api/status").json()["capabilities"] == {"speech_detection": "classifier"}
        with client.websocket_connect("/api/speech") as first:
            assert first.receive_json() == {"ready": True}
            first.send_bytes(b"\0\0" * 512)
            assert first.receive_json() == {"probability": 0.0, "speech_score": None}
            first.send_bytes(b"\0\0" * 512)
            assert first.receive_json()["probability"] == 256.0
            with client.websocket_connect("/api/speech") as second:
                assert second.receive_json() == {"ready": True}
                second.send_bytes(b"\0\0" * 512)
                assert second.receive_json()["probability"] == 0.0
