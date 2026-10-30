#!/usr/bin/env python3
"""Local camera and voice chat with LiquidAI/LFM3-Omni-2B.

Run after the model download finishes:
    uv run serve
    uv run serve --model /path/to/LFM3-Omni-2B

Then open http://localhost:8000 and allow camera/microphone access. A WebSocket
streams 16 kHz PCM microphone packets and camera frames to the server. After a
pause, the model answers with streamed text and 24 kHz PCM audio. Click Interrupt
& speak to stop a reply, or enable voice interruption when using headphones.
The model processes each completed utterance; incoming audio is buffered until
the turn ends. Smart Turn v3 can hold thinking pauses without cutting you off.
No separate ASR or TTS model.

Only cached model files are used by default. --download permits Hugging Face
downloads. Custom model code is loaded with trust_remote_code=True, as required
by this checkpoint. --ui-only serves the interface without loading weights.

To serve HTTPS, pass --ssl-certfile cert.pem --ssl-keyfile key.pem. Use a
certificate trusted by the browser when accessing the camera from another device.
The browser automatically uses a secure WebSocket when the page uses HTTPS.

Use --turn-detector smart --download once to fetch Smart Turn v3.2 into
resources/models/smart-turn-v3/. Later runs reuse it without downloading.
The default auto mode uses cached Smart Turn, or falls back to silence detection.

Model API: https://huggingface.co/LiquidAI/LFM3-Omni-2B
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import importlib
import io
import json
import logging
import struct
import sys
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if (CHECKOUT_ROOT / "shared").is_dir():
    sys.path.insert(0, str(CHECKOUT_ROOT))

from shared.paths import PROJECT_ROOT as ROOT, WEB_DIR  # noqa: E402
from turn_detection import load_turn_detector  # noqa: E402

LOG = logging.getLogger("lfm3")
MODEL_ID = "LiquidAI/LFM3-Omni-2B"
MODEL = ROOT / "resources/models/lfm3-omni-2b-pytorch"
INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
MAX_SECONDS = 30
MAX_FRAMES = 6
MAX_BODY = 5_000_000
SYSTEM_PROMPT = (
    "You are having a live video conversation. Listen to the user's speech and "
    "use the camera frames when relevant. Frames are in chronological order, "
    "with times relative to the start of the recording. If no frames are provided, "
    "you cannot see the user. Answer the user's spoken request directly. "
    "Keep replies brief and conversational. Respond with interleaved text and audio."
)


def pcm_wav(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def waveform_pcm(waveform) -> bytes:
    import numpy as np

    samples = waveform.detach().float().cpu().numpy().reshape(-1)
    if not np.isfinite(samples).all():
        raise RuntimeError("The audio decoder returned an invalid waveform.")
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


class GenerationCancelled(Exception):
    """Stop at the next model callback, including during speech generation."""


class ReplyStream:
    """Decode incremental codec frames while keeping its overlap and KV cache."""

    def __init__(self, engine, emit, cancel):
        self.engine, self.emit, self.cancel = engine, emit, cancel
        self.cache = engine.codec_cache_class(engine.codec.config)
        self.pending, self.pcm = [], []
        self.last_frame = None
        self.finished = False

    def check(self):
        if self.cancel.is_set():
            raise GenerationCancelled

    def put(self, codes):
        self.check()
        self.last_frame = codes
        self.pending.append(codes)
        if len(self.pending) >= 4:  # Four codec frames represent 320 ms of speech.
            self.flush(finalize=False)

    def flush(self, finalize):
        self.check()
        if self.last_frame is None:
            return
        codes = (self.engine.torch.stack(self.pending, dim=-1).unsqueeze(0) if self.pending
                 else self.last_frame.new_empty((1, self.last_frame.numel(), 0)))
        self.pending.clear()
        with self.engine.torch.inference_mode():
            output = self.engine.codec.decode(codes, streaming_cache=self.cache, finalize=finalize)
        pcm = waveform_pcm(output.audio_values[0])
        if pcm:
            self.pcm.append(pcm)
            self.emit({"type": "audio", "pcm": pcm})

    def end(self):
        if not self.finished:
            self.finished = True
            self.flush(finalize=True)


class TextStream:
    def __init__(self, processor, emit, cancel):
        self.processor, self.emit, self.cancel = processor, emit, cancel
        self.prompt = True
        self.tokens, self.text = [], ""

    def put(self, tokens):
        if self.cancel.is_set():
            raise GenerationCancelled
        if self.prompt:
            self.prompt = False
            return
        self.tokens.extend(tokens.reshape(-1).tolist())
        text = self.processor.decode(self.tokens, skip_special_tokens=True).replace("<think></think>", "").strip()
        if text != self.text and not text.endswith("\ufffd"):
            self.text = text
            self.emit({"type": "text", "text": text})

    def end(self):
        pass


def decode_base64(value: object, limit: int) -> bytes:
    if not isinstance(value, str) or len(value) > (limit + 2) // 3 * 4:
        raise ValueError("Missing or oversized media data.")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 media data.") from exc
    if len(data) > limit:
        raise ValueError("Media exceeds the size limit.")
    return data


def valid_session_id(value):
    return (isinstance(value, str) and 1 <= len(value) <= 64
            and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value))


def validate_turn(payload: object) -> tuple[str, bytes, list[tuple[float, bytes]]]:
    """Validate browser media before it reaches the model or session storage."""
    from PIL import Image, UnidentifiedImageError

    if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object.")
    session_id = payload.get("session_id")
    if not valid_session_id(session_id):
        raise ValueError("Invalid session ID.")
    audio = decode_base64(payload.get("audio"), INPUT_RATE * 2 * MAX_SECONDS + 4096)
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, INPUT_RATE, "NONE"):
                raise ValueError("Audio must be mono 16 kHz, 16-bit PCM WAV.")
            samples = wav.getnframes()
            if not INPUT_RATE // 10 <= samples <= INPUT_RATE * MAX_SECONDS:
                raise ValueError(f"Record between 0.1 and {MAX_SECONDS} seconds.")
            if len(wav.readframes(samples)) != samples * 2:
                raise ValueError("The audio file is truncated.")
            duration = samples / INPUT_RATE
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid WAV recording.") from exc
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or len(frames) > MAX_FRAMES:
        raise ValueError(f"Send at most {MAX_FRAMES} camera frames.")
    decoded = []
    last_time = -1.0
    for frame in frames:
        if not isinstance(frame, dict):
            raise TypeError("Invalid camera frame.")
        timestamp = frame.get("time")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not 0 <= timestamp <= duration + 0.5 or timestamp < last_time):
            raise ValueError("Camera timestamps must be chronological and inside the recording.")
        data = decode_base64(frame.get("image"), 300_000)
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG" or max(image.size) > 640 or min(image.size) < 1:
                    raise ValueError("Camera frames must be JPEG images up to 640 pixels per side.")
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid camera image.") from exc
        decoded.append((float(timestamp), data))
        last_time = timestamp
    return session_id, audio, decoded


@dataclass
class Turn:
    directory: tempfile.TemporaryDirectory
    user: dict
    assistant: dict


@dataclass
class Session:
    turns: list[Turn] = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)

    def clear(self):
        for turn in self.turns:
            turn.directory.cleanup()
        self.turns.clear()


def resolve_model(model, download=False):
    if Path(model).expanduser().resolve() == MODEL:
        if download:
            from huggingface_hub import snapshot_download

            snapshot_download(MODEL_ID, local_dir=MODEL)
        if not MODEL.is_dir():
            raise FileNotFoundError(
                f"Model folder not found: {MODEL}. Start once with --download or use --model."
            )
        return str(MODEL)
    return model


class Engine:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.ready = False
        self.turn_detector = None
        self.device = "unloaded"

    def load(self):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        device = self.args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        dtype = self.args.dtype
        if dtype == "auto":
            dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.startswith("cuda") else "float32"
        self.device = device
        model = resolve_model(self.args.model, self.args.download)
        kwargs = {"trust_remote_code": True, "local_files_only": not self.args.download}
        LOG.info("Loading %s on %s with %s", model, device, dtype)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map=device, **kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model, **kwargs)
        # The codec stays in its own float32 dtype, as in Liquid AI's example.
        self.codec = self.processor.audio_tokenizer.to(device=device).eval()
        self.codec_cache_class = importlib.import_module(type(self.codec).__module__).LiquidCodecStreamingCache
        self.context_length = getattr(self.model.config, "max_position_embeddings", 32768)
        self.turn_detector = load_turn_detector(self.args)
        self.ready = True
        LOG.info("Model ready")

    def close(self):
        for session in self.sessions.values():
            session.clear()
        self.sessions.clear()

    def get_session(self, session_id: str) -> Session:
        now = time.monotonic()
        for key in list(self.sessions):
            if now - self.sessions[key].touched > 3600:
                self.sessions.pop(key).clear()
        session = self.sessions.pop(session_id, None) or Session()
        session.touched = now
        self.sessions[session_id] = session
        while len(self.sessions) > 8:
            self.sessions.popitem(last=False)[1].clear()
        return session

    def reset(self, session_id: str):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "Wait for the current reply before clearing the conversation.")
        try:
            session = self.sessions.pop(session_id, None)
            if session:
                session.clear()
        finally:
            self.lock.release()

    def respond(self, session_id: str, audio: bytes, frames: list, emit=None, cancel=None) -> dict:
        if not self.ready:
            raise HTTPException(503, "UI-only mode. Restart without --ui-only to load the model.")
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "The model is answering another turn. Please try again shortly.")
        directory = None
        cancel = cancel or threading.Event()
        try:
            if cancel.is_set():
                raise GenerationCancelled
            started = time.monotonic()
            session = self.get_session(session_id)
            directory = tempfile.TemporaryDirectory(prefix="lfm3-turn-")
            root = Path(directory.name)
            audio_path = root / "user.wav"
            audio_path.write_bytes(audio)
            content = []
            for index, (timestamp, image) in enumerate(frames):
                image_path = root / f"frame-{index}.jpg"
                image_path.write_bytes(image)
                content.extend([
                    {"type": "text", "text": f"Camera at {timestamp:.1f}s:"},
                    {"type": "image", "image": str(image_path)},
                ])
            content.append({"type": "audio", "audio": str(audio_path)})
            user = {"role": "user", "content": content}
            # Reprocess the original recordings so follow-ups retain what was
            # actually heard and seen. Drop old turns before exceeding context.
            while True:
                conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
                for turn in session.turns:
                    conversation.extend([turn.user, turn.assistant])
                conversation.append(user)
                inputs = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                prompt_length = inputs["input_ids"].shape[1]
                if prompt_length + self.args.max_new_tokens <= self.context_length:
                    break
                del inputs
                if not session.turns:
                    raise HTTPException(413, "This turn exceeds the model context. Try a shorter recording.")
                session.turns.pop(0).directory.cleanup()
            inputs = inputs.to(self.device)
            speech = ReplyStream(self, emit, cancel) if emit else None
            streaming = ({"audio_streamer": speech, "streamer": TextStream(self.processor, emit, cancel)}
                         if emit else {})
            with self.torch.inference_mode():
                if cancel.is_set():
                    raise GenerationCancelled
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.args.max_new_tokens,
                    return_dict_in_generate=True, do_sample=False,
                    generation_mode="interleaved", audio_temperature=0.4,
                    audio_repetition_penalty=1.2,
                    **streaming,
                )
                end = int(outputs.sequence_lengths[0])
                text = self.processor.decode(
                    outputs.sequences[0, prompt_length:end], skip_special_tokens=True,
                ).replace("<think></think>", "").strip()
                audio_length = int(outputs.audio_lengths[0])
                if audio_length < 1:
                    raise RuntimeError("The model returned no audio. Try increasing --max-new-tokens.")
                if speech:
                    speech.end()
                    pcm = b"".join(speech.pcm)
                else:
                    pcm = waveform_pcm(self.codec.decode(
                        outputs.audio_codes[:1, :, :audio_length]
                    ).audio_values[0])
            if not pcm:
                raise RuntimeError("The audio decoder returned an empty waveform.")
            if cancel.is_set():
                raise GenerationCancelled
            session.turns.append(Turn(directory, user, {"role": "assistant", "content": text}))
            directory = None  # The session owns the files until this turn expires.
            while len(session.turns) > self.args.history_turns:
                session.turns.pop(0).directory.cleanup()
            return {
                "text": text, "audio": base64.b64encode(pcm_wav(pcm, OUTPUT_RATE)).decode("ascii"),
                "sample_rate": OUTPUT_RATE, "seconds": round(time.monotonic() - started, 2),
                "frames": len(frames),
            }
        except (HTTPException, GenerationCancelled):
            raise
        except Exception as exc:
            LOG.exception("Generation failed")
            if hasattr(self, "torch") and self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()
            raise HTTPException(500, f"Model inference failed: {exc}") from exc
        finally:
            if directory is not None:
                directory.cleanup()
            self.lock.release()


async def serve_live(websocket: WebSocket, engine: Engine):
    """Binary input is PCM16LE/16kHz. Binary output is uint32 turn ID + PCM16LE/24kHz.

    JSON start/frame/commit/cancel/reset messages delimit turns. Each reply JSON
    event carries the same turn ID, allowing the browser to discard stale output.
    Only the sender task writes to the socket after the initial ready message.
    """
    origin = websocket.headers.get("origin")
    if origin:
        from urllib.parse import urlsplit

        if urlsplit(origin).netloc != websocket.headers.get("host"):
            await websocket.close(code=1008)
            return
    await websocket.accept()
    await websocket.send_json({"type": "ready", "ready": engine.ready,
                               "input_rate": INPUT_RATE, "output_rate": OUTPUT_RATE,
                               "turn_detector": "smart-turn-v3" if engine.turn_detector else "silence",
                               "pause_ms": 250 if engine.turn_detector else 900,
                               "hold_ms": 2500})
    if not engine.ready:
        await websocket.close(code=1000)
        return
    queue = asyncio.Queue(maxsize=32)
    loop = asyncio.get_running_loop()
    worker, cancelled, recording, turn_id = None, threading.Event(), None, 0
    decision_task = None

    async def sender():
        while True:
            event = await queue.get()
            if event["type"] == "audio":
                await websocket.send_bytes(struct.pack("<I", event["id"]) + event["pcm"])
            else:
                await websocket.send_json(event)

    async def generate(session_id, audio, frames, reply_id, cancellation):
        async def terminal(event):
            while not cancellation.is_set():
                try:
                    await asyncio.wait_for(queue.put(event), timeout=0.1)
                    return
                except TimeoutError:
                    continue

        def emit(event):
            if cancellation.is_set():
                raise GenerationCancelled
            pending = asyncio.run_coroutine_threadsafe(queue.put({**event, "id": reply_id}), loop)
            try:
                while True:
                    try:
                        pending.result(timeout=0.1)
                        return
                    except TimeoutError:
                        if cancellation.is_set():
                            raise GenerationCancelled
            finally:
                if not pending.done():
                    pending.cancel()

        try:
            result = await run_in_threadpool(engine.respond, session_id, audio, frames, emit, cancellation)
            if not cancellation.is_set():
                result.pop("audio", None)  # PCM has already been streamed.
                await terminal({"type": "done", "id": reply_id, **result})
        except GenerationCancelled:
            pass
        except HTTPException as exc:
            if not cancellation.is_set():
                await terminal({"type": "error", "id": reply_id, "detail": exc.detail})

    async def stop_generation():
        cancelled.set()
        if worker:
            await worker

    async def check_turn(data, reply_id, check_id, pcm):
        started = time.monotonic()
        try:
            probability = await run_in_threadpool(engine.turn_detector.predict, pcm)
            complete = probability >= engine.args.turn_threshold
        except Exception:
            LOG.exception("Turn detection failed; falling back to silence timeout")
            probability, complete = None, False
        # A continuation, reset, or cancellation invalidates this decision.
        if recording is data and data["check"] == check_id:
            await queue.put({"type": "turn_decision", "id": reply_id, "check": check_id,
                             "complete": complete, "probability": probability,
                             "milliseconds": round((time.monotonic() - started) * 1000)})

    writer = asyncio.create_task(sender())
    try:
        while True:
            packet = await websocket.receive()
            if packet["type"] == "websocket.disconnect":
                break
            try:
                pcm = packet.get("bytes")
                if pcm is not None:
                    if recording is None:
                        continue
                    if len(pcm) > 16384 or len(pcm) % 2:
                        raise ValueError("Send PCM16 audio in packets of at most 16 KiB.")
                    recording["pcm"].extend(pcm)
                    if len(recording["pcm"]) > INPUT_RATE * 2 * MAX_SECONDS:
                        raise ValueError(f"Record at most {MAX_SECONDS} seconds per turn.")
                    continue
                raw = packet.get("text", "")
                if len(raw) > 410_000:
                    raise ValueError("Control message is too large.")
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise TypeError("Expected a control message.")
                kind = event.get("type")
                if kind in ("start", "reset"):
                    new_id = event.get("id")
                    if (isinstance(new_id, bool) or not isinstance(new_id, int)
                            or not turn_id < new_id <= 0xFFFFFFFF or not valid_session_id(event.get("session_id"))):
                        raise ValueError("Invalid turn or session ID.")
                    turn_id = new_id
                    cancelled.set()
                    recording = None
                    if kind == "reset":
                        await stop_generation()
                        await run_in_threadpool(engine.reset, event["session_id"])
                        await queue.put({"type": "reset", "id": turn_id})
                    else:
                        recording = {"session_id": event["session_id"], "pcm": bytearray(), "frames": [], "size": 0, "check": 0}
                elif kind == "cancel":
                    cancelled.set()
                    recording = None
                elif kind == "frame":
                    if recording is None:
                        continue
                    recording["size"] += len(raw)
                    if recording["size"] > MAX_BODY or len(recording["frames"]) >= 64:
                        raise ValueError("Too many camera frames.")
                    recording["frames"].append({"time": event.get("time"), "image": event.get("image")})
                elif kind == "pause":
                    if recording is None or engine.turn_detector is None:
                        continue
                    check_id = event.get("check")
                    if (isinstance(check_id, bool) or not isinstance(check_id, int)
                            or not recording["check"] < check_id <= 1000):
                        raise ValueError("Invalid turn check.")
                    recording["check"] = check_id
                    if len(recording["pcm"]) < INPUT_RATE // 5:
                        continue
                    if decision_task:
                        decision_task.cancel()
                    # Only the last eight seconds are needed; GPU generation can
                    # continue independently while this CPU classifier runs.
                    decision_task = asyncio.create_task(check_turn(
                        recording, turn_id, check_id, bytes(recording["pcm"][-INPUT_RATE * 16:]),
                    ))
                elif kind == "commit":
                    if recording is None:
                        raise ValueError("Start a recording before committing it.")
                    data, recording = recording, None
                    frames = data["frames"] if event.get("use_camera", True) else []
                    if len(frames) > MAX_FRAMES:
                        frames = [frames[round(i * (len(frames) - 1) / (MAX_FRAMES - 1))] for i in range(MAX_FRAMES)]
                    payload = {"session_id": data["session_id"], "audio": base64.b64encode(
                        pcm_wav(bytes(data["pcm"]), INPUT_RATE)).decode("ascii"), "frames": frames}
                    validated = await run_in_threadpool(validate_turn, payload)
                    await stop_generation()
                    cancelled = threading.Event()
                    await queue.put({"type": "generating", "id": turn_id})
                    worker = asyncio.create_task(generate(*validated, turn_id, cancelled))
                else:
                    raise ValueError("Unknown control message.")
            except (ValueError, TypeError, HTTPException) as exc:
                recording = None
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                await queue.put({"type": "error", "id": turn_id, "detail": detail})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        cancelled.set()
        if decision_task:
            decision_task.cancel()
            await asyncio.gather(decision_task, return_exceptions=True)
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
        if worker:
            await worker


def create_app(args, engine=None):
    engine = engine or Engine(args)

    @asynccontextmanager
    async def lifespan(app):
        if not args.ui_only:
            try:
                await run_in_threadpool(engine.load)
            except Exception:
                LOG.exception("Could not load model. Finish the download, check dependencies, or use --model PATH.")
                raise
        yield
        engine.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.engine = engine
    web_directory = WEB_DIR
    app.mount("/static", StaticFiles(directory=web_directory / "static"), name="static")

    @app.get("/", response_class=FileResponse)
    def index():
        return FileResponse(web_directory / "index.html", media_type="text/html",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        return {"ready": engine.ready, "model": args.model, "device": engine.device,
                "backend": "pytorch", "capabilities": {"speech_detection": "volume"},
                "max_seconds": MAX_SECONDS, "max_frames": MAX_FRAMES,
                "preview": getattr(engine, "preview", False),
                "turn_detector": "smart-turn-v3" if engine.turn_detector else "silence"}

    @app.websocket("/api/live")
    async def live(websocket: WebSocket):
        await serve_live(websocket, engine)

    @app.post("/api/turn")
    async def turn(request: Request):
        # Read with a cap even when Content-Length is absent.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY:
                raise HTTPException(413, "Recording is too large.")
        try:
            payload = json.loads(body)
            validated = await run_in_threadpool(validate_turn, payload)
        except (ValueError, TypeError, UnicodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse(await run_in_threadpool(engine.respond, *validated))

    @app.delete("/api/session/{session_id}")
    async def reset(session_id: str):
        await run_in_threadpool(engine.reset, session_id)
        return {"ok": True}

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(MODEL), help="Hugging Face ID or local model directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--history-turns", type=int, default=3, help="Past camera/audio turns retained per conversation")
    parser.add_argument("--download", action="store_true", help="Allow downloading missing model files")
    parser.add_argument("--ui-only", action="store_true", help="Serve the interface without loading the model")
    parser.add_argument("--ssl-certfile", help="PEM certificate file for HTTPS")
    parser.add_argument("--ssl-keyfile", help="PEM private key file for HTTPS")
    parser.add_argument("--turn-detector", choices=["auto", "smart", "silence"], default="auto",
                        help="Use cached Smart Turn if available, require it, or use silence only")
    parser.add_argument("--turn-model", help="Local Smart Turn v3.2 ONNX file")
    parser.add_argument("--turn-threshold", type=float, default=0.5, help="Smart Turn completion probability threshold")

    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 8192:
        parser.error("--max-new-tokens must be between 1 and 8192")
    if not 0 <= args.history_turns <= 8:
        parser.error("--history-turns must be between 0 and 8")
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("Supply both --ssl-certfile and --ssl-keyfile")
    if not 0 < args.turn_threshold < 1:
        parser.error("--turn-threshold must be between 0 and 1")
    if args.turn_model and args.turn_detector == "silence":
        parser.error("--turn-model cannot be combined with --turn-detector silence")
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    LOG.info("Open %s://localhost:%s", "https" if args.ssl_certfile else "http", args.port)
    uvicorn.run(create_app(args), host=args.host, port=args.port,
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)


if __name__ == "__main__":
    main()
