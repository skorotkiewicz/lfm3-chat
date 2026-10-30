"""Acoustic turn completion, independent of LFM3 and its GPU lock.

Smart Turn v3.2 uses Whisper Tiny features for the last eight seconds of audio.
The integration follows Parlor's use of Pipecat's classifier, while feature
extraction uses the upstream Transformers implementation already in this app.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

LOG = logging.getLogger("lfm3.turns")
MODEL_ID = "pipecat-ai/smart-turn-v3"
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    # A wheel installation has no checkout next to its modules. In that case,
    # retain downloaded models in the directory where the command was started.
    PROJECT_ROOT = Path.cwd()
MODEL_DIR = PROJECT_ROOT / "resources" / "models" / "smart-turn-v3"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 8


class SmartTurn:
    def __init__(self, path: str):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.features = WhisperFeatureExtractor(chunk_length=8)
        self.lock = threading.Lock()

    def predict(self, pcm: bytes) -> float:
        import numpy as np

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        audio = audio[-WINDOW_SAMPLES:]
        if audio.size < WINDOW_SAMPLES:
            audio = np.pad(audio, (WINDOW_SAMPLES - audio.size, 0))
        with self.lock:
            features = self.features(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="np",
                do_normalize=True, padding="max_length", max_length=WINDOW_SAMPLES,
                truncation=True,
            ).input_features
            outputs = self.session.run(None, {"input_features": features})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Smart Turn returned an invalid probability.")
        return probability


def load_turn_detector(args):
    if args.turn_detector == "silence":
        return None
    try:
        if args.turn_model:
            path = Path(args.turn_model)
            if not path.is_file():
                raise FileNotFoundError(f"Turn model not found: {path}")
        else:
            from huggingface_hub import hf_hub_download

            path = MODEL_DIR / MODEL_FILE
            if not path.is_file():
                # Reuse an existing Hugging Face download before attempting any
                # network access. Store a persistent project copy for next time.
                try:
                    cached = hf_hub_download(MODEL_ID, MODEL_FILE, local_files_only=True)
                except OSError:
                    if not args.download:
                        raise
                    path = hf_hub_download(MODEL_ID, MODEL_FILE, local_dir=MODEL_DIR)
                else:
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cached, path)
        detector = SmartTurn(str(path))
        LOG.info("Smart Turn v3.2 ready on CPU")
        return detector
    except Exception as exc:
        if args.turn_detector == "smart" or args.turn_model:
            raise RuntimeError(
                "Cannot load Smart Turn. Install project dependencies, then use "
                "--download to cache its model, or --turn-model /path/to/model.onnx. "
                "Use --turn-detector silence to run without it."
            ) from exc
        LOG.info("Smart Turn unavailable: %s. Using silence detection; enable Smart Turn with --turn-detector smart --download.", exc)
        return None
