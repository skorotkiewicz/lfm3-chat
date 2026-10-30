# /// script
# requires-python = ">=3.12"
# dependencies = ["onnxruntime", "transformers>=5.20.2", "numpy", "huggingface-hub"]
# ///
"""Verify real Smart Turn ONNX inference; retain its model in resources/models/."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-pytorch"))

from turn_detection import MODEL_DIR, MODEL_FILE, load_turn_detector

if __name__ == "__main__":
    detector = load_turn_detector(
        SimpleNamespace(
            turn_detector="smart",
            turn_model=None,
            download=True,
        )
    )
    result = detector.predict(bytes(32000))
    assert 0 <= result <= 1
    print(f"Smart Turn v3.2 ONNX inference passed: p(complete)={result:.4f}")
    print(f"Persistent model: {MODEL_DIR / MODEL_FILE}")
