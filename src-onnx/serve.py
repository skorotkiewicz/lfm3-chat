#!/usr/bin/env python3
"""Run camera and voice chat with the LFM3 Omni ONNX q4 or q4f16 bundle."""

import logging
import sys
from pathlib import Path

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if (CHECKOUT_ROOT / "shared").is_dir():
    sys.path.insert(0, str(CHECKOUT_ROOT))

from shared import server  # noqa: E402
from shared.paths import PROJECT_ROOT as ROOT  # noqa: E402
from shared.turn_detection import load_turn_detector  # noqa: E402

LOG = logging.getLogger("lfm3")
MODEL = ROOT / "resources/models/lfm3-omni-2b-onnx"


class Engine(server.Engine):
    backend = "onnx"

    def load(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Choose a runtime: uv run --project src-onnx --extra cpu serve "
                "or use --extra gpu on the GPU machine."
            ) from exc
        from lfm3_onnx.model import OmniModel, choose_device_precision, resolve_model

        self.device, self.precision = choose_device_precision(
            self.args.device, self.args.precision, ort.get_available_providers(),
        )
        folder = resolve_model(self.args.model, self.precision, self.args.download)
        LOG.info("Loading %s on %s with %s", folder, self.device, self.precision)
        self.model = OmniModel(folder, precision=self.precision, force_cpu=self.device == "cpu", tf32=False)
        self.model.load_sessions(ep=self.device)
        self.model.load_preprocessing()
        self.context_length = min(self.model.config.language.max_position_embeddings, 32768)
        self.turn_detector = load_turn_detector(self.args)
        self.ready = True
        LOG.info("ONNX model ready")


def create_app(args, engine=None):
    return server.create_app(args, engine if engine is not None else Engine(args))


def parse_args(argv=None):
    parser = server.argument_parser(__doc__)
    parser.add_argument("--model", default=str(MODEL),
                        help="Default Hugging Face ID or a local ONNX model folder")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["auto", "q4", "q4f16"], default="auto",
                        help="auto selects q4 for CPU or q4f16 for CUDA")
    args = server.validate_args(parser, parser.parse_args(argv))
    if args.device == "cuda" and args.precision == "q4":
        parser.error("Use --precision q4f16 for CUDA. Plain q4 runs attention on CPU.")
    if args.device == "cpu" and args.precision == "q4f16":
        parser.error("Use --precision q4 for CPU.")
    return args


def main(argv=None):
    args = parse_args(argv)
    server.run(args, Engine(args))


if __name__ == "__main__":
    main()
