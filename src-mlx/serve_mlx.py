#!/usr/bin/env python3
"""Apple Silicon MLX backend for the existing local camera and voice chat."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared import server as base  # noqa: E402
from shared.audio import CodecStream, waveform_pcm  # noqa: E402
from shared.turn_detection import load_turn_detector  # noqa: E402

MODEL = ROOT / "resources/models/lfm3-omni-2b-mlx-8bit"


def collect_media(conversation):
    """Retain markers in their originating turns and pass paths separately."""
    messages, images, audio = [], [], []
    for message in conversation:
        content = message["content"]
        if isinstance(content, list):
            parts = []
            for item in content:
                kind = item["type"]
                if kind == "image":
                    images.append(item["image"])
                    parts.append({"type": "image"})
                elif kind == "audio":
                    audio.append(item["audio"])
                    parts.append({"type": "audio"})
                else:
                    parts.append(dict(item))
            content = parts
        messages.append({"role": message["role"], "content": content})
    return messages, images, audio


class MLXModel:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.codec = model.codec()
        self.config = type("CodecConfig", (), {"codec_n_fft": self.codec.config.stft_n_fft})()

    def prepare(self, conversation, check=lambda: None):
        from mlx_vlm.prompt_utils import get_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        check()
        messages, images, audio = collect_media(conversation)
        prompt = get_chat_template(self.processor, messages, add_generation_prompt=True)
        inputs = prepare_inputs(
            self.processor, images=images or None, audio=audio or None, prompts=prompt,
            add_special_tokens=should_add_special_tokens(self.model.config.model_type, self.processor),
        )
        inputs.pop("attention_mask", None)
        inputs["input_ids"] = inputs["input_ids"].reshape(-1).tolist()
        check()
        return inputs

    def decode_audio(self, codes):
        import mlx.core as mx
        import numpy as np

        audio = self.codec.decode(mx.array(np.asarray(codes).T))
        mx.eval(audio)
        return np.asarray(audio)

    def reply(self, inputs, max_new_tokens, emit, check):
        import mlx.core as mx
        import numpy as np
        from mlx_vlm.generate.common import generation_stream, wired_limit

        tokenizer = self.processor.tokenizer
        config = self.model.config
        text_tokens, frames = [], []
        last_text = ""
        stream = CodecStream(self, emit, check) if emit else None
        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")
        with wired_limit(self.model, [generation_stream]), mx.stream(generation_stream):
            for token, codes in self.model.stream_slots(
                input_ids, max_tokens=max_new_tokens, generation_mode="interleaved",
                audio_temperature=0.4, audio_repetition_penalty=1.2, **inputs,
            ):
                check()
                if codes is not None:
                    if int(codes[0].item()) != config.audio_eos_code:
                        frame = np.asarray(codes)
                        frames.append(frame)
                        if stream:
                            stream.put(frame)
                elif token != config.audio_out_token_id:
                    text_tokens.append(token)
                    text = tokenizer.decode(text_tokens, skip_special_tokens=True).strip()
                    if emit and text != last_text and not text.endswith("\ufffd"):
                        last_text = text
                        emit({"type": "text", "text": text})
            if not frames:
                raise RuntimeError("The model returned no speech. Try increasing --max-new-tokens.")
            pcm = stream.end() if stream else waveform_pcm(self.decode_audio(np.stack(frames)))
        return tokenizer.decode(text_tokens, skip_special_tokens=True).strip(), pcm


class MLXEngine(base.Engine):
    backend = "mlx"

    def load(self):
        import mlx.core as mx
        from mlx_vlm import load

        if not mx.metal.is_available():
            raise RuntimeError("The MLX backend requires an available Apple GPU.")
        mx.set_default_device(mx.gpu)
        model, processor = load(self.args.model)
        self.model = MLXModel(model, processor)
        self.device = "gpu"
        self.precision = "8bit"
        self.context_length = min(model.config.text_config.max_position_embeddings, 32768)
        self.turn_detector = load_turn_detector(self.args)
        self.ready = True
        base.LOG.info("MLX model ready on Apple GPU")


def parse_args(argv=None):
    parser = base.argument_parser(__doc__)
    parser.add_argument("--model", default=str(MODEL), help="Local MLX model folder or Hugging Face ID")
    parser.add_argument("--device", default="auto", help="Leave as auto to use the Apple GPU")
    parser.add_argument("--precision", default="auto", help="Leave as auto for the 8-bit checkpoint")
    args = base.validate_args(parser, parser.parse_args(argv))
    if args.device != "auto" or args.precision != "auto":
        parser.error("The MLX backend uses the Apple GPU and the 8-bit checkpoint; omit --device and --precision.")
    args.precision = "8bit"
    return args


def main(argv=None):
    args = parse_args(argv)
    base.run(args, MLXEngine(args))


if __name__ == "__main__":
    main()
