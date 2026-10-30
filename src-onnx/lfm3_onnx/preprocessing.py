"""NumPy/Pillow/tokenizers preprocessing; importing this module needs no model runtime."""

import json
import math

import numpy as np
from PIL import Image


class Tokenizer:
    def __init__(self, directory):
        from jinja2 import nodes
        from jinja2.ext import Extension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer as Backend

        class Generation(Extension):
            # The checkpoint marks assistant spans for training. Inference only
            # needs the rendered body, without recording an assistant token mask.
            tags = {"generation"}

            def parse(self, parser):
                next(parser.stream)
                body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                return nodes.Scope(body)

        def raise_exception(message):
            raise ValueError(message)

        config = json.loads((directory / "tokenizer_config.json").read_text())
        self.special = {k: v for k, v in config.items() if k.endswith("_token") and isinstance(v, str)}
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[Generation],
        )
        environment.globals["raise_exception"] = raise_exception
        # Match Transformers' JSON rendering, including non-ASCII speech text.
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
        self.template = environment.from_string((directory / "chat_template.jinja").read_text())
        self.backend = Backend.from_file(str(directory / "tokenizer.json"))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            continue_final_message=False):
        if tokenize or continue_final_message:
            raise ValueError("This server renders complete messages with a generation prompt.")
        return self.template.render(
            **self.special, messages=messages, add_generation_prompt=add_generation_prompt,
            enable_thinking=False, preserve_thinking=False,
        )

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.backend.encode(text, add_special_tokens=add_special_tokens).ids}

    def decode(self, ids, skip_special_tokens=True):
        return self.backend.decode(list(ids), skip_special_tokens=skip_special_tokens)


def stft_to_wave(features, n_fft, hop_length):
    """Periodic Hann inverse STFT with the codec's same padding."""
    bins = n_fft // 2 + 1
    spectrum = np.exp(features[:, :bins]) * np.exp(1j * features[:, bins:])
    frames = np.fft.irfft(spectrum, n=n_fft, axis=-1)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    length = (len(frames) - 1) * hop_length + n_fft
    wave, envelope = np.zeros(length), np.zeros(length)
    for index, frame in enumerate(frames):
        offset = index * hop_length
        wave[offset:offset + n_fft] += frame * window
        envelope[offset:offset + n_fft] += window ** 2
    wave /= np.maximum(envelope, 1e-8)
    trim = (n_fft - hop_length) // 2
    return wave[trim:length - trim if trim else length].astype(np.float32)


def image_patches(image, config):
    """Qwen2-VL spatial block ordering, with duplicate frames for still images."""
    patch = config["patch_size"]
    merge = config["merge_size"]
    temporal = config["temporal_patch_size"]
    factor = patch * merge
    width, height = image.size
    if max(width, height) / min(width, height) > 200:
        raise ValueError("Image aspect ratio exceeds 200.")
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    minimum, maximum = config["size"]["shortest_edge"], config["size"]["longest_edge"]
    if h * w > maximum:
        beta = math.sqrt(height * width / maximum)
        h, w = (max(factor, math.floor(side / beta / factor) * factor) for side in (height, width))
    elif h * w < minimum:
        beta = math.sqrt(minimum / (height * width))
        h, w = (math.ceil(side * beta / factor) * factor for side in (height, width))
    image = image.convert("RGB").resize((w, h), Image.Resampling(config.get("resample", 3)))
    pixels = np.asarray(image, dtype=np.float32) * np.float32(config["rescale_factor"])
    pixels = (pixels - np.array(config["image_mean"], dtype=np.float32)) / np.array(config["image_std"], dtype=np.float32)
    pixels = np.repeat(pixels.transpose(2, 0, 1)[None], temporal, axis=0)
    gh, gw = h // patch, w // patch
    pixels = pixels.reshape(1, temporal, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    pixels = pixels.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    blocks = pixels.reshape(gh // merge, gw // merge, merge, merge, 3 * temporal * patch * patch)
    return np.ascontiguousarray(blocks), gh * gw // merge ** 2
