"""PCM encoding and incremental codec playback shared by model adapters."""

import io
import wave

import numpy as np


def pcm_wav(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def waveform_pcm(samples):
    samples = np.asarray(samples).reshape(-1)
    if not np.isfinite(samples).all():
        raise RuntimeError("The audio decoder returned an invalid waveform.")
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


class CodecStream:
    """Re-decode accumulated codes, retaining the inverse-STFT overlap until stable.

    The published codec graph has no cache inputs. Recomputing the prefix costs
    more than the PyTorch streaming codec, but allows playback before turn end.
    """

    def __init__(self, model, emit, check):
        self.model, self.emit, self.check = model, emit, check
        self.codes, self.parts = [], []
        self.offset = 0
        self.wave = None

    def put(self, codes):
        self.check()
        self.codes.append(np.array(codes, copy=True))
        if len(self.codes) % 4 == 0:
            self.flush(final=False)

    def flush(self, final):
        self.check()
        if self.wave is None or len(self.codes) % 4 or not final:
            self.wave = self.model.decode_audio(np.stack(self.codes))
        self.check()
        # Causal codec features are stable; only the inverse STFT right edge
        # changes when additional frames arrive. Retain a full FFT of samples.
        limit = len(self.wave) if final else max(0, len(self.wave) - self.model.config.codec_n_fft)
        pcm = waveform_pcm(self.wave[self.offset:limit])
        self.offset = limit
        if pcm:
            self.parts.append(pcm)
            self.emit({"type": "audio", "pcm": pcm})

    def end(self):
        self.flush(final=True)
        return b"".join(self.parts)
