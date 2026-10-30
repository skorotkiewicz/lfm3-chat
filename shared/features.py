"""NumPy mel features used by model preprocessing and CPU turn detection."""

import math
from types import SimpleNamespace

import numpy as np


def mel_filters(mel):
    """Slaney mel scale and area normalization, matching the checkpoint frontend."""
    # The Slaney scale is linear below 1 kHz and logarithmic above it.
    def hz_to_mel(hz):
        return hz / (200 / 3) if hz < 1000 else 15 + math.log(hz / 1000) / (math.log(6.4) / 27)

    mel_points = np.linspace(0, hz_to_mel(mel.sampling_rate / 2), mel.num_mel_bins + 2)
    hz_points = np.where(mel_points < 15, mel_points * (200 / 3),
                         1000 * np.exp((mel_points - 15) * (math.log(6.4) / 27)))
    frequencies = np.linspace(0, mel.sampling_rate / 2, mel.n_fft // 2 + 1)
    differences = np.diff(hz_points)
    slopes = hz_points[None, :] - frequencies[:, None]
    filters = np.maximum(0, np.minimum(-slopes[:, :-2] / differences[:-1],
                                       slopes[:, 2:] / differences[1:]))
    filters *= 2 / (hz_points[2:] - hz_points[:-2])
    return filters.astype(np.float32)


def whisper_features(audio):
    """Whisper Tiny's normalized 80-bin frontend for Smart Turn's eight-second window."""
    audio = np.asarray(audio, dtype=np.float32)
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    padded = np.pad(audio, (200, 200), mode="reflect")
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    segments = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160][:-1]
    power = np.abs(np.fft.rfft(segments * window, axis=-1)) ** 2
    filters = mel_filters(SimpleNamespace(sampling_rate=16000, n_fft=400, num_mel_bins=80))
    features = np.log10(np.maximum(filters.T @ power.T, 1e-10))
    features = np.maximum(features, features.max() - 8)
    return ((features + 4) / 4).astype(np.float32)[None]
