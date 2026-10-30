# LFM3 chat on Apple Silicon

This experimental backend runs LFM3-Omni on an Apple GPU using MLX. It reuses
the shared `shared/web/` UI, microphone and camera handling, conversation history, and
CPU Smart Turn detector. The model generates text and speech incrementally;
the codec decodes accumulated speech frames for playback while generation
continues. Shared Python modules live in `shared/`; this backend does not import the
ONNX model runtime.

From the repository root on an Apple Silicon Mac with Python 3.12 and `uv`:

```sh
uv venv --python 3.12 src-mlx/.venv
uv pip install --python src-mlx/.venv/bin/python -r src-mlx/requirements.txt
src-mlx/.venv/bin/hf download LiquidAI/LFM3-Omni-2B-MLX-8bit \
  --local-dir resources/models/lfm3-omni-2b-mlx-8bit
src-mlx/.venv/bin/python src-mlx/serve_mlx.py --port 8014 --download
```

Open http://localhost:8014 and allow microphone and camera access when needed.
The download is stored locally. To use another compatible local checkpoint,
pass `--model /path/to/model`. Use the 8-bit checkpoint above for the tested
configuration. This backend requires the Apple GPU; it does not silently fall
back to CPU. The first response can take longer while MLX compiles its kernels.

`--download` also fetches the Silero speech detector, a YAMNet audio-event
classifier, and the Smart Turn classifier. Silero and the community ONNX export
of Google's YAMNet are Apache-2.0 licensed. Omit `--download` on later starts.
Speech detection reduces false starts from
non-speech sounds; it cannot reliably remove music mixed with a person's voice
and may mistake singing for speech. When the speech detector is unavailable,
hands-free mode is disabled and the Speak button remains available.

For the optional Smart Turn model, start once with `--download`, or use the
existing cached model from the ONNX backend. Without it, auto mode falls back
to silence detection. Audio and image history is stored locally under
`resources/turns/` until the session is cleared or expires.

MLX-VLM's pinned revision is pre-release. Performance and output quality vary
with the recording, camera frames, response length, and retained history.
