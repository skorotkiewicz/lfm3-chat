# LFM3 video chat

Talk to [LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) through your
microphone and camera. The model replies with text and audio, and you can
interrupt it by speaking or pressing a button.

Start the server and open http://localhost:8000 on the same computer.
Camera and microphone access on localhost works without SSL certificates.

An [ONNX version](src-onnx/README.md) is available in `src-onnx/`, with a separate
uv environment. Use `q4` for CPU or `q4f16` for NVIDIA GPU. It supports the same
camera, voice, and interruption controls without installing PyTorch.

An experimental [Apple Silicon GPU version](src-mlx/README.md) uses MLX and
the 8-bit LFM3-Omni model.

To download ONNX weights separately, use the [bundle download commands](src-onnx/README.md#download-without-starting-the-server).

## Get started

You need [uv](https://docs.astral.sh/uv/) and a microphone. The project uses Python
3.12. Run these commands from the project folder on the computer hosting the model.

Install the dependencies:

```bash
uv sync
```

Already downloaded the model? Skip the next command and use your existing folder
with `--model`. Otherwise, download it once:

```bash
uv run hf download LiquidAI/LFM3-Omni-2B --local-dir resources/models/lfm3-omni-2b-pytorch
```

Start the chat:

```bash
uv run serve
```

Open http://localhost:8000 on the same computer. HTTPS is not needed for localhost.
The server uses existing model files by default; `--download` allows it to fetch
missing files. To use a model in the Hugging Face cache, pass
`--model LiquidAI/LFM3-Omni-2B`. The default local folder is
`resources/models/lfm3-omni-2b-pytorch/`.
You can also let the server download into that folder with `uv run serve --download`.
Existing model folders remain usable with `--model`; no files need to move.

## Have a conversation

1. Click **Start conversation** and allow microphone and camera access.
2. Speak, then pause for a reply. Audio plays as the model generates it.
3. Speak over a reply to interrupt it, or click **Interrupt & speak**. Headphones
   help prevent speaker echo from causing unwanted interruptions.
4. Open **Conversation settings** to turn camera sharing or hands-free speech on
   or off.
5. Click **End** to stop the camera and microphone. **Start fresh** clears the
   conversation history.

The assistant begins answering after your turn ends. It does not generate an
answer while you are still speaking.

## Pause to think with Smart Turn

Smart Turn v3.2 helps distinguish a thinking pause from the end of a sentence.
To enable it and download any missing model files on the first run:

```bash
uv run serve \
  --model resources/models/lfm3-omni-2b-pytorch \
  --turn-detector smart --download
```

After that, omit `--download`. The default `auto` setting uses Smart Turn when it
is available locally and falls back to silence detection otherwise. You can
choose silence detection explicitly with `--turn-detector silence`.

Smart Turn runs on the CPU. It gives you time to continue an unfinished thought,
with a short silence timeout so the conversation does not stay waiting forever.
You can always press **Send** to finish your turn immediately.

Its model is kept in `resources/models/smart-turn-v3/`. Keep that folder to reuse
the download. Deleting it removes the local model copy. For an existing ONNX file,
use `--turn-model /path/to/smart-turn-v3.2-cpu.onnx`.

## Connect from another device

To connect from another device, use HTTPS for camera and microphone access.
Start the server with `--host 0.0.0.0` to accept local network connections.
This setup is optional; local use on http://localhost:8000 needs no certificate.

Use a certificate trusted by the connecting device. For development, you can
create a self-signed certificate. Replace `<server-ip>` below with your server's
LAN address before running the command:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout key.pem -out cert.pem -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:<server-ip>"
chmod 600 key.pem
```

Then run:

```bash
uv run serve \
  --model resources/models/lfm3-omni-2b-pytorch \
  --host 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem
```

Trust the certificate on the connecting device, then open
https://<server-ip>:8000/ there. The browser automatically uses a secure WebSocket
for audio and video frames. If the page cannot connect, check that the server's
firewall allows your chosen port.

Both certificate flags are required together. `cert.pem` and `key.pem` are
ignored by Git. Keep the private key private.

## Development checks and preview

All backends use the frontend in `shared/web/`. Edit it once for PyTorch, ONNX, and
MLX. The UI uses the capabilities returned by `/api/status` to select speech
handling and show supported controls. Wheel and source builds include the UI.

The backend directories are `src-pytorch/`, `src-onnx/`, and `src-mlx/`.
`shared/` contains the frontend and the common ONNX/MLX server, audio helpers,
and speech and turn detection. Model loading stays in each backend. The
PyTorch launch command remains `uv run serve`.

Run the local tests:

```bash
uv run pytest
node --test tests/test-*.cjs
```

Preview the interface without loading LFM3 weights:

```bash
uv run --python 3.12 tests/live_fixture.py --port 8001
```

Open http://localhost:8001. The preview is labelled and uses fixed text, synthetic
audio, and scripted turn decisions. It demonstrates the interface and interaction
flow, not the model's response quality or speed.

To check loading and inference of the real Smart Turn model:

```bash
uv run --python 3.12 tests/check_smart_turn.py
```

This check downloads the classifier if missing and tests it on silence. It does
not measure accuracy on real speech. On their first run, `uv` may also install the
check scripts' Python dependencies.

Both scripts are saved under `tests/`. Downloaded model assets are kept under
`resources/`, which is ignored by Git. Your own LFM3 folder can be anywhere; pass
its path with `--model`.
