"""All server variants must serve the same checkout frontend."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_onnx import serve as onnx
from test_serve import serve as pytorch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("server", [pytorch, onnx], ids=["pytorch", "onnx"])
def test_servers_serve_the_shared_frontend(server):
    args = server.parse_args(["--ui-only"])
    with TestClient(server.create_app(args)) as client:
        assert client.get("/").text == (ROOT / "web/index.html").read_text()
        for path in (ROOT / "web/static").iterdir():
            if path.is_file():
                response = client.get(f"/static/{path.name}")
                assert response.status_code == 200
                assert response.content == path.read_bytes()
