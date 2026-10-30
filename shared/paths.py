"""Locate packaged assets separately from persistent model and turn storage."""

from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "pyproject.toml").is_file():
    PROJECT_ROOT = Path.cwd()
