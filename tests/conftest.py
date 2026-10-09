"""Make `mktstats` (src/) and the scripts importable without installing anything, and keep
the workshop's environment variables out of tests unless a test sets them."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SETTINGS = (
    "MKTSTATS_WORKED",
    "MKTSTATS_QUICK",
    "MKTSTATS_SABOTAGE",
    "MKTSTATS_INSTALL_SECONDS",
    "MKTSTATS_RUNTIME",
    "COLAB_RELEASE_TAG",
)


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
