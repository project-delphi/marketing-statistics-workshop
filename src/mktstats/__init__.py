"""Helpers for the Statistics for Marketing workshop labs.

Importing ``mktstats`` loads nothing heavy: no pymc, arviz, pytensor or econml. A lab's
install cell must be able to import ``mktstats.runtime`` before it installs the pinned
versions of those libraries, and a library imported at the wrong version stays in memory
until the runtime restarts. Submodules load on first use: ``mktstats.checks``,
``mktstats.harness``, ``mktstats.runtime``, ``mktstats.data``, ``mktstats.synth``,
``mktstats.uplift``.
"""

from __future__ import annotations

import importlib

# Keep in step with pyproject.toml (tests/test_package.py checks).
__version__ = "0.1.0"

_SUBMODULES = ("checks", "data", "harness", "runtime", "synth", "uplift")

__all__ = ["__version__", *_SUBMODULES]


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
