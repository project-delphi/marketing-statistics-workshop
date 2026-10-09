"""Importing mktstats must load nothing heavy: the install cell imports mktstats.runtime
before it installs the pinned pymc, arviz and pytensor."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import mktstats

ROOT = Path(__file__).resolve().parent.parent
HEAVY = ("pymc", "arviz", "pytensor", "pymc_marketing", "econml", "numba", "pandas", "numpy")


def loaded_after(statement: str) -> list[str]:
    code = f"{statement}\nimport sys\nprint(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return [m for m in out.stdout.strip().split(",") if m]


def test_import_mktstats_loads_nothing_heavy():
    assert loaded_after("import mktstats") == []


def test_runtime_and_harness_load_nothing_heavy():
    assert loaded_after("import mktstats.runtime, mktstats.harness") == []


def test_submodules_load_on_first_use():
    assert mktstats.checks.__name__ == "mktstats.checks"
    assert "checks" in dir(mktstats)


def test_version_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert mktstats.__version__ == project["version"]


def test_readme_exists_for_the_git_install():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert (ROOT / project["readme"]).exists(), "hatchling cannot build mktstats without it"
