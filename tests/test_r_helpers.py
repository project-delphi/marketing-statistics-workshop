"""R/mktstats.R, tested in R (tests/r/test_mktstats.R). Needs R >= 4.5 with jsonlite, as in
the workshop image; skipped elsewhere (the host's R may be older)."""

import shutil
import subprocess

import pytest

import common


def r_version_ok() -> bool:
    if shutil.which("Rscript") is None:
        return False
    out = subprocess.run(
        [
            "Rscript",
            "-e",
            'cat(getRversion() >= "4.5" && requireNamespace("jsonlite", quietly=TRUE))',
        ],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip().endswith("TRUE")


@pytest.mark.skipif(not r_version_ok(), reason="needs R >= 4.5 with jsonlite (the workshop image)")
def test_r_helpers():
    out = subprocess.run(
        ["Rscript", str(common.ROOT / "tests" / "r" / "test_mktstats.R"), str(common.ROOT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
