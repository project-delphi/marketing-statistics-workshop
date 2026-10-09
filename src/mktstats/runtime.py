"""Where a lab runs, and making the installed packages match the tested pins.

The generated install cell of every Python lab calls :func:`ensure` with that lab's pins
(from ``_variables.yml``) and the URL of ``environment/requirements.txt`` at the
workshop's ref, so pip resolves to exactly the versions CI and the Docker image test.

- On Colab and SageMaker, :func:`ensure` installs any pinned package that is missing or
  at another version.
- Anywhere else (your laptop, the workshop image, CI) it installs nothing and only warns.
- A library already imported at another version stays in memory until the runtime
  restarts. :func:`ensure` then stops the run with "Runtime > Restart session, then Run
  all" instead of letting later cells mix two versions.

Nothing here imports a heavy library; it only reads versions from package metadata and
from modules that are already loaded.
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping

COLAB, SAGEMAKER, LOCAL = "colab", "sagemaker", "local"
HOSTED = (COLAB, SAGEMAKER)
RESTART_MESSAGE = "Runtime > Restart session, then Run all"
# Libraries whose in-memory version must match the installed one: a preloaded older
# pymc (Colab ships one) would otherwise be used next to a newer pytensor or arviz.
GUARDED = ("pymc", "pytensor", "arviz", "pymc-marketing", "numba", "llvmlite")
# Distribution name -> import name, where they differ beyond "-" -> "_".
IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "google-meridian": "meridian",
    "tfp-causalimpact": "causalimpact",
}
# SageMaker Studio apps carry this metadata file (environment/aws/README.md). Not yet
# checked on a live Studio app: the AWS paths are documented, not run.
SAGEMAKER_METADATA = "/opt/ml/metadata/resource-metadata.json"


class RestartRequired(RuntimeError):
    """A library is loaded at another version than the one installed: restart the runtime."""


def detect(environ: Mapping[str, str] | None = None) -> str:
    """``"colab"``, ``"sagemaker"`` or ``"local"``. ``MKTSTATS_RUNTIME`` overrides (tests)."""
    env = os.environ if environ is None else environ
    forced = env.get("MKTSTATS_RUNTIME", "").strip().lower()
    if forced in (COLAB, SAGEMAKER, LOCAL):
        return forced
    if env.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return COLAB
    if (
        env.get("SAGEMAKER_APP_TYPE")
        or env.get("SAGEMAKER_SPACE_NAME")
        or os.path.exists(SAGEMAKER_METADATA)
    ):
        return SAGEMAKER
    return LOCAL


def normalize(name: str) -> str:
    """PEP 503 name: ``Scikit_Learn`` -> ``scikit-learn``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def import_name(dist: str) -> str:
    dist = normalize(dist)
    return IMPORT_NAMES.get(dist, dist.replace("-", "_"))


def installed_version(dist: str) -> str | None:
    try:
        return metadata.version(normalize(dist))
    except metadata.PackageNotFoundError:
        return None


def loaded_version(dist: str) -> str | None:
    """The version of an already imported module, or None if it is not imported."""
    module = sys.modules.get(import_name(dist))
    if module is None:
        return None
    return str(getattr(module, "__version__", "") or "") or None


def pins_of(requirements: Mapping[str, str] | Iterable[str]) -> dict[str, str]:
    """``{"pandas": "2.2.3"}`` from a mapping or from ``["pandas==2.2.3", ...]``."""
    if isinstance(requirements, Mapping):
        return {normalize(k): str(v) for k, v in requirements.items()}
    out = {}
    for spec in requirements:
        name, sep, version = spec.partition("==")
        if not sep or not version.strip():
            raise ValueError(f"{spec!r}: give an exact pin, name==version")
        out[normalize(name.strip())] = version.strip()
    return out


def mismatches(pins: Mapping[str, str]) -> dict[str, tuple[str | None, str]]:
    """``{dist: (installed or None, pinned)}`` for every pin not met."""
    out = {}
    for dist, want in pins.items():
        have = installed_version(dist)
        if have != want:
            out[dist] = (have, want)
    return out


def stale_modules(dists: Iterable[str] = GUARDED) -> dict[str, tuple[str, str]]:
    """``{dist: (loaded, installed)}`` for imported modules whose version on disk differs."""
    out = {}
    for dist in dists:
        loaded = loaded_version(dist)
        installed = installed_version(dist)
        if loaded and installed and loaded != installed:
            out[dist] = (loaded, installed)
    return out


def _pip(specs: list[str], constraints_url: str | None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *specs]
    if constraints_url:
        cmd += ["-c", constraints_url]
    print("Installing:", " ".join(specs))
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-15:])
        raise RuntimeError(
            f"pip could not install {' '.join(specs)}:\n{tail}\n"
            "Check the network connection and rerun this cell. If it fails again, tell the"
            " instructor and include the lines above."
        )


def ensure(
    requirements: Mapping[str, str] | Iterable[str],
    constraints_url: str | None = None,
    *,
    runtime: str | None = None,
    guarded: Iterable[str] = GUARDED,
) -> dict:
    """Make the installed versions match ``requirements`` (exact pins).

    Returns ``{"runtime", "installed", "mismatched"}``: what pip installed, and (locally)
    the pins that are not met. Raises :class:`RestartRequired` when a module that is
    already imported now differs from the version on disk.
    """
    pins = pins_of(requirements)
    where = runtime or detect()
    wrong = mismatches(pins)
    installed: list[str] = []
    loaded_before = {d for d in wrong if loaded_version(d)}
    if wrong and where in HOSTED:
        specs = [f"{d}=={want}" for d, (_, want) in sorted(wrong.items())]
        _pip(specs, constraints_url)
        installed = specs
        wrong = mismatches(pins)
        if wrong:
            raise RuntimeError(f"After installing, these pins are still not met: {wrong}")
    elif wrong:
        for dist, (have, want) in sorted(wrong.items()):
            what = f"{dist} {have} is installed" if have else f"{dist} is not installed"
            print(
                f"Warning: {what}; the labs are tested with {dist}=={want}"
                " (environment/requirements.txt). Nothing is installed outside Colab and"
                " SageMaker: use the workshop image or `uv pip install -r"
                " environment/requirements.txt`."
            )
    stale = stale_modules(set(guarded) | loaded_before)
    if stale:
        lines = [f"  {d}: {lo} in memory, {inst} installed" for d, (lo, inst) in stale.items()]
        print(
            "These libraries were imported before the install, at another version:\n"
            + "\n".join(lines)
            + f"\n{RESTART_MESSAGE}."
        )
        raise RestartRequired(RESTART_MESSAGE)
    return {"runtime": where, "installed": installed, "mismatched": sorted(wrong)}


def memory_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def info() -> dict:
    """What this runtime is: used by the environment check in Module 0 and run records."""
    return {
        "runtime": detect(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "cpus": os.cpu_count(),
        "memory_gb": memory_gb(),
        "colab_release": os.environ.get("COLAB_RELEASE_TAG"),
    }


def report() -> None:
    """One line for the install cell: where the lab runs and how long the install took."""
    i = info()
    seconds = os.environ.get("MKTSTATS_INSTALL_SECONDS")
    parts = [
        f"Runtime: {i['runtime']}",
        f"Python {i['python']}",
        f"{i['cpus']} CPUs",
    ]
    if i["memory_gb"]:
        parts.append(f"{i['memory_gb']} GB RAM")
    if i["colab_release"]:
        parts.append(f"Colab {i['colab_release']}")
    if seconds:
        parts.append(f"setup took {seconds} s")
    print(" · ".join(parts))
