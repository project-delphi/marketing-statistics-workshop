"""The exercise harness of every Python lab (CONTRIBUTING.md, "Harness API").

The generated harness cell calls :func:`start` and binds the result to ``workshop``:

- ``@workshop.solution(n)`` stores a reference solution. It replaces your own definition
  only in worked mode (``WORKED_EXAMPLE = True`` or ``MKTSTATS_WORKED=1``) or after
  ``workshop.use_reference(n)``.
- ``with workshop.checkpoint(n):`` runs the checks, says whose code they checked (yours or
  the reference), prints how to recover when they fail, and re-raises so Run all stops.
- ``workshop.run_record()`` prints the run record between markers and writes
  ``run_record.json`` (fields: CONTRIBUTING.md and runs/README.md).

IPython hooks are used only to time cells. Rerunning the harness cell (as Run all does)
removes only the hooks an earlier harness registered; Colab's own hooks stay.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _metadata
import json
import os
import platform
import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path

WORKED_ENV = "MKTSTATS_WORKED"
QUICK_ENV = "MKTSTATS_QUICK"
SABOTAGE_ENV = "MKTSTATS_SABOTAGE"
INSTALL_ENV = "MKTSTATS_INSTALL_SECONDS"
RECORDED_SETTINGS = (QUICK_ENV, SABOTAGE_ENV)
RECORD_BEGIN = "----- run record (paste into scripts/add_run_record.py) -----"
RECORD_END = "----- end of run record -----"
RECORD_FILE = "run_record.json"
RECORD_PACKAGES = (
    "mktstats",
    "pymc-marketing",
    "pymc",
    "pytensor",
    "arviz",
    "nutpie",
    "numba",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "econml",
    "lightgbm",
    "duckdb",
    "matplotlib",
)
# Printed by a checkpoint that stops a learner who has not written an exercise yet;
# scripts/test_notebooks.py looks for it in learner mode.
NOT_WRITTEN = "is not written yet"
_MISSING = object()


def _flag(name: str) -> bool | None:
    """True for "1"/"true", False for "0"/"false", None when unset or empty."""
    value = os.environ.get(name, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return None


def _ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _numbers(text: str) -> set[int]:
    out = set()
    for part in text.replace(" ", "").split(","):
        if part.isdigit():
            out.add(int(part))
    return out


def mktstats_commit() -> str | None:
    """The commit mktstats was installed from (pip's direct_url.json), when it was
    installed from git and it is the copy this process imported."""
    try:
        dist = _metadata.distribution("mktstats")
        text = dist.read_text("direct_url.json")
    except _metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        here = Path(__file__).resolve()
        root = Path(str(dist.locate_file(""))).resolve()
        if root not in here.parents:
            return None  # imported from a checkout, not from the installed copy
    except (OSError, TypeError):
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def package_versions(names=RECORD_PACKAGES) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            pass
    if "mktstats" not in out:
        from mktstats import __version__

        out["mktstats"] = __version__
    return out


class _Checkpoint(AbstractContextManager):
    def __init__(self, ws: Workshop, n, label):
        self.ws, self.n, self.label = ws, n, str(label if label is not None else n)
        self.whose = None

    def __enter__(self):
        self.whose = self.ws._whose(self.n)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ws._finish(self.n, self.label, self.whose, exc)
        return False  # re-raise: Run all stops at a failed checkpoint


class Workshop:
    """Keeps the folded solutions from replacing your code, and says whose code each
    checkpoint checked: yours, the reference solution, or code the lab provides."""

    def __init__(self, notebook, content_sha, deps_sha, ref, exercises, ns, old=None):
        self.notebook = notebook
        self.content_sha = content_sha
        self.deps_sha = deps_sha
        self.ref_name = ref
        self.exercises = {k: tuple(v) for k, v in (exercises or {}).items()}
        self.ns = ns
        # A rerun of the harness keeps the stored solutions but starts a new run.
        self.ref = getattr(old, "ref", {})
        self.stub = getattr(old, "stub", {})
        self.chosen = getattr(old, "chosen", set())
        self.results: dict[str, tuple[bool, str | None]] = {}
        self.verified: dict = {}
        self._verifying = None
        self._saved: dict = {}
        self.cell_seconds: list[float] = []
        self.cell_errors = 0
        self.started = time.time()
        self._t0 = None

    # ---- settings -------------------------------------------------------------------
    @property
    def worked(self) -> bool:
        env = _flag(WORKED_ENV)
        if env is not None:
            return env
        return bool(self.ns.get("WORKED_EXAMPLE"))

    @property
    def quick(self) -> bool:
        env = _flag(QUICK_ENV)
        if env is not None:
            return env
        return bool(self.ns.get("QUICK"))

    @property
    def sabotaged(self) -> set[int]:
        """Exercises that keep the stub even in worked mode (MKTSTATS_SABOTAGE="2,3"), so
        a whole run must stop at their checkpoints: a check that the checks can fail."""
        return _numbers(os.environ.get(SABOTAGE_ENV, ""))

    def settings(self) -> dict:
        out = {}
        if self.quick:
            out[QUICK_ENV] = "1"
        if os.environ.get(SABOTAGE_ENV):
            out[SABOTAGE_ENV] = os.environ[SABOTAGE_ENV]
        return out

    # ---- solutions ------------------------------------------------------------------
    def _is_reference(self, n, name, obj) -> bool:
        return any(obj is r for r in self.ref.get(n, {}).get(name, ()))

    def solution(self, n):
        """Decorator on a reference solution: stores it and, unless WORKED_EXAMPLE is on,
        leaves your own definition of the same name in place."""

        def bind(obj, name=None):
            name = name or obj.__name__
            self.ref.setdefault(n, {}).setdefault(name, []).append(obj)
            current = self.ns.get(name, _MISSING)
            yours_now = current is not _MISSING and not self._is_reference(n, name, current)
            if yours_now:
                self.stub.setdefault(n, {})[name] = current
            if self.worked and n not in self.sabotaged:
                return obj
            if yours_now:
                self.chosen.discard(n)  # your TODO cell ran again: back to your code
                return current
            if n in self.chosen and n not in self.sabotaged:
                return obj  # you chose the reference with use_reference(n)
            yours = self.stub.get(n, {}).get(name, _MISSING)
            if yours is not _MISSING:
                return yours
            if callable(obj):
                return self._not_run(n, name)
            print(f"[workshop] Run your TODO {n} cell: {name} is the reference until you do.")
            return obj

        return bind

    def _not_run(self, n, name):
        def missing(*args, **kwargs):
            raise NotImplementedError(
                f"{name}: run your TODO {n} cell first, or go on with workshop.use_reference({n!r})"
            )

        missing.__name__ = name
        return missing

    def solution_value(self, n, name, value):
        """The same as solution(n), for a value rather than a function or class."""
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            kind = type(value).__name__
            raise TypeError(f"solution_value({n!r}, {name!r}): use an object, not a plain {kind}")
        return self.solution(n)(value, name)

    def use_reference(self, n):
        """Go on with the reference solution of exercise n in place of yours."""
        if n not in self.ref:
            raise KeyError(f"Exercise {n} has no reference solution yet: run its Solution cell")
        for name, objs in self.ref[n].items():
            self.ns[name] = objs[-1]
        self.chosen.add(n)
        print(
            f"Exercise {n}: now using the REFERENCE solution. Rerun the cells below it."
            f" To switch back to your code, rerun your TODO {n} cell."
        )

    # ---- checkpoints ----------------------------------------------------------------
    def checkpoint(self, n=None, label=None) -> _Checkpoint:
        """``with workshop.checkpoint(n):`` around the checks of exercise n."""
        return _Checkpoint(self, n, label)

    def _whose(self, n) -> str | None:
        names = self.exercises.get(n, ())
        if not names:
            return None  # checks of provided code
        k = sum(self._is_reference(n, x, self.ns.get(x, _MISSING)) for x in names)
        if k == len(names):
            return "the REFERENCE solution"
        if k == 0:
            return "your code"
        return "partly the REFERENCE solution"

    def _finish(self, n, label, whose, exc) -> None:
        ok = exc is None
        if self._verifying is not None:
            self.verified[self._verifying] = ok
            if ok:
                print(f"[verify] Checkpoint {label} PASSED on the stub of exercise {n}.")
            else:
                print(f"[verify] Checkpoint {label} failed on the stub, as it should.")
            return
        self.results[label] = (ok, whose)
        if ok:
            on = f" on {whose}" if whose else ""
            print(f"[workshop] Checkpoint {label} passed{on}.")
            return
        if isinstance(exc, KeyboardInterrupt):
            print(f"[workshop] Checkpoint {label} was interrupted.")
        elif isinstance(exc, NotImplementedError):
            if whose is None:
                print(
                    f"[workshop] Checkpoint {label} uses an exercise you have not written yet."
                    " Finish it, or go on with workshop.use_reference(N) for that exercise."
                )
            else:
                print(
                    f"[workshop] Checkpoint {label}: TODO {n} {NOT_WRITTEN}. Write it, run"
                    " its cell, then rerun this one. To go on without it, run"
                    f" workshop.use_reference({n!r})"
                )
        elif whose == "the REFERENCE solution":
            print(
                f"[workshop] Checkpoint {label} failed on the reference solution: a problem"
                " with the lab or the runtime, not with your code. Tell the instructor."
            )
        elif whose is None:
            print(f"[workshop] Checkpoint {label} failed. Read the message above.")
        else:
            print(
                f"[workshop] Checkpoint {label} failed on {whose}. Read the message above,"
                f" fix TODO {n} and rerun its cell, or go on with"
                f" workshop.use_reference({n!r})."
            )

    # ---- verify mode (scripts/test_notebooks.py) -----------------------------------
    def _verify_begin(self, n):
        """Test-only: bind exercise n's stubs, so a copy of its checkpoint runs on them."""
        self._saved = {name: self.ns.get(name, _MISSING) for name in self.exercises.get(n, ())}
        for name, value in self.stub.get(n, {}).items():
            self.ns[name] = value
        self.verified.pop(n, None)
        self._verifying = n

    def _verify_end(self, n):
        """Test-only: restore the reference and require that the stub copy failed."""
        for name, value in self._saved.items():
            if value is _MISSING:
                self.ns.pop(name, None)
            else:
                self.ns[name] = value
        self._saved = {}
        self._verifying = None
        if self.verified.get(n) is not False:
            raise AssertionError(
                f"The checkpoint of exercise {n} passed on the unfinished stub (or did not"
                " run): it cannot tell a finished exercise from an unfinished one"
            )

    # ---- timing hooks -----------------------------------------------------------------
    def _pre(self, *args):
        self._t0 = time.time()

    def _post(self, result=None):
        if self._t0 is None:  # the harness cell itself: hooks were registered mid-cell
            return
        self.cell_seconds.append(round(time.time() - self._t0, 2))
        self._t0 = None
        if result is not None and not result.success and self._verifying is None:
            self.cell_errors += 1  # a verification copy fails on purpose

    # ---- summary and record -----------------------------------------------------------
    def summary(self):
        """Which checkpoints passed, and on whose code."""
        groups = {"your code": [], "the reference": [], "other checks": [], "failed": []}
        for label, (ok, whose) in self.results.items():
            if not ok:
                groups["failed"].append(label)
            elif whose is None:
                groups["other checks"].append(label)
            elif whose == "your code":
                groups["your code"].append(label)
            else:
                groups["the reference"].append(label)
        mode = "WORKED EXAMPLE: reference solutions" if self.worked else "your code"
        quick = ", QUICK settings" if self.quick else ""
        print(f"{self.notebook}, run with {mode}{quick}.")
        lines = {
            "your code": "Checkpoints passed on your code",
            "the reference": "Checkpoints passed on a reference solution",
            "other checks": "Other checks passed",
            "failed": "Checkpoints failed",
        }
        for group, labels in groups.items():
            print(f"  {lines[group]}: {', '.join(labels) or 'none'}")
        if self.worked:
            print("  A worked-example run shows how the lab goes, not that you did it.")

    def status(self) -> str:
        """pass: at least one checkpoint ran, all passed, and no cell failed."""
        ok = self.results and self.cell_errors == 0 and all(r[0] for r in self.results.values())
        return "pass" if ok else "fail"

    def record(self) -> dict:
        from mktstats import runtime

        install = os.environ.get(INSTALL_ENV)
        try:
            install_seconds = float(install) if install else None
        except ValueError:
            install_seconds = None
        rec = {
            "date": _dt.datetime.now(_dt.UTC).date().isoformat(),
            "notebook": self.notebook,
            "kernel": "python3",
            "content_sha": self.content_sha,
            "deps_sha": self.deps_sha,
            "ref": self.ref_name,
            "mktstats_commit": mktstats_commit(),
            "mode": "worked" if self.worked else "learner",
            "settings": self.settings(),
            "status": self.status(),
            "cell_errors": self.cell_errors,
            "seconds": round(time.time() - self.started, 1),
            "install_seconds": install_seconds,
            "cell_seconds": self.cell_seconds,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "runtime": runtime.detect(),
            "colab_release": os.environ.get("COLAB_RELEASE_TAG") or None,
            "cpus": os.cpu_count(),
            "memory_gb": runtime.memory_gb(),
            "packages": package_versions(),
            "checkpoints": {
                label: {"passed": ok, "whose": whose} for label, (ok, whose) in self.results.items()
            },
        }
        if self.verified:
            rec["verified"] = {str(k): v for k, v in self.verified.items()}
        return rec

    def run_record(self, path: str | os.PathLike = RECORD_FILE) -> None:
        """Print the run record between markers and write it to run_record.json. It holds
        no code, outputs, keys or names (runs/README.md)."""
        rec = self.record()
        text = json.dumps(rec)
        try:
            Path(path).write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"(Could not write {path}: {exc})")
        print(RECORD_BEGIN)
        print(text)
        print(RECORD_END)


def _is_harness_hook(callback) -> bool:
    # IPython 7 may store a backcall wrapper around the bound method (`__wrapped__`).
    owner = getattr(getattr(callback, "__wrapped__", callback), "__self__", None)
    cls = type(owner)
    return cls.__name__ == "Workshop" and cls.__module__ == __name__


def start(notebook, content_sha, deps_sha, ref, exercises, *, ns=None) -> Workshop:
    """Start (or restart) the harness for one notebook and return the ``workshop`` object.

    ``ns`` is the namespace the lab's cells run in: IPython's user namespace by default,
    else the caller's globals.
    """
    ip = _ipython()
    if ns is None:
        ns = ip.user_ns if ip is not None else sys._getframe(1).f_globals
    old = ns.get("workshop")
    if not (hasattr(old, "ref") and hasattr(old, "stub")):
        old = None
    ws = Workshop(notebook, content_sha, deps_sha, ref, exercises, ns, old)
    # The switches the lab's own cells read: the variables, overridden by the environment.
    ns["WORKED_EXAMPLE"] = ws.worked
    ns["QUICK"] = ws.quick
    if ip is not None:
        for event in ("pre_run_cell", "post_run_cell"):
            for callback in list(ip.events.callbacks[event]):
                if _is_harness_hook(callback):
                    ip.events.unregister(event, callback)
        ip.events.register("pre_run_cell", ws._pre)
        ip.events.register("post_run_cell", ws._post)
    mode = "WORKED EXAMPLE (reference solutions)" if ws.worked else "checking your code"
    print(f"Workshop harness ready: {mode}" + (", QUICK settings" if ws.quick else "") + ".")
    return ws
