"""Load and validate run records: the evidence that a notebook ran (runs/README.md).

A file holds one batch: top-level fields apply to every entry of `runs`, and an entry may
override them. `load_all()` returns one flat dict per entry, with `file` and `index` set.
Records come from `runs/*.json` (committed: local, Docker and Colab runs) and from
`runs/ci/**/*.json` (CI runs, checked out from the `evidence` branch by CI; not committed
on main).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ROOT = common.ROOT
RUNS = ROOT / "runs"
CI_RUNS = RUNS / "ci"

SCHEMA = 1
SOURCES = ("test_notebooks", "colab", "notebook", "backfill")
MODES = ("worked", "learner")
SCOPES = ("notebook", "partial")
STATUSES = ("pass", "fail")
KERNELS = tuple(common.KERNELS)
# Fields an entry may inherit from its batch.
INHERITED = (
    "date",
    "source",
    "env",
    "env_detail",
    "mode",
    "kernel",
    "settings",
    "commit",
    "ref",
    "mktstats_commit",
    "evidence",
    "content_sha",
    "deps_sha",
    "python",
    "r",
    "colab_release",
    "runtime",
    "platform",
    "cpus",
    "memory_gb",
    "packages",
    "install_seconds",
)
REQUIRED = (
    "date",
    "source",
    "env",
    "mode",
    "kernel",
    "notebook",
    "scope",
    "status",
    "seconds",
    "evidence",
)
ALLOWED = set(INHERITED) | {
    "notebook",
    "scope",
    "scope_note",
    "status",
    "seconds",
    "phases",
    "slowest_cell",
    "cell_seconds",
    "cell_errors",
    "checkpoints",
    "verified",
    "note",
    "file",
    "index",
}
BATCH_FIELDS = set(INHERITED) | {"schema", "runs"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA = re.compile(r"^[0-9a-f]{16}$")
SECRET = re.compile(
    r"(?<![A-Za-z0-9])(sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]+|hf_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"
)


def records_of(batch: dict, name: str) -> list[dict]:
    """The flat records of one parsed batch: shared fields, then each entry's own."""
    shared = {k: batch[k] for k in INHERITED if k in batch}
    return [
        {**shared, **entry, "file": name, "index": index}
        for index, entry in enumerate(batch.get("runs", []))
    ]


def batch_of(records: list[dict]) -> dict:
    """The inverse of records_of: fields every record shares move to the top."""
    flat = [{k: v for k, v in r.items() if k not in ("file", "index")} for r in records]
    shared = {}
    for key in INHERITED:
        values = [json.dumps(r.get(key, None), sort_keys=True) for r in flat]
        if flat and all(key in r for r in flat) and len(set(values)) == 1:
            shared[key] = flat[0][key]
    runs = [{k: v for k, v in r.items() if k not in shared} for r in flat]
    return {"schema": SCHEMA, **shared, "runs": runs}


def files(dirs: tuple[Path, ...] | None = None) -> list[Path]:
    out = []
    for d in dirs or (RUNS, CI_RUNS):
        if not d.exists():
            continue
        pattern = "**/*.json" if d == CI_RUNS else "*.json"
        out += sorted(d.glob(pattern))
    return out


def check_batch(
    name: str, text: str, v: dict, today: dt.date | None = None
) -> tuple[list[dict] | None, list[str]]:
    """(records, problems) for one batch file's text; records is None when unreadable."""
    envs = v["readiness"]["envs"]
    declared = common.notebook_by_id(v)
    errors = []
    if SECRET.search(text):
        errors.append(f"{name}: contains something that looks like a credential")
    try:
        batch = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, errors + [f"{name}: not valid JSON ({exc})"]
    if not isinstance(batch, dict) or not isinstance(batch.get("runs"), list):
        return None, errors + [f"{name}: must be an object with a list of runs"]
    if not all(isinstance(entry, dict) for entry in batch["runs"]):
        return None, errors + [f"{name}: every entry of runs must be an object"]
    if batch.get("schema") != SCHEMA:
        errors.append(f"{name}: schema must be {SCHEMA}")
    if not batch["runs"]:
        errors.append(f"{name}: no runs")
    for field in sorted(set(batch) - BATCH_FIELDS):
        errors.append(f"{name}: unknown top-level field {field} (only shared fields go here)")
    today = today or dt.date.today()
    records = records_of(batch, name)
    for i, r in enumerate(records):
        where = f"{name} runs[{i}] ({r.get('notebook')})"
        for field in REQUIRED:
            if field not in r:
                errors.append(f"{where}: missing {field}")
        for field in r:
            if field not in ALLOWED:
                errors.append(f"{where}: unknown field {field}")
        for field, allowed in (
            ("source", SOURCES),
            ("mode", MODES),
            ("scope", SCOPES),
            ("status", STATUSES),
            ("kernel", KERNELS),
        ):
            if field in r and r[field] not in allowed:
                errors.append(f"{where}: {field} {r[field]!r} not in {allowed}")
        env = r.get("env")
        if not isinstance(env, str) or env not in envs:
            errors.append(f"{where}: env {env!r} is not a key of readiness.envs")
        elif envs[env].get("documented_only"):
            errors.append(f"{where}: env {env} is documented only: it has no runs")
        nb = declared.get(r.get("notebook")) if isinstance(r.get("notebook"), str) else None
        if nb is None:
            errors.append(f"{where}: no notebook named {r.get('notebook')!r} in _variables.yml")
        elif r.get("kernel") in KERNELS and r["kernel"] != nb["kernel"]:
            errors.append(f"{where}: kernel {r['kernel']} but the notebook is {nb['kernel']}")
        if not DATE.match(str(r.get("date", ""))):
            errors.append(f"{where}: date must be YYYY-MM-DD")
        else:
            try:
                day = dt.date.fromisoformat(r["date"])
            except ValueError:
                errors.append(f"{where}: {r['date']} is not a real date")
            else:
                if day > today + dt.timedelta(days=1):
                    errors.append(f"{where}: {r['date']} is in the future")
        seconds = r.get("seconds")
        is_number = isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
        if seconds is not None and not (is_number and seconds >= 0):
            errors.append(f"{where}: seconds must be a non-negative number or null")
        install = r.get("install_seconds")
        if install is not None and not (
            isinstance(install, (int, float)) and not isinstance(install, bool) and install >= 0
        ):
            errors.append(f"{where}: install_seconds must be a non-negative number or null")
        if r.get("scope") == "partial" and not r.get("scope_note"):
            errors.append(f"{where}: a partial run needs a scope_note")
        settings = r.get("settings")
        if settings is not None and not (
            isinstance(settings, dict) and all(isinstance(x, str) for x in settings.values())
        ):
            errors.append(f"{where}: settings must be an object of strings, as in the environment")
        if r.get("source") == "backfill":
            if r.get("content_sha") or r.get("deps_sha"):
                errors.append(f"{where}: a backfilled record cannot carry content_sha or deps_sha")
        else:
            for field in ("content_sha", "deps_sha"):
                if not SHA.match(str(r.get(field) or "")):
                    errors.append(f"{where}: a record made by a tool needs the notebook's {field}")
        if r.get("source") == "colab":
            errors += [f"{where}: {p}" for p in colab_problems(r)]
    return records, errors


def colab_problems(r: dict) -> list[str]:
    """What a record must carry to count as a Colab run (add_run_record.py checks it too)."""
    out = []
    if not r.get("colab_release"):
        out.append("a Colab record needs colab_release (COLAB_RELEASE_TAG)")
    if r.get("install_seconds") is None:
        out.append("a Colab record needs install_seconds")
    if r.get("kernel") == "python3":
        if not r.get("python"):
            out.append("a Colab Python record needs the python version")
        if not r.get("mktstats_commit"):
            out.append("a Colab Python record needs mktstats_commit (mktstats installed from git)")
    elif r.get("kernel") == "ir":
        if not r.get("r"):
            out.append("a Colab R record needs the R version")
        if not r.get("ref"):
            out.append("a Colab R record needs the ref its helpers were sourced at")
    return out


def load_all(dirs: tuple[Path, ...] | None = None) -> list[dict]:
    records = []
    for path in files(dirs):
        records += records_of(json.loads(path.read_text(encoding="utf-8")), _name(path))
    return records


def _name(path: Path) -> str:
    try:
        return path.relative_to(RUNS).as_posix()
    except ValueError:
        return path.name


def load_valid(v: dict, dirs: tuple[Path, ...] | None = None) -> list[dict]:
    """All records, each file parsed once; stops with every problem listed."""
    records, errors = [], []
    for path in files(dirs):
        found, problems = check_batch(_name(path), path.read_text(encoding="utf-8"), v)
        errors += problems
        records += found or []
    if errors:
        raise SystemExit("Invalid run records:\n  " + "\n  ".join(errors))
    return records


def current_hashes(v: dict) -> dict[str, tuple[str, str]]:
    """{notebook id: (content_sha, deps_sha)} of the committed notebooks, as they are now."""
    return {
        e["id"]: (common.content_sha(e["path"]), common.deps_sha(v, e))
        for e in common.existing_notebooks(v)
    }
