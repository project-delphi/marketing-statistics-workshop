"""Execute lab notebooks top to bottom in a real kernel (nbclient).

Each notebook runs in a temporary directory with its own kernel (python3 or ir, from the
notebook's metadata), with MKTSTATS_REPO_ROOT pointing at this checkout (so the install
cell installs nothing) and MKTSTATS_QUICK=1 unless --full.

Modes:
  worked   (default) solutions bound (MKTSTATS_WORKED=1). After the first checkpoint of
           each exercise the runner injects three cells: a verify-begin cell that binds the
           exercise's stub, a copy of the checkpoint tagged raises-exception, and a
           verify-end cell that fails unless the copy failed. So one pass shows that the
           lab runs and that every checkpoint can fail on unfinished code (--no-verify
           skips this).
  learner  nothing written (MKTSTATS_WORKED=0): the run must stop at the first checkpoint
           with the harness's "TODO n is not written yet" recovery message.
  both     worked, then learner.

Run:
  python scripts/test_notebooks.py [--notebooks ID ...] [--mode worked|learner|both]
      [--full] [--no-verify] [--save DIR] [--record runs/FILE.json --env ENV]
  python scripts/test_notebooks.py --notebooks ID --verify-checkpoints   (= --mode worked)
  python scripts/test_notebooks.py --notebooks ID --learner              (= --mode learner)
  python scripts/test_notebooks.py --list-matrix   JSON list of {id, slug, kernel, path}

The exit code is 1 if any run failed; the --record file is written either way.

`--notebooks` takes ids (python/00-setup-warmup), stems (00-setup-warmup) or paths.
`--record` writes (or extends) a run-record batch (runs/README.md); `--env` names the
machine, a key of readiness.envs in _variables.yml.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import run_records  # noqa: E402

ROOT = common.ROOT
TIMEOUT = int(os.environ.get("MKTSTATS_CELL_TIMEOUT", "900"))
NOT_WRITTEN = "is not written yet"  # src/mktstats/harness.py and R/mktstats.R print it
RECORDED_SETTINGS = ("MKTSTATS_QUICK", "MKTSTATS_SABOTAGE")
PY_CHECKPOINT = re.compile(r"workshop\.checkpoint\(\s*(\d+)")
R_CHECKPOINT = re.compile(r"^\s*checkpoint\(\s*(\d+)", re.M)
PHASES = ("generated", "exercise", "solution", "checkpoint", "verify")


def tags(cell) -> list[str]:
    return cell.get("metadata", {}).get("tags", [])


def checkpoint_number(cell, kernel: str) -> int | None:
    pattern = PY_CHECKPOINT if kernel == "python3" else R_CHECKPOINT
    found = pattern.search(cell.source)
    return int(found.group(1)) if found else None


def inject_verification(nb, kernel: str) -> int:
    """After the first checkpoint of each exercise, run a copy of it on the stub."""
    begin = "workshop._verify_begin({n})" if kernel == "python3" else ".ws_verify_begin({n})"
    end = "workshop._verify_end({n})" if kernel == "python3" else ".ws_verify_end({n})"
    cells, seen = [], set()
    for cell in nb.cells:
        cells.append(cell)
        if cell.cell_type != "code" or "checkpoint" not in tags(cell) or "no-verify" in tags(cell):
            continue
        n = checkpoint_number(cell, kernel)
        if n is None or n in seen:
            continue
        seen.add(n)
        copy = nbformat.v4.new_code_cell(cell.source)
        copy.metadata["tags"] = ["raises-exception", "verify"]
        first = nbformat.v4.new_code_cell(begin.format(n=n))
        first.metadata["tags"] = ["verify"]
        last = nbformat.v4.new_code_cell(end.format(n=n))
        last.metadata["tags"] = ["verify"]
        cells += [first, copy, last]
    nb.cells = cells
    return len(seen)


def cell_seconds(cell) -> float:
    ex = cell.get("metadata", {}).get("execution", {})
    start, end = ex.get("iopub.status.busy"), ex.get("shell.execute_reply")
    if not start or not end:
        return 0.0
    parse = dt.datetime.fromisoformat
    delta = parse(end.replace("Z", "+00:00")) - parse(start.replace("Z", "+00:00"))
    return max(0.0, delta.total_seconds())


def phases(nb) -> dict[str, float]:
    """Seconds by the first matching tag; untagged cells (provided runs) are `other`."""
    out: dict[str, float] = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        key = next((t for t in PHASES if t in tags(cell)), "other")
        out[key] = round(out.get(key, 0.0) + cell_seconds(cell), 2)
    return out


def slowest(nb) -> dict | None:
    timed = [(cell_seconds(c), c.get("id")) for c in nb.cells if c.cell_type == "code"]
    if not timed:
        return None
    seconds, cell_id = max(timed)
    return {"id": cell_id, "seconds": round(seconds, 2)}


def printed(cell) -> str:
    out = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append("".join(o.get("text", "")))
        elif o.get("output_type") in ("display_data", "execute_result"):
            out.append(str(o.get("data", {}).get("text/plain", "")))
    return "".join(out)


def first_error(nb):
    for cell in nb.cells:
        if cell.cell_type == "code" and "raises-exception" not in tags(cell):
            if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                return cell
    return None


def learner_verdict(nb) -> tuple[bool, str]:
    """A learner run passes when it stops at the first checkpoint with the recovery message."""
    stopped = first_error(nb)
    checkpoints = [c for c in nb.cells if c.cell_type == "code" and "checkpoint" in tags(c)]
    if stopped is None:
        return False, "the run did not stop at an unwritten exercise"
    if not checkpoints or stopped is not checkpoints[0]:
        where = ",".join(tags(stopped)) or "untagged"
        return False, f"the run stopped in cell {stopped.get('id')} ({where}), not at checkpoint 1"
    if NOT_WRITTEN not in printed(stopped):
        return False, "checkpoint 1 failed without the 'not written yet' recovery message"
    return True, "stopped at checkpoint 1 with the recovery message"


def embedded(nb, field: str) -> str | None:
    """content_sha / deps_sha as the generator embedded them in the harness cell."""
    for cell in nb.cells:
        if cell.get("id") == common.HARNESS_ID:
            found = re.search(rf"{field}\s*=\s*['\"]([0-9a-f]+)['\"]", cell.source)
            return found.group(1) if found else None
    return None


def settings() -> dict:
    return {k: os.environ[k] for k in RECORDED_SETTINGS if os.environ.get(k) not in (None, "", "0")}


def kernel_available(name: str) -> bool:
    try:
        KernelSpecManager().get_kernel_spec(name)
        return True
    except NoSuchKernel:
        return False


def run(entry: dict, mode: str, verify: bool, save: Path | None, timeout: int) -> dict:
    path = ROOT / entry["path"]
    nb = nbformat.read(path, as_version=4)
    kernel = nb.metadata.get("kernelspec", {}).get("name", entry["kernel"])
    content, deps = embedded(nb, "content_sha"), embedded(nb, "deps_sha")
    verified = inject_verification(nb, kernel) if mode == "worked" and verify else 0
    os.environ["MKTSTATS_WORKED"] = "1" if mode == "worked" else "0"
    started = time.monotonic()
    ok, error, record = True, "", None
    with tempfile.TemporaryDirectory() as workdir:
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name=kernel,
            resources={"metadata": {"path": workdir}},
            record_timing=True,
        )
        try:
            client.execute()
        except CellExecutionError as exc:
            ok, error = False, str(exc)
        except (CellTimeoutError, DeadKernelError) as exc:
            ok, error = False, f"{type(exc).__name__}: {exc}"
        record_path = Path(workdir) / "run_record.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
    seconds = time.monotonic() - started
    if save is not None:
        out = save / Path(entry["path"]).parent.name
        out.mkdir(parents=True, exist_ok=True)
        suffix = "" if mode == "worked" else f"-{mode}"
        nbformat.write(nb, out / f"{path.stem}{suffix}.ipynb")
    if mode == "learner":
        status_ok, verdict = learner_verdict(nb)
        if ok:
            status_ok, verdict = False, "the run did not stop at an unwritten exercise"
    else:
        status_ok = ok and record is not None and record.get("status") == "pass"
        if ok and record is None:
            verdict = "the record cell did not write run_record.json"
        elif ok and record.get("status") != "pass":
            verdict = f"the notebook's own run record says {record.get('status')}"
        else:
            verdict = "ran to the end" if ok else "stopped with an error"
    return {
        "entry": entry,
        "kernel": kernel,
        "mode": mode,
        "ok": status_ok,
        "verdict": verdict,
        "error": error,
        "seconds": round(seconds, 1),
        "verified": verified,
        "content_sha": content,
        "deps_sha": deps,
        "record": record,
        "phases": phases(nb),
        "slowest_cell": slowest(nb),
        "settings": settings(),
    }


def git_commit() -> str | None:
    if os.environ.get("MKTSTATS_COMMIT"):
        return os.environ["MKTSTATS_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def entry_of(r: dict, env: str, v: dict) -> dict:
    rec = r["record"] or {}
    entry = {
        "date": dt.date.today().isoformat(),
        "source": "test_notebooks",
        "env": env,
        "env_detail": f"{platform.platform()}, runner Python {platform.python_version()}",
        "mode": r["mode"],
        "kernel": r["kernel"],
        "settings": r["settings"],
        "commit": git_commit(),
        "ref": str(v["repo"]["ref"]),
        "evidence": "scripts/test_notebooks.py --record",
        "content_sha": r["content_sha"],
        "deps_sha": r["deps_sha"],
        "notebook": r["entry"]["id"],
        "scope": "notebook",
        "status": "pass" if r["ok"] else "fail",
        "seconds": r["seconds"],
        "phases": r["phases"],
    }
    if r["slowest_cell"]:
        entry["slowest_cell"] = r["slowest_cell"]
    if r["mode"] == "worked":
        entry["verified"] = r["verified"]
    for field in (
        "python",
        "r",
        "runtime",
        "platform",
        "cpus",
        "memory_gb",
        "packages",
        "install_seconds",
        "checkpoints",
        "cell_errors",
    ):
        if rec.get(field) is not None:
            entry[field] = rec[field]
    if not r["ok"]:
        entry["note"] = r["verdict"]
    return entry


def write_record(path: Path, env: str, results: list[dict], v: dict) -> Path:
    """Write (or extend) one run-record batch file."""
    new = [entry_of(r, env, v) for r in results]
    old = []
    if path.exists():
        old = run_records.records_of(json.loads(path.read_text(encoding="utf-8")), path.name)
    body = json.dumps(run_records.batch_of(old + new), indent=2) + "\n"
    _, problems = run_records.check_batch(path.name, body, v)
    if problems:
        raise SystemExit("Run record not written:\n  " + "\n  ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def select(v: dict, wanted: list[str]) -> list[dict]:
    entries = common.existing_notebooks(v)
    if not wanted:
        return entries
    out = []
    for w in wanted:
        w = w.removesuffix(".ipynb")
        found = [
            e
            for e in entries
            if w in (e["id"], Path(e["path"]).stem, e["path"].removesuffix(".ipynb"))
        ]
        if not found:
            raise SystemExit(f"No such notebook: {w}")
        out += [e for e in found if e not in out]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebooks", nargs="*", default=[], help="ids, stems or paths")
    parser.add_argument("--mode", choices=("worked", "learner", "both"), default="worked")
    parser.add_argument(
        "--verify-checkpoints",
        action="store_true",
        help="worked mode with checkpoint verification (the default; kept for CI)",
    )
    parser.add_argument("--learner", action="store_true", help="the same as --mode learner")
    parser.add_argument("--full", action="store_true", help="full settings (no QUICK)")
    parser.add_argument("--no-verify", action="store_true", help="skip checkpoint verification")
    parser.add_argument("--save", type=Path, help="write executed notebooks here")
    parser.add_argument("--record", type=Path, help="write a run-record batch to this file")
    parser.add_argument("--env", help="a key of readiness.envs (with --record)")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="seconds per cell")
    parser.add_argument("--list-matrix", action="store_true", help="print notebooks as JSON")
    args = parser.parse_args(argv)
    v = common.load_variables()
    if args.list_matrix:
        entries = select(v, args.notebooks)
        print(
            json.dumps(
                [
                    {
                        "id": e["id"],
                        "slug": e["id"].replace("/", "-"),
                        "kernel": e["kernel"],
                        "path": e["path"],
                    }
                    for e in entries
                ]
            )
        )
        return 0
    if args.record and not args.env:
        parser.error("--record needs --env")
    if args.env and args.env not in v["readiness"]["envs"]:
        parser.error(f"--env {args.env}: not a key of readiness.envs")
    os.environ.setdefault("MKTSTATS_REPO_ROOT", str(ROOT))
    os.environ["MKTSTATS_QUICK"] = "0" if args.full else os.environ.get("MKTSTATS_QUICK", "1")
    os.environ.setdefault("MKTSTATS_CACHE", str(ROOT / ".cache" / "mktstats"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    entries = select(v, args.notebooks)
    if args.learner and args.verify_checkpoints:
        parser.error("--learner and --verify-checkpoints are separate runs")
    mode = "learner" if args.learner else "worked" if args.verify_checkpoints else args.mode
    modes = ["worked", "learner"] if mode == "both" else [mode]
    results, failures = [], 0
    for entry in entries:
        kernel = entry["kernel"]
        if not kernel_available(kernel):
            print(f"FAIL  {entry['id']}  no '{kernel}' kernel installed here", flush=True)
            failures += 1
            continue
        for mode in modes:
            r = run(entry, mode, not args.no_verify, args.save, args.timeout)
            results.append(r)
            extra = f", {r['verified']} checkpoints verified on stubs" if r["verified"] else ""
            word = "PASS" if r["ok"] else "FAIL"
            print(
                f"{word}  {entry['id']}  {mode}  {r['seconds']:.1f}s  ({r['verdict']}{extra})",
                flush=True,
            )
            if not r["ok"]:
                failures += 1
                if r["error"] and mode == "worked":
                    print(r["error"][-4000:])
    if args.record and results:
        print(f"Run record: {write_record(args.record, args.env, results, v)}")
    total = len(results) + sum(1 for e in entries if not kernel_available(e["kernel"]))
    print(f"{total - failures} of {total} runs passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
