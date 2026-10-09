"""Add a run record printed by a notebook's record cell (run_record()) to runs/.

On Colab, after Run all, copy everything between the "run record" marker lines that the
last code cell prints (or download run_record.json from the Files panel), then:

    pbpaste | python scripts/add_run_record.py
    python scripts/gen_tables.py

The env is read from the record: a Python record from Colab is colab-py, an R record is
colab-r (the teaching envs of readiness.envs for each kernel). A Colab record must carry
colab_release, install_seconds, the Python or R version, and mktstats_commit (Python: the
package came from git) or ref (R: the helpers were sourced at that ref). Records from
elsewhere need --env. A learner-mode record is refused: it shows the participant's work,
not that the lab runs as taught.

The record is written to runs/<date>-<env>-<language>-<notebook>.json; a second record
for the same day, env and notebook is added to the same file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import readiness  # noqa: E402
import run_records  # noqa: E402

BEGIN = "----- run record"
END = "----- end of run record"


def extract(text: str) -> dict:
    """The JSON object in pasted text, with or without the marker lines around it."""
    if BEGIN in text:
        text = text.split(BEGIN, 1)[1]
    if END in text:
        text = text.split(END, 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise SystemExit("No run record found in the input")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"The run record is not valid JSON: {exc}") from exc


def env_of(v: dict, record: dict, given: str | None) -> str:
    if given:
        return given
    if not record.get("colab_release"):
        raise SystemExit("Not a Colab run (no colab_release): name the machine with --env")
    env = readiness.teaching_env(v, record.get("kernel", ""))
    if env is None:
        raise SystemExit(f"No teaching env for kernel {record.get('kernel')!r} in readiness.envs")
    return env


def batch_from(v: dict, record: dict, env: str) -> dict:
    teaching = bool(v["readiness"]["envs"][env].get("teaching"))
    lang = (
        "Python " + str(record.get("python"))
        if record.get("kernel") == "python3"
        else ("R " + str(record.get("r")))
    )
    detail = [lang, f"{record.get('cpus')} CPUs"]
    if record.get("colab_release"):
        detail.insert(0, f"Colab {record['colab_release']}")
    entry = {
        "notebook": record["notebook"],
        "scope": "notebook",
        "status": record["status"],
        "seconds": record.get("seconds"),
    }
    for field in ("checkpoints", "cell_seconds", "cell_errors", "verified"):
        if record.get(field) is not None:
            entry[field] = record[field]
    batch = {
        "date": record["date"],
        "source": "colab" if teaching else "notebook",
        "env": env,
        "env_detail": ", ".join(detail),
        "mode": record["mode"],
        "kernel": record.get("kernel"),
        "settings": {k: str(x) for k, x in (record.get("settings") or {}).items()},
        "evidence": "run_record() output, added with scripts/add_run_record.py",
        "content_sha": record.get("content_sha"),
        "deps_sha": record.get("deps_sha"),
    }
    for field in (
        "ref",
        "mktstats_commit",
        "colab_release",
        "python",
        "r",
        "runtime",
        "platform",
        "cpus",
        "memory_gb",
        "packages",
        "install_seconds",
    ):
        if record.get(field) is not None:
            batch[field] = record[field]
    return {**batch, "runs": [entry]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", type=Path, help="default: read standard input")
    parser.add_argument("--env", help="a key of readiness.envs, if not a Colab run")
    parser.add_argument("--runs", type=Path, default=run_records.RUNS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    record = extract(text)
    v = common.load_variables()
    if record.get("mode") != "worked":
        raise SystemExit(
            "Not added: a learner-mode run is evidence of the participant's work, not that the"
            " lab runs as taught. Set WORKED_EXAMPLE, Run all, and paste that record."
        )
    for field in ("notebook", "date", "status", "kernel", "content_sha", "deps_sha"):
        if not record.get(field):
            raise SystemExit(f"Not added: the record has no {field}")
    env = env_of(v, record, args.env)
    if env not in v["readiness"]["envs"]:
        raise SystemExit(f"Not added: --env {env} is not a key of readiness.envs")
    if v["readiness"]["envs"][env].get("teaching"):
        problems = run_records.colab_problems(record)
        if problems:
            raise SystemExit("Not added:\n  " + "\n  ".join(problems))
    nb = common.notebook_by_id(v).get(record["notebook"])
    if nb is None:
        raise SystemExit(f"Not added: no notebook named {record['notebook']!r}")
    current = run_records.current_hashes(v).get(record["notebook"])
    if current != (record["content_sha"], record["deps_sha"]):
        print(
            "Note: the record was made against other code or deps than the committed notebook's;"
            " it is kept, and the readiness page marks it stale."
        )
    batch = batch_from(v, record, env)
    lang, stem = record["notebook"].split("/", 1)
    out = args.runs / f"{record['date']}-{env}-{lang}-{stem}.json"
    records = run_records.records_of(batch, out.name)
    if out.exists():
        records = (
            run_records.records_of(json.loads(out.read_text(encoding="utf-8")), out.name) + records
        )
    body = json.dumps(run_records.batch_of(records), indent=2) + "\n"
    _, problems = run_records.check_batch(out.name, body, v)
    if problems:
        raise SystemExit("Not added:\n  " + "\n  ".join(problems))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"Wrote {out}. Now run python scripts/gen_tables.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
