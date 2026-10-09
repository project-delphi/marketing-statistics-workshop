"""What is known about whether each notebook can be taught, and on what evidence.

Reads `readiness` in _variables.yml and the run records (scripts/run_records.py). Used by
scripts/gen_tables.py for the readiness pages and by scripts/release_check.py.

Nothing here reads the clock: the date to judge age against is passed in. The generated
pages pass the newest record's date, so they change only when the repository does; the
release check passes the release date.

Rules (runs/README.md):

- A record is **stale** when its content_sha or deps_sha differs from the notebook's now.
- **Teaching evidence** is the newest record that is: on an env with `teaching: true` for
  the notebook's kernel (Colab), worked mode, the whole notebook, settings without QUICK
  or SABOTAGE, not a backfill. It counts when it passed, is not stale, and is at most
  `readiness.max_run_age_days` old. A newer failure replaces an older pass.
- Envs with `documented_only: true` are always "documented, not run".
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import run_records  # noqa: E402

QUICK = "MKTSTATS_QUICK"
SABOTAGE = "MKTSTATS_SABOTAGE"


def _on(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def is_quick(r: dict) -> bool:
    return _on((r.get("settings") or {}).get(QUICK, ""))


def is_sabotaged(r: dict) -> bool:
    return bool((r.get("settings") or {}).get(SABOTAGE))


def is_stale(r: dict, current: tuple[str, str] | None) -> bool:
    if current is None:
        return True  # the notebook does not exist (any more)
    return (r.get("content_sha"), r.get("deps_sha")) != current


def teaching_env(v: dict, kernel: str) -> str | None:
    for key, env in v["readiness"]["envs"].items():
        if env.get("teaching") and env.get("kernel") == kernel:
            return key
    return None


def teaching_eligible(v: dict, r: dict) -> bool:
    env = v["readiness"]["envs"].get(r["env"], {})
    return (
        bool(env.get("teaching"))
        and env.get("kernel") == r.get("kernel")
        and r["mode"] == "worked"
        and r["scope"] == "notebook"
        and r["source"] != "backfill"
        and not is_quick(r)
        and not is_sabotaged(r)
    )


def newest(records: list[dict]) -> dict | None:
    """The most telling record: current code first, then the most recent; on the same date
    a failure over a pass, the whole notebook over a part, full settings over QUICK, then
    the later file and entry."""
    if not records:
        return None
    return max(
        records,
        key=lambda r: (
            not r.get("stale"),
            r["date"],
            r["status"] == "fail",
            r["scope"] == "notebook",
            not is_quick(r),
            r.get("file", ""),
            r.get("index", 0),
        ),
    )


def age_days(r: dict, as_of: dt.date | None) -> int | None:
    if as_of is None:
        return None
    return (as_of - dt.date.fromisoformat(r["date"])).days


def evidence(
    v: dict,
    entry: dict,
    records: list[dict],
    current: tuple[str, str] | None,
    as_of: dt.date | None,
) -> dict:
    """The evidence for one notebook: the newest record per env lane, the teaching verdict."""
    envs = v["readiness"]["envs"]
    mine = [{**r, "stale": is_stale(r, current)} for r in records if r["notebook"] == entry["id"]]
    lanes = {}
    home = teaching_env(v, entry["kernel"])
    for key, env in envs.items():
        if env.get("teaching") and key != home:
            continue  # the other language's Colab lane
        if env.get("documented_only"):
            lanes[key] = {"state": "documented", "record": None}
            continue
        worked = [r for r in mine if r["env"] == key and r["mode"] == "worked"]
        r = newest(worked)
        lanes[key] = {"state": "not run" if r is None else "run", "record": r}
    eligible = newest([r for r in mine if teaching_eligible(v, r)])
    limit = v["readiness"]["max_run_age_days"]
    ready, why = False, ""
    written = (common.ROOT / entry["path"]).exists()
    if not written:
        why = "not written yet"
    elif eligible is None:
        env_label = envs[home]["label"] if home else "its teaching runtime"
        why = f"no worked, full-settings run of the whole notebook on {env_label}"
    elif eligible.get("stale"):
        why = f"the newest teaching run ({eligible['date']}) predates the current code or deps"
    elif eligible["status"] != "pass":
        why = f"the newest teaching run failed ({eligible['date']})"
    elif as_of is not None and age_days(eligible, as_of) > limit:
        why = (
            f"the newest teaching run is {age_days(eligible, as_of)} days old"
            f" ({eligible['date']}); the limit is {limit}"
        )
    else:
        ready, why = True, f"passed on {eligible['date']}"
    learner = newest([r for r in mine if r["mode"] == "learner"])
    return {
        "id": entry["id"],
        "module": entry["module"],
        "kernel": entry["kernel"],
        "written": written,
        "lanes": lanes,
        "teaching": eligible,
        "learner": learner,
        "ready": ready,
        "why": why,
    }


def passed(r: dict | None) -> bool:
    return r is not None and r["status"] == "pass" and not r.get("stale")


def build(v: dict, records: list[dict], as_of: dt.date | None) -> dict:
    """Everything the readiness pages and the release check need, computed once."""
    current = run_records.current_hashes(v)
    entries = common.notebooks(v)
    ev = [evidence(v, e, records, current.get(e["id"]), as_of) for e in entries]
    ci_envs = [k for k, e in v["readiness"]["envs"].items() if e.get("ci")]
    other_envs = [
        k
        for k, e in v["readiness"]["envs"].items()
        if not e.get("teaching") and not e.get("ci") and not e.get("documented_only")
    ]

    def lane_passed(e: dict, keys: list[str]) -> bool:
        return any(passed(e["lanes"].get(k, {}).get("record")) for k in keys)

    summary = {
        "notebooks": len(ev),
        "written": sum(e["written"] for e in ev),
        "ready": sum(e["ready"] for e in ev),
        "ci_passed": sum(lane_passed(e, ci_envs) for e in ev),
        "elsewhere_passed": sum(lane_passed(e, other_envs) for e in ev),
        "elsewhere_full": sum(
            any(
                passed(e["lanes"].get(k, {}).get("record"))
                and not is_quick(e["lanes"][k]["record"])
                for k in other_envs
            )
            for e in ev
        ),
        "as_of": as_of.isoformat() if as_of else None,
        "max_age": v["readiness"]["max_run_age_days"],
    }
    summary["all_ready"] = summary["ready"] == summary["notebooks"] and summary["notebooks"] > 0
    return {"notebooks": ev, "summary": summary}


def newest_date(records: list[dict]) -> dt.date | None:
    """The date the generated pages judge age against: the newest record's."""
    dates = [r["date"] for r in records]
    return dt.date.fromisoformat(max(dates)) if dates else None


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "time not recorded"
    if seconds < 100:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 10:
        return f"{minutes:.1f} min"
    return f"{minutes:.0f} min"


def describe(r: dict) -> str:
    """'passed 2026-10-09, 41 s, QUICK settings' for one record."""
    word = "passed" if r["status"] == "pass" else "**failed**"
    text = f"{word} {r['date']}, {duration(r.get('seconds'))}"
    if is_quick(r):
        text += ", QUICK settings"
    if is_sabotaged(r):
        text += ", sabotage test"
    if r["scope"] == "partial":
        text += f" (part: {r.get('scope_note', '')})"
    if r.get("stale"):
        text += ", before the notebook or its deps last changed"
    return text
