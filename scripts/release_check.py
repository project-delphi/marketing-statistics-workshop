"""Can this commit be released as ready to teach? Lists every blocker, exits 1 if any.

A release of the core workshop (pre-work and Days 1-5) needs, as of the given date
(scripts/readiness.py has the rules):

- every core notebook declared in _variables.yml written, with teaching evidence: a
  passing, current (content_sha and deps_sha), worked, full-settings, whole-notebook run on
  its Colab runtime, no older than readiness.max_run_age_days;
- none of them declared `in_ref: false` (repo.ref does not contain it, so its Colab link
  cannot open it; the release commit that moves repo.ref removes the flag);
- on each CI env, a passing worked run of every core notebook's current code;
- every readiness item closed (readiness.items in _variables.yml).

The optional extension track (modules on a day with `track: extensions`) is checked by the
same rules and always reported, but it blocks the release only with --extensions. By
default an unwritten or unrun extension notebook is listed under "Extensions (not
required)" and the exit status is decided by the core alone, so the five-day workshop can
be released while extensions are still being built. Pass --extensions to release the
extensions as ready to teach too: then their blockers count like the core's.

Records dated after the release date (plus one day of slack: records carry the recorder's
date) are ignored. The same commit, date and flags always give the same answer. This gates
the "ready to teach" release (run it before tagging one), not the site deploy.

    python scripts/release_check.py [--as-of 2026-10-30] [--extensions]
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import readiness  # noqa: E402
import run_records  # noqa: E402


def item_closed(v: dict, item: dict) -> tuple[bool, str]:
    """(closed, why) for one readiness item, by its mechanical check."""
    check = item.get("check", {})
    if "exists" in check:
        paths = check["exists"] if isinstance(check["exists"], list) else [check["exists"]]
        missing = [p for p in paths if not (common.ROOT / p).exists()]
        if missing:
            return False, f"missing: {', '.join(missing)}"
        return True, "exists"
    if "var" in check:
        node = v
        try:
            for part in check["var"].split("."):
                node = node[part]
        except (KeyError, TypeError):
            return False, f"{check['var']} is missing from _variables.yml"
        return node == check.get("equals"), f"{check['var']} is {node!r}"
    if "manual" in check:
        return bool(check.get("done")), "done" if check.get("done") else "not done"
    return False, f"unknown check {check}"


def items(v: dict) -> list[dict]:
    raw = v["readiness"].get("items") or []
    pairs = raw.items() if isinstance(raw, dict) else enumerate(raw)
    out = []
    for key, item in pairs:
        closed, why = item_closed(v, item)
        out.append(
            {
                "id": str(item.get("id", key)),
                "title": item.get("title", ""),
                "closed": closed,
                "why": why,
            }
        )
    return out


def notebook_blockers(v: dict, e: dict) -> list[str]:
    """What keeps one notebook (readiness.evidence) from release: teaching evidence, CI."""
    out = []
    if not e["ready"]:
        out.append(f"{e['id']}: {e['why']}")
    if not e["written"]:
        return out
    if not e["in_ref"]:
        out.append(
            f"{e['id']}: declared `in_ref: false`, so its Colab link cannot open it: move"
            f" repo.ref ({v['repo']['ref']}) to a tag that contains it and remove the flag"
        )
    for env in [k for k, x in v["readiness"]["envs"].items() if x.get("ci")]:
        r = e["lanes"].get(env, {}).get("record")
        if r is None:
            out.append(f"{e['id']}: no CI run on {env}")
        elif r.get("stale"):
            out.append(f"{e['id']}: the newest CI run on {env} predates the current code")
        elif r["status"] != "pass":
            out.append(f"{e['id']}: the newest CI run on {env} failed ({r['date']})")
    return out


def check(v: dict, records: list[dict], as_of: dt.date) -> dict:
    """{"core": [...], "extensions": [...], "items": [...]}: blockers by part."""
    latest = (as_of + dt.timedelta(days=1)).isoformat()
    records = [r for r in records if r["date"] <= latest]
    report = readiness.build(v, records, as_of)
    out = {"core": [], "extensions": [], "items": []}
    for e in report["notebooks"]:
        part = "core" if e["track"] == common.CORE else "extensions"
        out[part] += notebook_blockers(v, e)
    for item in items(v):
        if not item["closed"]:
            out["items"].append(f"Item {item['id']} ({item['title']}): {item['why']}")
    return out


def blockers(
    v: dict, records: list[dict], as_of: dt.date, extensions: bool = False
) -> list[str]:
    """The blockers that fail the release: the core's and the items, plus the extensions'
    when `extensions` is set."""
    found = check(v, records, as_of)
    return found["core"] + (found["extensions"] if extensions else []) + found["items"]


def has_extensions(v: dict) -> bool:
    return any(e["track"] != common.CORE for e in common.notebooks(v))


def report_text(v: dict, found: dict, as_of: dt.date, extensions: bool) -> tuple[str, bool]:
    """(the text to print, whether the release is blocked)."""
    failing = found["core"] + found["items"] + (found["extensions"] if extensions else [])
    scope = "the core workshop and the extensions" if extensions else "the core workshop"
    if failing:
        lines = [f"Not ready to release {scope} as of {as_of}: {len(failing)} blocker(s).", ""]
        lines += [f"- {b}" for b in failing]
    else:
        lines = [f"Ready to release {scope} as of {as_of}: no blockers."]
    if has_extensions(v) and not extensions:
        lines += [""]
        if found["extensions"]:
            lines += [
                "Extensions (reported, not required; --extensions requires them):"
                f" {len(found['extensions'])} blocker(s) if required.",
                "",
            ]
            lines += [f"- {b}" for b in found["extensions"]]
        else:
            lines += ["Extensions (not required): no open items; --extensions would pass too."]
    return "\n".join(lines), bool(failing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=dt.date.today(),
        help="the release date (default: today)",
    )
    parser.add_argument(
        "--extensions",
        action="store_true",
        help="also require the optional extension notebooks (default: report them only)",
    )
    args = parser.parse_args(argv)
    v = common.load_variables()
    found = check(v, run_records.load_valid(v), args.as_of)
    text, blocked = report_text(v, found, args.as_of, args.extensions)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"## Release check\n\n{text}\n")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
