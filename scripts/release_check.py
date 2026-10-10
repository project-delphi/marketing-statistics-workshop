"""Can this commit be released as ready to teach? Lists every blocker, exits 1 if any.

A release needs, as of the given date (scripts/readiness.py has the rules):

- every notebook declared in _variables.yml written, with teaching evidence: a passing,
  current (content_sha and deps_sha), worked, full-settings, whole-notebook run on its
  Colab runtime, no older than readiness.max_run_age_days;
- on each CI env, a passing worked run of every notebook's current code;
- every readiness item closed (readiness.items in _variables.yml).

Records dated after the release date (plus one day of slack: records carry the recorder's
date) are ignored. The same commit and date always give the same answer. This gates the
"ready to teach" release (run it before tagging one), not the site deploy.

    python scripts/release_check.py [--as-of 2026-10-30]
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


def blockers(v: dict, records: list[dict], as_of: dt.date) -> list[str]:
    latest = (as_of + dt.timedelta(days=1)).isoformat()
    records = [r for r in records if r["date"] <= latest]
    report = readiness.build(v, records, as_of)
    out = []
    ci_envs = [k for k, e in v["readiness"]["envs"].items() if e.get("ci")]
    for e in report["notebooks"]:
        if not e["ready"]:
            out.append(f"{e['id']}: {e['why']}")
        if not e["written"]:
            continue
        for env in ci_envs:
            r = e["lanes"].get(env, {}).get("record")
            if r is None:
                out.append(f"{e['id']}: no CI run on {env}")
            elif r.get("stale"):
                out.append(f"{e['id']}: the newest CI run on {env} predates the current code")
            elif r["status"] != "pass":
                out.append(f"{e['id']}: the newest CI run on {env} failed ({r['date']})")
    for item in items(v):
        if not item["closed"]:
            out.append(f"Item {item['id']} ({item['title']}): {item['why']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=dt.date.today(),
        help="the release date (default: today)",
    )
    args = parser.parse_args(argv)
    v = common.load_variables()
    found = blockers(v, run_records.load_valid(v), args.as_of)
    if found:
        lines = [f"Not ready to release as of {args.as_of}: {len(found)} blocker(s).", ""]
        lines += [f"- {b}" for b in found]
    else:
        lines = [f"Ready to release as of {args.as_of}: no blockers."]
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"## Release check\n\n{text}\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
