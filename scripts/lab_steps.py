"""The steps of one lab, read from the headings of its source (CONTRIBUTING.md):

    # Part A · Title
    ## Exercise N · Title (M minutes)
    ## Stretch (optional) · Title
    ## Decision · Title

scripts/gen_tables.py turns them into `_includes/lab-NN.md` and counts the exercises for
`_includes/facts.md`. Nothing is invented: a step appears only if the source has a heading
for it, and a time only if the heading states one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import jupytext

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PART = re.compile(r"^#\s+Part\s+(\S+)\s+·\s+(.+?)\s*$")
EXERCISE = re.compile(r"^##\s+Exercise\s+(\d+)\s+·\s+(.+?)(?:\s*\((\d+)\s*minutes?\))?\s*$")
STRETCH = re.compile(r"^##\s+Stretch(?:\s*\(optional\))?(?:\s*·\s*(.+?))?\s*$")
DECISION = re.compile(r"^##\s+Decision(?:\s*·\s*(.+?))?\s*$")


def headings(entry: dict) -> list[str]:
    """Markdown headings of a notebook's source, in order (code fences skipped)."""
    path = common.ROOT / entry["source"]
    if not path.exists():
        return []
    nb = jupytext.read(path, fmt=common.KERNELS[entry["kernel"]]["fmt"])
    out = []
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        fenced = False
        for line in cell.source.splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
            elif not fenced and line.startswith("#"):
                out.append(line.rstrip())
    return out


def rows(entry: dict) -> list[dict]:
    """[{kind: part|exercise|decision|stretch, label, title, minutes}] in source order."""
    out = []
    for line in headings(entry):
        if m := PART.match(line):
            out.append({"kind": "part", "label": f"Part {m[1]}", "title": m[2], "minutes": None})
        elif m := EXERCISE.match(line):
            out.append(
                {
                    "kind": "exercise",
                    "label": f"Exercise {m[1]}",
                    "number": int(m[1]),
                    "title": m[2],
                    "minutes": int(m[3]) if m[3] else None,
                }
            )
        elif m := DECISION.match(line):
            out.append(
                {"kind": "decision", "label": "Decision", "title": m[1] or "", "minutes": None}
            )
        elif m := STRETCH.match(line):
            out.append(
                {
                    "kind": "stretch",
                    "label": "Stretch (optional)",
                    "title": m[1] or "",
                    "minutes": None,
                }
            )
    return out


def exercises(entry: dict) -> list[dict]:
    return [r for r in rows(entry) if r["kind"] == "exercise"]


def minutes(found: list[dict]) -> int:
    return sum(r["minutes"] or 0 for r in found if r["kind"] == "exercise")
