"""The optional extension track (modules on a day with `track: extensions`) is counted,
judged, released and linked apart from the core workshop (pre-work and Days 1-5): it never
makes the core look unready, never inflates the core's counts, and the core never leads
into it (CONTRIBUTING.md, "The core workshop and the optional extensions"). Notebook
footers: tests/test_footers.py."""

import copy
import datetime as dt

import pytest

import common
import gen_tables as g
import readiness
import release_check
import run_records

V = common.load_variables()
CURRENT = run_records.current_hashes(V)
DAY = dt.date(2026, 10, 9)
CORE_NB = [e for e in common.notebooks(V) if e["track"] == common.CORE]
EXT_NB = [e for e in common.notebooks(V) if e["track"] != common.CORE]
EXT_MODULES = [k for k in sorted(V["modules"]) if common.module_track(V, k) != common.CORE]


def rec(entry: dict, current: dict = CURRENT, **fields) -> dict:
    content, deps = current[entry["id"]]
    colab = "colab-py" if entry["kernel"] == "python3" else "colab-r"
    base = {
        "notebook": entry["id"],
        "kernel": entry["kernel"],
        "date": DAY.isoformat(),
        "source": "colab",
        "env": colab,
        "mode": "worked",
        "scope": "notebook",
        "status": "pass",
        "seconds": 100.0,
        "evidence": "test",
        "settings": {},
        "content_sha": content,
        "deps_sha": deps,
        "file": "test.json",
        "index": 0,
    }
    return {**base, **fields}


def all_passed(entries: list[dict], current: dict = CURRENT) -> list[dict]:
    """A passing Colab run and a passing CI run of every written notebook in `entries`."""
    out = []
    for e in entries:
        if e["id"] in current:
            out.append(rec(e, current))
            out.append(rec(e, current, env="ci-amd64", source="test_notebooks", index=1))
    return out


def test_there_is_an_extension_track_to_check():
    assert common.extension_days(V), "no day with track: extensions"
    assert EXT_NB and CORE_NB


def test_the_track_comes_from_the_day():
    assert common.day_track(V, "d1") == common.CORE
    assert common.day_track(V, "d0") == common.CORE  # the pre-work is part of the core
    for dkey in common.extension_days(V):
        for key, m in V["modules"].items():
            if m["day"] == dkey:
                assert common.module_track(V, key) == V["days"][dkey]["track"]


@pytest.mark.parametrize(
    "nums, text",
    [
        ([13, 14, 15, 16, 17], "Modules 13–17"),
        ([13], "Module 13"),
        ([17, 13, 15], "Modules 13, 15 and 17"),
    ],
)
def test_module_range(nums, text):
    assert common.module_range(nums) == text


# ---- readiness and the verdict -------------------------------------------------------------


def test_the_core_verdict_ignores_unready_extensions():
    report = readiness.build(V, all_passed(CORE_NB), DAY)
    s = report["summary"]
    assert s["notebooks"] == len(CORE_NB) and s["all_ready"]
    assert s["extensions"]["notebooks"] == len(EXT_NB)
    assert s["extensions"]["ready"] == 0 and not s["extensions"]["all_ready"]
    text = g.readiness_summary(V, report)
    assert ": ready to teach.**" in text
    assert f"of its {len(CORE_NB)} notebooks" in text
    assert "**Optional extensions**" in text and "not part of that verdict" in text
    assert f"0 of {len(EXT_NB)} notebooks have a current, passing Colab run" in text


def test_a_ready_extension_never_counts_toward_the_core():
    # Make the pre-work an extension day in a copy: its notebooks exist and can be ready.
    v = copy.deepcopy(V)
    v["days"]["d0"]["track"] = "extensions"
    current = run_records.current_hashes(v)
    moved = [e for e in common.notebooks(v) if e["module"] == "m00"]
    report = readiness.build(v, all_passed(common.notebooks(v), current), DAY)
    s = report["summary"]
    assert s["notebooks"] == len(CORE_NB) - len(moved)
    assert s["extensions"]["ready"] == len(moved)
    assert {e["id"] for e in report["notebooks"] if e["track"] != common.CORE} >= {
        e["id"] for e in moved
    }


def test_the_readiness_page_has_a_section_per_track():
    page = g.readiness_page(V, readiness.build(V, [], None))
    core, _, rest = page.partition("{#extensions}")
    assert "{#core}" in core and rest
    for e in EXT_NB:
        assert f"`{e['id']}`" not in core and f"`{e['id']}`" in rest
    for e in CORE_NB:
        assert f"`{e['id']}`" in core and f"`{e['id']}`" not in rest


# ---- counts, cards, path, schedule, notebooks --------------------------------------------------


def test_facts_count_the_core_and_name_the_extensions_apart():
    text = g.facts(V)
    n_py = sum(e["language"] == "python" for e in CORE_NB)
    n_r = sum(e["language"] == "r" for e in CORE_NB)
    assert f"**{len(CORE_NB)}** lab notebooks ({n_py} Python, {n_r} R)" in text
    assert f"**+{len(EXT_MODULES)}** optional extension modules" in text
    taught = [
        k
        for k, m in V["modules"].items()
        if common.module_track(V, k) == common.CORE and not m.get("optional")
    ]
    assert f"**{len(taught)}** modules + optional pre-work" in text


def test_the_extension_card_links_to_its_page_with_a_range():
    cards = g.days_cards(V)
    for dkey in common.extension_days(V):
        d = V["days"][dkey]
        nums = common.day_module_numbers(V, dkey)
        assert f"[{d['title']}](/{d['track']}.qmd){{.day-card-title}}" in cards
        assert f"[{common.module_range(nums)}, optional]{{.day-card-span}}" in cards


def test_the_core_path_stops_at_the_core_and_extensions_say_what_they_build_on():
    core, ext = g.path(V), g.path(V, core=False)
    for key in EXT_MODULES:
        link = g.module_link(key, V["modules"][key])
        assert link not in core and link in ext
    assert "Builds on [Module" in ext and "Builds on" not in core


def test_the_schedule_lists_the_extensions_without_a_timetable():
    head, _, ext = g.schedule(V).partition("{#extensions}")
    assert ext and "| Time |" not in ext and "self-paced" in ext
    # The pre-work keeps its wording; the extensions are done after the week, not before.
    assert "before the workshop" not in ext and "after the week" in ext
    assert "On your own, before the workshop: [Module 0 · " in head
    assert "before the workshop" not in g.run_sheet(V, common.extension_days(V)[0])
    for key in EXT_MODULES:
        assert g.module_link(key, V["modules"][key]) in ext
        assert g.module_link(key, V["modules"][key]) not in head


def test_the_notebooks_tables_split_by_track():
    core = g.notebooks_table(V, [])
    ext = g.notebooks_table(V, [], core=False)
    for e in EXT_NB:
        stem = e["path"].rsplit("/", 1)[-1].removesuffix(".ipynb")
        assert f"`{stem}`" in ext and f"`{stem}`" not in core


def test_an_extension_page_lists_its_declared_prerequisites():
    for key in EXT_MODULES:
        text = g.module_header(V, key)
        before = text.split("## Before you start", 1)[1]
        n = common.module_number(key)
        assert f"Module {n - 1} · " not in before  # not "the module before it" by number
        for item in V["modules"][key].get("prerequisites") or []:
            plain = g.link_modules(V, str(item))
            assert plain in before


@pytest.mark.parametrize(
    "text, nums",
    [
        ("Module 4: the shifted-beta-geometric model", [4]),
        ("Modules 8–10: the MMM chain", [8, 9, 10]),
        ("Modules 2 and 4", [2, 4]),
        ("Logistic regression coefficients", []),
    ],
)
def test_referenced_modules(text, nums):
    assert g.referenced_modules(text) == nums


# ---- the release check ---------------------------------------------------------------------------


def test_release_fails_on_the_core_and_requires_extensions_only_when_asked():
    records = all_passed(CORE_NB)
    assert release_check.blockers(V, records, DAY) == []
    found = release_check.check(V, records, DAY)
    assert found["core"] == [] and len(found["extensions"]) >= len(EXT_NB)
    required = release_check.blockers(V, records, DAY, extensions=True)
    assert {b.split(":", 1)[0] for b in required} == {e["id"] for e in EXT_NB}
    text, blocked = release_check.report_text(V, found, DAY, extensions=False)
    assert not blocked and "Ready to release the core workshop" in text
    assert "Extensions (reported, not required" in text
    text, blocked = release_check.report_text(V, found, DAY, extensions=True)
    assert blocked and "the core workshop and the extensions" in text


def test_release_still_fails_when_the_core_is_not_ready():
    found = release_check.check(V, all_passed(CORE_NB[1:]), DAY)
    assert any(b.startswith(f"{CORE_NB[0]['id']}:") for b in found["core"])
    _, blocked = release_check.report_text(V, found, DAY, extensions=False)
    assert blocked
