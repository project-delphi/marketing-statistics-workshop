"""The generated includes: graceful when content is missing, honest about where times come
from, and stable for a split module and a two-notebook lab."""

import copy

import common
import gen_tables as g
import lab_steps
import readiness
import run_records

V = common.load_variables()
PY = "python/00-setup-warmup"
CURRENT = run_records.current_hashes(V)


def record(**fields):
    content, deps = CURRENT[PY]
    base = {
        "notebook": PY,
        "kernel": "python3",
        "date": "2026-10-09",
        "source": "colab",
        "env": "colab-py",
        "mode": "worked",
        "scope": "notebook",
        "status": "pass",
        "seconds": 150.0,
        "evidence": "test",
        "settings": {},
        "content_sha": content,
        "deps_sha": deps,
        "file": "t.json",
        "index": 0,
    }
    return {**base, **fields}


def test_missing_content_renders_as_to_be_written():
    v = copy.deepcopy(V)
    v["modules"]["m01"].update(summary="", objectives=[], decision="")
    text = g.module_header(v, "m01")
    assert "*Summary to be written.*" in text
    assert "*Objectives to be written.*" in text
    assert "*Decision to be written.*" in text
    assert "::: {.module-summary}\n\n:::" not in text


def test_filled_content_is_shown():
    v = copy.deepcopy(V)
    v["modules"]["m01"].update(
        summary="Turn a log into a customer table.",
        objectives=["Compute RFM in SQL"],
        decision="Which customers to stop retargeting.",
    )
    text = g.module_header(v, "m01")
    assert "Turn a log into a customer table." in text
    assert "- Compute RFM in SQL" in text
    assert "**The decision.** Which customers to stop retargeting." in text
    assert "to be written" not in text.split("::: {.module-details}")[0]


def test_facts_count_exercises_from_the_notebooks():
    expected = sum(len(lab_steps.exercises(e)) for e in common.existing_notebooks(V))
    assert f"**{expected}** hands-on exercises" in g.facts(V)


def test_colab_times_never_come_from_other_machines():
    entry = common.notebook_by_id(V)[PY]
    docker = record(env="docker-arm64", source="test_notebooks")
    assert g.teaching_time(V, entry, [docker]) == "not yet timed on Colab"
    assert g.teaching_time(V, entry, [record()]) == "2.5 min on Colab (2026-10-09)"
    quick = record(settings={"MKTSTATS_QUICK": "1"})
    assert g.teaching_time(V, entry, [quick]) == "not yet timed on Colab"


def test_install_times_come_only_from_hosted_runtimes():
    local = record(
        env="docker-arm64", source="test_notebooks", runtime="local", install_seconds=0.0
    )
    assert "not yet measured" in g.install_times(V, [local])
    colab = record(runtime="colab", install_seconds=31.0)
    text = g.install_times(V, [colab])
    assert f"`{PY}` | 31 s | 2026-10-09" in text


def test_the_summary_never_claims_readiness_without_colab_runs():
    report = readiness.build(V, [record(env="docker-arm64", source="test_notebooks")], None)
    assert "not yet ready to teach" in g.readiness_summary(report)


def test_the_readiness_page_marks_documented_envs():
    report = readiness.build(V, [], None)
    assert "documented, not run" in g.readiness_page(V, report)


def split_variables():
    v = copy.deepcopy(V)
    v["modules"]["m12"]["minutes"] = {"briefing": 15, "lab": 180, "debrief": 60}
    v["days"]["d5"]["blocks"] = [
        {"kind": "warmup", "minutes": 15},
        {"kind": "module", "id": "m11"},
        {"kind": "break", "minutes": 15},
        {"kind": "module", "id": "m12", "part": "briefing"},
        {"kind": "module", "id": "m12", "part": "lab", "minutes": 80, "title": "Pair work 1"},
        {"kind": "lunch", "minutes": 50},
        {"kind": "module", "id": "m12", "part": "lab", "minutes": 100, "title": "Pair work 2"},
        {"kind": "break", "minutes": 10},
        {"kind": "module", "id": "m12", "part": "debrief", "title": "Presentations and scoring"},
        {"kind": "wrapup", "minutes": 20},
    ]
    return v


def test_a_module_split_by_part():
    v = split_variables()
    assert g.part_problems(v) == []
    assert g.part_times(v, "m12") == "11:25 briefing · 11:40 lab · 13:50 lab · 15:40 debrief"
    assert g.module_clock(v, "m12") == "11:25–13:00 · 13:50–15:30 · 15:40–16:40"
    schedule = g.schedule(v)
    assert "| 13:50–15:30 | [Lab · Pair work 2]{.slot-lab} | Module 12 |" in schedule
    assert "[Debrief · Presentations and scoring]{.slot-debrief}" in schedule
    assert "11:25 briefing · 11:40 lab" in g.module_header(v, "m12")


def test_split_parts_must_add_up_and_end_by_five():
    v = split_variables()
    v["days"]["d5"]["blocks"][4]["minutes"] = 70
    assert any("lab segments add up to 170 minutes, not 180" in p for p in g.part_problems(v))
    v = split_variables()
    v["days"]["d5"]["blocks"][5]["minutes"] = 90
    assert any("after 17:00" in p for p in g.part_problems(v))


def test_a_lab_in_two_languages_is_one_choice():
    assert g.alternatives(g.notebooks_of(V, "m00"))
    text = g.lab_block(V, "m00")
    assert "same lab in two languages: do one" in text
    assert "**Python subtotal: 3 exercises** | **34 min**" in text


def test_two_different_notebooks_share_the_slot():
    entries = g.notebooks_of(V, "m03")
    assert len(entries) == 2 and not g.alternatives(entries)
