"""Run records and the readiness rules: what counts as evidence that a lab can be taught
(runs/README.md, scripts/readiness.py, scripts/add_run_record.py, scripts/release_check.py)."""

import datetime as dt
import json
from pathlib import Path

import pytest

import add_run_record
import common
import readiness
import release_check
import run_records

V = common.load_variables()
CURRENT = run_records.current_hashes(V)
PY = "python/00-setup-warmup"
R = "r/00-setup-warmup"
DAY = dt.date(2026, 10, 9)


def rec(**fields):
    notebook = fields.get("notebook", PY)
    content, deps = CURRENT[notebook]
    base = {
        "notebook": notebook,
        "kernel": "python3" if notebook.startswith("python/") else "ir",
        "date": "2026-10-09",
        "source": "colab",
        "env": "colab-py" if notebook.startswith("python/") else "colab-r",
        "mode": "worked",
        "scope": "notebook",
        "status": "pass",
        "seconds": 120.0,
        "evidence": "test",
        "settings": {},
        "content_sha": content,
        "deps_sha": deps,
        "file": "test.json",
        "index": 0,
    }
    return {**base, **fields}


def evidence(records, notebook=PY, as_of=DAY):
    entry = common.notebook_by_id(V)[notebook]
    return readiness.evidence(V, entry, records, CURRENT[notebook], as_of)


# ---- readiness rules -------------------------------------------------------------------------


def test_a_fresh_colab_run_is_teaching_evidence():
    e = evidence([rec()])
    assert e["ready"] and e["why"] == "passed on 2026-10-09"


@pytest.mark.parametrize("field", ["content_sha", "deps_sha"])
def test_a_stale_record_does_not_count(field):
    e = evidence([rec(**{field: "f" * 16})])
    assert not e["ready"] and "predates" in e["why"]
    assert e["teaching"]["stale"]


@pytest.mark.parametrize(
    "fields",
    [
        {"settings": {"MKTSTATS_QUICK": "1"}},
        {"settings": {"MKTSTATS_SABOTAGE": "2"}},
        {"mode": "learner"},
        {"scope": "partial", "scope_note": "Part A only"},
        {"env": "docker-arm64", "source": "test_notebooks"},
        {"env": "macos-arm64", "source": "test_notebooks"},
        {"env": "ci-amd64", "source": "test_notebooks"},
        {"source": "backfill"},
    ],
    ids=["quick", "sabotage", "learner", "partial", "docker", "mac", "ci", "backfill"],
)
def test_only_worked_full_whole_colab_runs_are_teaching_evidence(fields):
    e = evidence([rec(**fields)])
    assert e["teaching"] is None and not e["ready"]
    assert "no worked, full-settings run" in e["why"]


def test_the_other_languages_colab_lane_does_not_count():
    e = evidence([rec(env="colab-r")])
    assert e["teaching"] is None
    assert "colab-r" not in e["lanes"]


def test_an_old_run_expires_only_against_a_given_date():
    assert evidence([rec()], as_of=None)["ready"]
    assert evidence([rec()], as_of=DAY + dt.timedelta(days=21))["ready"]
    late = evidence([rec()], as_of=DAY + dt.timedelta(days=22))
    assert not late["ready"] and "22 days old" in late["why"]


def test_a_newer_failure_replaces_an_older_pass():
    e = evidence([rec(), rec(date="2026-10-10", status="fail", index=1)])
    assert not e["ready"] and "failed" in e["why"]


def test_on_the_same_day_a_failure_wins():
    e = evidence([rec(status="fail"), rec(index=1)])
    assert not e["ready"]


def test_a_current_record_beats_a_newer_stale_one():
    e = evidence([rec(), rec(date="2026-10-10", content_sha="f" * 16, index=1)])
    assert e["ready"]


def test_documented_only_envs_are_never_run():
    e = evidence([])
    for key, env in V["readiness"]["envs"].items():
        if env.get("documented_only"):
            assert e["lanes"][key] == {"state": "documented", "record": None}


def test_lanes_show_the_newest_worked_run_per_env():
    docker = rec(env="docker-arm64", source="test_notebooks", settings={"MKTSTATS_QUICK": "1"})
    e = evidence([docker, rec(env="docker-arm64", mode="learner", source="test_notebooks")])
    assert e["lanes"]["docker-arm64"]["record"]["mode"] == "worked"
    assert e["learner"]["mode"] == "learner"
    assert "QUICK settings" in readiness.describe(e["lanes"]["docker-arm64"]["record"])


def test_unwritten_notebooks_are_not_ready():
    unwritten = [e for e in common.notebooks(V) if e["id"] not in CURRENT]
    if unwritten:
        report = readiness.build(V, [], None)
        row = next(n for n in report["notebooks"] if n["id"] == unwritten[0]["id"])
        assert row["why"] == "not written yet"


def test_readiness_never_reads_the_clock():
    source = (common.ROOT / "scripts" / "readiness.py").read_text(encoding="utf-8")
    assert "today(" not in source and "now(" not in source


def test_build_summary():
    report = readiness.build(V, [rec(), rec(notebook=R)], DAY)
    s = report["summary"]
    assert s["ready"] == 2 and s["notebooks"] == len(common.notebooks(V))
    assert s["all_ready"] == (s["ready"] == s["notebooks"])


# ---- validation ------------------------------------------------------------------------------


def batch(**overrides):
    content, deps = CURRENT[PY]
    entry = {"notebook": PY, "scope": "notebook", "status": "pass", "seconds": 3.0}
    body = {
        "schema": 1,
        "date": "2026-10-09",
        "source": "test_notebooks",
        "env": "docker-arm64",
        "mode": "worked",
        "kernel": "python3",
        "evidence": "test",
        "content_sha": content,
        "deps_sha": deps,
        "runs": [entry],
    }
    runs = overrides.pop("runs", None)
    body.update(overrides)
    if runs is not None:
        body["runs"] = runs
    return json.dumps(body)


def problems(text):
    return run_records.check_batch("t.json", text, V, today=DAY)[1]


def test_a_valid_batch_passes():
    assert problems(batch()) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"env": "laptop"}, "not a key of readiness.envs"),
        ({"env": "aws-ec2"}, "documented only"),
        ({"kernel": "ir"}, "kernel ir but the notebook is python3"),
        ({"content_sha": None}, "needs the notebook's content_sha"),
        ({"deps_sha": "abc"}, "needs the notebook's deps_sha"),
        ({"date": "2026-13-01"}, "not a real date"),
        ({"date": "2026-10-20"}, "in the future"),
        ({"mode": "unknown"}, "mode"),
        ({"evidence": "token sk-ant-abcdefghijklmnopqrstuvwxyz"}, "credential"),
        ({"surprise": 1}, "unknown top-level field"),
        (
            {
                "runs": [
                    {
                        "notebook": "python/99-nope",
                        "scope": "notebook",
                        "status": "pass",
                        "seconds": 1.0,
                    }
                ]
            },
            "no notebook named",
        ),
        (
            {"runs": [{"notebook": PY, "scope": "partial", "status": "pass", "seconds": 1.0}]},
            "scope_note",
        ),
        ({"source": "colab", "env": "colab-py"}, "colab_release"),
    ],
)
def test_the_validator_rejects_bad_records(overrides, expected):
    assert any(expected in p for p in problems(batch(**overrides))), problems(batch(**overrides))


def test_batches_round_trip():
    records = run_records.records_of(json.loads(batch()), "t.json")
    again = run_records.records_of(run_records.batch_of(records), "t.json")
    assert again == records


def test_committed_run_records_are_valid():
    run_records.load_valid(V)


# ---- add_run_record ----------------------------------------------------------------------------


def colab_record(**fields):
    content, deps = CURRENT[fields.get("notebook", PY)]
    base = {
        "date": "2026-10-09",
        "notebook": PY,
        "kernel": "python3",
        "content_sha": content,
        "deps_sha": deps,
        "ref": "main",
        "mktstats_commit": "0123456789abcdef0123456789abcdef01234567",
        "mode": "worked",
        "settings": {},
        "status": "pass",
        "seconds": 95.2,
        "install_seconds": 31.0,
        "python": "3.13.16",
        "colab_release": "release-colab-20261001",
        "cpus": 2,
        "runtime": "colab",
        "packages": {"pandas": "2.2.3"},
        "checkpoints": {"1": {"passed": True, "whose": "the REFERENCE solution"}},
    }
    return {**base, **fields}


def paste(tmp_path: Path, record: dict) -> Path:
    path = tmp_path / "pasted.txt"
    text = (
        "----- run record (paste into scripts/add_run_record.py) -----\n"
        + json.dumps(record)
        + "\n----- end of run record -----\n"
    )
    path.write_text(text)
    return path


def test_a_colab_python_record_is_added_as_colab_py(tmp_path):
    runs = tmp_path / "runs"
    assert add_run_record.main([str(paste(tmp_path, colab_record())), "--runs", str(runs)]) == 0
    (out,) = runs.glob("*.json")
    assert out.name == "2026-10-09-colab-py-python-00-setup-warmup.json"
    data = json.loads(out.read_text())
    assert data["env"] == "colab-py" and data["source"] == "colab"
    assert data["install_seconds"] == 31.0
    assert problems(out.read_text()) == []


def test_a_colab_r_record_needs_the_ref(tmp_path):
    record = colab_record(notebook=R, kernel="ir", r="4.6.1", mktstats_commit=None, ref=None)
    record.pop("python")
    with pytest.raises(SystemExit, match="ref"):
        add_run_record.main([str(paste(tmp_path, record)), "--runs", str(tmp_path)])
    record["ref"] = "main"
    assert add_run_record.main([str(paste(tmp_path, record)), "--runs", str(tmp_path)]) == 0
    assert list(tmp_path.glob("*colab-r-r-00-setup-warmup.json"))


@pytest.mark.parametrize("missing", ["mktstats_commit", "install_seconds", "python"])
def test_a_colab_record_without_its_authenticity_fields_is_refused(tmp_path, missing):
    record = colab_record()
    record.pop(missing)
    with pytest.raises(SystemExit, match="Not added"):
        add_run_record.main([str(paste(tmp_path, record)), "--runs", str(tmp_path)])


def test_a_record_without_colab_release_needs_an_env(tmp_path):
    record = colab_record(colab_release=None)
    with pytest.raises(SystemExit, match="Not a Colab run"):
        add_run_record.main([str(paste(tmp_path, record)), "--runs", str(tmp_path)])


def test_a_learner_record_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="learner-mode"):
        add_run_record.main(
            [str(paste(tmp_path, colab_record(mode="learner"))), "--runs", str(tmp_path)]
        )


# ---- release_check -------------------------------------------------------------------------------


def test_release_check_lists_blockers_until_colab_and_ci_have_passed():
    found = release_check.blockers(V, [], DAY)
    assert any(b.startswith(f"{PY}: no worked") for b in found)
    assert any(b.startswith(f"{PY}: no CI run") for b in found)
    ci = rec(env="ci-amd64", source="test_notebooks", settings={"MKTSTATS_QUICK": "1"}, index=1)
    found = release_check.blockers(V, [rec(), ci], DAY)
    assert not any(b.startswith(f"{PY}:") for b in found)


def test_release_check_ignores_records_from_after_the_release_date():
    later = rec(date="2026-10-20")
    found = release_check.blockers(V, [later], DAY)
    assert any(b.startswith(f"{PY}: no worked") for b in found)


def test_release_check_requires_runs_younger_than_the_limit():
    found = release_check.blockers(V, [rec()], DAY + dt.timedelta(days=30))
    assert any(f"{PY}: the newest teaching run is 30 days old" in b for b in found)
