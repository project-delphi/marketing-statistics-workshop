# Run records

One JSON file per batch of notebook runs: what ran, where, in which mode and with which
settings, whether it passed and how long it took. They are the only evidence the site, the
facilitator pages and the release check use for "this lab has run". Never state a run time
or a "has run on" claim by hand: add a record and rerun `python scripts/gen_tables.py`.

- `runs/*.json`: committed records from laptops, the Docker image and Colab.
- `runs/ci/**/*.json`: CI records. CI keeps them on the orphan `evidence` branch and checks
  that branch out into `runs/ci/` before generating the site; they are not committed on main.

`scripts/run_records.py` validates every file; `scripts/readiness.py` turns them into
evidence; `scripts/release_check.py` gates a ready-to-teach release.

## How records are made

| Where | How | `source` |
|---|---|---|
| Laptop, Docker image, CI | `python scripts/test_notebooks.py --notebooks ID --record runs/FILE.json --env ENV` (add `--mode both` for worked and learner runs, `--full` for full settings) | `test_notebooks` |
| Colab | Run all with `WORKED_EXAMPLE` ticked; copy the text between the run-record markers the last code cell prints (or `run_record.json`); `pbpaste \| python scripts/add_run_record.py` | `colab` |
| Elsewhere by hand (local Jupyter) | the same, with `--env` | `notebook` |
| An earlier report | copied by hand; never counts as teaching evidence | `backfill` |

A test run writes to the file named by `--record`; if it exists, the new entries are added.

## Staleness

A record names the code it ran with two hashes, both embedded in the notebook's harness cell
by `scripts/gen_notebooks.py`:

- `content_sha`: the notebook's own code cells (the five generated cells are excluded).
- `deps_sha`: the files listed in `modules.mNN.deps` (R files count only for R notebooks,
  `src/` only for Python notebooks, anything else for both), the notebook's pins (Python:
  the `install` list resolved against `packages` and `environment/requirements.txt`; R: the
  CRAN and GitHub packages and the P3M snapshot) and `repo.ref`.

A record whose hashes differ from the committed notebook's current ones is **stale**: it is
shown, marked "before the notebook or its deps last changed", and counts for nothing.

## What counts as teaching evidence

The newest record (a newer failure replaces an older pass; on the same date a failure wins)
that is all of:

- on the env with `teaching: true` for the notebook's kernel (`colab-py` for Python,
  `colab-r` for R);
- `mode: worked` (reference solutions bound), `scope: notebook` (Run all, top to bottom);
- full settings: `settings` has neither `MKTSTATS_QUICK` nor `MKTSTATS_SABOTAGE`;
- not a backfill; current `content_sha` and `deps_sha`;
- at most `readiness.max_run_age_days` old. The readiness pages judge age against the newest
  record in the repository (so they never change with the calendar); the release check
  judges it against the release date.

Envs with `documented_only: true` (the AWS paths) cannot have records and are always shown
as "documented, not run". Laptop, Docker and CI times are shown on their own lanes and are
never labelled as Colab times.

A Colab record must carry `colab_release` (Colab's `COLAB_RELEASE_TAG`), `install_seconds`,
the `python` or `r` version, and `mktstats_commit` (Python: mktstats was installed from git
at that commit) or `ref` (R: the helpers were sourced at that ref). `add_run_record.py`
refuses a record without them, and a learner-mode record.

## File format

```json
{
  "schema": 1,
  "date": "2026-10-09",
  "source": "test_notebooks",
  "env": "docker-arm64",
  "env_detail": "Linux-6.10-aarch64, runner Python 3.13.14",
  "kernel": "ir",
  "settings": {"MKTSTATS_QUICK": "1"},
  "commit": "1489d2e",
  "ref": "main",
  "evidence": "scripts/test_notebooks.py --record",
  "runs": [
    {"notebook": "r/00-setup-warmup", "mode": "worked", "scope": "notebook", "status": "pass",
     "seconds": 2.1, "content_sha": "…", "deps_sha": "…", "verified": 3}
  ]
}
```

Fields at the top apply to every entry in `runs`; an entry may override any of them. Only
shared fields may sit at the top; `notebook`, `scope`, `status` and `seconds` are per entry.

| Field | Values |
|---|---|
| `schema` | `1` |
| `date` | `YYYY-MM-DD`, the recorder's date |
| `source` | `test_notebooks`, `colab`, `notebook` or `backfill` (above) |
| `env` | a key of `readiness.envs` in `_variables.yml`, not a `documented_only` one |
| `env_detail` | free text: OS, machine, runner Python |
| `notebook` | the notebook id: `python/NN-slug` or `r/NN-slug` (its path under `labs/` without `.ipynb`) |
| `kernel` | `python3` or `ir`; must match the notebook |
| `mode` | `worked` (solutions bound) or `learner` (nothing written: the run must stop at the first checkpoint with the recovery message; `status: pass` means it did) |
| `scope` | `notebook` (Run all) or `partial` (say what in `scope_note`) |
| `status` | `pass` or `fail` |
| `seconds` | wall time of the scope, or `null` |
| `settings` | the workshop's environment settings that were on, as strings: `MKTSTATS_QUICK`, `MKTSTATS_SABOTAGE` |
| `content_sha`, `deps_sha` | 16 hex characters (above); required unless `source: backfill` |
| `commit` | the repository commit the runner was at |
| `ref` | `repo.ref` when the record was made (R records from Colab: the ref the helpers came from) |
| `mktstats_commit` | the commit mktstats was installed from (Python, from pip's `direct_url.json`); `null` when it was imported from a checkout |
| `install_seconds` | the install cell's own time; on a laptop, the image or CI it installs nothing |
| `python`, `r`, `runtime`, `platform`, `cpus`, `memory_gb`, `colab_release`, `packages` | from the notebook's own run record |
| `checkpoints` | `{label: {passed, whose}}` from the notebook's run record (`whose`: `your code`, `the REFERENCE solution`, or `null` for provided code) |
| `verified` | worked runs: how many checkpoints were also run on their stubs and failed, as they must |
| `phases`, `slowest_cell`, `cell_seconds`, `cell_errors` | timing detail: seconds by cell tag, the slowest cell, per-cell seconds (Python) |
| `note` | why a run failed, or anything a reader needs |

A record never holds code, outputs, keys or anyone's identity; the validator rejects text
that looks like a credential.
