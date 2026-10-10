# Statistics for Marketing

[![publish](https://github.com/project-delphi/marketing-statistics-workshop/actions/workflows/publish.yml/badge.svg)](https://github.com/project-delphi/marketing-statistics-workshop/actions/workflows/publish.yml)
[![notebooks](https://github.com/project-delphi/marketing-statistics-workshop/actions/workflows/notebooks.yml/badge.svg)](https://github.com/project-delphi/marketing-statistics-workshop/actions/workflows/notebooks.yml)

A five-day hands-on workshop on statistics for sales growth and customer lifetime value (CLV): who
your customers are and who is still active, what each one is worth, whether the marketing caused
the sales, where the next dollar should go, and whom to target. Labs in Python and R open in Google
Colab with one click; the same notebooks run locally in Docker and on AWS.

**Site:** https://project-delphi.github.io/marketing-statistics-workshop/

Every method is first fitted to seeded synthetic data with a known answer (so a lab can check
whether the method recovers the truth), then to a real dataset. Every module ends in a business
decision. Nothing on the site claims a notebook runs unless a run record says it ran: see the
[readiness page](https://project-delphi.github.io/marketing-statistics-workshop/readiness.html).

## Status

<!-- BEGIN status -->
> **As of 2026-10-10: not yet ready to teach.** 4 of 18 notebooks have a current, passing, worked, full-settings run of the whole notebook on Colab. 2 have passed on another machine (2 with full settings), and 0 on CI. [What has run, where](https://project-delphi.github.io/marketing-statistics-workshop/readiness.html).
<!-- END status -->

This summary counts the run records committed in `runs/`. CI records live on the `evidence` branch,
so the [live readiness page](https://project-delphi.github.io/marketing-statistics-workshop/readiness.html)
also counts CI runs.

## Run it

- **Colab:** open any lab from the [Notebooks](https://project-delphi.github.io/marketing-statistics-workshop/notebooks.html) page.
- **Docker** (R 4.6.1, Python 3.13, JupyterLab, Quarto 1.10.19):
  ```bash
  docker run --rm -p 127.0.0.1:8888:8888 \
    -e MKTSTATS_REPO_ROOT=/home/sagemaker-user/work -e PYTHONPATH=/home/sagemaker-user/work/src \
    -v "$PWD":/home/sagemaker-user/work ghcr.io/project-delphi/marketing-statistics-workshop/env:latest
  ```
  or build it yourself from [`environment/Dockerfile`](environment/Dockerfile).
- **Python only, with uv:** `uv venv --python 3.13 && uv pip install -r environment/requirements.txt && uv pip install -e . --no-deps`.
- **AWS:** [`environment/aws/`](environment/aws/) (documented and scripted, not run by the authors).

Details: [Setup](https://project-delphi.github.io/marketing-statistics-workshop/setup.html).

## How the repository fits together

| Path | What it is |
|---|---|
| `_variables.yml` | Single source of truth: days, modules, timings, objectives, decisions, notebooks, pins, readiness rules |
| `labs/src/` | Lab sources (jupytext percent format); `labs/python/`, `labs/r/` are generated from them |
| `src/mktstats/` | Python package the labs install: runtime checks, exercise harness, checks, loaders, synthetic generators, uplift metrics |
| `R/mktstats.R` | The R side of the harness and checks |
| `data/` | Committed synthetic data with `truth.json`, small real datasets with their licences ([data/README.md](data/README.md)) |
| `scripts/` | Generators (`gen_notebooks.py`, `gen_tables.py`, `make_synthetic.py`), the notebook runner, run records and readiness |
| `runs/` | Run records from local and Colab runs; CI records are on the `evidence` branch |
| `modules/`, `*.qmd`, `days/`, `slides/` | The Quarto site; `_includes/` is generated |
| `briefs/` | Lab briefs and the lab standard (author-facing) |
| `environment/` | Pinned requirements (compiled against Colab's own package list), R packages, Dockerfile, AWS |
| `.github/workflows/` | `notebooks.yml` (runs every notebook), `publish.yml` (tests, render, link check, deploy), `image.yml` |

Contributors and agents: [AGENTS.md](AGENTS.md) (roles and conventions), [CONTRIBUTING.md](CONTRIBUTING.md)
(the contracts between parts), [PLAN.md](PLAN.md) (plan and task list), [DECISIONS.md](DECISIONS.md)
(decisions with evidence).

After editing `_variables.yml` or a lab source, regenerate and check:

```bash
export PYTHONPATH=src
python scripts/gen_notebooks.py && python scripts/gen_tables.py
python scripts/test_notebooks.py --notebooks python/00-setup-warmup --verify-checkpoints
pytest -q
```

## Licences

Code (the `mktstats` package, R helpers, scripts, workflows, notebook code cells): [MIT](LICENSE).
Teaching content (pages, slides, figures, notebook text): [CC BY 4.0](LICENSE-content). Datasets keep
their own licences ([data/README.md](data/README.md)). Site and notebook infrastructure adapted from
[project-delphi/nlp-llms](https://github.com/project-delphi/nlp-llms) (MIT).
