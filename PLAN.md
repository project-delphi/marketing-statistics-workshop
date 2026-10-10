# PLAN — "Statistics for Marketing" — 5-day workshop (Quarto site + Colab labs + CI)

## Context
`/Users/ravikalia/Code/github.com/marketing-statistics-workshop` is empty (not a git repo). Goal: build, test and
publish a five-day hands-on workshop on statistics for marketing (sales growth + CLV) as a Quarto site on GitHub
Pages, with labs that run on Colab (Python + R), local Linux (Docker) and documented AWS paths. It is modelled on
project-delphi/nlp-llms (local clone at `/Users/ravikalia/Code/github.com/nlp-llms`, same owner, MIT code). The overriding
rule: nothing is claimed to run unless it ran, and a generated readiness page says where, when, with which settings,
and whether that evidence is still current.

## Decisions (confirmed with user)
- Public repo `project-delphi/marketing-statistics-workshop`; site `https://project-delphi.github.io/marketing-statistics-workshop/`.
- Colab runs: I drive real Colab through the user's Chrome (Claude in Chrome). I will ask before clicking Colab's
  "notebook not authored by Google — Run anyway" consent the first time.
- AWS: docs + scripts only, labelled "documented, not run". No cloud spend.
- Docker: start Docker Desktop; build native arm64 locally, amd64 in CI.
- Day 5 overload fixed by moving budget allocation to Day 4 and renumbering (10 = allocation, 11 = uplift).

## Verified facts that shape the design (checked 2026-10-09; re-verified against docs before each lab)
- Colab: Ubuntu 24.04, **Python 3.13.16, R 4.6.1**; preinstalled pymc 5.28.5 / arviz 0.22 / numpy 2.1.3 / sklearn 1.6.1;
  no GSL. → Target Python 3.13 and R 4.6.1 everywhere.
- pymc-marketing **1.2.0** (pins pymc 6.3.x, arviz ≥1.2): `from pymc_marketing.mmm import MMM` (old class removed);
  `build_model` → `add_lift_test_measurements`; `mmm.budget_optimizer(start,end).allocate_budget(...)`; CLV models take
  data in `.fit(data=)`; `ShiftedBetaGeoModel` for sBG. PyMC 6 = numba backend; ArviZ 1.x default interval 0.89 ETI →
  always pass interval type/prob explicitly.
- econml 0.17 (cp313 wheels) ✔. causalml (no cp313 wheel) ✘. scikit-uplift (stale) ✘ → own Qini/uplift code.
  tfp-causalimpact (no release since 2023) → optional, only if it runs. Meridian 2.1 conflicts with pymc-marketing
  (arviz<0.20, tfp-nightly) and wants a GPU → briefing mention only, no lab.
- R: CLVTools 0.12.1 (needs GSL), grf 2.6.1, CausalImpact 1.4.1/bsts 0.9.11, GeoLift + augsynth GitHub-only (pure R),
  Robyn 3.12.1 dormant + needs Python nevergrad → timeboxed spike.
- P3M `https://p3m.dev/cran/__linux__/noble/<date>` serves binaries for amd64 **and arm64** (needs R HTTPUserAgent).
  `rocker/r-ver:4.6.1` = noble, multi-arch, P3M preconfigured. Quarto latest stable 1.10.19 (local is 1.6.40).
- SageMaker: only new Studio (Classic ended); BYOI needs UID 1000/GID 100, port 8888, base_url `jupyterlab/default`.
- Actions: checkout@v7, upload-pages-artifact@v5, deploy-pages@v5, configure-pages@v6, upload-artifact@v7,
  download-artifact@v8, quarto-actions/setup@v2, lychee-action@v2, build-push-action@v7, login-action@v4.

## Curriculum (final numbering)
| Day (question) | Module | Lab notebook(s) | Lang (why) |
|---|---|---|---|
| Pre-work | 0 Env check, DuckDB/pandas & dplyr warm-up, coding agents | `python/00-setup-warmup`, `r/00-setup-warmup` | both |
| 1 Who are our customers, who is still with us? | 1 Transaction logs → RFM, cohorts (CDNOW, Online Retail II) | `python/01-customer-base-sql` | Py (DuckDB SQL) |
| | 2 BG/NBD + Pareto/NBD (synthetic recovery + CDNOW; P(alive); stop-retargeting decision) | `python/02-btyd` | Py (PyMC-Marketing) |
| | 3 Covariates (CLVTools `SetStaticCovariates`) + BTYD vs gradient boosting, run not asserted | `r/03-clvtools-covariates`, `python/03-btyd-vs-ml` | R reference + Py ML |
| 2 What is a customer worth? | 4 Gamma-Gamma (independence check first), discounted CLV, horizon/rate sensitivity, sBG contractual | `python/04-monetary-clv` | Py |
| | 5 CLV → CAC caps, segment value w/ intervals, retention break-even, warehouse write-back, CFO memo | `python/05-clv-decisions` | Py |
| 3 Did marketing cause sales? | 6 Power/MDE, SRM, peeking & sequential, CUPED, delta method, attribution ≠ incrementality | `python/06-experiments` | Py |
| | 7 Synthetic control + DiD + power-by-simulation; CausalImpact (bsts); GeoLift market selection demo; placebo tests on known-lift geo panel; real: Prop 99 | `python/07-synthetic-control-did`, `r/07-causalimpact-geolift` | R reference (CausalImpact, GeoLift) |
| 4 Where should the next dollar go? | 8 Bayesian MMM: adstock, saturation, seasonality, priors from spend share, PPCs, divergences, ROAS intervals | `python/08-bayesian-mmm` | Py |
| | 9 Lift-test calibration, time-slice CV, parameter stability, what MMM can't identify; Robyn comparison | `python/09-mmm-calibration`, `r/09-robyn` (conditional on spike S5, which passed) | Py; R (Robyn) |
| | 10 Budget allocation: PyMC-Marketing optimizer + from-scratch SLSQP, constraints, risk-aware via draws, CLV-weighted objective | `python/10-budget-allocation` | Py |
| 5 Whom to target, how to spend? | 11 Uplift on Hillstrom: grf causal forest (ATE, RATE/TOC); EconML `CausalForestDML(discrete_treatment=True)` + meta-learners; Qini on held-out; target where margin > offer cost | `r/11-grf-causal-forest`, `python/11-uplift-econml` | R reference (grf) + Py |
| | 12 Capstone (11:25–17:00): CLV → geo test → calibrated MMM → allocation + uplift → 5-slide brief; rubric | `python/12-capstone` | Py |

Two-module days (2, 3) get the upper-bound lab slots plus a 60-min applied clinic tied to that day's decision (Day 2: CFO-memo peer
review; Day 3: geo-test design review using the GeoLift market-selection demo). A generated test checks every day fits 09:00–17:00.
Total: 13 Python + 5 R notebooks (R Robyn was conditional on S5; it passed in Docker and on Colab). Real data: CDNOW, UCI Online Retail II, Hillstrom, Prop 99, Fader–Hardie sBG
retention counts; simulated: PyMC-Marketing `mmm_example.csv`, Robyn `dt_simulated_weekly` (labelled as simulated).

## Repository layout (changes vs. the prompt, with reasons)
As the prompt, plus:
- `_variables.yml` — single source of truth (modules, days, timings, notebooks, deps, readiness config, package pins + verified-on
  dates, `repo.ref`). `scripts/gen_tables.py` writes `_includes/*.md`, sidebar, README regions, `notebooks.qmd` table, `VERSIONS.md`,
  counts; CI fails on drift. (Pattern from nlp-llms.)
- `labs/src/NN-*.py|.R` — jupytext percent sources, **one-way** → `labs/python/*.ipynb`, `labs/r/*.ipynb` via
  `scripts/gen_notebooks.py` (adds header + Colab badge, harness/install cell, footer, solution-cell metadata, R kernelspec `ir`,
  strips outputs). Why: reviewable diffs and lint across 18 notebooks in two languages.
- `runs/` — committed run records (local, Colab). CI records live on an orphan **`evidence`** branch (durable, reproducible;
  no bot commits on main, no 90-day artifact expiry).
- `cran/src/contrib/` — augsynth + GeoLift source tarballs built at pinned SHAs (pure R, GPL-2/MIT, licences kept), served from
  Pages as a mini repo so a class doesn't hit GitHub's anonymous API rate limit.
- `scripts/`, `tests/`, `filters/` (nlp-llms Lua filters), `images/`, `spikes/` (throwaway, results in `DECISIONS.md`).

## Shared code
- **`src/mktstats`** (pyproject at repo root; deps numpy/pandas/scipy only; **lazy imports** of pymc/arviz/econml so importing it
  never preloads old pymc): `runtime.py` (detect Colab/SageMaker/local; install-missing with pins; refuse to continue and say
  "Restart session, then Run all" if pymc/arviz were already imported at old versions), `harness.py` (port of nlp-llms
  `scripts/harness.py`: `@workshop.solution(N)`, `checkpoint(N)`, `use_reference(N)`, `WORKED_EXAMPLE`, per-cell timing,
  `run_record()` incl. `COLAB_RELEASE_TAG`, Python version, key package versions, mktstats commit from `direct_url.json`,
  install seconds), `data.py` (cached loaders with sha256 + fallback URLs; parquet mirrors as GitHub Release assets only where the
  licence allows — e.g. UCI CC BY 4.0; Hillstrom from minethatdata.com + fallback, not mirrored), `synth/` (seeded generators:
  BTYD customers with true r, α, a, b and per-customer true alive status + acquisition-channel effect; Gamma-Gamma spend; weekly
  MMM with known adstock/saturation and ROAS defined as noiseless incremental sales ÷ spend over a stated window; geo panel with
  known lift; randomized email experiment with true per-row CATE), `checks.py` (named-interval checks, e.g. "true ROAS inside the
  94% HDI for ≥ k of K channels"), `uplift.py` (Qini/uplift curves, AUUC).
- **`R/mktstats.R`** sourced from raw GitHub at `repo.ref`: harness without Colab forms (`WORKED_EXAMPLE <- FALSE`, env override;
  `solution(n, name, fn)`, `checkpoint(n)`, `run_record()` printed), loaders for committed synthetic CSV + `truth.json`, checks.
- **Synthetic data committed** in `data/synthetic/` + `truth.json`, generated with pinned numpy and drift-gated; R and Python read the
  same files. Live generation only where learners vary parameters.

## Notebook contract (enforced by `tests/test_notebooks_structure.py`)
Header (badge at `repo.ref`, goals, time) → install/runtime cell (pins from `_variables.yml`) → `QUICK`/`WORKED_EXAMPLE` switches
(env `MKTSTATS_QUICK`, `MKTSTATS_WORKED`) → parts with `## Exercise N · title (M minutes)`: **Predict** → stub function raising
`NotImplementedError` → folded solution (`tags=["solution"]`) → **Explain** + `<details>` why → checkpoint. First checkpoint comes
before any model fit. Fit once per lab; checkpoints test pure functions or the single fit. Ends with a decision cell, stretch section,
run record, footer. Targets on Colab FULL: install ≤ 3 min, any cell ≤ 4 min, total compute ≤ 15 min (measured, reported).

## Environments
- `environment/Dockerfile`: `rocker/r-ver:4.6.1` + uv Python 3.13 venv (`requirements.txt` compiled by `uv pip compile` from
  `requirements.in`) + IRkernel system-wide + `renv.lock` restore from a dated P3M snapshot + augsynth/GeoLift from `cran/` + Quarto
  1.10.19 + JupyterLab 4; user UID 1000/GID 100, port 8888 (SageMaker BYOI-compatible). `.devcontainer/` uses it.
- Colab R notebooks set repos to the same dated P3M snapshot with HTTPUserAgent; GSL via `apt-get install libgsl27` (or r2u if
  enabled — spike decides). Measured install times go on `setup.qmd` from run records.
- Local: `uv` for Python; R and Quarto 1.10.19 via the Docker image (local R 4.0.5 / Quarto 1.6.40 are not used).
- `environment/aws/`: SageMaker BYOI (`build-push-ecr.sh` linux/amd64, `create-image.sh`), lifecycle-config alternative, EC2
  `user-data.sh` + `launch.sh` (JupyterLab from the GHCR image over an SSH tunnel), least-privilege IAM JSON, S3 bucket option,
  `teardown.sh`, cost warning + links to the SageMaker and EC2 pricing pages (no prices). Size justified from measured CI runner times,
  stated as such. shellcheck/hadolint in CI.

## CI / publishing
- `image.yml`: build amd64 image on `environment/**` changes → GHCR, tag = hash of env files.
- `notebooks.yml` (push/PR, weekly cron, dispatch): setup job emits a matrix from `_variables.yml`; one job per notebook in the image
  container; `scripts/test_notebooks.py` (nbclient, kernel from metadata — **replaces papermill** because verify mode must inject
  cells and settings are env vars) runs worked + verify-checkpoints (each checkpoint must fail on its stub) in one pass, then learner
  mode; uploads executed notebooks; on main a `collect` job appends run records to the `evidence` branch.
- `publish.yml` (push to main, `workflow_run` of notebooks, dispatch): drift gates (gen_tables, gen_notebooks, synthetic data), ruff,
  pytest (unit, checks-can-fail, readiness rules, quick recovery), checkout `evidence` into `runs/ci/`, gen readiness, `quarto render`
  (`freeze: auto`, figures mostly script-made), lychee `--offline` (blocking), upload-pages-artifact → deploy-pages.
- `links.yml`: monthly lychee over every page of the live site (from its sitemap, plus 404.html), external links included,
  and every dataset source, downloaded and checked against its sha256; non-blocking (separate from deploy), job summaries
  list failures. `release.yml`: monthly and on demand (with a class date), `release_check.py` on main
  with the `evidence` records (see DECISIONS.md, 2026-10-10: why main and not a tag push).
- Readiness: per notebook, evidence lanes CI / local Docker / Colab / AWS ("not run"). Staleness = notebook code-cell hash **plus** a
  hash of the module's declared deps (mktstats files, R helpers, synthetic data, pins) and the install ref. Ready to teach = Colab,
  worked, FULL, whole notebook, current hashes, ≤ 21 days, no fallback markers, and CI passing on current commit.

## Change since approval
- GeoLift and augsynth install from pinned GitHub **archive** URLs (`github.com/<repo>/archive/<sha>.tar.gz`,
  no GitHub API call) after their CRAN dependencies, instead of a self-hosted `cran/` repo. Simpler; the
  spike S3 measures it on Colab. If archive downloads prove unreliable in class, fall back to `cran/`.

## Task list
Owner in brackets: [AD] Academic Director, [PE] Pedagogy Expert, [UI] UI Expert, [TE] Technical Expert,
[Lead] the coordinating session (spikes on Colab, merges, final report).

### Phase 0 — skeleton and spikes
- [x] [Lead] git init, licences, pyproject, DECISIONS.md, public repo pushed
- [x] [Lead] S4 Docker arm64 build (3.0 GB, ~16 min cold; full R dependency check)
- [x] [Lead] S1 Chrome-driven Colab loop: open from GitHub, fresh runtime, Run all, read record
- [x] [Lead] S2 Colab Python: install time, BG/NBD, Pareto/NBD, Gamma-Gamma, MMM, causal forest timings
- [x] [Lead] S3 Colab R: repos, binary installs, GSL, GeoLift from archives, hidden-cell behaviour
- [x] [Lead] S5 Robyn (1-hour timebox) → lab, demo or comparison table only

### Phase 1 — walking skeleton
- [x] [TE] `src/mktstats`: runtime (detect Colab/SageMaker/local, install with constraints, restart guard), harness (solution/checkpoint/use_reference/run_record, verify hooks), lazy imports
- [x] [TE] `R/mktstats.R`: harness without forms, loaders, checks, run record
- [x] [TE] `scripts/gen_notebooks.py` (jupytext percent → ipynb; generated header/install/harness/record/footer cells; solution metadata; R kernelspec; strip outputs; idempotent)
- [x] [TE] `scripts/test_notebooks.py` (nbclient; kernel from metadata; worked + verify-checkpoints in one pass; learner mode; per-cell timing; `--record`)
- [x] [TE] `scripts/run_records.py`, `readiness.py`, `add_run_record.py`, `release_check.py` (content_sha + deps_sha staleness; Colab authenticity fields)
- [x] [TE] `scripts/gen_tables.py` (includes: facts, days, path, module headers, lab steps, schedule, notebooks table, readiness, VERSIONS.md, README status region)
- [x] [TE] tests: structure of notebooks, variables, runs/readiness rules, checks-can-fail
- [x] [UI] `_quarto.yml`, theme tokens (light/dark), filters, page shells for every navbar/footer page, landing layout
- [x] [Lead] `.github/workflows/publish.yml` (drift gates, ruff, pytest, evidence checkout, render, lychee offline, deploy-pages) + Pages enabled
- [x] [Lead] `.github/workflows/notebooks.yml` (matrix from `_variables.yml`, image container, collect → `evidence` branch) + `image.yml`
- [x] [TE] M0 Python + R notebooks (written; worked+verify and learner runs recorded on macOS and in Docker, full settings)
- [x] [Lead] M0 Colab runs recorded. Milestone: live site, honest readiness for 2 notebooks (2026-10-09; CI evidence branch live)

### Phase 2 — generators (freeze mktstats API at the end)
- [x] [TE] synth: BTYD retailer (true r, α, a, b, alive status, acquisition channel effect), Gamma-Gamma spend, weekly MMM (adstock, saturation, true ROAS), geo panel (true lift), email experiment (true CATE); `truth.json`; drift gate
- [x] [TE] recovery tests (MAP fast in CI; seeded multi-seed sweep is `pytest -m slow tests/test_synth_files.py`, to be scheduled weekly in CI)
- [x] [TE] loaders: CDNOW, Online Retail II, Hillstrom, mmm_example, Prop 99, sBG counts (sha256 + fallback)

### Phase 3–5 — labs (each: brief [AD] → review [PE] → build + run [TE] → Colab [Lead] → module page [AD])
- [x] M8 Bayesian MMM (built + Docker run; Colab run recorded 2026-10-10)
- [x] M2 BTYD — built + Docker run; Colab run recorded 2026-10-10
- [x] M11 uplift (EconML + grf): built + Docker run + Colab worked/full runs recorded for both notebooks (2026-10-09, again 2026-10-10); module page written
- [x] M7 geo (Python SC/DiD + R CausalImpact/GeoLift) — [TE] built + Docker run (runs/2026-10-09-day3-*.json); Colab run recorded 2026-10-10
- [x] M1 customer base (DuckDB) — built + Docker run; Colab run recorded 2026-10-10
- [x] M4 monetary value and CLV — [TE] built + Docker run (`runs/2026-10-09-day2-*.json`); Colab run recorded 2026-10-10
- [x] M6 experiments — [TE] built + Docker run (runs/2026-10-09-day3-*.json); Colab run recorded 2026-10-10
- [x] M3 covariates (CLVTools) + BTYD vs ML — built + Docker run (R and Python); Colab run recorded 2026-10-10
- [x] M5 CLV decisions — [TE] built + Docker run (`runs/2026-10-09-day2-*.json`); Colab run recorded 2026-10-10
- [x] M9 calibration and validation (+ Robyn per S5) — built + Docker run (runs/2026-10-09-m09-*.json); Colab run recorded 2026-10-10
- [x] M10 budget allocation (built + Docker run; Colab run recorded 2026-10-10)
- [x] M12 capstone + rubric [PE] (rubric, task, deliverables and timing done: `briefs/12-capstone-rubric.md`); [TE] lab built + Docker run (runs/2026-10-09-m12-*.json); Colab run recorded 2026-10-10

### Phase 6 — site and docs
- [x] [AD] lab briefs for all modules (`briefs/`), objectives/decisions in `_variables.yml`
- [x] [AD] `references.qmd` (every entry fetched; Verified + date + claim supported)
- [x] [PE] `prepare.qmd` entry check; `teach.qmd`; day warm-ups/wrap-ups; knowledge checks (also `days/_day-0-notes.md`…`_day-5-notes.md` with the Day 2/3 clinics and Day 5 run of show, and the lab standard `briefs/_lab-standard.md`)
- [x] [UI] landing page (hero, counts, day cards, path, integration diagram SVG), slides/welcome.qmd, browser check both themes (headless Chrome, 1280 and 375 px, against the generator's includes; the lead's check of the live site in Chrome, 2026-10-10: Days menu, theme toggle, no sideways scroll on 29 of 30 pages at 375 px, fixed with the equation-number overlap and KaTeX pin, DECISIONS.md)
- [x] [Lead] `setup.qmd` (Colab Python/R, local Docker/uv, AWS) with measured install times from records
- [x] [TE] `environment/aws/` (SageMaker BYOI, lifecycle config, EC2, IAM, S3, teardown) — documented, not run
- [x] [AD] `faq.qmd`

### Phase 7 — release
- [x] [Lead] `repo.ref` → tag; full Colab sweep; `release_check.py` passes; final report (`v2026.10.1`, which also pins the Python labs' data; all 18 notebooks swept on Colab 2026-10-10; `release_check.py` passes through `--as-of 2026-10-31`)
