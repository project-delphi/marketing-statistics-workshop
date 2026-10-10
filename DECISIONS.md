# Decisions and spike results

Each entry: what was decided, why, and the evidence (a command, a run, a URL with the date it was read).
Newest first within each section.

## Environment

### Python pins follow Colab's own package list (2026-10-09)
- `environment/requirements.txt` is compiled by `uv pip compile` from `environment/requirements.in` with
  `environment/colab-constraints.txt` (Colab's published `pip-freeze.txt`,
  https://github.com/googlecolab/backend-info, read 2026-10-09: Ubuntu 24.04.5, Python 3.13.16, R 4.6.1)
  as constraints. Only the packages pymc-marketing 1.2.0 needs newer are freed: pymc, pytensor, arviz,
  numba, llvmlite. Everything else (numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.16.3, …) stays
  at Colab's version, so CI and Docker run what a learner gets on Colab.
- Notebooks install with `pip install <pkgs> -c <raw URL of environment/requirements.txt at repo.ref>`,
  so Colab resolves to exactly the tested versions. Without constraints, a plain resolve picks pandas 3.0
  and scikit-learn 1.9, which Colab does not have.
- No uv.lock: one pin file is easier to explain than two.

### Library choices after checking current status (2026-10-09)
- pymc-marketing 1.2.0 (PyPI 2026-09-29): MMM is `pymc_marketing.mmm.MMM`; CLV models take data in `fit()`.
- econml 0.17.0 has cp313 wheels; causalml 0.17.0 does not (would compile on Colab) → not used.
- scikit-uplift last released 2022 → not used; Qini/uplift curves are implemented in `mktstats.uplift`.
- tfp-causalimpact last released 2023 → optional only, if it runs.
- google-meridian 2.1.0 pins arviz<0.20 and tfp-nightly, conflicting with pymc-marketing (arviz>=1.2),
  and recommends a GPU → mentioned in Module 9's briefing, no lab.
- Robyn 3.12.1: last commit 2025-06, needs Python nevergrad through reticulate → spike S5 decides.
- GeoLift and augsynth are GitHub-only and pure R → built at pinned commits into `cran/` and served
  from the site, so a class does not hit GitHub's anonymous API limit.

## Open issues found while building labs

### GeoLift() fails with two or more test markets at the pinned commits (2026-10-09, Module 7 build)
- `GeoLift()` errors inside augsynth ("Tibble columns must have compatible sizes") when the test group has
  two or more markets, reproduced on GeoLift's own chicago + portland example; one market works.
  `GeoLiftMarketSelection` and `GeoLiftPower` are not affected, so the Day 3 clinic demo runs.
- Consequence: multi-market post-test analysis with `GeoLift()` is not used in the labs; the Python
  synthetic control and R CausalImpact cover the analysis. Options to fix later: try another augsynth
  commit, or aggregate treated markets before calling `GeoLift()`. Not yet investigated.

### BLAS threads oversubscribe a CPU-limited container (2026-10-09, Module 7 build)
- In Docker with `--cpus 2` on a 10-core host, OpenBLAS starts 10 threads and SciPy's SLSQP slowed ~75×.
  Module 7's Python lab now calls `threadpool_limits(1, "blas")` (threadpoolctl 3.7.0, pinned).
- Colab and the GitHub runner expose only their own cores to the process, so they are not expected to be
  affected (unverified on Colab). Run records also report the host's core count (`os.cpu_count()`)
  under `--cpus 2`; their notes say so.

### Colab allows only a few concurrent sessions (2026-10-09)
- Opening a fourth notebook while three runtimes were alive showed "Too many sessions" on this account.
  Runtime > Manage sessions lists them; ending finished ones frees a slot. The Colab sweep runs at most
  three notebooks at a time and ends each session after its record is saved. Learners should close
  finished labs' runtimes (Runtime > Disconnect and delete runtime) before opening the next lab.

### `#@title` folds solution cells in Colab's R runtime too (2026-10-09, Module 0 R on Colab)
- The generated R solution cells start with `#@title Solution N — try it yourself first {display-mode: "form"}`;
  in the R runtime they render collapsed with "Show code", like Python. (The S3 spike cell without a
  `#@title` line was shown in full; that question is now closed.)

### Online Retail II is mirrored as a release asset (2026-10-09)
- The UCI download stalled for two agents during the build (one parse after download took ~196 s in
  total). UCI's licence is CC BY 4.0, so the original zip and the parsed parquet are attached to release
  `data-2026-10-09` with attribution. `load_online_retail_ii()` now tries the 7 MB parquet first
  (sha256-checked, cold load measured 11.6 s on the laptop), then UCI, then the zip mirror.
  CDNOW and Hillstrom have no explicit licence and are not mirrored.

### Module 9 build findings (2026-10-09)
- **Robyn 3.12.1 traps:** `robyn_run(quiet = TRUE)` fails ("object 'pb' not found"), so the lab hides
  progress with `capture.output`; Robyn seeds Nevergrad only when `seed` is an integer (`123L`; with `123`
  two identical runs differed, with `123L` two FULL runs matched exactly).
- **pymc-marketing 1.2.0 time-slice CV:** `summary.predictions` reads `X["date"]` literally (the lab renames
  `date_week` to `date`) and needs `original_scale_vars=["y"]`; CV forecasts are sampled without a seed.
- **Divergences on the confounded data:** `target_accept` 0.9 and 0.95 left a few divergences; 0.98 is clean
  on the laptop and in Docker for the two main fits (about a third slower). CV folds run clean at 0.95.
- **Cross-validation checks prediction, not cause:** the uncalibrated model forecast held-out weeks as well
  as the calibrated one while its search ROAS stayed wrong (5.2–5.5 vs 3.0) — the lab's main lesson.

### PyMC-Marketing's budget optimizer: tolerance in dollars, not 1e-9 (2026-10-10, Module 10)
- `BudgetOptimizer.allocate_budget` defaults to SLSQP with absolute `ftol=1e-9` on an objective of about
  $2.6M; that is below the rounding error of the sum, so SLSQP's line search fails ("Positive directional
  derivative") depending on the posterior draws: 5 of 20 seeds on arm64 and 5 of 20 on amd64. It failed on
  Colab and in CI, passed on the laptop by chance. With `ftol=0.01` (one cent) and a feasible start,
  0 of 40 failed and plans agreed within $1. The lab passes these options explicitly and explains why.

### Notebooks pinned to a release tag, now `v2026.10.1` (2026-10-10)
- `repo.ref` is a release tag, so every Colab badge, the mktstats install, the constraints file,
  the R helpers and the committed data read that tag. A push to main no longer changes what learners get. The alternative,
  freezing main for the class, would have blocked fixes and run records.
- The ref is part of every notebook's deps_sha, so the switch made all 18 Colab records stale and
  needed a full Colab sweep at the tag. A fix release does the same.
- `v2026.10.0` pinned the code but not the data: the Python install cell set `MKTSTATS_REF` as a
  Python variable without exporting it, so `mktstats.data` fetched `data/synthetic` and
  `truth.json` from main. `v2026.10.1` exports it (R already passed it as `mktstats.ref`). The data
  had not changed between the two, so the `v2026.10.0` runs used the same files.
- Run records are added on main after the tag (`repo.branch`), so links to `runs/` point at main.
  Readiness hashes main's files, not the tag's: if main's labs or deps move past the tag, the
  readiness page reports stale even though learners still get the tag. That errs on the safe side.

### Release check and external link check run monthly (2026-10-10)
- `release.yml` runs `release_check.py` on main with the `evidence` records, monthly and on demand
  (with a date, e.g. the class's Day 1). Not on a tag push: a tag is cut first and its Colab records
  are committed to main afterwards, so at the tag's own commit the check always fails. Monthly
  (the user's choice) rather than weekly: Colab runs count for only `readiness.max_run_age_days`,
  so the scheduled run is a reminder that is red in months without a recent sweep, not an early
  warning. Checking ahead for a class is the on-demand run with the class date.
- `links.yml` (monthly) replaces the weekly `links-external` job in `publish.yml`. It checks the
  live site (pages from its `sitemap.xml`) instead of rebuilding it, so a link check no longer
  renders and redeploys the site; the weekly redeploy still happens through the notebooks cron → publish.
- First run, by hand with lychee 0.24.2 (the action's default) on 2026-10-10: 29 pages, 2,017
  links, 57 errors. 49 were 403s from journal publishers behind doi.org (they refuse non-browser
  clients; doi.org itself redirects, and answers 404 for a wrong DOI), 6 were online.stat.psu.edu
  (incomplete certificate chain; the pages load with curl), 2 were transient 503s from github.com.
  The workflow accepts 403 and excludes online.stat.psu.edu.

## Spikes

(Results are added below as each spike runs.)

### S1 · Driving Colab from Chrome works (2026-10-09)
- Opening `colab.research.google.com/github/project-delphi/marketing-statistics-workshop/blob/main/<path>`,
  Runtime → Run all, and the one-time "not authored by Google — Run anyway" consent (approved by the user
  for this repo's notebooks in this session) runs a notebook on a fresh VM.
- Plain-text cell output is readable from the page DOM. Output that contains rich or JavaScript content
  (progress bars, `google.colab.files.download`) moves into a sandboxed iframe and is not readable.
  → The record cell prints plain text only (no `files.download`); labs pass `progressbar=False` where
  the API allows. `files.download` does work and lands in ~/Downloads without a prompt (fallback).
- Colab runtime seen: `release-colab-external-images_20261007-060045_RC00`, 2 CPUs, 13.6 GB RAM,
  Python 3.13.16, R 4.6.1, Ubuntu 24.04.5.

### S2 · Colab Python: install and model timings (2026-10-09, `spikes/results/s2-colab-python-2026-10-09.json`)
| Step (fresh runtime, 2 CPUs) | Colab s | laptop s (M-series, 10 cores) |
|---|---|---|
| pip install pymc-marketing 1.2.0 + econml 0.17.0 + nutpie with constraints | 31.1 | — |
| import pymc_marketing | 16.8 | 5.2 |
| BG/NBD MAP on CDNOW (2,357 customers) | 30.3 | 14.9 |
| BG/NBD MCMC 2×500 (first, incl. compile) / again with nutpie | 25.5 / 13.7 | 8.5 / 3.1 |
| Pareto/NBD MAP / MCMC 2×300 | 30.6 / **231.0** | 9.6 / 56.9 |
| Gamma-Gamma MCMC 2×500 + discounted CLV | 12.7 + 9.8 | 3.4 + 3.4 |
| MMM (mmm_example, 2 channels) 2×500 first (incl. numba compile) / again | 45.0 / 9.7 | 11.9 / 2.6 |
| EconML CausalForestDML, 20k rows, 200 trees | 19.1 | 17.4 |
- Installed versions on Colab equal the pins exactly (pymc 6.3.2, pytensor 3.3.3, arviz 1.3.0, numba
  0.62.1, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1): the constraints approach works.
- Decisions: Pareto/NBD uses MAP in labs (MCMC only as an optional stretch, with a warning). BG/NBD,
  Gamma-Gamma and MMM use MCMC with nutpie; budget ~35–45 s of first-compile per model in a fresh runtime.

### S3 · Colab R runtime (2026-10-09, `spikes/results/s3-colab-r-2026-10-09.json`)
- Colab's R starts with `repos = https://cran.rstudio.com` (source packages) and no bspm; GSL is absent.
  Setting the dated P3M snapshot + R HTTPUserAgent gives binaries.
- Measured: apt libgsl27 9.3 s; CLVTools 5.3 s; grf 4.4 s; CausalImpact (+bsts, Boom, BoomSpikeSlab)
  45.5 s; GeoLift CRAN deps 15.1 s; augsynth + GeoLift from pinned GitHub archives 23.9 s
  (≈104 s of installs in total). CLVTools Pareto/NBD with static covariates on apparelTrans 4.6 s;
  grf causal forest 20k rows 29 s; CausalImpact example 1.1 s. No errors. Versions equal the Docker image.
- A cell carrying `cellView: form` + `jupyter.source_hidden` metadata but no `#@title` line is shown in
  full in the R runtime. Whether a `#@title` line folds R cells is not yet known → checked on the first
  generated R lab; if it does not, R solutions are visible under a "Solution — try it yourself first" heading.

### S5 · Robyn (2026-10-09, `spikes/s5_robyn.R`, `spikes/results/s5-robyn-docker-arm64-2026-10-09.json`)
- The official demo flow (`robyn_inputs` → hyperparameters via `robyn_inputs(InputCollect=, hyperparameters=)`
  → `robyn_run` → `robyn_outputs` → `robyn_allocator`) runs on `dt_simulated_weekly` in the workshop
  image limited to 2 CPUs: Robyn 3.12.1 binary install 41 s, nevergrad 1.0.12 via uv into the image's
  Python 1 s, 500 iterations × 1 trial 36 s, outputs 4 s, allocator 1 s (89 s total).
- 500×1 is far below the demo's recommended 2000×5, so the lab presents Robyn as a method comparison
  on Robyn's own simulated data with that caveat stated, not as a production fit.
- Decision: keep `r/09-robyn` as a short paired notebook. Colab R still has to be proven (reticulate must
  find a Python with nevergrad inside Colab's R runtime); until a Colab run is recorded it is not
  "ready to teach". Robyn is effectively unmaintained (last commit 2025-06), which the briefing says.
- Proven on Colab (2026-10-10): `r/09-robyn` passed worked, FULL, whole-notebook runs at `main`,
  `v2026.10.0` and `v2026.10.1` (`runs/2026-10-10-colab-r-r-09-robyn.json`; at `v2026.10.1`
  install 65.5 s, 172 s in all).

### S4 · Docker image (2026-10-09)
- `rocker/r-ver:4.6.1` + uv Python 3.13 + pinned requirements + R packages from P3M 2026-10-01 + Quarto
  1.10.19: linux/arm64 build 15 m 50 s cold on an Apple Silicon laptop, 3.0 GB. A first build silently
  missed `Boom` (download timeout, warning only); the Dockerfile now raises R's timeout and fails if any
  package in the dependency tree is missing.
