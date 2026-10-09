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

### S4 · Docker image (2026-10-09)
- `rocker/r-ver:4.6.1` + uv Python 3.13 + pinned requirements + R packages from P3M 2026-10-01 + Quarto
  1.10.19: linux/arm64 build 15 m 50 s cold on an Apple Silicon laptop, 3.0 GB. A first build silently
  missed `Boom` (download timeout, warning only); the Dockerfile now raises R's timeout and fails if any
  package in the dependency tree is missing.
