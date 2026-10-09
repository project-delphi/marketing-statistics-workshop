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
