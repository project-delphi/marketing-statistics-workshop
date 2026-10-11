# Decisions and spike results

Each entry: what was decided, why, and the evidence (a command, a run, a URL with the date it was read).
Newest first within each section.

## Extensions track

### Five optional modules after the core week, built without staling the core (2026-10-10)
- The user asked for five additions: BTYD and Gamma-Gamma on messy real data, double machine
  learning for observational data, multi-touch attribution (Markov chains, Shapley values),
  survival analysis for contractual churn, and an MMM with time-varying effects and a Prophet
  baseline. They become Modules 13-17 on a new day `d6` ("Extensions") with no timed blocks,
  like the pre-work. The five timed days are full (a test checks 09:00-17:00) and were
  released at `v2026.10.1` and swept on Colab, so nothing was cut from them. Each extension
  keeps the core module format (briefing, 55-minute lab, debrief), so an instructor can swap
  one into a custom schedule.
- The core's run records hash each module's declared deps (`scripts/common.py` `deps_sha`).
  Modules 1, 3, 4 and 5 listed the whole `src/mktstats/synth` directory, so any new generator
  there would have staled them. They now list the same files one by one, in the same order;
  `deps_sha` of all 18 core notebooks was checked unchanged after the edit.
- Extension datasets keep their truth out of `truth.json`, which nine core modules hash: each
  writes `data/synthetic/truth_<name>.json` through the `EXTENSIONS` table in
  `scripts/make_synthetic.py`. Labs read them with `mktstats.data.repo_file`, and new helpers
  live in new files, so `data.py`, `checks.py` and `synth/__init__.py` are unchanged.
- Pins: `lifelines==0.30.3` (pure Python; PyPI, uploaded 2026-03-05) and `prophet==1.4.0`
  (Colab's preinstalled version, `environment/colab-constraints.txt`; PyPI has wheels for
  manylinux x86_64 and aarch64 and macOS arm64), with their dependencies autograd-gamma,
  cmdstanpy, holidays and stanio. Recompiling with the Colab constraints moved no existing
  pin. Module 9 lists `environment/requirements.txt` in its deps (the Robyn notebook reads
  nevergrad's pin from it), so its two run records read stale although nevergrad did not
  change; they are re-run in the next Colab sweep.
- The extension notebooks cannot run on Colab until a release tag contains them: Colab
  badges, the mktstats install and the data all point at `repo.ref`. A new tag changes every
  notebook's `deps_sha`, so it needs a full Colab sweep of the core as well.

### The extensions are judged and released apart from the core week (2026-10-10)
- `days.d6` has `track: extensions`. Its modules keep the module format, the notebook contract
  and the run records, but the readiness verdict, the README status line, the landing-page
  counts and the notebooks table are the core workshop's (pre-work and Days 1–5); the
  extensions get their own sentence, table and page (`extensions.qmd`), and the schedule lists
  them without a timetable. Without this, five unwritten optional modules made the released
  core read "16 of 23 … not yet ready" and "23 lab notebooks, 18 written so far".
- `release_check.py` fails on the core and the readiness items by default and always reports
  the extensions; `--extensions` (and `release.yml`'s `extensions` dispatch input) also requires
  them, so the core can be released while extensions are built.
- Notebook footers stay within a track: the capstone ends the core week and points to the
  Extensions page. Footers are outside `content_sha`, so all 18 core records stayed current
  (checked before and after). Extension module headers list their own `prerequisites`, not the
  module before them by number.
- Evidence: Quarto 1.10.19 render in the workshop image, 36 pages, no warnings; lychee 0.24.2
  `--offline` as in `publish.yml`: 2,485 links, 0 errors (also with `--include-fragments`).

### Module 15: double machine learning on histogram boosting, against an overlap-weighted truth (2026-10-10)
- `mktstats.synth.app_install` (seed 2035): 20,000 customers with nine pre-launch covariates;
  installing the app depends on them nonlinearly (loyal, recent, phone-using customers
  install), and the effect on 26-week spend varies with the customer. Nothing is hidden by
  default, so selection is on observables.
- Nuisance models are scikit-learn 1.6.1's histogram gradient boosting with
  `early_stopping=False` (`"auto"` switches early stopping on above 10,000 rows). LightGBM
  4.6.0 was not used: its macOS wheel needs Homebrew's libomp (it failed to import on the
  laptop without it), and inside EconML it raised feature-name warnings on every fit. econml
  0.17.0 still installs it as a dependency.
- Because the effect varies, residual-on-residual regression estimates the average effect
  weighted by e(1 − e), not the ATE; the checkpoints compare with that overlap-weighted truth
  ($14.28 against an ATE of $13.58 on the committed data), and the briefing derives it. Over
  30 seeds the 95% interval covered it in 27 of 30 (n = 20,000) and 29 of 30 (n = 10,000).
- EconML's `LinearDML` with the lab's folds reproduces the hand estimate to 2e-15; its
  interval is the hand HC0 interval times √(n/(n − 1)) (its default inference is HC1) and
  1.959964/1.96.
- On the committed data: naive $65.64, regression with linear controls $32.11, DML $13.51
  (95% CI $10.51 to $16.50). Leaving out one confounder (`mobile_share`) moves the estimate
  by +$6.39 and flips the decision. On a confounded subsample of the randomized Hillstrom test
  (outcome `visit`), the naive interval excluded the randomized benchmark (4.52 points) and
  `LinearDML` covered it in 30 of 30 spoiling seeds at full settings.
- A labeled check of provided data may now come before Checkpoint 1; the runner's learner
  verdict judges the first numbered checkpoint.

### Module 17: per-channel drift through PyMC-Marketing's HSGP; a Prophet baseline measured, not assumed (2026-10-10)
- pymc-marketing 1.2.0's `time_varying_media=True` is one multiplier shared by all channels
  (`mmm/mmm.py`; its docs notebook: "We assume all channels share the same time-dependent
  fluctuations"). Passing `SoftPlusHSGP.parameterize_from_data(..., dims=("date",
  "channel"), cov_func="matern52")` gives each channel its own mean-1 path with shared
  lengthscale and amplitude priors, so no custom PyMC model is needed. The user's example, a
  Gaussian random walk, is described in the briefing as the alternative.
  `parameterize_from_data` defaults to ExpQuad (Matérn 5/2 is only the default for
  `time_varying_media=True`), so the lab sets the kernel. All fits: 0 divergences at
  `target_accept` 0.98.
- `mktstats.synth.mmm_drift` (seed 2037): 156 weeks, Module 8's channels, social's effect
  fades from about 1.0 to 0.40 of its first-year level, with quarterly social pushes that
  make the drift identifiable. On the committed data the static MMM puts social's
  last-quarter ROAS at 3.88 (94% HDI 3.48 to 4.28) against a true 2.11 and says keep; the
  drifting model gives 2.53 (1.94 to 3.15) and says cut.
- Robyn 3.12.1 fits Prophet to the dependent variable and uses its trend, season and holiday
  components as regressors (`R/R/inputs.R`, read 2026-10-10). Fed into the MMM the same way,
  the Prophet baseline moved social's all-period ROAS from 3.90 to 2.90 against a true 4.00:
  Prophet's trend takes up the fading channel's contribution. In this simulation no variant
  tried helped every channel; the lab measures the practice and does not recommend it.
- Over 15 generator seeds (QUICK) every fit-against-truth check held in 15 of 15 except the
  Prophet checkpoint's fit check (12 of 15); the decision said cut in 14 of 15.
- Library traps: cmdstanpy 1.3.0 resets its logger to DEBUG on first use unless it already
  has a handler, so silencing Prophet needs a `NullHandler` first; without plotly, importing
  prophet logs an ERROR line; a lift-test row on a time-varying MMM needs a `date` equal to a
  model date.
- `scripts/test_notebooks.py --note` writes run conditions (Docker CPU quota, thread
  variables, time zone) into run records, so no record is edited by hand. Docker runs after
  midnight UTC need `-e TZ=<local zone>`, or the record carries tomorrow's date and fails the
  record check; runs from a worktree need `-e MKTSTATS_COMMIT`.

### Module 16 writes its own attribution code; ChannelAttribution is not used (2026-10-10)
- ChannelAttribution 2.2.5 (CRAN, published 2026-06-08) has first-, last- and linear-touch
  heuristics and k-order Markov models with removal effects, and no Shapley function
  (https://cran.r-project.org/web/packages/ChannelAttribution/refman/ChannelAttribution.html,
  read 2026-10-10). Its PyPI release 2.2.5 is a source distribution only (Cython and C++), so it
  would compile in the Colab install cell.
- Learners write the heuristics, the first-order transition matrix, the removal effect and exact
  Shapley values (32 coalitions for five channels), each checked on hand-worked examples; the
  absorbing-chain solve is a provided cell (Pedagogy review: the counterfactual is the lesson,
  the linear algebra is machinery). Checks live in `mktstats.attribution`.
- `mktstats.synth.mta` (seed 2036, 100,000 users): exposure follows latent intent, retargeting
  has no effect by construction, and each channel's true incremental conversions come from
  re-running every user's outcome with that channel removed, using the same draws. A holdout
  uses its own random stream, so the default data do not depend on it (tested).
- The CSV stores one letter per touch (1.34 MB instead of 3.5 MB). The market stays at 100,000
  users because the decision compares a holdout's MDE, which grows with the square root of the
  market, with a spend-fixed break-even: at 60,000 users the decision's economics changed
  (measured).
- Result on the committed data (FULL, the notebook's printed output at the commit of the run
  records): Markov and Shapley give retargeting 23.2% and 24.6% of the credit against no true
  effect, and none of the 25 true shares lies inside its 95% bootstrap interval. Across eight
  seeds, data-driven credit was no better than first-click or linear credit, so the lab claims
  "not incrementality", not "worse than heuristics". The decision compares the holdout's MDE,
  scaled to all users reached (about 250 conversions a month), with the break-even (278), not
  with the credited conversions, which would be a weaker bar.

### Module 13: a dirtied synthetic log, a maximum-likelihood helper, and Online Retail II (2026-10-10)
- The messy log (`mktstats.synth.messy`, seed 2033) is the retailer's customers 1–2,500 plus
  13 wholesale buyers, with the schema `load_online_retail_ii()` returns and error rates
  measured on Online Retail II (lines without a customer ID 22.4% against 22.5%, postage
  invoices 4.83% against 4.88%, duplicate lines 1.45% against 1.44%, cancellation lines 2.28%
  in both). Labels go only to `truth_messy.json`. Fewer customers than the prototype keep the
  plain CSV at 2.4 MB; it is not gzipped, because compressed bytes can differ between zlib
  builds and the drift gate compares bytes.
- The lesson the data support is aggregate versus individual: Module 1's cleaning rules move
  the 52-week forecast by +2.8% but move 449 of 2,513 customers' scores by more than 10%. On
  Online Retail II, customer 12346 is worth £18,026.64 over 26 weeks under Module 1's rules and
  £113.12 after cancellation matching (an order of 74,215 units cancelled the same day).
- The sweep of cleaning variants needs about 20 fits, so `mktstats.mle` maximizes
  pymc-marketing 1.2.0's closed-form BG/NBD and Gamma-Gamma likelihoods with SciPy (each fit
  under 0.3 s). It reproduces Fader and Hardie's notes 004 and 025 at their published
  precision and agrees with PyMC-Marketing's flat-prior MAP to 1.6e-5 relative or better on
  the lab's data; PyMC-Marketing is fitted once in the lab as the cross-check (Pedagogy
  review: approved because Modules 2 and 4 already show flat-prior MAP is maximum likelihood).
- Tolerances come from 30 independent generator seeds: the holdout forecast within 20% (the
  largest error was 17.8%; the BG/NBD over-predicts this Pareto/NBD holdout by 8% on average)
  and the capped population mean within 8%.
- Olist is not used: Kaggle's API metadata gives CC BY-NC-SA 4.0, Kaggle's docs say an account
  is needed, and only 2.2% of its customers bought on more than one day (measured on the files).
- Library traps found in the build: PyTensor turns Python floats that float32 holds exactly
  (2.0, 2.5) into float32 constants, so `pm.logp(BetaGeoNBD.dist(b=2.0, …))` computes
  `gammaln(b)` in single precision (pass `np.float64`; pymc 6.3.2, pytensor 3.3.3);
  `fit_summary()` after a MAP fit rounds its values (ArviZ `round_to="auto"`, arviz 1.3.0;
  read `model.fit_result`); DuckDB 1.3.2's ASOF join breaks exact ties differently from run to
  run with more than one thread, so the lab checks only counts that do not depend on the tie.
  The join matches 123 purchases twice on Online Retail II; the lab states the approximation
  (one-to-one pairing would change 17 customers' calibration data and one contact).

### Module 14: time zero, a fading covariate, interval checks and coverage (2026-10-10)
- `mktstats.synth.subscriptions` (seed 2034): 8,000 subscribers signing up from 2024-01-01 to
  2025-12-31, observed to 2026-06-30; Weibull proportional hazards (shape 0.8, scale 14
  months) in plan and onboarding; onboarding confounded by plan; an intro offer with hazard
  ratio 2.01 before month 3 and 1 after. The truth is analytic (restricted mean survival time
  in closed form). Time zero is the first billing date after a 14-day trial, so onboarding is
  known when the clock starts (no immortal-time bias); the fading covariate is the offer,
  known at sign-up, not early support tickets.
- Checks on library confidence intervals and a stratified bootstrap live in the new
  `mktstats/survival.py`; `checks.py`, which the core hashes, is unchanged.
- Coverage of the checkpoint rules with the reference solutions over 80 seeds: 78–80 for every
  rule except the two that rest on one 95% interval each, Cox onboarding (73 of 80) and the
  RMST bootstrap (72–73 of 80). The prototype's 38–40 of 40 came from one seed set. On the
  committed seed every rule holds and a correct solution reproduces the reference exactly.
- The build found that 50/50 weights in Checkpoint 5 let a plain average pass; they are now
  75/25.
- lifelines 0.30.3 traps: `predict` carries the last value flat past the data without a
  warning; `restricted_mean_survival_time(return_variance=True)` is the variance of min(T, t),
  not a standard error; `confidence_intervals_` is on the log scale and its column names carry
  the level; `CoxPHFitter.fit` uses every other column as a covariate, so passing the whole
  table raises `TypeError: Cannot cast DatetimeArray to dtype float64`.

### BLAS and OpenMP oversubscribe under a Docker CPU quota (2026-10-10, Module 16 build)
- In the workshop image with `--cpus 2` on a 10-core Mac, OpenBLAS 0.3.27 starts 10 threads:
  200 matrix-vector products (8,441 × 64) took 5.0 s, against 0.02 s with 1 or 2 threads. Module
  16's bootstrap took 13.5 s in Docker against 1.2 s natively until it used `np.bincount`
  (1.4 s). Docker `--cpus N` times of BLAS- or OpenMP-heavy cells can therefore overstate what a
  machine with N real cores (Colab: 2) takes. Later extension builds run their Docker records
  with `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to the quota and say so in the record.
  Run records report `os.cpu_count()` (the host's cores), not the quota.

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

### KaTeX pinned to 0.19.0; equation numbers never cover an equation (2026-10-10)
- The site loaded KaTeX from `cdn.jsdelivr.net/npm/katex@latest`, pandoc's default (pandoc 3.10 in
  Quarto 1.10.19), so the maths on every page followed KaTeX's releases with no commit here: six
  between 2026-09-23 and 2026-10-01, the last (0.19.0) with a breaking change to `strict`
  handling. `_quarto.yml` now sets `html-math-method: {method: katex, url: …katex@0.19.0/dist/}`
  (Quarto's schema allows `url`, `share/schema/document-options.yml`). 0.19.0 is what the live
  site already served and was checked with in Chrome on 2026-10-10: 74 of 74 formulas on
  Module 8 rendered, none as `.katex-error`, no console messages.
- KaTeX places an equation number with `position: absolute; right: 0`, so a display equation
  wider than the text column ran under its number. In Chrome at 1280 px (a 714 px column), 18
  numbered equations on Modules 3 and 7–12 overlapped their number, by up to 749 px (Module 9's
  lift-test row). Two fixes: `custom.scss` makes the number a flex item that stays at the right
  when there is room and otherwise follows the equation inside its scroll box (it uses 0.19's
  class names `katex-base` and `katex-tag`, another reason to pin); and the 15 widest equations,
  which joined several definitions with `\qquad`, now put one definition per line (`aligned` or
  `gathered`), with no symbol changed.

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
  live site (pages from its `sitemap.xml`, plus `404.html`, which the sitemap leaves out) instead of
  rebuilding it, so a link check no longer renders and redeploys the site; the scheduled redeploy
  still happens through the notebooks cron → publish (monthly since 2026-10-10, the user's choice).
  It also downloads every dataset source in `mktstats.data.DATASETS`, fallbacks included, with the
  loaders' own `_download` and checks its sha256: the loaders move to the next source silently when one fails or its hash differs, so the
  notebooks run does not notice a dead or changed source, and an HTTP status alone would not show a
  changed file. By hand in a clean container on 2026-10-10, all 9 sources matched. The site URL is
  read from `repo.site` in `_variables.yml`.
- First run, by hand with lychee 0.24.2 on 2026-10-10: 29 pages, 2,017 links, 57 errors. 49 were
  403s from journal publishers behind doi.org (they refuse non-browser clients), 6 were
  online.stat.psu.edu (incomplete certificate chain; the pages load with curl), 2 were transient 503s
  from github.com. Accepting 403 everywhere would hide a dead link on any host that answers 403
  (a private S3 object, say), so only those two hosts are treated differently: DOIs are checked at
  doi.org without following the redirect (a registered DOI answers 302, a wrong one 404, both
  checked), and online.stat.psu.edu with `--insecure`. Every other link must give 200, 203, 206
  or 429; 429 is accepted because it means the host is up but throttling the runner, and failing
  on it would make the run red for links that work. With these settings, by hand: 2,044 links on
  30 pages with 0 errors, and the 66 DOI links and 6 Penn State links all pass.
- lychee is pinned to v0.24.2, the default of lychee-action v2 when checked
  (https://github.com/lycheeverse/lychee-action/blob/v2/action.yml, read 2026-10-10), so a new
  default cannot change the status handling these settings rely on; publish.yml's blocking
  internal-link check uses the same version.
- First CI runs, dispatched on main on 2026-10-10: `release.yml` with `as_of=2026-10-31` printed
  "Ready to release as of 2026-10-31: no blockers" (run 38073181420). `links.yml` failed (run
  38073188370): 2,046 links, 24 errors, every one a 503 from a github.com file page
  (`/blob/<ref>/<path>`), 23 of them this repository's. The same URLs gave 503 by hand with curl,
  one at a time and with a browser user agent, while repository and `/tree/` pages gave 200, the
  same files at raw.githubusercontent.com gave 200, and githubstatus.com said all systems
  operational. So github.com refuses many file pages to logged-out clients (not concurrency).
  A GitHub token does not help: lychee 0.24.2's API fallback (`check_github` in
  `lychee-lib/src/checker/website.rs`, read 2026-10-10) checks only that the repository exists
  and reports any URL with a path as an error. Fix: `--remap` each `/blob/` link to
  raw.githubusercontent.com with the same owner, repository, ref and path. By hand with lychee
  0.24.2 in Docker: 2,046 links on 30 pages, 0 errors, and a planted link to a missing file at
  the tag gave 404.
- `notebooks.yml`'s schedule (the notebook matrix and the slow recovery tests) moved from weekly
  to monthly on 2026-10-10 at the user's request: no scheduled job in this repository runs weekly.
  It runs on the 1st at 04:17, before `links.yml` (05:41) and `release.yml` (06:23), so the release
  check reads that day's CI records: `collect` pushes them once the notebook jobs finish (3.5 minutes
  after the start in the dispatched run of 2026-10-10) and does not wait for the recovery job.
  That dispatched run (38073363414, on main) was the recovery job's first run in CI: `pytest -m
  slow`, 65 passed in 10 min 53 s; all 18 notebooks passed in the same run.

### Module 9's cross-validation cell reaches the 4-minute cell limit on Colab (2026-10-10)
- PLAN.md's Colab targets include "any cell ≤ 4 min". The time-slice cross-validation cell (4 folds ×
  2 chains × 500 draws, FULL) took 215 s, 241 s and 252 s in the three Colab runs at `main`,
  `v2026.10.0` and `v2026.10.1` (`runs/2026-10-10-colab-py-python-09-mmm-calibration.json`, the
  21st timed cell). The same notebook's two main fits come next (up to 209 s); outside Module 9 no
  cell took more than 161 s (the capstone). The whole notebook took 697 s at `v2026.10.1`, inside
  the 15-minute total.
- Not fixed in `v2026.10.1`: the overrun is 12 s, and changing the lab stales its Colab record
  (and a new tag stales all 18). The brief's rule is to cut folds before draws, so the next release
  should run 3 folds FULL (as QUICK does) or start the first fold later; then re-measure on Colab.

### Queued for the next release (lab changes that need a Colab re-sweep)
Changes to `labs/src`, `src/mktstats`, `R/` or the pins make the Colab records stale on main, and a
new tag makes all 18 stale, so these wait until the next sweep instead of going to main one by one.
- Module 9: the cross-validation cell (above).
- R Module 0: the lab carries its own CDNOW loader with one URL (pymc-marketing's `main` branch) and
  a visible `TODO(mktstats data loaders)` comment; `R/mktstats.R` has no CDNOW loader. Python's
  `mktstats.data` reads the same file from pymc-marketing's `1.2.0` tag first and `main` second.
  Move the loader into the R helpers with both sources in that order, and drop the comment.
- Harness (`src/mktstats/harness.py`, hashed by every Python notebook), found by the Extensions
  notebook reviews (2026-10-10): a failed checkpoint says "Read the message above", but in Jupyter
  and Colab the checkpoint's message is the last line of the error below it (the R helpers already
  say "below"); `use_reference(n)` says "Rerun the cells below it", but several labs' checkpoints
  create values later cells use, so it should say "rerun this exercise's checkpoint, then the cells
  below it"; a labeled check's recovery line says `use_reference(N)` without the number; the record
  printed inside a notebook carries the UTC date while `scripts/test_notebooks.py` uses the local
  date. `teach.qmd` tells instructors about the first two until then.
- Runner: Docker records report `cpus` from `os.cpu_count()` (the host's cores), not the
  `--cpus` quota; the time zone and the commit are passed by hand (`-e TZ`, `-e MKTSTATS_COMMIT`)
  and noted with `--note`. Reading the quota from the cgroup and documenting the Docker command in
  `runs/README.md` would remove the hand steps.

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
