# CONTRIBUTING — contracts every part of the build relies on

Read `AGENTS.md` first. This file fixes the interfaces between the parts so they can be built in
parallel: the notebook source format, the harness API (Python and R), the synthetic-data API, the run
record, and the HTML classes the generated includes use. Change a contract here first, then the code.

## Where to edit

| You want to change | Edit | Then run |
|---|---|---|
| A module's title, timing, objectives, decision | `_variables.yml` | `python scripts/gen_tables.py` |
| A lab | `labs/src/NN-slug.py` or `.R` | `python scripts/gen_notebooks.py` |
| Notebook header, install, harness, record or footer cells | `scripts/gen_notebooks.py`, `src/mktstats/harness.py`, `R/mktstats.R` | `python scripts/gen_notebooks.py` |
| Synthetic data or its truth | `src/mktstats/synth/` | `python scripts/make_synthetic.py` |
| Package pins | `environment/requirements.in`, `_variables.yml` `packages` | `uv pip compile …` (see the file), `python scripts/gen_tables.py` |
| Site look | `custom.scss`, `custom-dark.scss`, `_quarto.yml` | `quarto render` |

## Notebook source format (jupytext percent, one-way)

Sources live in `labs/src/`. `scripts/gen_notebooks.py` builds `labs/python/*.ipynb` and `labs/r/*.ipynb`
from them and from `_variables.yml`; never edit the `.ipynb` files. Cells are marked only with tags; the
generator adds the Colab/Jupyter metadata (`cellView: form`, `jupyter.source_hidden`) and the kernelspec
(`python3` or `ir`).

The generator writes five cells around the source: **header** (title, Colab badge at `repo.ref`, day,
minutes, goals, stack), **install** (runtime detection and pinned installs), **harness**
(`WORKED_EXAMPLE`, `QUICK`, the `workshop` object), **record** (summary + run record) and **footer**
(next lab, site, licences). A source file contains everything between the harness and the record.

Section and exercise headings are parsed by `scripts/gen_tables.py` for the lab step tables:

```
# Part A · Title                         (markdown H1 inside the notebook)
## Exercise N · Title (M minutes)
## Stretch (optional) · Title
## Decision · Title                      (the business decision cell(s); every lab has one)
```

### One exercise, Python

```python
# %% [markdown]
# ## Exercise 1 · Build the RFM table (8 minutes)
#
# **Predict.** Before you run anything: what share of CDNOW customers never bought again?
#
# **Task.** Write `build_rfm(transactions)` returning one row per customer with
# `frequency` (repeat purchases), `recency` (time of last purchase since the first) and `T` (age).

# %% tags=["exercise"]
def build_rfm(transactions):
    # TODO 1
    raise NotImplementedError("TODO 1")

# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def build_rfm(transactions):
    ...

# %% [markdown]
# **Explain.** Compare with your prediction. …
#
# <details><summary>Why this solution works</summary>
# …
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    rfm = build_rfm(tx)
    checks.rfm_table(rfm)
```

Rules: every exercise is a **function** (bindable to a reference, testable). The stub raises
`NotImplementedError("TODO N")`. A checkpoint is one `with workshop.checkpoint(N):` block using
`mktstats.checks` functions (or plain asserts with messages that say how to recover). The first
checkpoint comes before any model fit or long download. Fit each model once per lab; later checkpoints
test pure functions or that single fit. `QUICK` shortens sampling and subsamples data; it never skips a
checkpoint. Provided "Run" cells (no tag) carry the scaffolding. A Python solution cell starts
with `# @title Solution N — try it yourself first { display-mode: "form" }`; an R solution cell
with `#@title Solution N — try it yourself first { display-mode: "form" }`, under a markdown line
"**Solution N — try it yourself first.**" (Colab's R runtime may not fold it).

Colab moves the output of a cell that shows rich or JavaScript output (a progress bar,
`google.colab.files.download`) into a sandboxed frame that a script cannot read. So pass
`progressbar=False` to every fit or sampler call that accepts it, never call `files.download`,
and keep the record cell plain text (it prints the record and writes `run_record.json`).

A module with two notebooks of the same name (`labs/python/00-setup-warmup.ipynb` and
`labs/r/00-setup-warmup.ipynb`) offers one lab in two languages: a participant does one. Two
notebooks with different names (Modules 3, 7, 11) are both done in the lab slot, and the lab
step table adds up both against the slot.

Each notebook entry in `_variables.yml` has `install:`, the packages its install cell makes
sure of (Python: versions from `packages`, else `environment/requirements.txt`; R: CRAN
packages from the P3M snapshot and `github_packages` from their pinned archives). A notebook's
id in run records is its path under `labs/` without `.ipynb`: `python/00-setup-warmup`.

### One exercise, R

```r
# %% tags=["exercise"]
build_rfm <- function(transactions) {
  # TODO 1
  stop("TODO 1")
}

# %% [markdown]
# **Solution 1 — try it yourself first.** The next cell is the reference solution.

# %% tags=["solution"]
#@title Solution 1 — try it yourself first { display-mode: "form" }
solution(1, "build_rfm", function(transactions) {
  ...
})

# %% tags=["checkpoint"]
checkpoint(1, {
  rfm <- build_rfm(tx)
  check_rfm_table(rfm)
})
```

## Harness API

Python (`src/mktstats/harness.py`; the generated harness cell calls `start(...)` and binds `workshop`):

- `workshop.solution(n)` — decorator; stores the reference. Binds it only in worked mode
  (`WORKED_EXAMPLE = True` or env `MKTSTATS_WORKED=1`) or after `use_reference(n)`; otherwise the
  learner's definition stays. `workshop.solution_value(n, name, obj)` for non-callables.
- `with workshop.checkpoint(n):` — runs the checks; prints "passed on your code" / "passed on the
  REFERENCE solution" / a recovery message (`NotImplementedError` → "TODO n is not written yet …
  workshop.use_reference(n)"), then re-raises so Run all stops.
- `with workshop.checkpoint(label="..."):` — the same for checks of provided code (no exercise).
- `workshop.use_reference(n)`, `workshop.summary()`, `workshop.run_record()`.
- Test hooks used by `scripts/test_notebooks.py`: `workshop._verify_begin(n)`, `workshop._verify_end(n)`.
- Settings read from the environment: `MKTSTATS_QUICK`, `MKTSTATS_WORKED` (`1` forces worked mode,
  `0` forces learner mode), `MKTSTATS_SABOTAGE` (`"2,3"`: those exercises keep their stubs even in
  worked mode, so a whole run must stop at their checkpoints). The harness cell also imports
  `checks` (`mktstats.checks`) for the lab's checkpoints.
- IPython hooks only time cells; rerunning the harness removes only its own hooks.

R (`R/mktstats.R`, sourced by the install cell): `solution(n, name, fn)`, `checkpoint(n, expr)`,
`use_reference(n)`, `workshop_summary()`, `run_record()`, `.ws_verify_begin(n)`, `.ws_verify_end(n)`;
`WORKED_EXAMPLE <- FALSE` and `QUICK <- FALSE` are plain variables overridden by the same env vars.
No Colab forms in R. The generated harness cell calls `.ws_start(notebook, content_sha, deps_sha,
ref, exercises)`. Checks: `check_columns`, `check_rfm_table`, `check_in_interval`, `check_close`,
`check_frames_agree`. `mkt_data(path)` reads a committed file under `data/` (the checkout when
`MKTSTATS_REPO_ROOT` is set, else raw GitHub at the ref).

### Running and recording

```
python scripts/test_notebooks.py --notebooks python/00-setup-warmup            # worked + verify
python scripts/test_notebooks.py --notebooks 00-setup-warmup --mode both       # both languages, both modes
python scripts/test_notebooks.py --notebooks ID --learner                      # must stop at checkpoint 1
python scripts/test_notebooks.py --notebooks ID --full --record runs/FILE.json --env docker-arm64
python scripts/test_notebooks.py --list-matrix                                 # [{id, slug, kernel, path}]
```

Worked mode runs, after the first checkpoint of each exercise, a copy of it on the stub that
must fail (verify mode). `MKTSTATS_QUICK=1` unless `--full`. The R notebooks need the `ir`
kernel (the workshop image).

## Run record (printed between markers by the record cell; schema in `runs/README.md`)

`notebook`, `content_sha` (hash of the source's code cells, generated cells excluded), `deps_sha` (hash of
the files listed in `modules.mNN.deps` plus the pins, embedded by the generator), `ref`, `mktstats_commit`
(from `direct_url.json` when installed from git), `mode` (worked/learner), `settings` (QUICK etc.),
`status`, `seconds`, `install_seconds`, `cell_seconds` (Python), `python`/`r` version, `colab_release`
(`COLAB_RELEASE_TAG`), `cpus`, `packages` (key versions), `checkpoints` (label → passed, whose).
Also `date`, `kernel`, `runtime` (colab/sagemaker/local), `platform`, `memory_gb`, `cell_errors`
(Python) and `verified` (verify mode). `install_seconds` comes from `MKTSTATS_INSTALL_SECONDS`, which
the install cell sets.

## Synthetic data API (`mktstats.synth`)

All generators are seeded (numpy `Generator(PCG64(seed))`) and return a `SynthResult` with `.data`
(a dict of named DataFrames, also reachable as attributes: `res.transactions`) and `.truth` (a
JSON-serialisable dict). Committed copies with default seeds live in `data/synthetic/` (CSV) with all
truths in `data/synthetic/truth.json`, written by `scripts/make_synthetic.py` (`--check` is the drift
gate). R reads only the committed files. `truth.json` also holds a `tolerances` block: how close a
correct fit gets, measured over independent seeds; checkpoints and pages cite it. Fit-and-compare
helpers are in `mktstats.recovery`; loaders for committed files are `mktstats.data.load_synthetic(name)`
and `load_truth()`.

- `retailer(seed=2026, ...)` → tables `transactions` (customer_id, date, amount; every purchase,
  first included) and `customers` (customer_id, acquisition_channel, first_date, 0/1 indicators
  `channel_social`, `channel_referral`, truth columns `true_lambda`, `true_mu` (per week),
  `true_mean_spend`, `true_alive_at_cal_end`, `true_alive_at_end`). Pareto/NBD purchases and dropout
  with static covariates in the CLVTools / pymc-marketing parameterization
  (`alpha_i = alpha * exp(-gamma_purchase' x_i)`, `beta_i = beta * exp(-gamma_dropout' x_i)`; `search`
  is the reference level), Gamma-Gamma spend. Rates are per week; summarize with `time_unit="D",
  time_scaler=7`. Truth: `pnbd` {r, alpha, s, beta}, `covariates` {gamma_purchase, gamma_dropout per
  channel, columns}, `gamma_gamma` {p, q, v}, `population_mean_spend`, calibration/holdout dates,
  `value_by_channel` (true expected purchases, spend per transaction and discounted CLV per channel,
  for existing customers at the calibration end and for a new customer; horizon and discount rate
  stated).
- `btyd_bgnbd(seed=2027, ...)` → tables `transactions` (customer_id, date) and `rfm` (frequency,
  recency, T in weeks exactly as pymc-marketing's `rfm_summary(..., time_unit="D", time_scaler=7)`,
  test_frequency, test_T, true_lambda, true_p, true alive flags). Truth: `bgnbd` {r, alpha, a, b}.
- `mmm(seed=2028, ...)` → tables `weekly` (date_week, tv, search, social, display, price_index,
  holiday, t, y) and `lift_tests` (channel, x, delta_x, delta_y, sigma — the
  `MMM.add_lift_test_measurements` format — plus true_delta_y). Truth per channel: `adstock_alpha`,
  `saturation_lam` (scale-free), `saturation_beta_model_units` (what a default-scaled pymc-marketing
  MMM fitted on all rows reports) and `saturation_beta_sales_units`, `roas` (noiseless contribution ÷
  spend over all weeks, raw units) and `roas_with_carryover`, `contribution_share`, `share_of_sales`;
  the window and definitions; `true_optimal_allocation` (steady-state weekly split of a stated budget
  within stated bounds).
  `true_response(spend, truth["mmm"]["channels"][c])` gives a channel's true steady-state weekly
  response.
- `mmm(seed, confounded=True)` = `mmm_confounded(seed=2028)` (Module 9; files
  `mmm_confounded_weekly.csv`, `mmm_confounded_lift_tests.csv`, `mmm_confounded_latent.csv`, truth
  `mmm_confounded`) → the same `weekly` columns, with an unobserved AR(1) demand shock that
  multiplies `search` spend by `exp(0.15 d_t)` and adds `9,000 * d_t` to sales; tv, social,
  display, price and holiday equal `mmm()`'s. True ROAS stays causal (search 3.0); measured with
  `recovery.fit_mmm` (nutpie, 2 chains, 300 and 1000 draws) the uncalibrated 94% HDI for search
  lies above 3.0 and calibrating on the two search rows brings it inside a narrower HDI
  (`tolerances.mmm_confounded_mcmc`). Tables:
  `lift_tests` (channel, x, delta_x, delta_y, sigma, **date** = last week of the test, test_start,
  test_weeks, true_delta_y; one +30% test per channel plus a second search test that switches search
  off, `delta_x = -x`; sigma 8% of the lift; sorted by date) and `latent` (date_week, demand_shock,
  demand_effect, media_contribution_<channel>). Use `date` as `lift_test_date_column` in
  `TimeSliceCrossValidator.run`; both search tests end by week 101, so `n_init >= 101` keeps them in
  every fold. Truth adds `confounding` (mechanism; naive vs oracle OLS ROAS with the true transforms
  and the omitted-variable-bias identity) and `lift_tests.rows`.
- `geo_panel(seed=2029, ...)` → table `panel` (date, geo, region, sales, treated, post). Truth:
  `lift_pct`, `log_lift`, `incremental_sales`, `treated_geos`, test window, and `campaign` (cost =
  1.25% of each treated geo's mean weekly pre-period sales per test week, $25,000; incremental ROAS,
  break-even ROAS at 30% margin, incremental margin, net return).
- `email_experiment(seed=2030, n=20000, ...)` → table `experiment` (Hillstrom-like covariates,
  treatment, conversion, spend, per-row `true_cate`). Truth: `ate`, `margin`, `offer_cost`, decision
  rule, policy values (treat none / all / oracle).
- `channel_value(retailer_truth, {name: mmm_truth})` → truth block `channel_value`, the **shared
  channel vocabulary** (`mktstats.synth.channels`). Media channels (tv, search, social, display:
  where money is spent) and acquisition channels (search, social, referral: how a customer first
  arrived) are different lists; the e-mail `channel` column (Phone/Web/Multichannel) is a third
  thing. `media_to_acquisition` (tv → 30% search, 70% referral; search → search; social → 90%
  social, 10% referral; display → 60% search, 40% social), `cost_per_new_customer` (tv 150, search
  150, social 100, display 300 dollars; `new_customers_per_dollar` = 1 / cost, constant within the
  bounds), new-customer value in margin **excluding the first purchase** (the MMM's short-run sales
  already count it), and `long_run_optimal_allocation` for `mmm` and `mmm_confounded`: the steady
  weekly plan maximizing `margin * Σ r_c(x_c) + Σ x_c / cost_c * clv_c` within the same budget and
  bounds as `true_optimal_allocation` (concave plus linear, solved exactly by
  `mmm.optimal_allocation_value`), with `value_sensitivity` (how flat the objective is: a value
  tolerance of a few percent cannot tell the short-run and long-run plans apart).
- `capstone(seed=2032)` (Module 12; files `capstone_transactions.csv`, `capstone_customers.csv`,
  `capstone_mmm_weekly.csv`, `capstone_geo_panel.csv`, `capstone_email_experiment.csv`; truth
  `capstone`) → tables `transactions`, `customers` (retailer format, 4,000 customers), `weekly`
  (MMM format, confounded search, ends the week before the geo test), `panel` (geo format; in 8 of
  40 regions paid search was **switched off** for the last 10 weeks; each region responds as a
  scaled copy of the national curve, so the national-equivalent lift row `x, -x, r(0) - r(x)` lies
  on the MMM's true curve), `experiment` (e-mail format plus `split`, train/test halves; margin
  30%, offer cost $0.75). Truth: `scenario` (what pairs are told: dates, treated geos, base and
  changed spend, the share rule and national-equivalent rule, budget, bounds, current plan,
  channel link, offer economics) and `answers` per stage, each with its `checkpoint` tolerance:
  `stage1_customer_value` (36 months, 1% a month, Module 5's discrete convention, margin, by
  acquisition channel), `stage2_geo_test` (true sales change, lift, treated share, campaign
  economics, national-equivalent lift row), `stage3_mmm` (true ROAS), `stage4_allocation`
  (long-run optimal plan), `stage5_targeting` (policy values on all rows and on the test split);
  plus the full `retailer`, `mmm`, `geo_panel` and `email_experiment` truths. Measured reference
  results for each checkpoint are in `tolerances.capstone`.

Recovery helpers added for these: `recovery.sc_weights(y_pre, x_pre)` and
`recovery.synthetic_control(panel, treated_geos, test_start)` (treated mean vs controls, in-space
placebos, RMSPE-ratio p-value, relative placebo effects). The committed CSVs are listed in
`mktstats.data.SYNTHETIC_TABLES`, so `load_synthetic("capstone_geo_panel")` and the others read them.

Data loaders (`mktstats.data`) read env `MKTSTATS_CACHE` (download cache, default
`~/.cache/mktstats`), `MKTSTATS_OFFLINE` (no downloads), `MKTSTATS_REF` (git ref for committed files
when not in a checkout; the install cell sets it to `repo.ref`), `MKTSTATS_DATA_DIR` (override the local `data/` folder) and
`MKTSTATS_SABOTAGE=swap_rf` (the RFM helpers swap recency and frequency).

## Checks (`mktstats.checks`) — raise `AssertionError` with a message that says what to look at

Examples: `rfm_table(rfm)`, `in_interval(truth, draws, prob=0.94, kind="hdi", name=...)`,
`k_of_K_in_interval(...)`, `close(estimate, truth, rel=..., abs=..., name=...)`,
`probability(x)`, `monotone(x, increasing=True)`, `columns(df, required)`,
`frames_agree(left, right, on=..., names=(...))`, and `interval(draws, prob, kind)` (HDI or ETI
computed with numpy). Every check has a pytest test that feeds it a broken
input and expects failure.

## Markup classes used by generated includes (styled in `custom.scss`)

Reuse the class names of project-delphi/nlp-llms so includes and styles agree: `.hero`, `.eyebrow`,
`.hero-subtitle`, `.hero-lead`, `.hero-actions`, `.btn-primary-action`, `.facts`/`.fact`, `.day-grid`,
`.day-card`, `.day-card-title`, `.day-card-question`, `.day-card-span`, `.path`, `.path-day`,
`.path-day-label`, `.path-step`, `.path-step-optional`, `.path-num`, `.path-body`, `.path-title`, `.chip`,
`.module-header`, `.module-meta`, `.module-num`, `.module-time`, `.module-clock`, `.module-summary`,
`.module-actions`, `.btn-colab`, `.btn-quiet`, `.module-details`, `.module-outcomes`, `.prerequisites`,
`.module-decision`, `.recap`, `.lab-steps`, `.challenge`, `.self-check`, `.predict`, `.timetable`,
`.slot-briefing`, `.slot-lab`, `.slot-debrief`, `.slot-break`, `.module-cards`, `.module-card`,
`.readiness-table`, `.outcomes`, `.rhythm`, `.prose`, `.section-lede`.

## Pages and the explicit render list

`_quarto.yml` renders only the pages it lists. Partial content included into pages lives in files whose
names start with `_` (e.g. `days/_day-1-notes.md`, written by the Pedagogy Expert and included by the day
pages). Notebooks are resources, never executed at render time.
