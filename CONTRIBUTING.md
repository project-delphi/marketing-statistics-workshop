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
checkpoint. Provided "Run" cells (no tag) carry the scaffolding.

### One exercise, R

```r
# %% tags=["exercise"]
build_rfm <- function(transactions) {
  # TODO 1
  stop("TODO 1")
}

# %% tags=["solution"]
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
- `workshop.use_reference(n)`, `workshop.summary()`, `workshop.run_record()`.
- Test hooks used by `scripts/test_notebooks.py`: `workshop._verify_begin(n)`, `workshop._verify_end(n)`.
- Settings read from the environment: `MKTSTATS_QUICK`, `MKTSTATS_WORKED`, `MKTSTATS_SABOTAGE`.

R (`R/mktstats.R`, sourced by the install cell): `solution(n, name, fn)`, `checkpoint(n, expr)`,
`use_reference(n)`, `workshop_summary()`, `run_record()`, `.ws_verify_begin(n)`, `.ws_verify_end(n)`;
`WORKED_EXAMPLE <- FALSE` and `QUICK <- FALSE` are plain variables overridden by the same env vars.
No Colab forms in R.

## Run record (printed between markers by the record cell; schema in `runs/README.md`)

`notebook`, `content_sha` (hash of the source's code cells, generated cells excluded), `deps_sha` (hash of
the files listed in `modules.mNN.deps` plus the pins, embedded by the generator), `ref`, `mktstats_commit`
(from `direct_url.json` when installed from git), `mode` (worked/learner), `settings` (QUICK etc.),
`status`, `seconds`, `install_seconds`, `cell_seconds` (Python), `python`/`r` version, `colab_release`
(`COLAB_RELEASE_TAG`), `cpus`, `packages` (key versions), `checkpoints` (label → passed, whose).

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
- `geo_panel(seed=2029, ...)` → table `panel` (date, geo, region, sales, treated, post). Truth:
  `lift_pct`, `log_lift`, `incremental_sales`, `treated_geos`, test window.
- `email_experiment(seed=2030, n=20000, ...)` → table `experiment` (Hillstrom-like covariates,
  treatment, conversion, spend, per-row `true_cate`). Truth: `ate`, `margin`, `offer_cost`, decision
  rule, policy values (treat none / all / oracle).

Data loaders (`mktstats.data`) read env `MKTSTATS_CACHE` (download cache, default
`~/.cache/mktstats`), `MKTSTATS_OFFLINE` (no downloads), `MKTSTATS_REF` (git ref for committed files
when not in a checkout), `MKTSTATS_DATA_DIR` (override the local `data/` folder) and
`MKTSTATS_SABOTAGE=swap_rf` (the RFM helpers swap recency and frequency).

## Checks (`mktstats.checks`) — raise `AssertionError` with a message that says what to look at

Examples: `rfm_table(rfm)`, `in_interval(truth, draws, prob=0.94, kind="hdi", name=...)`,
`k_of_K_in_interval(...)`, `close(estimate, truth, rel=..., abs=..., name=...)`,
`probability(x)`, `monotone(x, increasing=True)`. Every check has a pytest test that feeds it a broken
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
