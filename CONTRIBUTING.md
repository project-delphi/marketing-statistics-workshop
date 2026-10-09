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

All generators are seeded and return a result with `.data` (DataFrame(s)) and `.truth` (dict). Committed
copies with default seeds live in `data/synthetic/` (CSV) with all truths in `data/synthetic/truth.json`,
written by `scripts/make_synthetic.py`. R reads only the committed files.

- `retailer(seed, ...)` — the running retailer story. Customers acquired over time through acquisition
  channels; purchases and dropout from a **Pareto/NBD** process with static covariate effects of the
  acquisition channel on purchase and dropout rates (CLVTools' parameterization), Gamma-Gamma spend.
  Data: `transactions` (customer_id, date, amount), `customers` (customer_id, acquisition_channel,
  first_date, and truth columns `true_alive_at_cal_end`, `true_alive_at_end`, `true_lambda`, `true_mu`,
  `true_mean_spend`). Truth: `pnbd` {r, alpha, s, beta}, `covariates` {gamma_purchase, gamma_dropout per
  channel}, `gamma_gamma` {p, q, v}, calibration/holdout dates.
- `btyd_bgnbd(seed, ...)` — plain **BG/NBD** customers for the recovery exercise. Truth: {r, alpha, a, b}.
- `mmm(seed, ...)` — weekly sales with channel spend (`tv`, `search`, `social`, `display`), geometric
  adstock, logistic saturation, trend, yearly seasonality, controls. Truth: per channel `adstock_alpha`,
  `saturation_lam`, `beta`, `roas` (noiseless incremental sales ÷ spend over the stated window, raw units),
  `contribution_share`; the window; the lift-test results a geo experiment would have produced.
- `geo_panel(seed, ...)` — weekly sales for ~40 geos with a campaign in treated geos from a known week.
  Truth: `lift_pct`, `incremental_sales`, treated geos, test window.
- `email_experiment(seed, ...)` — Hillstrom-like randomized email offer with heterogeneous effects.
  Truth: per-row `true_cate` column, `ate`, offer cost and margin assumptions.

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
