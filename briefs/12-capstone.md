# Lab brief · Module 12 · Capstone: one decision pipeline for the retailer

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert (who writes the rubric, `briefs/12-capstone-rubric.md`). |
| Notebook | `labs/src/12-capstone.py` → `labs/python/12-capstone.ipynb` |
| Slot | Day 5 from 11:25 (lead's schedule): briefing 15, pair work 80, lunch 50, pair work 100, break 10, presentations 60, wrap-up 20. `modules.m12.minutes`: briefing 15, lab 240 (80 + 50 + 100 + 10), debrief 60. |
| Budget | Pair work 180 min: open 5 + stages 150 + slack 25 (15 in block 1, 10 in block 2). Minutes are estimates until a pilot. |
| Work mode | Pairs, one notebook per pair, one runtime. |

## Question and decision

**Question.** Putting the week together: what should the retailer spend next quarter, where, and on whom?

**Decision.** Recommend next quarter's budget by channel and the customers to target with the retention
offer, with expected value, a 94% interval and the main risk to the recommendation
(`modules.m12.decision`), presented as a five-slide brief.

## What this lab fixes

Modules 1–11 each answered one question with its own data. Here each stage's output feeds the next, so an
error early on shows up in the final number, and every stage is checked against the truth before the next
one uses it.

## Data: one coherent synthetic scenario

All stages use one seeded scenario with consistent truth (open item: a `mktstats.synth.capstone(seed)`
wrapper or equivalent): the retailer's customers with acquisition channels (Stage 1), a geo panel for a
campaign on one **media channel of the MMM** whose true lift agrees with the MMM truth for that channel
(Stages 2–3), the weekly MMM with the same channel names (Stages 3–4), new customers per dollar by channel
(Stage 4), and the email experiment with true CATE (Stage 5). The capstone seed differs from the module labs'
seeds so answers cannot be copied, and the truth is revealed only in the decision cells.

## Provided toolkit

Folded "toolkit" cells hold reference implementations of the earlier labs' functions the pipeline needs
(RFM, `discounted_clv`, `sc_weights`, `permutation_pvalue`, `lift_test_row`, `response_curve`, `allocate`,
`targeting_profit`). Pairs may paste their own versions instead. Fits are provided cells with QUICK/FULL
settings (below). The stage exercises are the integration functions.

## Model fits (provided cells)

| Stage | Fit | QUICK | FULL | Estimate (until a run record) |
|---|---|---|---|---|
| 1 | `BetaGeoModel` with channel covariates, MCMC nutpie; `GammaGammaModel` MCMC | chains=2, draws=tune=300, 2,000-customer subsample | chains=2, draws=tune=1000 | about a minute plus first-fit compile |
| 3 | Calibrated MMM (`build_model` → `add_lift_test_measurements` → `fit`) | draws=tune=300 | draws=tune=1000 | under a minute |
| 5 | `CausalForestDML(discrete_treatment=True)` | 200 trees | 1,000 trees | under a minute |

No Pareto/NBD MCMC (231 s for 2×300 on Colab, lead's measurement). All fits `progressbar=False`.

## Stages

| Block | Stage | Minutes |
|---|---|---|
| Block 1 (80) | Open, install, read the scenario | 5 |
| | 1 · What is a new customer worth, by channel? | 30 |
| | 2 · Was the regional campaign incremental? | 30 |
| | *slack* | 15 |
| Lunch (50) | | |
| Block 2 (100) | 3 · Calibrate the MMM with the geo test | 25 |
| | 4 · Next quarter's budget | 20 |
| | 5 · Whom to send the retention offer | 20 |
| | 6 · The five-slide brief | 25 |
| | *slack* | 10 |
| Break (10) | | |

Each stage is one function with a checkpoint (a stage is longer than a lab exercise because pairs also read
the scenario, choose settings and discuss).

### Stage 1 · What is a new customer worth, by channel? (30 minutes)

- **Predict.** Rank the acquisition channels by 36-month new-customer value before fitting.
- **Function.** `clv_by_channel(model, gg_idata, channel_rows, months=36, monthly_rate=0.01, margin=0.30)
  -> xr.DataArray` (chain, draw, channel), as in Module 5 Exercise 1 (acquisition purchase plus discounted
  repeat purchases).
- **Checkpoint.** Each channel's posterior mean within 20% of the truth's `value_by_channel.new_customer` CLV including the first purchase, times the margin
  (BG/NBD approximates
  the Pareto/NBD generator; the Technical Expert confirms on the capstone seed) and the top channel is right.

### Stage 2 · Was the regional campaign incremental? (30 minutes)

- **Predict.** Will the placebo p-value be below 0.1?
- **Function.** `geo_readout(panel, treated_geos, test_start, test_end, base_weekly_spend,
  total_extra_spend) -> dict` with synthetic-control incremental sales, the placebo p-value and the lift-test
  row for Stage 3 (toolkit `sc_weights`, `permutation_pvalue`, `lift_test_row`).
- **Checkpoint.** Incremental sales within 15% of the truth; p-value ≤ 0.1 on the capstone seed; the row
  has `channel, x, delta_x, delta_y, sigma` in weekly units, with `sigma` from the placebo spread (the
  standard deviation of placebo effects).

### Stage 3 · Calibrate the MMM with the geo test (25 minutes)

- **Predict.** Will calibration move the tested channel's ROAS up or down?
- **Function.** `roas_summary(mmm, start_date, end_date, prob=0.94) -> pd.DataFrame` from
  `mmm.incrementality.contribution_over_spend(frequency="all_time", ...)`: mean, 94% HDI and
  P(ROAS > break-even) per channel. (Fit provided, using Stage 2's row.)
- **Checkpoint.** `checks.k_of_K_in_interval`: true ROAS inside the 94% HDI for at least 3 of 4 channels.

### Stage 4 · Next quarter's budget (20 minutes)

- **Predict.** Which channel gains the most budget once long-run customer value counts?
- **Function.** `recommend_budget(curves, clv, new_customers_per_dollar, bounds, weekly_total, margin) ->
  pd.Series`, maximizing Module 10's long-run value with the toolkit `allocate`, using posterior-mean curves
  from Stage 3 and Stage 1's CLV means.
- **Checkpoint.** Bounds and the weekly total hold; under the true parameters the plan's long-run value is
  within 3% of the true optimum (`true_optimal_allocation` with the long-run objective, proposed helper).

### Stage 5 · Whom to send the retention offer (20 minutes)

- **Predict.** What share of customers will clear margin × uplift > cost?
- **Function.** `retention_targets(cate_hat, margin, cost) -> pd.Series[bool]` and the expected profit.
- **Checkpoint.** On held-out data, the true profit of the rule (from `true_cate`) is positive and at least
  the true profit of mailing everyone (the scenario is built so heterogeneity matters; the Technical
  Expert confirms).

### Stage 6 · The five-slide brief (25 minutes)

- **Function.** `brief_numbers(...) -> dict` collecting every number the slides quote (budget by channel,
  expected incremental sales and 94% interval, probability the plan beats the current plan, CLV by channel
  and CAC caps, geo-test result, targeting list size and profit, the main risk). A provided cell renders a
  five-slide outline in markdown from it; pairs copy it into their slide tool.
- **Checkpoint.** All keys present; numbers agree with Stages 1–5 (no hand-typed numbers).
- **Slides (fixed order):** 1 · The recommendation (budget by channel and the offer list, with expected value
  and 94% interval). 2 · What a customer is worth by channel, and the CAC caps. 3 · The evidence that the
  marketing works: the geo test and what it changed in the MMM. 4 · The allocation and its risk (mean versus
  risk-averse plan, probability it beats the current plan). 5 · Whom to target, the main risk to the whole
  recommendation, and the next test to run.

## Decision · The recommendation (in the presentations)

Provided final cell, in the standard order: the **numbers** from `brief_numbers`; the **rules** from Modules
5, 9, 10 and 11, restated; the pair's **recommendation** sentence with its uncertainty and what would change
it. After presentations, a provided cell reveals the truth for every stage and the value of the true optimal
plan, so pairs see how far each stage's error carried. The cell prints the mode.

## Stretch (optional)

Rerun the pipeline with the uncalibrated MMM and quantify how much value calibration added; or swap the
spend-share prior for the default and repeat Stage 4.

## Known pitfalls

- Channel names must agree across the CLV table, the geo test and the MMM (open item).
- Per-period budgets in PyMC-Marketing's optimizer (Module 10) if pairs use it instead of the toolkit.
- Compute: the three fits must fit within the slack; QUICK is the fallback for a slow room and the notebook
  says that QUICK intervals are wider and may flip a close decision.
- Pairs should not "tune toward the truth": the truth is hidden until the end.

## APIs verified (and how)

No new APIs; every call is one verified in briefs 04, 05, 07, 08, 09, 10 and 11 (installed sources and the
Docker image, 2026-10-09).

## Open items for the Technical Expert

- `mktstats.synth.capstone(seed)` (or equivalent) producing one coherent scenario with a shared channel
  vocabulary, a geo-tested media channel whose true lift matches the MMM truth, new customers per dollar by
  channel, and a hidden truth object revealed by the final cell.
- The long-run version of `true_optimal_allocation`.
- Hand the Pedagogy Expert the final slide outline so the rubric matches it.
