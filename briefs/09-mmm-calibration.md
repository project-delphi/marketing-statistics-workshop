# Lab brief · Module 9 · Calibrating and validating an MMM

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert. |
| Notebooks | `labs/src/09-mmm-calibration.py` → `labs/python/09-mmm-calibration.ipynb` (core, decision); `labs/src/09-robyn.R` → `labs/r/09-robyn.ipynb` (short paired notebook, **conditional**: the Robyn spike passed in Docker; the Colab R run is pending) |
| Lab slot | 55 min (`modules.m09.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (Python 30 + R 10; limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Order | Open the **R** notebook first and start its install cell (Robyn plus the Python library Nevergrad through reticulate; Colab time not yet measured). Work the **Python** notebook's Parts A–C while it installs, then the R notebook's Part D, then the Python decision. If the R install fails or is not ready, skip Part D: the Python notebook stands alone and Exercise 5 becomes a stretch. |
| Day | Day 4, second module |

## Question and decision

**Question.** Can we trust the MMM's ROAS for each channel, and what would make it more trustworthy?

**Decision.** Which channel's ROAS is reliable enough to move budget on now, and which channel needs a new
lift test first.

## What this lab fixes

Module 8's MMM recovered the truth on clean synthetic data, but nothing in it guards against confounding:
a channel whose spend follows demand looks effective because sales and spend rise together.

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic MMM with a demand-following channel | `mktstats.synth.mmm(seed=<default>, confounded=True)` (proposed option): the same four channels as Module 8, with `search` spend responding to an unobserved demand shock that also raises sales; truth as in Module 8 plus `lift_tests` (the results a geo test on `search` would have produced: weekly spend level, spend change, incremental sales per week, its standard error, test dates) | everything in Python |
| Robyn `dt_simulated_weekly` (documented by Robyn as "Simulated MMM data") with `dt_prophet_holidays` | bundled with Robyn | R notebook |

## Model fits

Python (each once; QUICK draws=tune=300, FULL 1000; chains=2; nutpie; `progressbar=False`):
1. Uncalibrated MMM (Module 8's specification and priors).
2. Calibrated MMM: `mmm_cal.build_model(X, y)`, then `mmm_cal.add_lift_test_measurements(df_lift_test)`,
   then `mmm_cal.fit(...)`. (The lift test must be added after `build_model`; the method raises otherwise.)
3. Time-slice cross-validation: `cv = TimeSliceCrossValidator(n_init=..., forecast_horizon=8,
   date_column="date", step_size=...)` and `cv.run(X, y, mmm=<unfitted MMM with the calibrated spec>,
   df_lift_test=df_lift_test, lift_test_date_column="date")`, which fits one model per fold. QUICK: 3 folds,
   draws=tune=200; FULL: 4 folds, draws=tune=500. This is the longest cell; the Technical Expert measures it
   on Colab and reduces folds before draws if it exceeds 4 minutes. Estimated 2–3 minutes FULL until a run
   record exists (the lead measured a 2-channel fit at 9.7 s for 2×500 after compile; four channels and four
   folds are more).

R (provided cells, the spike's settings): `robyn_inputs(dt_input = dt_simulated_weekly, dt_holidays =
dt_prophet_holidays, date_var = "DATE", dep_var = "revenue", dep_var_type = "revenue", prophet_vars =
c("trend", "season", "holiday"), prophet_country = "DE", context_vars = c("competitor_sales_B", "events"),
paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"), paid_media_vars = c("tv_S",
"ooh_S", "print_S", "facebook_I", "search_clicks_P"), organic_vars = c("newsletter"), factor_vars =
c("events"), window_start = "2016-01-01", window_end = "2018-12-31", adstock = "geometric")` plus the
demo's hyperparameter ranges, then `robyn_run(InputCollect, iterations = 500, trials = 1)`, `robyn_outputs`
and `robyn_allocator(..., scenario = "max_response")`. The lead measured the whole chain at 89 s on 2 CPUs in
Docker. Robyn's demo recommends `iterations = 2000, trials = 5` for this dataset; the notebook must say in
the header and next to the call that 500 × 1 is far below that, so the results show how Robyn works, not
reliable estimates.

## Parts and exercises

| Notebook · Part | Exercise | Minutes |
|---|---|---|
| Python · A · From a geo test to the likelihood | 1 · A lift-test row in the model's units | 6 |
| | *Run: uncalibrated fit (provided)* | — |
| Python · B · What confounding does | 2 · Which channel is overcredited? | 7 |
| | *Run: calibrated fit (provided)* | — |
| | 3 · Before and after calibration | 7 |
| Python · C · Out of sample | 4 · Time-slice cross-validation scorecard | 10 |
| R · D · Another tool, another answer (conditional) | 5 · Robyn's marginal ROAS | 10 |
| **Exercises** | | **40** |
| Decision (Python notebook) | Move budget now, or test first? | 5 |

### Exercise 1 · A lift-test row in the model's units (6 minutes)

- **Predict.** A geo test ran 8 weeks, raised `search` spend by $80,000 in total from a base of $40,000 a
  week, and produced $200,000 of incremental sales (standard error $60,000). What are Δx and Δy per week?
- **Function.** `lift_test_row(channel, base_weekly_spend, total_extra_spend, total_incremental_sales,
  total_se, weeks) -> dict` with `channel`, `x` (weekly spend level before the test), `delta_x` (weekly
  spend change), `delta_y` (incremental sales per week) and `sigma` (standard error per week, total ÷
  weeks for a sum of weekly effects measured as one total).
- **Checkpoint.** The example above gives x = 40,000, Δx = 10,000, Δy = 25,000, σ = 7,500; the row built from
  `truth.lift_tests` matches the reference. First checkpoint, before any fit.
- **Explain.** PyMC-Marketing compares the lift with the saturation curve: saturation(x + Δx) −
  saturation(x) is conditioned on Δy with the given σ (method docstring); the curve is per period, so the
  test must be converted to weekly units, and the test should be long relative to the adstock's carryover
  for this steady-state comparison to hold. The method scales the row itself; pass original units.

### Exercise 2 · Which channel is overcredited? (7 minutes)

- **Predict.** One channel's spend rises when demand rises. Which one will the uncalibrated MMM overcredit:
  tv, search, social or display?
- **Function.** `bias_table(roas_draws, truth, prob=0.94) -> pd.DataFrame` with per channel the posterior
  mean, 94% HDI, true ROAS, relative bias (mean − truth) / truth and whether the truth is covered.
- **Checkpoint.** For `search` the posterior mean is above the truth and the truth lies outside the 94% HDI
  (the generator is built to produce this; the Technical Expert confirms on the committed seed in both
  modes); for at least 2 of the other 3 channels the truth is covered.
- **Explain.** The model cannot tell "search drives sales" from "demand drives both"; a good in-sample fit
  does not reveal it (Chan & Perry 2017). Point to the spend and sales plot in high-demand weeks.

### Exercise 3 · Before and after calibration (7 minutes)

- **Predict.** Will calibrating on a `search` lift test change only search's ROAS, or other channels' too?
- **Function.** `calibration_effect(roas_before, roas_after, truth, prob=0.94) -> pd.DataFrame` with per
  channel the HDI width before and after, coverage before and after, and the shift in posterior mean.
- **Checkpoint.** After calibration the truth for `search` lies inside its 94% HDI and the HDI is narrower
  than before; the table has all four channels.
- **Explain.** The lift test pins search's curve; the baseline and the other channels then absorb the
  sales search no longer explains, so their estimates move too. Calibration fixes what was tested, not
  everything (Zhang et al. 2024 put the same idea into priors).

### Exercise 4 · Time-slice cross-validation scorecard (10 minutes)

*The cross-validation cell runs first (the longest cell; 2–3 minutes FULL, estimated). Read the Predict
prompt while it runs.*

- **Predict.** Will the 94% predictive interval cover about 94% of held-out weeks, clearly fewer, or clearly
  more?
- **Function.** `cv_scorecard(pred: pd.DataFrame) -> pd.DataFrame` from `cv.summary.predictions(hdi_probs=
  (0.94,))` (long table with `cv, date, split, mean, median`, HDI bounds, `observed`): per fold, on the
  `test` rows, the MAPE of the posterior mean and the share of weeks inside the 94% HDI; plus an overall row.
- **Checkpoint.** Exact on a toy table; on the real table, one row per fold plus overall, coverage in [0, 1],
  MAPE ≥ 0. (No threshold on the result: the scorecard is what you read.)
- **Explain.** Provided after it: `cv.summary.param_stability(var_names=["saturation_beta"])` plotted per
  fold. A channel whose effect jumps between folds is identified by a few weeks of data; trust it less.

### Exercise 5 · Robyn's marginal ROAS (10 minutes, R, conditional)

*Provided before it: the Robyn chain above and the selected model ID.*

- **Predict.** Robyn ran 500 iterations × 1 trial instead of the recommended 2,000 × 5. Will its top models
  agree on which channel has the highest marginal ROAS: yes or no?
- **Function.** `robyn_marginal_roas(InputCollect, OutputCollect, select_model, channel, spend1, spend2)`
  returning (Δ response) ÷ (Δ spend) from two `robyn_response()` calls, the pattern in Robyn's demo
  (`(Response2$sim_mean_response - Response1$sim_mean_response) / (Response2$sim_mean_spend -
  Response1$sim_mean_spend)`), at the channel's mean weekly spend and 10% above it.
- **Checkpoint.** Finite and positive for every paid channel; recomputing with spend1 and spend2 swapped
  gives the same value.
- **Explain.** Robyn is ridge regression with Nevergrad searching hyperparameters; its "uncertainty" is the
  spread across Pareto-optimal models, not a posterior. Both tools need lift tests for the same reason.

### Decision · Move budget now, or test first? (5 minutes, Python notebook)

Provided cell, in the standard order:
- **Number:** per channel, calibrated ROAS mean and 94% HDI, HDI width relative to the mean, the shift
  caused by calibration (Exercise 3) and the parameter-stability flag (Exercise 4).
- **Rule:** "Move budget on a channel's ROAS only if its 94% HDI lies entirely on one side of the break-even
  ROAS (1 ÷ margin) in the calibrated model; otherwise run a lift test on the channel whose HDI straddles
  break-even with the most spend at stake."
- **Recommendation:** the learner's sentence naming the channels to act on and the one to test next, with
  the spend at stake, and what would change it. The cell prints the mode.

## Stretch (optional)

Fit the calibrated model with a second lift test on another channel and repeat Exercise 3. In the briefing
only, not in a lab: Google's Meridian (DECISIONS.md: package conflicts and a GPU recommendation).

## Known pitfalls

- `add_lift_test_measurements` before `build_model` raises; call it once per name (pass `name=` for a
  second call).
- Lift-test rows in weekly units; `sigma` must be positive; `delta_x` must not be 0.
- Cross-validation is the compute hot spot: fewer folds before fewer draws.
- Robyn writes plots and files to disk by default; send outputs to a temporary directory and keep
  `export = FALSE` where the function allows it.

## APIs verified (and how)

- pymc-marketing 1.2.0 installed source, 2026-10-09: `MMM.add_lift_test_measurements(df_lift_test,
  dist=Gamma, name="lift_measurements")` (columns `channel, x, delta_x, delta_y, sigma` plus one column per
  model dim; "The model has to be built before adding the lift tests"; it scales the rows itself through
  `scale_lift_measurements`); `TimeSliceCrossValidator(n_init, forecast_horizon, date_column, step_size=1,
  sampler_config=None)`, `.run(X, y, ..., mmm=, df_lift_test=, lift_test_date_column=, ...)`,
  `cv.summary.predictions(hdi_probs=(0.94,))` (columns listed above) and `param_stability(var_names=...)`.
- Robyn: function names, `robyn_inputs` arguments, the recommended `iterations = 2000, trials = 5` and the
  `robyn_response` marginal-ROAS pattern quoted from `demo/demo.R` on GitHub (WebFetch, 2026-10-09).
  `args(robyn_response)` was not checked in an image (Robyn is not in `mktstats-env:spike`); the Technical
  Expert checks it in the Robyn spike image.

## Open items for the Technical Expert

- `mmm(..., confounded=True)` with a demand shock shared by `search` spend and sales, and `lift_tests` in the
  truth.
- Add the R notebook to `modules.m09.notebooks` once the Colab R run is recorded.
