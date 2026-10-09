# Lab brief · Module 3 · Covariates in CLVTools, and BTYD versus machine learning

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert. |
| Notebooks | `labs/src/03-clvtools-covariates.R` → `labs/r/03-clvtools-covariates.ipynb` (Part A–B); `labs/src/03-btyd-vs-ml.py` → `labs/python/03-btyd-vs-ml.ipynb` (Part C) |
| Lab slot | 55 min for both notebooks (`modules.m03.minutes.lab`). Planned: 27 (R) + 25 (Python) + 3 decision = 55 min. |
| Day | Day 1, after lunch |

## Question and decision

**Question.** Do customers from different acquisition channels buy and leave at different rates, and does
a flexible machine-learning model forecast next-period purchases better than a BTYD model?

**Decision.** Whether channels differ enough to be treated differently (this feeds the CAC caps in
Module 5), and which method to use for next quarter's purchase forecast, based on the holdout error you
measured. The notebook must not assert a winner; it reports what the comparison showed on these data.

## What this lab fixes

Module 2 treats all customers as one population and never compared BTYD with a machine-learning
forecaster.

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic retailer (Pareto/NBD with channel effects on purchase and dropout, note-019 form) | R: committed CSVs via `R/mktstats.R` (transactions, customers, `truth.json`); Python: `mktstats.synth.retailer(seed=<default>)` | covariate recovery, ML comparison |
| CDNOW (real) | R: `data("cdnow", package = "CLVTools")` (2,357 customers, 6,696 rows, same-day purchases already combined); Python: `mktstats.data.load_cdnow()` | comparison with the Day 1 Python fit; ML comparison |

Covariate parameterization (Fader & Hardie note 019, cited by CLVTools' `pnbd` documentation and used by
PyMC-Marketing's `ParetoNBDModel`): α_i = α·exp(−γ_trans·z_i) and β_i = β·exp(−γ_life·z_i), so a positive
coefficient raises the purchase or dropout rate. The generator's `truth.covariates` must use this sign
convention and dummy coding with the first channel as the reference level (CLVTools names the
coefficients `trans.channelB`, `life.channelB`, … for a character column `channel`).

## Model fits

| Notebook | Fit | Settings | Timing we have |
|---|---|---|---|
| R | `pnbd()` on CDNOW, no covariates | MLE, default optimizer | — |
| R | `pnbd()` on the retailer with static covariates | MLE, default optimizer; fall back to `optimx.args = list(method = "Nelder-Mead")` if the fit returns NA | 1.1 s for 4,000 simulated customers, 23,554 transactions (Docker arm64 on a laptop, 2026-10-09; not Colab) |
| Python | `ParetoNBDModel` MAP on calibration data | flat priors | 4.3 s on CDNOW (laptop, 2026-10-09) |
| Python | `HistGradientBoostingRegressor(loss="poisson")` | defaults, `random_state` fixed | — |

## Parts and exercises

| Notebook · Part | Exercise | Minutes |
|---|---|---|
| R · A · CLVTools on the Day 1 data | 1 · Build the clv.data object and compare with Day 1 | 9 |
| R · B · Channel as a covariate | 2 · Fit the Pareto/NBD with the channel covariate | 10 |
| | 3 · Translate coefficients into rates | 8 |
| Python · C · BTYD versus gradient boosting | 4 · Features and a target without leakage | 9 |
| | 5 · A holdout scorecard | 8 |
| | 6 · Where does each method win? | 8 |
| Decision | Treat channels differently? Which forecaster? | 3 |
| **Total** | | **55** |

### Exercise 1 · Build the clv.data object and compare with Day 1 (9 minutes)

- **Predict.** Will CLVTools' Pareto/NBD on CDNOW give the same estimates as PyMC-Marketing's on Day 1?
  Which parameters might differ, and why?
- **Function.** `make_clvdata(tx, split)` returning `clvdata(tx, date.format = "ymd", time.unit = "week",
  estimation.split = split, name.id = "Id", name.date = "Date", name.price = "Price")`; then a provided
  `pnbd()` fit, and `compare_with_day1(fit, day1)` returning a data frame of parameter, CLVTools,
  PyMC-Marketing, relative difference.
- **Checkpoint.** `summary()` of the object reports 1,411 zero repeaters with `split = "1997-09-30"`
  (pass a date so the estimation period ends where Day 1's did; `split = 39` ends on 1997-10-01). All four
  parameters within 1% of the Day 1 values (Day 1 flat-prior MAP measured 2026-10-09: r = 0.553,
  α = 10.578, s = 0.606, β = 11.668; CLVTools with the date split gave 0.553, 10.578, 0.606, 11.664 the
  same day). The Technical Expert embeds the values from the recorded Module 2 run. This checkpoint runs
  before the covariate fit.
- **Explain.** Two independent implementations (maximum likelihood in R, flat-prior MAP in PyMC) agree
  when they see the same data. One extra day of data changes the dropout parameters most: with
  `split = 39` CLVTools gave s = 0.625, β = 12.244 against 0.606, 11.664, because dropout is never
  observed and the likelihood is flat in that direction.

### Exercise 2 · Fit the Pareto/NBD with the channel covariate (10 minutes)

- **Predict.** Which acquisition channel brings customers who buy more often but leave sooner?
- **Function.** `fit_channel_pnbd(clv, customers)`: `SetStaticCovariates(clv, data.cov.life = cov,
  data.cov.trans = cov, names.cov.life = "channel", names.cov.trans = "channel", name.id = "Id")` with
  `cov` holding `Id` and `channel`, then `pnbd()`. Return the fitted object.
- **Checkpoint.** For the 2(K − 1) channel coefficients, the truth lies inside `confint(fit, level = 0.9)`
  for at least 2(K − 1) − 1 of them (one miss allowed: six 90% intervals all cover only about 53% of the
  time). The Technical Expert confirms on the committed seed. Also: no NA coefficients (recovery message
  suggests Nelder-Mead).
- **Explain.** A covariate explains part of the heterogeneity that the gamma distributions otherwise
  absorb; the population parameters (r, α, s, β) now describe the reference channel.

### Exercise 3 · Translate coefficients into rates (8 minutes)

- **Predict.** A purchase coefficient of 0.3: by what factor does it change the average purchase rate?
- **Function.** `channel_rates(fit, channels)` returning per channel the mean purchase rate r/α_c and mean
  dropout rate s/β_c (per week) and the implied mean lifetime 1/(s/β_c) weeks (state that this is the
  rate at the population mean, not the mean of individual lifetimes).
- **Checkpoint.** Each channel's mean purchase rate within 10% of the true value computed from
  `truth.json` (`checks`-style relative tolerance with a message naming the channel).
- **Explain.** exp(γ) is a multiplicative effect, like a log-linear regression coefficient. This table is
  the input to the CAC caps in Module 5.

### Exercise 4 · Features and a target without leakage (9 minutes)

- **Predict.** If you build features from the whole log, how much will the machine-learning model's
  holdout error improve, and why is that improvement fake?
- **Function.** `make_features(tx, customers, cutoff) -> pd.DataFrame` using only transactions on or
  before `cutoff`: frequency, recency, T, mean spend, weeks since last purchase, purchases in the last 4
  and 13 weeks, channel dummies. Supervised design: train on features at `cutoff − H` with the target =
  purchases in (cutoff − H, cutoff]; predict with features at `cutoff` for the holdout window of length H.
- **Checkpoint.** Leakage test: features are identical when every transaction after `cutoff` is deleted
  or shuffled (`pd.testing.assert_frame_equal`), and the training target window ends at `cutoff`.
- **Explain.** BTYD needs no labelled training window; the machine-learning model does, which costs you
  H weeks of history.

### Exercise 5 · A holdout scorecard (8 minutes)

*Provided before this exercise: Pareto/NBD MAP (with channel covariates via `model_config`
`purchase_covariate_cols` and `dropout_covariate_cols` for the retailer; without for CDNOW) and the
gradient-boosting fit, each predicting holdout purchases.*

- **Predict.** Which method will have the lower error for customers with zero calibration purchases? For
  the heaviest buyers?
- **Function.** `holdout_scorecard(y_true, preds: dict[str, np.ndarray], groups) -> pd.DataFrame` with,
  per method and group (calibration frequency 0, 1, 2–3, 4–7, 8+), MAE, RMSE and total bias (Σ predicted −
  Σ actual).
- **Checkpoint.** On a fixed toy input with known answers the function returns the exact numbers; on the
  real inputs, the table has every method × group and the totals equal the sums. The checkpoint does
  **not** test which method wins.
- **Explain.** Read the table you got. State the result as measured, with the dataset and the split.

### Exercise 6 · Where does each method win? (8 minutes)

- **Predict.** What can the BTYD model tell you that gradient boosting cannot, even if gradient boosting
  has a lower error?
- **Function.** `bootstrap_difference(y_true, pred_a, pred_b, n_boot=1000, rng) -> dict` with the mean
  difference in absolute error between methods and a 95% bootstrap interval (resampling customers).
- **Checkpoint.** On a toy input where method A is exactly 1 better per customer, the mean difference is
  1 and the interval is [1, 1]; interval ordering low ≤ mean ≤ high on real data.
- **Explain.** BTYD gives P(alive), forecasts for any horizon and new customers, and parameters with
  meaning; gradient boosting uses any feature you have but forecasts only the horizon it was trained on.
  The literature has found simple heuristics competitive for some tasks (Wübben & von Wangenheim 2008),
  which is why we measure rather than assume.

### Decision · Treat channels differently? Which forecaster? (3 minutes)

Provided cell shows the Exercise 3 channel table and the Exercise 6 interval for both datasets. The
learner writes two sentences: whether channel differences are large enough to matter for acquisition
spending (Module 5 will price them) and which forecaster they would deploy, citing the interval.

## Provided scaffolding

Loaders; Pareto/NBD MAP and gradient-boosting fits; plotting of the scorecard. R: the plain `pnbd()` fit
in Exercise 1 and `predict(fit)` output (columns include `PAlive`, `CET`, `DERT`, `predicted.CLV`).

## Stretch (optional)

R: `predict(fit, uncertainty = "boots")` for bootstrap intervals on CET. Python: add the BTYD forecast as a
feature to gradient boosting and re-score.

## Known pitfalls

- CLVTools `pnbd()` can return NA coefficients with the default optimizer on some data (it did on the
  package's apparel example without covariates in our image, 2026-10-09); the message recommends
  `optimx.args = list(method = "Nelder-Mead")`.
- Character covariates are dummy-coded with the first level as reference; check the level order.
- `estimation.split` as a number counts periods from the first transaction; pass a date to match Day 1.
- LightGBM fails to load on macOS without `libomp` (seen in our local environment); use scikit-learn's
  `HistGradientBoostingRegressor`, which needs nothing extra.
- Holdout windows of different lengths for different customers bias the scorecard; use one window.

## APIs verified (and how)

- R, in the image `mktstats-env:spike` (R 4.6.1, CLVTools 0.12.1), 2026-10-09: `args(clvdata)`,
  `args(SetStaticCovariates)`, `args(pnbd)`, `predict` method arguments; ran `pnbd` with and without
  static covariates on a simulated 4,000-customer panel (all eight parameters inside 90% intervals) and on
  `data(cdnow)` with `estimation.split = "1997-09-30"` (a date string is accepted; 1,411 zero repeaters);
  read the `pnbd` help page (static covariates, reference to note 019).
- Python, installed source: `ParetoNBDModel` `default_model_config` keys `purchase_covariate_cols`,
  `dropout_covariate_cols`, `purchase_coefficient`, `dropout_coefficient`; covariates enter as
  `alpha_scale * exp(-dot(purchase_data, purchase_coefficient))` (pareto_nbd.py).
  `HistGradientBoostingRegressor(loss="poisson")` in scikit-learn 1.6.1.

## Open items for the Technical Expert

- Make sure `truth.covariates` uses the note-019 sign and reference-level coding shown above.
- Embed the Day 1 CDNOW Pareto/NBD values from the recorded Module 2 run for Exercise 1.
