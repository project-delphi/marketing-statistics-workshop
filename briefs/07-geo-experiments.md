# Lab brief · Module 7 · Geo-experiments and quasi-experiments

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert. |
| Notebooks | `labs/src/07-synthetic-control-did.py` → `labs/python/07-synthetic-control-did.ipynb` (Parts A–B, decision); `labs/src/07-causalimpact-geolift.R` → `labs/r/07-causalimpact-geolift.ipynb` (Part C in the lab; Part D in the clinic) |
| Lab slot | 55 min for both notebooks (`modules.m07.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (Python 28 + R 12; limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Order | Open the **R** notebook first and run its install cell (CausalImpact + GeoLift ≈ 90 s on Colab, lead's measurement 2026-10-09). While it installs, open the **Python** notebook, run its install (≈ 31 s + 17 s import) and work Parts A–B. Then do the R notebook's Part C. Return to the Python notebook for the decision. |
| Day | Day 3: briefing before lunch, lab after lunch, then the 60-minute geo-test design clinic, which uses the R notebook's Part D (GeoLift market selection) |

## Question and decision

**Question.** The retailer ran a regional campaign in some geos. Did it raise sales, and by how much?

**Decision.** Was the campaign incremental, and by how much? Report incremental sales with an interval and
the implied return on the campaign spend.

## What this lab fixes

Module 6 needs randomized individuals; a regional or broadcast campaign can only be switched on for whole
regions.

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic geo panel (the retailer's regional campaign) | Python `mktstats.synth.geo_panel(seed=<default>)`; R: committed CSV via `R/mktstats.R`; columns `geo, week, sales`; truth `lift_pct`, `incremental_sales`, `treated_geos`, `test_start`, `test_end`, campaign cost | everything |
| Proposition 99 panel (real; California and 38 control states, 1970–2000) | `mktstats.data.load_prop99()`, from the tidysynth `smoking` data: `state, year, cigsale` (+ predictors) | synthetic control on real data (provided cell) |
| GeoLift's bundled example (`data(GeoLift_PreTest)`: 40 locations × 90 days) | GeoLift package | clinic only |

Geo panel requirements (from GeoLift's DESCRIPTION: at least 25 pre-treatment periods and more than 20
geos): at least 40 geos and at least 52 pre-period weeks, about 6 treated geos, an 8-week test. The
generator should make parallel trends hold up to noise, so difference-in-differences is unbiased by
construction, and include one geo-specific trend so synthetic control has something to fix.

## Model fits

- Python: no sampler. Synthetic-control weights are a small constrained least-squares problem
  (`scipy.optimize.minimize(method="SLSQP")`); placebos refit it once per control geo (about 40 small fits,
  seconds; estimate); power simulation reuses difference-in-differences (vectorized).
- R: `CausalImpact(data, pre.period, post.period, model.args = list(niter = N))` with N = 1,000 QUICK and
  5,000 FULL (bsts MCMC). A 100-point example with `niter = 1000` ran in our Docker image (2026-10-09,
  not timed); Colab not yet measured.

## Parts and exercises

| Notebook · Part | Exercise | Minutes |
|---|---|---|
| Python · A · Two estimators | 1 · Difference-in-differences | 7 |
| | 2 · Synthetic-control weights | 9 |
| | *Run: the same estimator on Proposition 99 (provided)* | — |
| Python · B · Is it real, and could we have seen it? | 3 · In-space placebo test | 6 |
| | 4 · Power by simulation | 6 |
| R · C · A Bayesian structural time-series view | 5 · CausalImpact on the panel | 7 |
| | 6 · What contaminated controls do | 5 |
| **Exercises** | | **40** |
| Decision (Python notebook) | Was the campaign incremental? | 5 |
| R · D (clinic, not in lab minutes) | *GeoLift market selection and power (provided)* | — |

### Exercise 1 · Difference-in-differences (7 minutes)

- **Predict.** A provided plot shows treated and control geos' average weekly sales. Will the naive
  before/after change in treated geos be larger or smaller than the difference-in-differences estimate?
- **Function.** `did(panel, treated_geos, test_start, test_end) -> dict` with the effect per treated geo per
  week, the total incremental sales over the test, and the naive before/after change.
- **Checkpoint.** Total incremental sales within 10% of `truth.incremental_sales` (parallel trends hold by
  construction; the Technical Expert confirms the reference passes on the committed seed in both modes).
  First checkpoint; runs in milliseconds.
- **Explain.** The control geos' change is the counterfactual change for the treated geos; that only works
  if both would have moved in parallel without the campaign. Point to the pre-period lines.

### Exercise 2 · Synthetic-control weights (9 minutes)

- **Predict.** Will the synthetic control put weight on most control geos, or on a handful?
- **Function.** `sc_weights(y_pre: np.ndarray, X_pre: np.ndarray) -> np.ndarray`: weights w ≥ 0 with
  Σw = 1 minimizing ‖y_pre − X_pre·w‖² over the pre-period, with `scipy.optimize.minimize(method="SLSQP")`,
  bounds (0, 1) and an equality constraint. `y_pre` is the treated geos' total, `X_pre` the control geos.
- **Checkpoint.** Weights non-negative and summing to 1 (to 1e-6); pre-period RMSPE no worse than 1.05 ×
  the reference; incremental sales from the provided `sc_effect(weights, ...)` within 10% of truth.
- **Explain.** The weights build a comparison unit that tracks the treated geos before the campaign
  (Abadie, Diamond & Hainmueller 2010). Sparse weights are normal: a few similar geos explain the treated
  series.

*Provided after this exercise:* `sc_weights` applied to Proposition 99 (California versus 38 states,
1970–1988 pre-period, 1989–2000 after), with the gap plotted. Sentence before it: "Look at the sign and
size of the gap after 1988; Abadie et al. report about 26 fewer packs per capita by 2000 with
covariate-matched weights; this outcome-only version need not match exactly." Check: the post-1988 gap is
negative.

### Exercise 3 · In-space placebo test (6 minutes)

- **Predict.** If the campaign had no effect, what rank would the treated unit's post/pre fit ratio have
  among the 41 units: near the top, in the middle, or near the bottom?
- **Function.** `permutation_pvalue(treated_stat: float, placebo_stats: np.ndarray) -> float` =
  (1 + #{placebo ≥ treated}) / (1 + number of placebos). Provided: the loop that refits `sc_weights` with each
  control geo as a fake treated unit and computes its post/pre RMSPE ratio.
- **Checkpoint.** Correct on toy inputs (treated largest → 1/(J + 1)); on the panel the p-value equals the
  reference; on a provided copy of the panel with the lift removed, p > 0.1.
- **Explain.** With one treated unit there is no sampling distribution; the placebos supply one. The
  smallest p-value possible is 1/(J + 1), so 40 control geos limit you to p ≥ 0.024.

### Exercise 4 · Power by simulation (6 minutes)

- **Predict.** With these geos and an 8-week test, what is the chance of detecting a true 5% lift: under
  50%, 50 to 80%, or over 80%?
- **Function.** `geo_power(pre_panel, treated_geos, lift_pct, n_sims, rng) -> float`: in the pre-period only,
  pick random 8-week windows, inject `lift_pct` into the treated geos, estimate with `did`, and return the
  share of windows whose estimate exceeds the 95th percentile of estimates with no injected lift (computed
  in the same function from the same windows).
- **Checkpoint.** Power at lift 0 is between 0 and 0.15 (should be about 0.05); power increases with lift
  (`checks.monotone` over 0, 2, 5, 10%); 200 simulations FULL, 50 QUICK (seed shown).
- **Explain.** Power depends on how noisy the geos are relative to the lift and on the test length; this is
  what market selection in the clinic optimizes.

### Exercise 5 · CausalImpact on the panel (7 minutes, R)

- **Predict.** Will CausalImpact's 95% interval for cumulative incremental sales contain the truth: yes or
  no? Will it be wider or narrower than ±10% of the estimate?
- **Function.** `run_causal_impact(panel, treated_geos, test_start, test_end, niter)` building a matrix whose
  first column is the treated geos' total and whose other columns are the control geos, calling
  `CausalImpact(data, pre.period, post.period, model.args = list(niter = niter))`, and returning
  `impact$summary["Cumulative", c("AbsEffect", "AbsEffect.lower", "AbsEffect.upper")]`.
- **Checkpoint.** `truth$incremental_sales` lies inside [AbsEffect.lower, AbsEffect.upper] (95% posterior
  interval, CausalImpact's default `alpha = 0.05`, stated explicitly in the call).
- **Explain.** CausalImpact is a Bayesian synthetic control over time (Brodersen et al. 2015): a regression
  on the controls plus a local trend, forecast into the test window.

### Exercise 6 · What contaminated controls do (5 minutes, R)

- **Predict.** If one treated geo were wrongly included among the controls, would the estimated lift be
  larger, smaller or unchanged?
- **Function.** `contaminated_effect(panel, treated_geos, test_start, test_end, niter)`: rerun Exercise 5
  with the first treated geo moved into the controls; return the cumulative AbsEffect.
- **Checkpoint.** The contaminated estimate is smaller than the clean one.
- **Explain.** CausalImpact's documentation states its first assumption: the controls are not affected by
  the intervention. Spillover (shoppers crossing regions, national ads) violates it the same way.

### Decision · Was the campaign incremental? (5 minutes, Python notebook)

Provided cell, in the standard order:
- **Number:** synthetic-control incremental sales with the placebo p-value, difference-in-differences
  estimate, and (optional) the CausalImpact 95% interval typed into a provided variable from the R notebook;
  incremental margin = gross margin (30%, stated) × incremental sales; campaign cost from the truth's
  assumptions.
- **Rule:** "Call the campaign incremental if the placebo p-value is below 0.1 and the CausalImpact interval
  (if entered) excludes 0; call it profitable if incremental margin exceeds campaign cost at the lower end
  of the interval."
- **Recommendation:** the learner's sentence with incremental sales, return on spend and range, and what
  would change it (spillover; a longer test). The cell prints the mode.

## Part D · GeoLift market selection (clinic; provided cells, not timed in the lab)

R notebook, used by the Pedagogy Expert's geo-test design clinic:
`GeoDataRead(data = GeoLift_PreTest, date_id = "date", location_id = "location", Y_id = "Y", format =
"yyyy-mm-dd")`, then `GeoLiftMarketSelection(data, treatment_periods = c(15), N = c(2, 3), effect_size =
seq(0, 0.25, 0.05), lookback_window = 1, cpic = 7.5, budget = 100000, alpha = 0.1, Correlations = TRUE,
fixed_effects = TRUE, side_of_test = "one_sided", parallel = FALSE, print = FALSE)`, which returned
`BestMarkets`, `PowerCurves` and `parameters` in 19 s (Docker arm64 on a laptop, sequential, 2026-10-09; not
a Colab time). Keep these small settings. Optional: `GeoLiftPower` for the chosen markets.

## Stretch (optional)

Augmented synthetic control with `augsynth` on the panel (R); event-study plot of weekly
difference-in-differences effects with pre-period leads (Python).

## Known pitfalls

- Aggregate the treated geos before fitting the synthetic control (or fit one per geo and sum); do not mix.
- Use only pre-period data for weights and for the power simulation.
- CausalImpact needs the response in the first column; the controls must not be affected by the campaign.
- GeoLift needs `knitr` at install time: in our image, installing the pinned GeoLift archive failed with
  "dependency 'knitr' is not available" until `knitr` was installed (2026-10-09). Add it to
  `environment/r-packages.txt` and the Colab install cell.
- The tidysynth manual labels the unit of `cigsale` differently from the paper's "per-capita packs"; state
  the unit you use.

## APIs verified (and how)

- R, image `mktstats-env:spike` (CausalImpact 1.4.1, bsts 0.9.11): `args(CausalImpact)`; ran an example and
  read `names(impact$summary)` (`AbsEffect`, `AbsEffect.lower`, `AbsEffect.upper`, …, rows `Average`,
  `Cumulative`).
- GeoLift 2.7.5 and augsynth 0.2.0 installed from the pinned archives in a throwaway container (with `knitr`
  added): `args(GeoDataRead)`, `args(GeoLiftMarketSelection)`, `args(GeoLiftPower)`, `args(GeoLift)`,
  `args(augsynth)`; ran the market selection above.
- tidysynth `smoking`: 1,209 rows, columns `state, year, cigsale, lnincome, beer, age15to24, retprice`
  (read in the same container).
- Python: `scipy.optimize.minimize(method="SLSQP")` with bounds and equality constraints (SciPy docs,
  references.qmd).

## Open items for the Technical Expert

- `geo_panel` as specified above, with campaign cost in the truth.
- `load_prop99()` from the tidysynth data (check its licence before mirroring).
- Add `knitr` to the R package list for GeoLift.
