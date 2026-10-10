# Lab brief · Module 6 · Experiments done right

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written before the lab was built; the lab has since passed on Colab and in the workshop's Docker image (`runs/`; the Readiness page). Review: Pedagogy Expert. |
| Notebook | `labs/src/06-experiments.py` → `labs/python/06-experiments.ipynb` |
| Lab slot | 55 min (`modules.m06.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Day | Day 3, first module |

## Question and decision

**Question.** Did the email campaign cause extra revenue, and how sure are we?

**Decision.** Roll out the email campaign or not: report incremental revenue per customer with a 95%
confidence interval after the SRM check and CUPED, and compare the incremental margin with the cost per
email.

## What this lab fixes

Module 5 set CAC caps and a retention plan assuming marketing causes the purchases credited to it.
Nothing so far measured causation.

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic email experiment (randomized; the retailer's story) | `mktstats.synth.email_experiment(seed=<default>)` (on main: 20,000 rows, Hillstrom-like columns `recency, history_segment, history, mens, womens, zip_code, newbie, channel, treatment, conversion, spend, true_cate`; `history` is pre-period spend); truth `ate` (spend scale), `offer_cost`, `margin`, `base_conversion_control`, `conversion_treated` | power check, CUPED against truth, attribution, decision |
| Hillstrom e-mail experiment (real; 64,000 customers randomized to Mens e-mail, Womens e-mail, No e-mail; two-week outcomes) | `mktstats.data.load_hillstrom()`: `recency, history_segment, history, mens, womens, zip_code, newbie, channel, segment, visit, conversion, spend` | SRM, CUPED on real data (`history` = past-year spend), delta method |

## Model fits

None. Everything is closed-form or simulation; each simulation is vectorized and runs in seconds
(estimate). QUICK reduces simulation counts (stated beside each).

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · Before the test | 1 · Sample size and minimum detectable effect | 7 |
| | 2 · Sample-ratio mismatch | 5 |
| B · During the test | 3 · What peeking does to false positives | 9 |
| C · Reading the test | 4 · CUPED | 7 |
| | 5 · Delta-method standard error for a ratio metric | 7 |
| D · What the test measures | 6 · Attribution is not incrementality | 5 |
| **Exercises** | | **40** |
| Decision | Roll out the email? | 5 |

### Exercise 1 · Sample size and minimum detectable effect (7 minutes)

- **Predict.** To detect a $0.50 lift in spend per customer when spend has a standard deviation of $15
  (α = 0.05 two-sided, 80% power), how many customers per arm: under 10,000, 10,000–100,000, or over
  100,000?
- **Function.** `sample_size(sd, mde, alpha=0.05, power=0.8) -> int` per arm for a two-sample comparison of
  means (normal approximation), and `mde(n_per_arm, sd, alpha=0.05, power=0.8) -> float`.
- **Checkpoint.** `mde(sample_size(sd, d)) ≈ d` to 1%; and by simulation (2,000 experiments, seed shown;
  QUICK 500) at `n = sample_size(15, 0.5)` with a true lift of 0.5, empirical power is within 0.80 ± 0.03
  (±0.03 is about three binomial standard errors for 2,000 runs; QUICK uses ±0.06). This is the first
  checkpoint and runs in seconds.
- **Explain.** Compare with your guess (the formula gives about 14,100 per arm). Halving the MDE quadruples
  the sample, because the standard error falls with √n.

### Exercise 2 · Sample-ratio mismatch (5 minutes)

- **Predict.** Hillstrom assigned one third of customers to each arm. Will the chi-square test of the
  observed arm counts give a p-value above or below 0.05?
- **Function.** `srm_pvalue(counts: np.ndarray, expected_shares: np.ndarray) -> float` (chi-square goodness of
  fit, `scipy.stats.chisquare`).
- **Checkpoint.** Matches the reference on Hillstrom's arm counts; on a provided "broken logging" copy of the
  synthetic experiment (2% of treated rows dropped, seed shown) the p-value is below 0.001.
- **Explain.** An SRM means the randomization or the logging is broken, so the comparison is not
  trustworthy whatever the effect looks like (Fabijan et al. 2019; Kohavi, Tang & Xu 2020). Check it first.

### Exercise 3 · What peeking does to false positives (9 minutes)

- **Predict.** An A/A test (no true effect) is checked 10 times as data arrive and stopped at the first
  p < 0.05. Will the share of A/A tests declared "significant" be about 5%, 10%, 20% or 40%?
- **Function.** `peeking_fpr(n_per_arm, looks, n_sims, alpha, rng) -> float`: simulate A/A tests, run a
  two-sample z-test at `looks` equally spaced interim sample sizes, count a test as positive if any look has
  p < α, and return the share of positives. Vectorize over simulations.
- **Checkpoint.** With `looks=1`, the rate is within 0.05 ± 0.015 (4,000 simulations, about four standard
  errors; QUICK 1,000 and ±0.03); with `looks=10`, it is at least 0.10 and higher than with `looks=1`; the
  rate increases with the number of looks (`checks.monotone` over 1, 2, 5, 10).
- **Explain.** Each look is another chance for noise to cross the line. Fixes: decide the sample size in
  advance, use a group-sequential design with a stricter threshold, or use always-valid inference that
  allows continuous monitoring (Johari et al. 2022, briefing). Stretch: find by simulation the constant
  threshold that keeps 10 looks at 5%.

### Exercise 4 · CUPED (7 minutes)

- **Predict.** The correlation ρ between pre-period spend (`history`) and outcome spend in the synthetic
  experiment is shown above. Will CUPED cut the variance of the estimate by about ρ, by about ρ², or not at all?
- **Function.** `cuped(y, x, treatment) -> dict` with θ = cov(y, x) / var(x) on pooled data, adjusted outcome
  y − θ(x − x̄), and for raw and adjusted: effect (difference in means), standard error, 95% CI, plus the
  variance reduction 1 − var(y_adj)/var(y).
- **Checkpoint.** On the synthetic experiment: the adjusted 95% CI contains the true ATE (truth `ate`); the
  adjusted CI is narrower than the raw CI; the variance reduction equals the squared sample correlation to
  0.01. On Hillstrom (`history` as x, Womens e-mail versus No e-mail), the function runs and the reduction is
  between 0 and 1 (structural check; the real effect is not known).
- **Explain.** CUPED removes the part of the outcome that was predictable before the test, so it cannot bias
  a randomized comparison; the gain is about ρ² (Deng et al. 2013). Point to the two intervals.

### Exercise 5 · Delta-method standard error for a ratio metric (7 minutes)

- **Predict.** For revenue per visit (randomized by customer), will the correct standard error be larger
  or smaller than the naive one that treats every visit as independent?
- **Function.** `ratio_delta_se(num: np.ndarray, den: np.ndarray) -> tuple[float, float]` returning the ratio
  R = Σnum / Σden and its delta-method standard error from per-customer numerators and denominators:
  Var(R) ≈ [Var(n̄) − 2R·Cov(n̄, d̄) + R²·Var(d̄)] / d̄².
- **Checkpoint.** On Hillstrom (spend over visits, one arm), the delta-method SE is within 10% of a provided
  bootstrap SE (2,000 resamples of customers, seed shown; QUICK 500 and 20%); on a toy input with constant
  denominators it equals the ordinary standard error of the mean ratio.
- **Explain.** Visits from the same customer are correlated, so they are not independent observations; the
  delta method accounts for that in closed form (Deng, Knoblich & Lu 2018).

### Exercise 6 · Attribution is not incrementality (5 minutes)

- **Predict.** A last-touch report credits every conversion by an emailed customer to the email. Will that
  be more than, fewer than, or about the conversions the email caused?
- **Function.** `attributed_vs_incremental(df) -> dict` with `attributed` = conversions among treated
  customers (each was last touched by the email), and `incremental` = (conversion rate treated − conversion
  rate control) × number treated, with a 95% CI for `incremental`.
- **Checkpoint.** (`truth.conversion_treated` − `truth.base_conversion_control`) × number treated lies inside
  the CI; `attributed` exceeds the CI's upper bound (on main's truth the treated conversion rate is 0.063
  against 0.036 without the email, so more than half of the credited conversions would have happened
  anyway).
- **Explain.** Attribution counts who was touched before converting; incrementality counts who converted
  because of the touch. Customers who would have bought anyway still get touched, so touch-based credit
  overstates the effect. Large field experiments found observational methods often miss the
  experimental lift (Gordon et al. 2019).

### Decision · Roll out the email? (5 minutes)

Provided cell, in the standard order:
- **Number:** CUPED-adjusted incremental revenue per customer with its 95% CI (synthetic experiment), the
  SRM p-value, and incremental margin per email = margin × incremental revenue − offer cost (truth's
  `margin` and `offer_cost`, shown as assumptions).
- **Rule:** "Roll out if the SRM check passes (p > 0.001) and the lower end of the 95% CI of incremental
  margin per email is above 0." (A stricter rule than "the point estimate is positive"; say why: a rollout
  is hard to reverse.)
- **Recommendation:** the learner's sentence with the money per 100,000 emails and its range, and what would
  change it (a cheaper offer; a segment where the effect is larger: Module 11). Then the cell reveals the
  true ATE. The cell prints the mode.

## Stretch (optional)

Calibrate a constant z threshold for 10 looks by simulation (Exercise 3); run the full readout on Hillstrom
Mens versus Womens e-mail.

## Known pitfalls

- Hillstrom's `visit`, `conversion` and `spend` are two-week outcomes; `history` is the past year, so it is
  a valid pre-period covariate. Do not use post-period variables as CUPED covariates.
- Spend is mostly zeros and heavy-tailed: normal-approximation intervals need large samples (they are large
  here); say so.
- `scipy.stats.chisquare` expects counts, not shares, for both observed and expected.

## APIs verified (and how)

`scipy.stats.chisquare(f_obs, f_exp)` and `scipy.stats.norm` in SciPy 1.16.3 (pinned, Colab version). No
library-specific statistical API beyond these. Hillstrom's arms, outcomes and the data URL were checked on
the MineThatData post (references.qmd, 2026-10-09).

## Open items for the Technical Expert

- Resolved on main: `email_experiment` uses `history` as the pre-period covariate; there is no `visits` or
  `clicked` column, so the delta method uses Hillstrom (`visit`) and attribution uses last touch by the
  email.
- `load_hillstrom()` with lower-case column names as above.
