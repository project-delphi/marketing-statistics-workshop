# Lab brief · Module 2 · Buy-till-you-die models: BG/NBD and Pareto/NBD

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written, not run. Review: Pedagogy Expert. |
| Notebook | `labs/src/02-btyd.py` → `labs/python/02-btyd.ipynb` |
| Lab slot | 55 min (`modules.m02.minutes.lab`). Planned: 52 min. |
| Day | Day 1, second module |

## Question and decision

**Question.** Which customers are still with us, and how many purchases will they make?

**Decision.** Stop retargeting customers whose P(alive) is below τ = c / (m × u), where c is the
contact cost, m the margin per purchase and u the assumed probability that the ad causes a purchase
from an alive customer. Report how many customers and dollars that affects.

## What this lab fixes

Module 1's recency rule needs an arbitrary cutoff, gives no probability, cannot forecast, and drops
customers who are still alive.

## Notation (used again in Modules 3–5)

x = repeat purchases (frequency), t_x = time of the last purchase since the first (recency), T = time
since the first purchase (age), all in weeks. BG/NBD: purchase rate λ ~ Gamma(r, α) with α a **rate**
(numpy scale = 1/α); after each purchase the customer drops out with probability p ~ Beta(a, b).
Pareto/NBD: λ ~ Gamma(r, α), dropout rate μ ~ Gamma(s, β).

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic BG/NBD customers | `mktstats.synth.btyd_bgnbd(seed=<default>)`; truth {r, alpha, a, b} | recovery |
| CDNOW (real) | `mktstats.data.load_cdnow()` | benchmark, holdout, decision |

CDNOW split: calibration to **1997-09-30** (39 weeks, as in Fader, Hardie & Lee's note 004), holdout
1997-10-01 to 1998-06-30 (39 weeks). RFM with `rfm_summary(..., time_unit="D", time_scaler=7)` (see
Module 1, Exercise 3) or with the learner's own Module 1 SQL.

## Model fits (once each, all in provided cells)

| Fit | Method | QUICK | FULL | Laptop timing we have |
|---|---|---|---|---|
| BG/NBD, synthetic | MAP (default priors) | same | same | — |
| BG/NBD, synthetic | MCMC, `nuts_sampler="nutpie"` | chains=2, draws=300, tune=300 | chains=2, draws=1000, tune=1000 | — |
| BG/NBD, CDNOW | MAP with flat priors (`Prior("HalfFlat")` for r, alpha, a, b) | same | same | 3 s (2026-10-09) |
| BG/NBD, CDNOW | MCMC nutpie, default priors | chains=2, draws=300, tune=300 | chains=2, draws=1000, tune=1000 | 3.1 s for 2×500 (spike S2 record) |
| Pareto/NBD, CDNOW | **MAP only** (flat priors) | same | same | 4.3 s (2026-10-09); MCMC 2×300 took 57 s in spike S2, so no MCMC here |

All laptop timings are a 10-core Apple Silicon laptop (native environment), **not Colab** (2 vCPUs). Use
`chains=2` because Colab has 2 vCPUs. Call: `model.fit(data=rfm, method="mcmc", nuts_sampler="nutpie",
chains=2, draws=..., tune=..., random_seed=...)` (extra keywords go to the sampler).

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · The generative story | 1 · Simulate BG/NBD customers | 8 |
| | 2 · Sufficient statistics and the calibration/holdout split | 8 |
| B · Fit and check recovery | 3 · Parameter recovery table | 8 |
| | 4 · MAP versus MCMC | 6 |
| C · Real data: CDNOW | 5 · Benchmark and holdout tracking | 10 |
| D · Decision | 6 · A P(alive) threshold for retargeting | 8 |
| Decision | Whom do we stop retargeting? | 4 |
| **Total** | | **52** |

### Exercise 1 · Simulate BG/NBD customers (8 minutes)

- **Predict.** If the dropout probability p after each purchase is high, what happens to a customer's
  frequency and recency?
- **Function.** `simulate_bgnbd(r, alpha, a, b, T, n, rng) -> pd.DataFrame` with `frequency, recency, T,
  alive`: draw λ_i ~ Gamma(shape r, rate α), p_i ~ Beta(a, b); purchases arrive as a Poisson process;
  after each purchase the customer leaves with probability p_i; observe up to T.
- **Checkpoint.** For n = 20,000 at the true parameters and T = 52, mean frequency within 3% of the
  closed-form expected number of purchases for a new customer, E[X(52)] (Fader, Hardie & Lee 2005), or
  equivalently `BetaGeoModel.expected_purchases_new_customer(t=52)` evaluated at the true parameters
  (`checks.close(..., rel=0.03)`); `checks.rfm_table` on the output. This checkpoint runs before any fit.
- **Explain.** What r, α, a, b each control. Note the BG/NBD quirk: a customer can only leave right after
  a purchase, so a customer with x = 0 is always alive in this model.

### Exercise 2 · Sufficient statistics and the calibration/holdout split (8 minutes)

- **Predict.** If you swapped recency and T, which customers would look most alive?
- **Function.** `calibration_holdout(tx, cal_end: str, holdout_end: str) -> pd.DataFrame` with
  `customer_id, frequency, recency, T, holdout_purchases` (purchase days in the holdout window), weeks
  as days/7.
- **Checkpoint.** On CDNOW: 2,357 rows; exactly 1,411 with frequency 0; `checks.rfm_table`; total
  holdout purchases equal the reference. On the synthetic data: equals the generator's own summary.
- **Explain.** Why the split must be by time, not by customer, and why T differs between customers
  (different first-purchase dates).

### Exercise 3 · Parameter recovery table (8 minutes)

*Provided before this exercise: the two synthetic fits (MAP and MCMC).*

- **Predict.** Which parameter will have the widest 94% HDI relative to its value?
- **Function.** `recovery_table(idata, truth: dict, prob: float = 0.94) -> pd.DataFrame` with `param,
  truth, mean, hdi_low, hdi_high, covered`. Use `az.hdi(idata, prob=prob)` on the posterior (or flatten
  draws to 1-D first; a raw (chain, draw) array is summarized per chain).
- **Checkpoint.** `checks.k_of_K_in_interval` with k = 3 of K = 4 parameters inside the 94% HDI. (With
  four independent 94% intervals, all four cover only about 78% of the time; requiring 3 of 4 keeps a
  correct solution from failing on an unlucky seed. The Technical Expert confirms coverage on the
  committed seed in both QUICK and FULL.)
- **Explain.** a and b (the dropout process) are less well identified than r and α: dropout is never
  observed directly.

### Exercise 4 · MAP versus MCMC (6 minutes)

- **Predict.** Will the MAP estimate equal the posterior mean?
- **Function.** `compare_map_mcmc(map_idata, mcmc_idata) -> pd.DataFrame` with the relative difference
  between MAP and posterior mean per parameter.
- **Checkpoint.** Every MAP value lies inside the MCMC 94% HDI (`checks.in_interval`, one call per
  parameter).
- **Explain.** MAP is fast and good for point forecasts; it gives no uncertainty, which matters for
  customers near the decision threshold.

### Exercise 5 · Benchmark and holdout tracking (10 minutes)

*Provided before this exercise: the CDNOW fits (flat-prior MAP BG/NBD, MCMC BG/NBD, flat-prior MAP
Pareto/NBD).*

- **Predict.** Will the model over- or under-predict holdout purchases for customers with many
  calibration purchases?
- **Function.** `holdout_comparison(model, cal: pd.DataFrame, holdout_weeks: float) -> pd.DataFrame`
  grouping customers by calibration frequency (0, 1, …, 6, 7+) with mean actual and mean predicted holdout
  purchases, predicted by `model.expected_purchases(data=cal, future_t=holdout_weeks)` (posterior mean).
- **Checkpoint (two).**
  1. Benchmark: the flat-prior MAP BG/NBD on CDNOW reproduces the published maximum-likelihood estimates
     r = 0.243, α = 4.414, a = 0.793, b = 2.426 (Fader, Hardie & Lee, note 004) within 1%
     (`checks.close(rel=0.01)` per parameter). Reproduced on 2026-10-09 with PyMC-Marketing 1.2.0. If the
     RFM table is wrong in any way, this fails.
  2. Total predicted holdout purchases within a tolerance of actual set by the Technical Expert from the
     reference run (no looser than 15%); report BG/NBD and Pareto/NBD side by side.
- **Explain.** Default priors pull a and b away from the MLE on CDNOW (default-prior MAP gave a = 0.706,
  b = 2.092 on 2026-10-09); with 2,357 customers the prior still matters for the dropout parameters.
  BG/NBD and Pareto/NBD forecast similarly (Fader, Hardie & Lee 2005). Provided plot:
  `pymc_marketing.clv.plot_expected_purchases_over_time` (cumulative tracking plot over the holdout).

### Exercise 6 · A P(alive) threshold for retargeting (8 minutes)

- **Predict.** Contact cost $0.50, margin $20 per purchase, and 5% of alive customers buy because of the
  ad. What threshold τ do you get?
- **Function.** `retarget_threshold(contact_cost, margin, response_rate) -> float` (τ = c / (m × u)) and
  `retarget_flags(p_alive: pd.Series, tau: float) -> pd.Series[bool]` (True = keep retargeting).
- **Checkpoint.** τ arithmetic on three cases; on the synthetic data, customers flagged "stop" have a
  lower true-alive share than those kept, and the share of truly alive customers among "stop" equals the
  reference (needs per-customer `true_alive` in `btyd_bgnbd`, see open items).
- **Explain.** u is an assumption, not a measurement; Day 3 measures it with an experiment. For CDNOW the
  decision uses the **Pareto/NBD** P(alive), because BG/NBD gives P(alive) = 1 to every customer with no
  repeat purchase (checked 2026-10-09 with `expected_probability_alive`), which would never stop them.

### Decision · Whom do we stop retargeting? (4 minutes)

Provided cell for CDNOW at the end of calibration: Pareto/NBD P(alive) per customer
(`ParetoNBDModel.expected_probability_alive(data=cal)`), the τ from Exercise 6, and a table: customers
stopped, retargeting cost saved, expected margin forgone (Σ P(alive) × m × u over stopped customers),
with u at 2%, 5% and 10%. Rule in markdown: "Stop retargeting when P(alive) < τ; the decision is only as
good as u, which we have not measured yet."

## Stretch (optional)

`plot_probability_alive_matrix` for BG/NBD and Pareto/NBD side by side; Pareto/NBD MCMC with
`method="demz"` and a timer, compared with MAP.

## Known pitfalls

- Gamma rate versus scale (Exercise 1).
- `rfm_summary(time_unit="W")` merges same-week purchases; use days/7.
- BG/NBD P(alive | x = 0) = 1 by construction.
- `expected_purchases` returns an xarray with chain/draw dims (MAP: one draw); take the posterior mean
  explicitly.
- ArviZ 1.x default intervals are 89% ETI: always pass `prob=0.94` and the interval kind.

## APIs verified (and how)

Installed source, pymc-marketing 1.2.0, on 2026-10-09:
- `BetaGeoModel(model_config=..., sampler_config=...)`, `.fit(data, method="mcmc"|"map", ...,
  **sampler_kwargs)`, `.expected_purchases(data, future_t=)`, `.expected_probability_alive(data)`,
  `.expected_purchases_new_customer(data, t=)`, `.fit_summary()`.
- `ParetoNBDModel.fit(data, method="map")` (default method is MAP), `.expected_probability_alive(data,
  future_t=)`.
- `pymc_marketing.clv.plot_expected_purchases_over_time(model, purchase_history, customer_id_col,
  datetime_col, t, ...)`, `plot_probability_alive_matrix(model, ...)`.
- `ModelBuilder.fit` forwards extra keywords to `pymc.sample` for MCMC (read in `model_builder.py`).
- Ran: flat-prior MAP on CDNOW reproduces note 004 (above); BG/NBD P(alive) = 1 at x = 0.
- ArviZ 1.3.0: `az.hdi(data, prob=...)`, `az.summary(data, ci_prob=..., ci_kind="hdi")` signatures.

## Open items for the Technical Expert

- `btyd_bgnbd` should also return per-customer `true_alive` at the end of observation (for Exercise 6).
- Helper for the Exercise 1 closed form, tested against a 100,000-customer simulation.
- Confirm the `checks.k_of_K_in_interval` signature used in Exercise 3.
