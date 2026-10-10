# Lab brief · Module 10 · Budget allocation under uncertainty

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written before the lab was built; the lab has since passed on Colab and in the workshop's Docker image (`runs/`; the Readiness page). Review: Pedagogy Expert. |
| Notebook | `labs/src/10-budget-allocation.py` → `labs/python/10-budget-allocation.ipynb` |
| Lab slot | 55 min (`modules.m10.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Day | Day 4, after lunch |

## Question and decision

**Question.** Given diminishing returns, constraints and uncertainty, how should next quarter's media budget
be split across channels?

**Decision.** Recommend next quarter's weekly spend per channel within the agreed bounds, with expected
incremental sales and a 94% interval, and show how the recommendation changes under a risk-averse or a
CLV-weighted objective.

## What this lab fixes

Module 9 produced calibrated response curves but no rule for dividing a fixed budget.

## Notation

Weekly spend s_c held constant over the quarter (13 weeks). With normalized adstock, constant spend gives
adstocked spend equal to s_c, so the steady-state weekly response is r_c(s_c) = β_c · y_scale ·
saturation(s_c / x_scale_c; λ_c). Marginal ROAS m_c(s) = dr_c/ds. At an allocation that maximizes Σ r_c
subject to Σ s_c = B and bounds, every channel not at a bound has the same marginal ROAS.

## Data

- Synthetic MMM with lift tests: `mktstats.synth.mmm(seed=<default>, confounded=True)` as in Module 9 (or the
  default `mmm()` on main until the confounded option exists; same
  data, so the calibrated model carries over), truth per channel. The true steady-state weekly response can be computed from the truth on main:
  `saturation_beta_sales_units × logistic(saturation_lam × s / channel_scale)` per channel (a small
  `true_response` helper in `mktstats` would keep notebooks short).
- Budget and bounds from `truth.mmm.true_optimal_allocation` on main: weekly budget $72,000 (mean weekly
  total over the last 52 weeks, rounded), bounds 0.5 to 2.0 times each channel's mean weekly spend over the
  last 52 weeks, the current allocation, the optimal allocation, weekly contribution at both, marginal ROAS at
  the optimum and which channels sit at a bound.
- For Exercise 6: new customers acquired per dollar per channel and new-customer CLV per channel (proposed:
  `truth.new_customers_per_dollar` in the MMM truth, not on main yet; and the new-customer CLV from
  `truth.retailer.value_by_channel.new_customer`, with the
  channel names mapped; see open items).

## Model fits

One fit: the calibrated MMM from Module 9 (Module 8 specification + `add_lift_test_measurements` after
`build_model`), nutpie, chains=2, QUICK draws=tune=300, FULL 1000, `progressbar=False`; estimated under a
minute FULL on Colab plus the first-fit compile (estimate). Then `opt = mmm.budget_optimizer(start_date,
end_date)` for a 13-week future window. The optimizer compiles a PyTensor function: on Colab (2026-10-10,
worked, FULL) building it took 1.4 s and the first `allocate_budget` call, compile included, 31–36 s
over three runs (`runs/2026-10-10-colab-py-python-10-budget-allocation.json`).

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · Response curves | 1 · Steady-state response curve | 7 |
| | 2 · Marginal ROAS | 5 |
| | *Run: the calibrated fit (provided)* | — |
| B · Allocate | 3 · An SLSQP allocator | 10 |
| | 4 · PyMC-Marketing's optimizer, in its units | 6 |
| C · Uncertainty and the long run | 5 · Risk of an allocation | 7 |
| | 6 · A CLV-weighted objective | 5 |
| **Exercises** | | **40** |
| Decision | Next quarter's weekly budget | 5 |

### Exercise 1 · Steady-state response curve (7 minutes)

- **Predict.** Doubling TV's weekly spend from its current level: will weekly TV-driven sales roughly double,
  rise by less than half, or rise by more than double?
- **Function.** `response_curve(spend, beta, lam, x_scale, y_scale) -> np.ndarray`, vectorized over spend
  levels and over posterior draws (broadcasting), using logistic saturation from Module 8.
- **Checkpoint.** With the true parameters, matches `true_response` for each channel at 5 spend levels to
  1e-6 relative; increasing in spend; 0 at 0. First checkpoint, before the fit.
- **Explain.** The curve flattens because of saturation; adstock does not change the steady state when it is
  normalized.

### Exercise 2 · Marginal ROAS (5 minutes)

- **Predict.** At current spend, is marginal ROAS above or below average ROAS for every channel?
- **Function.** `marginal_roas(spend, curve_fn, eps=1.0) -> np.ndarray`: central difference
  (r(s + ε) − r(s − ε)) / 2ε.
- **Checkpoint.** Matches the analytic derivative of the logistic curve to 1e-4 relative; decreasing in
  spend (`checks.monotone(increasing=False)`).
- **Explain.** With diminishing returns, the next dollar always returns less than the average dollar; budget
  decisions are about the next dollar.

### Exercise 3 · An SLSQP allocator (10 minutes)

- **Predict.** Will the optimal plan give the most budget to the channel with the highest average ROAS: yes
  or no?
- **Function.** `allocate(total: float, bounds: dict[str, tuple[float, float]], curves: dict[str, Callable])
  -> pd.Series`: maximize Σ_c curves[c](s_c) subject to Σ s_c = total and the bounds, with
  `scipy.optimize.minimize(method="SLSQP")` (minimize the negative), starting from equal shares.
- **Checkpoint (three).** (1) Sum equals `total` to 1e-6 relative and bounds hold. (2) With the true curves,
  the plan's true response is within 0.5% of `truth.mmm.true_optimal_allocation.weekly_contribution_optimal`
  (on main the optimum has tv and search interior with equal marginal ROAS, and social and display at their
  lower bounds). (3) Channels strictly inside their bounds have marginal ROAS
  within 2% of each other.
- **Explain.** Compare with your guess. The optimum equalizes marginal, not average, ROAS: a channel with a
  high average ROAS may already be saturated.

### Exercise 4 · PyMC-Marketing's optimizer, in its units (6 minutes)

- **Predict.** If you passed the quarter's total budget (13 weeks) as `total_budget`, would the recommended
  weekly spend be right, 13 times too high, or 13 times too low?
- **Function.** `pymc_allocation(opt, weekly_total: float, bounds: dict) -> pd.Series` calling
  `opt.allocate_budget(total_budget=weekly_total, budget_bounds=bounds)` and returning
  `result.budgets.to_series()`.
- **Checkpoint.** Sums to `weekly_total` within 0.1%; within bounds; each channel within 5% of the total
  budget of your Exercise 3 allocation computed with posterior-mean curves (both maximize mean response;
  they differ only in that PyMC-Marketing averages response over draws and includes the adstock warm-up).
- **Explain.** In pymc-marketing 1.2.0 the budget is **per period**: the optimizer repeats it for each week of
  the window (read in `optimization_variables.py`: the budget is expanded over the date dimension, and
  budget-distribution factors are "fractions of the per-period budget level"). Its default method is SLSQP,
  the same algorithm as yours.

### Exercise 5 · Risk of an allocation (7 minutes)

*Provided before it: response draws for two plans (the mean-optimal plan and a more even plan) from
`opt.evaluate_response_distribution(plan)`.*

- **Predict.** Which plan has the better worst case (10% quantile of total response): the mean-optimal plan
  or the more even plan?
- **Function.** `allocation_risk(draws: np.ndarray, q: float = 0.10) -> dict` with `mean`, `q` quantile
  and `cvar` (mean of draws at or below the quantile).
- **Checkpoint.** Exact on a toy array; `cvar ≤ quantile ≤ mean` on the real draws.
- **Explain.** The mean-optimal plan concentrates spend where the posterior mean is high, which can also be
  where the posterior is widest. PyMC-Marketing can optimize a risk-aware utility directly (for example
  `pymc_marketing.mmm.utility.conditional_value_at_risk`, passed as `utility_function=` to
  `budget_optimizer`, which forwards it to `BudgetOptimizer`); stretch.

### Exercise 6 · A CLV-weighted objective (5 minutes)

- **Predict.** When the long-run value of newly acquired customers counts, which channel gains budget?
- **Function.** `long_run_value(plan: pd.Series, curves: dict, margin: float, new_customers_per_dollar: dict,
  clv: dict) -> float` = margin × Σ curves[c](s_c) + Σ s_c × new_customers_per_dollar[c] × clv[c] (one week).
- **Checkpoint.** With all CLVs set to 0 it equals margin × short-run response; raising one channel's CLV and
  re-allocating (provided call to your `allocate` with this objective) never lowers that channel's spend.
- **Explain.** Short-run sales understate channels that bring loyal customers; Module 5's CLV by channel puts
  a price on that. The weighting depends on CLV estimates and on acquisition counts that are themselves
  uncertain.

### Decision · Next quarter's weekly budget (5 minutes)

Provided cell, in the standard order:
- **Number:** for three plans (mean-optimal, risk-averse by 10% quantile, CLV-weighted): weekly spend per
  channel, expected incremental weekly sales with 94% interval, and the posterior probability that each plan
  beats the current plan, computed draw by draw from `evaluate_response_distribution` with paired draws.
- **Rule:** "Recommend the mean-optimal plan unless its 10% quantile is below the current plan's 10%
  quantile; use the CLV-weighted plan only if the CLV-by-channel estimates come from Module 5's model, and
  say so."
- **Recommendation:** the learner's sentence with the weekly spend by channel, expected incremental sales and
  range, the probability it beats the current plan, and what would change it (a new lift test from Module 9).
  The cell prints the mode.

## Stretch (optional)

Run `mmm.budget_optimizer(start, end, utility_function=conditional_value_at_risk(...))` and compare with
Exercise 5; add a minimum-spend constraint for a contractual TV commitment.

## Known pitfalls

- `total_budget` is per period (Exercise 4).
- Without `budget_bounds`, PyMC-Marketing warns and uses (0, total_budget) per channel.
- SLSQP is local: start from a feasible point and check the marginal-ROAS condition.
- Channel names must match across the MMM and the CLV-by-channel table (open item).

## APIs verified (and how)

pymc-marketing 1.2.0 installed source, 2026-10-09: `MMM.budget_optimizer(start_date, end_date, *,
budgets_to_optimize=None, cost_per_unit=None, compile_kwargs=None, **kwargs)` (kwargs forwarded to
`BudgetOptimizer`); `BudgetOptimizer.allocate_budget(total_budget, budget_bounds=None, x0=None,
minimize_kwargs=None, return_if_fail=False, callback=False)` returning `BudgetOptimizationResult` with
`budgets` (xarray, monetary units) and `scipy_result`, default `method="SLSQP"`; the per-period budget
expansion in `optimization_variables.py`; `evaluate_response_distribution(plan, response_variable=None)`
signature (its output dims were not checked); utility functions in `pymc_marketing.mmm.utility`
(`average_response`, `value_at_risk`, `conditional_value_at_risk`, `sharpe_ratio`, …). SciPy SLSQP bounds and
constraints: SciPy docs (references.qmd). The optimizer was not run in this check.

## Open items for the Technical Expert

- Resolved on main: `truth.mmm.true_optimal_allocation`. Optional: a `true_response` helper.
- One channel vocabulary across `retailer` (acquisition channels) and `mmm` (media channels), or an explicit
  mapping table in the truth, so Exercise 6 can join CLV to media spend.
- Check the output dims of `evaluate_response_distribution` and the time to compile the optimizer on Colab.
