# %% [markdown]
# <!--
# APIs this lab calls, checked against the installed source of pymc-marketing 1.2.0 and
# SciPy 1.16.3 (.venv/lib/python3.13/site-packages, 2026-10-09):
# - MMM(...) as in Module 8; .build_model(X, y); .add_lift_test_measurements(df) with columns
#   channel, x, delta_x, delta_y, sigma, after build_model (mmm/mmm.py); .fit(..., nuts_sampler=
#   "nutpie", target_accept=, random_seed=, progressbar=False).
# - idata.constant_data["channel_scale"] (max spend per channel) and ["target_scale"] (max sales):
#   the model's scaling, equal to truth channel_scale and y_scale on this data (measured).
# - MMM.budget_optimizer(start_date, end_date, *, budgets_to_optimize=None, cost_per_unit=None,
#   compile_kwargs=None, **kwargs) -> BudgetOptimizer (kwargs such as utility_function= are
#   forwarded). Its model's date axis is 8 weeks of history (carry-in), the decision weeks and
#   8 weeks of carry-over (_window_layout); num_periods = the decision weeks.
# - BudgetOptimizer.allocate_budget(total_budget, budget_bounds=None, x0=None,
#   minimize_kwargs=None, return_if_fail=False, callback=False) -> BudgetOptimizationResult with
#   .budgets (xarray over channel, money) and .scipy_result; default method SLSQP. total_budget
#   is per period: optimization_variables.py repeats the budget over the date dimension.
#   minimize_kwargs is merged over DEFAULT_MINIMIZE_KWARGS = {"method": "SLSQP", "options":
#   {"ftol": 1e-9, "maxiter": 1000}} (a shallow merge, so pass "method" and "options"); x0 may
#   be a labelled DataArray over channel. On failure it raises MinimizeException. With the
#   default ftol it raised "Positive directional derivative for linesearch" on Colab x86_64
#   (the lead's run, 2026-10-09); ftol is absolute, on an objective of about $2.6 million.
# - truth["channel_value"] (new_customers_per_dollar, clv_margin_by_media_channel,
#   media_to_acquisition, long_run_optimal_allocation.mmm) from mktstats.synth.channels.
# - BudgetOptimizer.evaluate_response_distribution(plan) -> DataArray with dims ("sample",):
#   total media contribution over the whole window (carry-in + decisions + carry-over) per
#   posterior draw, in sales units (measured: dims ("sample",), 2 x draws values).
# - pymc_marketing.mmm.utility.conditional_value_at_risk(confidence_level) (stretch only) scores
#   the (1 - confidence_level) tail of that window total. The core risk-averse plan does not use
#   the library's utilities: they score the window total, carry-in included, not the plan's
#   incremental weekly sales that this lab compares (a value_at_risk(0.9) plan scored a lower
#   10% quantile than the mean-optimal plan on that measure in a test run).
# - scipy.optimize.minimize(method="SLSQP", bounds=, constraints=[{"type": "eq", ...}]) clips
#   the starting point into the bounds (_slsqp_py.py).
# -->
#
# # Part A · Response curves
#
# Module 8 fitted a marketing mix model (MMM) and asked what each channel returned. Module 9
# calibrated it with lift tests. This lab asks the forward question: how should next quarter's
# weekly media budget be split across tv, search, social and display, given diminishing returns,
# the bounds the business agreed and the uncertainty in the model?
#
# The planning facts (from the synthetic truth file, as a planner would get them from finance):
# a weekly budget of \$72,000, the mean weekly total over the last 52 weeks; each channel between
# half and twice its mean weekly spend over those 52 weeks; and the current plan, the recent
# mix scaled to the budget.

# %%
import io
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import Image, display
from scipy.optimize import minimize

from mktstats.data import load_synthetic, load_truth
from mktstats.synth.mmm import steady_state_response

# Notices from inside the libraries that say nothing about this lab: a PyMC 6 deprecation,
# and two about optional progress-bar widgets.
warnings.filterwarnings("ignore", message=".*merge_dataset.*")
warnings.filterwarnings("ignore", message=".*ipywidgets.*")
warnings.filterwarnings("ignore", message=".*IProgress not found.*")


def show(fig):
    """Show a figure as a PNG on Colab, in Jupyter and in a headless run alike."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))


weekly = load_synthetic("mmm_weekly")
lift_tests = load_synthetic("mmm_lift_tests")
truth_all = load_truth()
truth = truth_all["mmm"]
plan_truth = truth["true_optimal_allocation"]  # planning facts, and the answer key for checks
CHANNELS = ["tv", "search", "social", "display"]
SEED = 2028  # every random step in this notebook uses this seed
GROSS_MARGIN = 0.30  # stated by the business, as in Module 8
WEEKS = 13  # next quarter
BUDGET = plan_truth["weekly_budget"]
BOUNDS = {c: tuple(plan_truth["bounds"][c]) for c in CHANNELS}
CURRENT = pd.Series(plan_truth["current_allocation"])[CHANNELS]


def true_response(spend, channel):
    """True steady-state weekly sales from a constant weekly spend on `channel` (from truth)."""
    t = truth["channels"][channel]
    return steady_state_response(spend, t["saturation_beta_sales_units"], t["saturation_lam"],
                                 t["channel_scale"])


pd.DataFrame({"current plan ($/week)": CURRENT,
              "lower bound": [BOUNDS[c][0] for c in CHANNELS],
              "upper bound": [BOUNDS[c][1] for c in CHANNELS]}).round(0)

# %% [markdown]
# ## Exercise 1 · Steady-state response curve (7 minutes)
#
# Hold weekly spend $s_c$ on channel $c$ constant for the quarter. Normalized adstock (Module 8)
# averages recent weeks with weights that sum to 1, so the adstock of a constant spend is that
# spend, and after a few weeks the channel adds the same sales every week: its *steady-state
# response*
#
# $$r_c(s) = \beta_c \cdot y_{\text{scale}} \cdot \frac{1 - e^{-\lambda_c s / x_c}}{1 + e^{-\lambda_c s / x_c}},$$
#
# where $\beta_c$ and $\lambda_c$ are the fitted saturation parameters, $x_c$ is the channel's
# largest weekly spend in the data and $y_{\text{scale}}$ the largest weekly sales (PyMC-Marketing
# divides by both before fitting, so $\beta_c$ is in units of peak weekly sales).
#
# **Predict.** tv's current plan is about \$31,600 a week. If tv's weekly spend doubled, would
# tv's weekly sales more than double, rise by between half and double, or rise by less than half?
#
# **Task.** Write `response_curve(spend, beta, lam, x_scale, y_scale)` returning $r(s)$. Use only
# NumPy operations, so that arrays broadcast: spend levels of shape `(k,)` with `beta` and `lam`
# of shape `(d, 1)` (one row per posterior draw) give a `(d, k)` array.

# %% tags=["exercise"]
def response_curve(spend, beta, lam, x_scale, y_scale):
    # TODO 1: beta * y_scale * logistic saturation of lam * spend / x_scale
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def response_curve(spend, beta, lam, x_scale, y_scale):
    z = lam * np.asarray(spend, dtype=float) / x_scale
    return beta * y_scale * (1 - np.exp(-z)) / (1 + np.exp(-z))


# %% [markdown]
# **Explain.** Doubling tv from its current plan raises its weekly sales by about 84% on the
# true curve (the next cell prints the numbers). Compare with your guess: which part of the
# formula makes the second \$31,600 earn less than the first?
#
# <details><summary>Why this solution works</summary>
#
# The formula is Module 8's logistic saturation, scaled back to dollars: $\beta_c y_{\text{scale}}$
# is the most the channel can add in a week and $\lambda_c / x_c$ how fast spend gets there. The
# curve is concave (it flattens), so each extra dollar adds less. Adstock does not appear: with
# normalized weights it moves effects in time but leaves a constant spend's weekly effect
# unchanged. Written with NumPy operations only, the same function evaluates one curve, a grid
# of spend levels, or thousands of posterior draws at once.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    levels = np.array([0.0, 5_000, 20_000, 40_000, 80_000])
    for c in CHANNELS:
        t = truth["channels"][c]
        got = response_curve(levels, t["saturation_beta_model_units"], t["saturation_lam"],
                             t["channel_scale"], truth["y_scale"])
        # saturation_beta_model_units is stored to 6 significant digits, hence rel=1e-5
        checks.close(got, true_response(levels, c), rel=1e-5,
                     name=f"response_curve for {c} with the true parameters")
    many = response_curve(levels, np.array([[0.1], [0.2], [0.3]]), 2.0, 30_000.0, 500_000.0)
    assert np.shape(many) == (3, 5), (
        f"Three draws of beta (shape (3, 1)) and five spend levels should give a (3, 5) array;"
        f" got shape {np.shape(many)}. Use NumPy operations (np.exp), not math.exp or a loop."
    )
    assert np.allclose(many[:, 0], 0.0), "Zero spend must give zero response."
    checks.monotone(many[1], increasing=True, strict=True, name="the response as spend grows")

# %%
s_now = CURRENT["tv"]
r1, r2 = true_response(s_now, "tv"), true_response(2 * s_now, "tv")
print(f"tv on the true curve: ${s_now:,.0f}/week -> ${r1:,.0f} of sales/week;"
      f" ${2 * s_now:,.0f}/week -> ${r2:,.0f} ({r2 / r1 - 1:+.0%})")

# %% [markdown]
# ## Exercise 2 · Marginal ROAS (5 minutes)
#
# The *marginal ROAS* $m_c(s) = dr_c/ds$ is what the next dollar on channel $c$ returns at weekly
# spend $s$. The *average* ROAS $r_c(s)/s$ is what all dollars so far returned on average.
#
# **Predict.** At the current plan, is marginal ROAS above or below average ROAS for every
# channel?
#
# **Task.** Write `marginal_roas(spend, curve_fn, eps=1.0)`: the central difference
# $(r(s + \varepsilon) - r(s - \varepsilon)) / 2\varepsilon$ for a function `curve_fn` of spend
# ($\varepsilon$ = one dollar).

# %% tags=["exercise"]
def marginal_roas(spend, curve_fn, eps=1.0):
    # TODO 2: central difference of curve_fn at spend
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def marginal_roas(spend, curve_fn, eps=1.0):
    spend = np.asarray(spend, dtype=float)
    return (curve_fn(spend + eps) - curve_fn(spend - eps)) / (2 * eps)


# %% [markdown]
# **Explain.** Compare with your guess, using the table below. Why does a budget decision
# depend on the marginal ROAS and not on the average ROAS that Module 8 reported?
#
# <details><summary>Why this solution works</summary>
#
# The central difference approximates the slope of the curve with an error that shrinks with
# $\varepsilon^2$; one dollar is tiny next to spend levels in the thousands. On a concave curve
# that starts at zero, the slope at $s$ is below the average slope from 0 to $s$, so marginal
# ROAS is below average ROAS at every spend level. A dollar moved from one channel to another
# loses that channel's marginal return and gains the other's: only marginal ROAS says whether
# the move pays.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    grid = np.array([2_000.0, 10_000, 30_000, 60_000])
    for c in CHANNELS:
        t = truth["channels"][c]
        big_b, lam, x = t["saturation_beta_sales_units"], t["saturation_lam"], t["channel_scale"]
        exact = big_b * lam / (2 * x) * (1 - np.tanh(lam * grid / (2 * x)) ** 2)
        got = marginal_roas(grid, lambda s, c=c: true_response(s, c))
        checks.close(got, exact, rel=1e-4, name=f"marginal ROAS of {c} against the exact slope")
        checks.monotone(got, increasing=False, strict=True,
                        name=f"marginal ROAS of {c} as spend grows")

# %%
pd.DataFrame({
    "current plan ($/week)": CURRENT,
    "average ROAS": [true_response(CURRENT[c], c) / CURRENT[c] for c in CHANNELS],
    "marginal ROAS": [float(marginal_roas(CURRENT[c], lambda s, c=c: true_response(s, c)))
                      for c in CHANNELS],
}).round(2)

# %% [markdown]
# **Run.** The calibrated model from Module 9, refitted here so this notebook stands alone:
# Module 8's specification and spend-share priors, plus one lift test per channel added to the
# likelihood. Read Exercise 3's Predict prompt while it runs.
#
# With QUICK on, the sampler draws 300 instead of 1,000 values per chain: intervals are rougher,
# and a decision near its threshold can flip. That is a lesson about sample size, not a bug.

# %%
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

X = weekly.drop(columns=["y"])
y = weekly["y"]
spend_share = weekly[CHANNELS].sum() / weekly[CHANNELS].sum().sum()
beta_sigma = xr.DataArray(len(CHANNELS) * spend_share.to_numpy(), dims="channel",
                          coords={"channel": CHANNELS})
mmm = MMM(
    date_column="date_week",
    channel_columns=CHANNELS,
    control_columns=["price_index", "holiday", "t"],
    target_column="y",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
    model_config={"saturation_beta": Prior("HalfNormal", sigma=beta_sigma, dims="channel")},
)
mmm.build_model(X, y)
mmm.add_lift_test_measurements(lift_tests[["channel", "x", "delta_x", "delta_y", "sigma"]])
DRAWS = 300 if QUICK else 1000
t_fit = time.time()
idata = mmm.fit(X, y, nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS,
                target_accept=0.9, random_seed=SEED, progressbar=False)
fit_seconds = time.time() - t_fit

# %%
post = idata.posterior.to_dataset().stack(sample=("chain", "draw"))
beta_draws = post["saturation_beta"].sel(channel=CHANNELS).transpose("sample", "channel").values
lam_draws = post["saturation_lam"].sel(channel=CHANNELS).transpose("sample", "channel").values
x_scale = idata.constant_data["channel_scale"].sel(channel=CHANNELS).values
y_scale = float(idata.constant_data["target_scale"])
n_div = int(idata["/sample_stats"]["diverging"].sum())
print(f"Fitted 2 chains x {DRAWS} draws in {fit_seconds:.1f} s"
      f" ({'QUICK' if QUICK else 'FULL'} settings); {n_div} divergences;"
      f" {beta_draws.shape[0]} posterior draws.")


def posterior_curve(j):
    """Channel j's weekly response at spend s, averaged over the posterior draws."""
    return lambda s: float(np.mean(response_curve(s, beta_draws[:, j], lam_draws[:, j],
                                                  x_scale[j], y_scale)))


mean_curves = {c: posterior_curve(j) for j, c in enumerate(CHANNELS)}
true_curves = {c: (lambda s, c=c: float(true_response(s, c))) for c in CHANNELS}

# %% [markdown]
# Look at where the true curve (black) lies inside each channel's 94% band, and how wide the
# band is at the spend levels between the bounds (shaded): that width is the model's
# uncertainty about what a plan will return.

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for j, (c, ax) in enumerate(zip(CHANNELS, axes, strict=True)):
    s = np.linspace(0, 1.1 * BOUNDS[c][1], 80)
    r = response_curve(s, beta_draws[:, [j]], lam_draws[:, [j]], x_scale[j], y_scale)
    band = np.array([checks.interval(r[:, i], 0.94, "hdi") for i in range(len(s))])
    ax.axvspan(BOUNDS[c][0] / 1e3, BOUNDS[c][1] / 1e3, color="grey", alpha=0.12,
               label="allowed spend")
    ax.fill_between(s / 1e3, band[:, 0] / 1e3, band[:, 1] / 1e3, alpha=0.35, label="94% HDI")
    ax.plot(s / 1e3, r.mean(axis=0) / 1e3, label="posterior mean")
    ax.plot(s / 1e3, true_response(s, c) / 1e3, "k--", label="truth")
    ax.set_title(c)
    ax.set_xlabel("weekly spend ($ thousand)")
axes[0].set_ylabel("weekly sales ($ thousand)")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
fig.tight_layout()
show(fig)

# %% [markdown]
# # Part B · Allocate
#
# The problem: choose weekly spend $s_c$ for every channel to maximize total weekly sales
# $\sum_c r_c(s_c)$, subject to $\sum_c s_c = B$ (the budget) and $\ell_c \le s_c \le u_c$ (the
# bounds). At the best plan, every channel strictly inside its bounds has the same marginal
# ROAS: otherwise moving a dollar from the lower to the higher one would add sales.
#
# ## Exercise 3 · An SLSQP allocator (10 minutes)
#
# SLSQP (sequential least squares quadratic programming) is a SciPy optimizer for smooth
# problems with bounds and equality constraints. PyMC-Marketing's budget optimizer uses it too.
#
# **Predict.** Will the best plan give the most budget to the channel with the highest average
# ROAS (search, in Module 8): yes or no?
#
# **Task.** Write `allocate(total, bounds, curves)`: `curves` maps each channel to a function of
# its weekly spend; `bounds` maps it to `(lower, upper)`. Maximize the sum of the curves by
# minimizing its negative with `scipy.optimize.minimize(..., method="SLSQP", bounds=...,
# constraints=[{"type": "eq", "fun": lambda s: s.sum() - total}])`, starting from equal shares
# (SLSQP moves a start that breaks a bound onto the bound). Return a `pd.Series` of spend
# indexed by channel.

# %% tags=["exercise"]
def allocate(total, bounds, curves):
    # TODO 3: minimize -sum(curves[c](s_c)) over s with SLSQP, the budget equality and bounds
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def allocate(total, bounds, curves):
    names = list(curves)
    result = minimize(
        lambda s: -sum(curves[c](v) for c, v in zip(names, s, strict=True)),
        x0=np.full(len(names), total / len(names)),
        method="SLSQP",
        bounds=[bounds[c] for c in names],
        constraints=[{"type": "eq", "fun": lambda s: s.sum() - total}],
    )
    if not result.success:
        raise RuntimeError(f"SLSQP did not converge: {result.message}")
    return pd.Series(result.x, index=names)


# %% [markdown]
# **Explain.** Compare with your guess. In the table below the best plan puts social and display
# at their lower bounds and splits the rest between tv and search where their marginal ROAS is
# equal. Why can a channel with a high average ROAS still get less money?
#
# <details><summary>Why this solution works</summary>
#
# SLSQP replaces the problem near the current point by a quadratic model with linearized
# constraints, solves that, and repeats. The response curves are concave, so the problem has one
# optimum and a local method finds it. At the optimum the channels inside their bounds share one
# marginal ROAS (the Lagrange multiplier of the budget constraint, about 1.53 dollars of sales
# per dollar on the true curves); a channel at its lower bound has a lower marginal ROAS even
# there. Average ROAS includes the steep first dollars, so it says little about the next dollar.
# SLSQP is local in general: on curves that are not concave, start it from several points.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    plan_check = allocate(BUDGET, BOUNDS, true_curves)
    assert isinstance(plan_check, pd.Series) and sorted(plan_check.index) == sorted(CHANNELS), (
        f"allocate should return a pd.Series indexed by channel {CHANNELS}; got"
        f" {type(plan_check).__name__} {getattr(plan_check, 'index', '')}."
    )
    plan_check = plan_check[CHANNELS].astype(float)
    checks.close(plan_check.sum(), BUDGET, rel=1e-6, name="the plan's total weekly spend")
    for c in CHANNELS:
        lo, hi = BOUNDS[c]
        assert lo - 1e-6 * BUDGET <= plan_check[c] <= hi + 1e-6 * BUDGET, (
            f"{c} gets ${plan_check[c]:,.0f}, outside its bounds [${lo:,.0f}, ${hi:,.0f}]."
            " Pass bounds=[bounds[c] for c in channels] in the same order as the spend vector."
        )
    got = sum(true_response(plan_check[c], c) for c in CHANNELS)
    best = plan_truth["weekly_contribution_optimal"]
    assert got >= best * (1 - 0.005), (
        f"On the true curves your plan earns ${got:,.0f} of weekly sales; the best plan earns"
        f" ${best:,.0f} (allowed: 0.5% less). Check that you minimize the NEGATIVE of the total"
        " and that the equality constraint is s.sum() - total."
    )
    slope = {c: (true_response(plan_check[c] + 1, c) - true_response(plan_check[c] - 1, c)) / 2
             for c in CHANNELS}
    inside = [c for c in CHANNELS
              if BOUNDS[c][0] + 1e-3 * BUDGET < plan_check[c] < BOUNDS[c][1] - 1e-3 * BUDGET]
    m_in = [slope[c] for c in inside]
    assert len(m_in) < 2 or max(m_in) <= 1.02 * min(m_in), (
        f"Channels inside their bounds should have equal marginal ROAS (within 2%); yours:"
        f" { {c: round(slope[c], 3) for c in inside} }. The optimizer stopped early: check its"
        " result.success and message."
    )

# %% [markdown]
# **Run.** The same allocator on the model's curves (each the posterior mean of the response),
# next to the plan on the true curves and the answer key. The last column scores every plan on
# the true curves.

# %%
plan_mean = allocate(BUDGET, BOUNDS, mean_curves)[CHANNELS]
plans_a = pd.DataFrame({
    "current": CURRENT,
    "your allocate, model curves": plan_mean,
    "your allocate, true curves": plan_check,
    "answer key": pd.Series(plan_truth["optimal_allocation"])[CHANNELS],
})
scored = plans_a.T.assign(**{"true weekly sales": [sum(true_response(p[c], c) for c in CHANNELS)
                                                   for _, p in plans_a.T.iterrows()]})
scored.round(0)

# %% [markdown]
# ## Exercise 4 · PyMC-Marketing's optimizer, in its units (6 minutes)
#
# PyMC-Marketing optimizes a budget for a future window of dates. `mmm.budget_optimizer(start,
# end)` builds the optimizer for the 13 weeks after the data end (provided below);
# `opt.allocate_budget(total_budget=..., budget_bounds=...)` runs it.
#
# **Predict.** If you passed the quarter's total budget (13 weeks of \$72,000) as `total_budget`,
# would the recommended weekly spend be right, 13 times too high, or 13 times too low?
#
# Two settings matter, both provided below. The optimizer stops when SLSQP's objective changes
# by less than `ftol`, in the objective's own units. Its default, `ftol=1e-9`, is absolute: on
# an objective of about \$2.6 million (mean media sales summed over the window) that asks for
# more precision than the rounding error of the sum, and SLSQP can then fail with "Positive
# directional derivative for linesearch" on one machine and not on another (it did on Colab's
# x86 runtime). `SLSQP_OPTIONS` asks for a cent instead. And a start inside the bounds (`x0`,
# your Exercise 3 plan) beats the default start, which splits the budget equally and breaks
# display's upper bound.
#
# **Task.** Write `pymc_allocation(opt, weekly_total, bounds, x0=None,
# minimize_kwargs=SLSQP_OPTIONS)`: call `opt.allocate_budget(total_budget=weekly_total,
# budget_bounds=bounds, x0=..., minimize_kwargs=minimize_kwargs)`, where `x0` is `None` or a plan
# (`pd.Series`) converted with the provided `plan_array(x0)`, and return the result's `.budgets`
# as a `pd.Series` (`.to_series()`).

# %%
start = weekly["date_week"].max() + pd.Timedelta(weeks=1)
end = start + pd.Timedelta(weeks=WEEKS - 1)
t_opt = time.time()
opt = mmm.budget_optimizer(start_date=start, end_date=end)
print(f"Optimizer for {start:%Y-%m-%d} to {end:%Y-%m-%d}: {opt.num_periods} decision weeks"
      f" (built in {time.time() - t_opt:.1f} s)")
# Keys checked in pymc-marketing 1.2.0 (BudgetOptimizer.allocate_budget merges them over its
# defaults {"method": "SLSQP", "options": {"ftol": 1e-9, "maxiter": 1000}}) and SciPy 1.16.3.
SLSQP_OPTIONS = {"method": "SLSQP", "options": {"ftol": 0.01, "maxiter": 1000}}


def plan_array(plan):
    """A plan (pd.Series by channel) as the labelled DataArray PyMC-Marketing expects."""
    return xr.DataArray(pd.Series(plan)[CHANNELS].to_numpy(dtype=float), dims=["channel"],
                        coords={"channel": CHANNELS})


# %% tags=["exercise"]
def pymc_allocation(opt, weekly_total, bounds, x0=None, minimize_kwargs=SLSQP_OPTIONS):
    # TODO 4: opt.allocate_budget(total_budget=weekly_total, budget_bounds=bounds,
    #         x0=None or plan_array(x0), minimize_kwargs=minimize_kwargs).budgets as a Series
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def pymc_allocation(opt, weekly_total, bounds, x0=None, minimize_kwargs=SLSQP_OPTIONS):
    result = opt.allocate_budget(
        total_budget=weekly_total,
        budget_bounds=bounds,
        x0=None if x0 is None else plan_array(x0),
        minimize_kwargs=minimize_kwargs,
    )
    return result.budgets.to_series()


# %% [markdown]
# **Explain.** In pymc-marketing 1.2.0 `total_budget` is a budget **per period** (per week
# here): the optimizer repeats it for each of the 13 weeks. Compare its plan with yours in the
# table below. Both maximize mean sales under the posterior; why are they not identical?
#
# <details><summary>Why this solution works</summary>
#
# The optimizer builds the model over the window: 8 weeks of history before it (their spend
# still carries into the window), the 13 decision weeks and 8 weeks after (so the carry-over of
# the last weeks' spend counts). The library's code expands the budget over the date dimension
# and describes any uneven split over weeks as "fractions of the per-period budget level", so a
# quarterly total would be spent every week: 13 times too much. Its default method is SLSQP,
# like yours. It differs from your plan only because it scores whole weeks with adstock warm-up
# and history instead of the steady state; the difference is a fraction of a percent of the
# budget.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    t_alloc = time.time()
    plan_pymc = pymc_allocation(opt, BUDGET, BOUNDS, x0=plan_mean)
    alloc_seconds = time.time() - t_alloc
    assert isinstance(plan_pymc, pd.Series) and sorted(plan_pymc.index) == sorted(CHANNELS), (
        f"Return result.budgets.to_series(): a pd.Series indexed by channel; got"
        f" {type(plan_pymc).__name__}."
    )
    plan_pymc = plan_pymc[CHANNELS].astype(float)
    checks.close(plan_pymc.sum(), BUDGET, rel=1e-3,
                 name="PyMC-Marketing's weekly total (pass the WEEKLY budget)")
    for c in CHANNELS:
        lo, hi = BOUNDS[c]
        assert lo - 1e-6 * BUDGET <= plan_pymc[c] <= hi + 1e-6 * BUDGET, (
            f"{c} gets ${plan_pymc[c]:,.0f}, outside [${lo:,.0f}, ${hi:,.0f}]: pass budget_bounds."
        )
    gap = (plan_pymc - plan_mean).abs() / BUDGET
    assert gap.max() <= 0.05, (
        f"PyMC-Marketing's plan differs from your Exercise 3 plan on the model's curves by up to"
        f" {gap.max():.1%} of the budget ({gap.idxmax()}); expected within 5%. Check the budget"
        " and the bounds you pass."
    )

# %%
print(f"allocate_budget took {alloc_seconds:.1f} s (the first call compiles the model)")
pd.DataFrame({"your allocate, model curves": plan_mean, "PyMC-Marketing": plan_pymc,
              "difference (% of budget)": 100 * (plan_pymc - plan_mean) / BUDGET}).round(1)

# %% [markdown]
# # Part C · Uncertainty and the long run
#
# A plan's sales are uncertain because the model's parameters are. For each posterior draw,
# `opt.evaluate_response_distribution(plan)` gives the total media contribution over the
# optimizer's window. The provided `weekly_sales_draws(plan)` turns that into the plan's
# incremental sales per week: it subtracts the same total with zero spend (what the history
# alone still contributes) and divides by the 13 weeks. Draw $i$ of two plans comes from the
# same parameter draw, so the two can be compared draw by draw.

# %%
zero_total =opt.evaluate_response_distribution(plan_array(pd.Series(0.0, index=CHANNELS)))


def weekly_sales_draws(plan):
    """Posterior draws of the plan's incremental sales per week over the 13-week window."""
    total = opt.evaluate_response_distribution(plan_array(plan))
    return (total - zero_total).values / WEEKS


draws_mean_opt = weekly_sales_draws(plan_pymc)
draws_current = weekly_sales_draws(CURRENT)
print(f"{draws_mean_opt.size} posterior draws per plan")

# %% [markdown]
# ## Exercise 5 · Risk of an allocation (7 minutes)
#
# Two summaries of a plan's bad case: the *10% quantile* (sales are below it in 10% of draws)
# and the *conditional value at risk* (CVaR), the mean of the draws at or below that quantile.
#
# **Predict.** Which plan has the better worst case (higher 10% quantile of weekly sales): the
# mean-optimal plan from Exercise 4, or the current plan, which is more even?
#
# **Task.** Write `allocation_risk(draws, q=0.10)` returning a dict with `mean`, `quantile`
# (`np.quantile(draws, q)`) and `cvar` (mean of the draws at or below the quantile).

# %% tags=["exercise"]
def allocation_risk(draws, q=0.10):
    # TODO 5: mean, the q quantile, and the mean of the draws at or below the quantile
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def allocation_risk(draws, q=0.10):
    draws = np.asarray(draws, dtype=float).ravel()
    cut = np.quantile(draws, q)
    return {"mean": draws.mean(), "quantile": cut, "cvar": draws[draws <= cut].mean()}


# %% [markdown]
# **Explain.** Compare with your guess, using the table below. It adds a third plan, provided:
# the *risk-averse* plan, which maximizes the 10% quantile instead of the mean (SLSQP again,
# started from the mean-optimal plan). Was the more even current plan safer? How far does the
# risk-averse plan move from the mean-optimal one?
#
# <details><summary>Why this solution works</summary>
#
# The quantile and CVaR read the left tail of the draws; CVaR also says how bad the tail is
# beyond the cut. Spreading money evenly is not the same as lowering risk: the current plan
# keeps money on channels with low marginal ROAS, so its whole distribution, tail included,
# sits lower. A risk-averse objective would move money away from a channel whose response is
# uncertain *at the margin*, giving up a little expected sales for a better bad case. Here it
# barely moves: at the mean-optimal plan tv and search have the same marginal ROAS, and most of
# the uncertainty is in search's overall level (the wide band in the response-curve plot),
# which every allowed plan carries, so no reallocation within the bounds removes it. A lift
# test that narrows search's curve would (Module 9). Whether a better tail is worth lower
# expected sales is a business choice; the rule in the Decision states it. With QUICK on,
# quantiles from 600 draws are rough.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    toy_draws = np.arange(1.0, 11.0)  # 1, 2, ..., 10
    toy_risk = allocation_risk(toy_draws, q=0.2)
    checks.columns(pd.DataFrame([toy_risk]), ["mean", "quantile", "cvar"],
                   name="the dict allocation_risk returns")
    checks.close(toy_risk["mean"], 5.5, abs=1e-12, name="the mean of 1..10")
    checks.close(toy_risk["quantile"], 2.8, abs=1e-12,
                 name="the 20% quantile of 1..10 (np.quantile interpolates: 2.8)")
    checks.close(toy_risk["cvar"], 1.5, abs=1e-12,
                 name="the CVaR of 1..10 at 20% (mean of 1 and 2, the draws at or below 2.8)")
    risk_opt = allocation_risk(draws_mean_opt)
    assert risk_opt["cvar"] <= risk_opt["quantile"] <= risk_opt["mean"], (
        f"On real draws CVaR <= quantile <= mean must hold; got {risk_opt}."
    )

# %%
t_risk = time.time()
risk_fit = minimize(
    lambda s: -np.quantile(weekly_sales_draws(pd.Series(s, index=CHANNELS)), 0.10),
    x0=plan_pymc.to_numpy(),  # start at the mean-optimal plan
    method="SLSQP",
    bounds=[BOUNDS[c] for c in CHANNELS],
    constraints=[{"type": "eq", "fun": lambda s: s.sum() - BUDGET}],
    # SciPy's default tolerance, on weekly sales of about $0.1 million: well above the rounding
    # error of these sums, unlike PyMC-Marketing's 1e-9 on a window total 20 times larger.
    options={"ftol": 1e-6, "maxiter": 200},
)
plan_risk = pd.Series(risk_fit.x, index=CHANNELS)
risk_seconds = time.time() - t_risk
draws_risk = weekly_sales_draws(plan_risk)
# Keep the result only if it is a valid plan that is at least as good in the tail; the quantile
# is flat and kinked here, so SLSQP may stop early. Otherwise the mean-optimal plan stands.
valid = (risk_fit.success and abs(plan_risk.sum() - BUDGET) <= 1.0
         and np.quantile(draws_risk, 0.10) >= np.quantile(draws_mean_opt, 0.10))
if not valid:
    print(f"SLSQP did not improve the 10% quantile ({risk_fit.message}): keeping the"
          " mean-optimal plan as the risk-averse plan.")
    plan_risk, draws_risk = plan_pymc.copy(), draws_mean_opt
risk_table = pd.DataFrame({
    name: {**{c: p[c] for c in CHANNELS}, **allocation_risk(d)}
    for name, p, d in [("current", CURRENT, draws_current),
                       ("mean-optimal", plan_pymc, draws_mean_opt),
                       ("risk-averse (best 10% quantile)", plan_risk, draws_risk)]
})
moved = (plan_risk - plan_pymc).abs().sum() / 2
print(f"Risk-averse plan: SLSQP from the mean-optimal plan, {risk_fit.nit} iterations in"
      f" {risk_seconds:.1f} s ({risk_fit.message}); it moved ${moved:,.0f} of the weekly budget.")
risk_table.round(0)

# %% [markdown]
# Look at the left tails: where each plan's 10% quantile (vertical line) sits.

# %%
fig, ax = plt.subplots(figsize=(8, 3))
for (name, d), style in zip({"current": draws_current, "mean-optimal": draws_mean_opt,
                             "risk-averse": draws_risk}.items(), ["-", "--", ":"], strict=True):
    hist, edges = np.histogram(d / 1e3, bins=40)
    line = ax.step(edges[:-1], hist, where="post", linestyle=style, label=name)[0]
    ax.axvline(np.quantile(d, 0.1) / 1e3, color=line.get_color(), linestyle=style, lw=1)
ax.set_xlabel("incremental weekly sales ($ thousand), per posterior draw")
ax.set_ylabel("draws")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
show(fig)

# %% [markdown]
# ## Exercise 6 · A CLV-weighted objective (5 minutes)
#
# Short-run sales miss what new customers buy later. Module 5 priced that: the *customer
# lifetime value* (CLV) of a new customer by acquisition channel. Add it to the objective:
#
# $$V(s) = m \sum_c r_c(s_c) + \sum_c s_c \cdot n_c \cdot \text{CLV}_c,$$
#
# where $m$ is the gross margin, $n_c$ the new customers acquired per dollar on channel $c$, and
# $\text{CLV}_c$ a new customer's future margin.
#
# Media channels (where money is spent) and acquisition channels (how the CRM records a new
# customer's first visit: search, social, referral) are different lists. The synthetic world
# links them, and the table below shows the link: tv viewers mostly arrive by word of mouth
# (recorded as referral) or by searching for the brand, so a tv-acquired customer is valued as
# that mix of acquisition channels. $\text{CLV}_c$ is the discounted margin on a new customer's
# *repeat* purchases over 52 weeks (Module 5's quantity): the first purchase is already in the
# MMM's short-run sales. Here these numbers come from the truth file; in practice the cost per
# new customer comes from attribution in the CRM and the CLV from Module 5's model, both with
# uncertainty.
#
# **Predict.** When the long-run value of new customers counts, which channel gains budget?
#
# **Task.** Write `long_run_value(plan, curves, margin, new_customers_per_dollar, clv)`: $V$ for a
# `plan` (a `pd.Series` of weekly spend by channel), summing over the channels in `plan`.

# %%
channel_value = truth_all["channel_value"]  # the synthetic world's link between the two lists
new_customers_per_dollar = channel_value["new_customers_per_dollar"]
clv = channel_value["clv_margin_by_media_channel"]  # future margin per new customer, $
mix = pd.DataFrame(channel_value["media_to_acquisition"]).T.loc[CHANNELS]
mix.columns = [f"share recorded as {a}" for a in mix.columns]
mix.assign(**{"cost per new customer ($)": [channel_value["cost_per_new_customer"][c]
                                           for c in CHANNELS],
              "CLV, margin ($)": [clv[c] for c in CHANNELS],
              "long-run margin per $ of spend": [new_customers_per_dollar[c] * clv[c]
                                                 for c in CHANNELS]}).round(3)


# %% tags=["exercise"]
def long_run_value(plan, curves, margin, new_customers_per_dollar, clv):
    # TODO 6: margin * curves[c](s) + s * new_customers_per_dollar[c] * clv[c], summed over plan
    raise NotImplementedError("TODO 6")


# %% tags=["solution"]
# @title Solution 6 — try it yourself first { display-mode: "form" }
@workshop.solution(6)
def long_run_value(plan, curves, margin, new_customers_per_dollar, clv):
    return sum(margin * curves[c](s) + s * new_customers_per_dollar[c] * clv[c]
               for c, s in plan.items())


# %% [markdown]
# **Explain.** Compare with your guess, using the plans below. The CLV term adds a constant
# amount per dollar to each channel's marginal value. Which column of the table above decides
# which channel gains, and how sure would you be of it with real data?
#
# <details><summary>Why this solution works</summary>
#
# The long-run term is linear in spend, so it raises each channel's marginal value by
# $n_c \cdot \text{CLV}_c$ (the last column) and the allocator moves money toward channels where
# that sum is largest, until diminishing short-run returns or a bound stop it. tv gains: it
# costs as much per new customer as search, but its customers arrive mostly by referral, the
# most loyal acquisition channel. The short-run curves are nearly flat around the optimum (tv
# and search share one marginal ROAS there), so a modest difference in long-run value per dollar
# moves thousands of dollars a week while total value hardly changes. With real data the cost
# per new customer and the CLV are estimates with intervals (Module 5); a CLV-weighted plan is
# only as good as those numbers.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(6):
    no_clv = {c: 0.0 for c in CHANNELS}
    short_run = GROSS_MARGIN * sum(mean_curves[c](plan_mean[c]) for c in CHANNELS)
    checks.close(long_run_value(plan_mean, mean_curves, GROSS_MARGIN, new_customers_per_dollar,
                                no_clv), short_run, rel=1e-9,
                 name="long_run_value with every CLV at 0 (should be margin x short-run sales)")
    extra = sum(plan_mean[c] * new_customers_per_dollar[c] * clv[c] for c in CHANNELS)
    checks.close(long_run_value(plan_mean, mean_curves, GROSS_MARGIN, new_customers_per_dollar,
                                clv), short_run + extra, rel=1e-9,
                 name="long_run_value with the CLVs")

    def long_run_curves(clv_values):
        """One curve per channel: the long-run value of spend s on that channel alone."""
        return {c: (lambda s, c=c: long_run_value(pd.Series({c: s}), mean_curves, GROSS_MARGIN,
                                                  new_customers_per_dollar, clv_values))
                for c in CHANNELS}

    plan_clv = allocate(BUDGET, BOUNDS, long_run_curves(clv))[CHANNELS]
    plan_more = allocate(BUDGET, BOUNDS, long_run_curves({**clv, "social": 2 * clv["social"]}))
    assert plan_more["social"] >= plan_clv["social"] - 1e-3 * BUDGET, (
        f"Doubling social's CLV lowered social's spend from ${plan_clv['social']:,.0f} to"
        f" ${plan_more['social']:,.0f}: the CLV term must add spend x customers per dollar x CLV."
    )
    # On the TRUE curves, the CLV-weighted plan must be the truth's long-run optimum.
    true_long_run = {c: (lambda s, c=c: long_run_value(pd.Series({c: s}), true_curves,
                                                       GROSS_MARGIN, new_customers_per_dollar,
                                                       clv)) for c in CHANNELS}
    plan_lr_true = allocate(BUDGET, BOUNDS, true_long_run)[CHANNELS]
    lr_key = pd.Series(channel_value["long_run_optimal_allocation"]["mmm"]["optimal_allocation"])
    off = (plan_lr_true - lr_key[CHANNELS]).abs() / BUDGET
    assert off.max() <= 0.01, (
        f"On the true curves the CLV-weighted plan should be the long-run optimum"
        f" {lr_key[CHANNELS].round(0).to_dict()}; yours is {plan_lr_true.round(0).to_dict()}"
        f" ({off.max():.1%} of the budget off for {off.idxmax()}). Check that the long-run term"
        " is spend x new customers per dollar x CLV and that the margin multiplies only the"
        " short-run sales."
    )

# %%
pd.DataFrame({"mean-optimal (short run)": plan_pymc, "CLV-weighted, model curves": plan_clv,
              "change": plan_clv - plan_pymc, "CLV-weighted, true curves": plan_lr_true,
              "answer key (long run)": lr_key[CHANNELS]}).round(0)

# %% [markdown]
# ## Decision · Next quarter's weekly budget
#
# **Number.** For each plan: weekly spend by channel, expected incremental weekly sales with a
# 94% HDI, the expected gain over the current plan with a 94% HDI, the 10% quantile, and the
# posterior probability that the plan beats the current plan, computed draw by draw.
#
# **Rule.** Recommend the mean-optimal plan unless its 10% quantile is below the current plan's
# 10% quantile; then recommend the risk-averse plan if its 10% quantile beats the current plan's,
# and otherwise keep the current plan. Use the CLV-weighted plan only if the cost per new
# customer and the CLV by channel are measured (Module 5's model and an acquisition count per
# channel), and say so. Here they come from the synthetic world's truth file, not from a model
# you fitted, so the CLV-weighted plan is shown as the long-run alternative, not recommended.

# %%
candidates = {"current": CURRENT, "mean-optimal": plan_pymc,
              "risk-averse (best 10% quantile)": plan_risk, "CLV-weighted": plan_clv}
rows = []
for name, plan in candidates.items():
    d = weekly_sales_draws(plan)
    gain = d - draws_current
    lo, hi = checks.interval(d, 0.94, "hdi")
    g_lo, g_hi = checks.interval(gain, 0.94, "hdi") if name != "current" else (0.0, 0.0)
    rows.append({"plan": name, **{c: plan[c] for c in CHANNELS},
                 "expected weekly sales": d.mean(), "HDI low": lo, "HDI high": hi,
                 "gain vs current": gain.mean(), "gain HDI low": g_lo, "gain HDI high": g_hi,
                 "10% quantile": np.quantile(d, 0.10),
                 "P(beats current)": float((gain > 0).mean()) if name != "current" else np.nan})
decision = pd.DataFrame(rows).set_index("plan")
q10 = decision["10% quantile"]
if q10["mean-optimal"] >= q10["current"]:
    choice = "mean-optimal"
elif q10["risk-averse (best 10% quantile)"] > q10["current"]:
    choice = "risk-averse (best 10% quantile)"
else:
    choice = "current"
mode = "QUICK run: treat this as a rough answer" if QUICK else "FULL run"
pick = decision.loc[choice]
print(f"Weekly budget ${BUDGET:,.0f} for {WEEKS} weeks; 94% HDIs; {mode}.")
print(f"10% quantiles: current ${q10['current']:,.0f}, mean-optimal ${q10['mean-optimal']:,.0f}"
      f" -> the rule picks the {choice} plan:")
print("  weekly spend " + ", ".join(f"{c} ${pick[c]:,.0f}" for c in CHANNELS))
print(f"  expected incremental sales ${pick['expected weekly sales']:,.0f} a week"
      f" (94% HDI ${pick['HDI low']:,.0f} to ${pick['HDI high']:,.0f})")
if choice != "current":
    print(f"  gain over the current plan ${pick['gain vs current']:,.0f} a week (94% HDI"
          f" ${pick['gain HDI low']:,.0f} to ${pick['gain HDI high']:,.0f});"
          f" P(beats current) = {pick['P(beats current)']:.2f}")
decision.T.round(2)

# %% [markdown]
# **Recommendation.** Write one sentence a manager could act on: the weekly spend by channel,
# the expected incremental sales and their range, the probability that the plan beats the
# current one, and what would change it (a new lift test on the channel with the widest
# response band, Module 9; measured acquisition costs and CLVs for the CLV-weighted plan).
#
# ```text
# Your sentence: ________________________________________________
# ```
#
# Then score every plan on the true curves, which a real planner never sees:

# %%
decision.assign(**{"true weekly sales": [sum(true_response(candidates[p][c], c) for c in CHANNELS)
                                         for p in decision.index]})[
    CHANNELS + ["expected weekly sales", "true weekly sales"]].round(0)

# %% [markdown]
# ## Stretch (optional) · Tail risk and a contract
#
# 1. Optimize the conditional value at risk instead of the 10% quantile:
#    `utility_function=conditional_value_at_risk(confidence_level=0.90)` from
#    `pymc_marketing.mmm.utility`. Compare its plan and its CVaR with Exercise 5's table.
# 2. tv has a contract: at least \$40,000 a week. Raise tv's lower bound and rerun your
#    `allocate` and PyMC-Marketing's optimizer. What does the contract cost in expected weekly
#    sales, with a 94% HDI?
#
# Each new optimizer compiles the model again (a few seconds to a minute).
#
# <details><summary>Code for both</summary>
#
# ```python
# from pymc_marketing.mmm.utility import conditional_value_at_risk
# opt_cvar = mmm.budget_optimizer(start_date=start, end_date=end,
#                                 utility_function=conditional_value_at_risk(confidence_level=0.90))
# plan_cvar = pymc_allocation(opt_cvar, BUDGET, BOUNDS, x0=plan_pymc)
# print(allocation_risk(weekly_sales_draws(plan_cvar)))
#
# bounds_contract = {**BOUNDS, "tv": (40_000.0, BOUNDS["tv"][1])}
# start_contract = allocate(BUDGET, bounds_contract, mean_curves)  # your Exercise 3 allocator
# plan_contract = pymc_allocation(opt, BUDGET, bounds_contract, x0=start_contract)
# cost = weekly_sales_draws(plan_pymc) - weekly_sales_draws(plan_contract)
# print(plan_contract.round(0).to_dict(), cost.mean(), checks.interval(cost, 0.94, "hdi"))
# ```
# </details>
