# %% [markdown]
# <!--
# APIs checked for this lab (2026-10-09) against the installed source of SciPy 1.16.3 in the
# workshop's environment: `scipy.optimize.minimize(fun, x0, jac=..., method="SLSQP", bounds=...,
# constraints=[{"type": "eq", "fun": ...}], options={"maxiter": ..., "ftol": ...})` (constraints as
# a list of dicts; "eq" means the function must equal zero) and `scipy.optimize.nnls(A, b)`, which
# returns `(x, rnorm)`. Everything else is NumPy, pandas and Matplotlib.
# -->
#
# **Order of work.** Module 7's lab has two notebooks in one 55-minute slot: this Python
# notebook, which you work first and finish with, and an R notebook,
# `labs/r/07-causalimpact-geolift.ipynb`.
#
# 1. Open the R notebook and run only its install cell: CausalImpact and GeoLift are the
#    slowest installs of the lab, so start them now and let them run.
# 2. Work Parts A and B here: Exercises 1 to 4, about 28 minutes.
# 3. About 33 minutes into the lab, switch to the R notebook even if Part B is unfinished
#    (`workshop.use_reference(n)` gets you past it): Part C there, Exercises 1 and 2, about 12
#    minutes, and its short decision cell.
# 4. Come back here for the Decision (5 minutes) and type CausalImpact's interval into it.
#
# **How this lab runs.** No sampler: every fit is a small least-squares problem that takes
# milliseconds. `QUICK` only shrinks the power simulation of Exercise 4 (counts printed below);
# with QUICK on, its power estimates are noisier.
#
# Terms used throughout:
#
# - A **geo** is a sales region. The retailer ran a regional campaign in some geos, the
#   **treated** geos, for ten weeks, the **test window**; the other geos are **controls**. The weeks
#   before the test window are the **pre-period**.
# - The **counterfactual** is what the treated geos would have sold in the test window without the
#   campaign. It is never observed; every method here estimates it from the controls.
# - **Incremental sales** = actual sales − counterfactual sales, and the **lift** = incremental
#   sales ÷ counterfactual sales.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from threadpoolctl import threadpool_limits

from mktstats.data import load_prop99, load_synthetic, load_truth

# Every fit here is tiny, so one linear-algebra thread is fastest. It also avoids a large
# slowdown where a container sees more cores than it is allowed to use.
threadpool_limits(1, user_api="blas")

panel = load_synthetic("geo_panel")
truth_all = load_truth()
truth = truth_all["geo_panel"]
TREATED = truth["treated_geos"]
TEST_START, TEST_END = pd.Timestamp(truth["test_start"]), pd.Timestamp(truth["test_end"])
LIFT_TOL = truth_all["tolerances"]["geo_did"]["lift_pct"]["abs"]  # percentage points, see Exercise 1

SEED = 2029  # the power simulation starts from np.random.default_rng(SEED + 4)
N_POWER_SIMS = 50 if QUICK else 200  # Exercise 4: simulated tests per lift

print(f"{panel['geo'].nunique()} geos x {panel['date'].nunique()} weeks;"
      f" {len(TREATED)} treated geos: {', '.join(TREATED)}")
print(f"Test window: {TEST_START:%Y-%m-%d} to {TEST_END:%Y-%m-%d} ({truth['test_weeks']} weeks);"
      f" pre-period: {truth['pre_weeks']} weeks")
print(f"QUICK = {QUICK}: {N_POWER_SIMS} simulated tests per lift in Exercise 4")
panel.head()

# %% [markdown]
# Look at whether the treated and control lines move together **before** the shaded test window.
# Each geo's sales are divided by its own pre-period average (1.0 = a normal pre-period week),
# so geos of very different sizes can share one axis.

# %%
pre_mean = panel[panel["date"] < TEST_START].groupby("geo")["sales"].mean()
indexed = panel.assign(index=panel["sales"] / panel["geo"].map(pre_mean),
                       group=np.where(panel["geo"].isin(TREATED), "treated geos", "control geos"))
lines = indexed.groupby(["date", "group"])["index"].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(lines.index, lines["control geos"], color="#4c72b0", label="control geos (average)")
ax.plot(lines.index, lines["treated geos"], color="#dd8452", linestyle="--", label="treated geos (average)")
ax.axvspan(TEST_START, TEST_END, color="grey", alpha=0.2, label="test window")
ax.set_ylabel("Sales index\n(1 = pre-period average)")
ax.set_xlabel("Week")
ax.legend(loc="upper left")
fig.tight_layout()

# %% [markdown]
# # Part A · Two estimators
#
# **Difference-in-differences (DiD)** takes the change in the treated geos from the pre-period to
# the test window and subtracts the change in the control geos over the same weeks. The controls'
# change stands in for what would have happened to the treated geos without the campaign. That
# only works under **parallel trends**: without the campaign, both groups would have moved by the
# same amount.
#
# The campaign multiplies sales, and the geos differ in size by a factor of ten or more, so we
# work with log sales: on the log scale a 5% lift adds the same amount to a small geo and a
# large one, and seasonality moves all geos by similar percentages. With $\bar{L}$ the mean of
# log weekly sales over the geo-weeks of a group and period,
#
# $$ \delta = \big(\bar{L}_{\text{treated, test}} - \bar{L}_{\text{treated, pre}}\big)
#            - \big(\bar{L}_{\text{control, test}} - \bar{L}_{\text{control, pre}}\big), \qquad
#    \text{lift} = e^{\delta} - 1, $$
#
# and incremental sales = actual treated sales in the test window × (1 − e^{−δ}): the actual sales
# minus the counterfactual, actual ÷ (1 + lift).

# %% [markdown]
# ## Exercise 1 · Difference-in-differences (7 minutes)
#
# **Predict.** Look at the plot above. Will the **naive** before/after change in the treated geos
# (test window against pre-period, ignoring the controls) be larger or smaller than the
# difference-in-differences lift?
#
# **Task.** Write `did(panel, treated_geos, test_start, test_end)` using only rows with
# `date <= test_end`, returning a dict with
#
# - `"lift_pct"`: 100 × (e^δ − 1);
# - `"incremental_sales"`: actual treated sales in the test window × (1 − e^{−δ});
# - `"effect_per_geo_week"`: incremental sales ÷ the number of treated geo-weeks in the test window;
# - `"naive_lift_pct"`: 100 × (e^{change of the treated geos alone} − 1).
#
# Hint: `np.log(p["sales"]).groupby([is_treated, in_test]).mean()` gives the four means at once,
# indexed by `(True/False, True/False)`.

# %% tags=["exercise"]
def did(panel, treated_geos, test_start, test_end):
    # TODO 1: the four means of log sales, delta, lift, incremental sales, naive change
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def did(panel, treated_geos, test_start, test_end):
    p = panel[panel["date"] <= test_end]
    is_treated = p["geo"].isin(treated_geos)
    in_test = p["date"] >= test_start
    m = np.log(p["sales"]).groupby([is_treated, in_test]).mean()
    change_treated = m[(True, True)] - m[(True, False)]
    delta = change_treated - (m[(False, True)] - m[(False, False)])
    incremental = p.loc[is_treated & in_test, "sales"].sum() * (1 - np.exp(-delta))
    return {"lift_pct": float(100 * np.expm1(delta)),
            "incremental_sales": float(incremental),
            "effect_per_geo_week": float(incremental / (is_treated & in_test).sum()),
            "naive_lift_pct": float(100 * np.expm1(change_treated))}


# %% [markdown]
# The checkpoint compares your lift with the true 5% lift. The tolerance, ±2.5 percentage
# points, is `tolerances.geo_did` in `data/synthetic/truth.json`: how far this log-scale
# difference-in-differences lands from the truth across 41 independently generated panels.

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    did_result = did(panel, TREATED, TEST_START, TEST_END)
    checks.columns(pd.DataFrame([did_result]),
                   ["lift_pct", "incremental_sales", "effect_per_geo_week", "naive_lift_pct"],
                   name="the dict did() returns")
    gap = abs(did_result["lift_pct"] - truth["lift_pct"])
    assert gap <= LIFT_TOL, (
        f"Your difference-in-differences lift is {did_result['lift_pct']:.2f}%; the truth is"
        f" {truth['lift_pct']:.1f}%, more than {LIFT_TOL} points away. Check that you subtract the"
        " CONTROL geos' change, use log sales, and use only rows up to test_end."
        " Fix TODO 1, or run workshop.use_reference(1)."
    )
    in_window = panel["geo"].isin(TREATED) & panel["date"].between(TEST_START, TEST_END)
    actual = panel.loc[in_window, "sales"].sum()
    checks.close(did_result["incremental_sales"], actual * (1 - 1 / (1 + did_result["lift_pct"] / 100)),
                 rel=1e-6, name="incremental sales implied by your lift (actual x (1 - 1/(1 + lift)))")
    checks.close(did_result["effect_per_geo_week"],
                 did_result["incremental_sales"] / (len(TREATED) * truth["test_weeks"]),
                 rel=1e-6, name="effect per treated geo-week")

print(f"Naive before/after change in treated geos: {did_result['naive_lift_pct']:+.2f}%")
print(f"Difference-in-differences lift:            {did_result['lift_pct']:+.2f}%"
      f" (truth {truth['lift_pct']:.1f}%)")
print(f"Incremental sales: {did_result['incremental_sales']:,.0f}"
      f" ({did_result['effect_per_geo_week']:,.0f} per treated geo-week)")
geo_region = panel.drop_duplicates("geo").set_index("geo")["region"]
print("Geos per region, treated / control:",
      ", ".join(f"{r} {n_t}/{n_c}" for r, n_t, n_c in zip(
          sorted(geo_region.unique()),
          geo_region[TREATED].value_counts().reindex(sorted(geo_region.unique()), fill_value=0),
          geo_region.drop(TREATED).value_counts().reindex(sorted(geo_region.unique()), fill_value=0))))

# %% [markdown]
# **Explain.** Compare with your prediction. Which part of the naive change was not caused by the
# campaign? Point to the control geos' line in the plot.
#
# <details><summary>Why this solution works</summary>
#
# The naive change includes everything that moved sales between the pre-period and the test
# window: growth over the two years and the season of the test weeks. The control geos had the
# same growth and season, so subtracting their change removes them and leaves the campaign. The
# estimate is still off by a little because each region has its own shocks, so trends are only
# approximately parallel: the treated and control geos are not spread over the regions in the same
# proportions (printed below the checkpoint). A longer test,
# more treated geos spread over all regions, or a better-matched comparison (next exercise)
# reduces that error.
# </details>

# %% [markdown]
# **Synthetic control** (Abadie, Diamond & Hainmueller 2010) builds the comparison from a
# weighted average of control geos, with weights chosen so that the average tracks the treated
# series in the pre-period. The weights are non-negative and add up to one, so the synthetic
# control is an interpolation between real geos, never an extrapolation.
#
# No weighted average of single geos can reach the total of eight treated geos, so we match
# **shapes**: the next cell divides the treated total and each control geo by its own pre-period
# average, fits the weights on these indexes, and `sc_effect` converts the result back to sales.

# %%
wide = panel.pivot(index="date", columns="geo", values="sales").sort_index()
CONTROLS = [g for g in wide.columns if g not in TREATED]
pre = np.asarray(wide.index < TEST_START)  # True in the pre-period, False in the test window
y_total = wide[TREATED].sum(axis=1).to_numpy(dtype=float)  # treated geos' total weekly sales
X_controls = wide[CONTROLS].to_numpy(dtype=float)  # one column per control geo
y_index = y_total / y_total[pre].mean()
X_index = X_controls / X_controls[pre].mean(axis=0)


def sc_effect(weights, y, X, pre):
    """The synthetic control of y from the columns of X, in the units of y.

    The weights were fitted on series indexed to their pre-period means; `pre` marks the
    pre-period weeks and every other week is in the test window.
    """
    y_idx, X_idx = y / y[pre].mean(), X / X[pre].mean(axis=0)
    synthetic = (X_idx @ weights) * y[pre].mean()
    incremental = (y - synthetic)[~pre].sum()
    return {"synthetic": synthetic,
            "incremental_sales": float(incremental),
            "lift_pct": float(100 * incremental / (y[~pre].sum() - incremental)),
            "pre_rmspe": float(np.sqrt(np.mean((y_idx - X_idx @ weights)[pre] ** 2)))}


print(f"y_index: {y_index.shape}, X_index: {X_index.shape} ({len(CONTROLS)} control geos)")

# %% [markdown]
# ## Exercise 2 · Synthetic-control weights (9 minutes)
#
# **Predict.** There are 32 control geos. Will the synthetic control put weight on most of them,
# or on a handful? Write a number of geos with a weight above 1%.
#
# **Task.** Write `sc_weights(y_pre, X_pre)` returning the weights w (one per column of `X_pre`)
# that minimize ‖y_pre − X_pre·w‖² subject to w ≥ 0 and Σw = 1. Use
# `minimize(loss, x0, method="SLSQP", bounds=[(0, 1)] * J, constraints=[{"type": "eq", "fun": ...}])`
# with equal starting weights `x0 = np.full(J, 1 / J)`. SLSQP (sequential least-squares
# programming) is SciPy's optimizer for smooth problems with bounds and equality constraints. Passing the gradient
# `jac=lambda w: -2 * X_pre.T @ (y_pre - X_pre @ w)` makes it faster and more precise.

# %% tags=["exercise"]
def sc_weights(y_pre, X_pre):
    # TODO 2: squared-error loss, SLSQP with bounds (0, 1) and a sum-to-one equality constraint
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def sc_weights(y_pre, X_pre):
    y_pre, X_pre = np.asarray(y_pre, dtype=float), np.asarray(X_pre, dtype=float)
    n_controls = X_pre.shape[1]
    result = minimize(
        lambda w: np.sum((y_pre - X_pre @ w) ** 2),
        np.full(n_controls, 1 / n_controls),
        jac=lambda w: -2 * X_pre.T @ (y_pre - X_pre @ w),
        method="SLSQP",
        bounds=[(0, 1)] * n_controls,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


# %% [markdown]
# The checkpoint also solves the same problem a second way, as a non-negative least-squares
# problem (`nnls`) with a heavily weighted extra row that forces the weights to add up to one,
# and checks that your pre-period fit is as good (within 5%).

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    weights = np.asarray(sc_weights(y_index[pre], X_index[pre]), dtype=float)
    assert weights.shape == (len(CONTROLS),), (
        f"sc_weights should return one weight per control geo ({len(CONTROLS)}), got shape"
        f" {weights.shape}. Return result.x."
    )
    assert weights.min() >= -1e-8 and abs(weights.sum() - 1) <= 1e-6, (
        f"The weights must be >= 0 and add up to 1; yours range from {weights.min():.3g} and add up"
        f" to {weights.sum():.6f}. Check bounds=[(0, 1)] * J and the equality constraint"
        " {'type': 'eq', 'fun': lambda w: w.sum() - 1}. Fix TODO 2, or run workshop.use_reference(2)."
    )
    sc_result = sc_effect(weights, y_total, X_controls, pre)
    big = 1e3  # weight of the sum-to-one row
    w_check, _ = nnls(np.vstack([X_index[pre], big * np.ones(len(CONTROLS))]), np.append(y_index[pre], big))
    best_rmspe = sc_effect(w_check, y_total, X_controls, pre)["pre_rmspe"]
    assert sc_result["pre_rmspe"] <= 1.05 * best_rmspe, (
        f"Your synthetic control misses the treated index by {sc_result['pre_rmspe']:.4f} per week in"
        f" the pre-period (root mean squared); the best possible is about {best_rmspe:.4f}. The"
        " optimizer stopped early: check the loss (sum of squared errors) and try"
        " options={'maxiter': 1000, 'ftol': 1e-12}."
    )
    assert abs(sc_result["lift_pct"] - truth["lift_pct"]) <= LIFT_TOL, (
        f"The lift from your weights is {sc_result['lift_pct']:.2f}%; the truth is"
        f" {truth['lift_pct']:.1f}% (tolerance {LIFT_TOL} points). Check that you fitted on the"
        " pre-period rows only (y_index[pre], X_index[pre])."
    )

print(f"Synthetic control: lift {sc_result['lift_pct']:+.2f}% (truth {truth['lift_pct']:.1f}%),"
      f" incremental sales {sc_result['incremental_sales']:,.0f} (truth {truth['incremental_sales']:,.0f})")
print(f"Pre-period fit error (RMSPE, index units): {sc_result['pre_rmspe']:.4f}")
top = pd.Series(weights, index=CONTROLS).sort_values(ascending=False)
print(f"Weights above 1%: {(top > 0.01).sum()} of {len(CONTROLS)} control geos. Largest:")
print(pd.DataFrame({"weight": top.head(8).round(3), "region": geo_region[top.head(8).index]}).to_string())

# %% [markdown]
# Look at how closely the dashed synthetic control follows the treated total before the test
# window, and at the gap that opens inside it. The *RMSPE* printed above (root mean squared
# prediction error) is the typical size of that gap per pre-period week, in index units: the
# smaller, the closer the fit.

# %%
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(wide.index, y_total / 1000, color="#dd8452", label="treated geos (actual)")
ax.plot(wide.index, sc_result["synthetic"] / 1000, color="#4c72b0", linestyle="--",
        label="synthetic control (counterfactual)")
ax.axvspan(TEST_START, TEST_END, color="grey", alpha=0.2, label="test window")
ax.set_ylabel("Weekly sales of the\ntreated geos (thousands)")
ax.set_xlabel("Week")
ax.legend(loc="upper left")
fig.tight_layout()

# %% [markdown]
# **Explain.** Compare with your prediction about the number of geos. Why does a total of eight
# geos from four regions need weight on many control geos, where one treated geo is often matched
# by a few? Point to the region column of the weights table.
#
# <details><summary>Why this solution works</summary>
#
# The weights build a comparison unit that tracks the treated series before the campaign, so the
# test-window gap is the campaign's effect plus whatever the controls fail to reproduce. The
# treated total mixes geos from every region, each with its own seasonal amplitude and regional
# shocks, so matching it needs control geos from every region; a single treated geo is usually
# matched by a few similar geos. Weights are often sparse (many exact zeros) because of the
# bounds: the optimum sits on the edge of the allowed region. A poor pre-period fit, or a treated
# unit outside the range of the controls, is the warning sign: then the method extrapolates and
# its estimate is not trustworthy.
# </details>

# %% [markdown]
# **The same estimator on real data: California's Proposition 99.** Synthetic control was made
# famous by a tobacco-control program, not a marketing campaign: in 1988 California passed
# Proposition 99, and Abadie, Diamond & Hainmueller (2010) estimated
# its effect on cigarette sales by building a synthetic California from 38 states without such a
# program. It is the canonical synthetic-control dataset, and the question is the one a geo test
# asks: what would have happened without the intervention? The outcome is per-capita cigarette
# sales in packs (`cigsale`), 1970–2000; the pre-period is 1970–1988.
#
# Look at the sign and size of the gap after 1988. Abadie et al. report about 26 fewer packs per
# capita by 2000 with weights that also match covariates (income, beer consumption, the share of
# young people, the retail price); this outcome-only version need not match exactly. (We divide
# every series by one common number before fitting; that leaves the weights unchanged and keeps
# the optimizer's numbers near 1.)

# %%
prop99 = load_prop99()
cig = prop99.pivot(index="year", columns="state", values="cigsale")
pre99 = np.asarray(cig.index <= 1988)
california = cig["California"].to_numpy(dtype=float)
donors = cig.drop(columns="California")
scale99 = california[pre99].mean()
w99 = np.asarray(sc_weights(california[pre99] / scale99, donors.to_numpy(dtype=float)[pre99] / scale99))
gap99 = california - donors.to_numpy(dtype=float) @ w99

with workshop.checkpoint(label="Prop 99"):
    rmspe99 = float(np.sqrt(np.mean(gap99[pre99] ** 2)))
    assert rmspe99 < 5, (
        f"Synthetic California misses the real one by {rmspe99:.1f} packs per year before 1989:"
        " the weights did not fit. Check sc_weights (TODO 2)."
    )
    assert gap99[~pre99].mean() < 0, "The gap after 1988 should be negative (fewer packs sold)."

print(f"Gap in 2000: {gap99[-1]:+.1f} packs per capita; average 1989-2000: {gap99[~pre99].mean():+.1f};"
      f" pre-1989 fit error (RMSPE) {rmspe99:.1f} packs")
print("Donor states with weight above 1%:",
      ", ".join(f"{s} {w:.2f}" for s, w in sorted(zip(donors.columns, w99), key=lambda x: -x[1]) if w > 0.01))
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(cig.index, gap99, color="black", marker="o", markersize=3, label="California - synthetic California")
ax.axhline(0, color="grey", linewidth=1)
ax.axvline(1988.5, color="grey", linestyle="--", label="Proposition 99 (1988)")
ax.set_ylabel("Gap in cigarette sales\n(packs per capita)")
ax.set_xlabel("Year")
ax.legend(loc="lower left")
fig.tight_layout()

# %% [markdown]
# # Part B · Is it real, and could we have seen it?
#
# With one treated unit there is no sampling distribution to compute a p-value from. An
# **in-space placebo test** makes one: pretend each control geo was treated, fit its synthetic
# control from the other controls, and see how often a fake treatment looks as large as the real
# one. The statistic is the **RMSPE ratio**: the root mean squared gap in the test window divided
# by the root mean squared gap in the pre-period. A large ratio means the fit was good before and
# broke in the test window. The next cell runs the 32 placebo fits.

# %%
def rmspe_ratio(gap, pre):
    """Root mean squared gap in the test window over the same in the pre-period."""
    return float(np.sqrt(np.mean(gap[~pre] ** 2)) / np.sqrt(np.mean(gap[pre] ** 2)))


weights = np.asarray(sc_weights(y_index[pre], X_index[pre]), dtype=float)
treated_gap = y_index - X_index @ weights
treated_ratio = rmspe_ratio(treated_gap, pre)
placebo_gaps, placebo_ratios = [], []
for j in range(len(CONTROLS)):
    others = np.delete(np.arange(len(CONTROLS)), j)
    w_j = np.asarray(sc_weights(X_index[pre, j], X_index[pre][:, others]), dtype=float)
    placebo_gaps.append(X_index[:, j] - X_index[:, others] @ w_j)
    placebo_ratios.append(rmspe_ratio(placebo_gaps[-1], pre))
placebo_ratios = np.array(placebo_ratios)
print(f"Treated RMSPE ratio: {treated_ratio:.2f}; placebo ratios from {placebo_ratios.min():.2f}"
      f" to {placebo_ratios.max():.2f}")

# %% [markdown]
# ## Exercise 3 · In-space placebo test (6 minutes)
#
# **Predict.** If the campaign had had no effect, where would the treated total's RMSPE ratio rank
# among the 33 ratios (the treated total and 32 placebos): near the top, in the middle, or near the
# bottom?
#
# **Task.** Write `permutation_pvalue(treated_stat, placebo_stats)` returning
# (1 + the number of placebos whose statistic is at least `treated_stat`) ÷ (1 + the number of
# placebos). The 1 in the numerator counts the treated unit itself.

# %% tags=["exercise"]
def permutation_pvalue(treated_stat, placebo_stats):
    # TODO 3: (1 + number of placebo_stats >= treated_stat) / (1 + number of placebos)
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def permutation_pvalue(treated_stat, placebo_stats):
    placebo_stats = np.asarray(placebo_stats, dtype=float)
    return float((1 + np.sum(placebo_stats >= treated_stat)) / (1 + placebo_stats.size))


# %% [markdown]
# The checkpoint also runs the test on a copy of the panel with the campaign removed (the treated
# geos' test-window sales divided by 1 + the true lift): there, the treated total should not stand
# out.

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    checks.close(permutation_pvalue(5.0, [1.0, 2.0, 3.0, 4.0]), 1 / 5, rel=1e-12,
                 name="permutation_pvalue(5, [1, 2, 3, 4]) (treated largest: 1 / (J + 1))")
    checks.close(permutation_pvalue(0.5, [1.0, 2.0, 3.0, 4.0]), 1.0, rel=1e-12,
                 name="permutation_pvalue(0.5, [1, 2, 3, 4]) (treated smallest)")
    checks.close(permutation_pvalue(2.0, [1.0, 2.0, 3.0]), 3 / 4, rel=1e-12,
                 name="permutation_pvalue(2, [1, 2, 3]) (a tie counts as at least as large)")
    p_value = permutation_pvalue(treated_ratio, placebo_ratios)
    checks.close(p_value, (1 + np.sum(placebo_ratios >= treated_ratio)) / (1 + len(placebo_ratios)),
                 rel=1e-12, name="the placebo p-value on the panel")
    y_no_lift = np.where(pre, y_total, y_total / (1 + truth["lift_pct"] / 100))
    y_no_lift_index = y_no_lift / y_no_lift[pre].mean()
    w_no_lift = np.asarray(sc_weights(y_no_lift_index[pre], X_index[pre]), dtype=float)
    p_no_lift = permutation_pvalue(rmspe_ratio(y_no_lift_index - X_index @ w_no_lift, pre), placebo_ratios)
    assert p_no_lift > 0.1, (
        f"With the campaign removed, the placebo p-value is {p_no_lift:.3f}; it should be above 0.1"
        " (nothing to find). Check that placebos at least as large as the treated statistic"
        " count (>=), and that you divide by 1 + the number of placebos."
    )
    print(f"Placebo p-value: {p_value:.3f} (smallest possible 1/{len(placebo_ratios) + 1} ="
          f" {1 / (len(placebo_ratios) + 1):.3f}); with the campaign removed: {p_no_lift:.3f}")

# %% [markdown]
# Look at whether the colored treated gap leaves the grey band of placebo gaps inside the test
# window, and stays inside it before.

# %%
fig, ax = plt.subplots(figsize=(8, 3.5))
for g in placebo_gaps:
    ax.plot(wide.index, 100 * g, color="grey", alpha=0.35, linewidth=0.8)
ax.plot([], [], color="grey", alpha=0.6, label=f"{len(placebo_gaps)} placebo geos")
ax.plot(wide.index, 100 * treated_gap, color="#dd8452", linewidth=2.2, label="treated geos")
ax.axvspan(TEST_START, TEST_END, color="grey", alpha=0.2, label="test window")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Gap: actual - synthetic\n(% of a pre-period week)")
ax.set_xlabel("Week")
ax.legend(loc="lower left")
fig.tight_layout()

# %% [markdown]
# **Explain.** Compare with your prediction. What is the smallest p-value this test can ever
# give with 32 control geos, and what would you need to make it smaller? Point to the printed line.
#
# <details><summary>Why this solution works</summary>
#
# Under "no effect anywhere", the treated unit is one more draw from the same process as the
# placebos, so its rank among the 33 ratios is equally likely to be any of 1 to 33; the p-value is
# its rank from the top divided by 33. That makes 1/33 ≈ 0.03 the smallest possible value: more
# control units, not more weeks, are the only way to get a smaller one. The ratio, rather than the
# test-window gap alone, keeps placebo geos that are hard to fit (large gaps everywhere) from
# looking like effects. With the campaign removed, the treated total sits among the placebos.
# </details>

# %% [markdown]
# **Power** is the chance that a test detects an effect of a given size. For a geo test it can be
# estimated before spending anything: take only pre-period data, where there was no campaign,
# pretend a 10-week window was the test, inject a known lift into the treated geos, and see how
# often the estimate stands out from what the same windows give without the injected lift.

# %%
pre_panel = panel[panel["date"] < TEST_START]
print(f"Pre-period panel: {pre_panel['date'].nunique()} weeks; a test window can start after week 26")

# %% [markdown]
# ## Exercise 4 · Power by simulation (6 minutes)
#
# **Predict.** With these geos and a 10-week test, what is the chance of detecting a true 5% lift:
# under 50%, 50 to 80%, or over 80%?
#
# **Task.** Write `geo_power(pre_panel, treated_geos, lift_pct, n_sims, rng)`:
#
# 1. take the sorted weeks of `pre_panel`, and draw `n_sims` window starts with
#    `rng.integers(26, n_weeks - 10 + 1, size=n_sims)` (at least 26 weeks before each window);
# 2. for each window, keep the rows up to the window's last week, and run `did` twice: on the data
#    as they are (no lift) and with the treated geos' sales in the window multiplied by
#    1 + lift_pct / 100;
# 3. return the share of windows whose with-lift estimate exceeds the 95th percentile
#    (`np.quantile(..., 0.95)`) of the no-lift estimates.

# %% tags=["exercise"]
def geo_power(pre_panel, treated_geos, lift_pct, n_sims, rng):
    # TODO 4: random 10-week windows in the pre-period, did() without and with the injected lift
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def geo_power(pre_panel, treated_geos, lift_pct, n_sims, rng):
    weeks = np.sort(pre_panel["date"].unique())
    no_lift, with_lift = [], []
    for s in rng.integers(26, len(weeks) - 10 + 1, size=n_sims):
        start, end = weeks[s], weeks[s + 9]
        window = pre_panel[pre_panel["date"] <= end]
        no_lift.append(did(window, treated_geos, start, end)["lift_pct"])
        boost = window["geo"].isin(treated_geos) & (window["date"] >= start)
        lifted = window.assign(sales=window["sales"] * np.where(boost, 1 + lift_pct / 100, 1.0))
        with_lift.append(did(lifted, treated_geos, start, end)["lift_pct"])
    return float(np.mean(np.array(with_lift) > np.quantile(no_lift, 0.95)))


# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    lifts = [0, 2, 5, 10]
    power = [geo_power(pre_panel, TREATED, lift, N_POWER_SIMS, np.random.default_rng(SEED + 4))
             for lift in lifts]
    checks.probability(power, name="geo_power results")
    assert power[0] <= 0.15, (
        f"With no injected lift the 'power' is {power[0]:.2f}; it should be about 0.05 (at most 0.15):"
        " the threshold is the 95th percentile of the no-lift estimates. Check that both estimates"
        " use the same windows. Fix TODO 4, or run workshop.use_reference(4)."
    )
    checks.monotone(power, increasing=True, name="power at lifts of 0, 2, 5 and 10%")
    assert power[-1] >= 0.5, (
        f"A 10% lift is detected in only {power[-1]:.0%} of windows. Check that the lift multiplies"
        " the treated geos' sales INSIDE the window only, and that did() sees the window as the test."
    )
    print("Power by true lift:", ", ".join(f"{lift}%: {p:.2f}" for lift, p in zip(lifts, power)),
          f"({N_POWER_SIMS} windows each)")

# %% [markdown]
# Look at the lift where the curve crosses the dashed 80% line: that is the smallest lift this
# design detects reliably, its minimum detectable effect.

# %%
extra = [0.5, 1, 1.5, 3, 4]
power_curve = dict(zip(lifts, power)) | {
    lift: geo_power(pre_panel, TREATED, lift, N_POWER_SIMS, np.random.default_rng(SEED + 4)) for lift in extra}
power_curve = dict(sorted(power_curve.items()))
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(list(power_curve), list(power_curve.values()), marker="o", color="#4c72b0")
ax.axhline(0.8, color="black", linestyle="--", label="80% power")
ax.axvline(truth["lift_pct"], color="#dd8452", linestyle=":", label="true lift of this campaign")
ax.set_xlabel("True lift injected into the treated geos (%)")
ax.set_ylabel("Power (share of\nwindows detected)")
ax.set_title(f"Power of a 10-week test with these geos ({N_POWER_SIMS} windows per point)")
ax.legend(loc="lower right")
fig.tight_layout()
detectable = [lift for lift, p in power_curve.items() if p >= 0.8]
print(f"Smallest lift on this grid with power >= 0.8: {min(detectable) if detectable else 'none'}%")

# %% [markdown]
# **Explain.** Compare with your prediction. Which two things would you change to detect a 1%
# lift reliably, and why does each help? Point to the curve.
#
# <details><summary>Why this solution works</summary>
#
# Each window's no-lift estimate is pure noise: regional shocks the controls do not share. Its
# spread sets the threshold, and power is the share of windows in which the true lift pushes the
# estimate past it. A lift several times larger than that noise is always detected; one of the
# same size is detected about half the time. Power rises with a longer test (noise averages out
# over more weeks), with more treated geos spread over all regions, and with controls that track
# the treated geos better. Choosing treated markets to maximize this power is what GeoLift's market
# selection does in the afternoon clinic.
# </details>

# %% [markdown]
# ## Decision · Was the campaign incremental?
#
# **The interval.** Synthetic control gives no standard error, so we measure how wrong it is
# where we know the answer: in every 10-week window of the pre-period (no campaign, true lift 0)
# we fit it on the weeks before the window and record its estimated lift. Those are its errors.
# The **90% placebo-in-time interval** is the estimate minus the 95th and 5th percentiles of
# those errors.
#
# **Optional: CausalImpact from the R notebook.** Type its cumulative 95% interval for incremental
# sales below, for example `causalimpact_interval = (48_000, 144_000)`.
#
# **Assumptions** (stated, not estimated): gross margin 30% of sales; campaign cost \$20,000.
# Change `CAMPAIGN_COST` in the next cell if your campaign cost differs.
#
# **The rule**, fixed before reading the numbers: call the campaign **incremental** if the
# in-space placebo p-value is below 0.1 and, if entered, the CausalImpact interval excludes 0.
# Call it **profitable** if the incremental margin at the lower end of the 90% interval exceeds the
# campaign cost.

# %%
causalimpact_interval = None  # (lower, upper) cumulative AbsEffect from the R notebook, or None
GROSS_MARGIN = 0.30  # assumption: margin per dollar of sales
CAMPAIGN_COST = 20_000  # assumption: $ spent on the campaign in the treated geos

errors = []
for s in range(26, int(pre.sum()) - 10 + 1):  # every 10-week window with 26 weeks before it
    in_fit = np.arange(s + 10) < s
    y_s, X_s = y_total[: s + 10], X_controls[: s + 10]
    w_s = sc_weights((y_s / y_s[in_fit].mean())[in_fit], (X_s / X_s[in_fit].mean(axis=0))[in_fit])
    errors.append(sc_effect(np.asarray(w_s, dtype=float), y_s, X_s, in_fit)["lift_pct"])
errors = np.array(errors)
lift_lo = sc_result["lift_pct"] - np.quantile(errors, 0.95)
lift_hi = sc_result["lift_pct"] - np.quantile(errors, 0.05)
actual = y_total[~pre].sum()
inc = sc_result["incremental_sales"]
inc_lo, inc_hi = (actual * x / (100 + x) for x in (lift_lo, lift_hi))
margin, margin_lo, margin_hi = (GROSS_MARGIN * x for x in (inc, inc_lo, inc_hi))

incremental = p_value < 0.1 and (causalimpact_interval is None or min(causalimpact_interval) > 0)
profitable = margin_lo > CAMPAIGN_COST
print("Number")
print(f"  Synthetic control: {inc:,.0f} incremental sales (lift {sc_result['lift_pct']:.1f}%), 90%"
      f" placebo-in-time interval {inc_lo:,.0f} to {inc_hi:,.0f} ({len(errors)} pre-period windows)")
print(f"  In-space placebo p-value: {p_value:.3f}")
print(f"  Difference-in-differences: {did_result['incremental_sales']:,.0f} (lift {did_result['lift_pct']:.1f}%)")
if causalimpact_interval is not None:
    print(f"  CausalImpact 95% interval (R notebook): {causalimpact_interval[0]:,.0f} to {causalimpact_interval[1]:,.0f}")
print(f"  Incremental margin at {GROSS_MARGIN:.0%}: ${margin:,.0f} (90% interval ${margin_lo:,.0f} to"
      f" ${margin_hi:,.0f}); return on ${CAMPAIGN_COST:,}: {margin / CAMPAIGN_COST:.2f}x"
      f" ({margin_lo / CAMPAIGN_COST:.2f}x to {margin_hi / CAMPAIGN_COST:.2f}x)")
print("Rule")
print("  Incremental if the placebo p < 0.1 (and the CausalImpact interval, if entered, excludes 0);")
print("  profitable if margin at the lower end of the 90% interval > campaign cost.")
print("Recommendation")
print(f"  Incremental: {'yes' if incremental else 'not shown'}. Profitable at the lower end:"
      f" {'yes' if profitable else 'not shown'}.")
if incremental and profitable:
    print("  Roll the campaign out to similar regions; plan a follow-up test to confirm the lift.")
elif incremental:
    print("  The campaign moved sales, but it has not been shown to pay back: extend or redesign the test.")
else:
    print("  Do not roll out on this evidence: the effect is not distinguishable from noise.")
print(f"  Mode: {'QUICK run' if QUICK else 'FULL run'} (QUICK changes only the power curve, not this readout).")

# %% [markdown]
# **Decide.** Write one sentence: was the campaign incremental, how much did it add (sales and
# margin, with the range), what return did it earn, and what would change your mind (spillover
# into control geos, which biases the estimate down; another campaign in the same weeks, which
# biases it up; a longer test).
#
# Then run the next cell, which reveals the truth the panel was generated from.

# %%
print(f"True lift: {truth['lift_pct']:.1f}%; true incremental sales: {truth['incremental_sales']:,.0f}"
      f" (inside the 90% interval: {inc_lo <= truth['incremental_sales'] <= inc_hi})")
print(f"True incremental margin: ${GROSS_MARGIN * truth['incremental_sales']:,.0f}"
      f" against a cost of ${CAMPAIGN_COST:,}")

# %% [markdown]
# ## Stretch (optional) · An event-study plot
#
# A difference-in-differences estimate averages over the whole test window. An **event study**
# shows it week by week: the gap in mean log sales between treated and control geos, minus its
# pre-period average. Before the test the bars (called leads) should hover around zero; if they
# trend, parallel trends is in doubt. Plot the last 30 weeks.

# %% tags=["solution"]
# @title Stretch solution — try it yourself first { display-mode: "form" }
log_wide = np.log(wide)
weekly_gap = log_wide[TREATED].mean(axis=1) - log_wide[CONTROLS].mean(axis=1)
event = 100 * (weekly_gap - weekly_gap[pre].mean())  # approximately % above the pre-period gap
relative_week = np.arange(len(event)) - int(pre.sum())
last = slice(-30, None)
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(relative_week[last], event.to_numpy()[last], color=np.where(pre[last], "#4c72b0", "#dd8452"))
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(truth["lift_pct"], color="black", linestyle=":", label="true lift")
ax.axvline(-0.5, color="grey", linestyle="--", label="test starts")
ax.set_xlabel("Week relative to the start of the test (0 = first test week)")
ax.set_ylabel("Treated - control gap\n(% vs pre-period)")
ax.legend(loc="upper left")
fig.tight_layout()
