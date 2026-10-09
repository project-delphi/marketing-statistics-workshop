# %% [markdown]
# <!--
# APIs this lab calls, checked against the installed source of pymc-marketing 1.2.0, pymc 6.3.2,
# pymc-extras 0.15.1 and ArviZ 1.3.0 (.venv/lib/python3.13/site-packages, 2026-10-09):
# - MMM(...), GeometricAdstock(l_max=8), LogisticSaturation(), Prior("HalfNormal", sigma=, dims=)
#   and .fit(X, y, nuts_sampler="nutpie", chains=, draws=, tune=, target_accept=, random_seed=,
#   progressbar=False) as in Module 8 (mmm/mmm.py, model_builder.py).
# - MMM.add_lift_test_measurements(df_lift_test, dist=pmd.Gamma, name="lift_measurements")
#   (mmm/mmm.py): raises RuntimeError unless build_model ran first; scales x, delta_x by the
#   channel scale and delta_y, sigma by the target scale itself (scale_lift_measurements), then
#   adds Gamma(mu=|s(x + delta_x) - s(x)|, sigma=sigma, observed=|delta_y|) with s the channel's
#   saturation curve (mmm/lift_test.py add_saturation_observations). Rows need delta_x * delta_y
#   >= 0 (assert_monotonic). Only the columns x, delta_x, delta_y, sigma and the model dims
#   (channel) are used; other columns (date, test_start, ...) are ignored.
# - MMM.incrementality.contribution_over_spend(frequency="all_time", start_date=, end_date=):
#   dims (chain, draw, channel); equals the truth's ROAS definition (Module 8, measured).
# - TimeSliceCrossValidator(n_init, forecast_horizon, date_column, step_size=1,
#   sampler_config=None) (mmm/time_slice_cross_validation.py). .run(X, y, sampler_config=,
#   mmm=, original_scale_vars=, df_lift_test=, lift_test_date_column=) deep-copies the unfitted
#   MMM per fold, calls build_model on the fold's training weeks, keeps the lift rows with
#   date <= the fold's last training week and raises if none are left, then fits with
#   sampler_config passed straight to pm.sample (create_sample_kwargs). It wraps the folds in a
#   tqdm progress bar and samples each fold's posterior predictive without a random seed.
#   Fold i trains on weeks < n_init + i * step_size and tests the next forecast_horizon weeks.
# - cv.summary.predictions(hdi_probs=(0.94,)) (mmm/summary/cv.py) needs
#   original_scale_vars=["y"] (posterior_predictive y_original_scale) and reads a column named
#   exactly "date" from each fold's X (X_train["date"]); columns split, cv, date, mean, median,
#   abs_error_94_lower, abs_error_94_upper (az.hdi at prob 0.94: a 94% HDI), observed.
#   cv.summary.param_stability(var_names=) returns the same summary per fold for parameters.
#   cv.cv_metadata is a list of dicts with each fold's X_train, y_train, X_test, y_test.
# - az.rhat, az.ess(method="bulk"), idata["/sample_stats"]["diverging"] as in Module 8.
# -->
#
# # Part A · From a geo test to the likelihood
#
# Module 8 fitted a *marketing mix model* (MMM) to the synthetic retailer and recovered every
# channel's true return on ad spend (ROAS). This lab uses the same retailer with one change: an
# unobserved shift in demand, say a competitor's stockout or a viral product, raises sales in
# some weeks, and the retailer's spend on **one** channel rises in the same weeks. The demand
# itself is not in the data. You will see what that does to the model, fix it with lift tests,
# and check the model out of sample.
#
# A *lift test* (Module 7) is a randomized experiment, usually across regions, that changes one
# channel's spend and measures the sales it caused. Because the change is randomized, hidden
# demand cannot bias it.
#
# The data: three years of weekly sales `y` and spend on tv, search, social and display, a price
# index, a holiday flag and a trend `t`. We generated it, so the true ROAS of every channel is in
# `truth`.

# %%
import io
import time
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import Image, display

from mktstats.data import load_synthetic, load_truth

# Notices from inside the libraries that say nothing about this lab: a PyMC 6 deprecation,
# two about optional progress-bar widgets, and a future xarray default that PyMC-Marketing's
# cross-validation triggers when it stacks folds of different lengths.
warnings.filterwarnings("ignore", message=".*merge_dataset.*")
warnings.filterwarnings("ignore", message=".*ipywidgets.*")
warnings.filterwarnings("ignore", message=".*IProgress not found.*")
warnings.filterwarnings("ignore", message=".*default value for join will change.*")


def show(fig):
    """Show a figure as a PNG on Colab, in Jupyter and in a headless run alike."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))


# PyMC-Marketing's cross-validation summary reads a column named exactly "date", so we use that
# name from the start (Module 8 called it date_week).
weekly = load_synthetic("mmm_confounded_weekly").rename(columns={"date_week": "date"})
lift_tests = load_synthetic("mmm_confounded_lift_tests")
truth = load_truth()["mmm_confounded"]
CHANNELS = ["tv", "search", "social", "display"]
SEED = 2028  # every random step in this notebook uses this seed
true_roas = {c: truth["channels"][c]["roas"] for c in CHANNELS}
print(f"{len(weekly)} weeks, {weekly['date'].min():%Y-%m-%d} to {weekly['date'].max():%Y-%m-%d}")
weekly.head()

# %% [markdown]
# Look at the spend lines against the sales line: which channel's spend moves up and down with
# sales from week to week, beyond the holiday peaks?

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for c, style in zip(CHANNELS, ["-", "--", "-.", ":"], strict=True):
    ax1.plot(weekly["date"], weekly[c] / 1e3, style, label=c)
ax1.set_ylabel("spend ($ thousand / week)")
ax1.legend(ncol=4, loc="upper left")
ax2.plot(weekly["date"], weekly["y"] / 1e3, color="black")
ax2.set_ylabel("sales ($ thousand / week)")
fig.tight_layout()
show(fig)

# %% [markdown]
# The retailer's geo-test team ran five tests over these three years. They report each test as
# totals over the test: the weekly spend before the test (`base_weekly_spend`), the extra spend
# over all test weeks (negative when the test cut spend; the second search test switched search
# off in the test regions, scaled to the national level), the incremental sales over all test
# weeks and its standard error.

# %%
test_reports = pd.DataFrame({
    "channel": lift_tests["channel"],
    "test_start": lift_tests["test_start"],
    "test_end": lift_tests["date"],
    "weeks": lift_tests["test_weeks"],
    "base_weekly_spend": lift_tests["x"],
    "total_extra_spend": lift_tests["delta_x"] * lift_tests["test_weeks"],
    "total_incremental_sales": lift_tests["delta_y"] * lift_tests["test_weeks"],
    "total_se": lift_tests["sigma"] * lift_tests["test_weeks"],
})
test_reports

# %% [markdown]
# ## Exercise 1 · A lift-test row in the model's units (6 minutes)
#
# PyMC-Marketing adds a lift test to the model as one more piece of data. For channel $c$ with
# saturation curve $s_c$ (Module 8, in sales per week), a test that moved weekly spend from $x$ to
# $x + \Delta x$ and measured a weekly sales change $\Delta y$ with standard error $\sigma$ adds
#
# $$|\Delta y| \sim \text{Gamma}\big(\text{mean} = |s_c(x + \Delta x) - s_c(x)|,\ \text{sd} = \sigma\big)$$
#
# to the likelihood (the method's docstring and source). The curve is per week, so the row must
# be in weekly units: `channel`, `x` (weekly spend before the test), `delta_x` (change in weekly
# spend), `delta_y` (incremental sales per week) and `sigma` (its standard error per week). A
# total over $W$ weeks divided by $W$ is a weekly average, and its standard error is divided by
# $W$ too.
#
# **Predict.** A geo test ran 8 weeks, raised search spend by \$80,000 in total from a base of
# \$40,000 a week, and produced \$200,000 of incremental sales (standard error \$60,000). Write
# down $\Delta x$ and $\Delta y$ per week before you go on.
#
# **Task.** Write `lift_test_row(channel, base_weekly_spend, total_extra_spend,
# total_incremental_sales, total_se, weeks)` returning a dict with the keys `channel`, `x`,
# `delta_x`, `delta_y` and `sigma`.

# %% tags=["exercise"]
def lift_test_row(channel, base_weekly_spend, total_extra_spend, total_incremental_sales,
                  total_se, weeks):
    # TODO 1: x is the base weekly spend; the other three are totals divided by the weeks
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def lift_test_row(channel, base_weekly_spend, total_extra_spend, total_incremental_sales,
                  total_se, weeks):
    return {"channel": channel, "x": float(base_weekly_spend),
            "delta_x": total_extra_spend / weeks,
            "delta_y": total_incremental_sales / weeks,
            "sigma": total_se / weeks}


# %% [markdown]
# **Explain.** Compare with your guess: $\Delta x$ = \$10,000 and $\Delta y$ = \$25,000 a week.
# Why would a test of only one or two weeks be a poor fit for this comparison, even with the
# same totals? (Hint: Module 8's adstock.)
#
# <details><summary>Why this solution works</summary>
#
# The model compares the test with the *steady-state* response curve: the sales one week of
# spend at a constant level brings once the carryover from earlier weeks has built up. A test
# reported as totals is converted to that unit by dividing by its length; $x$ is already
# weekly. The standard error of a total divided by a constant is divided by the same constant.
# The test should run long relative to the channel's carryover (the adstock), or the first
# weeks, before the carryover has built up, make the measured lift too small for the curve.
# Pass the rows in money: `add_lift_test_measurements` scales them to the model's units itself.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    row = lift_test_row("search", 40_000, 80_000, 200_000, 60_000, 8)
    assert isinstance(row, dict) and {"channel", "x", "delta_x", "delta_y", "sigma"} <= set(row), (
        f"lift_test_row should return a dict with channel, x, delta_x, delta_y and sigma; got"
        f" {row!r}."
    )
    for key, want in {"x": 40_000, "delta_x": 10_000, "delta_y": 25_000, "sigma": 7_500}.items():
        checks.close(row[key], want, rel=1e-9,
                     name=f"{key} for the Predict example (totals over 8 weeks become weekly)")
    for r in test_reports.itertuples():
        got = lift_test_row(r.channel, r.base_weekly_spend, r.total_extra_spend,
                            r.total_incremental_sales, r.total_se, r.weeks)
        want = lift_tests.loc[r.Index]
        for key in ["x", "delta_x", "delta_y", "sigma"]:
            checks.close(got[key], want[key], rel=1e-9,
                         name=f"{key} of the {r.channel} test ending {r.test_end:%Y-%m-%d}")

# %% [markdown]
# Your function turns every report into a row in the format `add_lift_test_measurements` reads.
# `date` (the test's last week) comes along for the cross-validation in Part C.

# %%
lift_rows = pd.DataFrame([
    lift_test_row(r.channel, r.base_weekly_spend, r.total_extra_spend, r.total_incremental_sales,
                  r.total_se, r.weeks)
    for r in test_reports.itertuples()
]).assign(date=test_reports["test_end"])
search_rows = lift_rows[lift_rows["channel"] == "search"].reset_index(drop=True)
lift_rows.round(0)

# %% [markdown]
# **Run.** Fit Module 8's model, unchanged: the same transforms, controls, seasonality and
# spend-share priors, on these data, with no lift tests. `make_mmm()` returns the unfitted model,
# so the calibrated fit and the cross-validation below get exactly the same specification.
#
# `target_accept=0.95` makes the sampler take smaller steps than Module 8's 0.9: with the lift
# tests added, the workshop's test fit at 0.9 left 3 divergent draws in 4,000 (what a divergence
# is: Module 8, Exercise 4). Read the next exercise's Predict prompt while this runs.
#
# With QUICK on, the sampler draws 300 instead of 2,000 values per chain: intervals are rougher
# and may not match the module page, and a decision near its threshold can flip. That is a
# lesson about sample size, not a bug.

# %%
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

X = weekly.drop(columns=["y"])
y = weekly["y"]
totals = weekly[CHANNELS].sum().to_numpy(dtype=float)
beta_sigma = xr.DataArray(len(CHANNELS) * totals / totals.sum(), dims="channel",
                          coords={"channel": CHANNELS})  # Module 8's spend-share prior scales


def make_mmm():
    """Module 8's specification, unfitted."""
    return MMM(
        date_column="date",
        channel_columns=CHANNELS,
        control_columns=["price_index", "holiday", "t"],
        target_column="y",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
        model_config={"saturation_beta": Prior("HalfNormal", sigma=beta_sigma, dims="channel")},
    )


DRAWS = 300 if QUICK else 2000
TARGET_ACCEPT = 0.95
mmm = make_mmm()
t_fit = time.time()
idata = mmm.fit(X, y, nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS,
                target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=False)
fit_seconds = time.time() - t_fit

# %%
PARAMS = ["adstock_alpha", "saturation_lam", "saturation_beta", "gamma_control",
          "gamma_fourier", "intercept_contribution", "y_sigma"]


def fit_report(idata, var_names=PARAMS):
    """Module 8's three sampler checks: divergences, largest R-hat, smallest bulk ESS."""
    rhat = az.rhat(idata, var_names=var_names)
    ess = az.ess(idata, var_names=var_names, method="bulk")
    return (f"divergences {int(idata['/sample_stats']['diverging'].sum())},"
            f" max R-hat {max(float(rhat[v].max()) for v in rhat.data_vars):.3f},"
            f" min bulk ESS {min(float(ess[v].min()) for v in ess.data_vars):.0f}")


w0, w1 = truth["roas_window"]["start"], truth["roas_window"]["end"]
roas_before = mmm.incrementality.contribution_over_spend(frequency="all_time", start_date=w0,
                                                         end_date=w1)
print(f"Uncalibrated fit: 2 chains x {DRAWS} draws in {fit_seconds:.1f} s"
      f" ({'QUICK' if QUICK else 'FULL'} settings); {fit_report(idata)}")
print("Module 8's bars: 0 divergences, R-hat <= 1.01, bulk ESS >= 400 (QUICK usually misses"
      " the last two).")

# %% [markdown]
# # Part B · What confounding does
#
# A variable that drives both a channel's spend and sales, and is missing from the model, is a
# *confounder*: the model cannot tell "the channel raised sales" from "something else raised
# both". `roas_before` holds each channel's ROAS for every posterior draw (dims chain, draw,
# channel), as in Module 8.
#
# ## Exercise 2 · Which channel is overcredited? (7 minutes)
#
# **Predict.** One channel's spend rises when demand rises: the retailer buys more of it when
# more people are already shopping. Which one will the uncalibrated model overcredit: tv,
# search, social or display? Write it down, with a reason.
#
# **Task.** Write `bias_table(roas_draws, true_roas, prob=0.94)` returning a DataFrame indexed
# by channel with columns `mean` (posterior mean ROAS), `hdi_low`, `hdi_high` (the `prob` HDI,
# from `checks.interval(draws, prob, "hdi")`), `true_roas`, `rel_bias` ((mean − truth) ÷ truth)
# and `covered` (True if the truth lies inside the interval).

# %% tags=["exercise"]
def bias_table(roas_draws, true_roas, prob=0.94):
    # TODO 2: one row per channel: mean, hdi_low, hdi_high, true_roas, rel_bias, covered
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def bias_table(roas_draws, true_roas, prob=0.94):
    rows = []
    for c in roas_draws.coords["channel"].values:
        d = roas_draws.sel(channel=c).values.ravel()
        lo, hi = checks.interval(d, prob, "hdi")
        t = true_roas[str(c)]
        rows.append({"channel": str(c), "mean": d.mean(), "hdi_low": lo, "hdi_high": hi,
                     "true_roas": t, "rel_bias": (d.mean() - t) / t, "covered": lo <= t <= hi})
    return pd.DataFrame(rows).set_index("channel")


# %% [markdown]
# **Explain.** Compare with your pick. The model fits the sales well, yet it gives search far
# more credit than search earned. Which feature of the data lets a model that fits well still be
# wrong about one channel? Point to the plot below.
#
# <details><summary>Why this solution works</summary>
#
# The model learns a channel's effect from how sales move when its spend moves. Here search spend
# rises in the weeks when hidden demand also raises sales, so part of the demand's effect is
# assigned to search: its ROAS is overstated and the interval, narrow enough to look
# trustworthy, sits above the truth. A good in-sample fit does not reveal this, because "search
# drives sales" and "demand drives both" predict the same sales (Chan and Perry 2017 discuss this
# limit of MMMs). The other channels' spend does not follow demand, so they stay close to their
# truths. What would fix it: a measurement of demand to add as a control, or a randomized test
# of search, which the next fit uses.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    bias = bias_table(roas_before, true_roas)
    checks.columns(bias.reset_index(), ["channel", "mean", "hdi_low", "hdi_high", "true_roas",
                                        "rel_bias", "covered"], name="bias_table(...)")
    assert sorted(bias.index) == sorted(CHANNELS), (
        f"Expected one row per channel {CHANNELS}, indexed by channel; got {list(bias.index)}."
    )
    for c in CHANNELS:
        d = roas_before.sel(channel=c).values.ravel()
        lo, hi = checks.interval(d, 0.94, "hdi")
        checks.close(bias.loc[c, "mean"], d.mean(), rel=1e-9, name=f"the mean ROAS of {c}")
        tol = 0.02 * (hi - lo)
        assert abs(bias.loc[c, "hdi_low"] - lo) <= tol and abs(bias.loc[c, "hdi_high"] - hi) <= tol, (
            f"{c}: your interval is [{bias.loc[c, 'hdi_low']:.3g}, {bias.loc[c, 'hdi_high']:.3g}];"
            f" the 94% HDI of the draws is [{lo:.3g}, {hi:.3g}]. Use checks.interval(draws, prob,"
            " 'hdi'): an HDI, at 94%, not ArviZ's default 89% equal-tailed interval."
        )
        checks.close(bias.loc[c, "rel_bias"], (d.mean() - true_roas[c]) / true_roas[c], rel=1e-9,
                     name=f"rel_bias of {c}: (mean - truth) / truth")
        assert bool(bias.loc[c, "covered"]) == (lo <= true_roas[c] <= hi), (
            f"{c}: 'covered' should be True exactly when the true ROAS lies inside the interval."
        )
    s = bias.loc["search"]
    assert s["mean"] > s["true_roas"] and s["hdi_low"] > s["true_roas"], (
        f"The data are built so that the uncalibrated model overcredits search; here its mean is"
        f" {s['mean']:.2f} and 94% HDI [{s['hdi_low']:.2f}, {s['hdi_high']:.2f}] against a truth"
        f" of {s['true_roas']}. If your table matches the checks above, the fit itself differs"
        " from the workshop's: tell the instructor."
    )
    checks.k_of_K_in_interval(
        {c: true_roas[c] for c in CHANNELS if c != "search"},
        {c: roas_before.sel(channel=c).values for c in CHANNELS if c != "search"},
        k=2, prob=0.94, kind="hdi", name="ROAS of the channels whose spend does not follow demand",
    )

# %%
bias.round(3)

# %% [markdown]
# The demand shift is in the generator's answer key, which a real analysis never sees. Look at
# whether search spend is higher in the weeks when hidden demand is higher.

# %%
latent = load_synthetic("mmm_confounded_latent")  # truth only: never give it to a model
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(latent["demand_shock"], weekly["search"] / 1e3, "o", alpha=0.6)
ax.set_xlabel("hidden demand shift (standard deviations)")
ax.set_ylabel("search spend ($ thousand / week)")
show(fig)
print(f"Correlation of search spend with the hidden demand: "
      f"{np.corrcoef(latent['demand_shock'], weekly['search'])[0, 1]:.2f}")

# %% [markdown]
# **Run.** Fit the same model again, now *calibrated*: `build_model` first, then
# `add_lift_test_measurements` with the two search rows from Exercise 1 (the method raises if
# the model is not built yet), then `fit`. The other three tests stay out for now (the stretch
# adds them). About as long as the first fit.

# %%
mmm_cal = make_mmm()
mmm_cal.build_model(X, y)
mmm_cal.add_lift_test_measurements(search_rows)
t_fit = time.time()
idata_cal = mmm_cal.fit(X, y, nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS,
                        target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=False)
fit_cal_seconds = time.time() - t_fit

# %%
roas_after = mmm_cal.incrementality.contribution_over_spend(frequency="all_time", start_date=w0,
                                                            end_date=w1)
print(f"Calibrated fit: 2 chains x {DRAWS} draws in {fit_cal_seconds:.1f} s;"
      f" {fit_report(idata_cal)}")

# %% [markdown]
# ## Exercise 3 · Before and after calibration (7 minutes)
#
# **Predict.** Will calibrating on the search tests change only search's ROAS, or the other
# channels' ROAS too? If others move, which way?
#
# **Task.** Write `calibration_effect(roas_before, roas_after, true_roas, prob=0.94)` returning
# a DataFrame indexed by channel with columns `mean_before`, `mean_after`, `shift` (after minus
# before), `width_before`, `width_after` (width of the `prob` HDI), `covered_before` and
# `covered_after`. You may call your `bias_table` from Exercise 2.

# %% tags=["exercise"]
def calibration_effect(roas_before, roas_after, true_roas, prob=0.94):
    # TODO 3: summarize both posteriors per channel and put them side by side
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def calibration_effect(roas_before, roas_after, true_roas, prob=0.94):
    b = bias_table(roas_before, true_roas, prob)
    a = bias_table(roas_after, true_roas, prob)
    return pd.DataFrame({
        "mean_before": b["mean"], "mean_after": a["mean"], "shift": a["mean"] - b["mean"],
        "width_before": b["hdi_high"] - b["hdi_low"], "width_after": a["hdi_high"] - a["hdi_low"],
        "covered_before": b["covered"], "covered_after": a["covered"],
    })


# %% [markdown]
# **Explain.** Compare with your prediction. Search moved most; which other channels moved, in
# which direction, and why would sales that search no longer explains end up there? Point to the
# `shift` column and the plot below.
#
# <details><summary>Why this solution works</summary>
#
# The lift tests pin search's response curve near the spend levels they tested, so search's
# ROAS falls to near the truth and its interval narrows. The sales that the uncalibrated model
# credited to search still have to be explained: the baseline and the other channels absorb
# them, so their estimates move too, here mostly up, with wide intervals. Calibration fixes
# what was tested, not everything; a channel without a test is only as good as the data on it
# (Zhang et al. 2024 put the same idea into priors instead of a likelihood). A test on a second
# channel would pin that one too: the stretch tries it.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    effect = calibration_effect(roas_before, roas_after, true_roas)
    checks.columns(effect.reset_index(), ["channel", "mean_before", "mean_after", "shift",
                                          "width_before", "width_after", "covered_before",
                                          "covered_after"], name="calibration_effect(...)")
    assert sorted(effect.index) == sorted(CHANNELS), (
        f"Expected the four channels {CHANNELS} as the index; got {list(effect.index)}."
    )
    for c in CHANNELS:
        d0 = roas_before.sel(channel=c).values.ravel()
        d1 = roas_after.sel(channel=c).values.ravel()
        (l0, h0), (l1, h1) = checks.interval(d0, 0.94, "hdi"), checks.interval(d1, 0.94, "hdi")
        checks.close(effect.loc[c, "shift"], d1.mean() - d0.mean(), rel=1e-6, abs=1e-9,
                     name=f"the shift of {c}'s mean ROAS (after minus before)")
        checks.close(effect.loc[c, "width_before"], h0 - l0, rel=0.02,
                     name=f"the width of {c}'s 94% HDI before calibration")
        checks.close(effect.loc[c, "width_after"], h1 - l1, rel=0.02,
                     name=f"the width of {c}'s 94% HDI after calibration")
        assert bool(effect.loc[c, "covered_after"]) == (l1 <= true_roas[c] <= h1), (
            f"{c}: covered_after should say whether the truth lies inside the calibrated 94% HDI."
        )
    s = effect.loc["search"]
    assert bool(s["covered_after"]) and s["width_after"] < s["width_before"], (
        f"After calibration the truth for search should lie inside a narrower 94% HDI; here"
        f" covered_after = {bool(s['covered_after'])}, width {s['width_before']:.2f} before and"
        f" {s['width_after']:.2f} after. If your table matches the checks above, the calibrated"
        " fit differs from the workshop's: check that the fit used search_rows, then tell the"
        " instructor."
    )

# %% [markdown]
# Look at search's bar before and after, against the truth (black diamond), and at which other
# bars moved.

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
pos = np.arange(len(CHANNELS))
for offset, draws, label, color, marker in [(0.15, roas_before, "uncalibrated", "C0", "o"),
                                             (-0.15, roas_after, "calibrated", "C1", "s")]:
    tab = bias_table(draws, true_roas)
    ax.hlines(pos + offset, tab["hdi_low"], tab["hdi_high"], lw=6, alpha=0.5, color=color,
              label=f"{label}, 94% HDI")
    ax.plot(tab["mean"], pos + offset, marker, color=color, label=f"{label}, mean")
ax.plot([true_roas[c] for c in CHANNELS], pos, "D", color="black", label="truth")
ax.set_yticks(pos, CHANNELS)
ax.set_xlabel("ROAS (dollars of sales per dollar of spend, all weeks)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
show(fig)
effect.round(3)

# %% [markdown]
# # Part C · Out of sample
#
# A model can fit the weeks it was trained on and still forecast badly. *Time-slice
# cross-validation* refits the model on the first weeks only and forecasts the next ones, then
# moves the cut forward and repeats. Each repetition is a *fold*. Fold $i$ trains on all weeks
# before its cut and tests on the 8 weeks after it; a lift test is used in a fold only if it
# ended before the fold's cut (otherwise the fold would learn from the future). Both search
# tests end by week 101, so every fold here has both.
#
# **Run.** Cross-validate the calibrated model. This is the longest cell: one fit per fold. To
# keep it short each fold draws 500 values per chain instead of 2,000 (QUICK: 3 folds of 200),
# enough for predictive intervals but not for Module 8's ESS bar on every parameter. The
# library samples each fold's forecast without a fixed seed, so the scorecard can change a
# little between runs. Read Exercise 4's Predict prompt while it runs.

# %%
from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator

HORIZON = 8  # weeks forecast in each fold
STEP = 8  # weeks the cut moves between folds
N_INIT = 132 if QUICK else 124  # training weeks of the first fold: 3 or 4 folds ending at week 156
CV_DRAWS = 200 if QUICK else 500
cv = TimeSliceCrossValidator(n_init=N_INIT, forecast_horizon=HORIZON, date_column="date",
                             step_size=STEP)
t_cv = time.time()
cv_idata = cv.run(
    X, y, mmm=make_mmm(), df_lift_test=search_rows, lift_test_date_column="date",
    original_scale_vars=["y", "channel_contribution"],
    sampler_config={"draws": CV_DRAWS, "tune": CV_DRAWS, "chains": 2, "nuts_sampler": "nutpie",
                    "target_accept": TARGET_ACCEPT, "random_seed": SEED, "progressbar": False},
)
cv_seconds = time.time() - t_cv

# %%
pred = cv.summary.predictions(hdi_probs=(0.94,))
folds = [str(f) for f in cv_idata["/cv_metadata"]["cv"].values]
print(f"{len(folds)} folds x 2 chains x {CV_DRAWS} draws in {cv_seconds:.1f} s;"
      f" divergences per fold:"
      f" {cv_idata['/sample_stats']['diverging'].sum(['chain', 'draw']).values.tolist()}")
pred.head()

# %% [markdown]
# `pred` has one row per fold, week and `split` (`train` or `test`); `mean` is the posterior
# mean forecast of sales, `abs_error_94_lower` and `abs_error_94_upper` are the bounds of its
# 94% HDI (PyMC-Marketing's names), `observed` the actual sales. A week outside a split has
# empty (NaN) forecasts in that split's rows.
#
# ## Exercise 4 · Time-slice cross-validation scorecard (10 minutes)
#
# **Predict.** Will the 94% interval cover about 94% of the held-out weeks, clearly fewer, or
# clearly more?
#
# **Task.** Write `cv_scorecard(pred)` returning a DataFrame indexed by fold (the `cv` labels)
# plus a last row `overall`, with columns `weeks` (number of test weeks), `mape` (mean absolute
# percentage error of `mean` against `observed`, as a fraction) and `coverage_94` (share of test
# weeks whose `observed` lies inside the 94% HDI). Use only `test` rows with a forecast. The
# `toy_pred` below has a known answer: try `cv_scorecard(toy_pred)`.

# %%
toy_pred = pd.DataFrame({
    "split": ["train", "test", "test", "test", "test", "test"],
    "cv": ["f0", "f0", "f0", "f1", "f1", "f1"],
    "date": pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27",
                            "2025-02-03", "2025-02-10"]),
    "mean": [100.0, 100.0, 200.0, 100.0, np.nan, 300.0],
    "median": [100.0, 100.0, 200.0, 100.0, np.nan, 300.0],
    "abs_error_94_lower": [90.0, 95.0, 150.0, 50.0, np.nan, 310.0],
    "abs_error_94_upper": [110.0, 105.0, 190.0, 150.0, np.nan, 400.0],
    "observed": [100.0, 100.0, 180.0, 120.0, 999.0, 290.0],
})


# %% tags=["exercise"]
def cv_scorecard(pred):
    # TODO 4: keep test rows with a forecast; per fold and overall: weeks, mape, coverage_94
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def cv_scorecard(pred):
    test = pred[(pred["split"] == "test") & pred["mean"].notna()].copy()
    test["ape"] = (test["mean"] - test["observed"]).abs() / test["observed"]
    test["inside"] = test["observed"].between(test["abs_error_94_lower"],
                                              test["abs_error_94_upper"])
    groups = [(str(k), g) for k, g in test.groupby("cv", sort=False)] + [("overall", test)]
    return pd.DataFrame(
        [{"cv": k, "weeks": len(g), "mape": g["ape"].mean(), "coverage_94": g["inside"].mean()}
         for k, g in groups]
    ).set_index("cv")


# %% [markdown]
# **Explain.** Compare the coverage with your guess. The model was calibrated, so its search
# ROAS is close to the truth; is that what made its forecasts good or bad? Think about what an
# *uncalibrated* model could use search spend for when forecasting sales in weeks of hidden
# demand.
#
# <details><summary>Why this solution works</summary>
#
# Only `test` rows of each fold are held out; `train` rows were fitted, and rows with an empty
# forecast belong to the other split. MAPE (mean absolute percentage error) says how far the
# forecast mean is from sales; coverage says whether the 94% intervals are honest: about 94% is
# right, clearly fewer means overconfident intervals. With 24 to 32 test weeks one week is 3 to
# 4 percentage points of coverage, and neighbouring weeks are not independent, so read a gap of
# a few points as noise. Cross-validation checks *prediction*, not *cause*: search spend rises
# with hidden demand in held-out weeks too, so an uncalibrated model that credits demand to
# search can forecast sales just as well (in a test run while this lab was built, 2026-10-09, it
# did slightly better) while its search ROAS stays wrong. A good scorecard cannot replace a
# lift test; the stretch lets you check this.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    toy = cv_scorecard(toy_pred)
    checks.columns(toy.reset_index(), ["cv", "weeks", "mape", "coverage_94"],
                   name="cv_scorecard(toy_pred)")
    assert list(toy.index) == ["f0", "f1", "overall"], (
        f"The toy scorecard should have the rows f0, f1, overall (in that order); got"
        f" {list(toy.index)}. Group the test rows by `cv`, then add a row for all of them."
    )
    toy_want = {"f0": (2, (0 + 20 / 180) / 2, 1.0), "f1": (2, (20 / 120 + 10 / 290) / 2, 0.5),
                "overall": (4, (0 + 20 / 180 + 20 / 120 + 10 / 290) / 4, 0.75)}
    for k, (weeks, mape, cover) in toy_want.items():
        assert int(toy.loc[k, "weeks"]) == weeks, (
            f"toy row {k}: {weeks} test weeks with a forecast, your scorecard says"
            f" {toy.loc[k, 'weeks']}. Drop the train rows and the rows whose `mean` is NaN."
        )
        checks.close(toy.loc[k, "mape"], mape, rel=1e-9,
                     name=f"toy row {k}: mape = mean of |mean - observed| / observed")
        checks.close(toy.loc[k, "coverage_94"], cover, rel=1e-9,
                     name=f"toy row {k}: share of weeks with observed inside the 94% bounds")
    scorecard = cv_scorecard(pred)
    assert list(scorecard.index) == [*map(str, folds), "overall"], (
        f"Expected one row per fold {folds} and then 'overall'; got {list(scorecard.index)}."
    )
    assert (scorecard.loc[list(map(str, folds)), "weeks"] == HORIZON).all(), (
        f"Each fold forecasts {HORIZON} weeks; your scorecard counts"
        f" {scorecard['weeks'].tolist()}."
    )
    checks.probability(scorecard["coverage_94"].to_numpy(), name="coverage_94")
    assert (scorecard["mape"] >= 0).all(), "MAPE cannot be negative: use absolute errors."

# %%
scorecard.round(3)

# %% [markdown]
# Second, *parameter stability*: does a fold that sees fewer weeks tell the same story about
# each channel? `cv.summary.param_stability(var_names=[...])` summarizes any model parameter per
# fold; here the channel's effect ceiling `saturation_beta` (in the model's scaled units; the
# scaling is the same in every fold here).

# %%
stability = cv.summary.param_stability(var_names=["saturation_beta"], hdi_probs=(0.94,))
stability.pivot(index="channel", columns="cv", values="mean").loc[CHANNELS].round(3)

# %% [markdown]
# Module 8 showed that $\lambda$ and $\beta$ trade off along a ridge, so $\beta$ alone can move
# between folds while the ROAS, which depends on both, does not. The decision rests on ROAS, so
# the provided `fold_roas` computes each fold's ROAS over its own training weeks. Look at which
# channels' dots move from fold to fold, and compare with the `saturation_beta` table above.
# Each panel has its own scale; the printed shares below put the channels on one scale.

# %%
def fold_roas(cv, cv_idata, prob=0.94):
    """Per fold and channel: ROAS over the fold's training weeks (mean and HDI)."""
    contrib = cv_idata["/posterior"]["channel_contribution_original_scale"]
    rows = []
    for label, meta in zip(contrib.coords["cv"].values, cv.cv_metadata, strict=True):
        fold = contrib.sel(cv=label).sum("date")  # weeks outside the fold's training are NaN
        for c in CHANNELS:
            d = (fold.sel(channel=c) / meta["X_train"][c].sum()).values.ravel()
            lo, hi = checks.interval(d, prob, "hdi")
            rows.append({"cv": str(label), "channel": c, "mean": d.mean(), "hdi_low": lo,
                         "hdi_high": hi})
    return pd.DataFrame(rows)


roas_folds = fold_roas(cv, cv_idata)
fig, axes = plt.subplots(1, len(CHANNELS), figsize=(11, 3), sharey=False)
for ax, c in zip(axes, CHANNELS, strict=True):
    f = roas_folds[roas_folds["channel"] == c].reset_index(drop=True)
    ax.vlines(f.index, f["hdi_low"], f["hdi_high"], lw=5, alpha=0.4, label="94% HDI")
    ax.plot(f.index, f["mean"], "o", label="mean")
    ax.axhline(true_roas[c], color="black", ls="--", lw=1, label="truth")
    ax.set_title(c)
    ax.set_xticks(f.index, [str(i + 1) for i in f.index])
    ax.set_xlabel("fold")
axes[0].set_ylabel("ROAS over the fold's training weeks")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
show(fig)

STABILITY_TOL = 0.10  # a ROAS that moves by more than a tenth between folds rests on a few weeks
spread = roas_folds.groupby("channel")["mean"].agg(lambda m: m.max() - m.min())
moves = (spread / bias_table(roas_after, true_roas)["mean"]).loc[CHANNELS]
print("Range of the fold means as a share of the calibrated ROAS:",
      {c: f"{moves[c]:.0%}" for c in CHANNELS})

# %% [markdown]
# ## Decision · Move budget now, or test first?
#
# **Number.** For each channel: the calibrated ROAS (mean and 94% HDI, Exercise 3), the
# posterior probability that it exceeds break-even, computed draw by draw, whether its ROAS
# moves between folds (Part C), and the spend at stake (the last 52 weeks' spend).
#
# **Rule.** As in Module 8: a dollar of sales earns the retailer its gross margin, 30%, so the
# break-even ROAS is 1 ÷ 0.30 ≈ 3.3. A channel *pays back* if P(ROAS > 3.3) ≥ 0.9 and *does not
# pay back* if P(ROAS < 3.3) ≥ 0.9; otherwise it is *undecided*. Move budget on a channel now
# only if it is decided **and** its ROAS moves by at most 10% between folds; otherwise test it
# first, and give the next lift test to the channel to test with the most spend at stake. The
# uncalibrated verdict is shown beside it: what you would have done without the lift tests.

# %%
GROSS_MARGIN = 0.30  # stated by the business, as in Module 8
BREAK_EVEN = 1 / GROSS_MARGIN
spend_52 = weekly[CHANNELS].tail(52).sum()


def verdict(draws):
    p = float((draws > BREAK_EVEN).mean())
    return p, ("pays back" if p >= 0.9 else "does not pay back" if p <= 0.1 else "undecided")


after = bias_table(roas_after, true_roas)
rows = []
for c in CHANNELS:
    p_after, v_after = verdict(roas_after.sel(channel=c).values.ravel())
    _, v_before = verdict(roas_before.sel(channel=c).values.ravel())
    stable = moves[c] <= STABILITY_TOL
    rows.append({"channel": c, "roas_mean": after.loc[c, "mean"],
                 "hdi_low": after.loc[c, "hdi_low"], "hdi_high": after.loc[c, "hdi_high"],
                 "P(ROAS > break-even)": p_after, "calibrated verdict": v_after,
                 "uncalibrated verdict": v_before, "moves between folds": not stable,
                 "spend at stake ($, 52 weeks)": spend_52[c],
                 "act now?": "act" if v_after != "undecided" and stable else "test first"})
decision = pd.DataFrame(rows).set_index("channel")
to_test = decision[decision["act now?"] == "test first"]
next_test = to_test["spend at stake ($, 52 weeks)"].idxmax() if len(to_test) else None
mode = "QUICK run: treat this as a rough answer" if QUICK else "FULL run"
print(f"Break-even ROAS {BREAK_EVEN:.2f} (gross margin {GROSS_MARGIN:.0%}); {mode}.")
print(f"Next lift test: {next_test or 'none needed by this rule'}")
decision.round(3)

# %% [markdown]
# **Recommendation.** Write one sentence a manager could act on: which channels' ROAS you would
# move budget on now and in which direction, which channel gets the next lift test and how much
# spend that decision concerns, how sure you are, and what would change it.
#
# ```text
# Your sentence: ________________________________________________
# ```
#
# Then compare with the truth, which a real analysis never sees:

# %%
decision.assign(true_roas=[true_roas[c] for c in CHANNELS],
                truly_pays_back=[true_roas[c] > BREAK_EVEN for c in CHANNELS])[
    ["roas_mean", "calibrated verdict", "uncalibrated verdict", "true_roas",
     "truly_pays_back"]].round(3)

# %% [markdown]
# ## Stretch (optional) · A second lift test, and cross-validation without calibration
#
# 1. The marketing team ran the test the decision asked for: `lift_rows` also holds tests on
#    tv, social and display. Fit the calibrated model with all five rows and repeat Exercise 3
#    against the search-only fit. Which intervals narrow, and does any verdict change? Read
#    `fit_report` first: more lift tests constrain the posterior more, and the sampler may find
#    it harder to explore.
# 2. Run the cross-validation without `df_lift_test` (the uncalibrated model) and compare its
#    scorecard and its fold-by-fold search ROAS with the calibrated one. Which model forecasts
#    better, and which one is right about search?
#
# Each costs about as long as its counterpart above.
#
# <details><summary>Code for both</summary>
#
# ```python
# mmm_all = make_mmm()
# mmm_all.build_model(X, y)
# mmm_all.add_lift_test_measurements(lift_rows)
# mmm_all.fit(X, y, nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS,
#             target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=False)
# print(fit_report(mmm_all.idata))
# roas_all = mmm_all.incrementality.contribution_over_spend(frequency="all_time",
#                                                           start_date=w0, end_date=w1)
# calibration_effect(roas_after, roas_all, true_roas).round(3)
#
# cv_plain = TimeSliceCrossValidator(n_init=N_INIT, forecast_horizon=HORIZON,
#                                    date_column="date", step_size=STEP)
# cv_plain_idata = cv_plain.run(
#     X, y, mmm=make_mmm(), original_scale_vars=["y", "channel_contribution"],
#     sampler_config={"draws": CV_DRAWS, "tune": CV_DRAWS, "chains": 2, "nuts_sampler": "nutpie",
#                     "target_accept": TARGET_ACCEPT, "random_seed": SEED, "progressbar": False})
# pd.concat({"calibrated": scorecard,
#            "uncalibrated": cv_scorecard(cv_plain.summary.predictions(hdi_probs=(0.94,)))},
#           axis=1).round(3)
# fold_roas(cv_plain, cv_plain_idata).query("channel == 'search'").round(2)
# ```
# </details>
