# %% [markdown]
# <!--
# Module 17 lab; brief: briefs/17-mmm-time-varying.md. Data: src/mktstats/synth/mmm_drift.py
# (seed 2037), committed as data/synthetic/mmm_drift_{weekly,latent,lift_tests}.csv and
# truth_mmm_drift.json. Checks and the reference pipeline: src/mktstats/drift.py.
# APIs this lab calls, checked 2026-10-10 against the installed source in
# .venv/lib/python3.13/site-packages (pymc-marketing 1.2.0, pymc 6.3.2, pymc-extras 0.15.1,
# nutpie 0.16.11, prophet 1.4.0, cmdstanpy 1.3.0, xarray 2025.12.0):
# - MMM(..., time_varying_media: bool | HSGPBase) (mmm/mmm.py): an HSGPBase instance is
#   registered on the time index 0..T-1 and creates media_temporal_latent_multiplier with the
#   instance's own dims; channel_contribution = baseline_channel_contribution x multiplier.
#   time_varying_media=True would build one multiplier with dims ("date",), shared by all
#   channels (create_hsgp_from_config(dims=("date", *self.dims)), default media_tvp_config
#   HSGPKwargs(m=200, ls_mu=5, ...), Matern 5/2 in mmm/tvp.py).
# - SoftPlusHSGP.parameterize_from_data(X, dims, X_mid=None, eta_mass=0.05, eta_upper=1.0,
#   ls_lower=1.0, ls_upper=None, ls_mass=0.9, cov_func=CovFunc.ExpQuad, centered=False,
#   drop_first=True, ...) (mmm/hsgp.py): eta Exponential with P(eta > eta_upper) = eta_mass;
#   with ls_upper, ls InverseGamma with mass 0.9 in [ls_lower, ls_upper]; m and L from
#   approx_hsgp_hyperparams (Riutort-Mayol et al.). Its default kernel is ExpQuad; this lab sets
#   cov_func="matern52". With X = 0..155, ls 20 to 104 and Matern 5/2: m = 56 basis functions,
#   L = 426.4 (printed by the pipeline in mktstats.drift). SoftPlusHSGP.create_variable divides
#   softplus(f) by its mean over "date", so the multiplier has mean 1 over the fitted weeks.
# - MMM.sample_prior_predictive(X, y, samples=, random_seed=): prior draws in idata["prior"];
#   .fit(X, y, nuts_sampler="nutpie", chains=2, cores=2, draws=, tune=, target_accept=,
#   random_seed=, progressbar=False); idata.constant_data["target_scale"]; posterior "mu".
# - MMM.incrementality.contribution_over_spend(frequency="all_time", start_date=, end_date=)
#   (mmm/incrementality.py): spend zeroed in the window, carry-in kept, window clamped to the
#   fitted dates. Over all weeks it equals the window definition to 4e-14 (measured).
# - MMM.add_lift_test_measurements(df) after build_model: with time-varying media the rows need
#   a `date` column equal to a model date (lift_test.py exact_row_indices); the saturation
#   curve is multiplied by the multiplier at that date (create_time_varying_saturation).
# - prophet.Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
#   holidays=, uncertainty_samples=0) (forecaster.py); .add_regressor(name); .fit(df);
#   .predict(df) has trend, yearly, holidays and yhat; yhat = trend * (1 + multiplicative
#   terms) + additive terms. Prophet logs through cmdstanpy at INFO level.
# -->
#
# **How this lab runs.** Three MMM fits, each once, all provided: a static model (Part B), a
# model whose channel effects may drift (Part C, the longest cell) and a model with a Prophet
# baseline (Part D). Each fit cell says how long it takes and what to do while it runs. `QUICK`
# (in the harness cell) draws 300 instead of 1,000 values per chain: intervals are wider and may
# not match the module page, and a decision near its threshold can flip. That is a lesson about
# sample size, not a bug.
#
# The data are simulated: three years of a retailer's weekly sales and spend on four channels.
# Because the simulation knows each channel's true effect in every week, you can check every
# model against the truth, which real data never allow.

# %%
import io
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import Image, display

from mktstats import drift
from mktstats.data import repo_file

# Notices from inside the libraries that say nothing about this lab: a PyMC 6 deprecation and
# two about optional progress-bar widgets.
warnings.filterwarnings("ignore", message=".*merge_dataset.*")
warnings.filterwarnings("ignore", message=".*ipywidgets.*")
warnings.filterwarnings("ignore", message=".*IProgress not found.*")


def show(fig):
    """Show a figure as a PNG on Colab, in Jupyter and in a headless run alike."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))


SEED = 2037  # every random step in this notebook uses this seed
DRAWS = 300 if QUICK else 1000  # draws (and tuning steps) per chain in every fit
CHANNELS = ["tv", "search", "social", "display"]
STYLES = {"tv": "-", "search": "--", "social": "-.", "display": ":"}
GROSS_MARGIN = 0.30  # stated by the business (Modules 8 and 9)
BREAK_EVEN = 1 / GROSS_MARGIN

weekly = pd.read_csv(repo_file("synthetic/mmm_drift_weekly.csv"), parse_dates=["date_week"])
latent = pd.read_csv(repo_file("synthetic/mmm_drift_latent.csv"), parse_dates=["date_week"])
truth = json.loads(repo_file("synthetic/truth_mmm_drift.json").read_text())
WINDOWS = {k: (w["start"], w["end"]) for k, w in truth["windows"].items()}
X = weekly.drop(columns=["y"])
y = weekly["y"]
# Spend and the true contributions (dollars) as arrays with named dims (date, channel)
spend = xr.DataArray(weekly[CHANNELS].to_numpy(dtype=float), dims=("date", "channel"),
                     coords={"date": weekly["date_week"].to_numpy(), "channel": CHANNELS})
true_contrib = xr.DataArray(
    latent[[f"contribution_{c}" for c in CHANNELS]].to_numpy(dtype=float),
    dims=("date", "channel"), coords={"date": latent["date_week"].to_numpy(), "channel": CHANNELS})

print(f"{len(weekly)} weeks, {weekly['date_week'].min():%Y-%m-%d} to"
      f" {weekly['date_week'].max():%Y-%m-%d}; {DRAWS} draws per chain"
      f" ({'QUICK' if QUICK else 'FULL'} settings)")
print("Windows:", {k: f"{a} to {b}" for k, (a, b) in WINDOWS.items()})
weekly.head()

# %% [markdown]
# # Part A · ROAS depends on the window
#
# **Ad fatigue** (wear-out): a channel's effect per dollar falls over time, because the same
# creative is shown too often, an audience is used up, or a platform changes. The
# **effectiveness multiplier** of a channel in a week says how strongly it works in that week
# compared with its own average over the three years: 1 is average, 0.5 is half as well. Its
# path over the weeks is the channel's **relative path**.
#
# Look at the spend lines first: tv in bursts with weeks off air, search rising and falling with
# demand, social with a four-week push every quarter, display small and noisy.

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for c in CHANNELS:
    ax1.plot(weekly["date_week"], weekly[c] / 1e3, STYLES[c], label=c)
ax1.set_ylabel("spend ($ thousand / week)")
ax1.legend(ncol=4, loc="upper left")
ax2.plot(weekly["date_week"], weekly["y"] / 1e3, color="black")
ax2.set_ylabel("sales ($ thousand / week)")
fig.tight_layout()
show(fig)

# %% [markdown]
# The next plot shows each channel's true relative path, from the simulation's hidden truth
# (`latent`, never given to a model). Look at social's line: its effect per dollar holds for a
# year, then falls to under half.

# %%
fig, ax = plt.subplots(figsize=(10, 3.2))
constant = [c for c in CHANNELS if np.allclose(latent[f"multiplier_rel_{c}"], 1.0)]
ax.plot(latent["date_week"], latent[f"multiplier_rel_{constant[0]}"], "-", color="grey", lw=3,
        label=f"{', '.join(constant)} (1 in every week)")
for c in CHANNELS:
    if c not in constant:
        ax.plot(latent["date_week"], latent[f"multiplier_rel_{c}"], STYLES[c], lw=2.5, label=c)
ax.set_ylabel("true relative path\n(1 = the channel's average)")
ax.legend(loc="lower left")
show(fig)

# %% [markdown]
# Two ideas from Module 8, and one new one:
#
# - **Adstock carryover**: part of a week's effect lands in later weeks.
# - **ROAS over a window**: the sales a channel produced in those weeks per dollar it spent in
#   those weeks; carryover from earlier spend counts, carryover after the window does not. For
#   channel $c$ and a set of weeks $W$,
#   $$\text{ROAS}_c(W) = \frac{\sum_{t \in W} \text{contrib}_{c,t}}{\sum_{t \in W} x_{c,t}},$$
#   with $\text{contrib}_{c,t}$ the sales channel $c$ produced in week $t$ and $x_{c,t}$ its
#   spend. Over all weeks it is Module 8's ROAS.
# - **Break-even ROAS**: the ROAS at which the gross margin on the sales a channel produced pays
#   for its spend: 1 ÷ 0.30 ≈ 3.33 at the retailer's 30% margin.
#
# `WINDOWS` names three windows: `"all"` (all 156 weeks), `"first_year"` (the first 52 weeks) and
# `"last_quarter"` (the last 13 weeks), each a pair of date strings `(start, end)`.

# %% [markdown]
# ## Exercise 1 · ROAS over a window (7 minutes)
#
# **Predict.** Social's true ROAS over all three years is 4.0, above the break-even ROAS of 3.3.
# Will its ROAS in the last quarter be above break-even, between 2.5 and 3.3, or below 2.5? Write
# your guess in the cell below.

# %% [markdown]
# *Your guess* (double-click to edit):

# %% [markdown]
# **Task.** Write `window_roas(contrib, spend, start, end)` returning ROAS over the weeks from
# `start` to `end`, both included. `contrib` is a DataArray of contributions in dollars with dims
# `(..., date, channel)`: any leading dims, such as `chain` and `draw` of posterior draws.
# `spend` has dims `(date, channel)`. Return a DataArray with dims `(..., channel)`. You will use
# the same function on the truth now and on posterior draws later.

# %% tags=["exercise"]
def window_roas(contrib, spend, start, end):
    # TODO 1: select the window's weeks in both arrays with .sel(date=slice(start, end)) (both
    # ends included); sum each over "date" by name (.sum("date")); divide. xarray matches the
    # remaining dims by name, so leading dims such as chain and draw survive.
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def window_roas(contrib, spend, start, end):
    sales = contrib.sel(date=slice(start, end)).sum("date")
    cost = spend.sel(date=slice(start, end)).sum("date")
    return sales / cost


# %% [markdown]
# The checkpoint runs three toys, then the truth. (1) One channel over four weeks from
# 2023-01-02, spend 4, 3, 2, 2 and contribution 1, 1, 1, 5, over the window 2023-01-09 to
# 2023-01-23, gives (1 + 1 + 5) ÷ (3 + 2 + 2) = 1.0. (2) A contribution of exactly twice the spend
# gives 2 for every channel. (3) Draws with dims `(chain, draw, date, channel)` of sizes
# (2, 3, 156, 4) give shape (2, 3, 4). (4) On `true_contrib`, the three windows equal the truth
# file's ROAS to a relative 1e-9. It prints the true ROAS by window.

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    roas_true = drift.check_window_roas(window_roas, true_contrib, spend, WINDOWS, truth)
    print("True ROAS by window (dollars of sales per dollar of spend):")
    print(pd.DataFrame({k: v.to_series() for k, v in roas_true.items()}).round(2).to_string())

# %% [markdown]
# **Explain.** Compare with your guess. Which weeks dominate an all-period average: the many good
# weeks or the few recent ones? Point to the plot of the true relative paths.
#
# <details><summary>Why this solution works</summary>
#
# ROAS over a window is a ratio of sums, so the all-period ROAS weights each week by its spend:
# two good years outweigh one bad quarter, and the average stays above break-even long after the
# channel stopped paying back (social: 4.00 over all weeks, 5.00 in the first year, 2.11 in the
# last quarter). A decision about next quarter needs the recent window. A mean of weekly ratios
# would weight a week of small spend as much as a push week, and is undefined in tv's weeks off
# air.
# </details>

# %% [markdown]
# # Part B · What the static model says about now
#
# **Run: the static fit.** A **static MMM** is Module 8's model: one effect size per channel, the
# same in every week. Same specification as Module 8 (geometric adstock with up to 8 weeks of
# carryover, logistic saturation, spend-share priors on the effect sizes; controls price index,
# holiday flag and trend; two orders of yearly seasonality), sampled with nutpie, two chains.
# `target_accept=0.98` makes the sampler take small steps: at 0.95 a test fit of this model had a
# divergence.
#
# It takes about a minute and a half on Colab (an estimate until a Colab record exists). While it
# runs, read Exercise 2's Predict, write your two signs, and start writing `residual_spend_corr`:
# its toy check needs no fit.
#
# With QUICK on, intervals are wider and may not match the module page; a decision near its
# threshold can flip. That is a lesson about sample size, not a bug.

# %%
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation, SoftPlusHSGP

totals = weekly[CHANNELS].sum().to_numpy(dtype=float)
beta_sigma = xr.DataArray(len(CHANNELS) * totals / totals.sum(), dims="channel",
                          coords={"channel": CHANNELS})  # Module 8's spend-share prior scales
FIT = {"nuts_sampler": "nutpie", "chains": 2, "cores": 2, "draws": DRAWS, "tune": DRAWS,
       "target_accept": 0.98, "random_seed": SEED}  # every fit below adds progressbar=False


def media_priors():
    return {"saturation_beta": Prior("HalfNormal", sigma=beta_sigma, dims="channel")}


mmm_static = MMM(
    date_column="date_week", channel_columns=CHANNELS, target_column="y",
    control_columns=["price_index", "holiday", "t"],
    adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
    yearly_seasonality=2, model_config=media_priors(),
)
t_fit = time.time()
idata_static = mmm_static.fit(X, y, progressbar=False, **FIT)
static_seconds = time.time() - t_fit

# %% [markdown]
# **94% HDI** (recall, Module 8): the narrowest interval that holds 94% of the posterior draws.
# The next cell turns the posterior's weekly channel contributions into dollars and applies your
# `window_roas` to every draw. The labeled check compares it, over all weeks, with
# PyMC-Marketing's own ROAS (`contribution_over_spend`), and checks social's last quarter against
# the truth.

# %%
def target_scale(idata):
    """PyMC-Marketing's scale for sales: model units x target_scale = dollars."""
    return float(np.asarray(idata.constant_data["target_scale"]).squeeze())


def roas_table(draws_by_window, windows, prob=0.94):
    """Mean and HDI of ROAS draws per channel and window, next to the truth."""
    key = {"all": "roas", "first_year": "roas_first_year", "last_quarter": "roas_last_quarter"}
    rows = []
    for c in CHANNELS:
        for k in windows:
            d = draws_by_window[k].sel(channel=c).values.ravel()
            lo, hi = checks.interval(d, prob, "hdi")
            rows.append({"channel": c, "window": k, "mean": d.mean(), "hdi_low": lo,
                         "hdi_high": hi, "truth": truth["channels"][c][key[k]]})
    return pd.DataFrame(rows).set_index(["channel", "window"])


contrib_static = idata_static.posterior["channel_contribution"] * target_scale(idata_static)
roas_cos = mmm_static.incrementality.contribution_over_spend(frequency="all_time")
print(f"Static fit: {static_seconds:.0f} s, {drift.divergences(idata_static)} divergences.")
with workshop.checkpoint(label="static model"):
    roas_draws_static = drift.check_static_model(window_roas, contrib_static, spend, roas_cos,
                                                 WINDOWS, truth)

# %% [markdown]
# Look at social's two rows: the static model gives about the same number for both windows.

# %%
roas_table(roas_draws_static, ["all", "last_quarter"]).round(2)

# %% [markdown]
# <details><summary>Two definitions of ROAS over a window</summary>
#
# PyMC-Marketing's `contribution_over_spend(start_date=, end_date=)` uses a counterfactual: it sets
# the channel's spend to zero in the window, keeps the carryover from earlier weeks, and adds up
# the sales lost from the window's first week on. Over all weeks the two definitions agree (the
# check above confirmed it to 1e-8); for one quarter they differ slightly (social's true
# last-quarter ROAS is 2.11 by the window definition and 2.04 by the counterfactual one). This
# lab uses the window definition throughout, which one function computes for the truth and for
# posterior draws alike.
# </details>

# %% [markdown]
# ## Exercise 2 · Residuals that drift (8 minutes)
#
# On real data there is no truth. This exercise finds the drift from the static model alone. A
# **residual** is actual sales minus the model's fitted sales (posterior mean), in dollars. The
# next cell computes `resid` (one per week) and `year`.

# %%
mu_mean = idata_static.posterior["mu"].mean(("chain", "draw")).values * target_scale(idata_static)
resid = pd.Series(y.to_numpy(dtype=float) - mu_mean, index=weekly.index, name="residual")
year = weekly["date_week"].dt.year
print(f"Residuals: mean ${resid.mean():,.0f}, sd ${resid.std():,.0f} a week;"
      f" weeks per year {year.value_counts().sort_index().to_dict()}")

# %% [markdown]
# **Predict.** The static model gives social one effect size for all three years. In 2023, when
# social worked better than its three-year average, will the residuals be positively or
# negatively correlated with social's weekly spend? And in 2025? Write two signs in the cell
# below.

# %% [markdown]
# *Your guess* (double-click to edit):

# %% [markdown]
# **Task.** Write `residual_spend_corr(resid, spend, year)` returning a DataFrame with one row
# per year (index: the year) and one column per channel, each the Pearson correlation between
# the residuals and that channel's weekly spend within the year. `resid` (Series), `spend`
# (DataFrame, one column per channel) and `year` (Series) share `weekly`'s index.

# %% tags=["exercise"]
def residual_spend_corr(resid, spend, year):
    # TODO 2: for each year yr in sorted(year.unique()), select that year's rows with
    # year == yr; spend_part.corrwith(resid_part) gives one Pearson correlation per column
    # (pandas 2.2.3: method="pearson" is the default); one row per year, one column per channel
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def residual_spend_corr(resid, spend, year):
    rows = {}
    for yr in sorted(year.unique()):
        rows[yr] = spend[year == yr].corrwith(resid[year == yr])
    return pd.DataFrame(rows).T.rename_axis("year")


# %% [markdown]
# The checkpoint uses a toy that `np.corrcoef` answers exactly: `year` 2023 four times and 2024
# four times; spend a = 1, 2, 3, 4, 1, 2, 3, 4 and b = 4, 1, 3, 2, 2, 2, 1, 5; `resid` = 1, 2, 3,
# 10, 4, 3, 2, 1 (2023: a 0.8854, b −0.3162; 2024: a −1.0, b −0.5963). Then it compares your
# table on the static fit with the reference, and checks that social's correlation falls most
# from 2023 to 2025 and ends below −0.3.

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    corr_table = drift.check_residual_spend_corr(residual_spend_corr, resid, weekly[CHANNELS],
                                                 year)
    print("Correlation of the static model's residuals with weekly spend, by year:")
    print(corr_table.round(2).to_string())

# %% [markdown]
# **Explain.** Compare with your two signs. Why does a positive correlation in 2023 mean the model
# gives social too little credit then, and a negative one in 2025 too much? Point to social's
# column in the yearly table, and to one other channel's column for contrast.
#
# <details><summary>Why this solution works</summary>
#
# With one effect size for all weeks, the model fits the average. In 2023 sales jump more than
# the model predicts when social spends (positive residuals in push weeks: +0.40); in 2025 they
# jump less (negative: −0.57). The other channels' correlations stay small in every year. A sign
# change in a channel's residual correlation is a cheap warning to run before believing an
# all-period ROAS. What would change it: spend that never varies within a year leaves nothing to
# correlate.
# </details>

# %% [markdown]
# # Part C · Letting the effect drift
#
# The **drifting MMM** is the same model with each channel's effect multiplied by its own
# multiplier $m_c(t)$ in every week: the quantity whose true value Part A plotted. Its prior is a
# **Gaussian process prior**: a prior over smooth curves, where each draw is a whole curve (the
# plot after the next cell shows 20 of them). Its **lengthscale** is roughly how many weeks a
# curve takes to change appreciably; it is set here so that a curve changes over months, not
# weeks. Its **amplitude** is how far a curve can move away from its average. The model's
# multiplier has **mean 1** over the three years, like the true relative path, so the two can be
# compared directly.
#
# **Run.** Build the drifting model and draw 1,000 times from its prior (seconds).

# %%
def drifting_mmm():
    """Fit 1's specification plus one effectiveness multiplier per channel and week."""
    multiplier = SoftPlusHSGP.parameterize_from_data(
        # a fast approximation of a Gaussian process, kept positive and scaled to average 1
        # over the fitted weeks
        X=np.arange(len(weekly)),  # the weeks 0..155, the model's time index
        dims=("date", "channel"),  # one curve per channel (time_varying_media=True would give
        # one curve shared by all channels)
        ls_lower=20, ls_upper=104,  # the curve changes over months, not weeks: 90% prior
        # probability that the lengthscale is 20 to 104 weeks
        eta_upper=2.0,  # how far a curve can move: a 5% prior probability that the amplitude
        # exceeds 2
        cov_func="matern52",  # how smooth the curve is; the choice PyMC-Marketing makes for its
        # own time_varying_media=True (this function's default is "expquad")
    )
    return MMM(
        date_column="date_week", channel_columns=CHANNELS, target_column="y",
        control_columns=["price_index", "holiday", "t"],
        adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
        yearly_seasonality=2, time_varying_media=multiplier, model_config=media_priors(),
    )


mmm_drift = drifting_mmm()
mmm_drift.sample_prior_predictive(X, y, samples=1000, random_seed=SEED)
prior_mult = mmm_drift.idata["prior"]["media_temporal_latent_multiplier"]  # (chain, draw, date, channel)
print(dict(prior_mult.sizes))

# %% [markdown]
# Look at how far the prior's curves wander from 1, and how fast they change.

# %%
rng_plot = np.random.default_rng(SEED)
pick = rng_plot.choice(prior_mult.sizes["draw"], size=20, replace=False)
fig, ax = plt.subplots(figsize=(10, 3.6))
for i, d in enumerate(pick):
    ax.plot(weekly["date_week"], prior_mult.sel(channel="social").isel(chain=0, draw=d),
            color="tab:blue", alpha=0.45, lw=1, label="prior draws (20)" if i == 0 else None)
ax.plot(latent["date_week"], latent["multiplier_rel_social"], "k--", lw=2.5,
        label="social's true relative path")
ax.axhline(1, color="grey", lw=0.8)
ax.set_ylabel("social's multiplier")
ax.set_yscale("log")
ax.set_title("20 draws from the multiplier's prior (log scale: 0.5 and 2 are equally far from 1)")
ax.legend(loc="upper left")
show(fig)

# %% [markdown]
# ## Exercise 3 · How much drift does the prior allow? (6 minutes)
#
# The **drift ratio** of a multiplier is its average in the last quarter divided by its average
# in the first year: the share of its first-year effect a channel keeps. The prior draws
# (`prior_mult`; Module 8's prior predictive check) are draws from the model before it has seen
# any sales.
#
# **Predict.** Over all 1,000 prior draws, in what share will social keep half or less of its
# first-year effect in the last quarter (a drift ratio of 0.5 or less): under 3%, 3 to 20%, or
# over 20%? Use the plot of 20 curves to estimate it. Write your guess in the cell below.

# %% [markdown]
# *Your guess* (double-click to edit):

# %% [markdown]
# **Task.** Write `drift_ratio(mult, early, late)` returning the mean of `mult` over the `late`
# weeks divided by its mean over the `early` weeks. `early` and `late` are `(start, end)` pairs
# of date strings, as in `WINDOWS`; `mult` has dims `(..., date, channel)`; return dims
# `(..., channel)`.

# %% tags=["exercise"]
def drift_ratio(mult, early, late):
    # TODO 3: the same window selection as Exercise 1, .sel(date=slice(*late)), with
    # .mean("date"); divide the late mean by the early mean
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def drift_ratio(mult, early, late):
    return mult.sel(date=slice(*late)).mean("date") / mult.sel(date=slice(*early)).mean("date")


# %% [markdown]
# The checkpoint uses (1) a multiplier of 1 in every week, which gives 1; (2) the linear path
# $m(t) = 2 - t/155$ over the 156 weeks, with `WINDOWS["first_year"]` (52 weeks) and
# `WINDOWS["last_quarter"]` (13 weeks), which gives 0.56591; (3) the prior draws: the result has
# dims `(chain, draw, channel)` and is positive and finite.

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    prior_ratio = drift.check_drift_ratio(drift_ratio, weekly["date_week"], WINDOWS, prior_mult)
    print(dict(prior_ratio.sizes))

# %% [markdown]
# The share of prior draws in which social keeps half or less of its first-year effect, the share
# at or below social's true drift ratio, and the prior's 94% equal-tailed interval (3rd to 97th
# percentile):

# %%
social_prior = prior_ratio.sel(channel="social").values.ravel()
true_ratio = truth["channels"]["social"]["drift_ratio"]
lo, hi = checks.interval(social_prior, 0.94, "eti")
print(f"P(drift ratio <= 0.5) = {np.mean(social_prior <= 0.5):.1%}")
print(f"Social's true drift ratio: {true_ratio:.2f}; P(drift ratio <= {true_ratio:.2f}) ="
      f" {np.mean(social_prior <= true_ratio):.1%}")
print(f"Prior 94% equal-tailed interval of the drift ratio: {lo:.2f} to {hi:.2f}")

# %% [markdown]
# **Explain.** Compare with your guess. Social's true drift ratio is printed above. Does this
# prior rule out a fall that large, allow it but treat it as unusual, or expect it? Point to the
# printed share.
#
# <details><summary>Why this solution works</summary>
#
# It allows the fall but treats it as unusual (7.4% of prior draws at or below the true 0.42), so
# the data have to show it. A prior that almost never allows a large fall cannot find one; a prior
# that allows anything follows noise. The lengthscale prior keeps the path smooth (changes over
# months, not weeks); the amplitude prior keeps it near 1 unless the data insist. This is Module
# 8's prior predictive check applied to the drift: it takes seconds and catches a prior that
# rules the answer out before the longest fit of the lab. What would change it: a smaller
# amplitude bound (`eta_upper=1.0` instead of 2.0) puts less prior mass on large falls.
# </details>

# %% [markdown]
# **Run: the drifting fit**, the longest cell of the lab: about three minutes on Colab (an
# estimate until a Colab record exists). While it runs, read Exercise 4's Predict, write your
# guess, and write `path_coverage`: its toy checks need no fit. Run its cells when the fit has
# finished.

# %%
t_fit = time.time()
idata_drift = mmm_drift.fit(X, y, progressbar=False, **FIT)
drift_seconds = time.time() - t_fit

# %%
mult = idata_drift.posterior["media_temporal_latent_multiplier"]  # (chain, draw, date, channel)
contrib_drift = idata_drift.posterior["channel_contribution"] * target_scale(idata_drift)
print(f"Drifting fit: {drift_seconds:.0f} s, {drift.divergences(idata_drift)} divergences;"
      f" multiplier draws {dict(mult.sizes)}")

# %% [markdown]
# ## Exercise 4 · Does the band cover the true path? (9 minutes)
#
# The **coverage** of a band is the share of weeks in which the true value lies inside that
# week's 94% HDI.
#
# **Predict.** A 94% band that is right covers the truth in about 94% of weeks. In what share of
# the 156 weeks will social's true relative path lie inside the model's 94% HDI: 90% or more, 50
# to 90%, or under 50%? Write your guess in the cell below.

# %% [markdown]
# *Your guess* (double-click to edit):

# %% [markdown]
# **Task.** Write `path_coverage(mult_draws, true_path, prob=0.94)`: `mult_draws` has dims
# `(chain, draw, date)` for one channel; `true_path` is an array with one true value per week.
# For each week compute the HDI of that week's draws, `checks.interval(draws, prob, "hdi")`, which
# returns `(low, high)`; return the share of weeks whose true value lies inside, as a float.

# %% tags=["exercise"]
def path_coverage(mult_draws, true_path, prob=0.94):
    # TODO 4: for k in range(mult_draws.sizes["date"]): mult_draws.isel(date=k).values holds
    # week k's draws from both chains; low, high = checks.interval(those draws, prob, "hdi");
    # is true_path[k] inside? Return the share of weeks inside (about five lines).
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def path_coverage(mult_draws, true_path, prob=0.94):
    true_path = np.asarray(true_path, dtype=float)
    inside = []
    for k in range(mult_draws.sizes["date"]):
        lo, hi = checks.interval(mult_draws.isel(date=k).values, prob, "hdi")
        inside.append(lo <= true_path[k] <= hi)
    return float(np.mean(inside))


# %% [markdown]
# The checkpoint runs four toys with 2 chains × 500 draws × 156 weeks first: (a) draws tightly
# around a known path (sd 0.01) cover it in every week, 1.0; (b) the same draws shifted by +5
# cover it in none, 0.0; (c) skewed draws (the same 1,000 exact quantiles of an Exponential every
# week) with true values 0.01 and 2.5 in alternate weeks give 1.0 with the 94% HDI
# [0.0005, 2.805] and 0.5 with an equal-tailed or an 89% interval; (d) toy (a)'s draws against the
# path reversed in time are covered only where the two cross. Then, on the fit, your coverage of
# each channel must equal the reference's; social's must be at least 0.5 and the other channels'
# at least 0.9 (the truth is `latent["multiplier_rel_<channel>"]`).

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    coverage = drift.check_path_coverage(path_coverage, mult, latent)
    print("Share of weeks whose true relative path lies inside the 94% HDI:",
          {c: round(v, 2) for c, v in coverage.items()})

# %% [markdown]
# Look at where the band misses the true path: which weeks, and on which side.

# %%
social_draws = mult.sel(channel="social").stack(sample=("chain", "draw")).transpose("date", "sample")
band = np.array([checks.interval(row, 0.94, "hdi") for row in social_draws.values])
true_social = latent["multiplier_rel_social"].to_numpy()
missed = (true_social < band[:, 0]) | (true_social > band[:, 1])
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.fill_between(weekly["date_week"], band[:, 0], band[:, 1], alpha=0.3,
                label="model: 94% HDI band")
ax.plot(weekly["date_week"], social_draws.mean("sample"), lw=2, label="model: posterior mean")
ax.plot(latent["date_week"], true_social, "k--", lw=2, label="truth: relative path")
ax.plot(latent["date_week"][missed], true_social[missed], "x", color="black", ms=5,
        label="weeks the band misses")
ax.axhline(1, color="grey", lw=0.8)
ax.set_ylabel("social's multiplier (1 = average)")
ax.legend(loc="lower left", ncol=2)
show(fig)


def week_ranges(weeks):
    """[4, 5, 6, 9] -> "4-6, 9" (weeks counted from 1)."""
    parts, start = [], None
    for i, w in enumerate(weeks):
        if start is None:
            start = w
        if i + 1 == len(weeks) or weeks[i + 1] != w + 1:
            parts.append(f"{start}" if start == w else f"{start}-{w}")
            start = None
    return ", ".join(parts) or "none"


above = (np.flatnonzero(true_social > band[:, 1]) + 1).tolist()
below = (np.flatnonzero(true_social < band[:, 0]) + 1).tolist()
print(f"The band misses {missed.sum()} of {len(missed)} weeks (counting from week 1): the truth"
      f" lies above the band in weeks {week_ranges(above)}, below it in weeks"
      f" {week_ranges(below)}.")

# %% [markdown]
# **Explain.** Compare with your guess. Where does the band miss: at the two ends of the history,
# in the middle, or at random? Point to the plot of the band, the true path and the model's mean
# path.
#
# <details><summary>Why this solution works</summary>
#
# The prior pulls the path toward its average, so the band sits below the truth in the first
# months and above it for most of the last year (FULL run: weeks 5 to 23 and 122 to 152; at week
# 144 the model's mean is 0.68 against a true 0.55). The model sees the decline (the drift
# ratio's HDI lies below 1, in the next cell) but understates it (0.57 against a true 0.42),
# which makes its current ROAS too high, not too low. The effect is identified by the quarterly pushes: the sales jump after
# each push shrinks over time. Without the pushes a fading effect would look like a falling
# baseline, and the model could not tell them apart. What would change it: more pushes, a longer
# history after the decline, or a lift test now (stretch A).
# </details>

# %% [markdown]
# **Labeled check "drifting model".** With your `drift_ratio` on the posterior draws: social's
# drift ratio 94% HDI should lie entirely below 1, and the other three channels' HDIs should
# contain 1. The channels whose HDI lies below 1 are the **fading** channels; the Decision uses
# them.

# %%
ratio_post = drift_ratio(mult, WINDOWS["first_year"], WINDOWS["last_quarter"])
with workshop.checkpoint(label="drifting model"):
    ratio_table = drift.check_drifting_model(ratio_post, truth)
fading = [c for c in CHANNELS if ratio_table.loc[c, "hdi_high"] < 1]
print("Drift ratio (last quarter / first year), 94% HDI, against the truth:")
print(ratio_table.round(2).to_string())
print("Fading channels:", fading)

# %% [markdown]
# Social's current-quarter ROAS, static against drifting model, next to the truth. Look at which
# interval contains the truth.

# %%
roas_draws_drift = {k: window_roas(contrib_drift, spend, *WINDOWS[k]) for k in WINDOWS}
current = pd.concat({"static": roas_table(roas_draws_static, ["last_quarter"]).droplevel("window"),
                     "drifting": roas_table(roas_draws_drift, ["last_quarter"]).droplevel("window")},
                    names=["model"]).swaplevel()
current = current.reindex(pd.MultiIndex.from_product([CHANNELS, ["static", "drifting"]],
                                                     names=["channel", "model"]))
print("ROAS in the last quarter (94% HDI), static against drifting model:")
current.round(2)

# %% [markdown]
# Search's and display's intervals are wide in every fit: search's spend follows demand, so the
# data separate its effect from the trend poorly, and display is small. Letting every channel
# drift widens them further, because a channel's level and its path now trade off against each
# other. The decision below concerns the fading channel.

# %% [markdown]
# # Part D · A Prophet baseline
#
# **Prophet** (Taylor & Letham 2018) is a forecasting model that adds a trend, a yearly seasonal
# pattern and holiday effects, fitted to dates and sales only. Its trend is piecewise linear: a
# **changepoint** is a week where the trend may change slope; Prophet places 25 candidates in the
# first 80% of the history and keeps the slope changes small unless the data insist. Robyn, Meta's
# open-source MMM, uses it this way: it fits Prophet to the sales and adds the trend, season and
# holiday components to its regression as baseline variables; paid media are not Prophet
# regressors. The **baseline** is the part of sales the MMM credits to everything except media:
# trend, season, holidays, price. Part D tests against the truth what a baseline fitted to the same
# sales does to channel ROAS.
#
# **Run.** The next cell defines `prophet_components` (provided), fits Prophet once (seconds) and
# checks the result.

# %%
import logging

# Prophet logs every fit through cmdstanpy at INFO level; keep warnings and errors only. The
# NullHandler stops cmdstanpy from resetting the level when it sets up its own logger.
logging.getLogger("cmdstanpy").addHandler(logging.NullHandler())
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)  # off Colab: "plotly failed", harmless
from prophet import Prophet

holidays = pd.DataFrame({"holiday": "peak_week",
                         "ds": weekly.loc[weekly["holiday"] == 1, "date_week"]})


def prophet_components(weekly, holidays):
    """Prophet's trend, yearly season and holiday components of sales (dollars), one row per
    week in the order of `weekly`, and its fitted sales yhat (for the check)."""
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    holidays=holidays, uncertainty_samples=0)
    df = pd.DataFrame({"ds": weekly["date_week"], "y": weekly["y"].astype(float)})
    model.fit(df)  # a MAP fit that draws no progress bar (unlike the MMM fits: progressbar=False)
    forecast = model.predict(df)
    return pd.DataFrame({"trend": forecast["trend"].to_numpy(),
                         "season": forecast["yearly"].to_numpy(),
                         "holiday": forecast["holidays"].to_numpy(),
                         "yhat": forecast["yhat"].to_numpy()}, index=weekly.index)


components = prophet_components(weekly, holidays)
with workshop.checkpoint(label="prophet"):
    drift.check_prophet(components, weekly)
components.head()

# %% [markdown]
# Look at the trend's growth: the true baseline grows by \$500 a week.

# %%
growth = (components["trend"].iloc[-1] - components["trend"].iloc[0]) / (len(weekly) - 1)
holiday_effect = components.loc[weekly["holiday"] == 1, "holiday"].mean()
print(f"Prophet's trend grows by ${growth:,.0f} a week on average over the three years"
      f" (true baseline: ${truth['baseline']['trend_per_week']:,.0f}).")
print(f"Prophet's holiday effect: ${holiday_effect:,.0f} a holiday week"
      f" (true: ${truth['baseline']['holiday_effect']:,.0f}).")

# %% [markdown]
# Prophet's trend minus the true baseline trend (top), and social's true contribution (bottom).
# The gap is large because Prophet's trend also carries the average level of everything the
# truth keeps out of its trend, media included; read its slope, not its level. Look at whether
# the gap falls when social's contribution falls.

# %%
gap = components["trend"] - latent["baseline_trend"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax1.plot(weekly["date_week"], gap / 1e3, color="tab:blue", lw=2)
ax1.set_ylabel("Prophet's trend minus\ntrue trend ($ thousand)")
ax2.plot(latent["date_week"], latent["contribution_social"] / 1e3, color="tab:orange", ls="-.")
ax2.set_ylabel("social's true\ncontribution ($ thousand)")
fig.tight_layout()
show(fig)
print(f"Correlation between the gap and social's true contribution:"
      f" {np.corrcoef(gap, latent['contribution_social'])[0, 1]:.2f}")

# %% [markdown]
# **Run: the MMM with Prophet's components as its baseline.** An **informed prior** is a prior
# centered on what we already believe: here each Prophet component enters the MMM with a
# coefficient whose prior is centered on 1 with standard deviation 0.1, so the model may rescale
# it by about ±20% (94% of a Normal lies within ±1.88 standard deviations). The media part stays
# static, so the only difference from the static fit is the baseline: Prophet's trend, season and
# holiday, divided by the maximum of sales (PyMC-Marketing divides sales by their maximum and
# leaves controls as they are, so a coefficient of 1 means "use Prophet's component as it is"),
# plus the price index; no trend `t`, holiday flag or Fourier terms of its own.
#
# It takes about a minute on Colab (an estimate until a Colab record exists). While it runs, read
# Exercise 5's Predict, write your guess, and write `baseline_comparison`.

# %%
y_max = float(y.max())
X_prophet = weekly[["date_week", *CHANNELS, "price_index"]].assign(
    prophet_trend=components["trend"] / y_max, prophet_season=components["season"] / y_max,
    prophet_holiday=components["holiday"] / y_max)
CONTROLS = ["price_index", "prophet_trend", "prophet_season", "prophet_holiday"]
on_controls = {"control": CONTROLS}
informed = Prior("Normal", mu=xr.DataArray([0.0, 1.0, 1.0, 1.0], dims="control", coords=on_controls),
                 sigma=xr.DataArray([2.0, 0.1, 0.1, 0.1], dims="control", coords=on_controls),
                 dims="control")  # price: Normal(0, 2), the default; each component: Normal(1, 0.1)
mmm_prophet = MMM(
    date_column="date_week", channel_columns=CHANNELS, target_column="y",
    control_columns=CONTROLS, adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
    yearly_seasonality=None, model_config={**media_priors(), "gamma_control": informed},
)
t_fit = time.time()
idata_prophet = mmm_prophet.fit(X_prophet, y, progressbar=False, **FIT)
prophet_seconds = time.time() - t_fit

# %%
contrib_prophet = idata_prophet.posterior["channel_contribution"] * target_scale(idata_prophet)
roas_draws_prophet = window_roas(contrib_prophet, spend, *WINDOWS["all"])
true_roas = {c: truth["channels"][c]["roas"] for c in CHANNELS}
print(f"Prophet-baseline fit: {prophet_seconds:.0f} s, {drift.divergences(idata_prophet)}"
      " divergences.")

# %% [markdown]
# ## Exercise 5 · Did the Prophet baseline help? (10 minutes)
#
# **Predict.** With Prophet's components as an informed prior on the baseline, will social's
# all-period ROAS rise, stay within 0.3 of the static model's value (the table after the static
# fit), or fall by more than 0.3? The truth is 4.0. Write your guess in the cell below.

# %% [markdown]
# *Your guess* (double-click to edit):

# %% [markdown]
# **Task.** Write `baseline_comparison(roas_own, roas_prophet, true_roas, prob=0.94)`. Both inputs
# are all-period ROAS draws from your `window_roas`, dims `(chain, draw, channel)`: `roas_own` from
# the static model (its own baseline), `roas_prophet` from the Prophet-baseline model. `true_roas`
# is a dict {channel: true ROAS}. Return one row per channel and baseline, index
# `(channel, baseline)` with `baseline` in `{"own", "prophet"}`, and the columns `mean` (posterior
# mean), `hdi_low`, `hdi_high` (the `prob` HDI), `true_roas`, `abs_error` (absolute error of the
# mean) and `covered` (True if the HDI contains the truth).

# %% tags=["exercise"]
def baseline_comparison(roas_own, roas_prophet, true_roas, prob=0.94):
    # TODO 5: loop over {"own": roas_own, "prophet": roas_prophet}.items(), then over the
    # channels (draws.coords["channel"].values); d = draws.sel(channel=c).values.ravel();
    # checks.interval(d, prob, "hdi"); one dict per row; finish with
    # pd.DataFrame(rows).set_index(["channel", "baseline"])
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def baseline_comparison(roas_own, roas_prophet, true_roas, prob=0.94):
    rows = []
    for baseline, draws in {"own": roas_own, "prophet": roas_prophet}.items():
        for c in draws.coords["channel"].values:
            d = draws.sel(channel=c).values.ravel()
            lo, hi = checks.interval(d, prob, "hdi")
            t = true_roas[str(c)]
            rows.append({"channel": str(c), "baseline": baseline, "mean": d.mean(),
                         "hdi_low": lo, "hdi_high": hi, "true_roas": t,
                         "abs_error": abs(d.mean() - t), "covered": lo <= t <= hi})
    return pd.DataFrame(rows).set_index(["channel", "baseline"])


# %% [markdown]
# The checkpoint uses a toy first: for two channels a and b, the "own" draws are the 1,000 exact
# quantiles of a lognormal with median 2 and log-sd 0.3, the Prophet draws the same with median 3;
# the true ROAS is 2.5 for a and 1.10 for b. Reference: own mean 2.0919, 94% HDI
# [1.0209, 3.2904]; Prophet mean 3.1379, [1.5314, 4.9357]; b is covered under "own" (1.10 lies
# just inside the HDI) and not under "prophet". Means must match to 1e-9 and interval bounds to
# within 2% of the interval's width. Then on the fits: for social, the Prophet-baseline mean is
# below the own-baseline mean, its 94% HDI excludes the true all-period ROAS, and the own-baseline
# HDI covers it.

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    comparison = drift.check_baseline_comparison(baseline_comparison, roas_draws_static["all"],
                                                 roas_draws_prophet, true_roas)


def mean_hdi(row):
    return f"{row['mean']:.2f} [{row['hdi_low']:.2f}, {row['hdi_high']:.2f}]"


print("All-period ROAS, mean [94% HDI], with each baseline:")
pd.DataFrame({
    "own baseline": comparison.xs("own", level="baseline").apply(mean_hdi, axis=1),
    "Prophet baseline": comparison.xs("prophet", level="baseline").apply(mean_hdi, axis=1),
    "truth": pd.Series(true_roas).round(2),
}).loc[CHANNELS]

# %% [markdown]
# **Explain.** Compare with your prediction. Which channel lost credit, and where did that credit
# go? Point to the trend-gap plot in Part D.
#
# <details><summary>Why this solution works</summary>
#
# Prophet's trend already contains the slow part of social's effect, so with a coefficient held
# near 1 the baseline keeps it and the MMM gives social less (FULL run: 3.90 with the model's own
# baseline, 2.90 with Prophet's, truth 4.00; tv also lost credit). A fading channel lowers sales
# slowly, and a slow change is what a trend with changepoints is built to follow; Prophet cannot
# know that the fall came from social. Two more problems follow from fitting Prophet to the same
# sales the MMM explains: the MMM treats Prophet's components as known data, so the uncertainty in
# Prophet's own fit never reaches the ROAS intervals; and the intervals look reasonable while
# excluding the truth: they did not become narrower, they moved. What would change it: a baseline
# the MMM estimates jointly with the media (Module 8's trend and Fourier terms), or Prophet told
# about spend (stretch B).
# </details>

# %% [markdown]
# ## Decision · Cut or keep social next quarter?
#
# **The number.** For each fading channel: the drifting model's last-quarter ROAS (posterior mean
# and 94% HDI) and the posterior probability that it is below break-even, computed draw by draw;
# beside it, for contrast, the static model's all-period ROAS with the same probability.
#
# **The rule.** A dollar of sales earns the 30% gross margin, so break-even ROAS is 1 ÷ 0.30 ≈
# 3.33. Cut a channel's budget next quarter if P(current-quarter ROAS < 3.33) ≥ 0.9; keep it if
# P(> 3.33) ≥ 0.9; otherwise run a lift test first (Module 9). We assume next quarter's effect per
# dollar equals last quarter's: beyond the last week the model's multiplier drifts back toward its
# average of 1, so we do not extrapolate it. Under saturation the last dollar returns less than
# the average dollar, so an average below break-even means the marginal dollar is below it too.
# The 0.9 is a business choice: how sure we want to be before moving money.
#
# The next cell applies the rule in code to the channels in `fading` (from the labeled check
# after Exercise 4), not to a channel typed by hand; the other channels' ROAS are in the table
# after Exercise 4.

# %%
def verdict(draws):
    p_below = float(np.mean(draws < BREAK_EVEN))
    if p_below >= 0.9:
        return "cut", p_below
    if 1 - p_below >= 0.9:
        return "keep", p_below
    return "lift test first", p_below


def money(x):
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


mode = "FULL run." if not QUICK else "QUICK run: treat this as a rough answer."
lq_start, lq_end = WINDOWS["last_quarter"]
for c in fading:
    now = roas_draws_drift["last_quarter"].sel(channel=c).values.ravel()
    avg = roas_draws_static["all"].sel(channel=c).values.ravel()
    v_now, p_now = verdict(now)
    v_avg, p_avg = verdict(avg)
    lo_now, hi_now = checks.interval(now, 0.94, "hdi")
    lo_avg, hi_avg = checks.interval(avg, 0.94, "hdi")
    spent = float(spend.sel(date=slice(lq_start, lq_end), channel=c).sum())
    net = spent * (GROSS_MARGIN * now - 1)  # margin on the sales it produced, net of its cost
    lo_net, hi_net = checks.interval(net, 0.94, "hdi")
    print(f"Number          : {c}, last quarter, drifting model: ROAS {now.mean():.2f}"
          f" (94% HDI {lo_now:.2f} to {hi_now:.2f}),")
    print(f"                  P(ROAS < {BREAK_EVEN:.2f}) = {p_now:.2f}.")
    print(f"                  {c}, all weeks, static model: ROAS {avg.mean():.2f} (94% HDI"
          f" {lo_avg:.2f} to {hi_avg:.2f}), P(ROAS > {BREAK_EVEN:.2f}) = {1 - p_avg:.3f}.")
    print(f"Scale           : {c} spent {money(spent)} last quarter; net of that spend, the"
          f" margin on the sales it")
    print(f"                  produced was about {money(net.mean())} (94% HDI {money(lo_net)} to"
          f" {money(hi_net)}).")
    print(f"Rule            : cut if P(current-quarter ROAS < {BREAK_EVEN:.2f}) >= 0.9; keep if"
          f" P(> {BREAK_EVEN:.2f}) >= 0.9;")
    print("                  otherwise run a lift test first.")
    print(f"Verdict         : drifting model, last quarter: {v_now}. Static model, all weeks:"
          f" {v_avg}.")
    print("Assumes         : next quarter's effect per dollar equals last quarter's.")
    print("What would change it: a creative refresh that a test shows has brought the effect"
          " back above")
    print("                  break-even; a lift test now (stretch A).")
    print("Your sentence   : ________________________________________________")
print(mode)

# %% [markdown]
# **Decide.** Write one sentence: what to do with social's budget next quarter (its last-quarter
# spend is printed above), how sure you are, how your answer differs from the static model's, and
# what would change it.
#
# ```text
# Your sentence: ________________________________________________
# ```
#
# In Exercise 4 the drifting model understated the fall: its current-quarter ROAS sits above the
# truth, not below it. So a "cut" from it is not the model exaggerating the fade. Then run the
# next cell, which shows the truth the data were generated from.

# %%
for c in fading:
    t_lq = truth["channels"][c]["roas_last_quarter"]
    print(f"{c}: true ROAS {truth['channels'][c]['roas']:.2f} over all weeks,"
          f" {truth['channels'][c]['roas_first_year']:.2f} in the first year and {t_lq:.2f} in the"
          f" last quarter: {'below' if t_lq < BREAK_EVEN else 'above'} break-even"
          f" ({BREAK_EVEN:.2f}) now, {'above' if truth['channels'][c]['roas'] > BREAK_EVEN else 'below'}"
          " it on average.")

# %% [markdown]
# ## Stretch (optional) · A lift test now, and Prophet told about spend
#
# Nothing later depends on these. Each refit takes about as long as the fit it repeats.
#
# **A · A lift test now.** `mmm_drift_lift_tests.csv` holds one randomized social test (Module 9):
# +30% weekly spend for 8 weeks ending 2025-11-17. Add it to a fresh drifting model:
# `drifting_mmm()`, then `build_model(X, y)`, then `add_lift_test_measurements(df)`, then fit.
# With time-varying media, PyMC-Marketing multiplies the saturation curve by the multiplier at
# the test's `date`, which must be one of the model's weeks, so a recent test pins the current
# level. Compare social's last-quarter ROAS, its HDI and P(below break-even) with the Decision.
#
# **B · Tell Prophet about spend.** Copy `prophet_components` into a new function that calls
# `model.add_regressor(c)` for each channel before fitting, and puts each spend column into the
# frame passed to `fit` and to `predict` (Prophet's documentation: a regressor "must be known for
# both the history and for future dates"). Rebuild the components, refit the Prophet-baseline
# model, and compare with `baseline_comparison`. Which channels move toward the truth?

# %% tags=["solution"]
# @title Stretch A solution — try it yourself first { display-mode: "form" }
lift_tests = pd.read_csv(repo_file("synthetic/mmm_drift_lift_tests.csv"),
                         parse_dates=["date", "test_start"])
mmm_lift = drifting_mmm()
mmm_lift.build_model(X, y)
mmm_lift.add_lift_test_measurements(lift_tests[["channel", "x", "delta_x", "delta_y", "sigma",
                                                "date"]])
idata_lift = mmm_lift.fit(X, y, progressbar=False, **FIT)
lift_lq = window_roas(idata_lift.posterior["channel_contribution"] * target_scale(idata_lift),
                      spend, *WINDOWS["last_quarter"]).sel(channel="social").values.ravel()
for name, d in [("without the test", roas_draws_drift["last_quarter"].sel(channel="social")
                 .values.ravel()), ("with the test", lift_lq)]:
    lo, hi = checks.interval(d, 0.94, "hdi")
    print(f"social, last quarter, {name}: ROAS {d.mean():.2f} (94% HDI {lo:.2f} to {hi:.2f}),"
          f" P(< {BREAK_EVEN:.2f}) = {np.mean(d < BREAK_EVEN):.2f}")
print(f"Truth: {truth['channels']['social']['roas_last_quarter']:.2f};"
      f" {drift.divergences(idata_lift)} divergences.")

# %% tags=["solution"]
# @title Stretch B solution — try it yourself first { display-mode: "form" }
def prophet_components_with_spend(weekly, holidays):
    """prophet_components with each channel's spend as an extra regressor."""
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    holidays=holidays, uncertainty_samples=0)
    df = pd.DataFrame({"ds": weekly["date_week"], "y": weekly["y"].astype(float)})
    for c in CHANNELS:
        model.add_regressor(c)
        df[c] = weekly[c].astype(float)
    model.fit(df)  # a MAP fit that draws no progress bar (unlike the MMM fits: progressbar=False)
    forecast = model.predict(df)
    return pd.DataFrame({"trend": forecast["trend"].to_numpy(),
                         "season": forecast["yearly"].to_numpy(),
                         "holiday": forecast["holidays"].to_numpy()}, index=weekly.index)


components_b = prophet_components_with_spend(weekly, holidays)
X_b = weekly[["date_week", *CHANNELS, "price_index"]].assign(
    prophet_trend=components_b["trend"] / y_max, prophet_season=components_b["season"] / y_max,
    prophet_holiday=components_b["holiday"] / y_max)
mmm_b = MMM(
    date_column="date_week", channel_columns=CHANNELS, target_column="y",
    control_columns=CONTROLS, adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
    yearly_seasonality=None, model_config={**media_priors(), "gamma_control": informed},
)
idata_b = mmm_b.fit(X_b, y, progressbar=False, **FIT)
roas_draws_b = window_roas(idata_b.posterior["channel_contribution"] * target_scale(idata_b),
                           spend, *WINDOWS["all"])
pd.concat({"Prophet": comparison.xs("prophet", level="baseline"),
           "Prophet with spend": baseline_comparison(roas_draws_static["all"], roas_draws_b,
                                                     true_roas).xs("prophet", level="baseline")},
          names=["baseline"]).swaplevel().sort_index(level="channel", sort_remaining=False)[
    ["mean", "hdi_low", "hdi_high", "true_roas", "covered"]].round(2)
