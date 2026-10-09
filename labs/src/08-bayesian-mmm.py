# %% [markdown]
# <!--
# APIs this lab calls, checked against the installed source of pymc-marketing 1.2.0, pymc 6.3.2,
# pymc-extras 0.15.1 and ArviZ 1.3.0 (.venv/lib/python3.13/site-packages, 2026-10-09):
# - pymc_marketing.mmm.MMM(date_column=, channel_columns=, target_column=, adstock=, saturation=,
#   control_columns=, yearly_seasonality=, model_config=) (mmm/mmm.py); .build_model(X, y);
#   .sample_prior_predictive(X, y, samples=, random_seed=) returns a Dataset whose y is in model
#   units (sales / max sales); .fit(X, y, nuts_sampler="nutpie", chains=, draws=, tune=,
#   target_accept=, random_seed=, progressbar=) returns an xarray DataTree;
#   .sample_posterior_predictive(X, extend_idata=True, random_seed=, progressbar=) (y in model units);
#   .incrementality.contribution_over_spend(frequency="all_time", start_date=, end_date=)
#   (mmm/incrementality.py) has dims (chain, draw, channel). Over all weeks it equals the sum of
#   channel_contribution x target_scale / spend to 1e-15 (measured), the definition of truth roas.
# - GeometricAdstock(l_max=8) (normalize=True by default), LogisticSaturation(); default priors
#   adstock_alpha Beta(1, 3), saturation_lam Gamma(3, 1), saturation_beta HalfNormal(2)
#   (mmm/components/adstock.py, saturation.py).
# - mmm.transformers.geometric_adstock(x, alpha, l_max, *, dim, normalize) and
#   logistic_saturation(x, lam) take xtensors: pytensor.xtensor.as_xtensor(x, dims=("date",)).
# - pymc_extras.prior.Prior("HalfNormal", sigma=<DataArray over channel>, dims="channel").
# - idata["/sample_stats"]["diverging"]; az.rhat, az.ess(method="bulk"),
#   az.summary(var_names=, ci_prob=, ci_kind=, round_to="none").
# - After a nutpie fit pymc-marketing recomputes deterministics with a progress bar it does not
#   let us switch off (model_builder.py, pm.compute_deterministics); the fit cell prints nothing
#   else, so its numbers are printed in the next cell.
# -->
#
# # Part A · The two transforms
#
# A *marketing mix model* (MMM) explains weekly sales with a baseline (trend, seasonality, price,
# holidays) plus one term per media channel. Each channel's term passes the channel's spend
# through two transforms and multiplies the result by an effect size:
#
# 1. **Adstock** spreads one week's spend over the following weeks: people remember an ad.
# 2. **Saturation** bends the curve: each extra dollar returns less than the one before.
#
# You write both and check them against PyMC-Marketing's versions. Then you fit the model and
# ask what each channel's dollars returned.
#
# The data: three years of weekly sales and spend on four channels (tv, search, social, display)
# for the workshop's synthetic retailer, with a price index and a holiday flag. We generated it,
# so we know every true parameter; they are in `truth`.

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

from mktstats.data import load_mmm_example, load_synthetic, load_truth

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
truth = load_truth()["mmm"]
CHANNELS = ["tv", "search", "social", "display"]
SEED = 2028  # every random step in this notebook uses this seed
print(f"{len(weekly)} weeks, {weekly['date_week'].min():%Y-%m-%d} to"
      f" {weekly['date_week'].max():%Y-%m-%d}")
weekly.head()

# %% [markdown]
# Look at how differently the channels are bought: tv in bursts with weeks off, search steady and
# rising with the season, social with a few four-week pushes, display small and slowly growing.
# How much a channel's spend moves on its own decides how well its effect can be measured.

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
for c, style in zip(CHANNELS, ["-", "--", "-.", ":"], strict=True):
    ax1.plot(weekly["date_week"], weekly[c] / 1e3, style, label=c)
ax1.set_ylabel("spend ($ thousand / week)")
ax1.legend(ncol=4, loc="upper left")
ax2.plot(weekly["date_week"], weekly["y"] / 1e3, color="black")
ax2.set_ylabel("sales ($ thousand / week)")
fig.tight_layout()
show(fig)

# %% [markdown]
# ## Exercise 1 · Geometric adstock (7 minutes)
#
# Geometric adstock with decay $\alpha$ (between 0 and 1) and maximum lag $L = 8$ weeks replaces
# spend $x_t$ by a weighted sum of this week's and the previous $L-1$ weeks' spend:
#
# $$a_t = \sum_{l=0}^{L-1} w_l\, x_{t-l}, \qquad
#   w_l = \frac{\alpha^l}{\sum_{k=0}^{L-1} \alpha^k} \quad\text{(normalized: the weights sum to 1)}.$$
#
# Weeks before the first one count as zero spend. Without normalization, $w_l = \alpha^l$.
#
# **Predict.** With $\alpha = 0.5$ and normalized weights, what share of one week's spend lands
# in the week of spend itself: about a quarter, about a half, or about three quarters? Write
# your guess down.
#
# **Task.** Write `geometric_adstock(x, alpha, l_max=8, normalize=True)` for a 1-D NumPy array
# `x` of weekly spend, returning an array of the same length.

# %% tags=["exercise"]
def geometric_adstock(x, alpha, l_max=8, normalize=True):
    # TODO 1: weights alpha**l for l = 0 .. l_max - 1 (divided by their sum if normalize),
    # then out[t] = sum over l of w[l] * x[t - l], with zero spend before the first week.
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def geometric_adstock(x, alpha, l_max=8, normalize=True):
    x = np.asarray(x, dtype=float)
    w = alpha ** np.arange(l_max)
    if normalize:
        w = w / w.sum()
    # np.convolve(x, w)[t] = sum_l w[l] * x[t - l]; the first len(x) values have zero history.
    return np.convolve(x, w)[: len(x)]


# %% [markdown]
# **Explain.** Compare with your guess: the weights for $\alpha = 0.5$ are 0.502, 0.251, 0.126,
# ... so half of the effect lands in the week of spend. Which property of normalized adstock
# keeps the total effect of a week's spend the same whatever $\alpha$ is?
#
# <details><summary>Why this solution works</summary>
#
# Adstock is a convolution of the spend series with the weights, so `np.convolve` computes it
# directly; keeping the first `len(x)` values means weeks before the data count as zero spend.
# Normalized weights sum to 1, so a dollar's effect is spread over later weeks but its total is
# unchanged: $\alpha$ only decides *when* the effect arrives. A larger $\alpha$ spreads it more
# (tv in this data has $\alpha = 0.6$, search 0.2). Without normalization a larger $\alpha$ would
# also make the total larger, and the effect size would absorb the difference.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    from pymc_marketing.mmm import transformers  # PyMC-Marketing's version, for comparison
    from pytensor.xtensor import as_xtensor

    impulse = np.zeros(12)
    impulse[0] = 1.0
    w_true = 0.5 ** np.arange(8) / (0.5 ** np.arange(8)).sum()
    got = np.asarray(geometric_adstock(impulse, alpha=0.5), dtype=float)
    assert got.shape == impulse.shape, (
        f"geometric_adstock returned {got.shape[0]} values for {impulse.shape[0]} weeks: keep"
        " only the first len(x) values of the convolution."
    )
    assert np.allclose(got, np.r_[w_true, np.zeros(4)], atol=1e-12), (
        f"One week of spend (1 in week 0) should give the weights {np.round(w_true, 3).tolist()}"
        f" in weeks 0-7 and 0 after; your function gives {np.round(got, 3).tolist()}. Check that"
        " week t uses spend from weeks t, t-1, ... (not later weeks) and that the weights are"
        " divided by their sum."
    )
    x_check = np.random.default_rng(SEED).uniform(0, 1, 52)
    for a, norm in [(0.3, True), (0.7, True), (0.7, False)]:
        ref = transformers.geometric_adstock(
            as_xtensor(x_check, dims=("date",)), alpha=a, l_max=8, dim="date", normalize=norm
        ).eval()
        checks.close(geometric_adstock(x_check, a, 8, norm), ref, abs=1e-8,
                     name=f"your adstock (alpha={a}, normalize={norm}) against PyMC-Marketing's")

# %% [markdown]
# ## Exercise 2 · Logistic saturation (5 minutes)
#
# Logistic saturation with parameter $\lambda > 0$ maps adstocked spend $a \ge 0$ to a number
# between 0 and 1:
#
# $$s(a) = \frac{1 - e^{-\lambda a}}{1 + e^{-\lambda a}}.$$
#
# In the model the channel's effect is $\beta \cdot s(a)$, so $\beta$ is the most the channel can
# ever add in a week and $\lambda$ says how fast it gets there.
#
# **Predict.** At $\lambda a = 1$, what share of the maximum effect does logistic saturation
# give: under 50%, about 50%, or over 50%?
#
# **Task.** Write `logistic_saturation(x, lam)` for a NumPy array `x`.

# %% tags=["exercise"]
def logistic_saturation(x, lam):
    # TODO 2: (1 - exp(-lam * x)) / (1 + exp(-lam * x))
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def logistic_saturation(x, lam):
    e = np.exp(-lam * np.asarray(x, dtype=float))
    return (1 - e) / (1 + e)


# %% [markdown]
# **Explain.** $s = 0.462$ at $\lambda a = 1$: just under half. Each extra dollar returns less
# than the one before. Why does that make "the ROAS of a channel" depend on how much you
# already spend on it?
#
# <details><summary>Why this solution works</summary>
#
# The formula is $\tanh(\lambda a / 2)$: zero at zero spend, rising steeply at first and
# flattening towards 1. Its slope (the return on the next dollar) falls as spend grows, so the
# average return over all dollars is higher than the return on the last one. Adstock followed by
# saturation gives the *response curve* that Module 10 uses to divide a budget.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    from pymc_marketing.mmm import transformers
    from pytensor.xtensor import as_xtensor

    grid = np.linspace(0, 3, 61)
    for lam in (0.5, 2.0, 4.0):
        ref = transformers.logistic_saturation(as_xtensor(grid, dims=("date",)), lam=lam).eval()
        checks.close(logistic_saturation(grid, lam), ref, abs=1e-8,
                     name=f"your saturation (lam={lam}) against PyMC-Marketing's")
    out = logistic_saturation(grid, 2.0)
    checks.monotone(out, increasing=True, name="logistic_saturation over increasing spend")
    assert 0 <= out.min() and out.max() < 1, (
        f"Saturation should lie in [0, 1); yours runs from {out.min():.3g} to {out.max():.3g}."
        " Check the signs in the formula."
    )

# %% [markdown]
# # Part B · Priors before data
#
# A *prior* is what the model believes about a parameter before it sees sales. PyMC-Marketing
# divides each channel's spend by its maximum and sales by their maximum before fitting, so the
# channel effect $\beta_c$ is in units of peak weekly sales: $\beta_c = 0.2$ means channel $c$
# can add at most a fifth of the best week's sales. Its default prior is HalfNormal with scale
# 2 for every channel.
#
# ## Exercise 3 · Priors from spend shares (7 minutes)
#
# Our design choice: without other information, expect a channel to matter in proportion to
# what the business spends on it. So channel $c$'s prior scale is
# $\sigma_c = C \cdot \text{share}_c$, where $\text{share}_c$ is its share of total spend over all
# weeks and $C$ the number of channels (so the scales average 1).
#
# **Predict.** Before seeing sales, should the channel with the largest spend get a wider or a
# narrower prior on its effect than the smallest channel?
#
# **Task.** Write `beta_prior_sigma(spend)`: `spend` is a DataFrame with one column per channel
# (weekly spend); return a NumPy array of the scales $\sigma_c$ in the column order.

# %% tags=["exercise"]
def beta_prior_sigma(spend):
    # TODO 3: each channel's share of total spend, times the number of channels
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def beta_prior_sigma(spend):
    totals = spend.sum(axis=0).to_numpy(dtype=float)
    return len(totals) * totals / totals.sum()


# %% [markdown]
# **Explain.** tv gets the widest prior and display the narrowest. Compare with the default
# (scale 2 for every channel): what does the default say about how much display, a fifth of
# tv's spend, could add to sales?
#
# <details><summary>Why this solution works</summary>
#
# A HalfNormal prior with scale $\sigma$ puts about 95% of its mass below $2\sigma$, so the
# scale caps what the model expects a channel can do before seeing data. The default gives a
# small channel the same room as the largest one. Spend shares encode a weak, defensible belief
# (the business spends where it expects returns) and still let the data move each effect. They
# are not evidence: a lift test (Module 9) is.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    sigma = np.asarray(beta_prior_sigma(weekly[CHANNELS]), dtype=float)
    assert sigma.shape == (len(CHANNELS),), (
        f"Expected one scale per channel ({len(CHANNELS)}), got shape {sigma.shape}."
    )
    assert (sigma > 0).all(), f"Every scale must be positive; got {np.round(sigma, 3).tolist()}."
    checks.close(sigma.sum(), len(CHANNELS), rel=1e-9,
                 name="the sum of the scales (they should average 1, so sum to the channel count)")
    order = weekly[CHANNELS].sum().sort_values().index.tolist()
    got_order = [CHANNELS[i] for i in np.argsort(sigma)]
    assert got_order == order, (
        f"The scales should rank the channels as their total spend does ({order}, smallest"
        f" first); yours rank them {got_order}. Use each channel's share of total spend."
    )

# %% [markdown]
# **Run.** Build the model with your priors. `MMM` takes the column names, PyMC-Marketing's
# versions of your two transforms, the controls (price index, holiday flag, a linear trend `t`)
# and two orders of yearly seasonality (sine and cosine terms). Then draw 500 fake sales series
# from the priors alone: the *prior predictive check*.

# %%
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

X = weekly.drop(columns=["y"])
y = weekly["y"]
beta_sigma = xr.DataArray(beta_prior_sigma(weekly[CHANNELS]), dims="channel",
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
prior = mmm.sample_prior_predictive(X, y, samples=500, random_seed=SEED)
y_scale = float(weekly["y"].max())  # PyMC-Marketing's target scale: model units x y_scale = $


def weekly_band(draws, prob=0.94):
    """Per-week HDI of draws shaped (weeks, samples)."""
    return np.array([checks.interval(row, prob, "hdi") for row in np.asarray(draws)])


prior_y = prior["y"].transpose("date", "sample").values * y_scale
prior_band = weekly_band(prior_y)
media_share = (mmm.idata["/prior"]["channel_contribution"].sum(["date", "channel"]).values.ravel()
               * y_scale / weekly["y"].sum())
lo, hi = checks.interval(media_share, 0.94, "hdi")
print(f"Prior 94% HDI of weekly sales, widest week: ${prior_band[:, 0].min() / 1e6:,.1f}M to"
      f" ${prior_band[:, 1].max() / 1e6:,.1f}M (observed sales: up to ${y_scale / 1e6:.2f}M)")
print(f"Prior 94% HDI of media's share of all sales: {lo:.0%} to {hi:.0%}")

# %% [markdown]
# Look at whether the observed sales (black) lie inside the prior's 94% band, and whether the
# band allows absurd values: negative sales, or ten times the observed.

# %%
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.fill_between(weekly["date_week"], prior_band[:, 0] / 1e6, prior_band[:, 1] / 1e6,
                alpha=0.3, label="prior predictive, 94% HDI")
ax.plot(weekly["date_week"], weekly["y"] / 1e6, color="black", label="observed")
ax.set_ylabel("sales ($ million / week)")
ax.legend(loc="upper left")
show(fig)

# %% [markdown]
# The band covers the data, and it is absurd: it allows negative sales and sales hundreds of
# times the observed, more so every week (compare the two printed numbers). The main cause is
# not the channel priors. The trend control `t` runs from 0 to 155 and its coefficient has the
# default prior Normal(0, 2) on sales divided by their maximum, so the trend alone may move
# sales by hundreds of times by the last week. The channel priors are vague too, on purpose:
# media's share of all sales may be anywhere from a fraction of sales to several times all
# sales (the second printed line). Here 156 weeks of data pin all of these down, so we keep
# them; on your own data, rescale `t` (to years, say) or narrow `gamma_control`, and run this
# check again.
#
# **Run.** Fit the model by MCMC with the nutpie sampler: two chains. `target_accept=0.9` makes
# the sampler take smaller steps than the default 0.8; on this model 0.8 left a divergence in
# one of six test runs (what a divergence is: Exercise 4). Read the next exercise's Predict
# prompt while it runs.
#
# With QUICK on, the sampler draws 300 instead of 2,000 values per chain: intervals are rougher
# and may not match the module page, and a decision near its threshold can flip. That is a
# lesson about sample size, not a bug.

# %%
DRAWS = 300 if QUICK else 2000
t_fit = time.time()
idata = mmm.fit(X, y, nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS,
                target_accept=0.9, random_seed=SEED, progressbar=False)
fit_seconds = time.time() - t_fit

# %%
mmm.sample_posterior_predictive(X, extend_idata=True, random_seed=SEED, progressbar=False)
print(f"Fitted 2 chains x {DRAWS} draws in {fit_seconds:.1f} s"
      f" ({'QUICK' if QUICK else 'FULL'} settings).")

# %% [markdown]
# # Part C · Can we trust the fit?
#
# MCMC draws are only a sample from the posterior if the sampler worked. Three checks, before
# reading any result:
#
# - **Divergences**: steps where the sampler's numerical path broke down, usually in a region
#   of the posterior that is hard to explore. Draws near them may be biased. We want 0.
# - **R-hat** compares the chains with each other (and the halves of each chain): near 1 when
#   they agree. We want at most 1.01 (Vehtari et al. 2021).
# - **Bulk ESS**, the effective sample size: how many independent draws the correlated draws
#   are worth for estimating the posterior's center. We want at least 400 (same source).
#
# ## Exercise 4 · Diagnostics before results (7 minutes)
#
# **Predict.** Will the fit have no divergences, a few (under 10), or many?
#
# **Task.** Write `diagnose(idata, var_names=PARAMS)` returning a dict with `divergences` (the
# number of divergent draws, summed from `idata["/sample_stats"]["diverging"]`), `max_rhat`
# (largest `az.rhat` over the parameters in `var_names`) and `min_ess_bulk` (smallest
# `az.ess(..., method="bulk")`). The provided `toy` below is a fake posterior with known
# problems to test on: `diagnose(toy, ["adstock_alpha"])`.

# %%
PARAMS = ["adstock_alpha", "saturation_lam", "saturation_beta", "gamma_control",
          "gamma_fourier", "intercept_contribution", "y_sigma"]

# A fake posterior with known problems: chain 1 is stuck away from chain 0, and 3 draws diverged.
rng_toy = np.random.default_rng(SEED)
toy_alpha = rng_toy.normal(0.5, 0.05, size=(2, 200, 2))
toy_alpha[1] += 0.3
toy_div = np.zeros((2, 200), dtype=bool)
toy_div[0, [10, 70, 150]] = True
toy = xr.DataTree.from_dict({
    "posterior": xr.Dataset({"adstock_alpha": (("chain", "draw", "channel"), toy_alpha)},
                            coords={"chain": [0, 1], "draw": np.arange(200),
                                    "channel": ["a", "b"]}),
    "sample_stats": xr.Dataset({"diverging": (("chain", "draw"), toy_div)},
                               coords={"chain": [0, 1], "draw": np.arange(200)}),
})


# %% tags=["exercise"]
def diagnose(idata, var_names=PARAMS):
    # TODO 4: count divergences; largest R-hat and smallest bulk ESS over var_names
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def diagnose(idata, var_names=PARAMS):
    rhat = az.rhat(idata, var_names=var_names)
    ess = az.ess(idata, var_names=var_names, method="bulk")
    return {
        "divergences": int(idata["/sample_stats"]["diverging"].sum()),
        "max_rhat": max(float(rhat[v].max()) for v in rhat.data_vars),
        "min_ess_bulk": min(float(ess[v].min()) for v in ess.data_vars),
    }


# %% [markdown]
# **Explain.** Compare with your guess. Which parameter has the smallest ESS (the table below
# names it), and why might two parameters that can trade off against each other, such as a
# channel's $\lambda$ and $\beta$, be slow to explore?
#
# <details><summary>Why this solution works</summary>
#
# `az.rhat` and `az.ess` return one value per parameter element (per channel, per control);
# the worst one decides whether the fit can be trusted. Over the range of spend a channel
# actually had, a steeper curve (larger $\lambda$) with a lower ceiling (smaller $\beta$) can fit
# the data about as well as a flatter one with a higher ceiling. The sampler then moves slowly
# along that ridge, so some channel's $\lambda$ and $\beta$ have the lowest ESS, even when its
# ROAS, which depends on both together, is estimated well. More draws (or a parameterization
# without the ridge) raise the ESS; a divergence calls for a smaller step (`target_accept`
# closer to 1) or a different prior.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    diag = diagnose(idata)
    checks.columns(pd.DataFrame([diag]), ["divergences", "max_rhat", "min_ess_bulk"],
                   name="the dict diagnose(idata) returns")
    summary = az.summary(idata, var_names=PARAMS, ci_prob=0.94, ci_kind="hdi", round_to="none")
    n_div = int(idata["/sample_stats"]["diverging"].sum())
    assert diag["divergences"] == n_div, (
        f"The fit has {n_div} divergent draws; diagnose says {diag['divergences']}. Sum the"
        " boolean array idata['/sample_stats']['diverging'] over chains and draws."
    )
    checks.close(diag["max_rhat"], summary["r_hat"].max(), rel=1e-6,
                 name="max_rhat (compare az.summary's r_hat column)")
    checks.close(diag["min_ess_bulk"], summary["ess_bulk"].min(), rel=1e-6,
                 name="min_ess_bulk (compare az.summary's ess_bulk column)")
    toy_diag = diagnose(toy, ["adstock_alpha"])
    assert toy_diag["divergences"] == 3, (
        f"The toy posterior has 3 divergent draws; diagnose says {toy_diag['divergences']}."
    )
    assert toy_diag["max_rhat"] > 1.1, (
        f"The toy's chain 1 is stuck away from chain 0, so R-hat must be far above 1.01;"
        f" diagnose gives {toy_diag['max_rhat']:.3f}. Take the largest value, not the mean."
    )

# %% [markdown]
# The gate: read results only if all three pass. With QUICK on, the R-hat and ESS bars usually
# fail (300 draws per chain are too few); that is expected, and the results below are rough.
# The FULL run draws 2,000 per chain so that it can pass them.

# %%
gate = {
    "divergences = 0": diag["divergences"] == 0,
    "max R-hat <= 1.01": diag["max_rhat"] <= 1.01,
    "min bulk ESS >= 400": diag["min_ess_bulk"] >= 400,
}
print(f"Divergences {diag['divergences']}, max R-hat {diag['max_rhat']:.3f},"
      f" min bulk ESS {diag['min_ess_bulk']:.0f} ({'QUICK' if QUICK else 'FULL'} run)")
for rule, ok in gate.items():
    print(f"  {'pass' if ok else 'FAIL'}  {rule}")
print("Lowest ESS:", summary["ess_bulk"].nsmallest(3).round(0).to_dict())

# %% [markdown]
# The second check is the *posterior predictive check*: data simulated from the fitted model
# should look like the data. Look for weeks where the black line leaves the band, and whether
# they cluster (holidays, the start of the series).

# %%
post_y = mmm.idata["/posterior_predictive"]["y"].stack(sample=("chain", "draw"))
post_band = weekly_band(post_y.transpose("date", "sample").values * y_scale)
outside = (weekly["y"] < post_band[:, 0]) | (weekly["y"] > post_band[:, 1])
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.fill_between(weekly["date_week"], post_band[:, 0] / 1e3, post_band[:, 1] / 1e3, alpha=0.3,
                label="posterior predictive, 94% HDI")
ax.plot(weekly["date_week"], weekly["y"] / 1e3, color="black", lw=1, label="observed")
ax.plot(weekly["date_week"][outside], weekly["y"][outside] / 1e3, "x", color="black",
        label="outside the band")
ax.set_ylabel("sales ($ thousand / week)")
ax.legend(loc="upper left")
show(fig)
print(f"{outside.sum()} of {len(weekly)} weeks fall outside the 94% band"
      f" (about {0.06 * len(weekly):.0f} expected by chance)")

# %% [markdown]
# # Part D · What each dollar returned
#
# The *return on ad spend* (ROAS) of a channel over a window is the sales its spend caused,
# divided by that spend. "Caused" means a counterfactual: sales as fitted minus sales had the
# channel spent nothing in the window. PyMC-Marketing computes it for every posterior draw:

# %%
w0, w1 = truth["roas_window"]["start"], truth["roas_window"]["end"]
roas_draws = mmm.incrementality.contribution_over_spend(frequency="all_time",
                                                        start_date=w0, end_date=w1)
true_roas = {c: truth["channels"][c]["roas"] for c in CHANNELS}
print(roas_draws.dims, dict(roas_draws.sizes))

# %% [markdown]
# ## Exercise 5 · ROAS against the truth (9 minutes)
#
# **Predict.** Which channel will have the widest ROAS interval: tv, search, social or display?
# Pick one, and look at the spend plot at the top for a reason.
#
# **Task.** Write `roas_table(roas_draws, true_roas, prob=0.94)` returning a DataFrame indexed by
# channel with columns `mean` (posterior mean ROAS), `hdi_low`, `hdi_high` (the `prob` HDI:
# `checks.interval(draws, prob, "hdi")` computes it), `true_roas` and `covered` (True if the
# truth lies inside the interval). `roas_draws.sel(channel=c).values` gives one channel's draws.

# %% tags=["exercise"]
def roas_table(roas_draws, true_roas, prob=0.94):
    # TODO 5: one row per channel: mean, hdi_low, hdi_high, true_roas, covered
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def roas_table(roas_draws, true_roas, prob=0.94):
    rows = []
    for c in roas_draws.coords["channel"].values:
        d = roas_draws.sel(channel=c).values.ravel()
        lo, hi = checks.interval(d, prob, "hdi")
        t = true_roas[str(c)]
        rows.append({"channel": str(c), "mean": d.mean(), "hdi_low": lo, "hdi_high": hi,
                     "true_roas": t, "covered": lo <= t <= hi})
    return pd.DataFrame(rows).set_index("channel")


# %% [markdown]
# **Explain.** Compare with your pick. search has the widest interval although it has the
# highest true ROAS. Look at its line in the spend plot: does it vary much from week to week,
# and does it move with something else that also drives sales?
#
# <details><summary>Why this solution works</summary>
#
# The model learns a channel's effect from weeks where its spend changes while everything else
# does not. tv switches on and off, so its effect stands out and its interval is narrow. search
# spend varies little and rises with the season, which also raises sales: the model cannot
# separate the two well, so many ROAS values fit and the interval is wide. Four 94% intervals
# all cover their truths only about 78% of the time (0.94⁴), so the check asks for at least 3 of
# 4. A channel whose spend followed demand *exactly* would be worse: Module 9 shows how a lift
# test fixes that.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    table = roas_table(roas_draws, true_roas)
    checks.columns(table.reset_index(), ["channel", "mean", "hdi_low", "hdi_high", "true_roas",
                                         "covered"], name="roas_table(...)")
    assert sorted(table.index) == sorted(CHANNELS), (
        f"Expected one row per channel {CHANNELS}, indexed by channel; got {list(table.index)}."
    )
    for c in CHANNELS:
        d = roas_draws.sel(channel=c).values.ravel()
        lo, hi = checks.interval(d, 0.94, "hdi")
        checks.close(table.loc[c, "mean"], d.mean(), rel=1e-9, name=f"the mean ROAS of {c}")
        tol = 0.02 * (hi - lo)
        assert abs(table.loc[c, "hdi_low"] - lo) <= tol and abs(table.loc[c, "hdi_high"] - hi) <= tol, (
            f"{c}: your interval is [{table.loc[c, 'hdi_low']:.3g}, {table.loc[c, 'hdi_high']:.3g}];"
            f" the 94% HDI of the draws is [{lo:.3g}, {hi:.3g}]. Use checks.interval(draws, prob,"
            " 'hdi'): an HDI, at 94%, not ArviZ's default 89% equal-tailed interval."
        )
        assert bool(table.loc[c, "covered"]) == (lo <= true_roas[c] <= hi), (
            f"{c}: 'covered' should be True exactly when the true ROAS lies inside the interval."
        )
    roas_check = checks.k_of_K_in_interval(
        true_roas, {c: roas_draws.sel(channel=c).values for c in CHANNELS}, k=3, prob=0.94,
        kind="hdi", name="channel ROAS",
    )

# %% [markdown]
# Look at whether the true ROAS (black diamond) falls inside each channel's bar, and which bars
# are wide.

# %%
fig, ax = plt.subplots(figsize=(7, 3))
pos = np.arange(len(CHANNELS))
ax.hlines(pos, table["hdi_low"], table["hdi_high"], lw=6, alpha=0.5, label="94% HDI")
ax.plot(table["mean"], pos, "o", label="posterior mean")
ax.plot(table["true_roas"], pos, "D", color="black", label="truth")
ax.set_yticks(pos, CHANNELS)
ax.set_xlabel("ROAS (dollars of sales per dollar of spend, all weeks)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
show(fig)
table.round(3)

# %% [markdown]
# The same check for the adstock decay $\alpha$ (provided, no exercise): does the model recover
# how long each channel's effect lasts?

# %%
alpha_draws = {c: idata.posterior["adstock_alpha"].sel(channel=c).values for c in CHANNELS}
true_alpha = {c: truth["channels"][c]["adstock_alpha"] for c in CHANNELS}
with workshop.checkpoint(label="adstock alpha"):
    alpha_check = checks.k_of_K_in_interval(true_alpha, alpha_draws, k=3, prob=0.94, kind="hdi",
                                            name="adstock decay alpha")
    checks.in_interval(true_alpha["tv"], alpha_draws["tv"], 0.94, "hdi", name="tv's adstock alpha")
pd.DataFrame({c: {"truth": true_alpha[c], "mean": alpha_draws[c].mean(),
                  "hdi_low": alpha_check[c][0], "hdi_high": alpha_check[c][1],
                  "covered": alpha_check[c][2]} for c in CHANNELS}).T.round(3)

# %% [markdown]
# # Part E · A dataset without truth
#
# On real data nobody knows the true ROAS. Next, the same workflow on `mmm_example.csv`, a demo
# file from the PyMC-Marketing repository: 179 weeks of sales `y`, two media variables `x1` and
# `x2`, two event flags and a trend. Its documentation calls it a simulated example and gives no
# true parameters, so we cannot score the fit, only check it. `x1` and `x2` are not dollar spend
# (they run between 0 and 1), so we use default priors and look at shares of the media effect,
# not ROAS.
#
# **Run.** Fit it with the same settings (about as long as the first fit).

# %%
example = load_mmm_example()
mmm_ex = MMM(
    date_column="date_week",
    channel_columns=["x1", "x2"],
    control_columns=["event_1", "event_2", "t"],
    target_column="y",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)
t_fit = time.time()
idata_ex = mmm_ex.fit(example.drop(columns=["y"]), example["y"], nuts_sampler="nutpie", chains=2,
                      draws=DRAWS, tune=DRAWS, target_accept=0.9, random_seed=SEED,
                      progressbar=False)
fit_ex_seconds = time.time() - t_fit

# %%
diag_ex = diagnose(idata_ex)
print(f"mmm_example: fitted in {fit_ex_seconds:.1f} s; divergences {diag_ex['divergences']},"
      f" max R-hat {diag_ex['max_rhat']:.3f}, min bulk ESS {diag_ex['min_ess_bulk']:.0f}")

# %% [markdown]
# ## Exercise 6 · Contribution shares on `mmm_example` (5 minutes)
#
# **Predict.** Which channel contributes more to sales in `mmm_example`, `x1` or `x2`?
#
# **Task.** Write `contribution_share(idata)` returning a DataFrame indexed by channel with
# columns `mean`, `hdi_low`, `hdi_high` (94% HDI): each channel's share of the total media
# contribution. Compute the share *per draw*: sum `idata.posterior["channel_contribution"]` over
# `date`, divide by its sum over channels, then summarize each channel's draws.

# %% tags=["exercise"]
def contribution_share(idata):
    # TODO 6: per draw, channel contribution summed over dates / total over channels
    raise NotImplementedError("TODO 6")


# %% tags=["solution"]
# @title Solution 6 — try it yourself first { display-mode: "form" }
@workshop.solution(6)
def contribution_share(idata):
    contrib = idata.posterior["channel_contribution"].sum("date")
    share = contrib / contrib.sum("channel")
    rows = {}
    for c in share.coords["channel"].values:
        d = share.sel(channel=c).values.ravel()
        lo, hi = checks.interval(d, 0.94, "hdi")
        rows[str(c)] = {"mean": d.mean(), "hdi_low": lo, "hdi_high": hi}
    return pd.DataFrame(rows).T.rename_axis("channel")


# %% [markdown]
# **Explain.** Compare with your guess. Without truth, what protects you from a wrong answer
# here: name two checks from this notebook, and one kind of outside evidence.
#
# <details><summary>Why this solution works</summary>
#
# Shares are computed per draw, so every draw's shares sum to 1 and the interval of a share
# carries the posterior's uncertainty about both channels at once. Dividing the posterior means
# instead would give a single number with no honest interval. Without truth only the sampler
# diagnostics, the predictive checks and outside evidence protect you; the strongest outside
# evidence is an experiment such as a geo lift test (Modules 7 and 9). We report shares, not
# ROAS, because `x1` and `x2` are not dollars: `contribution_over_spend` on this file returns
# values in the thousands per unit, which is not a return on a dollar.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(6):
    shares = contribution_share(idata_ex)
    checks.columns(shares.reset_index(), ["channel", "mean", "hdi_low", "hdi_high"],
                   name="contribution_share(idata_ex)")
    assert sorted(shares.index) == ["x1", "x2"], (
        f"Expected one row per channel (x1, x2), indexed by channel; got {list(shares.index)}."
    )
    checks.close(shares["mean"].sum(), 1.0, abs=1e-9, name="the sum of the mean shares")
    checks.probability(shares[["hdi_low", "hdi_high"]].to_numpy(), name="the share intervals")
    contrib_ref = idata_ex.posterior["channel_contribution"].sum("date")
    share_ref = contrib_ref / contrib_ref.sum("channel")
    for c in ["x1", "x2"]:
        d = share_ref.sel(channel=c).values.ravel()
        checks.close(shares.loc[c, "mean"], d.mean(), rel=1e-9,
                     name=f"the mean share of {c} (compute the share in every draw, then average)")

# %%
shares.round(3)

# %% [markdown]
# ## Decision · Which channels pay back?
#
# **Number.** For each channel: ROAS (posterior mean and 94% HDI, from Exercise 5) and the
# posterior probability that ROAS exceeds the break-even ROAS, computed draw by draw.
#
# **Rule.** A dollar of sales earns the retailer its gross margin, 30% here, so a channel pays
# back if 0.30 × ROAS > 1: the break-even ROAS is 1 ÷ 0.30 ≈ 3.3. Classify a channel as *pays
# back* if P(ROAS > 3.3) ≥ 0.9, as *does not pay back* if P(ROAS < 3.3) ≥ 0.9, and otherwise as
# *undecided*. The 0.9 is a business choice: how sure we want to be before moving money.

# %%
GROSS_MARGIN = 0.30  # stated by the business
BREAK_EVEN = 1 / GROSS_MARGIN
rows = []
for c in CHANNELS:
    d = roas_draws.sel(channel=c).values.ravel()
    p_above = float((d > BREAK_EVEN).mean())
    verdict = ("pays back" if p_above >= 0.9 else
               "does not pay back" if 1 - p_above >= 0.9 else "undecided")
    rows.append({"channel": c, "roas_mean": d.mean(), "hdi_low": table.loc[c, "hdi_low"],
                 "hdi_high": table.loc[c, "hdi_high"], "P(ROAS > break-even)": p_above,
                 "verdict": verdict})
decision = pd.DataFrame(rows).set_index("channel")
mode = "QUICK run: treat this as a rough answer" if QUICK else "FULL run"
print(f"Break-even ROAS {BREAK_EVEN:.2f} (gross margin {GROSS_MARGIN:.0%}); {mode}.")
print(f"Diagnostics gate: {'passed' if all(gate.values()) else 'NOT passed'}"
      f" ({', '.join(k for k, ok in gate.items() if not ok) or 'all three checks pass'}).")
decision.round(3)

# %% [markdown]
# **Recommendation.** Write one sentence a manager could act on: which channels pay back, which
# do not, which are undecided, how sure you are, and what would change the answer (for an
# undecided channel: a lift test, Module 9).
#
# ```text
# Your sentence: ________________________________________________
# ```
#
# Then compare with the truth, which a real analysis never sees:

# %%
decision.assign(true_roas=[true_roas[c] for c in CHANNELS],
                truly_pays_back=[true_roas[c] > BREAK_EVEN for c in CHANNELS])[
    ["roas_mean", "verdict", "true_roas", "truly_pays_back"]].round(3)

# %% [markdown]
# ## Stretch (optional) · Default priors, or another saturation curve
#
# 1. Refit with PyMC-Marketing's default priors (drop `model_config`) and compare the ROAS
#    intervals with Exercise 5's. Which channels' intervals change most, and why those?
# 2. Replace `LogisticSaturation()` with `HillSaturation()` and compare the posterior predictive
#    fit and the ROAS table.
#
# Each refit takes about as long as the first fit.
#
# <details><summary>Code for the first one</summary>
#
# ```python
# mmm_default = MMM(date_column="date_week", channel_columns=CHANNELS,
#                   control_columns=["price_index", "holiday", "t"], target_column="y",
#                   adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
#                   yearly_seasonality=2)
# mmm_default.fit(X, y, nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS,
#                 target_accept=0.9, random_seed=SEED, progressbar=False)
# roas_default = mmm_default.incrementality.contribution_over_spend(
#     frequency="all_time", start_date=w0, end_date=w1)
# pd.concat({"spend-share priors": roas_table(roas_draws, true_roas),
#            "default priors": roas_table(roas_default, true_roas)}, axis=1).round(2)
# ```
# </details>
