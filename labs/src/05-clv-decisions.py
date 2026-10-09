# ---
# jupyter:
#   mktstats:
#     api_checks: |
#       Technical Expert, 2026-10-09. Checked against the installed source of
#       pymc-marketing 1.2.0, pymc 6.3.2, duckdb 1.3.2 (.venv/lib/python3.13/site-packages):
#       - clv/models/beta_geo.py: model_config keys purchase_covariate_cols and
#         dropout_covariate_cols. With covariates, alpha = alpha_scale * exp(-x'gamma_alpha),
#         a = a_scale * exp(x'gamma_a), b = b_scale * exp(x'gamma_b), all per customer.
#         _extract_predictive_variables applies the same covariates to prediction data, so
#         expected_purchases_new_customer(data=rows with customer_id and covariates, t=) gives
#         FHL 2005 eq. 9 per row, dims (chain, draw, customer_id); expected_purchases(data,
#         future_t=) and expected_probability_alive(data) likewise.
#       - clv/models/gamma_gamma.py: posterior p, q, v; population mean spend p v / (q - 1);
#         expected_customer_spend(data) -> (chain, draw, customer_id).
#       - clv/models/basic.py: thin_fit_result(keep_every).
#       - model_builder.py fit(data, method="mcmc", progressbar=, random_seed=, nuts_sampler=,
#         chains=, draws=, tune=). For models with deterministics it calls
#         pm.compute_deterministics without progressbar, so a "Computing ..." bar prints.
#       - duckdb 1.3.2: CREATE TABLE ... PRIMARY KEY (customer_id, model_version); INSERT OR
#         REPLACE INTO ... SELECT from a registered DataFrame; DESCRIBE returns column_name and
#         column_type; connections close with .close().
# ---

# %% [markdown]
# # Part A · What can we pay to acquire a customer?
#
# Module 4 put a dollar value on customers, with an interval. This lab turns that value into
# decisions: the most we can pay to acquire a customer in each channel, which segments are worth
# more, whether a retention offer pays back, and a scores table that other systems can read.
#
# Terms used throughout:
#
# - **Customer acquisition cost (CAC)**: what marketing spends, on average, to win one new customer
#   through a channel. A **CAC cap** is the most we are willing to pay.
# - **Margin CLV**: customer lifetime value in gross margin, not revenue: spend times the gross
#   margin rate, discounted month by month. Every CLV in this lab is margin.
# - **Acquisition channel**: how the retailer first won the customer (search, social or referral).
#   The channels are the covariates of Module 3.
#
# The data is the synthetic retailer of Module 4 (5,000 customers, calibration year 2024), whose
# true new-customer value per channel is known.
#
# **QUICK.** With QUICK on, the models are fitted on a random 2,000 of the 5,000 customers with
# 2 × 300 draws, instead of everyone with 2 × 500. Intervals are then wider and may not match the
# module page, and a decision near its threshold can flip. That is a lesson about sample size, not
# a bug.

# %%
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pymc_extras.prior import Prior
from pymc_marketing import clv

from mktstats import data, recovery, synth

SEED = 2026                     # the retailer's seed; also picks the QUICK subset of customers
FIT_SEED = 505                  # the seed of every sampler in this lab
CHAINS = 2
DRAWS = 300 if QUICK else 500   # draws per chain, and as many tuning steps
THIN = 2                        # per-customer arrays keep every 2nd draw (300 or 500 in all)
N_CUSTOMERS = 2000 if QUICK else None  # None: all 5,000
INTERVAL = 0.94                 # every interval here is a 94% highest-density interval (HDI)
WEEKS_PER_MONTH = 30.4375 / 7   # PyMC-Marketing's days per month, in weeks
CHANNELS = ["search", "social", "referral"]
COVARIATES = ["channel_social", "channel_referral"]  # search is the reference channel

# %% [markdown]
# ### The assumptions, in one place
#
# Change them here and rerun the cells below; the checkpoints use these defaults.

# %%
MARGIN = 0.30             # gross margin as a share of spend (an assumption, not from data)
MONTHLY_RATE = 0.008      # discount rate, 0.8% a month: about 10% a year, as in the truth
HORIZON_MONTHS = 12       # acquisition decisions look 12 months ahead (the truth uses 52 weeks)
CAMPAIGN_COST = 2.00      # retention offer: $2 per targeted customer
CURRENT_CAC = {"search": 45.0, "social": 32.0, "referral": 55.0}  # assumed for this exercise

# %%
retailer = synth.retailer(seed=SEED)
truth = retailer.truth
tx, customers = retailer.transactions, retailer.customers
rfm = data.retailer_rfm(tx, customers, truth)
if N_CUSTOMERS is not None:
    rng = np.random.default_rng(SEED)
    keep = rng.choice(rfm["customer_id"].to_numpy(), size=N_CUSTOMERS, replace=False)
    rfm = rfm[rfm["customer_id"].isin(keep)].reset_index(drop=True)

# One row per channel, shaped like a customer of that channel: what a new customer looks like.
channel_rows = pd.DataFrame({"customer_id": CHANNELS,
                             "channel_social": [0, 1, 0], "channel_referral": [0, 0, 1]})
new_truth = truth["value_by_channel"]["new_customer"]
print(f"{len(rfm):,} customers in the fit; by channel:"
      f" {rfm['acquisition_channel'].value_counts().reindex(CHANNELS).to_dict()}")
channel_rows

# %% [markdown]
# ## Exercise 1 · New-customer CLV by channel (10 minutes)
#
# A CAC cap is about a customer we have not acquired yet, so it needs the value of a **new**
# customer of each channel, not of the customers we already have. A new customer brings:
#
# value<sub>c</sub> = m · s̄ · (1 + Σ<sub>k=1..H</sub> n<sub>ck</sub> / (1 + d)<sup>k</sup>),
#
# where m is the margin rate, s̄ = p·v/(q − 1) the population mean spend per purchase from the
# Gamma-Gamma posterior (one value per draw), n<sub>ck</sub> the expected **repeat** purchases of a
# new channel-c customer in month k, d the monthly discount rate and H the horizon. The 1 is the
# acquisition purchase itself, in month 0, not discounted: the BG/NBD counts only repeat purchases.
#
# **Predict.** Rank the channels by the 12-month value of a new customer, highest first. Is the top
# channel worth less than 1.5 times, 1.5 to 3 times, or more than 3 times the bottom one?
#
# **Task.** Write `new_customer_clv(model, gg_idata, channel_rows, months, monthly_rate, margin)`
# returning an `xr.DataArray` with dims `(chain, draw, channel)`. Get cumulative repeat purchases
# from `model.expected_purchases_new_customer(data=channel_rows, t=k * WEEKS_PER_MONTH)` for
# k = 0, 1, …, months (dims `(chain, draw, customer_id)`, where `customer_id` holds the channel
# names), take differences to get purchases in each month, and rename `customer_id` to `channel`.
# Checkpoint 1 first tests your function on a toy model, before any fit.

# %%
class ToyModel:
    """A stand-in for a fitted transaction model, for checkpoint 1: a new customer of channel c
    makes rate[c] repeat purchases per week, in both of its two draws (one chain)."""

    def __init__(self, rate):
        self.rate = rate

    def expected_purchases_new_customer(self, data=None, *, t=None):
        per_week = np.array([self.rate[c] for c in data["customer_id"]])
        values = np.broadcast_to(per_week * t, (1, 2, len(per_week)))
        return xr.DataArray(values, dims=("chain", "draw", "customer_id"),
                            coords={"customer_id": data["customer_id"].to_numpy()})


toy_model = ToyModel({"search": 0.10, "social": 0.05, "referral": 0.20})
# Population mean spend p v / (q - 1): $10 in the first draw, $20 in the second.
toy_gg = SimpleNamespace(posterior=xr.Dataset({"p": (("chain", "draw"), [[2.0, 4.0]]),
                                               "q": (("chain", "draw"), [[3.0, 3.0]]),
                                               "v": (("chain", "draw"), [[10.0, 10.0]])}))

# %% tags=["exercise"]
def new_customer_clv(model, gg_idata, channel_rows, months, monthly_rate, margin):
    # TODO 1: spend per draw; repeat purchases in each month; discount; add the first purchase
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def new_customer_clv(model, gg_idata, channel_rows, months, monthly_rate, margin):
    post = gg_idata.posterior
    spend = post["p"] * post["v"] / (post["q"] - 1)  # population mean spend, per draw
    steps = np.arange(0, months + 1)
    cumulative = xr.concat(
        [model.expected_purchases_new_customer(data=channel_rows, t=k * WEEKS_PER_MONTH)
         for k in steps],
        dim=xr.DataArray(steps, dims="month", name="month"),
    )
    in_month = cumulative.diff("month")  # repeat purchases in month k = 1..months
    discount = (1 + monthly_rate) ** -in_month["month"]
    value = margin * spend * (1 + (in_month * discount).sum("month"))  # 1 = acquisition purchase
    return value.rename(customer_id="channel").transpose("chain", "draw", "channel")


# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    toy = new_customer_clv(toy_model, toy_gg, channel_rows, 3, 0.01, 0.5)
    assert isinstance(toy, xr.DataArray) and set(toy.dims) == {"chain", "draw", "channel"}, (
        f"Checkpoint 1: expected an xr.DataArray with dims (chain, draw, channel), got"
        f" {getattr(toy, 'dims', type(toy).__name__)}. Rename customer_id to channel. To move on,"
        " run workshop.use_reference(1)."
    )
    spend_toy = np.array([10.0, 20.0])
    discounted_months = sum(1.01 ** -k for k in range(1, 4))
    for c, rate in toy_model.rate.items():
        want = 0.5 * spend_toy * (1 + rate * WEEKS_PER_MONTH * discounted_months)
        got = toy.sel(channel=c).values.ravel()
        if np.allclose(got, want - 0.5 * spend_toy):
            cause = ("It is short by exactly the margin on one purchase: add the acquisition"
                     " purchase (month 0, not discounted).")
        elif np.allclose(got, 0.5 * spend_toy * (1 + rate * WEEKS_PER_MONTH * 3)):
            cause = "It is not discounted: divide month k by (1 + monthly_rate) ** k."
        else:
            cause = ("Check that you difference the cumulative purchases (months 0 to `months`)"
                     " and multiply by the margin and the spend p * v / (q - 1) of each draw.")
        assert np.allclose(got, want, rtol=1e-9), (
            f"Checkpoint 1: on the toy model, {c} should be worth {np.round(want, 4).tolist()} in"
            f" its two draws; your function gives {np.round(got, 4).tolist()}. {cause} To move"
            " on, run workshop.use_reference(1)."
        )

# %% [markdown]
# ### Run · Fit the transaction and spend models
#
# The transaction model is a BG/NBD with the channel indicators on both the purchase and the
# dropout process (Module 3's covariates). The retailer's data come from a Pareto/NBD, whose MCMC
# fit is too slow for a lab (231 s for 2 × 300 draws on CDNOW on Colab, DECISIONS.md spike S2), and
# the CAC caps need a posterior, not a point estimate. The BG/NBD approximates the generator, which
# is why the next check allows a wider gap to the truth than a recovery check would. The spend
# model is Module 4's Gamma-Gamma with the same weak priors. Both draw 2 × 500 (2 × 300 with
# QUICK); the BG/NBD is the slowest cell of the lab (about 30 s on a laptop with FULL settings,
# longer on Colab): read Exercise 2 while it runs.

# %%
t0 = time.time()
bgnbd = clv.BetaGeoModel(model_config={"purchase_covariate_cols": COVARIATES,
                                       "dropout_covariate_cols": COVARIATES})
with warnings.catch_warnings():
    # pymc-marketing 1.2.0 passes PyMC an argument PyMC has deprecated (merge_dataset): the
    # FutureWarning is about the library, not your code. It also prints a "Computing ..." bar.
    warnings.simplefilter("ignore", FutureWarning)
    bgnbd.fit(data=rfm[["customer_id", "frequency", "recency", "T", *COVARIATES]], method="mcmc",
              nuts_sampler="nutpie", chains=CHAINS, draws=DRAWS, tune=DRAWS,
              random_seed=FIT_SEED, progressbar=False)
print(f"BG/NBD with channels: {CHAINS} chains x {DRAWS} draws on {len(rfm):,} customers in"
      f" {time.time() - t0:.1f} s, {int(bgnbd.idata.sample_stats['diverging'].sum())} divergent"
      " transitions.")

# %%
gg_priors = {name: Prior(dist, **kw) for name, (dist, kw) in recovery.WEAK_GAMMA_GAMMA_PRIORS.items()}
t0 = time.time()
gg = clv.GammaGammaModel(model_config=gg_priors)
gg.fit(data=rfm.loc[rfm["frequency"] > 0, ["customer_id", "frequency", "monetary_value"]],
       method="mcmc", nuts_sampler="nutpie", chains=CHAINS, draws=DRAWS, tune=DRAWS,
       random_seed=FIT_SEED, progressbar=False)
print(f"Gamma-Gamma: {CHAINS} chains x {DRAWS} draws in {time.time() - t0:.1f} s,"
      f" {int(gg.idata.sample_stats['diverging'].sum())} divergent transitions.")

# %% [markdown]
# Now the same function on the fitted models, against the truth: the true new-customer value per
# channel, `truth["value_by_channel"]["new_customer"][c]["discounted_clv_including_first_purchase"]`
# times the margin. The truth is revenue over 52 weeks discounted continuously at 10% a year; this
# lab uses 12 months of 30.4375 days discounted at the end of each month at 0.8%. That convention
# gap is well under 1%. The tolerance is 15%: the reference run's largest gap is 7% (QUICK and
# FULL), the rest is the BG/NBD's misspecification and sampling noise.

# %% tags=["checkpoint"]
with workshop.checkpoint(1, label="1 · against the truth"):
    new_clv = new_customer_clv(bgnbd, gg.idata, channel_rows, HORIZON_MONTHS, MONTHLY_RATE, MARGIN)
    true_value = {c: MARGIN * new_truth[c]["discounted_clv_including_first_purchase"]
                  for c in CHANNELS}
    for c in CHANNELS:
        got = float(new_clv.sel(channel=c).mean())
        gap = got / true_value[c] - 1
        assert abs(gap) <= 0.15, (
            f"Checkpoint 1: a new {c} customer is worth ${got:.2f} of margin in your posterior"
            f" mean; the truth is ${true_value[c]:.2f} ({gap:+.0%}, allowed 15%). If the toy"
            " check passed, look at the units: t in weeks (k * WEEKS_PER_MONTH), the margin,"
            " and months 1 to 12. To move on, run workshop.use_reference(1)."
        )
    best_true = max(true_value, key=true_value.get)
    best_est = str(new_clv.mean(("chain", "draw")).idxmax("channel").item())
    assert best_est == best_true, (
        f"Checkpoint 1: the most valuable channel is {best_true} in truth but {best_est} in your"
        " posterior means. To move on, run workshop.use_reference(1)."
    )


def summary_row(x):
    lo, hi = checks.interval(x, INTERVAL, "hdi")
    return {"mean": float(x.mean()), "hdi_low": lo, "hdi_high": hi}


pd.DataFrame({c: {**summary_row(new_clv.sel(channel=c)), "truth": true_value[c]}
              for c in CHANNELS}).T.round(2)

# %% [markdown]
# **Explain.** Compare with your ranking. Which part of the formula did the covariates change
# between channels, and what would the values be if you had forgotten the acquisition purchase?
#
# <details><summary>Why this solution works</summary>
#
# The BG/NBD and Pareto/NBD count **repeat** purchases, so a new customer's first purchase must be
# added separately: here it is m · s̄, about $9 of margin, a large share of a social customer's
# value. The channel covariates scale the purchase rate (α) and the dropout parameters (a, b), so a
# referral customer buys more often and stays longer than a social one; the spend per purchase is
# the same population mean for every channel, because the Gamma-Gamma model here has no
# covariates. Each draw of the purchase model is paired with a draw of the spend model, so the
# interval carries both uncertainties.
# </details>

# %% [markdown]
# ## Exercise 2 · CAC caps under a stated risk rule (6 minutes)
#
# A cap at the posterior mean pays, on average, what a new customer is worth. A more cautious
# **risk rule** sets the cap at a low quantile of the posterior: with the 20th percentile, the
# expected value of a new customer is above the cap with posterior probability 0.8.
#
# **Predict.** Will a cap at the 20th percentile be less than 5%, 5 to 15%, or more than 15% below
# a cap at the mean?
#
# **Task.** Write `cac_cap(clv_draws, q=0.2)` returning a `pd.Series` indexed by channel: the `q`
# quantile of each channel's posterior draws, or the posterior mean when `q` is `None`.

# %% tags=["exercise"]
def cac_cap(clv_draws, q=0.2):
    # TODO 2: per channel, the q quantile over (chain, draw), or the mean when q is None
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def cac_cap(clv_draws, q=0.2):
    if q is None:
        caps = clv_draws.mean(("chain", "draw"))
    else:
        caps = clv_draws.quantile(q, dim=("chain", "draw"))
    return pd.Series(caps.values, index=[str(c) for c in caps["channel"].values], name="cac_cap")


# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    cap_mean = cac_cap(new_clv, q=None)
    cap_20 = cac_cap(new_clv, q=0.2)
    for caps in (cap_mean, cap_20):
        assert isinstance(caps, pd.Series) and sorted(caps.index) == sorted(CHANNELS), (
            f"Checkpoint 2: expected a pd.Series indexed by {CHANNELS}; got"
            f" {type(caps).__name__} with index {list(getattr(caps, 'index', []))}. To move on,"
            " run workshop.use_reference(2)."
        )
    flat_draws = {c: new_clv.sel(channel=c).values.ravel() for c in CHANNELS}
    for c in CHANNELS:
        checks.close(cap_mean[c], flat_draws[c].mean(), rel=1e-9,
                     name=f"the mean-rule cap for {c}")
        checks.close(cap_20[c], np.quantile(flat_draws[c], 0.2), rel=0.005,
                     name=f"the 20th-percentile cap for {c}")
        assert cap_20[c] < cap_mean[c], (
            f"Checkpoint 2: for {c} the 20th-percentile cap ({cap_20[c]:.2f}) should be below the"
            f" mean ({cap_mean[c]:.2f}). Did you take the quantile over draws, not channels? To"
            " move on, run workshop.use_reference(2)."
        )
        checks.monotone([cac_cap(new_clv, q=x)[c] for x in (0.1, 0.2, 0.5)], increasing=True,
                        name=f"the {c} cap at q = 0.1, 0.2, 0.5")

caps = pd.DataFrame({"current CAC": CURRENT_CAC, "cap: mean": cap_mean, "cap: 20th pct": cap_20})
caps["20th pct below mean"] = (1 - caps["cap: 20th pct"] / caps["cap: mean"]).map("{:.1%}".format)
caps.round(2)

# %% [markdown]
# **Explain.** Compare with your prediction. The interval here is about the *expected* value of a
# new customer, not about how much individual customers differ. Why is it narrow, and what does a
# cap set from it assume about the customers a channel brings in?
#
# <details><summary>Why this solution works</summary>
#
# The posterior is about the population parameters, which thousands of customers pin down, so the
# expected value of a new customer is known to within a few percent; individual customers differ
# far more. A cap from this posterior is a cap on the **average** cost per acquired customer. It
# also assumes that each channel's acquisitions are **incremental**: that those customers would
# not have arrived anyway. Nothing here tests that; Day 3 does. A lower quantile protects against
# estimation error, not against paying for customers the channel did not cause.
# </details>

# %% [markdown]
# # Part B · Segments and retention
#
# Next, existing customers. The cell below computes, from every 2nd posterior draw, each customer's
# **P(alive)** (the probability they have not yet churned, given their history), expected purchases
# in each of the next 12 months, expected spend, and 12-month margin CLV, with Module 4's
# `discounted_clv`. One feature of the BG/NBD to keep in mind: it lets a customer leave only right
# after a repeat purchase, so a customer with no repeat purchase gets P(alive) = 1, even though
# their low purchase rate gives them few expected purchases. Segments combine the acquisition channel with how many repeat purchases the
# customer made in 2024 (0, 1 to 3, 4 or more). Tenure would not separate anyone here: the
# retailer acquired all its customers in the first half of 2024, so at the end of the year every
# customer has been with it for 6 to 12 months.

# %%
def discounted_clv(purchases_by_month, spend, monthly_rate):
    """Module 4, Exercise 3: sum over months of spend x purchases / (1 + rate) ** month."""
    discount = (1 + monthly_rate) ** -purchases_by_month["month"]
    return (purchases_by_month * discount).sum("month") * spend


t0 = time.time()
bgnbd_thin, gg_thin = bgnbd.thin_fit_result(THIN), gg.thin_fit_result(THIN)
rfm_scores = rfm[["customer_id", "frequency", "recency", "T", "monetary_value", *COVARIATES]]
p_alive = bgnbd_thin.expected_probability_alive(data=rfm_scores)
months = np.arange(1, HORIZON_MONTHS + 1)
cumulative = xr.concat(
    [bgnbd_thin.expected_purchases(data=rfm_scores, future_t=k * WEEKS_PER_MONTH) for k in months],
    dim=xr.DataArray(months, dims="month", name="month"),
)
purchases_by_month = (cumulative - cumulative.shift(month=1, fill_value=0.0)).transpose(
    "chain", "draw", "customer_id", "month")
spend = gg_thin.expected_customer_spend(data=rfm_scores)
clv_draws = MARGIN * discounted_clv(purchases_by_month, spend, MONTHLY_RATE)
clv_draws = clv_draws.transpose("chain", "draw", "customer_id")

band = pd.cut(rfm["frequency"], [-1, 0, 3, np.inf], labels=["0 repeat", "1-3 repeat", "4+ repeat"])
segments = pd.Series((rfm["acquisition_channel"] + " · " + band.astype(str)).to_numpy(),
                     index=rfm["customer_id"].to_numpy(), name="segment")
base_total = clv_draws.sum("customer_id")
base_lo, base_hi = checks.interval(base_total, INTERVAL, "hdi")
print(f"clv_draws {clv_draws.dims} {clv_draws.shape}, computed in {time.time() - t0:.1f} s")
print(f"12-month margin CLV of the {len(rfm):,} customers: ${float(base_total.mean()):,.0f}"
      f" (94% HDI ${base_lo:,.0f} to ${base_hi:,.0f}).")
segments.value_counts().sort_index()

# %% [markdown]
# ## Exercise 3 · Segment value and real differences (8 minutes)
#
# **Predict.** Two segments have overlapping 94% intervals for mean CLV per customer. Can the
# posterior probability that one is more valuable than the other still be above 0.9: yes or no?
#
# **Task.** Write two functions. `segment_value(clv_draws, segments)` returns a DataFrame with one
# row per segment and columns `segment`, `customers`, `total_mean`, `total_low`, `total_high`
# (the segment's total CLV: posterior mean and 94% HDI) and `per_customer_mean`,
# `per_customer_low`, `per_customer_high` (the same for the mean CLV per customer).
# `prob_greater(clv_draws, segments, a, b)` returns the posterior probability that segment `a`'s
# mean CLV per customer exceeds segment `b`'s: compare the two means draw by draw. `segments` is in
# the same customer order as `clv_draws`; `(segments == name).to_numpy()` selects a segment's
# customers with `clv_draws.isel(customer_id=mask)`.

# %% tags=["exercise"]
def segment_value(clv_draws, segments):
    # TODO 3: per segment, total and per-customer mean CLV, each with its 94% HDI
    raise NotImplementedError("TODO 3")


def prob_greater(clv_draws, segments, a, b):
    # TODO 3: share of draws in which segment a's mean CLV per customer exceeds segment b's
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def segment_value(clv_draws, segments):
    rows = []
    for name in sorted(segments.unique()):
        part = clv_draws.isel(customer_id=(segments == name).to_numpy())
        total, average = part.sum("customer_id"), part.mean("customer_id")
        t_lo, t_hi = checks.interval(total, 0.94, "hdi")
        a_lo, a_hi = checks.interval(average, 0.94, "hdi")
        rows.append({"segment": name, "customers": part.sizes["customer_id"],
                     "total_mean": float(total.mean()), "total_low": t_lo, "total_high": t_hi,
                     "per_customer_mean": float(average.mean()), "per_customer_low": a_lo,
                     "per_customer_high": a_hi})
    return pd.DataFrame(rows)


@workshop.solution(3)
def prob_greater(clv_draws, segments, a, b):
    mean_a = clv_draws.isel(customer_id=(segments == a).to_numpy()).mean("customer_id")
    mean_b = clv_draws.isel(customer_id=(segments == b).to_numpy()).mean("customer_id")
    return float((mean_a > mean_b).mean())


# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    seg_table = segment_value(clv_draws, segments)
    checks.columns(seg_table, ["segment", "customers", "total_mean", "total_low", "total_high",
                               "per_customer_mean", "per_customer_low", "per_customer_high"],
                   name="segment_value(...)")
    assert int(seg_table["customers"].sum()) == len(segments) and seg_table["segment"].is_unique, (
        f"Checkpoint 3: the segments should cover all {len(segments):,} customers once; yours"
        f" cover {int(seg_table['customers'].sum()):,}. To move on, run workshop.use_reference(3)."
    )
    checks.close(seg_table["total_mean"].sum(), float(base_total.mean()), rel=1e-6,
                 name="the sum of the segment totals (it must equal the base total)")
    checks.close(seg_table["per_customer_mean"] * seg_table["customers"], seg_table["total_mean"],
                 rel=1e-6, name="per-customer mean x customers (it must equal the segment total)")
    inside = (seg_table["total_low"] <= seg_table["total_mean"]) & (
        seg_table["total_mean"] <= seg_table["total_high"])
    assert inside.all(), (
        "Checkpoint 3: a segment's posterior mean lies outside its HDI: check the column order."
        " To move on, run workshop.use_reference(3)."
    )
    a, b = "search · 4+ repeat", "search · 1-3 repeat"
    p_ab, p_ba = prob_greater(clv_draws, segments, a, b), prob_greater(clv_draws, segments, b, a)
    checks.probability([p_ab, p_ba], name="prob_greater(...)")
    checks.close(p_ab + p_ba, 1.0, abs=1e-9, name="P(a > b) + P(b > a)")
    mean_of = {s: clv_draws.isel(customer_id=(segments == s).to_numpy()).mean("customer_id")
               for s in (a, b)}
    checks.close(p_ab, float((mean_of[a] > mean_of[b]).mean()), abs=1e-9,
                 name=f"P({a} > {b}), compared draw by draw")
seg_table.round(1)

# %% [markdown]
# The table below lists every pair of segments whose 94% intervals for mean CLV per customer
# overlap, with the posterior probability that the first is worth more. Look for pairs with a
# probability above 0.9.

# %%
pairs = []
for i, r1 in seg_table.iterrows():
    for _, r2 in seg_table.iloc[i + 1:].iterrows():
        overlap = r1["per_customer_low"] <= r2["per_customer_high"] and (
            r2["per_customer_low"] <= r1["per_customer_high"])
        if overlap:
            hi_seg, lo_seg = (r1, r2) if r1["per_customer_mean"] >= r2["per_customer_mean"] else (r2, r1)
            pairs.append({"higher": hi_seg["segment"], "lower": lo_seg["segment"],
                          "P(higher > lower)": prob_greater(clv_draws, segments,
                                                            hi_seg["segment"], lo_seg["segment"])})
pd.DataFrame(pairs).sort_values("P(higher > lower)", ascending=False).round(3)

# %% [markdown]
# **Explain.** Compare with your yes or no. Why can two overlapping intervals still give a
# probability above 0.9? Think about what the two segments' draws share.
#
# <details><summary>Why this solution works</summary>
#
# Every customer's CLV is computed from the same posterior draws of the population parameters, so
# when a draw has a high purchase rate, every segment's value is high in that draw. The two
# segments' means move together, and their **difference** varies much less than either mean. The
# overlap of two separate intervals ignores that shared movement. Compare segments with the
# posterior of their difference (or the probability that one exceeds the other), never by whether
# their intervals overlap.
# </details>

# %% [markdown]
# ## Exercise 4 · Retention-campaign break-even (8 minutes)
#
# A retention offer costs $2 per targeted customer. It pays back if it raises the customer's
# purchases over the next 12 months by enough margin to cover the $2. The **break-even lift** is
# that relative increase:
#
# break-even lift<sub>i</sub> = cost / (margin per purchase<sub>i</sub> × expected
# purchases<sub>i</sub>),
#
# where margin per purchase<sub>i</sub> = m × expected spend<sub>i</sub> and expected
# purchases<sub>i</sub> are over the next 12 months (posterior means; undiscounted).
#
# **Predict.** Will the break-even lift be smaller for customers with high P(alive) or with low
# P(alive)?
#
# **Task.** Write `breakeven_lift(cost, margin_per_purchase, expected_purchases)` returning a numpy
# array, and `campaign_table(lift, segments)` returning a DataFrame with one row per segment and
# columns `segment`, `customers`, `median_lift`, and `below_5pct`, `below_10pct`, `below_20pct`
# (the number of customers whose break-even lift is below 5%, 10% and 20%).

# %%
margin_per_purchase = (MARGIN * spend.mean(("chain", "draw"))).to_numpy()
expected_purchases = purchases_by_month.sum("month").mean(("chain", "draw")).to_numpy()
p_alive_mean = p_alive.mean(("chain", "draw")).to_numpy()

# %% tags=["exercise"]
def breakeven_lift(cost, margin_per_purchase, expected_purchases):
    # TODO 4: cost / (margin per purchase x expected purchases), as a numpy array
    raise NotImplementedError("TODO 4")


def campaign_table(lift, segments):
    # TODO 4: per segment: customers, median lift, counts below 5%, 10% and 20%
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def breakeven_lift(cost, margin_per_purchase, expected_purchases):
    return cost / (np.asarray(margin_per_purchase, float) * np.asarray(expected_purchases, float))


@workshop.solution(4)
def campaign_table(lift, segments):
    df = pd.DataFrame({"segment": np.asarray(segments), "lift": np.asarray(lift, float)})
    return (
        df.groupby("segment")["lift"]
        .agg(customers="size", median_lift="median",
             below_5pct=lambda x: int((x < 0.05).sum()),
             below_10pct=lambda x: int((x < 0.10).sum()),
             below_20pct=lambda x: int((x < 0.20).sum()))
        .reset_index()
    )


# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    hand = breakeven_lift(2.0, np.array([3.0, 9.0, 6.0]), np.array([2.0, 4.0, 0.5]))
    want = np.array([2 / 6, 2 / 36, 2 / 3])
    assert np.allclose(np.asarray(hand, float), want, rtol=1e-12), (
        f"Checkpoint 4: for cost $2, margins per purchase $3, $9, $6 and expected purchases 2, 4,"
        f" 0.5, the break-even lifts are {np.round(want, 4).tolist()}; yours are"
        f" {np.round(np.asarray(hand, float), 4).tolist()}. Divide the cost by margin per purchase"
        " times expected purchases, and return a fraction (0.05 means 5%), not a percentage. To"
        " move on, run workshop.use_reference(4)."
    )
    checks.monotone(breakeven_lift(2.0, 9.0, np.array([0.5, 1.0, 2.0, 4.0, 8.0])),
                    increasing=False, strict=True,
                    name="Break-even lift as expected purchases grow")
    lift = np.asarray(breakeven_lift(CAMPAIGN_COST, margin_per_purchase, expected_purchases), float)
    table = campaign_table(lift, segments)
    checks.columns(table, ["segment", "customers", "median_lift", "below_5pct", "below_10pct",
                           "below_20pct"], name="campaign_table(...)")
    seg = segments.to_numpy()
    for row in table.itertuples():
        mine = lift[seg == row.segment]
        want_row = (len(mine), float(np.median(mine)), int((mine < 0.05).sum()),
                    int((mine < 0.10).sum()), int((mine < 0.20).sum()))
        got_row = (int(row.customers), float(row.median_lift), int(row.below_5pct),
                   int(row.below_10pct), int(row.below_20pct))
        same = (got_row[0] == want_row[0] and got_row[2:] == want_row[2:]
                and abs(got_row[1] - want_row[1]) <= 1e-9 * want_row[1])
        assert same, (
            f"Checkpoint 4: for {row.segment} expected (customers, median, below 5%, 10%, 20%) ="
            f" {tuple(round(x, 4) for x in want_row)}; yours are"
            f" {tuple(round(x, 4) for x in got_row)}. Lifts are fractions: 5% is 0.05. To move"
            " on, run workshop.use_reference(4)."
        )
table.round(3)

# %% [markdown]
# Among repeat buyers (dots), look at where the customers with P(alive) near 1 sit compared with
# those near 0. The crosses at P(alive) = 1 are customers with no repeat purchase, whom the BG/NBD
# always counts as alive. The dashed lines are lifts of 5%, 10% and 20%.

# %%
repeat_buyer = rfm["frequency"].to_numpy() > 0
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.scatter(p_alive_mean[repeat_buyer], lift[repeat_buyer], s=4, alpha=0.4, marker="o",
           label="at least one repeat purchase")
ax.scatter(p_alive_mean[~repeat_buyer], lift[~repeat_buyer], s=12, alpha=0.4, marker="x",
           label="no repeat purchase")
for level in (0.05, 0.10, 0.20):
    ax.axhline(level, linestyle="--", color="grey", linewidth=1)
ax.set_yscale("log")
ax.set_xlabel("P(alive) at the end of 2024 (posterior mean)")
ax.set_ylabel("Break-even lift (log scale; 0.1 = 10%)")
ax.set_title("Lift a $2 offer needs to pay back, by customer")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Explain.** Compare with your prediction. The customers whose break-even lift is smallest are
# the ones the model expects to buy most. Why does that make them a risky target for the offer?
#
# <details><summary>Why this solution works</summary>
#
# The same $2 is spread over more expected margin when a customer is likely alive and buys often,
# so the lift needed to recover it is small; for a customer who has probably left, the expected
# margin is near zero and the needed lift is enormous. P(alive) alone is not the whole story: the
# customers with no repeat purchase have P(alive) = 1 in the BG/NBD but buy rarely, so their
# median break-even lift is 40 to 50% in the reference runs: expected purchases are what matter.
# And likely-alive frequent buyers are often the customers who would buy anyway: what matters is
# whether the offer **causes** extra purchases, which this model cannot tell. That is an uplift question (Module 11), and Day 3 shows how to
# measure a campaign's effect with an experiment.
# </details>

# %% [markdown]
# # Part C · Production
#
# A scoring job writes its results where other systems can read them. Here the "warehouse" is a
# DuckDB file, `warehouse.duckdb`, in the notebook's working directory. On Colab that directory
# disappears with the runtime, and the file with it. The cell below builds the scores (P(alive) and
# 12-month margin CLV with its 94% HDI, per customer) and opens an empty warehouse: rerunning it
# starts from scratch.

# %%
def hdi_by_customer(draws, prob=0.94):
    """Per-customer HDI of draws with dims (chain, draw, customer_id), computed as
    checks.interval does: the shortest window holding `prob` of the sorted draws."""
    x = np.sort(draws.stack(sample=("chain", "draw")).transpose("customer_id", "sample").to_numpy(),
                axis=1)
    n = x.shape[1]
    k = max(1, min(n - 1, int(np.ceil(prob * n)) - 1))
    i = (x[:, k:] - x[:, : n - k]).argmin(axis=1)
    rows = np.arange(len(x))
    return x[rows, i], x[rows, i + k]


clv_low, clv_high = hdi_by_customer(clv_draws, INTERVAL)
scores = pd.DataFrame({
    "customer_id": rfm["customer_id"].astype(str).to_numpy(),
    "p_alive": p_alive_mean,
    "clv_mean": clv_draws.mean(("chain", "draw")).to_numpy(),
    "clv_hdi_low": clv_low,
    "clv_hdi_high": clv_high,
    "segment": segments.to_numpy(),
})
MODEL_VERSION = "bgnbd-channels+gamma-gamma-v1" + ("-quick" if QUICK else "")
SCORED_AT = truth["calibration_end"]  # the date the scores are as of: reruns give the same rows

WAREHOUSE = Path("warehouse.duckdb")
try:
    con.close()  # a connection from an earlier run of this cell
except NameError:
    pass
for path in (WAREHOUSE, Path(f"{WAREHOUSE}.wal")):
    path.unlink(missing_ok=True)
con = duckdb.connect(str(WAREHOUSE))
print(f"{len(scores):,} scores, model version {MODEL_VERSION}, as of {SCORED_AT};"
      f" warehouse at {WAREHOUSE.resolve()}")
scores.head()

# %% [markdown]
# ## Exercise 5 · Write scores to the warehouse (8 minutes)
#
# **Predict.** If you run the scoring job twice for the same model version, how many rows should
# the table have per customer: one or two?
#
# **Task.** Write `write_scores(con, scores, model_version, scored_at)` that creates the table
# `customer_scores` if it does not exist, with exactly these columns and a primary key on
# `(customer_id, model_version)`:
#
# ```
# customer_id VARCHAR, p_alive DOUBLE, clv_mean DOUBLE, clv_hdi_low DOUBLE,
# clv_hdi_high DOUBLE, segment VARCHAR, model_version VARCHAR, scored_at DATE
# ```
#
# then writes `scores` with `INSERT OR REPLACE` (so a rerun replaces rows instead of adding them),
# and returns the number of rows for `model_version`. `con.register("new_scores", df)` makes a
# DataFrame visible to SQL; `CAST(x AS DATE)` turns `"2024-12-29"` into a date.

# %% tags=["exercise"]
def write_scores(con, scores, model_version, scored_at):
    # TODO 5: CREATE TABLE IF NOT EXISTS ... PRIMARY KEY; INSERT OR REPLACE; return the count
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def write_scores(con, scores, model_version, scored_at):
    con.execute("""
        CREATE TABLE IF NOT EXISTS customer_scores (
            customer_id VARCHAR, p_alive DOUBLE, clv_mean DOUBLE, clv_hdi_low DOUBLE,
            clv_hdi_high DOUBLE, segment VARCHAR, model_version VARCHAR, scored_at DATE,
            PRIMARY KEY (customer_id, model_version)
        )
    """)
    con.register("new_scores", scores)
    con.execute("""
        INSERT OR REPLACE INTO customer_scores
            (customer_id, p_alive, clv_mean, clv_hdi_low, clv_hdi_high, segment,
             model_version, scored_at)
        SELECT customer_id, p_alive, clv_mean, clv_hdi_low, clv_hdi_high, segment,
               ?, CAST(? AS DATE)
        FROM new_scores
    """, [model_version, scored_at])
    con.unregister("new_scores")
    return con.execute("SELECT COUNT(*) FROM customer_scores WHERE model_version = ?",
                       [model_version]).fetchone()[0]


# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    n_first = write_scores(con, scores, MODEL_VERSION, SCORED_AT)
    n_again = write_scores(con, scores, MODEL_VERSION, SCORED_AT)
    schema = con.execute("DESCRIBE customer_scores").df()
    got_schema = dict(zip(schema["column_name"], schema["column_type"], strict=True))
    want_schema = {"customer_id": "VARCHAR", "p_alive": "DOUBLE", "clv_mean": "DOUBLE",
                   "clv_hdi_low": "DOUBLE", "clv_hdi_high": "DOUBLE", "segment": "VARCHAR",
                   "model_version": "VARCHAR", "scored_at": "DATE"}
    assert got_schema == want_schema, (
        f"Checkpoint 5: customer_scores has columns {got_schema}; expected exactly {want_schema}."
        " Fix the CREATE TABLE, then rerun the cell that opens the warehouse (it starts an empty"
        " one) and this checkpoint. To move on, run workshop.use_reference(5)."
    )
    rows_in_table = con.execute("SELECT COUNT(*) FROM customer_scores").fetchone()[0]
    assert n_first == n_again == rows_in_table == len(scores), (
        f"Checkpoint 5: after writing {len(scores):,} scores twice, the table has"
        f" {rows_in_table:,} rows and your function returned {n_first:,} then {n_again:,}; all"
        f" four should be {len(scores):,}. Without a primary key and INSERT OR REPLACE, a rerun"
        " duplicates rows. To move on, run workshop.use_reference(5)."
    )
    bad = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE customer_id IS NULL OR p_alive IS NULL OR clv_mean IS NULL
                             OR clv_hdi_low IS NULL OR clv_hdi_high IS NULL OR segment IS NULL
                             OR model_version IS NULL OR scored_at IS NULL) AS nulls,
            COUNT(*) FILTER (WHERE p_alive < 0 OR p_alive > 1) AS bad_p_alive,
            COUNT(*) FILTER (WHERE NOT (clv_hdi_low <= clv_mean AND clv_mean <= clv_hdi_high))
                AS bad_interval,
            COUNT(*) FILTER (WHERE scored_at <> CAST(? AS DATE) OR model_version <> ?) AS bad_tags
        FROM customer_scores
    """, [SCORED_AT, MODEL_VERSION]).df().iloc[0].to_dict()
    assert all(v == 0 for v in bad.values()), (
        f"Checkpoint 5: rows that break the table's rules: {bad}. Check that every column is"
        " written from `scores`, the model version and the date. To move on, run"
        " workshop.use_reference(5)."
    )
con.execute("SELECT segment, COUNT(*) AS customers, ROUND(AVG(p_alive), 3) AS mean_p_alive,"
            " ROUND(SUM(clv_mean), 0) AS clv_total FROM customer_scores"
            " GROUP BY segment ORDER BY segment").df()

# %% [markdown]
# **Explain.** Compare with your prediction. Suppose the CRM team reads this table next month and
# a number looks odd. Which columns let them find out which model produced it, as of when, and how
# sure it was?
#
# <details><summary>Why this solution works</summary>
#
# The primary key makes a customer's score unique per model version, and `INSERT OR REPLACE`
# turns a rerun into an update rather than a duplicate, so the job can be run again safely
# (idempotent). Storing the interval next to the mean keeps the uncertainty with the number, the
# model version says which model and settings produced it, and the as-of date says which data it
# describes. Together they make a scoring job reproducible and auditable. A new model version adds
# new rows next to the old ones, so the two can be compared before switching.
# </details>

# %% [markdown]
# ### Run · The numbers for the CFO memo
#
# The cell below collects the numbers from this lab into `memo_numbers` and fills them into a
# one-page memo template. You do not write the memo here: the Day 2 clinic starts with 15 minutes of
# writing it from this output. Copy the printed template into your notes.

# %%
in_warehouse = con.execute(
    "SELECT COUNT(*), SUM(clv_mean) FROM customer_scores WHERE model_version = ?",
    [MODEL_VERSION]).fetchone()
p_beats_cac = {c: float((new_clv.sel(channel=c) > CURRENT_CAC[c]).mean()) for c in CHANNELS}
memo_numbers = {
    "base_clv": (float(base_total.mean()), base_lo, base_hi),
    "cac_caps": {c: {"current_cac": CURRENT_CAC[c], "cap_mean": float(cap_mean[c]),
                     "cap_20th_pct": float(cap_20[c]), "p_value_above_cac": p_beats_cac[c]}
                 for c in CHANNELS},
    "retention": {"cost_per_customer": CAMPAIGN_COST,
                  "customers": int(table["customers"].sum()),
                  "below_10pct": int(table["below_10pct"].sum()),
                  "median_lift": float(np.median(lift))},
    "assumptions": {"margin": MARGIN, "monthly_rate": MONTHLY_RATE, "horizon_months": HORIZON_MONTHS,
                    "interval": "94% HDI", "mode": "QUICK" if QUICK else "FULL",
                    "model_version": MODEL_VERSION, "scored_at": SCORED_AT},
}
b = memo_numbers["base_clv"]
caps_lines = "\n".join(
    f"   - {c}: current CAC ${v['current_cac']:.0f}; cap ${v['cap_mean']:.2f} at the mean,"
    f" ${v['cap_20th_pct']:.2f} at the 20th percentile; P(value > current CAC) = "
    f"{v['p_value_above_cac']:.2f}"
    for c, v in memo_numbers["cac_caps"].items())
r = memo_numbers["retention"]
print(f"""MEMO (template: write the headline and the recommendation in the clinic)
To: CFO   From: Customer analytics   Re: acquisition caps and the retention offer

Headline: ______________________________________________ (one sentence: the decision, in dollars)

1. What our customers are worth. {in_warehouse[0]:,} existing customers: 12-month margin CLV
   ${b[0]:,.0f} (94% HDI ${b[1]:,.0f} to ${b[2]:,.0f}), at {MARGIN:.0%} gross margin and
   {MONTHLY_RATE:.1%} a month. Scores are in customer_scores ({MODEL_VERSION}, as of {SCORED_AT}).
2. What we can pay to acquire a customer (12-month margin value of a new customer):
{caps_lines}
3. The retention offer at ${r['cost_per_customer']:.2f} per customer: the median customer needs a
   {r['median_lift']:.0%} lift in purchases to pay back; {r['below_10pct']:,} of {r['customers']:,}
   customers need less than 10%.
4. Assumptions: margin {MARGIN:.0%}; horizon {HORIZON_MONTHS} months; discount {MONTHLY_RATE:.1%} a
   month; current CACs as given; spend independent of purchase frequency (Module 4); each
   channel's acquisitions are incremental (not tested: Day 3); the offer's lift is not measured
   (Day 3).
Recommendation and what would change it: ________________________________________
({'QUICK' if QUICK else 'FULL'} run)""")

# %% [markdown]
# ## Decision · CAC caps and the retention campaign
#
# Two decisions: the most to pay per acquired customer in each channel, and whether to run the
# retention offer, and for whom. The rules are stated before the numbers are read.

# %%
mode = "QUICK run: treat this as a rough answer." if QUICK else "FULL run."
print("Number          : per channel, 12-month margin value of a new customer (94% HDI) and caps:")
for c in CHANNELS:
    s = summary_row(new_clv.sel(channel=c))
    print(f"                  {c:<9} value ${s['mean']:.2f} (${s['hdi_low']:.2f} to"
          f" ${s['hdi_high']:.2f}); cap ${cap_mean[c]:.2f} (mean rule),"
          f" ${cap_20[c]:.2f} (20th-percentile rule); current CAC ${CURRENT_CAC[c]:.0f};"
          f" P(value > CAC) = {p_beats_cac[c]:.2f}")
best = table.sort_values("median_lift").iloc[0]
print(f"                  Retention: median break-even lift {np.median(lift):.0%};"
      f" {int((lift < 0.10).sum()):,} of {len(lift):,} customers need less than 10%; the segment"
      f" with the smallest median is {best['segment']} ({best['median_lift']:.1%}). By segment:")
for line in table.assign(median_lift=table["median_lift"].map("{:.1%}".format)).to_string(
        index=False).splitlines():
    print(" " * 18 + line)
print("Rule            : keep acquiring through a channel while its current CAC is below the cap")
print("                  under the risk rule you chose (say which); run the retention offer for a")
print("                  segment only if its break-even lift is below what an experiment has shown")
print("                  is plausible.")
print("Recommendation  : (yours) one sentence per decision, with the probability from the Number")
print("                  line and what would change it. The caps assume each channel's acquisitions")
print("                  are incremental, and no experiment has measured the offer's lift yet: the")
print("                  retention decision is provisional until Day 3.")
print("Your sentence   : ________________________________________________")
print(f"Mode            : {mode}")

# %% [markdown]
# **Decide.** Write one sentence per decision in the cell below, in the form "Based on [number and
# interval], I would [action], because [rule, with its cost or margin], unless [what would change
# it]."

# %% [markdown]
# ## Stretch (optional) · Payback period by channel
#
# How many months does a new customer take to pay back the current CAC? For each draw, find the
# first month in which the cumulative discounted margin (acquisition purchase included) reaches the
# CAC, over up to 36 months, and report the median and 94% HDI per channel.

# %%
payback_months = np.arange(0, 37)
cum_new = xr.concat(
    [bgnbd.expected_purchases_new_customer(data=channel_rows, t=k * WEEKS_PER_MONTH)
     for k in payback_months],
    dim=xr.DataArray(payback_months, dims="month", name="month"),
).rename(customer_id="channel")
post = gg.idata.posterior
spend_new = post["p"] * post["v"] / (post["q"] - 1)
in_month_new = cum_new.diff("month")
cum_margin = MARGIN * spend_new * (1 + ((in_month_new * (1 + MONTHLY_RATE) ** -in_month_new["month"])
                                        .cumsum("month")))
rows = []
for c in CHANNELS:
    reached = (cum_margin.sel(channel=c) >= CURRENT_CAC[c]).transpose("chain", "draw", "month")
    first = xr.where(reached.any("month"), reached.argmax("month") + 1, np.inf).values.ravel()
    finite = first[np.isfinite(first)]
    lo_p, hi_p = checks.interval(finite, INTERVAL, "hdi") if finite.size > 1 else (np.nan, np.nan)
    rows.append({"channel": c, "current CAC": CURRENT_CAC[c],
                 "share of draws paid back in 36 months": float(np.isfinite(first).mean()),
                 "median months": float(np.median(first)), "hdi_low": lo_p, "hdi_high": hi_p})
pd.DataFrame(rows).round(2)
