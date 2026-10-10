# ---
# jupyter:
#   mktstats:
#     verified: >-
#       2026-10-09, Technical Expert, against installed source. pymc-marketing 1.2.0:
#       BetaGeoModel(model_config=) defaults r ~ Weibull(2, 1), alpha ~ Weibull(2, 10),
#       a = phi_dropout * kappa_dropout, b = (1 - phi_dropout) * kappa_dropout (a and b are
#       Deterministics in the posterior); passing "a" and "b" in model_config replaces them;
#       ModelBuilder.fit(data, method="map"|"mcmc", progressbar=, random_seed=, **kwargs ->
#       pymc.sample, e.g. nuts_sampler, chains, draws, tune); fit_summary() gives a Series for
#       MAP; expected_purchases(data, future_t=), expected_probability_alive(data) (x = 0 gives
#       exactly 1), ParetoNBDModel.fit (default method "map"), expected_probability_alive;
#       clv.plot_expected_purchases_over_time(model, purchase_history, customer_id_col,
#       datetime_col, t, t_start_eval=, time_unit=, time_scaler=). pymc_extras.prior.Prior.
#       arviz 1.3.0 az.hdi pools chain and draw for a DataArray but gives one interval per
#       chain for a raw 2-D array. Ran: flat-prior MAP on CDNOW reproduces note 004 (r 0.2426,
#       alpha 4.414, a 0.7930, b 2.426); E[X(t)] closed form (FHL 2005 eq. 9) matches a
#       100,000-customer simulation (2.766 vs 2.759 at t = 52).
# ---

# %% [markdown]
# # Part A · The generative story
#
# Module 1's recency rule needed an arbitrary cutoff and gave no probability. A
# *buy-till-you-die* (BTYD) model describes how each customer buys while they are a customer
# and how they silently stop. This lab fits two of them with PyMC-Marketing.
#
# **BG/NBD** (beta-geometric/negative binomial, Fader, Hardie & Lee 2005). Time is in weeks.
#
# - While alive, customer $i$ buys at random times at rate $\lambda_i$ per week (a Poisson
#   process). Across customers, $\lambda_i \sim \text{Gamma}(r, \alpha)$, with $\alpha$ a
#   **rate**: the mean is $r/\alpha$, and NumPy's `gamma` takes `scale = 1/alpha`.
# - Right after each repeat purchase the customer leaves for good with probability $p_i$, with
#   $p_i \sim \text{Beta}(a, b)$ across customers (mean $a/(a+b)$).
#
# **Pareto/NBD** keeps the purchase process but lets a customer leave at any moment, at rate
# $\mu_i \sim \text{Gamma}(s, \beta)$.
#
# For each customer the data are $x$ (repeat purchases, *frequency*), $t_x$ (time of the last
# purchase since the first, *recency*) and $T$ (time since the first purchase, *age*), as in
# Module 1.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing import clv
from scipy.special import hyp2f1

from mktstats import data

# pymc 6 warns about an argument pymc-marketing 1.2.0 passes internally; nothing to act on
warnings.filterwarnings("ignore", message=".*merge_dataset.*", category=FutureWarning)
truth = data.load_truth()["btyd_bgnbd"]["bgnbd"]  # the true r, alpha, a, b
SEED = 2027  # every random step below uses this seed (or SEED + 1)
DRAWS = 300 if QUICK else 1000  # MCMC draws and tuning steps per chain
print("True parameters:", truth)
print(f"MCMC: 2 chains x {DRAWS} draws ({'QUICK' if QUICK else 'FULL'} settings)")


def expected_purchases_new(r, alpha, a, b, t):
    """E[X(t)]: expected repeat purchases in the first t weeks of a new BG/NBD customer
    (Fader, Hardie & Lee 2005, equation 9)."""
    return ((a + b - 1) / (a - 1)
            * (1 - (alpha / (alpha + t)) ** r * hyp2f1(r, b, a + b - 1, t / (alpha + t))))


print(f"E[X(52)] at the true parameters: {expected_purchases_new(**truth, t=52):.3f}")

# %% [markdown]
# ## Exercise 1 · Simulate BG/NBD customers (7 minutes)
#
# **Predict.** Raise the average dropout probability from 0.1 to 0.5 (from $a=1, b=9$ to
# $a=5, b=5$). Will the mean frequency go up, down or stay the same? And the mean recency?
# Write both guesses down.
#
# **Task.** Write `simulate_bgnbd(r, alpha, a, b, T, n, rng)` returning a DataFrame with one
# row per customer and the columns `frequency`, `recency`, `T` and `alive` (True if the
# customer has not left by `T`). Every customer makes a first purchase at time 0 and is
# watched until `T` weeks. For each customer: draw $\lambda$ with
# `rng.gamma(shape=r, scale=1/alpha)` and $p$ with `rng.beta(a, b)`; then repeat: wait
# `rng.exponential(1/lam)` weeks for the next purchase; stop if it falls after `T`; otherwise
# count it, and with probability $p$ the customer leaves (`rng.random() < p`).

# %% tags=["exercise"]
def simulate_bgnbd(r, alpha, a, b, T, n, rng):
    # TODO 1: draw lam and p per customer, then simulate purchases until T or dropout
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def simulate_bgnbd(r, alpha, a, b, T, n, rng):
    lam = rng.gamma(shape=r, scale=1 / alpha, size=n)  # alpha is a rate
    p = rng.beta(a, b, size=n)
    rows = []
    for i in range(n):
        t, x, last, alive = 0.0, 0, 0.0, True
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x, last = x + 1, t
            if rng.random() < p[i]:  # leaves right after this repeat purchase
                alive = False
                break
        rows.append((x, last, T, alive))
    return pd.DataFrame(rows, columns=["frequency", "recency", "T", "alive"])


# %% [markdown]
# **Explain.** Run the next cell to check your prediction. Then say what each parameter
# controls: which two set how often an alive customer buys, and which two set how soon they
# leave?
#
# <details><summary>Why this solution works</summary>
#
# The waiting times of a Poisson process with rate $\lambda$ are exponential with mean
# $1/\lambda$, so adding them up gives the purchase times. $r$ and $\alpha$ set the purchase
# rates ($r/\alpha$ purchases per week on average; a small $r$ means very unequal customers);
# $a$ and $b$ set the dropout probabilities. A higher dropout probability lowers the mean
# frequency, and it lowers the mean recency too: customers stop buying earlier, so their last
# purchase comes earlier. A quirk of the BG/NBD: a customer can leave only right after a
# purchase, so a customer with no repeat purchase is always alive in this model.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    sim_check = simulate_bgnbd(**truth, T=52, n=20_000, rng=np.random.default_rng(SEED))
    checks.columns(sim_check, ["frequency", "recency", "T", "alive"],
                   name="simulate_bgnbd(...)")
    assert len(sim_check) == 20_000, f"Expected 20,000 rows (n), got {len(sim_check):,}."
    f = sim_check["frequency"].to_numpy(dtype=float)
    rec = sim_check["recency"].to_numpy(dtype=float)
    assert (f >= 0).all() and np.allclose(f, np.round(f)), (
        "frequency counts repeat purchases: whole numbers 0, 1, 2, ..."
    )
    assert ((rec >= 0) & (rec <= 52)).all(), "recency is a time between 0 and T = 52 weeks."
    assert ((f == 0) == (rec == 0)).all(), (
        "recency should be 0 exactly for customers with no repeat purchase: it is the time of"
        " the last repeat purchase."
    )
    assert sim_check.loc[f == 0, "alive"].astype(bool).all(), (
        "Some customers with no repeat purchase are marked as gone. In the BG/NBD a customer"
        " can leave only right after a repeat purchase."
    )
    expected = expected_purchases_new(**truth, t=52)
    # 3% is about three standard errors of the mean of 20,000 customers (sd of frequency
    # about 4.0), so a correct simulator fails only by rare bad luck.
    try:
        checks.close(f.mean(), expected, rel=0.03,
                     name="the mean frequency of 20,000 simulated customers over 52 weeks")
    except AssertionError as err:
        raise AssertionError(
            f"{err} The closed form E[X(52)] is {expected:.3f}. A mean far off usually means"
            " the Gamma rate and scale are swapped (scale = 1/alpha), or customers can leave"
            " before their first repeat purchase. To move on now, run"
            " workshop.use_reference(1)."
        ) from None

# %%
for a_, b_ in [(1, 9), (5, 5)]:
    s = simulate_bgnbd(truth["r"], truth["alpha"], a_, b_, T=52, n=5_000,
                       rng=np.random.default_rng(SEED))
    print(f"mean dropout probability {a_ / (a_ + b_):.1f}: mean frequency"
          f" {s['frequency'].mean():.2f}, mean recency {s['recency'].mean():.1f} weeks,"
          f" alive at week 52 {s['alive'].mean():.0%}")

# %% [markdown]
# ## Exercise 2 · Sufficient statistics and the calibration/holdout split (8 minutes)
#
# Real data come as a transaction log. To test a model we split it **by time**: fit on the
# *calibration* period, then compare its forecast with what customers did in the *holdout*
# period that follows. CDNOW: calibration to 1997-09-30 (39 weeks), holdout 1997-10-01 to
# 1998-06-30 (39 weeks), as in Fader, Hardie & Lee's BG/NBD note.
#
# **Predict.** If you swapped recency and $T$, would customers who have been silent for a long
# time look more alive or less alive than they are? Write one word down.
#
# **Task.** Write `calibration_holdout(tx, cal_end, holdout_end)` for a log with columns
# `customer_id` and `date`. Return one row per customer whose first purchase is on or before
# `cal_end`, with `customer_id`, `frequency`, `recency`, `T` (purchase days up to `cal_end`,
# weeks as days / 7, as in Module 1) and `holdout_purchases` (the number of purchase days
# after `cal_end` and on or before `holdout_end`; 0 if none). `cal_end` and `holdout_end`
# are date strings.

# %%
cdnow = data.load_cdnow().rename(columns={"id": "customer_id"})[["customer_id", "date"]]
synth_info = data.load_truth()["btyd_bgnbd"]  # calibration_end, holdout_end of the cohort
tx_synth = data.load_synthetic("bgnbd_transactions")  # a committed synthetic BG/NBD log
rfm_synth = data.load_synthetic("bgnbd_rfm")  # the generator's own summary of that log
print(f"CDNOW: {len(cdnow):,} rows; synthetic log: {len(tx_synth):,} rows")

# %% tags=["exercise"]
def calibration_holdout(tx, cal_end, holdout_end):
    # TODO 2: purchase days -> calibration summary per customer -> holdout counts
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def calibration_holdout(tx, cal_end, holdout_end):
    cal_end, holdout_end = pd.Timestamp(cal_end), pd.Timestamp(holdout_end)
    days = tx[["customer_id", "date"]].drop_duplicates()  # one row per purchase day
    cal = days.loc[days["date"] <= cal_end].groupby("customer_id")["date"]
    first, last = cal.min(), cal.max()
    out = pd.DataFrame({
        "frequency": cal.size() - 1,
        "recency": (last - first).dt.days / 7,
        "T": (cal_end - first).dt.days / 7,
    })
    hold = days.loc[(days["date"] > cal_end) & (days["date"] <= holdout_end)]
    out["holdout_purchases"] = hold.groupby("customer_id").size().reindex(out.index).fillna(0)
    return out.rename_axis("customer_id").reset_index()


# %% [markdown]
# **Explain.** Compare with your prediction. Then say why the split must be by time and not by
# customer, and why $T$ differs between customers.
#
# <details><summary>Why this solution works</summary>
#
# Only purchases up to `cal_end` enter $x$, $t_x$ and $T$, so the holdout cannot leak into
# the fit. A split by customer would fit and test on the same calendar period and say nothing
# about forecasting the future, which is the decision we need. $T$ differs because customers
# made their first purchase on different dates. Swapping recency and $T$ makes every customer
# look as if their last purchase were at the cutoff, so lapsed customers look alive.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    cal = calibration_holdout(cdnow, "1997-09-30", "1998-06-30")
    checks.columns(cal, ["customer_id", "frequency", "recency", "T", "holdout_purchases"],
                   name="calibration_holdout(cdnow, ...)")
    assert len(cal) == 2357, f"CDNOW has 2,357 customers, all acquired by March 1997; got {len(cal):,}."
    # checks.rfm_table compares frequency with recency in days, the unit purchases are counted in
    checks.rfm_table(cal.assign(recency=cal["recency"] * 7, T=cal["T"] * 7),
                     customer_id="customer_id", name="the CDNOW calibration table (in days)")
    n_zero = int((cal["frequency"] == 0).sum())
    assert n_zero == 1411, (
        f"{n_zero:,} customers have frequency 0; the BG/NBD note reports 1,411. Count purchase"
        " days (drop same-day duplicates) up to 1997-09-30 only."
    )
    total = int(cal["holdout_purchases"].sum())
    assert total == 1882, (
        f"Total holdout purchases: {total:,}; the reference (and CLVTools) count 1,882 purchase"
        " days from 1997-10-01 to 1998-06-30."
    )
    ours = calibration_holdout(tx_synth, synth_info["calibration_end"], synth_info["holdout_end"])
    generator = rfm_synth.rename(columns={"test_frequency": "holdout_purchases"})
    checks.frames_agree(ours, generator[["customer_id", "frequency", "recency", "T",
                                         "holdout_purchases"]],
                        on="customer_id", names=("calibration_holdout(tx_synth, ...)",
                                                 "the generator's own summary"))

# %% [markdown]
# # Part B · Fit and check recovery
#
# Before trusting a model on real data, check that it gets known answers back. The next cell
# simulates a cohort of 4,000 customers at the true parameters, with the same story as your
# `simulate_bgnbd` written without the loop (so every participant fits exactly the same
# cohort, whatever order their own function draws random numbers in). It then fits the BG/NBD
# two ways, with PyMC-Marketing's default priors. (A *prior* is the spread of parameter values
# the model finds plausible before it sees data; the *posterior* is that spread after the data
# have updated it.)
#
# - **MAP** (maximum a posteriori): the single most probable parameter values. Fast; no
#   uncertainty.
# - **MCMC** (Markov chain Monte Carlo, here the nutpie sampler): thousands of draws from the
#   posterior distribution, so every parameter comes with an interval.
#
# The first fit in a fresh runtime compiles the model (30 to 45 seconds on Colab). While it
# runs, read Exercise 3's Predict prompt. With QUICK on, MCMC draws fewer samples, so
# intervals are rougher and may not match the module page; that is a lesson about sample
# size, not a bug.

# %%
def simulate_cohort(r, alpha, a, b, T, n, rng):
    """Exercise 1's story without the loop: a customer who never left would make
    Poisson(lam * T) purchases; they leave after the Geometric(p)-th repeat purchase; the last
    of x uniform purchase times out of N is T times a Beta(x, N - x + 1) draw."""
    lam = rng.gamma(shape=r, scale=1 / alpha, size=n)
    p = rng.beta(a, b, size=n)
    leave_after = rng.geometric(p)
    n_if_alive = rng.poisson(lam * T)
    x = np.minimum(n_if_alive, leave_after)
    recency = np.where(x > 0, T * rng.beta(np.maximum(x, 1), n_if_alive - x + 1), 0.0)
    return pd.DataFrame({"frequency": x, "recency": recency, "T": float(T),
                         "alive": n_if_alive < leave_after})


# Seed of the cohort. Even with the right model, "3 of 4 true values inside their 94% HDI"
# fails on some cohorts by chance: r and alpha (and a and b) move together, so they miss in
# pairs. In a scan of seeds 2028-2047 (2 chains x 1000 draws) the check passed on 17 of 20
# cohorts, and each failure was one such pair. This seed's cohort passes in QUICK and FULL;
# Exercise 3's Explain says why other cohorts can miss.
COHORT_SEED = 2031
sim = simulate_cohort(**truth, T=52, n=4_000, rng=np.random.default_rng(COHORT_SEED))
sim_rfm = sim.assign(customer_id=np.arange(1, len(sim) + 1))[
    ["customer_id", "frequency", "recency", "T"]]
print(f"Simulated cohort: mean frequency {sim['frequency'].mean():.3f}"
      f" (closed form {expected_purchases_new(**truth, t=52):.3f})")

# %%
bg_sim_map = clv.BetaGeoModel()
bg_sim_map.fit(data=sim_rfm, method="map", progressbar=False, random_seed=SEED)
bg_sim = clv.BetaGeoModel()
bg_sim.fit(data=sim_rfm, method="mcmc", nuts_sampler="nutpie", chains=2, draws=DRAWS,
           tune=DRAWS, random_seed=SEED, progressbar=False)
idata = bg_sim.idata  # the posterior draws: idata.posterior["r"] has dims (chain, draw)
print(bg_sim.fit_summary(var_names=list(truth), ci_prob=0.94, ci_kind="hdi").iloc[:, :4])

# %% [markdown]
# ## Exercise 3 · Parameter recovery table (8 minutes)
#
# A *94% HDI* (highest density interval) is the narrowest interval that holds 94% of the
# posterior draws.
#
# **Predict.** Which parameter will have the widest 94% HDI relative to its mean: $r$,
# $\alpha$, $a$ or $b$? Pick one.
#
# **Task.** Write `recovery_table(idata, truth, prob=0.94)` returning one row per parameter in
# `truth` with the columns `param`, `truth`, `mean` (posterior mean), `hdi_low`, `hdi_high`
# (the `prob` HDI) and `covered` (True if the truth lies inside). The draws of parameter `p`
# are `idata.posterior[p]`. For the HDI use `checks.interval(draws, prob, "hdi")`, which pools
# the chains, or `az.hdi(idata.posterior[p], prob=prob)`; a raw NumPy array of shape
# (chain, draw) given to `az.hdi` gets one interval per chain.

# %% tags=["exercise"]
def recovery_table(idata, truth, prob=0.94):
    # TODO 3: one row per parameter: truth, posterior mean, HDI, covered
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def recovery_table(idata, truth, prob=0.94):
    rows = []
    for p, true_value in truth.items():
        draws = idata.posterior[p].to_numpy()
        lo, hi = checks.interval(draws, prob, "hdi")
        rows.append({"param": p, "truth": true_value, "mean": draws.mean(),
                     "hdi_low": lo, "hdi_high": hi, "covered": lo <= true_value <= hi})
    return pd.DataFrame(rows)


# %% [markdown]
# **Explain.** Compare with your prediction, using the `width / mean` column below. Why are the
# dropout parameters harder to pin down than the purchase parameters?
#
# <details><summary>Why this solution works</summary>
#
# The posterior mean and the HDI summarize the draws of each parameter, pooled over both
# chains. $a$ and $b$ are less well identified than $r$ and $\alpha$ because dropout is never
# observed: the data show purchases, and a long silence is consistent with a customer who left
# and with one who buys rarely. Even when the model is exactly right, a 94% interval misses
# the truth 6% of the time. Here the misses come in pairs: the data pin down the mean purchase
# rate $r/\alpha$ much better than $r$ and $\alpha$ separately, so if $r$ is too high, so is
# $\alpha$ (and likewise $a$ and $b$). That is why the check asks for 3 of 4, and why on some
# other simulated cohorts even 3 of 4 fails by chance.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    table = recovery_table(idata, truth, prob=0.94)
    checks.columns(table, ["param", "truth", "mean", "hdi_low", "hdi_high", "covered"],
                   name="recovery_table(idata, truth)")
    assert sorted(table["param"]) == sorted(truth), (
        f"One row per parameter {sorted(truth)}; got {table['param'].tolist()}."
    )
    for row in table.itertuples():
        draws = idata.posterior[row.param].to_numpy()
        lo, hi = checks.interval(draws, 0.94, "hdi")
        tol = 0.02 * draws.std()  # az.hdi and checks.interval can differ by one draw
        checks.close([row.mean, row.hdi_low, row.hdi_high], [draws.mean(), lo, hi],
                     abs=tol, name=f"the mean and 94% HDI of {row.param} (pooled over chains)")
        assert bool(row.covered) == bool(lo <= truth[row.param] <= hi), (
            f"covered is wrong for {row.param}: is the truth inside [hdi_low, hdi_high]?"
        )
    checks.k_of_K_in_interval(truth, {p: idata.posterior[p].to_numpy() for p in truth}, k=3,
                              prob=0.94, kind="hdi", name="r, alpha, a, b (simulated cohort)")

# %% [markdown]
# ### Run · MAP versus MCMC
#
# Look at whether the MAP value and the posterior mean agree, and note that MAP gives one
# number with no interval.

# %%
map_values = bg_sim_map.fit_summary()
comparison = table.assign(**{"width / mean": (table["hdi_high"] - table["hdi_low"]) / table["mean"],
                              "MAP": [map_values[p] for p in table["param"]]})
comparison["MAP inside the 94% HDI"] = ((comparison["MAP"] >= comparison["hdi_low"])
                                        & (comparison["MAP"] <= comparison["hdi_high"]))
comparison.round(3)

# %% [markdown]
# MAP is fast and good enough for point forecasts. It says nothing about uncertainty, which
# matters for customers near a decision threshold. The lab uses MAP alone only where MCMC is
# too slow for a class: the Pareto/NBD below (2 chains of 300 draws took 231 s on Colab).
#
# # Part C · Real data: CDNOW
#
# Three fits on the CDNOW calibration table from Exercise 2. The first uses flat priors, so its
# MAP is the maximum-likelihood estimate that Fader, Hardie & Lee published; it is the
# benchmark for the whole pipeline.

# %%
cal_cdnow = cal[["customer_id", "frequency", "recency", "T"]]
bg_flat = clv.BetaGeoModel(model_config={k: Prior("HalfFlat") for k in ("r", "alpha", "a", "b")})
bg_flat.fit(data=cal_cdnow, method="map", progressbar=False, random_seed=SEED)
print("BG/NBD, flat priors, MAP:", bg_flat.fit_summary()[list(truth)].round(4).to_dict())

# %% tags=["checkpoint"]
with workshop.checkpoint(label="benchmark"):
    published = {"r": 0.243, "alpha": 4.414, "a": 0.793, "b": 2.426}  # BG/NBD note 004
    estimate = bg_flat.fit_summary()
    for p, value in published.items():
        try:
            checks.close(estimate[p], value, rel=0.01,
                         name=f"the flat-prior MAP of {p} on CDNOW (published {value})")
        except AssertionError as err:
            raise AssertionError(
                f"{err} This fit used your calibration table: check Exercise 2 (purchase days,"
                " weeks as days / 7, the 1997-09-30 cutoff)."
            ) from None

# %% [markdown]
# Next, the BG/NBD with default priors by MCMC (about 10 s on a laptop, longer on Colab), and
# the Pareto/NBD with flat priors by MAP.

# %%
bg_cdnow = clv.BetaGeoModel()
bg_cdnow.fit(data=cal_cdnow, method="mcmc", nuts_sampler="nutpie", chains=2, draws=DRAWS,
             tune=DRAWS, random_seed=SEED, progressbar=False)
print(bg_cdnow.fit_summary(var_names=list(truth), ci_prob=0.94, ci_kind="hdi").iloc[:, :4])

# %%
pnbd = clv.ParetoNBDModel(
    model_config={k: Prior("HalfFlat") for k in ("r", "alpha", "s", "beta")})
pnbd.fit(data=cal_cdnow, method="map", progressbar=False, random_seed=SEED)
pnbd_values = pnbd.fit_summary()[["r", "alpha", "s", "beta"]]
print("Pareto/NBD, flat priors, MAP:", pnbd_values.round(6).to_dict())

# %% [markdown]
# ## Exercise 4 · Benchmark and holdout tracking (9 minutes)
#
# **Predict.** For customers with 7 or more calibration purchases, will the model's holdout
# forecast be above or below what they bought? For customers with none?
#
# **Task.** Write `holdout_comparison(model, cal, holdout_weeks)` returning one row per group
# of calibration frequency `"0"`, `"1"`, ..., `"6"`, `"7+"` (in that order) with the columns
# `group`, `customers`, `actual` (mean `holdout_purchases`) and `predicted` (mean forecast).
# The forecast per customer is
# `model.expected_purchases(data=cal, future_t=holdout_weeks).mean(("chain", "draw"))`: an
# array indexed by `customer_id` (`.to_series()` turns it into a Series). With MAP there is
# one draw, so the mean changes nothing.

# %% tags=["exercise"]
def holdout_comparison(model, cal, holdout_weeks):
    # TODO 4: forecast per customer, then mean actual and mean forecast by frequency group
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def holdout_comparison(model, cal, holdout_weeks):
    forecast = model.expected_purchases(data=cal, future_t=holdout_weeks)
    forecast = forecast.mean(("chain", "draw")).to_series()
    df = pd.DataFrame({"x": cal["frequency"].clip(upper=7).astype(int),
                       "actual": cal["holdout_purchases"],
                       "predicted": cal["customer_id"].map(forecast)})
    out = df.groupby("x").agg(customers=("actual", "size"), actual=("actual", "mean"),
                              predicted=("predicted", "mean")).reset_index()
    out["group"] = out["x"].map(lambda v: "7+" if v == 7 else str(v))
    return out[["group", "customers", "actual", "predicted"]]


# %% [markdown]
# **Explain.** Compare with your prediction, group by group, in the table under the
# checkpoint. Where do the two models agree? Point to the group with the largest gap between
# `actual` and `predicted`: does the forecast fall short there or overshoot?
#
# <details><summary>Why this solution works</summary>
#
# `expected_purchases` gives each customer's expected purchases in the next `holdout_weeks`
# given their $x$, $t_x$ and $T$; averaging within frequency groups shows whether the model is
# right for light and heavy buyers, not only in total. Both models predict a total a little
# below what happened here, and the BG/NBD and Pareto/NBD forecast almost the same numbers, as
# Fader, Hardie & Lee found. Default priors pull $a$ and $b$ away from the published estimates
# (compare `bg_cdnow` with `bg_flat`): with 2,357 customers the prior still matters for the
# dropout parameters, which the data identify least well.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    HOLDOUT_WEEKS = 39  # 1997-10-01 to 1998-06-30
    groups = ["0", "1", "2", "3", "4", "5", "6", "7+"]
    actual_total = int(cal["holdout_purchases"].sum())
    tables = {}
    for name, model in (("BG/NBD", bg_cdnow), ("Pareto/NBD", pnbd)):
        comp = holdout_comparison(model, cal, HOLDOUT_WEEKS)
        checks.columns(comp, ["group", "customers", "actual", "predicted"],
                       name=f"holdout_comparison({name})")
        assert comp["group"].astype(str).tolist() == groups, (
            f"Groups should be {groups} in that order; got {comp['group'].tolist()}."
        )
        assert comp["customers"].sum() == len(cal), (
            f"{name}: the groups hold {comp['customers'].sum():,} customers, not {len(cal):,}."
            " Clip frequency at 7 so every customer lands in one group, '0' to '7+'."
        )
        checks.close((comp["customers"] * comp["actual"]).sum(), actual_total, abs=1e-6,
                     name=f"{name}: actual holdout purchases summed over groups")
        predicted_total = float((comp["customers"] * comp["predicted"]).sum())
        direct = float(model.expected_purchases(data=cal_cdnow, future_t=HOLDOUT_WEEKS)
                       .mean(("chain", "draw")).sum())
        checks.close(predicted_total, direct, rel=1e-6,
                     name=f"{name}: predicted purchases summed over groups")
        # The reference forecasts are 10 to 12% below the 1,882 purchases that happened
        # (measured 2026-10-09); 15% leaves room for MCMC noise in QUICK runs.
        checks.close(predicted_total, actual_total, rel=0.15,
                     name=f"{name}: predicted total holdout purchases (actual {actual_total:,})")
        tables[name] = comp.set_index("group")
    side_by_side = tables  # kept only when every check above passed
    print(f"Holdout total: actual {actual_total:,}; BG/NBD"
          f" {(side_by_side['BG/NBD'].eval('customers * predicted')).sum():,.0f}; Pareto/NBD"
          f" {(side_by_side['Pareto/NBD'].eval('customers * predicted')).sum():,.0f}")

# %% [markdown]
# Mean holdout purchases per customer, actual and predicted, by calibration frequency:

# %%
pd.concat(side_by_side, axis=1).round(2)

# %% [markdown]
# Look at whether the forecast (dotted line) tracks the actual cumulative repeat purchases
# (solid line) after week 39 (dashed red line), where the holdout begins. The forecast for the
# holdout uses only the calibration fit (flat-prior BG/NBD, MAP).

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
clv.plot_expected_purchases_over_time(
    bg_flat, purchase_history=cdnow, customer_id_col="customer_id", datetime_col="date",
    t=(pd.Timestamp("1998-06-30") - cdnow["date"].min()).days, time_unit="D", time_scaler=7,
    t_start_eval=39,  # the plot's x-axis counts weeks (one point per time_scaler days)
    ax=ax, xlabel="Weeks since 1997-01-01", ylabel="Cumulative repeat purchases",
    style=["-", ":"],
)
ax.legend(["actual", "BG/NBD forecast", "holdout starts"])
fig.tight_layout()

# %% [markdown]
# # Part D · Toward the decision
#
# *P(alive)* is the probability that a customer has not yet left, given their $x$, $t_x$ and
# $T$. Retargeting an alive customer costs $c$ (the contact cost) and brings an extra purchase
# with probability $u$ (the response the ad causes), worth margin $m$. Retargeting is worth it
# when the expected gain beats the cost: $P(\text{alive}) \times m \times u > c$, that is,
# when $P(\text{alive})$ exceeds the threshold $\tau = c / (m \times u)$.
#
# ## Exercise 5 · A P(alive) threshold for retargeting (8 minutes)
#
# **Predict.** Contact cost \$0.50, margin \$20 per purchase, and 5% of alive customers buy
# because of the ad. What threshold $\tau$ do you get?
#
# **Task.** Write `retarget_threshold(contact_cost, margin, response_rate)` returning $\tau$,
# and `retarget_flags(p_alive, tau)` returning a boolean Series that is True for customers to
# **keep** retargeting ($P(\text{alive}) \ge \tau$).

# %% tags=["exercise"]
def retarget_threshold(contact_cost, margin, response_rate):
    # TODO 5: tau = c / (m * u)
    raise NotImplementedError("TODO 5")


def retarget_flags(p_alive, tau):
    # TODO 5: True = keep retargeting
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def retarget_threshold(contact_cost, margin, response_rate):
    return contact_cost / (margin * response_rate)


@workshop.solution(5)
def retarget_flags(p_alive, tau):
    return pd.Series(p_alive) >= tau


# %% [markdown]
# **Explain.** Compare with your prediction: $\tau = 0.50 / (20 \times 0.05) = 0.5$. $u$ is an
# assumption here, not a measurement: what would you need to measure it (Day 3)? And why does
# the CDNOW decision below use the Pareto/NBD's P(alive), not the BG/NBD's?
#
# <details><summary>Why this solution works</summary>
#
# The threshold is where expected margin equals cost. A higher margin or response rate lowers
# it (retarget more people); a higher cost raises it. If $\tau > 1$, no customer is worth
# retargeting under these assumptions. $u$ comes from an experiment that retargets a random
# half of customers (Module 6). The BG/NBD gives P(alive) = 1 to every customer with no repeat
# purchase, because in that model a customer can only leave right after a purchase, so a
# BG/NBD rule would never stop retargeting the 1,411 CDNOW customers who never came back. The
# Pareto/NBD lets them leave at any time.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    for (c, m, u), tau_true in [((0.50, 20, 0.05), 0.5), ((1.00, 10, 0.20), 0.5),
                                ((0.30, 30, 0.01), 1.0)]:
        checks.close(retarget_threshold(c, m, u), tau_true, rel=1e-9,
                     name=f"retarget_threshold({c}, {m}, {u})")
    toy = retarget_flags(pd.Series([0.2, 0.5, 0.9]), 0.5)
    assert list(pd.Series(toy).astype(bool)) == [False, True, True], (
        "retarget_flags should be True (keep) when P(alive) >= tau: [False, True, True] for"
        f" [0.2, 0.5, 0.9] and tau 0.5; got {list(toy)}."
    )
    tau = retarget_threshold(0.50, 20, 0.05)
    p_alive_sim = bg_sim.expected_probability_alive(data=sim_rfm).mean(("chain", "draw"))
    p_alive_sim = p_alive_sim.to_series().reindex(sim_rfm["customer_id"]).to_numpy()
    keep = pd.Series(retarget_flags(pd.Series(p_alive_sim), tau)).to_numpy(dtype=bool)
    assert (keep == (p_alive_sim >= tau)).all(), "Flags must equal P(alive) >= tau."
    alive = sim["alive"].to_numpy(dtype=bool)
    share_stop, share_keep = alive[~keep].mean(), alive[keep].mean()
    assert share_stop < share_keep, (
        f"Customers flagged 'stop' should be less often alive than those kept; got"
        f" {share_stop:.1%} versus {share_keep:.1%}."
    )
    print(f"Simulated cohort, tau = {tau:.2f}: {(~keep).sum():,} to stop, of whom"
          f" {share_stop:.1%} are truly alive; {keep.sum():,} to keep, {share_keep:.1%} alive.")

# %% [markdown]
# ## Decision · Whom do we stop retargeting?
#
# **The rule, stated before the numbers.** Stop retargeting a CDNOW customer at the end of the
# calibration period when $P(\text{alive}) \times m \times u < c$, with $c$ = \$0.50 per
# contact and $m$ = \$20 margin per purchase. P(alive) comes from the Pareto/NBD. The response
# rate $u$ has not been measured, so the table repeats the rule for $u$ = 2%, 5% and 10%. For
# each, it counts the customers to stop, the contact cost saved, and the expected margin given
# up ($\sum P(\text{alive}) \times m \times u$ over the stopped customers).

# %%
CONTACT_COST, MARGIN = 0.50, 20.0
p_alive = pnbd.expected_probability_alive(data=cal_cdnow).mean(("chain", "draw"))
p_alive = p_alive.to_series().reindex(cal_cdnow["customer_id"]).to_numpy()
rows = []
for u in (0.02, 0.05, 0.10):
    tau_u = retarget_threshold(CONTACT_COST, MARGIN, u)
    stop = ~pd.Series(retarget_flags(pd.Series(p_alive), tau_u)).to_numpy(dtype=bool)
    rows.append({"u": u, "tau": round(tau_u, 3), "customers to stop": int(stop.sum()),
                 "share of customers": round(stop.mean(), 3),
                 "cost saved ($)": round(stop.sum() * CONTACT_COST, 2),
                 "margin given up ($)": round((p_alive[stop] * MARGIN * u).sum(), 2)})
decision = pd.DataFrame(rows)
p_alive_bg = bg_flat.expected_probability_alive(data=cal_cdnow).mean(("chain", "draw")).to_numpy()
print(f"At u = 5% the BG/NBD would stop {(p_alive_bg < 0.5).sum():,} customers, none of the"
      f" {(cal_cdnow['frequency'] == 0).sum():,} without a repeat purchase (its P(alive) is 1).")
print("Mode:", "QUICK run: treat this as a rough answer" if QUICK else "FULL run",
      "| The Pareto/NBD is a MAP fit, so the spread comes from u, not from a posterior.")
decision

# %% [markdown]
# **Recommendation.** Write one sentence: how many customers to stop retargeting and the money
# involved, how sure you are (the range over $u$), and what would change it: a measured $u$
# (Day 3) or a posterior for the Pareto/NBD (stretch).
#
# Your sentence: ________________________________________________

# %% [markdown]
# ## Stretch (optional) · Day-level data, P(alive) maps and a posterior decision
#
# 1. **Purchases recorded by day.** Fit `clv.BetaGeoModel()` by MAP to `rfm_synth` (the
#    committed synthetic cohort, whose purchases were recorded by calendar day) and compare
#    $r$ and $\alpha$ with `truth`. The fit sees slightly fewer purchases than the BG/NBD
#    expects, because two purchases on one day count once. Does that push $r/\alpha$ (the
#    mean purchase rate) up or down?
# 2. **P(alive) maps.** Draw `clv.plot_probability_alive_matrix(bg_flat)` and
#    `clv.plot_probability_alive_matrix(pnbd)` side by side. Where do they differ most?
# 3. **A posterior decision.** Fit `clv.ModifiedBetaGeoModel()` (MBG/NBD, which lets a
#    customer leave right after the first purchase too) by MCMC with nutpie on `cal_cdnow`,
#    apply the rule to each posterior draw of P(alive), and report for the customers near
#    $\tau$ the probability that stopping beats retargeting. This adds a minute or more of
#    compute on Colab. (A Pareto/NBD by MCMC took 231 s on Colab for 2 x 300 draws.)
