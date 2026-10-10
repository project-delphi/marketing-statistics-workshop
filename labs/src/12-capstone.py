# %% [markdown]
# <!--
# APIs this lab calls, checked 2026-10-09 against the installed source in
# .venv/lib/python3.13/site-packages (pymc-marketing 1.2.0, pymc 6.3.2, pymc-extras 0.15.1,
# ArviZ 1.3.0, econml 0.17.0, scikit-learn 1.6.1, SciPy 1.16.3, threadpoolctl 3.7.0), then by
# running every call below:
# - pymc_marketing.clv.BetaGeoModel(model_config={"purchase_covariate_cols": [...],
#   "dropout_covariate_cols": [...]}) (clv/models/beta_geo.py): with covariates
#   alpha_i = alpha_scale * exp(-x_i . purchase_coefficient_alpha), a_i and b_i scaled by
#   exp(x_i . dropout_coefficient_a|b) (_extract_predictive_variables);
#   .fit(data=, method="mcmc", nuts_sampler="nutpie", chains=, draws=, tune=, random_seed=,
#   progressbar=False); .expected_purchases_new_customer(data=, t=) -> DataArray over
#   (chain, draw, customer_id) (FHL 2005 eq. 9), covariates read from the data's columns.
# - GammaGammaModel(model_config={"p", "q", "v": pymc_extras.prior.Prior}) (clv/models/
#   gamma_gamma.py; package defaults Weibull(2, 1), Weibull(2, 1), Weibull(2, 10)); posterior
#   p, q, v; mean spend per purchase p v / (q - 1).
# - mktstats.data.rfm_summary(transactions, "customer_id", "date", "amount",
#   observation_period_end, "D", 7) (pymc-marketing's rfm_summary conventions, in weeks).
# - mktstats.recovery.synthetic_control(panel, treated_geos, test_start) (this repository):
#   SLSQP weights >= 0 summing to 1 for the treated mean, in-space placebos, RMSPE-ratio
#   p-value, placebo_relative_effects and placebo_pre_rmspe.
# - pymc_marketing.mmm.MMM(date_column=, channel_columns=, target_column=, control_columns=,
#   adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(), yearly_seasonality=);
#   .build_model(X, y); .add_lift_test_measurements(df) with channel, x, delta_x, delta_y, sigma
#   after build_model (mmm/mmm.py): a cut (delta_x < 0, delta_y < 0) is accepted because
#   lift_test.assert_monotonic only needs delta_x * delta_y >= 0 and the likelihood uses
#   absolute values; .fit(X, y, nuts_sampler="nutpie", chains=, draws=, tune=, target_accept=,
#   random_seed=, progressbar=False); .incrementality.contribution_over_spend(frequency=
#   "all_time", start_date=, end_date=) (mmm/incrementality.py) -> ROAS draws by channel;
#   idata.constant_data["channel_scale"], ["target_scale"]; idata["/sample_stats"]["diverging"].
#   pymc-marketing recomputes deterministics after a nutpie fit with a progress bar it does not
#   let us switch off, so the fit cells print nothing else.
# - arviz.rhat(idata, var_names=).
# - econml.dml.CausalForestDML(model_y=, model_t=, discrete_treatment=True, n_estimators=
#   (divisible by subforest_size=4), random_state=); .fit(Y, T, X=); .effect(X).
# - scipy.optimize.minimize(method="SLSQP", bounds=, constraints=[{"type": "eq", ...}]);
#   threadpoolctl.threadpool_limits(limits, user_api) as a context manager.
# -->
#
# # Part A · The scenario and the toolkit
#
# **The question.** Putting the week together: what should the retailer spend next quarter,
# where, and on whom? You answer it as a pair, in this notebook, and present the answer as a
# five-slide brief.
#
# **How the capstone works.** Five stages, one per day of the week. Each stage asks for one
# short function (an *exercise*), runs a provided model fit or analysis, and ends with a
# **checkpoint** that compares your result with the scenario's hidden answers. Checkpoints 1 to
# 5 are the rubric's checkpoints A to E. Each stage's output feeds the next, so an error early
# on shows up in the final number: that is why every stage is checked before the next one uses
# it. If a stage has not passed by its time, run `workshop.use_reference(n)` for it, move on,
# and say so on your last slide.
#
# | Block | Stage | Minutes |
# |---|---|---|
# | Block 1 | Open, run the setup cells, read the scenario | 5 |
# | | 1 · What is a new customer worth, by channel? | 30 |
# | | 2 · Is search incremental? | 30 |
# | | *slack* | 15 |
# | Lunch | | |
# | Block 2 | 3 · Calibrate the marketing mix model with the geo test | 25 |
# | | 4 · Next quarter's budget | 20 |
# | | 5 · Whom to send the retention offer | 20 |
# | | The five-slide brief | 25 |
# | | *slack* | 10 |
#
# The minutes are estimates until a pilot measures them.
#
# **Compute.** Five model fits run in provided cells: a BG/NBD and a Gamma-Gamma model
# (Stage 1), the marketing mix model twice (Stage 3) and a causal forest (Stage 5). They use
# fewer draws than the module labs (500 per chain instead of 1,000 to 2,000) so that the whole
# pipeline runs in a few minutes; each fit prints how long it took. **QUICK** draws 200 per chain
# and grows 200 trees instead of 1,000: intervals are wider and may not match the module pages,
# and a decision near its threshold can flip. That is a lesson about sample size, not a bug.
#
# **Names used throughout.** *Margin*: the retailer keeps 30% of each sales dollar (gross
# margin). *Media channels* (tv, search, social, display) are where money is spent.
# *Acquisition channels* (search, social, referral) are how a customer first arrived. They are
# different lists that share two names. A *94% HDI* (highest density interval) is the shortest
# interval that holds 94% of the posterior draws; every interval in this notebook is named.

# %%
import io
import time
import warnings
from statistics import NormalDist

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from mktstats import data
from mktstats.recovery import synthetic_control

# Notices from inside the libraries that say nothing about this lab: a PyMC 6 deprecation,
# and two about optional progress-bar widgets.
warnings.filterwarnings("ignore", message=".*merge_dataset.*")
warnings.filterwarnings("ignore", message=".*ipywidgets.*")
warnings.filterwarnings("ignore", message=".*IProgress not found.*")


def usd(x, digits=0):
    """Dollars with the sign in front: usd(-1234.5) -> '-$1,234'."""
    return f"{'-' if x < 0 else ''}${abs(x):,.{digits}f}"


truth_all = data.load_truth()["capstone"]
scenario = truth_all["scenario"]  # what the pairs are told
answers = truth_all["answers"]  # hidden answers: only the checkpoints and the last cells read them
tx = data.load_synthetic("capstone_transactions")  # customer_id, date, amount
customers = data.load_synthetic("capstone_customers")
panel = data.load_synthetic("capstone_geo_panel")  # date, geo, region, sales, treated, post
weekly = data.load_synthetic("capstone_mmm_weekly")  # date_week, tv, search, social, display, ...
email = data.load_synthetic("capstone_email_experiment")  # e-mail test, with a train/test split

SEED = 2032  # every random step in this notebook uses this seed
DRAWS = 200 if QUICK else 500  # MCMC draws and tuning steps per chain, every fit
N_TREES = 200 if QUICK else 1000  # causal forest trees (divisible by 4)
MARGIN = scenario["margin"]
CHANNELS = scenario["mmm"]["channels"]  # media channels
ACQUISITION = ["search", "social", "referral"]  # acquisition channels; search is the reference
print(scenario["story"])
print(f"\nSettings: {'QUICK' if QUICK else 'FULL'}; {DRAWS} draws per chain, {N_TREES} trees;"
      f" margin {MARGIN:.0%}.")

# %% [markdown]
# The facts the business gives you. Read them once now; each stage refers back to them.

# %%
gt, bud, link = scenario["geo_test"], scenario["budget"], scenario["channel_link"]
offer = scenario["retention_offer"]
cv = scenario["customer_value"]
print(f"Customer value: {cv['months']} months, {cv['monthly_discount_rate']:.0%} a month;"
      f" {cv['convention']}.")
print(f"Geo test on {gt['channel']}: {gt['spend_change_description']} in"
      f" {len(gt['treated_geos'])} regions ({', '.join(gt['treated_geos'])}) from"
      f" {gt['test_start']} to {gt['test_end']} ({gt['test_weeks']} weeks); national search"
      f" spend before the test {usd(gt['base_weekly_spend_national'])} a week; search spend"
      f" saved in the treated regions {usd(-gt['spend_change_total'])}.")
print(f"MMM data: weekly, {scenario['mmm']['window_start']} to {scenario['mmm']['window_end']}"
      " (it ends the week before the geo test).")
print(f"Budget: ${bud['weekly_budget']:,.0f} a week for {bud['planning_weeks']} weeks"
      f" ({bud['budget_rule']}); bounds: {bud['bounds_rule']}.")
print(f"Retention offer: costs ${offer['offer_cost']:.2f} per customer; rule: {offer['rule']};"
      f" score on {offer['held_out']}.")
pd.DataFrame({
    "current plan ($/week)": bud["current_allocation"],
    "lower bound": {c: b[0] for c, b in bud["bounds"].items()},
    "upper bound": {c: b[1] for c, b in bud["bounds"].items()},
    "cost per new customer ($)": link["cost_per_new_customer"],
    **{f"new customers recorded as {a}": {m: s[a] for m, s in link["media_to_acquisition"].items()}
       for a in ACQUISITION},
}).loc[CHANNELS].round(2)

# %% [markdown]
# The last table links the two channel lists, an assumption the business states rather than an
# estimate: of the new customers tv brings, 30% are recorded as search customers (they search
# for the brand) and 70% as referrals. Media spend per new customer is constant within the
# bounds.
#
# **The toolkit.** The next cell holds functions from earlier labs that the pipeline reuses:
# Module 7's synthetic control (`geo_readout`), Module 10's response curve and SLSQP allocator
# (`response_curve`, `allocate`), Module 11's features and a held-out profit estimate
# (`features`, `targeting_profit`), and a plotting helper. It is folded; open it to read them,
# or paste your own versions from the module labs.

# %%
# @title Toolkit: functions from Modules 7, 10 and 11 (open to read) { display-mode: "form" }
def show(fig):
    """Show a figure as a PNG on Colab, in Jupyter and in a headless run alike."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))


def geo_readout(panel, treated_geos, test_start):
    """Module 7: synthetic control for the treated geos' mean weekly sales, with in-space
    placebos (each control geo refitted against the others). Returns a dict with
    `incremental` (sales change over the test window, all treated geos), `p_value` (RMSPE
    ratio rank), `relative_effect`, `synthetic_post_total` (the counterfactual sales of the
    treated geos in the window), `gap` (treated mean minus synthetic, every week),
    `placebo_relative_effects` and `placebo_pre_rmspe` (each placebo's pre-period fit error,
    relative to its own level)."""
    with threadpool_limits(1, "blas"):  # tiny problems: one BLAS thread is fastest
        return synthetic_control(panel, treated_geos, test_start)


def response_curve(spend, beta, lam, x_scale, y_scale):
    """Module 10: steady-state weekly sales of a constant weekly spend (NumPy, broadcasts)."""
    z = lam * np.asarray(spend, dtype=float) / x_scale
    return beta * y_scale * (1 - np.exp(-z)) / (1 + np.exp(-z))


def allocate(total, bounds, curves):
    """Module 10: maximize the sum of `curves[c](spend_c)` with SLSQP, subject to the total
    and the bounds; returns a pd.Series of weekly spend by channel."""
    names = list(curves)
    with threadpool_limits(1, "blas"):
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


def features(df):
    """Module 11: the numeric features of the e-mail test (categories as 0/1 indicators)."""
    return df[["recency", "history", "mens", "womens", "newbie"]].astype(float).assign(
        zip_rural=(df["zip_code"] == "Rural").astype(float),
        zip_suburban=(df["zip_code"] == "Surburban").astype(float),
        channel_web=(df["channel"] == "Web").astype(float),
        channel_multichannel=(df["channel"] == "Multichannel").astype(float),
    )


def targeting_profit(send, y, t, margin, offer_cost, prob=0.94):
    """Module 11: profit per customer of sending the offer to the customers in `send`,
    estimated from a randomized test without any true effects: share sent x (margin x
    (mean spend with the offer - mean spend without, among those sent) - offer cost).
    Returns (estimate, low, high), the interval a normal approximation at `prob`."""
    send, y, t = (np.asarray(a) for a in (send, y, t))
    share = send.mean()
    if share == 0:
        return 0.0, 0.0, 0.0
    y1, y0 = y[send & (t == 1)], y[send & (t == 0)]
    estimate = share * (margin * (y1.mean() - y0.mean()) - offer_cost)
    se = share * margin * np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = NormalDist().inv_cdf(0.5 + prob / 2)
    return float(estimate), float(estimate - z * se), float(estimate + z * se)


print("Toolkit ready: geo_readout, response_curve, allocate, features, targeting_profit, show")

# %% [markdown]
# # Part B · Stage 1: what is a new customer worth, by channel?
#
# **Customer lifetime value** (CLV) here is the margin a newly acquired customer brings over
# the next 36 months, in today's dollars. Module 5's convention: the acquisition purchase counts
# in month 0, not discounted; then in each month $k = 1, \dots, 36$ the customer makes
# $\Delta N_k$ expected repeat purchases of average value $\bar s$, discounted by $(1 + d)^k$:
#
# $$\text{CLV} = m\,\bar s\,\Big(1 + \sum_{k=1}^{36} \frac{\Delta N_k}{(1 + d)^k}\Big),
# \qquad \Delta N_k = N(k) - N(k - 1),$$
#
# where $N(k)$ is the expected number of repeat purchases of a new customer by the end of month
# $k$ (from a buy-till-you-die model, Module 2), $\bar s$ the mean spend per purchase (from the
# Gamma-Gamma model, Module 4), $d = 1\%$ the monthly discount rate and $m = 30\%$ the margin.
#
# A first look at the customers, before any model: how often each acquisition channel's
# customers came back.

# %%
rfm = data.rfm_summary(tx, "customer_id", "date", "amount",
                       observation_period_end=tx["date"].max(), time_unit="D", time_scaler=7)
rfm = rfm.merge(customers[["customer_id", "acquisition_channel", "channel_social",
                           "channel_referral"]], on="customer_id")
print(f"{len(rfm):,} customers, purchases {tx['date'].min():%Y-%m-%d} to"
      f" {tx['date'].max():%Y-%m-%d}; recency and T in weeks")
rfm.groupby("acquisition_channel").agg(
    customers=("customer_id", "size"), mean_repeat_purchases=("frequency", "mean"),
    share_never_returned=("frequency", lambda f: (f == 0).mean()),
    mean_spend_per_purchase=("monetary_value", lambda v: v[v > 0].mean()),
).loc[ACQUISITION].round(2)

# %% [markdown]
# ## Exercise 1 · New-customer value from a purchase curve (30 minutes)
#
# **Predict.** Rank the three acquisition channels by the 36-month margin value of a new
# customer, highest first. Is the top channel worth less than 1.5 times, 1.5 to 3 times, or more
# than 3 times the bottom one? Write both guesses down. (The table above is your evidence.)
#
# **Task.** Write `new_customer_value(cum_purchases, mean_spend, monthly_rate, margin,
# first_purchase=True)`:
#
# - `cum_purchases`: an array whose first axis is the month $k = 0, 1, \dots, 36$, holding
#   $N(k)$ (so `cum_purchases[0]` is 0); any further axes (posterior draws, channels) carry
#   through;
# - `mean_spend`: $\bar s$, a number or an array that broadcasts against those further axes;
# - return the CLV above, with the 1 for the acquisition purchase only when `first_purchase`
#   is True. With `cum_purchases` of shape `(37, draws, channels)` the result has shape
#   `(draws, channels)`.
#
# Hint: `np.diff(cum_purchases, axis=0)` gives $\Delta N_k$; reshape the discount factors to
# `(-1,) + (1,) * (cum_purchases.ndim - 1)` so they broadcast.

# %% tags=["exercise"]
def new_customer_value(cum_purchases, mean_spend, monthly_rate, margin, first_purchase=True):
    # TODO 1: margin * mean_spend * (first purchase + discounted monthly repeat purchases)
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def new_customer_value(cum_purchases, mean_spend, monthly_rate, margin, first_purchase=True):
    cum = np.asarray(cum_purchases, dtype=float)
    per_month = np.diff(cum, axis=0)  # expected repeat purchases in month k = 1..36
    k = np.arange(1, cum.shape[0]).reshape((-1,) + (1,) * (cum.ndim - 1))
    repeat = (per_month / (1 + monthly_rate) ** k).sum(axis=0)
    return margin * mean_spend * (repeat + (1.0 if first_purchase else 0.0))


# %% [markdown]
# The checkpoint runs before any model is fitted. It tries your function on a two-month toy
# curve, then on the purchase curve of the model that **generated** the data (a Pareto/NBD
# process with known parameters): there your function must reproduce the answer key exactly,
# both with and without the first purchase.

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    toy = new_customer_value(np.array([0.0, 1.0, 1.5]), 10.0, 0.10, 0.5)
    checks.close(toy, 0.5 * 10 * (1 + 1 / 1.1 + 0.5 / 1.1**2), rel=1e-9,
                 name="new_customer_value on the toy curve [0, 1, 1.5], spend 10, 10% a month,"
                      " margin 0.5 (expected 5 x (1 + 1/1.1 + 0.5/1.21))")
    toy_repeat = new_customer_value(np.array([0.0, 1.0, 1.5]), 10.0, 0.10, 0.5,
                                    first_purchase=False)
    checks.close(toy_repeat, 0.5 * 10 * (1 / 1.1 + 0.5 / 1.1**2), rel=1e-9,
                 name="new_customer_value without the first purchase (drop the 1)")
    many = new_customer_value(np.zeros((37, 4, 3)), np.ones((4, 1)), 0.01, 0.3)
    assert np.shape(many) == (4, 3), (
        f"cum_purchases of shape (37, 4, 3) should give shape (4, 3) (sum over months only);"
        f" got {np.shape(many)}. Sum over axis 0 and keep the other axes."
    )
    # The generator's curve: Pareto/NBD with the true parameters (mktstats.synth.channels)
    from mktstats.synth.channels import pnbd_expected_purchases_new

    rt = truth_all["retailer"]
    t_weeks = np.arange(cv["months"] + 1) * cv["weeks_per_month"]
    for c in ACQUISITION:
        a_c = rt["pnbd"]["alpha"] * np.exp(-rt["covariates"]["gamma_purchase"][c])
        b_c = rt["pnbd"]["beta"] * np.exp(-rt["covariates"]["gamma_dropout"][c])
        curve = pnbd_expected_purchases_new(t_weeks, rt["pnbd"]["r"], a_c, rt["pnbd"]["s"], b_c)
        key = answers["stage1_customer_value"]["by_acquisition_channel"][c]
        for first, field in [(True, "margin_including_first_purchase"),
                             (False, "margin_excluding_first_purchase")]:
            got = float(new_customer_value(curve, rt["population_mean_spend"],
                                           cv["monthly_discount_rate"], MARGIN, first))
            assert abs(got / key[field] - 1) < 1e-4, (
                f"On the generator's purchase curve your function gives ${got:,.2f} for a new"
                f" {c} customer ({'with' if first else 'without'} the first purchase), which"
                " is not the answer key's value. Usual causes: discounting month k by"
                " (1 + d)**(k - 1) or (1 + d)**k starting at k = 0, summing cum_purchases"
                " instead of its monthly differences, or forgetting the margin. Fix TODO 1, or"
                " run workshop.use_reference(1)."
            )

# %% [markdown]
# **Run.** Fit the two models once, on every purchase to date: the BG/NBD (Module 2) with the
# acquisition channel as a covariate on both the purchase rate and the dropout (Module 3), and
# the Gamma-Gamma spend model (Module 4) on customers with a repeat purchase. Both by MCMC with
# the nutpie sampler. The data were generated by a Pareto/NBD process; the BG/NBD approximates
# it and samples much faster (a Pareto/NBD MCMC fit took four minutes on Colab in the lead's
# test), which is why the next checkpoint allows 20%.
#
# The Gamma-Gamma priors are weakly informative half-normals: the package defaults pull the
# shape parameter $p$ far below its value in a fit of this size (measured in
# `mktstats.recovery`), while the mean spend is recovered either way.
#
# While it runs, read Stage 2's Predict prompt (Part C).

# %%
from pymc_extras.prior import Prior
from pymc_marketing import clv

COVARIATES = ["channel_social", "channel_referral"]  # 0/1; search is the reference channel
t_fit = time.time()
bg = clv.BetaGeoModel(model_config={"purchase_covariate_cols": COVARIATES,
                                    "dropout_covariate_cols": COVARIATES})
bg.fit(data=rfm[["customer_id", "frequency", "recency", "T", *COVARIATES]], method="mcmc",
       nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS, random_seed=SEED,
       progressbar=False)
gg = clv.GammaGammaModel(model_config={"p": Prior("HalfNormal", sigma=10),
                                       "q": Prior("HalfNormal", sigma=10),
                                       "v": Prior("HalfNormal", sigma=100)})
gg.fit(data=rfm.loc[rfm["frequency"] > 0, ["customer_id", "frequency", "monetary_value"]],
       method="mcmc", nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS,
       random_seed=SEED, progressbar=False)
stage1_seconds = time.time() - t_fit

# %% [markdown]
# Turn the fits into value draws: $N(k)$ for a new customer of each channel at every month end,
# for every posterior draw, and the mean spend $p v / (q - 1)$ per draw. Draw $i$ of the two
# models are paired (they are separate models, so any pairing is valid).

# %%
channel_rows = pd.DataFrame({"customer_id": ACQUISITION,
                             "channel_social": [0, 1, 0], "channel_referral": [0, 0, 1]})
cum_purchases = np.stack([
    np.asarray(bg.expected_purchases_new_customer(data=channel_rows, t=k * cv["weeks_per_month"])
               .transpose("chain", "draw", "customer_id")).reshape(-1, len(ACQUISITION))
    for k in range(cv["months"] + 1)
])  # (months + 1, draws, channels)
gg_post = gg.idata.posterior
spend_draws = np.asarray(gg_post["p"] * gg_post["v"] / (gg_post["q"] - 1)).reshape(-1)
value_draws = new_customer_value(cum_purchases, spend_draws[:, None],
                                 cv["monthly_discount_rate"], MARGIN)  # (draws, channels)
n_div1 = int(bg.idata["/sample_stats"]["diverging"].sum())
print(f"Stage 1 fits: {stage1_seconds:.1f} s for 2 chains x {DRAWS} draws each"
      f" ({'QUICK' if QUICK else 'FULL'}); BG/NBD divergences: {n_div1};"
      f" {value_draws.shape[0]} posterior draws of value per channel.")

# %% tags=["checkpoint"]
with workshop.checkpoint(1, label="1 · fitted values"):
    key1 = answers["stage1_customer_value"]
    tol1 = key1["checkpoint"]["rel"]  # 0.20: BG/NBD approximates the Pareto/NBD generator
    assert value_draws.shape == (cum_purchases.shape[1], len(ACQUISITION)), (
        f"Expected value draws of shape (draws, channels) = {(cum_purchases.shape[1], 3)};"
        f" got {value_draws.shape}. Check TODO 1 sums over axis 0 only."
    )
    for j, c in enumerate(ACQUISITION):
        est = value_draws[:, j].mean()
        rel = est / key1["by_acquisition_channel"][c]["margin_including_first_purchase"] - 1
        assert abs(rel) <= tol1, (
            f"The posterior mean value of a new {c} customer, ${est:,.2f}, is"
            f" {'above' if rel > 0 else 'below'} the true value by more than {tol1:.0%}. The"
            " fit is provided: check that TODO 1 passed its first checkpoint, that the margin"
            " is 0.30 and that the month is 30.4375 / 7 weeks. Then tell the instructor."
        )
    top = ACQUISITION[int(np.argmax(value_draws.mean(axis=0)))]
    assert top == key1["top_channel"], (
        f"Your top channel is {top}, not the true top channel. Look at the covariate"
        " coefficients: a channel's customers differ in how often they buy and how soon they"
        " leave. Then tell the instructor."
    )

# %% [markdown]
# Look at the order of the bars and how much the 94% intervals overlap: is the ranking you
# predicted clear, or could two channels swap?

# %%
value_hdi = np.array([checks.interval(value_draws[:, j], 0.94, "hdi")
                      for j in range(len(ACQUISITION))])
CAC_QUANTILE = 0.20  # risk rule from Module 5: pay at most the 20th percentile of value
stage1 = pd.DataFrame({
    "value_mean": value_draws.mean(axis=0),
    "hdi_low": value_hdi[:, 0], "hdi_high": value_hdi[:, 1],
    "cac_cap": np.quantile(value_draws, CAC_QUANTILE, axis=0),
    "value_without_first_purchase": new_customer_value(
        cum_purchases, spend_draws[:, None], cv["monthly_discount_rate"], MARGIN,
        first_purchase=False).mean(axis=0),
}, index=pd.Index(ACQUISITION, name="acquisition channel"))
fig, ax = plt.subplots(figsize=(7, 2.6))
pos = np.arange(len(ACQUISITION))
ax.hlines(pos, stage1["hdi_low"], stage1["hdi_high"], lw=7, alpha=0.45, label="94% HDI")
ax.plot(stage1["value_mean"], pos, "o", label="posterior mean")
ax.plot(stage1["cac_cap"], pos, "|", color="black", markersize=14, label="CAC cap (20% quantile)")
ax.set_yticks(pos, ACQUISITION)
ax.set_xlabel("36-month margin value of a new customer ($, first purchase included)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
show(fig)
stage1.round(2)

# %% [markdown]
# **Explain.** Compare with your ranking. Which part of the model makes the channels differ:
# how often their customers buy, how soon they stop, or how much they spend per purchase? Point
# to the table at the top of this part and to the model's covariates.
#
# <details><summary>Why this solution works</summary>
#
# The BG/NBD forecasts repeat purchases only; the acquisition purchase is added by hand (the
# 1 in the formula), which is why the function has a switch for it. Differencing the cumulative
# curve gives each month's purchases, so each month can get its own discount factor. The
# channels differ only through the covariates: the model lets a channel change both the
# purchase rate and the dropout, and spend per purchase is the same for everyone (the
# Gamma-Gamma model has no covariates). A channel whose customers buy more often and leave
# later is worth more. The CAC cap (customer acquisition cost) is the 20th percentile of value:
# paying at most that leaves an 80% posterior probability that a new customer is worth it.
# Stage 4 uses the value **without** the first purchase, because the marketing mix model's
# short-run sales already contain it. A longer horizon, a lower discount rate or a Pareto/NBD
# fit would change the numbers; the ranking comes from both the purchase rate and the
# dropout, which the covariates move in the same direction for each channel.
# </details>

# %% [markdown]
# # Part C · Stage 2: is search incremental?
#
# Last autumn the retailer **switched paid search off** in 8 of its 40 regions for 10 weeks:
# a *holdout* geo test. If search causes sales, those regions sold less than they would have.
# The question for the business: did search pay for itself? The question for Stage 3: the test
# is the evidence that calibrates the marketing mix model (MMM), so it has to be turned into the
# form the MMM accepts.
#
# A **lift test row** for PyMC-Marketing's `add_lift_test_measurements` describes one experiment
# on the national weekly scale: at weekly spend `x`, changing spend by `delta_x` changed weekly
# sales by `delta_y`, with standard error `sigma`. The test ran in regions holding a share $f$
# of national sales, for $W$ weeks, and each region's response is a scaled copy of the national
# one (the scenario's rule). So the national equivalent of the test is
#
# $$\Delta x = \frac{\text{spend change}}{f\,W}, \qquad
#   \Delta y = \frac{\text{incremental sales}}{f\,W}, \qquad
#   \sigma = \frac{\text{standard error of the incremental sales}}{f\,W},$$
#
# with $x$ the national weekly search spend before the test.
#
# The standard error comes from the placebos. A **placebo** pretends a control region was
# treated and estimates its "effect", which is pure noise. Each placebo's effect is measured
# **relative to its own level** (its gap divided by its synthetic sales), so large and small
# regions are comparable; its spread, times the treated regions' synthetic sales, is the
# standard error. A placebo that its synthetic control could not fit before the test (pre-period
# fit error, RMSPE, more than twice the median placebo's) says nothing about noise and is left
# out.

# %% [markdown]
# ## Exercise 2 · The lift test the MMM needs (30 minutes)
#
# **Predict.** The 8 regions saved \$31,200 of search spend. At a 30% margin, the cut paid
# for itself in short-run margin if the regions lost less than \$104,000 of sales
# (0.30 × \$104,000 = \$31,200). Did they lose more or less than \$104,000? And will the
# placebo p-value be below 0.1? Write both down.
#
# **Task.** Write `lift_test_row(readout, treated_share, test_weeks, base_weekly_spend,
# spend_change_total)` returning a one-row DataFrame with columns `channel` ("search"), `x`,
# `delta_x`, `delta_y`, `sigma`, from the formulas above. `readout` is `geo_readout`'s dict:
# `incremental`, `synthetic_post_total`, `placebo_relative_effects` and `placebo_pre_rmspe`.
# Take the standard deviation of the kept placebos' relative effects with `ddof=1`.

# %% tags=["exercise"]
def lift_test_row(readout, treated_share, test_weeks, base_weekly_spend, spend_change_total):
    # TODO 2: national weekly delta_x, delta_y and sigma (placebo spread x synthetic total)
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def lift_test_row(readout, treated_share, test_weeks, base_weekly_spend, spend_change_total):
    scale = treated_share * test_weeks  # national weeks the test stands for
    pre_fit = np.asarray(readout["placebo_pre_rmspe"])
    kept = pre_fit <= 2 * np.median(pre_fit)  # placebos whose synthetic control fitted
    rel = np.asarray(readout["placebo_relative_effects"])[kept]
    se_total = rel.std(ddof=1) * readout["synthetic_post_total"]
    return pd.DataFrame([{"channel": "search", "x": base_weekly_spend,
                          "delta_x": spend_change_total / scale,
                          "delta_y": readout["incremental"] / scale,
                          "sigma": se_total / scale}])


# %% [markdown]
# **Run.** The readout: Module 7's synthetic control for the treated regions' mean, and its 32
# placebos (a second or two). Then the treated regions' share of sales before the test, which
# the scenario says to compute from the panel.

# %%
readout = geo_readout(panel, gt["treated_geos"], gt["test_start"])
before = panel["date"] < pd.Timestamp(gt["test_start"])
sales_before = panel[before].groupby("geo")["sales"].sum()
treated_share = float(sales_before[gt["treated_geos"]].sum() / sales_before.sum())
print(f"Synthetic control: sales change in the treated regions over the test"
      f" {usd(readout['incremental'])} ({100 * readout['relative_effect']:+.1f}%);"
      f" placebo p-value {readout['p_value']:.3f} ({len(readout['placebo_ratios'])} placebos)")
print(f"Treated regions' share of sales before the test: {treated_share:.1%}")

# %% [markdown]
# Look at whether the synthetic control (dashed) tracks the treated regions before the shaded
# test window, and at the gap that opens inside it.

# %%
wide = panel.pivot(index="date", columns="geo", values="sales").sort_index()
treated_mean = wide[gt["treated_geos"]].mean(axis=1)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(wide.index, treated_mean / 1e3, label="treated regions (mean, actual)")
ax.plot(wide.index, (treated_mean - readout["gap"]) / 1e3, "--",
        label="synthetic control (counterfactual)")
ax.axvspan(pd.Timestamp(gt["test_start"]), pd.Timestamp(gt["test_end"]), color="grey",
           alpha=0.2, label="search switched off")
ax.set_ylabel("weekly sales per region\n($ thousand)")
ax.legend(loc="upper left", fontsize=8)
show(fig)

# %% [markdown]
# **Explain.** Compare with both predictions. Why does a cut, rather than a campaign that adds
# spend, measure what the MMM gets wrong about search (Module 9: search spend follows demand)?
# And why must the placebo spread be relative to each region's level? Point to the row and to
# the two placebos the rule drops (printed under the checkpoint).
#
# <details><summary>Why this solution works</summary>
#
# Dividing by $f W$ turns eight regions over ten weeks into "the whole country for one week",
# which is the scale of the MMM's weekly data; the scenario's rule that each region responds
# as a scaled copy of the national curve is what makes that division valid. Switching search
# off measures the whole contribution of the current spend, $r(x) - r(0)$: the level of the
# curve, which is exactly what a hidden demand shock inflates in the MMM. A test that doubles
# spend measures only the curve above $x$ and leaves the level free (measured on this scenario:
# it moved the calibrated search ROAS away from the truth). Raw dollar gaps of single regions
# are not comparable: a region's gap scales with its size, and the largest regions cannot be
# matched by a weighted average of smaller ones, so their placebo gaps are huge. Relative
# effects remove the size; dropping placebos that did not fit before the test removes the
# regions synthetic control cannot represent. Without that, `sigma` would be larger than the
# effect itself and the test would barely move the MMM.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    toy_readout = {"incremental": -1000.0, "synthetic_post_total": 20_000.0,
                   "placebo_relative_effects": np.array([0.01, -0.01, 0.02, -0.02, 0.9]),
                   "placebo_pre_rmspe": np.array([0.01, 0.01, 0.01, 0.01, 0.2])}
    toy_row = lift_test_row(toy_readout, 0.25, 10, 500.0, -1250.0)
    checks.columns(toy_row, ["channel", "x", "delta_x", "delta_y", "sigma"],
                   name="lift_test_row(...)")
    assert len(toy_row) == 1 and toy_row["channel"].iloc[0] == "search", (
        "Return one row with channel 'search'."
    )
    checks.close(toy_row["delta_x"].iloc[0], -1250 / 2.5, rel=1e-9,
                 name="toy delta_x (spend change / (share x weeks) = -1250 / 2.5)")
    checks.close(toy_row["delta_y"].iloc[0], -1000 / 2.5, rel=1e-9,
                 name="toy delta_y (incremental sales / (share x weeks))")
    checks.close(toy_row["sigma"].iloc[0], np.std([0.01, -0.01, 0.02, -0.02], ddof=1) * 20_000
                 / 2.5, rel=1e-9,
                 name="toy sigma (sd of the 4 kept placebo effects x 20,000 / 2.5; the fifth"
                      " placebo's pre-period error is 20 times the median, so it is dropped)")
    key2 = answers["stage2_geo_test"]
    rel_inc = readout["incremental"] / key2["incremental_sales"] - 1
    assert abs(rel_inc) <= key2["checkpoint"]["incremental_sales"]["rel"], (
        f"The synthetic control's sales change, ${readout['incremental']:,.0f}, is more than 15%"
        " from the truth. The readout is provided: check that the panel and the treated regions"
        " were not changed, then tell the instructor."
    )
    assert readout["p_value"] <= key2["checkpoint"]["placebo_p_value_max"], (
        f"The placebo p-value is {readout['p_value']:.3f}, above 0.1. The readout is provided:"
        " tell the instructor."
    )
    lift_row = lift_test_row(readout, treated_share, gt["test_weeks"],
                             gt["base_weekly_spend_national"], gt["spend_change_total"])
    row, key_row = lift_row.iloc[0], key2["national_equivalent_lift_test"]
    checks.close(row["x"], gt["base_weekly_spend_national"], rel=1e-9,
                 name="x (national weekly search spend before the test)")
    checks.close(row["delta_x"], key_row["delta_x"], rel=0.01,
                 name="delta_x (spend change / (treated share x test weeks); negative for a cut)")
    assert row["delta_y"] < 0 and abs(row["delta_y"] / key_row["delta_y"] - 1) <= 0.15, (
        f"delta_y is ${row['delta_y']:,.0f} a week; it should be the sales change divided by"
        " (treated share x test weeks), negative for a cut, within 15% of the true national"
        " lift. Check the share (about 0.17) and the number of weeks (10)."
    )
    assert 0 < row["sigma"] <= 0.25 * abs(row["delta_y"]), (
        f"sigma is ${row['sigma']:,.0f} a week against a lift of ${abs(row['delta_y']):,.0f}."
        " The placebo spread should be a small fraction of the effect (p = 0.03). Check that"
        " you dropped placebos whose pre-period error is more than twice the median, used the"
        " RELATIVE effects, multiplied by synthetic_post_total and divided by share x weeks."
        " Fix TODO 2, or run workshop.use_reference(2)."
    )

# %%
dropped = readout["placebo_pre_rmspe"] > 2 * np.median(readout["placebo_pre_rmspe"])
z94 = NormalDist().inv_cdf(0.97)  # 94% normal interval
se_total = lift_row["sigma"].iloc[0] * treated_share * gt["test_weeks"]
inc = readout["incremental"]
inc_lo, inc_hi = inc - z94 * se_total, inc + z94 * se_total
roas_cut = inc / gt["spend_change_total"]  # sales lost per dollar saved
stage2 = {"incremental": inc, "low": inc_lo, "high": inc_hi, "p_value": readout["p_value"],
          "lift_pct": 100 * readout["relative_effect"], "roas": roas_cut,
          "roas_low": inc_hi / gt["spend_change_total"],
          "roas_high": inc_lo / gt["spend_change_total"]}
print(f"Placebos dropped (poor pre-period fit): {int(dropped.sum())} of {dropped.size};"
      f" kept placebos' relative effects: sd {np.std(readout['placebo_relative_effects'][~dropped], ddof=1):.4f}")
print(f"Sales lost in the treated regions: ${-inc:,.0f} (94% interval ${-inc_hi:,.0f} to"
      f" ${-inc_lo:,.0f}, normal approximation from the placebo spread) for"
      f" ${-gt['spend_change_total']:,.0f} of search spend saved")
print(f"Sales per search dollar in those regions: {roas_cut:.2f} (94% interval"
      f" {stage2['roas_low']:.2f} to {stage2['roas_high']:.2f}); break-even at"
      f" {MARGIN:.0%} margin: {1 / MARGIN:.2f}")
lift_row.round(1)

# %% [markdown]
# **Is search incremental?** Read the printed lines: did the treated regions lose sales, and
# does the placebo p-value say that few placebo regions moved as much? Then compare the sales
# per search dollar with the break-even of 1 ÷ 0.30 ≈ 3.33: in short-run margin alone, did a
# dollar of search in those regions pay back? Stage 4 adds what short-run margin misses: the
# future purchases of the customers search acquires.
#
# **Lunch.** If checkpoint 2 has passed, you can run the two fits of Part D before you leave;
# keep the tab open.

# %% [markdown]
# # Part D · Stage 3: calibrate the marketing mix model with the geo test
#
# The MMM (Module 8) explains weekly national sales by a baseline plus one adstocked,
# saturating term per media channel. Its **ROAS** (return on ad spend) for a channel is the
# sales its spend caused over the data window divided by that spend. In this retailer's data,
# search spend rises with demand that the model cannot see (Module 9), so the model can credit
# search for sales the demand caused. **Calibration** adds Stage 2's lift test to the model's
# likelihood: parameter values whose curve disagrees with the experiment become unlikely.
#
# **Run.** First the model **without** the test, for comparison: Module 8's specification
# (8-week geometric adstock, logistic saturation, price, holiday and trend as controls, two
# yearly Fourier terms) with PyMC-Marketing's default priors.

# %%
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

X = weekly.drop(columns=["y"])
y = weekly["y"]


def make_mmm():
    """Module 8's model, default priors."""
    return MMM(date_column="date_week", channel_columns=CHANNELS, target_column="y",
               control_columns=["price_index", "holiday", "t"],
               adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(),
               yearly_seasonality=2)


t_fit = time.time()
mmm_raw = make_mmm()
mmm_raw.fit(X, y, nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS, target_accept=0.9,
            random_seed=SEED, progressbar=False)
fit_raw_seconds = time.time() - t_fit

# %%
W0, W1 = scenario["mmm"]["window_start"], scenario["mmm"]["window_end"]
roas_raw = mmm_raw.incrementality.contribution_over_spend(frequency="all_time", start_date=W0,
                                                          end_date=W1)
search_raw = roas_raw.sel(channel="search").values.ravel()
lo_raw, hi_raw = checks.interval(search_raw, 0.94, "hdi")
print(f"Uncalibrated fit: {fit_raw_seconds:.1f} s; search ROAS {search_raw.mean():.2f}"
      f" (94% HDI {lo_raw:.2f} to {hi_raw:.2f})")

# %% [markdown]
# ## Exercise 3 · ROAS with intervals, and the break-even test (25 minutes)
#
# **Predict.** The uncalibrated model's search ROAS is printed above; Stage 2 printed the sales
# per search dollar the geo test measured. After calibration, will search's ROAS move up or
# down, and will its 94% HDI get narrower or wider? Write both down.
#
# **Task.** Write `roas_summary(roas_draws, break_even, prob=0.94)` returning a DataFrame
# indexed by channel with columns `mean`, `hdi_low`, `hdi_high` (the `prob` HDI from
# `checks.interval(draws, prob, "hdi")`) and `p_above_break_even`, the share of draws above
# `break_even`. `roas_draws.sel(channel=c).values.ravel()` gives one channel's draws; the
# channels are `roas_draws.coords["channel"].values`.

# %% tags=["exercise"]
def roas_summary(roas_draws, break_even, prob=0.94):
    # TODO 3: one row per channel: mean, hdi_low, hdi_high, p_above_break_even
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def roas_summary(roas_draws, break_even, prob=0.94):
    rows = {}
    for c in roas_draws.coords["channel"].values:
        d = roas_draws.sel(channel=c).values.ravel()
        lo, hi = checks.interval(d, prob, "hdi")
        rows[str(c)] = {"mean": d.mean(), "hdi_low": lo, "hdi_high": hi,
                        "p_above_break_even": float((d > break_even).mean())}
    return pd.DataFrame(rows).T.rename_axis("channel")


# %% [markdown]
# **Run.** The calibrated model: the same specification, built, then Stage 2's row added to
# the likelihood, then fitted.

# %%
t_fit = time.time()
mmm = make_mmm()
mmm.build_model(X, y)
mmm.add_lift_test_measurements(lift_row[["channel", "x", "delta_x", "delta_y", "sigma"]])
mmm.fit(X, y, nuts_sampler="nutpie", chains=2, draws=DRAWS, tune=DRAWS, target_accept=0.9,
        random_seed=SEED, progressbar=False)
fit_cal_seconds = time.time() - t_fit

# %%
roas_cal = mmm.incrementality.contribution_over_spend(frequency="all_time", start_date=W0,
                                                      end_date=W1)
PARAMS = ["adstock_alpha", "saturation_lam", "saturation_beta"]
fit_checks = {}
for name, model in [("uncalibrated", mmm_raw), ("calibrated", mmm)]:
    rhat = az.rhat(model.idata, var_names=PARAMS)
    fit_checks[name] = {"divergences": int(model.idata["/sample_stats"]["diverging"].sum()),
                        "max_rhat": max(float(rhat[v].max()) for v in rhat.data_vars)}
print(f"Calibrated fit: {fit_cal_seconds:.1f} s. Fit checks (want 0 divergences, R-hat <= 1.01):")
print(pd.DataFrame(fit_checks).T.round(3).to_string())
for name, f in fit_checks.items():
    if f["divergences"] > 0 or f["max_rhat"] > 1.01:
        print(f"The {name} fit misses a check: its chains do not fully agree on every"
              f" parameter with {DRAWS} draws each. Read its intervals as rough"
              + (" (Stage 4 uses this fit: say so on slide 5)." if name == "calibrated" else
                 "; without the test, the data cannot pin down search's curve."))

# %% [markdown]
# **Explain.** Compare with your prediction, using the table and plot below. The lift test is
# one experiment on one channel; why did search's interval change so much more than the
# others'? And where must the sales go that the model no longer gives to search? Point to the
# other channels' rows: if they barely move, what is left to explain those sales?
#
# <details><summary>Why this solution works</summary>
#
# Each row summarizes one channel's posterior draws: the mean, the shortest interval holding
# 94% of the draws, and the posterior probability that the channel pays back in short-run
# margin (ROAS above $1/m$). Without the test, the model sees search spend and sales rise
# together and cannot tell how much of that is demand, so search's ROAS is high and its interval
# wide. The switch-off test pins the level of search's curve at the current spend, so the
# posterior keeps only curves that agree with it: search's ROAS falls and its interval narrows.
# The other channels change little because the test says nothing about them, so the sales
# search no longer explains go mostly to the baseline (intercept, trend, season, controls),
# where the hidden demand belonged all along. One test fixes the level at one spend, not the
# shape of the curve above it: Stage 4 leans on that shape.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    import xarray as xr

    toy_draws = xr.DataArray(np.stack([np.arange(1.0, 101.0), np.full(100, 2.0)], axis=-1),
                             dims=("draw", "channel"), coords={"channel": ["a", "b"]})
    toy_table = roas_summary(toy_draws, 50.0)
    checks.columns(toy_table.reset_index(), ["channel", "mean", "hdi_low", "hdi_high",
                                             "p_above_break_even"], name="roas_summary(...)")
    assert sorted(toy_table.index) == ["a", "b"], (
        f"Expected one row per channel, indexed by channel; got {list(toy_table.index)}."
    )
    checks.close(toy_table.loc["a", "mean"], 50.5, rel=1e-9, name="the toy mean of 1..100")
    checks.close(toy_table.loc["a", "p_above_break_even"], 0.5, abs=1e-9,
                 name="the toy share of 1..100 above 50 (strictly above)")
    lo_a, hi_a = checks.interval(np.arange(1.0, 101.0), 0.94, "hdi")
    assert (toy_table.loc["a", "hdi_low"], toy_table.loc["a", "hdi_high"]) == (lo_a, hi_a), (
        "Use checks.interval(draws, prob, 'hdi') for the interval: a 94% HDI, not ArviZ's"
        " default 89% equal-tailed interval."
    )
    stage3 = roas_summary(roas_cal, 1 / MARGIN)
    stage3_raw = roas_summary(roas_raw, 1 / MARGIN)
    true_roas = answers["stage3_mmm"]["true_roas"]
    k_in, _ = answers["stage3_mmm"]["checkpoint"]["k_of_K_inside"]
    checks.k_of_K_in_interval(true_roas, {c: roas_cal.sel(channel=c).values for c in CHANNELS},
                              k=k_in, prob=0.94, kind="hdi", name="calibrated channel ROAS")

# %% [markdown]
# Look at search: how far its bar moves and how much shorter it gets, against the dashed
# break-even line.

# %%
fig, ax = plt.subplots(figsize=(7, 3))
pos = np.arange(len(CHANNELS))
for shift, table, label in [(0.15, stage3_raw, "uncalibrated"), (-0.15, stage3, "calibrated")]:
    t_ = table.loc[CHANNELS]
    ax.hlines(pos + shift, t_["hdi_low"], t_["hdi_high"], lw=6, alpha=0.5,
              label=f"{label}, 94% HDI", color="C0" if shift > 0 else "C1")
    ax.plot(t_["mean"], pos + shift, "o" if shift > 0 else "s", color="C0" if shift > 0 else "C1")
ax.axvline(1 / MARGIN, color="black", linestyle="--", lw=1, label="break-even (1 / margin)")
ax.set_yticks(pos, CHANNELS)
ax.set_xlabel("ROAS (sales $ per $ of spend, MMM window)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
show(fig)
pd.concat({"uncalibrated": stage3_raw, "calibrated": stage3}, axis=1).loc[CHANNELS].round(2)

# %% [markdown]
# # Part E · Stage 4: next quarter's budget
#
# Module 10's long-run objective values a steady weekly plan $s$ (weekly spend $s_c$ on each
# media channel) as
#
# $$V(s) = m \sum_c r_c(s_c) + \sum_c s_c\, n_c\, \text{CLV}_c,$$
#
# where $r_c$ is channel $c$'s response curve (weekly sales, from the calibrated MMM), $n_c$
# the new customers per dollar (1 ÷ the cost per new customer, from the scenario) and
# $\text{CLV}_c$ the future margin of a new customer that media channel brings: Stage 1's
# value by acquisition channel **without the first purchase** (the MMM's short-run sales
# already count it), averaged with the scenario's media-to-acquisition shares. The bounds and
# the budget are the scenario's.
#
# **Run.** The inputs: the posterior-mean response curves from Part D and the value of each
# media channel's new customer, per posterior draw, from Stage 1.

# %%
post = mmm.idata.posterior.to_dataset().stack(sample=("chain", "draw"))
beta_draws = post["saturation_beta"].sel(channel=CHANNELS).transpose("sample", "channel").values
lam_draws = post["saturation_lam"].sel(channel=CHANNELS).transpose("sample", "channel").values
x_scale = mmm.idata.constant_data["channel_scale"].sel(channel=CHANNELS).values
y_scale = float(mmm.idata.constant_data["target_scale"])
mean_curves = {c: (lambda s, j=j: float(np.mean(response_curve(
    s, beta_draws[:, j], lam_draws[:, j], x_scale[j], y_scale)))) for j, c in enumerate(CHANNELS)}

repeat_value_draws = new_customer_value(cum_purchases, spend_draws[:, None],
                                        cv["monthly_discount_rate"], MARGIN,
                                        first_purchase=False)  # (draws, acquisition channels)
shares = np.array([[link["media_to_acquisition"][m][a] for a in ACQUISITION] for m in CHANNELS])
clv_media_draws = repeat_value_draws @ shares.T  # (draws, media channels)
clv_media = dict(zip(CHANNELS, clv_media_draws.mean(axis=0), strict=True))
ncpd = {c: link["new_customers_per_dollar"][c] for c in CHANNELS}
BUDGET = float(bud["weekly_budget"])
BOUNDS = {c: tuple(bud["bounds"][c]) for c in CHANNELS}
CURRENT = pd.Series(bud["current_allocation"])[CHANNELS]
pd.DataFrame({"CLV of a new customer ($ margin, no first purchase)": clv_media,
              "new customers per $": ncpd,
              "future margin per $ of spend": {c: ncpd[c] * clv_media[c] for c in CHANNELS},
              "marginal ROAS at the current plan": {
                  c: (mean_curves[c](CURRENT[c] + 1) - mean_curves[c](CURRENT[c] - 1)) / 2
                  for c in CHANNELS}}).loc[CHANNELS].round(3)

# %% [markdown]
# ## Exercise 4 · A budget that counts new customers' future margin (20 minutes)
#
# **Predict.** Compared with the plan that maximizes short-run margin only, which channel gains
# the most budget once new customers' future margin counts: tv, search, social or display? The
# table above has what you need: the future margin per dollar adds to $m$ times the marginal
# ROAS.
#
# **Task.** Write `recommend_budget(curves, clv_media, new_customers_per_dollar, bounds,
# weekly_total, margin)`: build one function per channel, $s \mapsto m\,r_c(s) + s\,n_c\,
# \text{CLV}_c$, and pass them to the toolkit's `allocate(weekly_total, bounds, ...)`. Return
# its `pd.Series` of weekly spend. (Inside a dict comprehension, bind the channel with
# `lambda s, c=c: ...`, or every function uses the last channel.)

# %% tags=["exercise"]
def recommend_budget(curves, clv_media, new_customers_per_dollar, bounds, weekly_total, margin):
    # TODO 4: long-run value per channel (margin x curve + spend x customers per $ x CLV), allocate
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def recommend_budget(curves, clv_media, new_customers_per_dollar, bounds, weekly_total, margin):
    long_run = {c: (lambda s, c=c: margin * curves[c](s)
                    + s * new_customers_per_dollar[c] * clv_media[c]) for c in curves}
    return allocate(weekly_total, bounds, long_run)


# %% [markdown]
# **Explain.** Compare with your prediction, using the plans table below the checkpoint. The
# future-margin term is a constant per dollar; why can a small constant move thousands of
# dollars a week between two channels? Then look at search: the plans put far more on it than
# today. What about search's curve does Stage 2's test not tell us? Point to the marginal ROAS
# interval printed on slide 4 of the brief.
#
# <details><summary>Why this solution works</summary>
#
# Each channel's long-run value is concave (short-run margin, which saturates) plus linear (new
# customers' future margin), so the problem has one optimum and SLSQP finds it; the channels
# inside their bounds end up with equal marginal long-run value. Near the optimum the
# short-run curves of tv and search have almost the same slope, so a few cents per dollar of
# future margin shift a lot of money toward the channel whose new customers are worth more
# (tv brings referral customers). The checkpoint runs your function on the **true** curves and
# values, where it must land on the true long-run optimum: a plan that ignores future margin
# misses it by more than the 5% tolerance. On the model's curves the plan is only as good as the
# curves: the switch-off test fixed search's curve at its current spend, not its slope above
# it, so the model is unsure how fast search saturates, and where the curve is unknown the
# posterior mean bends less than a real saturating curve. The risk-averse plan maximizes the
# value in the worst 10% of draws; when it barely moves, the uncertainty is shared by every
# allowed plan and no split within the bounds removes it. A second test that raises search
# spend would measure the slope the plan depends on.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    from mktstats.synth.mmm import steady_state_response

    key4 = answers["stage4_allocation"]
    tch = truth_all["mmm"]["channels"]
    true_curves = {c: (lambda s, c=c: float(steady_state_response(
        s, tch[c]["saturation_beta_sales_units"], tch[c]["saturation_lam"],
        tch[c]["channel_scale"]))) for c in CHANNELS}
    true_clv = key4["clv_margin_by_media_channel"]

    def true_long_run_value(plan):
        """The true weekly long-run value of a plan (the answer key's objective)."""
        return sum(MARGIN * true_curves[c](plan[c])
                   + plan[c] * key4["long_run_value_per_dollar_of_new_customers"][c]
                   for c in CHANNELS)

    plan_true = recommend_budget(true_curves, true_clv, ncpd, BOUNDS, BUDGET, MARGIN)
    assert isinstance(plan_true, pd.Series) and sorted(plan_true.index) == sorted(CHANNELS), (
        f"recommend_budget should return allocate's pd.Series indexed by channel; got"
        f" {type(plan_true).__name__}."
    )
    optimum = pd.Series(key4["optimal_allocation"])[CHANNELS]
    off = (plan_true[CHANNELS] - optimum).abs() / BUDGET
    tol_spend = key4["checkpoint"]["spend_by_channel"]["abs_share_of_budget"]  # 0.05
    assert off.max() <= tol_spend, (
        f"On the true curves and true customer values your plan puts"
        f" ${plan_true[off.idxmax()]:,.0f} a week on {off.idxmax()}, {off.max():.0%} of the"
        f" budget away from the true long-run optimum (allowed {tol_spend:.0%}). Usual causes:"
        " the future-margin term is missing or not multiplied by spend, the margin is applied"
        " to it as well, or every lambda uses the last channel (bind c=c). Fix TODO 4, or run"
        " workshop.use_reference(4)."
    )
    share_of_best = true_long_run_value(plan_true) / key4["weekly_value_optimal"]
    assert share_of_best >= 1 - key4["checkpoint"]["value"]["rel"], (
        f"Your plan reaches {share_of_best:.1%} of the true optimal long-run value (allowed:"
        " at least 97%)."
    )
    plan_long = recommend_budget(mean_curves, clv_media, ncpd, BOUNDS, BUDGET, MARGIN)[CHANNELS]
    checks.close(plan_long.sum(), BUDGET, rel=1e-6, name="the plan's total weekly spend")
    for c in CHANNELS:
        assert BOUNDS[c][0] - 1e-6 * BUDGET <= plan_long[c] <= BOUNDS[c][1] + 1e-6 * BUDGET, (
            f"{c} gets ${plan_long[c]:,.0f}, outside its bounds; pass bounds to allocate."
        )

# %% [markdown]
# **Run.** Four plans, each scored in every posterior draw: weekly long-run value (margin on
# short-run sales plus new customers' future margin), its gain over the current plan with a 94%
# HDI, and the probability that it beats the current plan. Draw $i$ pairs MMM draw $i$ with
# Stage 1 draw $i$. The *short-run* plan ignores future margin; the *risk-averse* plan
# maximizes the 10% quantile of long-run value (SLSQP started from the long-run plan, as in
# Module 10).

# %%
ncpd_vec = np.array([ncpd[c] for c in CHANNELS])


def long_run_value_draws(plan):
    """Weekly long-run value of a plan in every posterior draw ($ margin)."""
    x = np.array([plan[c] for c in CHANNELS], dtype=float)
    sales = response_curve(x, beta_draws, lam_draws, x_scale, y_scale)  # (draws, channels)
    return MARGIN * sales.sum(axis=1) + (x * ncpd_vec * clv_media_draws).sum(axis=1)


def sales_draws(plan):
    """Weekly media-driven sales of a plan in every posterior draw ($)."""
    x = np.array([plan[c] for c in CHANNELS], dtype=float)
    return response_curve(x, beta_draws, lam_draws, x_scale, y_scale).sum(axis=1)


def marginal_roas_draws(spend, c):
    """Posterior draws of channel c's marginal ROAS at a weekly spend (central difference)."""
    j = CHANNELS.index(c)
    up, down = (response_curve(spend + e, beta_draws[:, j], lam_draws[:, j], x_scale[j], y_scale)
                for e in (1.0, -1.0))
    return (up - down) / 2


plan_short = recommend_budget(mean_curves, {c: 0.0 for c in CHANNELS}, ncpd, BOUNDS, BUDGET,
                              MARGIN)[CHANNELS]
t_risk = time.time()
with threadpool_limits(1, "blas"):
    risk_fit = minimize(lambda s: -np.quantile(long_run_value_draws(dict(zip(CHANNELS, s,
                                                                             strict=True))), 0.10),
                        x0=plan_long.to_numpy(), method="SLSQP",
                        bounds=[BOUNDS[c] for c in CHANNELS],
                        constraints=[{"type": "eq", "fun": lambda s: s.sum() - BUDGET}])
plan_risk = pd.Series(risk_fit.x, index=CHANNELS)
plans = {"current": CURRENT, "short-run": plan_short, "long-run": plan_long,
         "risk-averse": plan_risk}
value_current = long_run_value_draws(CURRENT)
rows = {}
for name, plan in plans.items():
    v = long_run_value_draws(plan)
    gain = v - value_current
    g_lo, g_hi = checks.interval(gain, 0.94, "hdi") if name != "current" else (0.0, 0.0)
    s_gain = sales_draws(plan) - sales_draws(CURRENT)
    rows[name] = {**{c: plan[c] for c in CHANNELS}, "weekly value": v.mean(),
                  "gain vs current": gain.mean(), "gain HDI low": g_lo, "gain HDI high": g_hi,
                  "P(beats current)": float((gain > 0).mean()) if name != "current" else np.nan,
                  "10% quantile of value": np.quantile(v, 0.10),
                  "extra weekly sales": s_gain.mean()}
stage4 = pd.DataFrame(rows).T
print(f"Risk-averse plan: SLSQP, {risk_fit.nit} iterations, {time.time() - t_risk:.1f} s"
      f" ({risk_fit.message}). Values are weekly $ of margin; 94% HDIs.")
stage4.round(2)

# %% [markdown]
# Look at where each plan's weekly gain over the current plan lies across the posterior draws,
# and whether any of them reaches below zero.

# %%
fig, ax = plt.subplots(figsize=(8, 3))
for (name, plan), style in zip([(k, plans[k]) for k in ("short-run", "long-run", "risk-averse")],
                               ["-", "--", ":"], strict=True):
    gain = long_run_value_draws(plan) - value_current
    hist, edges = np.histogram(gain / 1e3, bins=40)
    ax.step(edges[:-1], hist, where="post", linestyle=style, label=name)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("weekly long-run value gain over the current plan ($ thousand of margin), per draw")
ax.set_ylabel("draws")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
show(fig)

# %% [markdown]
# # Part F · Stage 5: whom to send the retention offer
#
# The retailer e-mailed an offer to a random half of 20,000 customers (Module 11). The
# **uplift**, or CATE (conditional average treatment effect), of a customer is how much more
# they spend *because of* the offer. The offer pays for itself for a customer when
# margin × uplift > offer cost, that is, when the uplift exceeds \$0.75 ÷ 0.30 = \$2.50 of
# spend. Customers with `split == "train"` fit the model; customers with `split == "test"` are
# **held out**: every score below uses only them.
#
# **Run.** A causal forest (EconML's `CausalForestDML`, Module 11) on the training half, then
# its estimated uplift for every held-out customer.

# %%
from econml.dml import CausalForestDML
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

train, test = email[email["split"] == "train"], email[email["split"] == "test"]
t_fit = time.time()
forest = CausalForestDML(model_y=HistGradientBoostingRegressor(random_state=SEED),
                         model_t=HistGradientBoostingClassifier(random_state=SEED),
                         discrete_treatment=True, n_estimators=N_TREES, random_state=SEED)
# EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.
forest.fit(train["spend"].to_numpy(), train["treatment"].to_numpy(), X=features(train).to_numpy())
cate_hat = pd.Series(forest.effect(features(test).to_numpy()), index=test.index)
print(f"Causal forest: {N_TREES} trees in {time.time() - t_fit:.1f} s;"
      f" {len(train):,} training and {len(test):,} held-out customers."
      f" Estimated uplift: mean ${cate_hat.mean():.2f}, middle 80% ${cate_hat.quantile(0.1):.2f}"
      f" to ${cate_hat.quantile(0.9):.2f}")

# %% [markdown]
# ## Exercise 5 · The targeting rule (20 minutes)
#
# **Predict.** The rule sends the offer when the estimated uplift exceeds \$2.50. What share of
# the held-out customers will it send it to: under 20%, 20 to 50%, or over 50%? (The printed
# mean and middle 80% of the estimates are your scale.)
#
# **Task.** Write `retention_targets(cate_hat, margin, offer_cost)` returning a boolean
# `pd.Series` with the same index as `cate_hat`: True where margin × uplift > offer cost
# (strictly greater: at equality the offer only breaks even).

# %% tags=["exercise"]
def retention_targets(cate_hat, margin, offer_cost):
    # TODO 5: True where margin x estimated uplift > offer cost
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def retention_targets(cate_hat, margin, offer_cost):
    return pd.Series(margin * np.asarray(cate_hat, dtype=float) > offer_cost,
                     index=getattr(cate_hat, "index", None))


# %% [markdown]
# **Explain.** Compare with your prediction. The rule ignores how much a customer spends
# without the offer; why is that right for this decision, and which Module 2 quantity would be
# the wrong thing to target on? Point to the held-out profit table below the checkpoint.
#
# <details><summary>Why this solution works</summary>
#
# The offer's cost is paid for every customer who gets it, and it earns only the margin on the
# spend it causes, so the rule compares margin × uplift with the cost, customer by customer.
# What a customer would spend anyway does not enter: a loyal customer likely to buy (high
# P(alive)) may have zero uplift, and mailing them only costs money. The checkpoint scores the
# rule with the true effects of the held-out customers, which a real analysis never has; the
# table below estimates the same profit from the experiment's outcomes alone, as you would on
# real data, so its interval is wide. A noisier uplift model, a higher offer cost or a lower
# margin would shrink the list.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    toy_send = retention_targets(pd.Series([0.0, 2.0, 2.5, 3.0, 10.0], index=list("abcde")),
                                 0.30, 0.75)
    assert isinstance(toy_send, pd.Series) and list(toy_send.index) == list("abcde"), (
        "Return a pd.Series with the same index as cate_hat."
    )
    assert toy_send.tolist() == [False, False, False, True, True], (
        f"For uplifts [0, 2, 2.5, 3, 10], margin 0.30 and cost 0.75 the rule should send to the"
        f" last two only (0.30 x 2.5 = 0.75 only breaks even); got {toy_send.tolist()}. Compare"
        " margin x uplift with the cost, strictly greater."
    )
    send = retention_targets(cate_hat, offer["margin"], offer["offer_cost"]).astype(bool)
    key5 = answers["stage5_targeting"]["test_split"]
    true_gain = offer["margin"] * test["true_cate"].to_numpy() - offer["offer_cost"]
    rule_value = float((send.to_numpy() * true_gain).mean())
    assert 0 < send.mean() < 1, (
        f"The rule sends the offer to {send.mean():.0%} of customers; it should pick some, not"
        " all or none. Check the margin (0.30) and the cost (0.75)."
    )
    assert rule_value > 0 and rule_value >= key5["treat_all"], (
        f"With the true effects, your rule earns ${rule_value:.3f} per held-out customer, less"
        " than mailing everyone. Check that you compare margin x uplift with the cost, not"
        " uplift with the cost. Fix TODO 5, or run workshop.use_reference(5)."
    )

# %%
y_test, t_test = test["spend"].to_numpy(), test["treatment"].to_numpy()
policies = {"send to the rule's list": send.to_numpy(),
            "send to everyone": np.ones(len(test), dtype=bool),
            "send to no one": np.zeros(len(test), dtype=bool)}
stage5 = pd.DataFrame({name: dict(zip(["profit per customer", "low", "high"],
                                      targeting_profit(s, y_test, t_test, offer["margin"],
                                                       offer["offer_cost"]), strict=True))
                       for name, s in policies.items()}).T
stage5["share sent"] = [s.mean() for s in policies.values()]
print(f"Held-out customers: {len(test):,}; profit per customer in $, 94% interval (normal"
      " approximation from the experiment's outcomes):")
stage5.round(3)

# %% [markdown]
# # Part G · The five-slide brief (25 minutes)
#
# The next cell collects every number your slides need from the stages above (nothing typed by
# hand) and prints a five-slide outline. Copy it into your slide tool, then write each slide's
# headline sentence yourselves. The rubric (`briefs/12-capstone-rubric.md`) scores it.

# %%
WEEKS = int(bud["planning_weeks"])
pick_name = ("long-run" if stage4.loc["long-run", "10% quantile of value"]
             >= stage4.loc["current", "10% quantile of value"] else "risk-averse")
pick = stage4.loc[pick_name]
moved = plans[pick_name] - CURRENT
mroas_hdi = {c: checks.interval(marginal_roas_draws(plans[pick_name][c], c), 0.94, "hdi")
             for c in CHANNELS}
widest = max(CHANNELS, key=lambda c: mroas_hdi[c][1] - mroas_hdi[c][0])
reference_stages = sorted(getattr(workshop, "chosen", set()))
brief = {
    "plan": pick_name,
    "budget": {c: float(plans[pick_name][c]) for c in CHANNELS},
    "weekly_gain": (pick["gain vs current"], pick["gain HDI low"], pick["gain HDI high"]),
    "p_beats_current": pick["P(beats current)"],
    "value": stage1[["value_mean", "hdi_low", "hdi_high", "cac_cap"]].to_dict("index"),
    "geo": stage2,
    "roas": {"uncalibrated": stage3_raw.loc[CHANNELS].to_dict("index"),
             "calibrated": stage3.loc[CHANNELS].to_dict("index")},
    "offer": {"share": float(send.mean()), "n": int(send.sum()), "n_test": len(test),
              **stage5.loc["send to the rule's list", ["profit per customer", "low", "high"]],
              "all": stage5.loc["send to everyone", "profit per customer"]},
    "widest_marginal": widest,
    "mode": "QUICK" if QUICK else "FULL",
}
lines = [
    "SLIDE 1 · The recommendation",
    "  Weekly budget: " + ", ".join(f"{c} ${v:,.0f}" for c, v in brief["budget"].items())
    + f" (the {pick_name} plan; current: "
    + ", ".join(f"{c} ${CURRENT[c]:,.0f}" for c in CHANNELS) + ")",
    f"  Expected gain over the current plan: ${brief['weekly_gain'][0]:,.0f} of margin a week"
    f" (94% HDI ${brief['weekly_gain'][1]:,.0f} to ${brief['weekly_gain'][2]:,.0f}),"
    f" ${WEEKS * brief['weekly_gain'][0]:,.0f} over the {WEEKS}-week quarter;"
    f" P(beats current) = {brief['p_beats_current']:.2f}",
    f"  Retention offer: send to {brief['offer']['share']:.0%} of customers"
    f" ({brief['offer']['n']:,} of {brief['offer']['n_test']:,} held out), estimated"
    f" ${brief['offer']['profit per customer']:.2f} per customer (94% interval"
    f" ${brief['offer']['low']:.2f} to ${brief['offer']['high']:.2f}) vs"
    f" ${brief['offer']['all']:.2f} for mailing everyone",
    "SLIDE 2 · What a new customer is worth (36 months, 1% a month, margin, first purchase in)",
    *[f"  {c}: ${v['value_mean']:,.0f} (94% HDI ${v['hdi_low']:,.0f} to ${v['hdi_high']:,.0f});"
      f" CAC cap at the 20th percentile ${v['cac_cap']:,.0f}" for c, v in brief["value"].items()],
    "SLIDE 3 · What search caused (geo test) and what it changed in the MMM",
    f"  Switching search off in {len(gt['treated_geos'])} of 40 regions for {gt['test_weeks']}"
    f" weeks lost ${-stage2['incremental']:,.0f} of sales (94% interval ${-stage2['high']:,.0f}"
    f" to ${-stage2['low']:,.0f}; placebo p = {stage2['p_value']:.2f}) for"
    f" ${-gt['spend_change_total']:,.0f} saved: {stage2['roas']:.2f} sales per $ (break-even"
    f" {1 / MARGIN:.2f})",
    f"  Search ROAS (94% HDI): uncalibrated {brief['roas']['uncalibrated']['search']['mean']:.2f}"
    f" ({brief['roas']['uncalibrated']['search']['hdi_low']:.2f} to"
    f" {brief['roas']['uncalibrated']['search']['hdi_high']:.2f}), calibrated"
    f" {brief['roas']['calibrated']['search']['mean']:.2f}"
    f" ({brief['roas']['calibrated']['search']['hdi_low']:.2f} to"
    f" {brief['roas']['calibrated']['search']['hdi_high']:.2f})",
    "SLIDE 4 · Where the next dollar goes, and its risk",
    *[f"  {name}: " + ", ".join(f"{c} ${stage4.loc[name, c]:,.0f}" for c in CHANNELS)
      + f"; gain ${stage4.loc[name, 'gain vs current']:,.0f}/week, P(beats current)"
      f" {stage4.loc[name, 'P(beats current)']:.2f}"
      for name in ("short-run", "long-run", "risk-averse")],
    f"  Largest move: {moved.abs().idxmax()} {moved[moved.abs().idxmax()]:+,.0f} a week;"
    f" widest 94% HDI of marginal ROAS at the plan: {widest} ({mroas_hdi[widest][0]:.2f} to"
    f" {mroas_hdi[widest][1]:.2f} sales $ per extra $)",
    "SLIDE 5 · Whom to target, the limits, the next test",
    "  Targeting rule: send when 0.30 x estimated uplift > $0.75, scored on held-out customers",
    "  Assumptions to name: the media-to-acquisition shares and costs per new customer; one test"
    " fixes search's level, not its slope above current spend; uplift model carries over to"
    " new customers",
    f"  Disclosure: {brief['mode']} run; reference stages: "
    + (", ".join(str(n) for n in reference_stages) or "none")
    + ("; WORKED_EXAMPLE (all reference)" if WORKED_EXAMPLE else ""),
]
print("\n".join(lines))

# %% [markdown]
# ## Decision · The recommendation
#
# **Number.** Printed by the next cell from the stages above: the weekly budget by channel,
# its expected long-run value gain over the current plan with a 94% HDI and the posterior
# probability it beats the current plan; the offer list and its held-out profit with a 94%
# interval.
#
# **Rules** (stated before reading the numbers):
#
# - **Budget** (Modules 9 and 10): recommend the long-run plan (short-run margin plus new
#   customers' future margin, calibrated MMM, Stage 1 values without the first purchase) unless
#   its 10% quantile of weekly value is below the current plan's; then recommend the
#   risk-averse plan.
# - **Customer value** (Module 5): pay at most the 20th percentile of a new customer's value to
#   acquire one through a channel (the CAC cap).
# - **Offer** (Module 11): send it where margin × estimated uplift > \$0.75.

# %%
mode = "QUICK run: treat this as a rough answer" if QUICK else "FULL run"
print(f"Number ({mode})")
print("  Weekly budget, " + pick_name + " plan: "
      + ", ".join(f"{c} ${plans[pick_name][c]:,.0f}" for c in CHANNELS))
print(f"  Long-run value gain over the current plan: ${pick['gain vs current']:,.0f} a week of"
      f" margin (94% HDI ${pick['gain HDI low']:,.0f} to ${pick['gain HDI high']:,.0f});"
      f" P(beats current) = {pick['P(beats current)']:.2f}")
print(f"  Offer: {send.mean():.0%} of customers; ${brief['offer']['profit per customer']:.2f}"
      f" per customer (94% interval ${brief['offer']['low']:.2f} to"
      f" ${brief['offer']['high']:.2f}) against ${brief['offer']['all']:.2f} for mailing everyone")
print("Rule")
print(f"  Long-run plan unless its 10% quantile is below the current plan's (here: the {pick_name}"
      " plan); offer where 0.30 x uplift > $0.75.")
print("Recommendation: your sentence below.")

# %% [markdown]
# **Recommendation.** Write one sentence a CFO could act on: the weekly spend by channel and
# the money at stake with its interval, how sure you are (the probability), whom to send the
# offer to and what it earns, and what would change it (the main risk on slide 4 and the next
# test you would run).
#
# ```text
# Your sentence: ________________________________________________
# ```
#
# **After the presentations**, run the next cell. It reveals the truth for every stage, which
# a real analysis never sees, and how far each stage's error carried into the plan.

# %%
from mktstats.synth.mmm import steady_state_response

key4 = answers["stage4_allocation"]
tch = truth_all["mmm"]["channels"]


def true_value(plan):
    """True weekly long-run value of a plan ($ margin)."""
    return sum(MARGIN * float(steady_state_response(plan[c], tch[c]["saturation_beta_sales_units"],
                                                    tch[c]["saturation_lam"],
                                                    tch[c]["channel_scale"]))
               + plan[c] * key4["long_run_value_per_dollar_of_new_customers"][c]
               for c in CHANNELS)


v_best = key4["weekly_value_optimal"]
truth_curves = {c: (lambda s, c=c: float(steady_state_response(
    s, tch[c]["saturation_beta_sales_units"], tch[c]["saturation_lam"], tch[c]["channel_scale"])))
    for c in CHANNELS}
reveal_plans = {**plans, "true optimum": pd.Series(key4["optimal_allocation"])[CHANNELS],
                "true curves, your CLV (Stage 3 error removed)": recommend_budget(
                    truth_curves, clv_media, ncpd, BOUNDS, BUDGET, MARGIN)[CHANNELS],
                "your curves, true CLV (Stage 1 error removed)": recommend_budget(
                    mean_curves, key4["clv_margin_by_media_channel"], ncpd, BOUNDS, BUDGET,
                    MARGIN)[CHANNELS]}
print("Stage 1 · value of a new customer (margin, first purchase in): truth vs your 94% HDI")
for j, c in enumerate(ACQUISITION):
    t_ = answers["stage1_customer_value"]["by_acquisition_channel"][c]["margin_including_first_purchase"]
    lo, hi = value_hdi[j]
    print(f"  {c:9s} truth ${t_:7.2f}; yours ${value_draws[:, j].mean():7.2f} [{lo:.2f}, {hi:.2f}]"
          f" inside: {lo <= t_ <= hi}")
k2 = answers["stage2_geo_test"]
print(f"Stage 2 · sales change: truth {usd(k2['incremental_sales'])}; yours"
      f" {usd(stage2['incremental'])} [{usd(stage2['low'])}, {usd(stage2['high'])}]; true sales"
      f" per search $ {k2['campaign']['incremental_roas']:.2f}")
print("Stage 3 · ROAS: truth vs calibrated mean [94% HDI] (uncalibrated mean)")
for c in CHANNELS:
    print(f"  {c:8s} truth {answers['stage3_mmm']['true_roas'][c]:.2f}; calibrated"
          f" {stage3.loc[c, 'mean']:.2f} [{stage3.loc[c, 'hdi_low']:.2f},"
          f" {stage3.loc[c, 'hdi_high']:.2f}] ({stage3_raw.loc[c, 'mean']:.2f})")
print("Stage 4 · plans scored on the TRUE curves and values (share of the true optimal value)")
print(pd.DataFrame({name: {**{c: p[c] for c in CHANNELS},
                           "true value / optimum": true_value(p) / v_best}
                    for name, p in reveal_plans.items()}).T.round(3).to_string())
k5 = answers["stage5_targeting"]["test_split"]
print(f"Stage 5 · true profit per held-out customer: your rule ${rule_value:.3f}; mail everyone"
      f" ${k5['treat_all']:.3f}; perfect targeting ${k5['oracle']:.3f}")

# %% [markdown]
# ## Stretch (optional) · What calibration was worth
#
# 1. Rerun Stage 4 with the **uncalibrated** model's curves (`mmm_raw`) and score the plan on
#    the true curves with `true_value`. How much weekly value did one geo test add?
# 2. Value new customers **with** the first purchase in Stage 4 (`first_purchase=True`). The
#    first purchase is then counted twice, once in the MMM's sales and once in CLV: does it
#    change the plan here, and why might it elsewhere?
#
# <details><summary>Code for the first one</summary>
#
# ```python
# post_raw = mmm_raw.idata.posterior.to_dataset().stack(sample=("chain", "draw"))
# b_raw = post_raw["saturation_beta"].sel(channel=CHANNELS).transpose("sample", "channel").values
# l_raw = post_raw["saturation_lam"].sel(channel=CHANNELS).transpose("sample", "channel").values
# xs_raw = mmm_raw.idata.constant_data["channel_scale"].sel(channel=CHANNELS).values
# ys_raw = float(mmm_raw.idata.constant_data["target_scale"])
# raw_curves = {c: (lambda s, j=j: float(np.mean(response_curve(
#     s, b_raw[:, j], l_raw[:, j], xs_raw[j], ys_raw)))) for j, c in enumerate(CHANNELS)}
# plan_raw = recommend_budget(raw_curves, clv_media, ncpd, BOUNDS, BUDGET, MARGIN)
# print(plan_raw.round(0).to_dict(), true_value(plan_raw) / v_best,
#       true_value(plan_long) / v_best)
# ```
# </details>
