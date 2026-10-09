# ---
# jupyter:
#   mktstats:
#     verified: >-
#       2026-10-09, Technical Expert, against installed source. pymc-marketing 1.2.0
#       ParetoNBDModel: default_model_config keys r, alpha, s, beta, purchase_coefficient,
#       dropout_coefficient, purchase_covariate_cols, dropout_covariate_cols; with covariates
#       alpha_i = alpha_scale * exp(-purchase_data . purchase_coefficient) (same for beta), and
#       fit_summary() names them alpha_scale, beta_scale, purchase_coefficient[col];
#       fit(data, method="map", progressbar=, random_seed=); expected_purchases(data, future_t=).
#       Flat priors (HalfFlat, Flat coefficients) give the CLVTools maximum-likelihood estimates
#       within 1.1% on the retailer (dropout coefficients differ most). scikit-learn 1.6.1
#       HistGradientBoostingRegressor(loss="poisson", random_state=) (no progress output). LightGBM is not used: it needs libomp
#       on macOS.
# ---

# %% [markdown]
# # Part C · BTYD versus gradient boosting
#
# The two exercises here follow Exercises 1 to 3 of the R notebook (`03-clvtools-covariates`).
# The question: for next quarter's purchases per customer, does a flexible machine-learning
# model forecast better than a buy-till-you-die (BTYD) model? We measure it on held-out data
# instead of assuming an answer, on two datasets: the synthetic retailer and CDNOW.
#
# The forecast horizon is $H$ = 13 weeks (a quarter) after each dataset's cutoff date.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing import clv
from sklearn.ensemble import HistGradientBoostingRegressor

from mktstats import data, synth

SEED = 2027  # seeds the gradient boosting and the bootstrap
H = 13  # forecast horizon, weeks
retailer = synth.retailer()  # seed 2026, the same data as data/synthetic/retailer_*.csv
tx = retailer.transactions  # customer_id, date, amount
customers = retailer.customers  # acquisition channel and the 0/1 indicators
truth = retailer.truth
cdnow = data.load_cdnow().rename(columns={"id": "customer_id", "spent": "amount"})
cdnow = cdnow[["customer_id", "date", "amount"]]
CUTOFFS = {"retailer": pd.Timestamp(truth["calibration_end"]), "CDNOW": pd.Timestamp("1997-09-30")}


def purchases_between(tx, start, end):
    """Purchase days per customer after `start` and on or before `end` (a Series)."""
    window = tx.loc[(tx["date"] > start) & (tx["date"] <= end), ["customer_id", "date"]]
    return window.drop_duplicates().groupby("customer_id").size()


print(f"Retailer: {len(tx):,} purchases, cutoff {CUTOFFS['retailer']:%Y-%m-%d};"
      f" CDNOW: {len(cdnow):,} purchases, cutoff {CUTOFFS['CDNOW']:%Y-%m-%d}; H = {H} weeks")

# %% [markdown]
# ## Exercise 1 · Features and a target without leakage (9 minutes)
#
# A machine-learning model needs labelled examples. The design: build each customer's
# features at the date `cutoff - H`, and use the purchases in the following $H$ weeks, up to
# `cutoff`, as the target. Train on those. Then build the same features at `cutoff` and
# predict the $H$ weeks after it, which the model has never seen. *Leakage* is any
# information from after the date the features describe; it makes a model look better than
# it will be in use.
#
# **Predict.** If the features were built from the whole log, including the holdout weeks,
# would the gradient-boosting holdout error look better, worse or the same? Would that be a
# real improvement?
#
# **Task.** Write `make_features(tx, customers, cutoff)` using only transactions on or before
# `cutoff` (a Timestamp). Count purchase **days** (sum the amounts of a customer's purchases
# on one day). Return one row per customer whose first purchase is on or before `cutoff`, with
# `customer_id` and:
#
# - `frequency`, `recency`, `T`: as in Module 2 (weeks as days / 7);
# - `mean_spend`: the mean amount per purchase day, first purchase included;
# - `weeks_since_last`: weeks from the last purchase day to `cutoff`;
# - `purchases_4w`, `purchases_13w`: purchase days in the last 28 and 91 days up to `cutoff`
#   (a purchase `d` days before `cutoff` counts when `d < 28`, or `d < 91`);
# - if `customers` is not None, its columns whose names start with `channel_`.

# %% tags=["exercise"]
def make_features(tx, customers, cutoff):
    # TODO 1: filter to date <= cutoff first, then one row per customer
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def make_features(tx, customers, cutoff):
    past = tx.loc[tx["date"] <= cutoff]  # nothing after the cutoff enters below
    days = past.groupby(["customer_id", "date"], as_index=False)["amount"].sum()
    g = days.groupby("customer_id")
    first, last = g["date"].min(), g["date"].max()
    days_ago = (cutoff - days["date"]).dt.days
    out = pd.DataFrame({
        "frequency": g.size() - 1,
        "recency": (last - first).dt.days / 7,
        "T": (cutoff - first).dt.days / 7,
        "mean_spend": g["amount"].mean(),
        "weeks_since_last": (cutoff - last).dt.days / 7,
        "purchases_4w": (days_ago < 28).groupby(days["customer_id"]).sum(),
        "purchases_13w": (days_ago < 91).groupby(days["customer_id"]).sum(),
    }).rename_axis("customer_id").reset_index()
    if customers is not None:
        channel_cols = [c for c in customers.columns if c.startswith("channel_")]
        out = out.merge(customers[["customer_id", *channel_cols]], on="customer_id", how="left")
    return out


# %% [markdown]
# **Explain.** Compare with your prediction. Then say what the BTYD model needs that the
# machine-learning model does not, and the other way round.
#
# <details><summary>Why this solution works</summary>
#
# Filtering to `date <= cutoff` before anything else guarantees that no later purchase can
# reach a feature, which is what the checkpoint tests by deleting and scrambling everything
# after the cutoff. With features from the whole log, the holdout error would look better
# but the model would be using the answer: in real use, the future is not in the log. A
# BTYD model needs no labelled training window; it extrapolates from each customer's $x$,
# $t_x$ and $T$. The machine-learning model learns the mapping from features to the next
# $H$ weeks from an earlier window, which costs $H$ weeks of history and fixes the horizon.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    toy_tx = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 3],
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-03-01", "2024-03-20",
                                "2024-04-15"]),
        "amount": [10.0, 5.0, 20.0, 8.0, 50.0],
    })
    toy_customers = pd.DataFrame({"customer_id": [1, 2, 3], "channel_social": [0, 1, 0]})
    got = make_features(toy_tx, toy_customers, pd.Timestamp("2024-03-31"))
    expected = pd.DataFrame({
        "customer_id": [1, 2], "frequency": [1, 0], "recency": [60 / 7, 0.0],
        "T": [90 / 7, 11 / 7], "mean_spend": [17.5, 8.0], "weeks_since_last": [30 / 7, 11 / 7],
        "purchases_4w": [0, 1], "purchases_13w": [2, 1], "channel_social": [0, 1],
    })
    checks.columns(got, list(expected.columns), name="make_features(toy_tx, ...)")
    try:
        pd.testing.assert_frame_equal(
            got[list(expected.columns)].sort_values("customer_id").reset_index(drop=True),
            expected, check_dtype=False, rtol=1e-9)
    except AssertionError as err:
        raise AssertionError(
            f"On the toy log (customer 1 buys twice on 1 Jan and once on 1 Mar 2024, customer 2"
            f" on 20 Mar, customer 3 only after the 31 Mar cutoff) the features differ:\n{err}\n"
            "Check: purchase days, not rows; customer 3 must be absent; 1 Mar is 30 days before"
            " the cutoff. To move on now, run workshop.use_reference(1)."
        ) from None
    cutoff = CUTOFFS["retailer"]
    features = make_features(tx, customers, cutoff)
    late = tx["date"] > cutoff
    scrambled = tx.copy()
    rng = np.random.default_rng(SEED)
    scrambled.loc[late, "customer_id"] = rng.permutation(scrambled.loc[late, "customer_id"])
    scrambled.loc[late, "amount"] = rng.permutation(scrambled.loc[late, "amount"].to_numpy())
    for label, other in (("deleted", tx.loc[~late]), ("scrambled", scrambled)):
        try:
            pd.testing.assert_frame_equal(features, make_features(other, customers, cutoff))
        except AssertionError:
            raise AssertionError(
                f"The features change when the purchases after the cutoff are {label}: they leak"
                " the future. Filter tx to date <= cutoff before computing anything. To move on"
                " now, run workshop.use_reference(1)."
            ) from None

# %% [markdown]
# ### Run · The training and holdout design
#
# For each dataset: training features at `cutoff - H` with the target in the $H$ weeks up to
# `cutoff`; holdout features at `cutoff` with the target in the $H$ weeks after it.

# %%
design = {}
for name, (log, cust) in {"retailer": (tx, customers), "CDNOW": (cdnow, None)}.items():
    cutoff = CUTOFFS[name]
    train_cutoff = cutoff - pd.Timedelta(weeks=H)
    train = make_features(log, cust, train_cutoff)
    train["target"] = train["customer_id"].map(purchases_between(log, train_cutoff, cutoff)).fillna(0)
    test = make_features(log, cust, cutoff)
    test["target"] = test["customer_id"].map(
        purchases_between(log, cutoff, cutoff + pd.Timedelta(weeks=H))).fillna(0)
    design[name] = {"train": train, "test": test, "train_window_end": cutoff}
    print(f"{name}: {len(train):,} training customers (features at {train_cutoff:%Y-%m-%d}),"
          f" {len(test):,} holdout customers; holdout purchases {int(test['target'].sum()):,}")

# %% tags=["checkpoint"]
with workshop.checkpoint(label="design"):
    for name, d in design.items():
        assert d["train_window_end"] == CUTOFFS[name], (
            f"{name}: the training target must end at the cutoff, where the holdout begins."
        )
        assert d["test"]["T"].min() >= 0 and d["train"]["T"].max() <= d["test"]["T"].max(), (
            f"{name}: the training features describe an earlier date than the holdout features."
        )

# %% [markdown]
# ### Run · Fit both forecasters
#
# - **BTYD:** the Pareto/NBD of Module 2 by MAP with flat priors (so it is the
#   maximum-likelihood fit), on the holdout features at `cutoff`. For the retailer the two
#   channel indicators enter the purchase and dropout rates, as in the R notebook; compare its
#   coefficients with CLVTools' there. It forecasts with `expected_purchases(future_t=H)`.
# - **Gradient boosting:** scikit-learn's `HistGradientBoostingRegressor` with a Poisson loss
#   (suited to counts), default settings, trained on the training design.
#
# Both take seconds (the first Pareto/NBD fit compiles the model first).

# %%
def fit_both(name):
    train, test = design[name]["train"], design[name]["test"]
    channel_cols = [c for c in test.columns if c.startswith("channel_")]
    config = {k: Prior("HalfFlat") for k in ("r", "alpha", "s", "beta")}
    if channel_cols:
        config |= {"purchase_coefficient": Prior("Flat"), "dropout_coefficient": Prior("Flat"),
                   "purchase_covariate_cols": channel_cols, "dropout_covariate_cols": channel_cols}
    rfm = test[["customer_id", "frequency", "recency", "T", *channel_cols]]
    btyd = clv.ParetoNBDModel(model_config=config)
    btyd.fit(data=rfm, method="map", progressbar=False, random_seed=SEED)
    forecast = btyd.expected_purchases(data=rfm, future_t=H).mean(("chain", "draw")).to_series()
    feature_cols = [c for c in train.columns if c not in ("customer_id", "target")]
    gbm = HistGradientBoostingRegressor(loss="poisson", random_state=SEED)  # no progress bar
    gbm.fit(train[feature_cols], train["target"])
    preds = {"BTYD (Pareto/NBD)": test["customer_id"].map(forecast).to_numpy(),
             "Gradient boosting": gbm.predict(test[feature_cols])}
    summary = btyd.fit_summary()
    return preds, summary[[k for k in summary.index if "[" not in k or "coefficient" in k]]


results = {}
for name in design:
    preds, params = fit_both(name)
    results[name] = preds
    print(f"{name} Pareto/NBD (flat priors, MAP):", params.round(4).to_dict())

# %% [markdown]
# The scorecard groups customers by calibration frequency: 0, 1, 2–3, 4–7 and 8 or more.

# %%
def frequency_group(frequency):
    """Calibration frequency in five groups; the labels sort in this order."""
    bins = pd.cut(pd.Series(frequency).to_numpy(), [-1, 0, 1, 3, 7, np.inf],
                  labels=["0", "1", "2-3", "4-7", "8+"])
    return np.asarray(bins.astype(str))


pd.Series(frequency_group(design["retailer"]["test"]["frequency"])).value_counts().sort_index()

# %% [markdown]
# ## Exercise 2 · A holdout scorecard (9 minutes)
#
# **Predict.** For customers with no purchase after their first one (calibration frequency 0),
# which method will have the lower mean absolute error: BTYD or gradient boosting? And for
# customers with 8 or more?
#
# **Task.** Write `holdout_scorecard(y_true, preds, groups)` where `y_true` is an array of
# actual holdout purchases, `preds` a dict {method name: array of forecasts} and `groups` an
# array of group labels (one per customer). Return one row per method and group with the
# columns `method`, `group`, `customers`, `MAE` (mean absolute error), `RMSE` (root mean
# squared error) and `bias` (sum of forecasts minus sum of actuals). List the groups in sorted
# order, `sorted(set(groups))`.

# %% tags=["exercise"]
def holdout_scorecard(y_true, preds, groups):
    # TODO 2: errors per method, then MAE, RMSE and bias per group
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def holdout_scorecard(y_true, preds, groups):
    rows = []
    y_true, groups = np.asarray(y_true, dtype=float), np.asarray(groups)
    for method, forecast in preds.items():
        error = np.asarray(forecast, dtype=float) - y_true
        for g in sorted(set(groups)):
            e = error[groups == g]
            rows.append({"method": method, "group": g, "customers": e.size,
                         "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
                         "bias": e.sum()})
    return pd.DataFrame(rows)


# %% [markdown]
# **Explain.** Compare with your prediction and point to the group where the methods differ
# most. What does each method use to forecast a customer, and why does that matter most in
# that group?
#
# <details><summary>Why this solution works</summary>
#
# MAE weighs every error equally; RMSE weighs large errors more; bias shows whether a method
# forecasts too many or too few purchases in total, which matters for planning stock or
# budget. The BTYD model extrapolates from each customer's own $x$, $t_x$ and $T$ through a
# model of buying and leaving, so it says something sensible for any customer and any
# horizon. Gradient boosting learns from other customers' features how many purchases
# followed in the training window, so it can use any feature but forecasts only the horizon
# it was trained on, and the training window is earlier than the forecast window. Which wins
# depends on the data; the scorecard measures it.
# </details>

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    toy = holdout_scorecard(np.array([0, 1, 2, 3]),
                            {"A": np.array([0, 1, 2, 3]), "B": np.array([1, 1, 1, 1])},
                            np.array(["0", "0", "1", "1"]))
    checks.columns(toy, ["method", "group", "customers", "MAE", "RMSE", "bias"],
                   name="holdout_scorecard(toy)")
    want = pd.DataFrame({"method": ["A", "A", "B", "B"], "group": ["0", "1", "0", "1"],
                         "customers": [2, 2, 2, 2], "MAE": [0.0, 0.0, 0.5, 1.5],
                         "RMSE": [0.0, 0.0, np.sqrt(0.5), np.sqrt(2.5)],
                         "bias": [0.0, 0.0, 1.0, -3.0]})
    try:
        pd.testing.assert_frame_equal(toy[list(want.columns)].reset_index(drop=True), want,
                                      check_dtype=False, rtol=1e-9)
    except AssertionError as err:
        raise AssertionError(
            f"On the toy input, method B forecasts 1 for actuals 0, 1 (group 0) and 2, 3 (group"
            f" 1): MAE 0.5 and 1.5, RMSE sqrt(0.5) and sqrt(2.5), bias +1 and -3.\n{err}\n"
            "To move on now, run workshop.use_reference(2)."
        ) from None
    scorecards = {}
    for name, preds in results.items():
        test = design[name]["test"]
        groups = frequency_group(test["frequency"])
        card = holdout_scorecard(test["target"].to_numpy(), preds, groups)
        assert len(card) == len(preds) * len(set(groups)), (
            f"{name}: one row per method and group ({len(preds)} x {len(set(groups))})."
        )
        for method, forecast in preds.items():
            rows = card[card["method"] == method]
            assert rows["customers"].sum() == len(test), f"{name}, {method}: every customer once."
            checks.close(rows["bias"].sum(), forecast.sum() - test["target"].sum(), abs=1e-6,
                         name=f"{name}, {method}: bias summed over groups")
        scorecards[name] = card

# %% [markdown]
# The scorecards for both datasets:

# %%
pd.concat(scorecards, names=["dataset", "row"]).round(3)

# %% [markdown]
# Look at the groups where the two bars of a pair differ most, and whether that is the same
# group in both datasets.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (name, card) in zip(axes, scorecards.items(), strict=True):
    wide = card.pivot(index="group", columns="method", values="MAE")
    wide.plot.bar(ax=ax, color=["0.3", "0.7"], edgecolor="black", rot=0)
    for bar in ax.patches[len(wide):]:
        bar.set_hatch("//")  # the second method is hatched as well as lighter
    ax.set_title(f"{name}: holdout MAE by calibration frequency")
    ax.set_xlabel("Calibration frequency group")
    ax.set_ylabel(f"Mean absolute error (purchases in {H} weeks)")
    ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ### Run · Is the difference larger than chance?
#
# A *bootstrap* interval: resample customers with replacement 1,000 times, recompute the
# difference in MAE (BTYD minus gradient boosting) each time, and take the middle 95%. Look at
# whether the interval includes 0; if it does, these data do not separate the methods.

# %%
rng = np.random.default_rng(SEED)
bootstrap = {}
for name, preds in results.items():
    y = design[name]["test"]["target"].to_numpy()
    err_btyd = np.abs(preds["BTYD (Pareto/NBD)"] - y)
    err_gbm = np.abs(preds["Gradient boosting"] - y)
    idx = rng.integers(0, len(y), size=(1000, len(y)))
    diffs = err_btyd[idx].mean(axis=1) - err_gbm[idx].mean(axis=1)
    low, high = np.quantile(diffs, [0.025, 0.975])
    bootstrap[name] = {"MAE BTYD": err_btyd.mean(), "MAE boosting": err_gbm.mean(),
                       "difference": err_btyd.mean() - err_gbm.mean(),
                       "95% interval low": low, "95% interval high": high}
pd.DataFrame(bootstrap).T.round(4)

# %% [markdown]
# Beyond the error, BTYD gives P(alive), forecasts for any horizon and for customers not yet
# acquired, and parameters with a meaning. Simple heuristics have been found competitive with
# BTYD models for some tasks (Wübben & von Wangenheim 2008), which is why the lab measures
# instead of assuming.
#
# ## Decision · Which forecaster for next quarter?
#
# **The rule, stated before the numbers.** Deploy the forecaster with the lower holdout MAE
# only if the 95% bootstrap interval of the difference excludes 0. Otherwise prefer BTYD for
# what it adds (P(alive), any horizon). The channel question was decided in the R notebook:
# treat a channel differently if its rate differs from search by more than 20% with a 90%
# interval that excludes no difference.

# %%
for name, b in bootstrap.items():
    lower = "BTYD" if b["difference"] < 0 else "gradient boosting"
    separated = b["95% interval low"] > 0 or b["95% interval high"] < 0
    print(f"{name}: lower MAE: {lower} (difference {b['difference']:+.4f} purchases per customer,"
          f" 95% interval {b['95% interval low']:+.4f} to {b['95% interval high']:+.4f});"
          f" the interval {'excludes' if separated else 'includes'} 0.")
print("Mode:", "QUICK run: treat this as a rough answer" if QUICK else "FULL run",
      "(QUICK does not change this notebook: the fits are MAP and gradient boosting)")

# %% [markdown]
# **Recommendation.** Write two sentences. First, the channel decision from the R notebook.
# Second, which forecaster you would deploy for next quarter, how sure you are (the interval),
# and what would change it (a different horizon, more holdout weeks, other features).
#
# Your sentences: ________________________________________________

# %% [markdown]
# ## Stretch (optional) · Give gradient boosting the BTYD forecast
#
# Fit the Pareto/NBD at `cutoff - H` on the training features, add its `expected_purchases`
# over $H$ weeks as a feature of the training set (and the forecast from the fit at `cutoff`
# as the same feature of the holdout set), refit gradient boosting and re-score. Does the
# combination beat both? Which feature does the model lean on most?
