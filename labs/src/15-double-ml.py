# ---
# jupyter:
#   api_checked: |
#     2026-10-10, against the installed source of econml 0.17.0, scikit-learn 1.6.1,
#     statsmodels 0.15.0 and numpy 2.1.3 (.venv/lib/python3.13/site-packages), then by running
#     every call below:
#     econml/dml/dml.py: LinearDML(*, model_y, model_t, discrete_treatment=False, cv=2,
#       random_state=None, ...); fit(Y, T, *, X=None, W=None, ..., inference='auto');
#       _FirstStageWrapper.predict returns predict_proba(...)[:, 1:] for a discrete treatment and
#       raises AttributeError "Cannot use a classifier as a first stage model when the target is
#       continuous!" for a classifier with discrete_treatment=False; sensitivity_interval,
#       robustness_value (lines 633-739).
#     econml/_ortho_learner.py: an iterable of (train, test) splits passed as cv is used as given
#       (check_cv); an integer cv becomes StratifiedKFold(shuffle=True, random_state) for a
#       discrete treatment.
#     econml/_cate_estimator.py: ate(X=None, *, T0=0, T1=1), ate_interval(..., alpha=0.05),
#       intercept__inference(); econml/inference/_inference.py: StatsModelsInference(cov_type=
#       'HC1') is LinearDML's default inference (HC1 = HC0 x sqrt(n / (n - 1)) here).
#     econml/dml/_rlearner.py: nuisance_scores_y, nuisance_scores_t (the first-stage models'
#       own score on each held-out fold: R^2 for the regressor, accuracy for the classifier).
#     econml/dml/causal_forest.py: CausalForestDML(..., n_estimators=100, min_samples_leaf=5,
#       subforest_size=4, random_state=None, ...). econml/dr/_drlearner.py: LinearDRLearner(*,
#       model_propensity, model_regression, min_propensity=1e-6, cv=2, random_state=None, ...).
#     sklearn: HistGradientBoostingRegressor / Classifier(..., early_stopping='auto' (on above
#       10,000 rows), random_state=None); cross_val_predict(estimator, X, y, *, cv, method);
#       StratifiedKFold / KFold(n_splits, *, shuffle, random_state); clone; r2_score;
#       roc_auc_score. statsmodels: OLS(...).fit(cov_type='HC1'); add_constant puts the
#       constant first.
#     mktstats.dml (this repository): reference solutions and the checkpoint checks.
#   note: This header is notebook metadata for maintainers; scripts/gen_notebooks.py drops it.
# ---

# %% [markdown]
# **How this lab runs.** The data are simulated: 20,000 customers of the retailer, some of whom
# installed its new app. Because we simulated them, we know how much the app changed each
# customer's spending, so you can grade every estimate against the truth, which real data never
# allow. Every model here is fitted in seconds.
#
# **QUICK** (in the harness cell) keeps a random 10,000 of the 20,000 customers and half of the
# Hillstrom experiment, with the same models; it does not make the lab much faster. With QUICK
# on, intervals are wider and will not match the module page, and a decision near its threshold
# can flip. That is a lesson about sample size, not a bug.

# %%
import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import LinearDML
from IPython.display import Image, display
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from mktstats import dml
from mktstats.data import load_hillstrom, repo_file

SEED = 2035  # the data's seed; every random step below (subsamples, folds, models) uses it
N_QUICK = 10_000  # customers kept with QUICK on


def show(fig):
    """Show a figure as a PNG on Colab, in Jupyter and in a headless run alike."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))


print(f"QUICK = {QUICK}: {'10,000 of the 20,000 customers, half of Hillstrom' if QUICK else 'all 20,000 customers, all of Hillstrom'}.")

# %% [markdown]
# # Part A · Why the obvious answers are wrong
#
# This is **observational data**: nobody assigned the app. Customers chose to install it, and
# the customers who chose it differ from those who did not (**self-selection**). A **confounder**
# (a common cause) is a variable that moves both install and spend, such as having bought in the
# last few weeks; it makes installers spend more whether or not the app does anything.
#
# This lab tries the obvious estimates first, then double machine learning, against a truth we
# know because we simulated the customers.
#
# The next cell loads the customers and the truth. Each customer's **true effect** is how much
# more they spend over the 26 weeks after the launch window with the app than without it, known
# here only because we simulated the data. It prints three averages of it: the **ATE** (average
# treatment effect) over all customers, the **ATT** over installers and the **ATC** over
# non-installers. `features(df)` returns the nine covariates a model may use, measured in the
# 12 months before the launch; the three `true_*` columns are for checking, never for a model.

# %%
truth = json.loads(repo_file("synthetic/truth_app_install.json").read_text())
customers_all = pd.read_csv(repo_file("synthetic/app_install_customers.csv"))
if QUICK:  # a seeded random 10,000 of the 20,000 customers, in their original order
    keep = np.sort(np.random.default_rng(SEED).choice(len(customers_all), N_QUICK, replace=False))
    customers = customers_all.iloc[keep].reset_index(drop=True)
else:
    customers = customers_all

FEATURES = ["tenure_months", "orders_12m", "days_since_last_order", "avg_order_value",
            "email_open_rate", "mobile_share", "loyalty_member", "channel_social",
            "channel_referral"]


def features(df):
    """The nine covariates as a matrix; channel as two 0/1 indicators (search is the reference)."""
    out = df.assign(channel_social=(df["acquisition_channel"] == "social").astype(float),
                    channel_referral=(df["acquisition_channel"] == "referral").astype(float))
    return out[FEATURES].to_numpy(dtype=float)


y = customers["spend_next_26w"].to_numpy(dtype=float)  # Y: dollars over 26 weeks
t = customers["app_installed"].to_numpy()  # T: 1 if the customer installed the app
W = features(customers)  # X, called W in code (EconML's name for covariates to adjust for)
e_true = customers["true_propensity"].to_numpy()  # truth: each customer's chance of installing
tau_true = customers["true_effect"].to_numpy()  # truth: each customer's effect of the app


def overlap_truth(rows=slice(None)):
    """The true overlap-weighted average effect over the given rows (Exercise 3 says what it is)."""
    w = e_true[rows] * (1 - e_true[rows])
    return float(np.sum(w * tau_true[rows]) / np.sum(w))


true_avg = {
    "ATE": (tau_true.mean(), "all customers"),
    "ATT": (np.sum(e_true * tau_true) / e_true.sum(), "installers (each weighted by their chance of installing)"),
    "ATC": (np.sum((1 - e_true) * tau_true) / (1 - e_true).sum(), "non-installers (weighted by their chance of not installing)"),
}
print(f"{len(customers):,} customers; {t.mean():.1%} installed the app in the launch weeks.")
print(f"Mean spend over the next 26 weeks: installers ${y[t == 1].mean():.2f},"
      f" non-installers ${y[t == 0].mean():.2f}; {np.mean(y == 0):.0%} of customers spent $0.")
print("True average effect of the app on 26-week spend:")
for name, (value, who) in true_avg.items():
    print(f"  {name}  ${value:6.2f}  over {who}")

# %% [markdown]
# The first check makes sure the data loaded as the lab expects, before anything is fitted.

# %% tags=["checkpoint"]
with workshop.checkpoint(label="data"):
    dml.check_data(customers, customers_all, truth, features, FEATURES,
                   n_expected=N_QUICK if QUICK else len(customers_all))

# %% [markdown]
# ## Exercise 1 · Naive and regression estimates against the truth (8 minutes)
#
# **Predict.** Installers spent far more than non-installers in the next 26 weeks (printed
# above), and the true averages in the table lie between about \$12 and \$18. A regression of
# spend on install *and* all nine covariates: will its estimate for install be above \$50,
# between \$25 and \$50, or below \$25? Write your guess in the cell below.
#
# **Task.** Write `baseline_estimates(y, t, W)` returning a `pd.DataFrame` with index
# `["naive", "regression"]` and columns `estimate, se, ci_low, ci_high`:
#
# - the **naive difference** is the mean spend of installers minus the mean spend of
#   non-installers, $\bar Y_1 - \bar Y_0$, with standard error
#   $\sqrt{s_1^2/n_1 + s_0^2/n_0}$ ($s^2$ the sample variance of each group, $n$ its size);
# - the regression with **linear controls** enters the covariates as straight-line terms: the
#   coefficient on `t` from an OLS of `y` on a constant, `t` and `W` (the call is in the cell
#   below), with its **robust (HC1) standard error**, one that does not assume every customer's
#   spend is equally noisy (heavy spenders are noisier);
# - both intervals are **95% normal-approximation confidence intervals**: the estimate ± 1.96
#   standard errors.

# %% tags=["exercise"]
def baseline_estimates(y, t, W):
    # TODO 1: naive = mean of y where t == 1 minus mean where t == 0, se from np.var(..., ddof=1);
    # regression: fit = sm.OLS(y, sm.add_constant(np.column_stack([t, W]))).fit(cov_type="HC1")
    # (statsmodels draws no progress bar, so there is no progressbar=False to pass); the
    # coefficient on t is fit.params[1], its standard error fit.bse[1]; intervals: +- 1.96 se
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def baseline_estimates(y, t, W):
    y1, y0 = y[t == 1], y[t == 0]
    naive = y1.mean() - y0.mean()
    naive_se = np.sqrt(np.var(y1, ddof=1) / len(y1) + np.var(y0, ddof=1) / len(y0))
    # statsmodels, EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.
    fit = sm.OLS(y, sm.add_constant(np.column_stack([t, W]))).fit(cov_type="HC1")
    est = np.array([naive, fit.params[1]])
    se = np.array([naive_se, fit.bse[1]])
    return pd.DataFrame({"estimate": est, "se": se, "ci_low": est - 1.96 * se,
                         "ci_high": est + 1.96 * se}, index=["naive", "regression"])


# %% [markdown]
# The checkpoint first tries your function on six made-up customers with $y = 2 + 3t + 5w$
# exactly (`t = [0, 0, 0, 1, 1, 1]`, `w = [0, 1, 2, 1, 2, 3]`): by hand the naive difference is
# 8 (the effect 3 plus 5 times the gap of 1 in mean `w`) with standard error √(50/3) ≈ 4.082,
# and the regression gives 3 with standard error 0 (no noise). Then it compares your numbers on
# the app data with a reference, and both intervals with the true ATT.

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    base = dml.check_baseline_estimates(baseline_estimates, y, t, W, att=true_avg["ATT"][0])
    print(base.round(2).to_string())
    print(f"True averages: ATE ${true_avg['ATE'][0]:.2f}, ATT ${true_avg['ATT'][0]:.2f},"
          f" ATC ${true_avg['ATC'][0]:.2f}. Both 95% intervals lie above all three.")

# %% [markdown]
# Look at how both curves bend in the first 60 days: a straight line in days cannot follow that.

# %%
days = customers["days_since_last_order"]
edges = [0, 7, 14, 21, 30, 45, 60, 90, 120, 180, 240, 365]
bins = pd.cut(days, edges, include_lowest=True)
by_bin = pd.DataFrame({"days": days, "install": t, "spend": y}).groupby(bins, observed=True)
mid = by_bin["days"].mean()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
ax1.plot(mid, 100 * by_bin["install"].mean(), marker="o", color="#4c72b0")
ax1.set_ylabel("Installed the app (%)")
ax1.set_title("Install rate and spend by days since the last order (binned)")
ax2.plot(mid, by_bin["spend"].mean(), marker="s", color="#dd8452")
ax2.set_ylabel("Mean spend, next 26 weeks ($)")
ax2.set_xlabel("Days since the last order, at launch")
for ax in (ax1, ax2):
    ax.axvspan(0, 60, color="grey", alpha=0.12)
    ax.grid(alpha=0.3)
show(fig)

# %% [markdown]
# **Explain.** Compare with your guess. The regression adjusts for all nine covariates and still
# lands far above every true average. Which feature of the data does a straight-line control
# miss? Point to the plot.
#
# <details><summary>Why this solution works</summary>
#
# Customers who bought in the last few weeks are both far more likely to install (they were
# shopping during the launch weeks) and far heavier spenders. A linear term in days fits the
# average slope and leaves the sharp early drop in the residual, where it still moves with
# install, so part of the confounding is credited to the app. The right transformation,
# exp(−days / 30), and the product of orders and order value would fix this regression, but you
# would have to know the shape. Double machine learning learns the shape from the data instead.
# </details>

# %% [markdown]
# # Part B · Double machine learning by hand
#
# **Double machine learning (DML)** estimates the effect after removing, from both spend and
# install, what machine-learning models predict from the covariates. Its assumption is **no
# unobserved confounding**: every common cause of install and spend is among the covariates.
# The data cannot test it; the Decision returns to it.

# %% [markdown]
# ## Exercise 2 · Cross-fitted predictions (8 minutes)
#
# DML needs two predictions per customer from the covariates alone: $\hat q(x)$, the expected
# spend, and $\hat e(x)$, the predicted probability that a customer like $x$ installs (the
# **propensity score**). Each customer's predictions must come from a model that never saw that
# customer: split the customers into $K = 5$ groups (**folds**) and predict each fold with a
# model fitted on the other four, so no customer is predicted by a model that saw them. This is
# **cross-fitting** (Chernozhukov et al. 2018); the predictions are **out-of-fold**.
#
# **Predict.** The checkpoint below fits a model that memorizes its training data (one nearest
# neighbor) to 40 made-up customers whose spends are all different. For the customers it was
# trained on, what share of its predictions will equal the customer's own spend exactly: about
# 0%, about 50% or 100%? And for each customer predicted out of fold, by a model fitted on the
# other folds? Write your guess in the cell below.
#
# **Task.** Write `cross_fit(model, X, target, folds)` returning a NumPy array with one
# prediction per row of `X`. For each `(train, test)` pair in `folds`, fit a fresh copy of the
# model, `clone(model)`, on `X[train]` and `target[train]`, and store its predictions for
# `X[test]` at the positions `test`: the probability of class 1, `predict_proba(X[test])[:, 1]`,
# if the model has `predict_proba` (a classifier), else `predict(X[test])`.

# %% tags=["exercise"]
def cross_fit(model, X, target, folds):
    # TODO 2: out = np.full(len(target), np.nan); for train, test in folds:
    # fitted = clone(model).fit(X[train], target[train]) (scikit-learn draws no progress bar, so
    # there is no progressbar=False to pass); out[test] = fitted.predict_proba(X[test])[:, 1] if
    # hasattr(fitted, "predict_proba") else fitted.predict(X[test]); return out
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def cross_fit(model, X, target, folds):
    out = np.full(len(target), np.nan)
    for train, test in folds:
        # EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.
        fitted = clone(model).fit(X[train], target[train])
        if hasattr(fitted, "predict_proba"):
            out[test] = fitted.predict_proba(X[test])[:, 1]
        else:
            out[test] = fitted.predict(X[test])
    return out


# %% [markdown]
# The checkpoint uses 40 made-up customers (seed `SEED`) and five folds: one nearest neighbor
# must never predict a customer's own spend out of fold; with `LinearRegression()` your result
# must equal scikit-learn's `cross_val_predict` on the same folds; and with a
# `KNeighborsClassifier` on made-up install labels it must equal the probabilities of class 1.

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    toy = dml.check_cross_fit(cross_fit, seed=SEED)
    print(f"In sample: {toy['in_sample']} of {toy['n']} predictions equal the customer's own spend."
          f" Out of fold: {toy['out_of_fold']} of {toy['n']}.")

# %% [markdown]
# **Explain.** Compare with your guess. Exercise 3 works with what the predictions leave over:
# spend minus predicted spend. What would be left over from in-sample predictions like the
# toy's, and what could an estimate learn from it?
#
# <details><summary>Why this solution works</summary>
#
# Nothing would be left: every leftover would be 0, so there would be no variation to learn the
# effect from. A flexible model fitted and evaluated on the same customers reproduces part of
# their noise, so its leftovers are too small and no longer behave like the leftovers for a new
# customer. Out-of-fold predictions are honest: every customer is predicted by a model that did
# not see them, and with five folds each model still learns from 80% of the data. The provided
# run after Exercise 3 shows the same thing on the app data.
# </details>
#
# **Run: the two prediction models (provided).** Gradient-boosted trees (scikit-learn's
# histogram gradient boosting, as in Module 11) predict spend and install, with the five folds
# built once here and reused for the rest of the lab. Two scores say how good the out-of-fold
# predictions are: **R²**, the share of the variation in spend that they explain, and **AUC**,
# the chance that a random installer gets a higher predicted probability than a random
# non-installer (0.5 is guessing, 1 is perfect). This cell fits 10 models, two per fold. While
# it runs, read Exercise 3's Predict.

# %%
BOOST = dict(
    max_iter=300,  # boosting rounds, each adding one small tree
    learning_rate=0.05,  # how much each tree corrects the ones before it
    max_leaf_nodes=15,  # small trees
    min_samples_leaf=50,  # at least 50 customers per leaf: smooth predictions
    early_stopping=False,  # 'auto' would hold out a random 10% inside every fold above 10,000 rows
    random_state=SEED,
)
model_y = HistGradientBoostingRegressor(**BOOST)  # spend
model_t = HistGradientBoostingClassifier(**BOOST)  # install: a classifier
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(W, t))  # stratified on install

y_hat = cross_fit(model_y, W, y, folds)
t_hat = cross_fit(model_t, W, t, folds)
print(f"Out-of-fold R² for spend: {r2_score(y, y_hat):.2f}. AUC for install: {roc_auc_score(t, t_hat):.2f}.")
if QUICK:
    print("With QUICK on, you have half the customers: intervals are wider and will not match the"
          " module page, and a decision near its threshold can flip. That is a lesson about sample"
          " size, not a bug.")

# %% [markdown]
# ## Exercise 3 · Residual on residual, with a standard error (9 minutes)
#
# With the out-of-fold predictions, the **residuals** are what the predictions leave over:
# $\tilde Y = Y - \hat q(X)$ (spend minus predicted spend) and $\tilde T = T - \hat e(X)$
# (install minus predicted probability). **Partialling out** regresses the spend residual on the
# install residual, with no intercept; the slope is the effect:
#
# $$\hat\theta = \frac{\sum_i \tilde T_i \tilde Y_i}{\sum_i \tilde T_i^2}, \qquad
# \widehat{\mathrm{se}}(\hat\theta) = \frac{\sqrt{\sum_i \tilde T_i^2 (\tilde Y_i - \hat\theta \tilde T_i)^2}}{\sum_i \tilde T_i^2},
# \qquad \text{95\% CI: } \hat\theta \pm 1.96\,\widehat{\mathrm{se}}.$$
#
# The standard error is the heteroskedasticity-robust one called HC0. The idea is the one behind
# controls in linear regression (the coefficient on $T$ equals the slope of the $Y$ residual on
# the $T$ residual after both are regressed on the controls), with the linear regressions
# replaced by cross-fitted machine learning. $\tilde T$ is the part of installing the covariates
# cannot predict: the "as good as random" variation the effect is learned from.
#
# **Predict.** Exercise 1's regression gave about \$31 to \$32 (printed above), and the true
# averages lie between about \$12 and \$18. Where will the DML estimate land: near the
# regression, about halfway, or among the true averages? And the half-width of its 95%
# interval: about \$1, \$3 or \$10? Write your guess in the cell below.
#
# **Task.** Write `dml_estimate(y, t, y_hat, t_hat)` returning a dict with `theta`, `se`,
# `ci_low` and `ci_high` from the formulas above.

# %% tags=["exercise"]
def dml_estimate(y, t, y_hat, t_hat):
    # TODO 3: the two residuals; theta = sum(T~ Y~) / sum(T~^2), no intercept;
    # se = sqrt(sum(T~^2 (Y~ - theta T~)^2)) / sum(T~^2); interval theta +- 1.96 se
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def dml_estimate(y, t, y_hat, t_hat):
    u, v = y - y_hat, t - t_hat  # the spend residual and the install residual
    theta = np.sum(v * u) / np.sum(v * v)
    se = np.sqrt(np.sum(v**2 * (u - theta * v) ** 2)) / np.sum(v * v)
    return {"theta": theta, "se": se, "ci_low": theta - 1.96 * se, "ci_high": theta + 1.96 * se}


# %% [markdown]
# The checkpoint compares your interval with the true *overlap-weighted* average effect, not
# the ATE. When the effect differs between customers, residual-on-residual regression averages
# their effects with weight $e(1 - e)$: largest (0.25) for customers whose install was a
# toss-up, near 0 for those almost sure to install or not. Exercise 4 shows who those customers
# are.
#
# It first tries your function on four made-up customers (`t = [1, 0, 1, 0]`,
# `t_hat = [0.5, 0.5, 0.25, 0.25]`, `y = [0, 0, 3, 2]`, `y_hat = [1, 2, 1, 3]`): by hand
# $\hat\theta = 2.25 / 1.125 = 2$ and the standard error is √10 / 3 ≈ 1.054.

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    mine = dml.check_dml_estimate(dml_estimate, y, t, y_hat, t_hat, target=overlap_truth(),
                                  ate=true_avg["ATE"][0])
    print(f"DML estimate: ${mine['theta']:.2f}, 95% CI [${mine['ci_low']:.2f}, ${mine['ci_high']:.2f}]"
          f" (± ${1.96 * mine['se']:.2f}).")
    print(f"True overlap-weighted average effect: ${overlap_truth():.2f}; the ATE is"
          f" ${true_avg['ATE'][0]:.2f}. Exercise 1's regression: ${base.loc['regression', 'estimate']:.2f}.")

# %% [markdown]
# **Explain.** Compare with your guess and with Exercise 1's regression, which used the same
# nine covariates. What did DML do with them that the regression did not? Point to the recency
# plot from Exercise 1.
#
# <details><summary>Why this solution works</summary>
#
# DML learned the curved relationships between the covariates and both install and spend, such
# as the sharp drop over the first weeks since the last order, and removed them from both before
# relating the leftovers. The regression could remove only straight-line relationships, so part
# of the confounding stayed in its estimate. What would change the result: a common cause that
# is not among the covariates; the Decision shows one.
# </details>
#
# **Run: what skipping cross-fitting does (provided).** The next cell fits a booster that
# memorizes its data (deep trees, 600 rounds, leaves of 2 customers) on *all* customers, predicts
# the same customers (an **in-sample prediction**: a prediction for a customer the model was
# fitted on) and calls your `dml_estimate`. Look at the standard error, and at how close the
# in-sample predictions of install come to the actual 0s and 1s.

# %%
OVERFIT = dict(BOOST, max_iter=600, learning_rate=0.1, max_leaf_nodes=255, min_samples_leaf=2)
# EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.
y_in = HistGradientBoostingRegressor(**OVERFIT).fit(W, y).predict(W)
t_in = HistGradientBoostingClassifier(**OVERFIT).fit(W, t).predict_proba(W)[:, 1]
in_sample = dml_estimate(y, t, y_in, t_in)
se_ratio = in_sample["se"] / mine["se"]
print(pd.DataFrame({
    "estimate": [mine["theta"], in_sample["theta"]],
    "se": [mine["se"], in_sample["se"]],
    "mean |T - e_hat|": [np.mean(np.abs(t - t_hat)), np.mean(np.abs(t - t_in))],
}, index=["cross-fitted (5 folds)", "in sample, no cross-fitting"]).round(4).to_string())
print(f"Without cross-fitting the standard error is {se_ratio:.1f} times the cross-fitted one,"
      f" and the point estimate (${in_sample['theta']:.2f} here) changes from seed to seed.")

# %% tags=["checkpoint"]
with workshop.checkpoint(label="no cross-fitting"):
    dml.check_no_cross_fitting(in_sample["se"], mine["se"], min_ratio=2.0)

# %% [markdown]
# <details><summary>Why the standard error grows</summary>
#
# A model that memorizes the data predicts every install almost exactly (the mean of
# $|T - \hat e|$ in sample, printed above, is close to 0), so almost no "as good as random"
# variation is left and the estimate divides noise by almost nothing. With cross-fitting, small
# errors in $\hat q$ and $\hat e$ do not shift $\hat\theta$ to first order (the briefing's name
# for this is Neyman orthogonality), which is why imperfect machine-learning predictions still
# give a valid interval (Chernozhukov et al. 2018; EconML's DML user guide).
# </details>

# %% [markdown]
# # Part C · Whose effect is it?
#
# **Overlap** means every kind of customer had some chance both to install and not to:
# $0 < e(x) < 1$. As Checkpoint 3 said, the residual-on-residual estimate is a weighted average
# of the customers' effects with weights $e(1 - e)$, the **overlap weights** (Li, Morgan &
# Zaslavsky 2018): 0.25 at $e = 0.5$, near 0 at either end.
#
# Look at the left edge: customers almost sure not to install have no installers like them to
# compare with.

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.6))
grid = np.linspace(0, 1, 41)
ax.hist(t_hat[t == 0], bins=grid, histtype="step", linewidth=1.8, color="#4c72b0",
        label="non-installers")
ax.hist(t_hat[t == 1], bins=grid, histtype="stepfilled", alpha=0.35, hatch="//",
        color="#dd8452", edgecolor="#dd8452", label="installers")
for edge in (0.05, 0.95):
    ax.axvline(edge, color="black", linestyle=":", linewidth=1)
ax.set_xlabel("Predicted probability of installing (out of fold)")
ax.set_ylabel("Customers")
ax.set_title("Overlap: predicted probability of installing, by what customers did")
ax.legend()
show(fig)
outside = (t_hat < 0.05) | (t_hat > 0.95)
print(f"{outside.mean():.1%} of customers have a predicted probability outside [0.05, 0.95]"
      f" ({np.mean(t_hat < 0.05):.1%} below 0.05, {np.mean(t_hat > 0.95):.2%} above 0.95).")

# %% [markdown]
# ## Exercise 4 · Overlap and trimming (7 minutes)
#
# **Trimming** drops the customers whose $\hat e$ lies outside $[\ell, h]$ before estimating. It
# changes which customers the estimate describes. Crump et al. (2009) give $[0.1, 0.9]$ as a
# rule of thumb for the average effect; the lab starts gentler, at $[0.05, 0.95]$.
#
# **Predict.** The cell above printed the share of customers with $\hat e$ outside
# $[0.05, 0.95]$. What share of the total weight $\hat e(1 - \hat e)$ do they carry: about the
# same, much less (around 1%), or much more? Write your guess in the cell below.
#
# **Task.** Write `trimmed_estimate(y, t, y_hat, t_hat, low, high)`: keep the rows with
# `low <= t_hat <= high`, call your `dml_estimate` on them, and add to its dict `share_kept`
# (the share of customers kept) and `weight_share_dropped` (the dropped customers' share of the
# total weight $\hat e(1 - \hat e)$, from `t_hat`). The predictions come from Exercise 2;
# nothing is refitted.

# %%
LOW, HIGH = 0.05, 0.95  # drops only customers nearly sure either way: their weight is under 0.0475, a fifth of the largest

# %% tags=["exercise"]
def trimmed_estimate(y, t, y_hat, t_hat, low, high):
    # TODO 4: keep = (t_hat >= low) & (t_hat <= high); out = dml_estimate on the kept rows of all
    # four arrays; w = t_hat * (1 - t_hat); add share_kept = keep.mean() and
    # weight_share_dropped = w[~keep].sum() / w.sum()
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def trimmed_estimate(y, t, y_hat, t_hat, low, high):
    keep = (t_hat >= low) & (t_hat <= high)
    out = dict(dml_estimate(y[keep], t[keep], y_hat[keep], t_hat[keep]))
    w = t_hat * (1 - t_hat)
    out["share_kept"] = keep.mean()
    out["weight_share_dropped"] = w[~keep].sum() / w.sum()
    return out


# %% [markdown]
# The checkpoint first uses six made-up customers (`t_hat = [0.02, 0.10, 0.30, 0.50, 0.70,
# 0.97]`, `low, high = 0.05, 0.95`): rows 2 to 5 are kept, `share_kept` is 4/6 and
# `weight_share_dropped` is 0.0487 / 0.8087 ≈ 0.0602. On the app data it checks the share kept,
# the estimate on the kept rows, that less weight than customers is dropped, and that the true
# overlap-weighted effect *of the kept customers* lies inside your 95% interval.

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    trimmed = dml.check_trimmed_estimate(trimmed_estimate, dml_estimate, y, t, y_hat, t_hat,
                                         LOW, HIGH, target_fn=overlap_truth)
    print(f"Trimmed to [{LOW}, {HIGH}]: kept {trimmed['share_kept']:.1%} of customers, dropped"
          f" {trimmed['weight_share_dropped']:.1%} of the weight. Estimate ${trimmed['theta']:.2f}"
          f" [${trimmed['ci_low']:.2f}, ${trimmed['ci_high']:.2f}] against the kept customers'"
          f" truth ${overlap_truth(trimmed['mask']):.2f} (untrimmed: ${mine['theta']:.2f}).")

# %% [markdown]
# **Run: whose effect the estimate describes (provided).** The table compares the customers on
# average, the installers, the non-installers, and the customers weighted by
# $\hat e(1 - \hat e)$, the ones the estimate describes. Below it, two stricter trims next to
# the true effect of the customers each keeps. Look at the last column: how does the customer the
# estimate describes differ from the average customer?

# %%
cols = ["orders_12m", "days_since_last_order", "loyalty_member", "mobile_share", "avg_order_value"]
weight = t_hat * (1 - t_hat)
profile = pd.DataFrame({
    "all customers": customers[cols].mean(),
    "installers": customers.loc[t == 1, cols].mean(),
    "non-installers": customers.loc[t == 0, cols].mean(),
    "whose effect the estimate describes": customers[cols].mul(weight, axis=0).sum() / weight.sum(),
})
print(profile.round(2).to_string())
print()
for low, high in [(LOW, HIGH), (0.1, 0.9), (0.2, 0.8)]:
    r = trimmed_estimate(y, t, y_hat, t_hat, low, high)
    kept = (t_hat >= low) & (t_hat <= high)
    print(f"Trim [{low}, {high}]: kept {r['share_kept']:5.1%}, estimate ${r['theta']:5.2f}"
          f" [${r['ci_low']:5.2f}, ${r['ci_high']:5.2f}], true effect of the kept ${overlap_truth(kept):5.2f}")

# %% [markdown]
# **Explain.** Compare with your guess. Trimming dropped the share of customers printed above the
# exercise but moved the estimate by well under a dollar: why so little? Then read the profile:
# which customers does the estimate describe, and which does it say least about?
#
# <details><summary>Why this solution works</summary>
#
# Residual-on-residual regression already gives customers who were almost sure to install, or
# almost sure not to, almost no weight, because their install leaves little unexplained
# variation; trimming them removes little. Estimators that divide by $\hat e$, such as the
# doubly robust average effect in the stretch, react much more to extreme propensity scores. The
# estimate describes customers whose install was in doubt: more recent, more mobile and more
# loyal than average. Trimming changes the population, and so the target: at $[0.2, 0.8]$ about
# half the customers remain and the true average among them is higher. Which average matters
# depends on the decision. For all 20,000 customers the truth file gives \$13.58 (all
# customers), \$17.64 (installers), \$11.87 (non-installers) and \$14.28 (overlap-weighted).
# </details>

# %% [markdown]
# # Part D · The library, checked on a real experiment

# %% [markdown]
# ## Exercise 5 · LinearDML: your estimate, then a confounded experiment (8 minutes)
#
# EconML's **`LinearDML`** does Exercises 2 and 3 in one call. Its two prediction models are
# called **nuisance models**: needed for the estimate, not of interest themselves. Four
# settings:
#
# - `model_y`, `model_t`: the same two boosted models as in Exercise 2;
# - `discrete_treatment=True`: install is a category (0 or 1), so `model_t` must be a
#   classifier;
# - `cv=folds`: our five folds (its default is `cv=2`);
# - `random_state=seed`.
#
# Then fit it with `X=None` and `W=W`. **`W`** holds the covariates to adjust for (controls);
# **`X`** holds covariates the effect may vary with: none here, so one effect. You use it twice:
# on the app data, where it must reproduce your Exercise 3 numbers, and on a real experiment
# that we spoil on purpose.
#
# **Predict (1 of 2).** With the same folds, the same two models and the same seed, will
# EconML's estimate equal yours to the cent, differ by a few cents, or differ by dollars? Write
# your guess in the cell below.
#
# **Task.** Write `fit_linear_dml(y, t, W, folds, model_y, model_t, seed)` returning the fitted
# `LinearDML` with the four settings above.

# %% tags=["exercise"]
def fit_linear_dml(y, t, W, folds, model_y, model_t, seed):
    # TODO 5: LinearDML(model_y=..., model_t=..., discrete_treatment=..., cv=...,
    # random_state=...), then its fit method with y, t, X=None, W=W; return the fitted estimator
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def fit_linear_dml(y, t, W, folds, model_y, model_t, seed):
    est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, cv=folds,
                    random_state=seed)
    # EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.
    return est.fit(y, t, X=None, W=W)


# %% [markdown]
# The checkpoint fits your `LinearDML` on the app data (10 models, as in Exercise 2) and
# compares it with your Exercise 3 result: the estimate to a millionth, the 95% interval to a
# ten-thousandth. Then the true overlap-weighted effect must lie inside EconML's interval.

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    est, econ_ate, econ_lo, econ_hi = dml.check_linear_dml(
        fit_linear_dml, y, t, W, folds, model_y, model_t, SEED, mine, target=overlap_truth())
    print(f"EconML: ${econ_ate:.6f} [${econ_lo:.4f}, ${econ_hi:.4f}]."
          f" Yours: ${mine['theta']:.6f} [${mine['ci_low']:.4f}, ${mine['ci_high']:.4f}].")
    print(est.intercept__inference().summary_frame(alpha=0.05).round(4).to_string())
    print("Out-of-fold R² of model_y per fold:", np.round(np.ravel(est.nuisance_scores_y), 3).tolist())
    print("Out-of-fold accuracy of model_t per fold:", np.round(np.ravel(est.nuisance_scores_t), 3).tolist())

# %% [markdown]
# **Run: a real benchmark (provided).** One way to test an observational method on real data
# is to take a randomized experiment, whose simple comparison is the right answer, spoil the
# randomization with a known rule, and see whether the method recovers the experiment (the
# design of LaLonde 1986). The data: Hillstrom's Womens E-Mail and No E-Mail customers (Module 6),
# outcome `visit` (visited the website within two weeks). The **benchmark** is the difference in
# visit rates on all of them, in **percentage points** (pp: a difference between two
# percentages, so 10% to 14.5% is 4.5 pp).
#
# The rule, applied by the provided `confound(df, seed)`: keep every No E-Mail customer; keep a
# Womens E-Mail customer with probability 0.9 if `recency <= 3` (months since the last purchase)
# or `history >= 250` (dollars spent last year), and with probability 0.2 otherwise. The
# e-mailed customers left in the sample are then mostly recent or big spenders, who visit more
# anyway.
#
# The next cell downloads Hillstrom on first use (about 4 MB, checked with SHA-256, then
# cached), prints the benchmark and spoils the experiment.

# %%
hill = load_hillstrom()
hill = hill[hill["segment"].isin(["Womens E-Mail", "No E-Mail"])].reset_index(drop=True)
hill["treatment"] = (hill["segment"] == "Womens E-Mail").astype(int)
if QUICK:  # a seeded random half of the customers
    half = np.sort(np.random.default_rng(SEED).choice(len(hill), len(hill) // 2, replace=False))
    hill = hill.iloc[half].reset_index(drop=True)
benchmark, benchmark_se = dml.visit_benchmark(hill)
print(f"Hillstrom, Womens E-Mail vs No E-Mail: {len(hill):,} customers. Benchmark: e-mail raised"
      f" the visit rate by {100 * benchmark:.2f} pp (standard error {100 * benchmark_se:.2f}).")


def confound(df, seed):
    """Spoil the experiment: keep every control; keep an e-mailed customer with probability 0.9
    if recent (recency <= 3) or a big spender (history >= 250), else 0.2."""
    favored = (df["recency"] <= 3) | (df["history"] >= 250)
    p_keep = np.where(favored, 0.9, 0.2)
    u = np.random.default_rng(seed).uniform(size=len(df))
    return df[(df["treatment"] == 0) | (u < p_keep)].reset_index(drop=True)


spoiled = confound(hill, SEED)
kept = dml.kept_shares(hill, spoiled)
print(f"Kept: {kept['control']:.2f} of controls, {kept['favored']:.2f} of e-mailed recent or big"
      f" spenders, {kept['other']:.2f} of other e-mailed customers.")
print(f"The spoiled sample: {len(spoiled):,} customers, {spoiled['treatment'].mean():.1%} e-mailed.")

# %% [markdown]
# **Predict (2 of 2).** On the spoiled sample, will the naive difference in visit rates be above,
# below or about the benchmark printed above? Write your guess in the cell below.
#
# The next cell runs *your* `baseline_estimates` and `fit_linear_dml` on the spoiled sample, with
# the same boosted models and five stratified folds built on the sample.

# %%
y_s = spoiled["visit"].to_numpy(dtype=float)
t_s = spoiled["treatment"].to_numpy()
W_s = dml.hillstrom_features(spoiled)  # Module 11's nine Hillstrom features
folds_s = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(W_s, t_s))
base_s = baseline_estimates(y_s, t_s, W_s)
est_s = fit_linear_dml(y_s, t_s, W_s, folds_s, model_y, model_t, SEED)
lo_s, hi_s = (float(np.ravel(v)[0]) for v in est_s.ate_interval(alpha=0.05))
table_s = pd.DataFrame({
    "estimate (pp)": [benchmark, *base_s["estimate"], float(np.ravel(est_s.ate())[0])],
    "95% CI low": [benchmark - 1.96 * benchmark_se, *base_s["ci_low"], lo_s],
    "95% CI high": [benchmark + 1.96 * benchmark_se, *base_s["ci_high"], hi_s],
}, index=["benchmark (the experiment)", "naive (spoiled)", "regression (spoiled)", "LinearDML (spoiled)"])
print((100 * table_s).round(2).to_string())

# %% tags=["checkpoint"]
with workshop.checkpoint(5, label="5 on Hillstrom"):
    dml.check_hillstrom(kept, (base_s.loc["naive", "ci_low"], base_s.loc["naive", "ci_high"]),
                        (lo_s, hi_s), benchmark)

# %% [markdown]
# **Explain.** Compare with your two guesses. On the spoiled sample the naive comparison missed
# the experiment and `LinearDML` recovered it. What did we know about this sample that you never
# know at work? Point to the rule above.
#
# <details><summary>Why this solution works</summary>
#
# We built the selection, so we *know* it depends only on columns the model sees (`recency` and
# `history`). In real observational data that is an assumption, and nothing in the data can
# confirm it. LaLonde (1986) found that many econometric methods failed to reproduce an
# experiment's results, and in 15 advertising experiments at Facebook, matching and regression
# methods overestimated the experimental effects (Gordon et al. 2019). Linear regression also
# came close here: visiting rises smoothly with past spend and recency, so linear controls
# capture most of this selection; in the app data the confounding ran through a sharp curve. One
# more reason this benchmark works: the e-mail's effect on visits is about the same for the
# customers the rule favors and for the rest (4.55 and 4.50 pp on all of Hillstrom's Womens
# E-Mail and No E-Mail customers), so reweighting does not change the target. A rule that
# selected on `womens` (customers who bought women's merchandise last year: 7.31 pp against
# 1.11 pp for the others) would make the sample's overlap-weighted effect differ from the
# benchmark even with perfect adjustment. On the library: it matched your Exercise 3 numbers because it used
# your folds and models; with its default `cv=2`, or with `mc_iters` (repeated cross-fitting
# with new folds), it would differ. Its interval is slightly wider only because its default
# standard error (HC1) multiplies HC0 by √(n / (n − 1)). In practice use the library (tested,
# with more inference options and heterogeneous effects through `X`), and keep the two-step
# version in your head to check it.
# </details>

# %% [markdown]
# ## Decision · Pay customers to install?
#
# **The decision.** Should the retailer pay customers a \$2.50 credit for installing the app?
# A **persuaded install** is an install that happens because of the credit and would not have
# happened without it. The credit pays back within 26 weeks if the extra spend per persuaded
# install, times the 30% gross margin (Module 5), exceeds \$2.50. The **break-even effect** is
# the extra spend per persuaded install at which the margin pays back the credit:
# \$2.50 ÷ 0.30 = \$8.33 of spend over 26 weeks.
#
# **The rule**, fixed before any number: offer the \$2.50 credit only if the lower end of the
# 95% CI of incremental margin per persuaded install is above \$2.50 (the same as: the effect's
# CI lies above \$8.33). The largest credit the rule allows is that lower end.
#
# **It assumes** (1) every common cause of install and spend is in the data; (2) everyone who
# installs after the offer is persuaded by it (customers who did not install in the launch
# window would not install in the next 26 weeks without the credit), so the credit is the cost
# per persuaded install; if some would install anyway, the cost per persuaded install is the
# credit divided by the persuaded share, which only a randomized test of the credit can measure
# (Module 6); (3) the customers the credit persuades gain about as much as the customers the
# estimate describes (Exercise 4's profile). For a small credit, (3) holds approximately in this
# synthetic world, because of the form of its install model; in general it is an assumption.
#
# Run the cells, write your sentence, then look at the truth.
#
# **Run: what a hidden confounder would do (provided).** The next cell refits $\hat q$ and
# $\hat e$ without `mobile_share`, as if the CRM extract had no web analytics, and, for
# contrast, without `email_open_rate`, and prints each estimate and its shift from yours.
# `mobile_share` drives both install and spend; `email_open_rate` drives only install.

# %%
hidden = {}
for left_out in ["mobile_share", "email_open_rate"]:
    cols_kept = [i for i, c in enumerate(FEATURES) if c != left_out]
    W_drop = W[:, cols_kept]
    r = dml_estimate(y, t, cross_fit(model_y, W_drop, y, folds), cross_fit(model_t, W_drop, t, folds))
    hidden[left_out] = r
    print(f"Without {left_out:<15}: ${r['theta']:.2f} [${r['ci_low']:.2f}, ${r['ci_high']:.2f}],"
          f" shift {r['theta'] - mine['theta']:+.2f} from yours (${mine['theta']:.2f});"
          f" the true overlap-weighted effect, ${overlap_truth():.2f},"
          f" {'inside' if r['ci_low'] <= overlap_truth() <= r['ci_high'] else 'outside'} the interval.")
shift = hidden["mobile_share"]["theta"] - mine["theta"]

# %% tags=["checkpoint"]
with workshop.checkpoint(label="hidden confounder"):
    dml.check_hidden_confounder(shift, min_shift=3.0)

# %% [markdown]
# The decision cell turns your Exercise 3 interval into money and applies the rule; then it
# applies the same rule to your interval moved down by the `mobile_share` shift: a common cause
# that strong, unobserved and pushing the same way.

# %%
MARGIN = truth["decision"]["margin"]  # 30% gross margin on spend (Module 5)
CREDIT = truth["decision"]["credit"]  # dollars paid per install
BREAK_EVEN = CREDIT / MARGIN  # extra spend per persuaded install at which the credit pays back

effect, effect_lo, effect_hi = mine["theta"], mine["ci_low"], mine["ci_high"]
m, m_lo, m_hi = MARGIN * effect, MARGIN * effect_lo, MARGIN * effect_hi
offer = m_lo > CREDIT
s_lo, s_hi, s_point = effect_lo - shift, effect_hi - shift, effect - shift
offer_shifted = MARGIN * s_lo > CREDIT
mode = "QUICK run: treat this as a rough answer" if QUICK else "FULL run"

print(f"Number            : each persuaded install adds ${m:.2f} of margin over 26 weeks"
      f" (95% CI ${m_lo:.2f} to ${m_hi:.2f}):")
print(f"                    an effect of ${effect:.2f} of spend (95% CI ${effect_lo:.2f} to"
      f" ${effect_hi:.2f}) at a {MARGIN:.0%} margin.")
print(f"                    Break-even: ${CREDIT:.2f} credit / {MARGIN:.0%} = ${BREAK_EVEN:.2f} of"
      f" extra spend; the estimate is ${abs(effect - BREAK_EVEN):.2f}"
      f" {'above' if effect > BREAK_EVEN else 'below'} it.")
print(f"                    Per 1,000 persuaded installs: about ${1000 * m:,.0f} of margin"
      f" (${1000 * m_lo:,.0f} to ${1000 * m_hi:,.0f}) for ${1000 * CREDIT:,.0f} of credits.")
print(f"Rule              : offer the ${CREDIT:.2f} credit only if the lower end of the 95% CI of"
      f" margin per persuaded install is above ${CREDIT:.2f} (the same as: the effect's CI lies"
      f" above ${BREAK_EVEN:.2f}).")
print("Assumes           : (1) every common cause of install and spend is in the data; (2) everyone"
      " who installs after the offer is persuaded by it; (3) they gain about as much as the"
      " customers the estimate describes (Exercise 4).")
if offer:
    print(f"Recommendation    : offer the credit: the lower end, ${m_lo:.2f}, is above ${CREDIT:.2f}."
          f" The largest credit the rule allows is ${m_lo:.2f}.")
else:
    print(f"Recommendation    : do not offer the credit: the lower end, ${m_lo:.2f}, is not above"
          f" ${CREDIT:.2f}. A randomized test of the credit (Module 6) would settle it.")
print(f"Hidden confounder : leaving out mobile_share moved the estimate by {shift:+.2f}. A common"
      " cause that strong, unobserved and pushing the same way, would shift your interval to"
      f" ${s_lo:.2f} to ${s_hi:.2f} of spend (${MARGIN * s_lo:.2f} to ${MARGIN * s_hi:.2f} of"
      f" margin): by the rule, {'still offer' if offer_shifted else 'do not offer'}. The point"
      f" estimate, ${s_point:.2f}, would {'stay above' if s_point > BREAK_EVEN else 'fall below'}"
      f" the ${BREAK_EVEN:.2f} break-even.")
print("What would change it: a common cause missing from the data (above); customers who would"
      " install anyway taking the credit (a randomized test of the credit measures that, Module 6);"
      " a larger credit, which reaches customers who gain less.")
print(f"Mode              : {mode}.")
print("Your sentence     : ________________________________________________")

# %% [markdown]
# **Decide.** Write your sentence:
#
# ```text
# (Offer / Do not offer) the $2.50 install credit: each persuaded install adds about $____ of
# margin over 26 weeks (95% CI $____ to $____). This assumes every common cause of install and
# spend is in our data; the one I fear most is ____, and if it were as strong as mobile_share
# the answer would be ____.
# ```
#
# Name the confounder you fear most: for example purchase intent at launch (customers about to
# buy install the app to do it), or phone use, if the extract lacks it. Then run the next cell,
# which reveals the truth the data were generated from.

# %%
inc = truth["incentive"]
by_k = {row["odds_multiplier"]: row for row in inc["table"]}
print(f"A credit that doubles the odds of installing (a customer with a 20% chance would have a"
      f" 33% chance) persuades {inc['share_persuaded']:.1%} of customers, whose true average effect"
      f" is ${inc['effect_persuaded']:.2f}: a true margin of ${MARGIN * inc['effect_persuaded']:.2f}"
      f" per persuaded install against the ${CREDIT:.2f} credit.")
print(f"Your 95% CI for the effect, ${effect_lo:.2f} to ${effect_hi:.2f},"
      f" {'contains' if effect_lo <= inc['effect_persuaded'] <= effect_hi else 'misses'} it.")
print(f"For comparison: the true average effect on non-installers is"
      f" ${truth['estimands']['atc']['value']:.2f}; a credit that multiplied the odds by ten would"
      f" persuade {by_k[10.0]['share_persuaded']:.1%} of customers with a true average effect of"
      f" ${by_k[10.0]['effect_persuaded']:.2f}.")
print("The credit really pays in this synthetic world; the lesson is how close one hidden"
      " variable comes to overturning it, and that bigger incentives reach customers who gain less.")

# %% [markdown]
# ## Stretch (optional) · Who gains most, a different average, and how strong a confounder
#
# Nothing later depends on these.
#
# 1. **Who gains most?** Fit `CausalForestDML(model_y=..., model_t=..., discrete_treatment=True,
#    cv=folds, n_estimators=200, min_samples_leaf=20, random_state=SEED)` with `X` =
#    `orders_12m, days_since_last_order, avg_order_value, loyalty_member, mobile_share` (the
#    effect may vary with these) and `W` = the other four covariates. Compare `effect(X)` with
#    `true_effect`: the Spearman correlation, and the mean by quartile of `true_baseline`. This
#    builds on Module 11, now with a learned propensity.
# 2. **A different average.** `LinearDRLearner(model_propensity=..., model_regression=...,
#    cv=folds, min_propensity=0.01, random_state=SEED)`, fitted with `X=None, W=W`, estimates
#    the average effect for *all* customers (doubly robust). Compare its interval with the true
#    ATE, then try other `min_propensity` values and watch the interval: this estimator divides
#    by $\hat e$.
# 3. **How strong would a hidden confounder have to be?** EconML 0.17's
#    `est.robustness_value(null_hypothesis=BREAK_EVEN)` and `est.sensitivity_interval(alpha=0.05,
#    c_y=..., c_t=...)`, which its documentation says are based on Chernozhukov, Cinelli, Newey,
#    Sharma & Syrgkanis ("Long story short"). The documentation describes `c_y` and `c_t` only
#    as "the level of confounding" from 0 to 1: read the paper's definitions before quoting a
#    number, and compare with the `mobile_share` experiment above.

# %% tags=["solution"]
# @title Stretch solutions — try them yourself first { display-mode: "form" }
from econml.dml import CausalForestDML
from econml.dr import LinearDRLearner
from scipy.stats import spearmanr

# 1. Who gains most? (EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.)
X_cols = ["orders_12m", "days_since_last_order", "avg_order_value", "loyalty_member", "mobile_share"]
x_idx = [FEATURES.index(c) for c in X_cols]
w_idx = [i for i in range(len(FEATURES)) if i not in x_idx]
forest = CausalForestDML(model_y=model_y, model_t=model_t, discrete_treatment=True, cv=folds,
                         n_estimators=200, min_samples_leaf=20, random_state=SEED)
forest.fit(y, t, X=W[:, x_idx], W=W[:, w_idx])
tau_hat = np.ravel(forest.effect(W[:, x_idx]))
quartile = pd.qcut(customers["true_baseline"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
print(f"1. Spearman correlation of the forest's effects with the true effects:"
      f" {spearmanr(tau_hat, tau_true).statistic:.2f}")
print(pd.DataFrame({"forest": pd.Series(tau_hat).groupby(quartile.to_numpy(), observed=True).mean(),
                    "truth": pd.Series(tau_true).groupby(quartile.to_numpy(), observed=True).mean()})
      .round(2).rename_axis("quartile of true_baseline").to_string())

# 2. A different average (doubly robust, all customers)
dr = LinearDRLearner(model_propensity=model_t, model_regression=model_y, cv=folds,
                     min_propensity=0.01, random_state=SEED).fit(y, t, X=None, W=W)
dr_lo, dr_hi = (float(np.ravel(v)[0]) for v in dr.ate_interval(alpha=0.05))
print(f"2. Doubly robust average effect: ${float(np.ravel(dr.ate())[0]):.2f} [${dr_lo:.2f}, ${dr_hi:.2f}];"
      f" true ATE ${true_avg['ATE'][0]:.2f}.")

# 3. How strong would a hidden confounder have to be?
print(f"3. Robustness value against the break-even ${BREAK_EVEN:.2f}:"
      f" {est.robustness_value(null_hypothesis=BREAK_EVEN, alpha=0.05):.3f};"
      f" against 0: {est.robustness_value(null_hypothesis=0, alpha=0.05):.3f}.")
for c in (0.01, 0.05):
    lo_c, hi_c = (float(np.ravel(v)[0]) for v in est.sensitivity_interval(alpha=0.05, c_y=c, c_t=c))
    print(f"   Sensitivity interval with c_y = c_t = {c}: {lo_c:.2f} to {hi_c:.2f} dollars")
