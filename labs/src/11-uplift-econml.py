# ---
# jupyter:
#   api_checked: |
#     2026-10-09, against the installed source of econml 0.17.0, scikit-learn 1.6.1 and
#     scipy 1.16.3 (.venv/lib/python3.13/site-packages), then by running every call below:
#     econml.dml.CausalForestDML(*, model_y, model_t, discrete_treatment=False, drate=True,
#       n_estimators=100, subforest_size=4, random_state=None, ...); n_estimators must be
#       divisible by subforest_size. Methods: fit(Y, T, *, X, W=None, ...) returns self;
#       effect(X, *, T0=0, T1=1); effect_interval(X, *, T0=0, T1=1, alpha=0.05);
#       ate_interval(X, *, T0=0, T1=1, alpha=0.05) (stderr of the CATE at the mean X: EconML
#       calls it a conservative upper bound); ate__inference() (doubly robust ATE on the
#       training data, from out-of-bag predictions; conf_int(alpha=0.05)).
#     econml.metalearners: SLearner(*, overall_model), TLearner(*, models),
#       XLearner(*, models, cate_models=None, propensity_model=LogisticRegression());
#       fit(Y, T, *, X); TLearner keeps its fitted outcome models in .models
#       ([control, treated], clones of the one model given).
#     sklearn.ensemble.HistGradientBoostingRegressor / Classifier(..., random_state=None);
#     sklearn.linear_model.LogisticRegression(max_iter=100, ...);
#     scipy.stats.spearmanr(a, b).statistic.
#     mktstats.uplift (this repository): qini_curve, qini_coefficient(normalize=False), auuc,
#       targeting_rule, policy_value_true.
#   note: This header is notebook metadata for maintainers; scripts/gen_notebooks.py drops it.
# ---

# %% [markdown]
# # Part B · Uplift models against the truth
#
# **Order of work.** This lab has two notebooks in one 55-minute slot, and this Python notebook
# comes second. You have just run the install cell above. Now open the R notebook
# `11-grf-causal-forest` (from the
# [notebooks page](https://project-delphi.github.io/marketing-statistics-workshop/notebooks.html))
# and work Part A there: Exercises 1 to 3, about 20 minutes. About 28 minutes into the lab,
# come back here for Part B (Exercises 1 to 3 of this notebook, about 20 minutes) and the
# decision (5 minutes).
#
# **The experiment.** Our retailer e-mailed an offer to a random half of 20,000 customers and
# recorded what each spent over the next two weeks. The customers look like Hillstrom's
# (Part A), but we generated them, so for every customer we know the true effect of the offer
# on their spend, `true_cate`. That lets us grade each uplift model against the truth before
# trusting it on real data.
#
# **Terms.**
#
# - Treatment $T_i \in \{0, 1\}$: customer $i$ got the offer. Outcome $Y_i$: dollars spent in
#   the two weeks after. Features $X_i$: recency, history, mens, womens, zip code type, newbie,
#   channel, as in Hillstrom.
# - The **conditional average treatment effect** (CATE, also *uplift*) is
#   $\tau(x) = E[Y(1) - Y(0) \mid X = x]$: how much more a customer like $x$ spends *because of*
#   the offer. An uplift model estimates it as $\hat\tau(x)$.
# - **Held out**: every grade in this notebook (accuracy, Qini, profit) uses customers the
#   models never saw.
#
# **QUICK.** With QUICK on, the causal forest grows 200 trees instead of 1,000. Its estimates
# are noisier and the numbers will not match the module page; a decision near its threshold can
# flip. That is a lesson about sample size, not a bug.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import spearmanr

from mktstats import uplift

SEED = 2026  # one visible seed: the 50/50 split, every model, the random scores
N_TREES = 200 if QUICK else 1000  # causal forest trees; must be divisible by 4 (subforest_size)
print(f"QUICK = {QUICK}: the causal forest grows {N_TREES} trees.")

# %% [markdown]
# ## Exercise 1 · How close are the effect estimates? (8 minutes)
#
# A provided cell below fits four uplift models on the training half, once:
#
# - **S-learner** ("single"): one model of $Y$ on $X$ and $T$ together;
#   $\hat\tau(x)$ = its prediction with $T = 1$ minus its prediction with $T = 0$.
# - **T-learner** ("two"): one model per arm; $\hat\tau(x)$ = the difference of their
#   predictions.
# - **X-learner**: a T-learner first, then models of each customer's imputed effect, blended
#   with the propensity score (the probability of getting the offer).
# - **Causal forest (DML)**: double machine learning first removes what $X$ predicts about $Y$
#   and about $T$, then grows a forest that splits where the *effect* differs (EconML's
#   `CausalForestDML`).
#
# All four use gradient-boosted trees as their base models.
#
# **Predict.** Which model will rank customers best by their true effect: S-learner,
# T-learner, X-learner or causal forest? Write one name down.
#
# **Task.** Write `cate_accuracy(cate_hat, true_cate)`. `cate_hat` is a DataFrame with one
# column per model holding $\hat\tau$ for each held-out customer; `true_cate` is an array of
# their true effects. Return a DataFrame with one row per model (the index is `cate_hat`'s
# column names, in order) and two columns:
#
# - `rmse`: the root mean squared error $\sqrt{\operatorname{mean}((\hat\tau - \tau)^2)}$, in
#   dollars;
# - `spearman`: the Spearman rank correlation of $\hat\tau$ and $\tau$ (1 = same order,
#   0 = unrelated), `scipy.stats.spearmanr(a, b).statistic`.

# %% tags=["exercise"]
def cate_accuracy(cate_hat, true_cate):
    # TODO 1: one row per column of cate_hat, with rmse and spearman against true_cate
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def cate_accuracy(cate_hat, true_cate):
    true_cate = np.asarray(true_cate, dtype=float)
    rows = {}
    for name in cate_hat.columns:
        estimate = cate_hat[name].to_numpy(dtype=float)
        rows[name] = {
            "rmse": float(np.sqrt(np.mean((estimate - true_cate) ** 2))),
            "spearman": float(spearmanr(estimate, true_cate).statistic),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# %% [markdown]
# The checkpoint tries your function on four made-up customers first; the models come after.

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    toy_hat = pd.DataFrame({"same order": [1.0, 2.0, 3.0, 4.0], "reversed": [4.0, 3.0, 2.0, 1.0]})
    toy_true = np.array([1.0, 2.0, 3.0, 5.0])
    toy = cate_accuracy(toy_hat, toy_true)
    checks.columns(toy, ["rmse", "spearman"], name="cate_accuracy(toy_hat, toy_true)")
    assert list(toy.index) == ["same order", "reversed"], (
        f"The rows should be the models, in cate_hat's column order: ['same order', 'reversed'];"
        f" got {list(toy.index)}. Build one row per column of cate_hat and use the column name as"
        " the index."
    )
    checks.close(toy.loc["same order", "rmse"], 0.5, abs=1e-9,
                 name="The RMSE of 'same order' (errors 0, 0, 0, -1: sqrt(1/4))")
    checks.close(toy.loc["reversed", "rmse"], np.sqrt(27 / 4), abs=1e-9,
                 name="The RMSE of 'reversed' (errors 3, 1, -1, -4: sqrt(27/4))")
    checks.close(toy.loc["same order", "spearman"], 1.0, abs=1e-9,
                 name="The Spearman correlation of 'same order' (same ranking; a Pearson"
                      " correlation would give 0.98, so check you used spearmanr)")
    checks.close(toy.loc["reversed", "spearman"], -1.0, abs=1e-9,
                 name="The Spearman correlation of 'reversed' (opposite ranking)")

# %% [markdown]
# ### Run: the data, the split and four uplift models (provided)
#
# The next cell loads the synthetic experiment and its truth, and splits the customers 50/50 at
# random (seed `SEED`) into a **training half**, which fits the models, and an **evaluation
# half**, which only grades them. A model graded on the customers it was fitted to looks
# better than it is.

# %%
from mktstats.data import load_synthetic, load_truth

truth = load_truth()["email_experiment"]
email = load_synthetic("email_experiment")
MARGIN, OFFER_COST = truth["margin"], truth["offer_cost"]


def features(df):
    """The numeric feature matrix: Hillstrom's columns, categories as 0/1 indicators
    (reference levels Urban and Phone)."""
    return df[["recency", "history", "mens", "womens", "newbie"]].astype(float).assign(
        zip_rural=(df["zip_code"] == "Rural").astype(float),
        zip_suburban=(df["zip_code"] == "Surburban").astype(float),
        channel_web=(df["channel"] == "Web").astype(float),
        channel_multichannel=(df["channel"] == "Multichannel").astype(float),
    )


in_train = np.random.default_rng(SEED).permutation(len(email)) < len(email) // 2
train, evaluation = email[in_train], email[~in_train]
X_train, X_eval = features(train).to_numpy(), features(evaluation).to_numpy()
y_train, t_train = train["spend"].to_numpy(), train["treatment"].to_numpy()
y_eval, t_eval = evaluation["spend"].to_numpy(), evaluation["treatment"].to_numpy()
true_cate_eval = evaluation["true_cate"].to_numpy()

print(f"Training half: {len(train):,} customers; evaluation half: {len(evaluation):,}.")
print(f"Share who got the offer: {t_train.mean():.1%}. Share who bought: {(y_train > 0).mean():.1%}.")
print(f"True average effect (ATE) over all 20,000 customers: ${truth['ate']:.2f} of spend.")
print(f"Margin {MARGIN:.0%}, offer cost ${OFFER_COST:.2f}: worth sending when the effect exceeds"
      f" ${OFFER_COST / MARGIN:.2f}. Truly worth it for {truth['share_should_treat']:.1%} of customers.")

# %% [markdown]
# **Run.** The next cell fits the four models on the training half (about 40 seconds in the
# recorded Colab FULL run, with 1,000 trees; QUICK grows 200). While it runs, look at your
# prediction for Exercise 1.

# %%
from econml.dml import CausalForestDML
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


def make_models(n_trees, seed):
    """The four uplift models, not yet fitted, each with gradient-boosted trees inside."""

    def boosted():
        return HistGradientBoostingRegressor(random_state=seed)

    return {
        "S-learner": SLearner(overall_model=boosted()),
        "T-learner": TLearner(models=boosted()),
        "X-learner": XLearner(models=boosted(), propensity_model=LogisticRegression(max_iter=1000)),
        "causal forest": CausalForestDML(
            model_y=boosted(),
            model_t=HistGradientBoostingClassifier(random_state=seed),  # T is binary: a classifier
            discrete_treatment=True,
            n_estimators=n_trees,
            random_state=seed,
        ),
    }


def fit_models(models, y, t, X):
    """Fit each model on the same data and say how long it took."""
    # EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.
    for name, model in models.items():
        started = time.time()
        model.fit(y, t, X=X)
        print(f"{name:>14} fitted in {time.time() - started:5.1f} s")
    return models


models = fit_models(make_models(N_TREES, SEED), y_train, t_train, X_train)
cate_hat = pd.DataFrame({name: model.effect(X_eval) for name, model in models.items()})
cf = models["causal forest"]

# %% [markdown]
# Now grade the four models against the true effects of the held-out customers, and compare the
# causal forest's average effect with the truth. `ate__inference()` is its doubly robust
# estimate of the ATE (computed on the training half from *out-of-bag* predictions: each
# customer's prediction comes only from trees that did not see that customer);
# `ate_interval(X_eval)` averages the forest's predictions over the evaluation half, with a
# deliberately conservative interval (EconML's documentation calls it an upper bound).

# %%
accuracy = cate_accuracy(cate_hat, true_cate_eval)
ate_lo, ate_hi = (float(np.ravel(v)[0]) for v in cf.ate__inference().conf_int(alpha=0.05))
avg_lo, avg_hi = (float(np.ravel(v)[0]) for v in cf.ate_interval(X_eval, alpha=0.05))
print(accuracy.assign(spread=cate_hat.std()).round(3))
print(f"(spread: standard deviation of each model's estimates; of the true effects: {true_cate_eval.std():.3f})")
print(f"\nCausal forest, doubly robust ATE: ${float(np.ravel(cf.ate_)[0]):.2f},"
      f" 95% interval [${ate_lo:.2f}, ${ate_hi:.2f}]")
print(f"Causal forest, mean prediction on the evaluation half: ${cate_hat['causal forest'].mean():.2f},"
      f" 95% interval [${avg_lo:.2f}, ${avg_hi:.2f}]")
print(f"Truth: ATE ${truth['ate']:.2f} (all customers), ${true_cate_eval.mean():.2f} (evaluation half)")

# %% [markdown]
# The next check grades the provided fit as well as your function: the causal forest's 95%
# doubly robust interval must contain the true ATE, and its rank correlation must be within
# 0.05 of what the reference run measured (0.61 with QUICK's 200 trees, 0.62 with 1,000).

# %% tags=["checkpoint"]
with workshop.checkpoint(1, label="1 on the four models"):
    assert list(accuracy.index) == list(cate_hat.columns), (
        f"Expected one row per model {list(cate_hat.columns)}; got {list(accuracy.index)}."
    )
    for name in cate_hat.columns:
        est = cate_hat[name].to_numpy()
        checks.close(accuracy.loc[name, "rmse"], np.sqrt(np.mean((est - true_cate_eval) ** 2)),
                     rel=1e-9, name=f"The RMSE of the {name}")
        checks.close(accuracy.loc[name, "spearman"], spearmanr(est, true_cate_eval).statistic,
                     rel=1e-9, name=f"The Spearman correlation of the {name}")
    assert ate_lo <= truth["ate"] <= ate_hi, (
        f"The true ATE, ${truth['ate']:.2f}, is outside the causal forest's 95% interval"
        f" [${ate_lo:.2f}, ${ate_hi:.2f}]. The fit is provided: tell the instructor."
    )
    floor = (0.61 if QUICK else 0.62) - 0.05
    rho = accuracy.loc["causal forest", "spearman"]
    assert rho >= floor, (
        f"The causal forest's rank correlation with the true effects is {rho:.3f}, below {floor:.2f}"
        " (the reference run's value minus 0.05). The fit is provided: check that N_TREES and SEED"
        " were not changed, then tell the instructor."
    )

# %% [markdown]
# **Explain.** Compare with your prediction. Which model has the lowest RMSE, and is it also the
# one with the highest rank correlation? Then compare the `spread` column with the spread of the
# true effects printed under the table: why is the T-learner's spread the largest?
#
# <details><summary>Why this solution works</summary>
#
# RMSE measures how far the estimated effects are from the true ones in dollars; the Spearman
# correlation only asks whether the order is right, which is what a targeting list needs. Every
# model's estimates are more spread out than the true effects: fewer than 5% of customers buy,
# so much of what a model learns from spend is noise, and the noise shows up as made-up
# differences between customers. The T-learner fits a separate model to each arm, so the noise
# of both lands in their difference; the S-learner shares one model across the arms and is less
# noisy. The X-learner and the DML causal forest target the difference itself (Künzel et al.
# 2019; Chernozhukov et al. 2018), and the forest's *honest* splits (each tree chooses its
# splits on one part of its sample and estimates effects on another), averaged over many trees,
# damp the noise most, which is why its RMSE is the lowest here. More customers, or an outcome
# with less noise than spend, would improve all four.
# </details>

# %% [markdown]
# ## Exercise 2 · Qini curves on held-out data (6 minutes)
#
# Rank the held-out customers by a model's $\hat\tau$, highest first, and walk down the list.
# After the top $k$ customers the **Qini curve** is
#
# $$g(k) = Y_T(k) - Y_C(k)\,\frac{N_T(k)}{N_C(k)},$$
#
# where $Y_T(k), Y_C(k)$ are the total spend of the treated and control customers among the top
# $k$, and $N_T(k), N_C(k)$ their numbers. $g(k)$ estimates the extra spend the offer caused
# among the treated customers in the top $k$, in dollars (Radcliffe & Surry 2011). Random
# targeting gives the straight line from 0 to $g(n)$. The **Qini coefficient** is the area
# between the curve and that line, with the share of customers targeted (0 to 1) on the x
# axis: larger is a better ranking, and about 0 is no better than random. It measures
# ranking, not the size of the effect.
#
# The scores to compare, one per held-out customer: the four models' $\hat\tau$, a random score
# (the floor) and the true effect itself (the best possible ranking).

# %%
scores = {name: cate_hat[name].to_numpy() for name in cate_hat.columns}
scores["random"] = np.random.default_rng(SEED).uniform(size=len(y_eval))
scores["true CATE"] = true_cate_eval

# %% [markdown]
# **Predict.** Will the model with the best Spearman correlation in Exercise 1 also have the
# largest Qini coefficient: yes or no?
#
# **Task.** Write `qini_table(y, t, scores)`: `scores` is a dict {name: array}. Return a
# DataFrame with one row per name (in the dict's order, names as the index) and one column
# `qini` from `uplift.qini_coefficient(y, t, score)`.

# %% tags=["exercise"]
def qini_table(y, t, scores):
    # TODO 2: one row per score, with its Qini coefficient
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def qini_table(y, t, scores):
    return pd.DataFrame(
        {"qini": [uplift.qini_coefficient(y, t, score) for score in scores.values()]},
        index=list(scores),
    )


# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    qini = qini_table(y_eval, t_eval, scores)
    checks.columns(qini, ["qini"], name="qini_table(y_eval, t_eval, scores)")
    assert list(qini.index) == list(scores), (
        f"Expected one row per score, in order {list(scores)}; got {list(qini.index)}."
        " Use the dict's keys as the index."
    )
    for name, score in scores.items():
        checks.close(qini.loc[name, "qini"], uplift.qini_coefficient(y_eval, t_eval, score),
                     rel=1e-9, name=f"The Qini coefficient of {name!r} (normalize=False, the default)")
    assert qini["qini"].idxmax() == "true CATE", (
        f"The true effects should rank customers best, but {qini['qini'].idxmax()!r} has the"
        " largest Qini coefficient. Check that each row uses its own score."
    )
    # How far from 0 a random ranking lands on these customers: 200 random scores.
    noise = np.array([
        uplift.qini_coefficient(y_eval, t_eval, np.random.default_rng(SEED + i).uniform(size=len(y_eval)))
        for i in range(1, 201)
    ])
    random_tolerance = float(np.quantile(np.abs(noise), 0.99))
    assert abs(qini.loc["random", "qini"]) <= random_tolerance, (
        f"The random score's Qini coefficient is {qini.loc['random', 'qini']:,.0f}; 99% of 200 random"
        f" scores land within ±{random_tolerance:,.0f}. Check that the 'random' row uses the random"
        " score."
    )

# %% [markdown]
# The table adds the area under the uplift curve above random (AUUC, the same idea with the
# uplift curve of Gutierrez & Gérardy 2017) and each Qini as a share of the true effects' Qini.
# Then look at the plot: how far each model's curve rises above the dashed random line, and how
# much room is left below the true effects' curve.

# %%
qini_summary = qini.assign(
    auuc=[uplift.auuc(y_eval, t_eval, s, subtract_random=True) for s in scores.values()],
    share_of_true=qini["qini"] / qini.loc["true CATE", "qini"],
)
print(f"Held-out customers: {len(y_eval):,}. A random score lands within ±{random_tolerance:,.0f}"
      " of 0 (99% of 200 random scores).")
print(qini_summary.round({"qini": 0, "auuc": 0, "share_of_true": 2}))

fig, ax = plt.subplots(figsize=(7, 4.5))
styles = {  # line style and marker, so the curves differ without color
    "S-learner": (":", "o"), "T-learner": ("-.", "^"), "X-learner": ((0, (5, 1)), "v"),
    "causal forest": ("-", "s"), "true CATE": ("-", None),
}
for name, (style, marker) in styles.items():
    share, gain = uplift.qini_curve(y_eval, t_eval, scores[name])
    ax.plot(share, gain, linestyle=style, marker=marker, markevery=0.1, markersize=5,
            linewidth=2.8 if name == "true CATE" else 1.6,
            color="black" if name == "true CATE" else None, label=name)
ax.plot([0, 1], [0, gain[-1]], linestyle="--", color="grey", label="random targeting")
ax.set_xlabel("Share of held-out customers targeted, highest estimated effect first")
ax.set_ylabel("Extra spend caused among treated targeted ($)")
ax.set_title(f"Qini curves on the evaluation half (n = {len(y_eval):,})")
ax.legend(frameon=False, fontsize=9)
display(fig)
plt.close(fig)

# %% [markdown]
# **Explain.** Compare with your prediction. Did the best rank correlation also give the largest
# Qini coefficient? The Qini curve is built from observed spend, not from the true effects: what
# does that add to the comparison, and what does the gap between the random line and the curves
# tell you?
#
# <details><summary>Why this solution works</summary>
#
# Spearman's correlation compares the ranking with the true effects, which only synthetic data
# give. The Qini coefficient grades the same ranking with observed spend of treated and control
# customers, which you also have on real data, so it is the metric you can use at work. Because
# observed spend is noisy, two models with similar rank correlations can swap places on Qini,
# and a random score lands anywhere inside the noise band printed above the table. Only
# differences larger than that band mean anything. A Qini coefficient says nothing about
# whether the effects are large enough to pay for the offer: that is Exercise 3.
# </details>

# %% [markdown]
# ## Exercise 3 · Profit of a targeting rule (6 minutes)
#
# Margin $m$ = 0.35 (35 cents of profit per dollar of spend) and offer cost $c$ = \$0.50 come
# from the truth file. The rule: send the offer when $m\,\hat\tau(x) > c$, that is when the
# estimated effect exceeds $c / m \approx$ \$1.43 of spend. Profit per customer of a rule,
# against sending nobody the offer:
#
# $$\frac{1}{n}\sum_{i \text{ targeted}} \left(m\,\tau_i - c\right).$$
#
# With the model's own $\hat\tau_i$ in place of $\tau_i$ this is the profit the model *expects*;
# with the true $\tau_i$ (known here because we generated the data) it is what the rule really
# earns.
#
# **Predict.** Will targeting by the causal forest make more true profit than mailing everyone:
# yes or no?
#
# **Task.** Write `targeting_profit(cate_hat, true_cate, margin, cost)` returning a dict with
#
# - `share_targeted`: the share of customers with `margin * cate_hat > cost`;
# - `expected_profit`: the model's view, the formula with `cate_hat`, dollars per customer;
# - `true_profit`: the truth, the formula with `true_cate`, dollars per customer.
#
# Divide by all $n$ customers, not by the number targeted.

# %% tags=["exercise"]
def targeting_profit(cate_hat, true_cate, margin, cost):
    # TODO 3: who is targeted, the profit the model expects, the profit it really makes
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def targeting_profit(cate_hat, true_cate, margin, cost):
    cate_hat = np.asarray(cate_hat, dtype=float)
    true_cate = np.asarray(true_cate, dtype=float)
    send = margin * cate_hat > cost
    return {
        "share_targeted": float(send.mean()),
        "expected_profit": float(np.mean(send * (margin * cate_hat - cost))),
        "true_profit": float(np.mean(send * (margin * true_cate - cost))),
    }


# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    toy = targeting_profit([5.0, 1.0, 2.0, -1.0], [4.0, 2.0, 0.0, 3.0], margin=0.5, cost=1.0)
    hint = (" Only the first of the 4 toy customers clears the bar (0.5 x 5 = 2.5 > 1; 0.5 x 2 = 1"
            " is not above 1), and profits are divided by all 4 customers.")
    checks.close(toy["share_targeted"], 0.25, abs=1e-12, name="The toy share_targeted." + hint)
    checks.close(toy["expected_profit"], 0.375, abs=1e-12,
                 name="The toy expected_profit, (0.5 x 5 - 1) / 4." + hint)
    checks.close(toy["true_profit"], 0.25, abs=1e-12, name="The toy true_profit, (0.5 x 4 - 1) / 4." + hint)
    cf_hat = cate_hat["causal forest"].to_numpy()
    mine = targeting_profit(cf_hat, true_cate_eval, MARGIN, OFFER_COST)
    rule = uplift.targeting_rule(cf_hat, MARGIN, OFFER_COST)
    checks.close(mine["true_profit"], uplift.policy_value_true(true_cate_eval, rule, MARGIN, OFFER_COST),
                 rel=1e-9, name="The causal forest's true profit per customer (mktstats.uplift agrees?)")
    every_cate = email["true_cate"].to_numpy()
    oracle_all = targeting_profit(every_cate, every_cate, MARGIN, OFFER_COST)["true_profit"]
    checks.close(oracle_all, truth["policy_value_per_customer"]["oracle"], rel=1e-4,
                 name="Targeting all 20,000 customers by their true effect: profit per customer vs truth.json")
    oracle_eval = targeting_profit(true_cate_eval, true_cate_eval, MARGIN, OFFER_COST)["true_profit"]
    for name in cate_hat.columns:
        model_profit = targeting_profit(cate_hat[name], true_cate_eval, MARGIN, OFFER_COST)["true_profit"]
        assert model_profit <= oracle_eval + 1e-12, (
            f"The {name} earns ${model_profit:.3f} per customer, more than targeting by the true"
            f" effects (${oracle_eval:.3f}), which is impossible: check that true_profit uses true_cate."
        )

# %% [markdown]
# Look at the gap between `expected_profit` and `true_profit` for each model, and at where each
# model's true profit falls between mailing everyone and the best possible rule (true CATE).

# %%
rows = {name: targeting_profit(cate_hat[name], true_cate_eval, MARGIN, OFFER_COST) for name in cate_hat}
rows["true CATE (best possible)"] = targeting_profit(true_cate_eval, true_cate_eval, MARGIN, OFFER_COST)
everyone = float(np.mean(MARGIN * true_cate_eval - OFFER_COST))
rows["mail everyone"] = {"share_targeted": 1.0, "expected_profit": np.nan, "true_profit": everyone}
rows["mail nobody"] = {"share_targeted": 0.0, "expected_profit": 0.0, "true_profit": 0.0}
profit = pd.DataFrame.from_dict(rows, orient="index")
print("Dollars per held-out customer, against sending nobody the offer:")
print(profit.round(3))

# %% [markdown]
# **Explain.** Compare with your prediction. For the causal forest, is the profit it expects
# above or below what it really makes? Which part of a model's error moves the cutoff: getting
# the order wrong, or getting the size of the effects wrong?
#
# <details><summary>Why this solution works</summary>
#
# The rule compares each estimated effect with a cutoff, \$1.43. A model can rank customers well
# and still misjudge the size of their effects, which moves who falls above the cutoff: too
# large and it sends offers that do not pay, too small and it skips customers who would have
# paid. The gap between expected and true profit measures that error in dollars. On real data
# you cannot compute the true profit; the Decision cell estimates it with doubly robust scores
# from held-out customers, as you did in R.
# </details>

# %% [markdown]
# ## Decision · Whom to send the offer?
#
# The evaluation half stands in for next month's mailing list: customers the models never saw.
# The cell below works only with what you would have at work (spend, who got the offer, the
# models), then reveals the truth.
#
# **Number.** For each held-out customer a doubly robust (AIPW) score estimates their effect
# without using the truth:
# $\Gamma_i = \hat\mu_1(x_i) - \hat\mu_0(x_i) + \frac{T_i (Y_i - \hat\mu_1(x_i))}{e} -
# \frac{(1 - T_i)(Y_i - \hat\mu_0(x_i))}{1 - e}$, where $\hat\mu_1, \hat\mu_0$ are the
# T-learner's two outcome models (fitted on the training half) and $e$ = 0.5 is the known
# chance of getting the offer. The profit per customer of a rule is the mean of
# $\text{send}_i\,(m\,\Gamma_i - c)$, with a 95% interval of ± 1.96 standard errors.
#
# **Rule** (stated before reading the result): send the offer when margin × estimated uplift
# > cost per offer, here $0.35\,\hat\tau(x) > \$0.50$.
#
# If you wrote down the Hillstrom numbers from the R notebook's Decision cell, type them into
# `HILLSTROM_FROM_R` first (or leave `None`).

# %%
# Your Hillstrom numbers from the R notebook: share targeted (0 to 1) and dollars per 1,000.
HILLSTROM_FROM_R = {"share_targeted": None, "profit_per_1000": None}

control_model, treated_model = models["T-learner"].models  # fitted on the training half
mu0, mu1 = control_model.predict(X_eval), treated_model.predict(X_eval)
e = truth["treat_prob"]  # 0.5: the offer was assigned at random
gamma = mu1 - mu0 + t_eval * (y_eval - mu1) / e - (1 - t_eval) * (y_eval - mu0) / (1 - e)


def mean_interval(terms, z=1.96):
    """Mean and its 95% normal interval (± 1.96 standard errors), per 10,000 customers."""
    mean, se = terms.mean(), terms.std(ddof=1) / np.sqrt(len(terms))
    return 10_000 * mean, 10_000 * (mean - z * se), 10_000 * (mean + z * se)


def usd(x):
    """Whole dollars with the sign in front: -$156, $2,152."""
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


send = uplift.targeting_rule(cf_hat, MARGIN, OFFER_COST)
forest_terms = send * (MARGIN * gamma - OFFER_COST)
everyone_terms = MARGIN * gamma - OFFER_COST
forest_est = mean_interval(forest_terms)
everyone_est = mean_interval(everyone_terms)
gain_est = mean_interval(forest_terms - everyone_terms)
lower90, _ = cf.effect_interval(X_eval, alpha=0.1)
sure = float(np.mean(np.ravel(lower90)[send == 1] > OFFER_COST / MARGIN))

if gain_est[1] > 0:
    advice = (f"send the offer only to the {send.sum():,} customers the causal forest selects"
              f" ({send.mean():.0%}): about {usd(gain_est[0])} more per 10,000 customers than mailing"
              f" everyone (95% interval {usd(gain_est[1])} to {usd(gain_est[2])}).")
elif gain_est[2] < 0:
    advice = (f"mail everyone: the forest's list earns {usd(-gain_est[0])} less per 10,000 customers"
              f" (95% interval {usd(-gain_est[2])} to {usd(-gain_est[1])} less).")
else:
    advice = (f"the held-out data cannot tell the forest's list ({send.mean():.0%} of customers) from"
              f" mailing everyone: {usd(gain_est[0])} per 10,000 customers, 95% interval"
              f" {usd(gain_est[1])} to {usd(gain_est[2])}. The list sends {1 - send.mean():.0%} fewer"
              " offers; a larger holdout would settle which earns more.")

print(f"Number         : the causal forest targets {send.sum():,} of {len(y_eval):,} held-out customers"
      f" ({send.mean():.1%}).")
print(f"                 Estimated profit (AIPW) per 10,000 customers: forest's list {usd(forest_est[0])}"
      f" (95% interval {usd(forest_est[1])} to {usd(forest_est[2])}),")
print(f"                 mail everyone {usd(everyone_est[0])} ({usd(everyone_est[1])} to"
      f" {usd(everyone_est[2])}), mail nobody $0.")
print(f"                 {sure:.0%} of targeted customers have a 90% CATE interval entirely above"
      f" ${OFFER_COST / MARGIN:.2f}.")
if HILLSTROM_FROM_R["share_targeted"] is not None:
    print(f"                 Hillstrom (R): {HILLSTROM_FROM_R['share_targeted']:.0%} targeted,"
          f" ${HILLSTROM_FROM_R['profit_per_1000']} per 1,000 customers.")
print(f"Rule           : send the offer when {MARGIN} x estimated uplift > ${OFFER_COST:.2f} cost per offer.")
print(f"Recommendation : {advice}")
print("What would change it: a different offer cost (the cutoff is cost / margin), or drift in who"
      " receives the e-mails.")
print(f"Mode           : {'QUICK run: treat this as a rough answer' if QUICK else 'FULL run'}.")
print("Your sentence  : ________________________________________________")

# %% [markdown]
# Write your sentence before you run the next cell. Then compare with the truth: what the
# forest's targeting list really earns, and what mailing everyone and the best possible rule
# earn, on the same customers.

# %%
true_forest = uplift.policy_value_true(true_cate_eval, send, MARGIN, OFFER_COST)
true_best = uplift.policy_value_true(
    true_cate_eval, uplift.targeting_rule(true_cate_eval, MARGIN, OFFER_COST), MARGIN, OFFER_COST
)
print(f"True profit per 10,000 customers: forest's list {usd(10_000 * true_forest)}, mail everyone"
      f" {usd(10_000 * everyone)}, best possible {usd(10_000 * true_best)}, mail nobody $0.")
inside = forest_est[1] <= 10_000 * true_forest <= forest_est[2]
print(f"The AIPW 95% interval for the list's profit, {usd(forest_est[1])} to {usd(forest_est[2])},"
      f" {'contains' if inside else 'misses'} the true value.")
print(f"The list really earns {usd(10_000 * (true_forest - everyone))} more than mailing everyone per"
      f" 10,000 customers; the best possible rule would earn {usd(10_000 * (true_best - everyone))} more.")

# %% [markdown]
# ## Stretch (optional) · The same four models on the real Hillstrom test
#
# Fit the same four models on Hillstrom's Womens e-mail and No e-mail arms (Part A's data) and
# compare their Qini coefficients on the held-out half. There is no truth here: only the Qini
# table and the random score's noise band tell you whether a model ranks better than chance.
# The cell downloads Hillstrom once (checked with SHA-256, cached) and grows 200 trees in the
# causal forest; QUICK keeps a random half of the customers. It took about 15 seconds in the
# recorded Colab FULL run.

# %%
from mktstats.data import load_hillstrom

hill = load_hillstrom()
hill = hill[hill["segment"].isin(["Womens E-Mail", "No E-Mail"])].reset_index(drop=True)
if QUICK:
    hill = hill.sample(frac=0.5, random_state=SEED).reset_index(drop=True)
X_h = features(hill).to_numpy()
y_h, t_h = hill["spend"].to_numpy(), (hill["segment"] == "Womens E-Mail").astype(int).to_numpy()
train_h = np.random.default_rng(SEED).permutation(len(hill)) < len(hill) // 2
hill_models = fit_models(make_models(200, SEED), y_h[train_h], t_h[train_h], X_h[train_h])
hill_scores = {name: model.effect(X_h[~train_h]) for name, model in hill_models.items()}
hill_scores["random"] = np.random.default_rng(SEED).uniform(size=int((~train_h).sum()))
hill_qini = qini_table(y_h[~train_h], t_h[~train_h], hill_scores)
hill_noise = np.array([
    uplift.qini_coefficient(y_h[~train_h], t_h[~train_h],
                            np.random.default_rng(SEED + i).uniform(size=int((~train_h).sum())))
    for i in range(1, 201)
])
print(f"Held-out Hillstrom customers: {int((~train_h).sum()):,}. Random scores land within"
      f" ±{np.quantile(np.abs(hill_noise), 0.99):,.0f} of 0 (99% of 200).")
print(hill_qini.round(0))

# %% [markdown]
# Which models clear the random band? Compare with the RATE interval from Part A: do the two
# tools, in two languages, tell the same story about Hillstrom?
