"""Customers who chose to install the retailer's app, with known effects (Module 15, double
machine learning; briefs/15-double-ml.md, "The synthetic world").

The retailer announced a shopping app by e-mail and on the checkout page during four launch
weeks. Customers who open e-mails, shop on their phones or had bought recently were more likely
to install, and loyal, frequent, recent customers also spend most in the next 26 weeks: that is
the confounding. Every arrow into install (T) and spend (Y) starts at a column of the table, so
"no unobserved confounding" holds given all nine covariates.

1. latent activity a ~ Normal(0, 1) (never saved; it only correlates the covariates);
2. covariates: channel, tenure, orders, days since last order, order value, e-mail open rate,
   mobile share, loyalty (formulas in ``FORMULAS``);
3. install: eta(x) = T0 + T_RECENT r30 + T_ORDERS log(1 + orders) + T_MOBILE mobile
   + T_EMAIL email + T_LOYAL loyalty + T_SOCIAL [social] + T_REFERRAL [referral], with
   r30 = exp(-days / 30); e(x) = expit(eta); T = 1[U < e(x)], U ~ Uniform(0, 1);
4. spend without the app: r26(x) = R_SCALE orders (R_BASE + R_RECENT r30)(1 + R_LOYAL loyalty)
   expected orders, mu0(x) = r26 aov (M_BASE + M_MOBILE mobile) expected dollars;
5. effect: tau(x) = TAU0 + TAU_SLOPE mu0(x) dollars over 26 weeks (heavier spenders gain more);
6. outcome: a purchase with probability p(x) = 1 - exp(-r26), which the app does not change;
   given one, spend ~ Gamma(SPEND_SHAPE, mean (mu0 + T tau) / p), rounded to cents; else 0.
   So E[Y | X, T] = mu0(X) + T tau(X) (up to the rounding to cents).

Draws come from one stream in this order: a, channel, tenure, orders, days, order value, e-mail
rate, mobile share, loyalty, the install uniforms U, the purchase uniforms, the spend amounts.

The truth columns ``true_propensity`` (e), ``true_effect`` (tau) and ``true_baseline`` (mu0)
are saved rounded (6, 4 and 4 decimals) and every estimand in the truth is computed from the
saved values, so a lab can recompute any of them on the rows it uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from mktstats.synth._common import SynthResult, make_rng, sig

SEED = 2035
N = 20_000
CHANNELS = ("search", "social", "referral")
CHANNEL_PROBS = (0.45, 0.35, 0.20)
# The nine covariates a model may use (channel as two 0/1 indicators, search the reference).
COVARIATES = ("tenure_months", "orders_12m", "days_since_last_order", "avg_order_value",
              "email_open_rate", "mobile_share", "loyalty_member", "channel_social",
              "channel_referral")
COLUMNS = ("customer_id", "acquisition_channel", "tenure_months", "orders_12m",
           "days_since_last_order", "avg_order_value", "email_open_rate", "mobile_share",
           "loyalty_member", "app_installed", "spend_next_26w", "true_propensity", "true_effect",
           "true_baseline")

PARAMS = {
    # install log-odds eta(x)
    "t0": -4.2, "t_recent": 3.0, "t_orders": 0.5, "t_mobile": 2.4, "t_email": 1.2,
    "t_loyal": 0.5, "t_social": 0.4, "t_referral": -0.2, "recent_days": 30.0,
    # expected orders in the next 26 weeks without the app, r26(x)
    "r_scale": 0.3, "r_base": 0.3, "r_recent": 1.6, "r_loyal": 0.2,
    # expected spend without the app, mu0(x) = r26 * aov * (m_base + m_mobile * mobile)
    "m_base": 0.5, "m_mobile": 1.0,
    # effect tau(x) = tau0 + tau_slope * mu0(x)
    "tau0": 8.0, "tau_slope": 0.12,
    # spend given a purchase ~ Gamma(shape, mean)
    "spend_shape": 3.0,
}

FORMULAS = {
    "a": "a ~ Normal(0, 1), latent activity, not saved; never enters T or Y",
    "acquisition_channel": "Categorical(search 0.45, social 0.35, referral 0.20)",
    "tenure_months": "clip(round(3 + Gamma(shape 2, scale 12)), 3, 96)",
    "orders_12m": "1 + Poisson(exp(0.55 + 0.6 a + 0.15 [referral]))",
    "days_since_last_order": "clip(round(365 Beta(1, 1 + 0.6 orders_12m)), 1, 365)",
    "avg_order_value": "round(exp(Normal(log 55 + 0.15 [referral] + 0.1 a, sd 0.45)), 2)",
    "email_open_rate": "round(Beta(6 mu, 6 (1 - mu)), 3), mu = expit(-0.8 + 0.5 a)",
    "mobile_share": ("round(Beta(4 mu, 4 (1 - mu)), 3), mu = expit(-0.3 + 0.9 [social] "
                     "+ 0.2 [search] - 0.01 (tenure_months - 24))"),
    "loyalty_member": "Bernoulli(expit(-1.3 + 0.7 a + 0.015 tenure_months))",
    "r30": "exp(-days_since_last_order / recent_days)",
    "eta": ("t0 + t_recent r30 + t_orders log(1 + orders_12m) + t_mobile mobile_share "
            "+ t_email email_open_rate + t_loyal loyalty_member + t_social [social] "
            "+ t_referral [referral]"),
    "true_propensity": "e(x) = expit(eta)",
    "app_installed": "T = 1[U < e(x)], U ~ Uniform(0, 1)",
    "r26": "r_scale orders_12m (r_base + r_recent r30) (1 + r_loyal loyalty_member)",
    "true_baseline": "mu0(x) = r26 avg_order_value (m_base + m_mobile mobile_share)",
    "true_effect": "tau(x) = tau0 + tau_slope mu0(x), dollars over 26 weeks",
    "spend_next_26w": ("with probability p = 1 - exp(-r26): Gamma(shape spend_shape, mean "
                       "(mu0 + T tau) / p), rounded to cents; otherwise 0. "
                       "E[Y | X, T] = mu0(X) + T tau(X)"),
}

# Which covariates enter install (eta) and spend (mu0, and so tau) directly. A covariate that
# enters both is a confounder; the others are correlated with the confounders through a.
CONFOUNDERS = {
    "acquisition_channel": "install",
    "tenure_months": "neither",
    "orders_12m": "both",
    "days_since_last_order": "both",
    "avg_order_value": "spend",
    "email_open_rate": "install",
    "mobile_share": "both",
    "loyalty_member": "both",
}

# The decision (an assumption of the synthetic world): a $2.50 credit for installing, priced at
# the retailer's 30% gross margin (Module 5) over the 26-week horizon of the outcome.
MARGIN = 0.30
CREDIT = 2.50
HORIZON_WEEKS = 26
ODDS_MULTIPLIER = 2.0
INCENTIVE_TABLE = (1.25, 1.5, 2.0, 3.0, 5.0, 10.0)

DESCRIPTION = (
    "The retailer launched a shopping app and announced it by e-mail and on the checkout page "
    "during 4 launch weeks. Customers who open e-mails, shop on their phones or had bought "
    "recently were more likely to install it, and loyal, frequent, recent customers also spend "
    "most in the next 26 weeks: installers would have spent more without the app. Each customer "
    "has a true propensity to install e(x), a true expected spend without the app mu0(x) and a "
    "true effect of the app tau(x) = 8 + 0.12 mu0(x) dollars over 26 weeks. Every common cause "
    "of install and spend is among the nine covariates in the file (channel as two "
    "indicators), so the assumption of no unobserved confounding holds given all of them; "
    "drop one that enters both (mobile_share, for example) and it fails."
)

ESTIMAND_DEFINITIONS = {
    "ate": "mean of true_effect over all n customers (average treatment effect)",
    "att": ("sum(e tau) / sum(e) with e = true_propensity: the average effect among installers, "
            "each customer weighted by their probability of installing"),
    "atc": ("sum((1 - e) tau) / sum(1 - e): the average effect among non-installers, each "
            "customer weighted by their probability of not installing"),
    "effect_overlap": ("sum(e (1 - e) tau) / sum(e (1 - e)): the overlap-weighted average "
                       "effect, which residual-on-residual (partialling-out) DML estimates when "
                       "effects vary (briefs/15-double-ml.md, 'What theta is when effects vary')"),
    "naive_expected": ("sum(e (mu0 + tau)) / sum(e) - sum((1 - e) mu0) / sum(1 - e): the "
                       "population difference in mean spend between installers and "
                       "non-installers, what a naive comparison estimates"),
}

DECISION_RULE = (
    "Offer the $2.50 install credit only if the lower end of the 95% confidence interval of "
    "the incremental margin per persuaded install over 26 weeks (margin x effect) is above the "
    "credit; equivalently, the effect's interval lies above break_even_effect = credit / margin."
)
PERSUASION_ASSUMPTION = (
    "Each customer installs in the launch window if U < e(x). A credit that multiplies the odds "
    "of installing by k makes them install if U < expit(eta(x) + log k); the persuaded "
    "customers are those in between, with weight w = expit(eta + log k) - e(x). Nobody who "
    "failed to install in the window would install later without the credit, so every credit "
    "paid buys a persuaded install (cost per persuaded install = credit). As k -> 1, w is "
    "proportional to e (1 - e), the overlap weights: a property of the logistic install model, "
    "not a general law."
)

# Measured coverage and minima behind the lab's checkpoints: mktstats.dml.measure_app and
# measure_hillstrom on 30 seeds per mode, 2026-10-10, macOS arm64 with OMP_NUM_THREADS=2
# (scikit-learn 1.6.1, econml 0.17.0, statsmodels 0.15.0, numpy 2.1.3). The slow tests in
# tests/test_synth_app_install.py rerun the same sweeps.
TOLERANCES = {
    "basis": (
        "Generator seeds 2035-2064, each run as the lab runs it: the data, the QUICK subsample "
        "(10,000 of 20,000 customers), the 5 stratified folds and the boosted models all from "
        "the seed (mktstats.dml.measure_app). Coverage counts how often the true value lies "
        "inside the 95% normal-approximation confidence interval (HC0 for the hand-written "
        "estimate, EconML's HC1 for LinearDML). Checkpoints compare with effect_overlap "
        "recomputed on the rows in use."
    ),
    "n_seeds": 30,
    "interval": "95% normal-approximation confidence interval, estimate +- 1.96 se",
    "checkpoint_1": {
        "claim": "the naive and the regression 95% intervals both lie above the true ATT",
        "full": {"holds": 30, "regression_min": 28.18, "regression_max": 37.03},
        "quick": {"holds": 30, "regression_min": 29.15, "regression_max": 38.22},
    },
    "checkpoint_3": {
        "target": "effect_overlap on the rows in use",
        "full": {"covered": 27, "missed_seeds": [2040, 2044, 2045], "mean_error": -0.197,
                 "error_sd": 1.656, "mean_se": 1.556, "covered_ate": 27},
        "quick": {"covered": 29, "missed_seeds": [2059], "mean_error": -0.019,
                  "error_sd": 2.276, "mean_se": 2.215, "covered_ate": 28},
    },
    "checkpoint_4": {
        "trim": [0.05, 0.95],
        "target": "effect_overlap over the kept rows",
        "full": {"covered": 27, "weight_dropped_below_customers_dropped": 30},
        "quick": {"covered": 29, "weight_dropped_below_customers_dropped": 30},
    },
    "checkpoint_5": {
        "estimate_rel_tolerance": 1e-6,
        "bounds_rel_tolerance": 1e-4,
        "bounds_note": ("EconML's bounds are compared with the hand interval widened by "
                        "sqrt(n / (n - 1)) (HC1 vs HC0); EconML also uses the exact normal "
                        "quantile 1.959964 where the lab uses 1.96"),
        "full": {"covered": 27, "max_rel_diff_estimate": 2.3e-15, "max_rel_diff_bounds": 2.8e-6},
        "quick": {"covered": 29, "max_rel_diff_estimate": 2.0e-15,
                  "max_rel_diff_bounds": 2.4e-5},
    },
    "no_cross_fitting": {
        "check": "in-sample standard error >= min_ratio x the cross-fitted one",
        "min_ratio": 2.0,
        "full": {"holds": 30, "min": 4.00, "median": 4.58, "max": 6.49},
        "quick": {"holds": 30, "min": 34.14, "median": 39.44, "max": 50.25},
    },
    "hidden_confounder": {
        "check": "the estimate without mobile_share minus the estimate with it > min_shift",
        "min_shift": 3.0,
        "full": {"holds": 30, "min": 4.33, "median": 6.11, "max": 7.20},
        "quick": {"holds": 30, "min": 4.67, "median": 5.84, "max": 8.20},
    },
    "hillstrom": {
        "basis": ("Womens E-Mail vs No E-Mail, outcome visit; spoiling seeds 2035-2064 "
                  "(mktstats.dml.measure_hillstrom); the folds and models use the same seed"),
        "kept_tolerance": 0.02,
        "full": {"kept_ok": 30, "naive_excludes_benchmark": 30, "linear_dml_covers": 30},
        "quick_lab_half": {"half_seed": 2035, "kept_ok": 30, "naive_excludes_benchmark": 29,
                           "linear_dml_covers": 30},
        "quick_random_halves": {"kept_ok": 30, "naive_excludes_benchmark": 27,
                                "linear_dml_covers": 30},
    },
    "decision": {
        "full": {"offer": 28, "offer_after_mobile_share_shift": 1},
        "quick": {"offer": 21, "offer_after_mobile_share_shift": 2},
    },
}


def _weighted(w: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def covariates(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """The pre-launch covariates (draws 1 to 9 of the stream)."""
    a = rng.normal(size=n)
    channel = rng.choice(np.array(CHANNELS, dtype=object), size=n, p=CHANNEL_PROBS)
    referral = (channel == "referral").astype(float)
    tenure = np.clip(np.round(3 + rng.gamma(2.0, 12.0, n)), 3, 96).astype(int)
    orders = 1 + rng.poisson(np.exp(0.55 + 0.6 * a + 0.15 * referral))
    days = np.clip(np.round(365 * rng.beta(1.0, 1.0 + 0.6 * orders)), 1, 365).astype(int)
    aov = np.round(np.exp(rng.normal(np.log(55) + 0.15 * referral + 0.1 * a, 0.45)), 2)
    mu_email = expit(-0.8 + 0.5 * a)
    email = np.round(rng.beta(6 * mu_email, 6 * (1 - mu_email)), 3)
    mu_mobile = expit(-0.3 + 0.9 * (channel == "social") + 0.2 * (channel == "search")
                      - 0.01 * (tenure - 24))
    mobile = np.round(rng.beta(4 * mu_mobile, 4 * (1 - mu_mobile)), 3)
    loyal = rng.binomial(1, expit(-1.3 + 0.7 * a + 0.015 * tenure)).astype(int)
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "acquisition_channel": channel.astype(str),
        "tenure_months": tenure,
        "orders_12m": orders.astype(int),
        "days_since_last_order": days,
        "avg_order_value": aov,
        "email_open_rate": email,
        "mobile_share": mobile,
        "loyalty_member": loyal,
    })


def install_logit(df: pd.DataFrame, p: dict = PARAMS) -> np.ndarray:
    """eta(x), the log-odds of installing."""
    r30 = np.exp(-df["days_since_last_order"].to_numpy(float) / p["recent_days"])
    ch = df["acquisition_channel"].to_numpy()
    return (p["t0"] + p["t_recent"] * r30
            + p["t_orders"] * np.log1p(df["orders_12m"].to_numpy(float))
            + p["t_mobile"] * df["mobile_share"].to_numpy(float)
            + p["t_email"] * df["email_open_rate"].to_numpy(float)
            + p["t_loyal"] * df["loyalty_member"].to_numpy(float)
            + p["t_social"] * (ch == "social") + p["t_referral"] * (ch == "referral"))


def expected_orders(df: pd.DataFrame, p: dict = PARAMS) -> np.ndarray:
    """r26(x), expected orders in the 26 weeks after the launch window without the app."""
    r30 = np.exp(-df["days_since_last_order"].to_numpy(float) / p["recent_days"])
    return (p["r_scale"] * df["orders_12m"].to_numpy(float) * (p["r_base"] + p["r_recent"] * r30)
            * (1 + p["r_loyal"] * df["loyalty_member"].to_numpy(float)))


def baseline_spend(df: pd.DataFrame, p: dict = PARAMS) -> np.ndarray:
    """mu0(x) = E[Y(0) | x], dollars over 26 weeks."""
    return (expected_orders(df, p) * df["avg_order_value"].to_numpy(float)
            * (p["m_base"] + p["m_mobile"] * df["mobile_share"].to_numpy(float)))


def estimands(e: np.ndarray, tau: np.ndarray, mu0: np.ndarray) -> dict[str, float]:
    """The averages of the effect in ESTIMAND_DEFINITIONS (unrounded), from the truth columns."""
    e, tau, mu0 = (np.asarray(x, dtype=float) for x in (e, tau, mu0))
    return {
        "ate": float(np.mean(tau)),
        "att": _weighted(e, tau),
        "atc": _weighted(1 - e, tau),
        "effect_overlap": _weighted(e * (1 - e), tau),
        "naive_expected": _weighted(e, mu0 + tau) - _weighted(1 - e, mu0),
    }


def persuaded(eta: np.ndarray, e: np.ndarray, tau: np.ndarray, k: float) -> dict[str, float]:
    """The customers a credit that multiplies the odds of installing by k persuades."""
    w = expit(eta + np.log(k)) - e
    return {"odds_multiplier": float(k), "effect_persuaded": _weighted(w, tau),
            "share_persuaded": float(np.mean(w))}


def app_install(seed: int = SEED, *, n: int = N, **params) -> SynthResult:
    """Customers, their app install (T), their next 26 weeks of spend (Y) and the truth.

    ``params`` overrides entries of :data:`PARAMS` (for experiments; the committed file uses
    the defaults). Returns ``SynthResult`` with one table, ``customers`` (columns
    :data:`COLUMNS`).
    """
    unknown = set(params) - set(PARAMS)
    if unknown:
        raise TypeError(f"unknown parameters {sorted(unknown)}; known: {sorted(PARAMS)}")
    p = {**PARAMS, **params}
    rng = make_rng(seed)
    df = covariates(rng, n)
    eta = install_logit(df, p)
    e = expit(eta)
    t = (rng.uniform(size=n) < e).astype(int)
    r26 = expected_orders(df, p)
    mu0 = baseline_spend(df, p)
    tau = p["tau0"] + p["tau_slope"] * mu0
    p_buy = 1 - np.exp(-r26)
    buys = rng.uniform(size=n) < p_buy
    amount = rng.gamma(p["spend_shape"], (mu0 + t * tau) / p_buy / p["spend_shape"])
    df["app_installed"] = t
    df["spend_next_26w"] = np.where(buys, np.round(amount, 2), 0.0)
    df["true_propensity"] = np.round(e, 6)
    df["true_effect"] = np.round(tau, 4)
    df["true_baseline"] = np.round(mu0, 4)
    df = df[list(COLUMNS)]

    e_s, tau_s, mu0_s = (df[c].to_numpy(float) for c in
                         ("true_propensity", "true_effect", "true_baseline"))
    est = estimands(e_s, tau_s, mu0_s)
    y = df["spend_next_26w"].to_numpy(float)
    table = [{k: sig(v) for k, v in persuaded(eta, e, tau, m).items()} for m in INCENTIVE_TABLE]
    main = next(r for r in table if r["odds_multiplier"] == ODDS_MULTIPLIER)
    q = np.quantile(e_s, [0.01, 0.05, 0.5, 0.95, 0.99])
    truth = {
        "seed": seed,
        "n": n,
        "description": DESCRIPTION,
        "columns": list(COLUMNS),
        "covariates": list(COVARIATES),
        "truth_columns": ["true_propensity", "true_effect", "true_baseline"],
        "parameters": dict(p),
        "formulas": FORMULAS,
        "estimands": {k: {"value": sig(v), "definition": ESTIMAND_DEFINITIONS[k]}
                      for k, v in est.items()},
        "incentive": {
            "odds_multiplier": ODDS_MULTIPLIER,
            "effect_persuaded": main["effect_persuaded"],
            "share_persuaded": main["share_persuaded"],
            "table": table,
            "assumption": PERSUASION_ASSUMPTION,
        },
        "decision": {
            "margin": MARGIN,
            "credit": CREDIT,
            "horizon_weeks": HORIZON_WEEKS,
            "break_even_effect": sig(CREDIT / MARGIN),
            "rule": DECISION_RULE,
            "persuasion_assumption": PERSUASION_ASSUMPTION,
            "true_margin_per_persuaded_install": sig(MARGIN * main["effect_persuaded"]),
        },
        "confounders": CONFOUNDERS,
        "install_rate": sig(df["app_installed"].mean()),
        "spend": {
            "mean_installers": sig(y[t == 1].mean()),
            "mean_non_installers": sig(y[t == 0].mean()),
            "share_zero": sig(np.mean(y == 0)),
            "max": sig(y.max()),
        },
        "propensity": {
            "quantiles": {"1%": sig(q[0]), "5%": sig(q[1]), "50%": sig(q[2]), "95%": sig(q[3]),
                          "99%": sig(q[4])},
            "share_below_0.05": sig(np.mean(e_s < 0.05)),
            "share_above_0.95": sig(np.mean(e_s > 0.95)),
        },
        "tolerances": TOLERANCES,
    }
    return SynthResult("app_install", {"customers": df}, truth)
