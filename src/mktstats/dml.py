"""Double machine learning for Module 15: reference implementations of the lab's functions,
the checkpoint checks (each failure message says what failed with the numbers, the likely
cause, what to check and how to recover), and the measurements behind the lab's tolerances.

Notation (briefs/15-double-ml.md): T install (0/1), Y spend over 26 weeks, X the nine
covariates (``W`` in code: EconML calls controls W), q(x) = E[Y | x] and e(x) = P(T = 1 | x)
predicted out of fold as ``y_hat`` and ``t_hat``; residuals Y~ = y - y_hat, T~ = t - t_hat;

    theta = sum(T~ Y~) / sum(T~^2),   se = sqrt(sum(T~^2 (Y~ - theta T~)^2)) / sum(T~^2)  (HC0),

95% interval theta +- 1.96 se. When effects vary, theta targets the overlap-weighted average
effect sum(e (1 - e) tau) / sum(e (1 - e)).

scikit-learn, statsmodels and EconML are imported inside the functions that use them, so
``import mktstats`` stays light (pyproject.toml).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from mktstats.synth import app_install as synth

Z95 = 1.96  # 95% normal-approximation interval: estimate +- 1.96 standard errors
FEATURES = synth.COVARIATES
INDEX = ["naive", "regression"]
COLUMNS = ["estimate", "se", "ci_low", "ci_high"]
DML_KEYS = ["theta", "se", "ci_low", "ci_high"]
N_QUICK = 10_000
# The nuisance models (both the regressor and the classifier): histogram gradient boosting.
# early_stopping=False must be explicit: 'auto' switches it on above 10,000 rows (scikit-learn
# 1.6.1 docstring), which would hold out a random validation set inside every fold.
BOOST = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
             early_stopping=False)
# The overfitting booster of the "no cross-fitting" demonstration (fitted on all rows).
OVERFIT = dict(max_iter=600, learning_rate=0.1, max_leaf_nodes=255, min_samples_leaf=2,
               early_stopping=False)
HILLSTROM_FEATURES = ["recency", "history", "mens", "womens", "newbie", "zip_rural",
                      "zip_suburban", "channel_web", "channel_multichannel"]
# The rule that spoils the Hillstrom experiment: keep every No E-Mail customer; keep a Womens
# E-Mail customer with probability 0.9 if recent (recency <= 3 months) or a big spender
# (history >= $250 last year), else with probability 0.2.
KEEP_FAVORED, KEEP_OTHER = 0.9, 0.2
KEPT_TOLERANCE = 0.02  # about 3 binomial standard errors at QUICK (0.006), 5 at FULL


# ---------------------------------------------------------------------------- references
def features(df: pd.DataFrame) -> np.ndarray:
    """The nine covariates as a float matrix (channel as two indicators, search the
    reference); the true_* columns never enter."""
    out = df.assign(channel_social=(df["acquisition_channel"] == "social").astype(float),
                    channel_referral=(df["acquisition_channel"] == "referral").astype(float))
    return out[list(FEATURES)].to_numpy(dtype=float)


def interval(estimate: float, se: float) -> tuple[float, float]:
    return estimate - Z95 * se, estimate + Z95 * se


def baseline_estimates(y, t, W) -> pd.DataFrame:
    """Naive difference in means (unpooled standard error) and the coefficient on t in an OLS
    of y on t and W with a constant (HC1 standard error); 95% normal intervals."""
    import statsmodels.api as sm

    y, t = np.asarray(y, dtype=float), np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    naive = y1.mean() - y0.mean()
    naive_se = np.sqrt(np.var(y1, ddof=1) / len(y1) + np.var(y0, ddof=1) / len(y0))
    fit = sm.OLS(y, sm.add_constant(np.column_stack([t, W]))).fit(cov_type="HC1")
    est = np.array([naive, fit.params[1]])
    se = np.array([naive_se, fit.bse[1]])
    return pd.DataFrame({"estimate": est, "se": se, "ci_low": est - Z95 * se,
                         "ci_high": est + Z95 * se}, index=INDEX)


def cross_fit(model, X, target, folds) -> np.ndarray:
    """Out-of-fold predictions: for each (train, test), a fresh clone fitted on train predicts
    test (the probability of class 1 for a classifier)."""
    from sklearn.base import clone

    target = np.asarray(target)
    out = np.full(len(target), np.nan)
    for train, test in folds:
        fitted = clone(model).fit(X[train], target[train])
        if hasattr(fitted, "predict_proba"):
            out[test] = fitted.predict_proba(X[test])[:, 1]
        else:
            out[test] = fitted.predict(X[test])
    return out


def dml_estimate(y, t, y_hat, t_hat) -> dict[str, float]:
    """Residual on residual with no intercept, HC0 standard error, 95% normal interval."""
    u = np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
    v = np.asarray(t, dtype=float) - np.asarray(t_hat, dtype=float)
    theta = np.sum(v * u) / np.sum(v * v)
    se = np.sqrt(np.sum(v**2 * (u - theta * v) ** 2)) / np.sum(v * v)
    lo, hi = interval(theta, se)
    return {"theta": float(theta), "se": float(se), "ci_low": float(lo), "ci_high": float(hi)}


def trimmed_estimate(y, t, y_hat, t_hat, low, high) -> dict[str, float]:
    """dml_estimate on the rows with low <= t_hat <= high, plus the share of customers kept and
    the share of the overlap weight t_hat (1 - t_hat) dropped."""
    y, t, y_hat, t_hat = (np.asarray(a, dtype=float) for a in (y, t, y_hat, t_hat))
    keep = (t_hat >= low) & (t_hat <= high)
    out = dml_estimate(y[keep], t[keep], y_hat[keep], t_hat[keep])
    w = t_hat * (1 - t_hat)
    out["share_kept"] = float(keep.mean())
    out["weight_share_dropped"] = float(w[~keep].sum() / w.sum())
    return out


def fit_linear_dml(y, t, W, folds, model_y, model_t, seed):
    """EconML's LinearDML with the lab's folds and models: one constant effect (X=None)."""
    from econml.dml import LinearDML

    est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, cv=folds,
                    random_state=seed)
    return est.fit(y, t, X=None, W=W)


def overlap_truth(e, tau, rows=None) -> float:
    """sum(e (1 - e) tau) / sum(e (1 - e)) over ``rows`` (a mask or index; all by default)."""
    e, tau = np.asarray(e, dtype=float), np.asarray(tau, dtype=float)
    if rows is not None:
        e, tau = e[rows], tau[rows]
    w = e * (1 - e)
    return float(np.sum(w * tau) / np.sum(w))


def true_averages(e, tau) -> dict[str, float]:
    """ATE, ATT and ATC (true-propensity weights) and the overlap-weighted average."""
    e, tau = np.asarray(e, dtype=float), np.asarray(tau, dtype=float)
    return {"ATE": float(tau.mean()), "ATT": float(np.sum(e * tau) / np.sum(e)),
            "ATC": float(np.sum((1 - e) * tau) / np.sum(1 - e)),
            "overlap": overlap_truth(e, tau)}


def boosted(seed: int, params: dict = BOOST):
    """(model_y, model_t): the regressor and classifier with the lab's settings."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    return (HistGradientBoostingRegressor(**params, random_state=seed),
            HistGradientBoostingClassifier(**params, random_state=seed))


def quick_rows(n: int, seed: int, size: int = N_QUICK) -> np.ndarray:
    """The QUICK subsample: ``size`` of ``n`` rows, seeded, in their original order."""
    return np.sort(np.random.default_rng(seed).choice(n, size, replace=False))


def folds_for(W, t, seed: int, k: int = 5) -> list:
    from sklearn.model_selection import StratifiedKFold

    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(W, t))


# ------------------------------------------------------------------- Hillstrom benchmark
def hillstrom_arms(hill: pd.DataFrame) -> pd.DataFrame:
    """Womens E-Mail (treatment 1) and No E-Mail (0) customers, with a fresh index."""
    out = hill[hill["segment"].isin(["Womens E-Mail", "No E-Mail"])].reset_index(drop=True)
    return out.assign(treatment=(out["segment"] == "Womens E-Mail").astype(int))


def hillstrom_features(df: pd.DataFrame) -> np.ndarray:
    """Module 11's Hillstrom features (zip_code spells 'Surburban'), as a float matrix."""
    out = df[["recency", "history", "mens", "womens", "newbie"]].astype(float).assign(
        zip_rural=(df["zip_code"] == "Rural").astype(float),
        zip_suburban=(df["zip_code"] == "Surburban").astype(float),
        channel_web=(df["channel"] == "Web").astype(float),
        channel_multichannel=(df["channel"] == "Multichannel").astype(float),
    )
    return out[HILLSTROM_FEATURES].to_numpy(dtype=float)


def favored(df: pd.DataFrame) -> pd.Series:
    """Recent (recency <= 3 months) or big spenders (history >= $250 last year)."""
    return (df["recency"] <= 3) | (df["history"] >= 250)


def confound(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Keep every control; keep a treated customer with probability 0.9 if favored, else 0.2."""
    p_keep = np.where(favored(df), KEEP_FAVORED, KEEP_OTHER)
    u = np.random.default_rng(seed).uniform(size=len(df))
    keep = (df["treatment"] == 0) | (u < p_keep)
    return df[keep].reset_index(drop=True)


def kept_shares(full: pd.DataFrame, spoiled: pd.DataFrame) -> dict[str, float]:
    """Share of each group kept by confound(): control, treated favored, treated other."""
    def counts(df):
        f = favored(df)
        return {"control": int((df["treatment"] == 0).sum()),
                "favored": int(((df["treatment"] == 1) & f).sum()),
                "other": int(((df["treatment"] == 1) & ~f).sum())}

    a, b = counts(full), counts(spoiled)
    return {k: b[k] / a[k] for k in a}


def visit_benchmark(df: pd.DataFrame) -> tuple[float, float]:
    """Difference in visit rates, treated minus control, and its unpooled standard error."""
    y, t = df["visit"].to_numpy(dtype=float), df["treatment"].to_numpy()
    y1, y0 = y[t == 1], y[t == 0]
    return (float(y1.mean() - y0.mean()),
            float(np.sqrt(np.var(y1, ddof=1) / len(y1) + np.var(y0, ddof=1) / len(y0))))


# ------------------------------------------------------------------------------ checks
def _recover(n: int) -> str:
    return f" To move on now, run `workshop.use_reference({n})` and come back to this later."


INSTRUCTOR = " Tell the instructor."


def _close(a, b, rel=0.0, abs=1e-9) -> bool:
    return bool(np.all(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
                       <= abs + rel * np.abs(np.asarray(b, dtype=float))))


def check_data(customers: pd.DataFrame, customers_all: pd.DataFrame, truth: dict,
               features_fn: Callable, feature_names, n_expected: int) -> None:
    """The labeled `data` check: rows, columns, the feature matrix and the truth file."""
    problems = []
    if len(customers) != n_expected:
        problems.append(f"{len(customers):,} customers in use, expected {n_expected:,}")
    if list(customers.columns) != list(synth.COLUMNS):
        problems.append(f"columns {list(customers.columns)}, expected the 14 columns"
                        f" {list(synth.COLUMNS)}")
    names = list(feature_names)
    leaked = [c for c in names if str(c).startswith("true_")]
    if leaked:
        problems.append(f"the features include {leaked}, which only the simulation knows")
    if len(names) != len(FEATURES):
        problems.append(f"{len(names)} feature names, expected 9")
    W = np.asarray(features_fn(customers), dtype=float)
    if W.shape != (len(customers), len(FEATURES)):
        problems.append(f"features(customers) has shape {W.shape}, expected"
                        f" ({len(customers):,}, 9)")
    elif not np.array_equal(W, features(customers)):
        problems.append("features(customers) differs from the lab's nine covariates"
                        f" {list(FEATURES)}")
    est = truth.get("estimands", {})
    missing = [k for k in ("ate", "att", "atc", "effect_overlap") if k not in est]
    if missing:
        problems.append(f"truth['estimands'] lacks {missing}")
    else:
        e = customers_all["true_propensity"].to_numpy(dtype=float)
        tau = customers_all["true_effect"].to_numpy(dtype=float)
        mu0 = customers_all["true_baseline"].to_numpy(dtype=float)
        again = synth.estimands(e, tau, mu0)
        off = [k for k in ("ate", "att", "atc", "effect_overlap")
               if not _close(again[k], est[k]["value"], rel=1e-5)]
        if off:
            problems.append(f"the true averages {off} recomputed from the CSV differ from"
                            " truth_app_install.json (the two files are from different versions)")
    if problems:
        raise AssertionError(
            "The app data did not load as expected: " + "; ".join(problems) + ". Rerun the"
            " install cell and then this cell; if it still fails, tell the instructor.")


# The toy of Checkpoint 1: y = 2 + 3 t + 5 w exactly (no noise).
TOY1_T = np.array([0, 0, 0, 1, 1, 1])
TOY1_W = np.array([[0.0], [1.0], [2.0], [1.0], [2.0], [3.0]])
TOY1_Y = 2 + 3 * TOY1_T + 5 * TOY1_W[:, 0]


def _table(got, what: str, n: int) -> pd.DataFrame:
    if not isinstance(got, pd.DataFrame) or list(got.index) != INDEX or list(
            got.columns) != COLUMNS:
        shape = (f"index {list(got.index)} and columns {list(got.columns)}"
                 if isinstance(got, pd.DataFrame) else f"a {type(got).__name__}")
        raise AssertionError(
            f"Checkpoint {n}: {what} returned {shape}. Return a DataFrame with index"
            " ['naive', 'regression'] and columns estimate, se, ci_low, ci_high, in that order."
            + _recover(n))
    return got.astype(float)


def check_baseline_estimates(fn: Callable, y, t, W, att: float) -> pd.DataFrame:
    """Checkpoint 1: the toy worked by hand, the app data against the reference, then both
    intervals against the true ATT. Returns the app-data table."""
    got = _table(fn(TOY1_Y, TOY1_T, TOY1_W), "baseline_estimates on the toy", 1)
    naive, reg = got.loc["naive"], got.loc["regression"]
    if not _close(naive["estimate"], 8.0):
        raise AssertionError(
            f"Checkpoint 1: on the toy (t = {TOY1_T.tolist()}, w = {TOY1_W[:, 0].tolist()},"
            f" y = 2 + 3t + 5w) your naive estimate is {naive['estimate']:.4g}; by hand it is 8:"
            " mean y of the installers (15) minus mean y of the others (7). Check which group"
            " comes first: installers (t == 1) minus non-installers." + _recover(1))
    if _close(naive["se"], np.sqrt(100 / 9)):
        raise AssertionError(
            f"Checkpoint 1: on the toy your naive standard error is {naive['se']:.4g}; by hand it"
            " is √(25/3 + 25/3) ≈ 4.082. You used the population variance: use the sample"
            " variance, `np.var(x, ddof=1)` (pandas `.var()` does this by default)." + _recover(1))
    if not _close(naive["se"], np.sqrt(50 / 3)):
        raise AssertionError(
            f"Checkpoint 1: on the toy your naive standard error is {naive['se']:.4g}; by hand it"
            " is √(s₁²/n₁ + s₀²/n₀) = √(25/3 + 25/3) ≈ 4.082, with the sample variances of y in"
            " each group." + _recover(1))
    wrong = {
        2.0: "you took the constant; `sm.add_constant` puts it first, so the coefficient on `t`"
             " is `params[1]`.",
        5.0: "you took the coefficient on `w`; `t` is the first column after the constant,"
             " `params[1]`.",
        23 / 7: "the regression has no constant; wrap the columns in `sm.add_constant(...)`.",
        8.0: "the regression left out the covariates; pass `np.column_stack([t, W])`.",
    }
    if not _close(reg["estimate"], 3.0, abs=1e-9):
        cause = next((m for v, m in wrong.items() if _close(reg["estimate"], v, abs=1e-6)),
                     "the coefficient on `t` in `sm.OLS(y, sm.add_constant(np.column_stack([t,"
                     " W]))).fit(cov_type=\"HC1\")` is `params[1]`.")
        raise AssertionError(
            f"Checkpoint 1: on the toy (y = 2 + 3t + 5w exactly) your regression estimate is"
            f" {reg['estimate']:.4g}; it should be 3, the coefficient of t. Likely cause: "
            + cause + _recover(1))
    if not _close(reg["se"], 0.0, abs=1e-6):
        raise AssertionError(
            f"Checkpoint 1: on the toy your regression standard error is {reg['se']:.4g}; the"
            " toy has no noise, so it is 0. Take it from the same fit as the estimate,"
            " `fit.bse[1]`." + _recover(1))
    for row in INDEX:
        r = got.loc[row]
        if not _close([r["ci_low"], r["ci_high"]], interval(r["estimate"], r["se"]), abs=1e-9):
            raise AssertionError(
                f"Checkpoint 1: on the toy the {row} interval is [{r['ci_low']:.4g},"
                f" {r['ci_high']:.4g}]; a 95% interval is estimate ± 1.96 × se ="
                f" [{r['estimate'] - Z95 * r['se']:.4g}, {r['estimate'] + Z95 * r['se']:.4g}]."
                + _recover(1))

    got = _table(fn(y, t, W), "baseline_estimates(y, t, W)", 1)
    ref = baseline_estimates(y, t, W)
    for row in INDEX:
        g, r = got.loc[row], ref.loc[row]
        if not _close(g["estimate"], r["estimate"], rel=1e-9):
            hint = ("mean spend of installers minus mean spend of non-installers." if
                    row == "naive" else "the coefficient on `t` from"
                    " `sm.OLS(y, sm.add_constant(np.column_stack([t, W]))).fit(cov_type=\"HC1\")`.")
            raise AssertionError(
                f"Checkpoint 1: on the app data your {row} estimate is {g['estimate']:.4f}; the"
                f" reference gives {r['estimate']:.4f}. Check: " + hint + _recover(1))
        if not _close(g["se"], r["se"], rel=1e-9):
            hint = ("use the unpooled formula √(s₁²/n₁ + s₀²/n₀), not a pooled variance." if
                    row == "naive" else "fit with `cov_type=\"HC1\"`; the default assumes every"
                    " customer's spend is equally noisy.")
            raise AssertionError(
                f"Checkpoint 1: on the app data your {row} standard error is {g['se']:.4f}; the"
                f" reference gives {r['se']:.4f}. Likely cause: " + hint + _recover(1))
        if not _close([g["ci_low"], g["ci_high"]], [r["ci_low"], r["ci_high"]], rel=1e-9):
            raise AssertionError(
                f"Checkpoint 1: on the app data your {row} interval is [{g['ci_low']:.2f},"
                f" {g['ci_high']:.2f}]; the reference gives [{r['ci_low']:.2f},"
                f" {r['ci_high']:.2f}]. A 95% interval is estimate ± 1.96 × se." + _recover(1))
    below = [row for row in INDEX if not ref.loc[row, "ci_low"] > att]
    if below:
        raise AssertionError(
            f"Checkpoint 1: the {' and '.join(below)} 95% interval does not lie above the true"
            f" ATT, ${att:.2f} (naive [{ref.loc['naive', 'ci_low']:.2f},"
            f" {ref.loc['naive', 'ci_high']:.2f}], regression"
            f" [{ref.loc['regression', 'ci_low']:.2f}, {ref.loc['regression', 'ci_high']:.2f}])."
            " Your function matches the reference, so this compares the data with the truth, not"
            " your code: tell the instructor.")
    return got


def toy_cross_fit(seed: int):
    """Checkpoint 2's toy: 40 rows, 2 features, distinct continuous targets, balanced labels,
    and its folds (KFold for the regressors, StratifiedKFold on the labels)."""
    from sklearn.model_selection import KFold, StratifiedKFold

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 2))
    target = np.round(rng.normal(50, 20, size=40), 2)
    while len(np.unique(target)) < 40:  # distinct targets (always true for this seed)
        target = np.round(rng.normal(50, 20, size=40), 2)
    labels = rng.permutation(np.repeat([0, 1], 20))
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    sfolds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, labels))
    return X, target, labels, folds, sfolds


def _predictions(out, n: int) -> np.ndarray:
    try:
        arr = np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        arr = np.array([np.nan])
    if arr.shape != (n,) or not np.isfinite(arr).all():
        raise AssertionError(
            f"Checkpoint 2: cross_fit returned shape {arr.shape} with"
            f" {int(np.sum(~np.isfinite(arr))) if arr.ndim else 'no'} missing values; it must"
            f" return {n} numbers. Return one prediction per row of `X`, filling every fold's"
            " test rows." + _recover(2))
    return arr


def check_cross_fit(fn: Callable, seed: int) -> dict[str, int]:
    """Checkpoint 2 (toy inputs only): out-of-fold means out of fold; the same answers as
    scikit-learn's cross_val_predict; probabilities of class 1 for a classifier. Returns the
    in-sample and out-of-fold counts of exact predictions of the 1-nearest-neighbor model."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    X, target, labels, folds, sfolds = toy_cross_fit(seed)
    n = len(target)
    one_nn = KNeighborsRegressor(n_neighbors=1)
    oof = _predictions(fn(one_nn, X, target, folds), n)
    hits = int(np.sum(np.isclose(oof, target, rtol=0, atol=1e-9)))
    if hits:
        raise AssertionError(
            f"Checkpoint 2: {hits} of {n} out-of-fold predictions equal the customer's own spend:"
            " those customers were in the model's training data. For each fold fit on"
            " `X[train]` and predict `X[test]`; a model fitted on all rows and scored on them gets"
            f" {n} of {n}." + _recover(2))
    linear = _predictions(fn(LinearRegression(), X, target, folds), n)
    want = cross_val_predict(LinearRegression(), X, target, cv=folds)
    if not np.allclose(linear, want, rtol=0, atol=1e-10):
        raise AssertionError(
            "Checkpoint 2: with LinearRegression() your predictions differ from scikit-learn's"
            " cross_val_predict on the same folds (largest gap"
            f" {np.max(np.abs(linear - want)):.3g})."
            " Put each fold's predictions at its test positions, `out[test] = ...`; appending fold"
            " after fold puts them in the wrong order. Fit a fresh `clone(model)` on each fold's"
            " training rows." + _recover(2))
    knn = KNeighborsClassifier(n_neighbors=5)
    probs = _predictions(fn(knn, X, labels, sfolds), n)
    want = cross_val_predict(knn, X, labels, cv=sfolds, method="predict_proba")[:, 1]
    if not np.allclose(probs, want, rtol=0, atol=1e-10):
        if np.all(np.isin(probs, [0.0, 1.0])):
            cause = ("for a classifier, store `predict_proba(X[test])[:, 1]`, the probability of"
                     " class 1, not `predict`'s 0/1 labels.")
        elif np.allclose(probs, 1 - want, rtol=0, atol=1e-10):
            cause = ("column 1 is the probability of installing; column 0 is the probability of"
                     " not installing. Store `predict_proba(X[test])[:, 1]`.")
        else:
            cause = ("for a model with `predict_proba`, store `predict_proba(X[test])[:, 1]`"
                     " at the test positions.")
        raise AssertionError(
            "Checkpoint 2: with KNeighborsClassifier(n_neighbors=5) your predictions differ from"
            " cross_val_predict(..., method='predict_proba')[:, 1] (first values"
            f" {np.round(probs[:4], 2).tolist()} vs {np.round(want[:4], 2).tolist()}): " + cause
            + _recover(2))
    from mktstats import checks

    checks.probability(probs, name="cross_fit's predictions for a classifier")
    # The fit below is the in-sample answer to the Predict; it never touches your function.
    # EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.
    in_sample = KNeighborsRegressor(n_neighbors=1).fit(X, target).predict(X)
    return {"n": n, "in_sample": int(np.sum(np.isclose(in_sample, target, rtol=0, atol=1e-9))),
            "out_of_fold": hits}


# The toy of Checkpoint 3: T~ = (0.5, -0.5, 0.75, -0.25) has a nonzero mean on purpose.
TOY3 = {"y": np.array([0.0, 0.0, 3.0, 2.0]), "t": np.array([1, 0, 1, 0]),
        "y_hat": np.array([1.0, 2.0, 1.0, 3.0]), "t_hat": np.array([0.5, 0.5, 0.25, 0.25])}


def _dml_dict(got, what: str, n: int) -> dict[str, float]:
    if not isinstance(got, dict) or not set(DML_KEYS) <= set(got):
        have = sorted(got) if isinstance(got, dict) else type(got).__name__
        raise AssertionError(
            f"Checkpoint {n}: {what} returned {have}; it must return a dict with the keys"
            f" {DML_KEYS}." + _recover(n))
    return {k: float(np.ravel(got[k])[0]) for k in got}


def _toy3_wrong() -> tuple[dict, dict]:
    """The likely wrong theta and se values on the toy, with their messages."""
    y, t, y_hat, t_hat = TOY3["y"], TOY3["t"], TOY3["y_hat"], TOY3["t_hat"]
    u, v = y - y_hat, t - t_hat
    theta = np.sum(v * u) / np.sum(v * v)
    r = u - theta * v
    n = len(y)
    thetas = {
        float(np.sum(t * u) / np.sum(t * t)): "you used `t`, not the install residual"
                                               " `t - t_hat`.",
        float(np.sum(v * y) / np.sum(v * v)): "you used `y`, not the spend residual"
                                               " `y - y_hat`.",
        float(np.polyfit(v, u, 1)[0]): "the formula has no intercept: θ̂ = Σ T̃Ỹ / Σ T̃²."
                                       " `np.polyfit` or a regression with a constant adds one.",
        float(-theta): "only one residual has its sign flipped; both are actual minus"
                       " predicted.",
    }
    se = np.sqrt(np.sum(v**2 * r**2)) / np.sum(v * v)
    ses = {
        float(np.sqrt(np.sum(r**2) / (n - 1) / np.sum(v * v))):
            "that is the classical standard error, which assumes equal noise for every"
            " customer; use √(Σ T̃² (Ỹ − θ̂T̃)²) / Σ T̃².",
        float(se * np.sqrt(n / (n - 1))):
            "that is HC1 (statsmodels' `cov_type=\"HC1\"`), which multiplies by √(n / (n − 1));"
            " the formula here is HC0. On 20,000 customers the two differ by 0.003%, on 4 toy"
            " rows by 15%.",
        float(se**2): "that is the squared standard error; take the square root of"
                      " Σ T̃² (Ỹ − θ̂T̃)², then divide by Σ T̃².",
        float(np.sqrt(np.sum(v**2 * r**2)) / np.sqrt(np.sum(v * v))):
            "divide by Σ T̃², not by its square root.",
        float(np.sqrt(np.sum(v**2 * u**2)) / np.sum(v * v)):
            "inside the standard error the residual is Ỹ − θ̂T̃, not Ỹ.",
    }
    return thetas, ses


def check_dml_estimate(fn: Callable, y, t, y_hat, t_hat, target: float, ate: float) -> dict:
    """Checkpoint 3: the toy exactly, the app data against the reference, then the provided
    predictions against the true overlap-weighted effect. Returns the app-data result."""
    got = _dml_dict(fn(**TOY3), "dml_estimate on the toy", 3)
    want = dml_estimate(**TOY3)
    thetas, ses = _toy3_wrong()
    toy = "t = [1, 0, 1, 0], t_hat = [0.5, 0.5, 0.25, 0.25], y = [0, 0, 3, 2], y_hat = [1, 2, 1, 3]"
    if not _close(got["theta"], want["theta"], abs=1e-12):
        cause = next((m for v, m in thetas.items() if _close(got["theta"], v, abs=1e-3)),
                     "θ̂ = Σ T̃Ỹ / Σ T̃² with T̃ = t − t_hat and Ỹ = y − y_hat.")
        raise AssertionError(
            f"Checkpoint 3: on the toy ({toy}) your θ̂ is {got['theta']:.4g}; by hand it is"
            " 2.25 / 1.125 = 2. Likely cause: " + cause + _recover(3))
    if not _close(got["se"], want["se"], abs=1e-12):
        cause = next((m for v, m in ses.items() if _close(got["se"], v, abs=1e-3)),
                     "use √(Σ T̃² (Ỹ − θ̂T̃)²) / Σ T̃².")
        raise AssertionError(
            f"Checkpoint 3: on the toy your standard error is {got['se']:.4g}; by hand it is"
            f" √10 / 3 ≈ {want['se']:.4f}. Likely cause: " + cause + _recover(3))
    if not _close([got["ci_low"], got["ci_high"]], [want["ci_low"], want["ci_high"]],
                  abs=1e-12):
        cause = ("a 95% interval uses 1.96, not 1.645." if _close(
            got["ci_low"], want["theta"] - 1.645 * want["se"], abs=1e-6)
            else "the interval is θ̂ ± 1.96 se.")
        raise AssertionError(
            f"Checkpoint 3: on the toy your interval is [{got['ci_low']:.3f},"
            f" {got['ci_high']:.3f}];"
            f" by hand it is [{want['ci_low']:.3f}, {want['ci_high']:.3f}]: " + cause + _recover(3))
    got = _dml_dict(fn(y, t, y_hat, t_hat), "dml_estimate(y, t, y_hat, t_hat)", 3)
    ref = dml_estimate(y, t, y_hat, t_hat)
    if not _close([got[k] for k in DML_KEYS], [ref[k] for k in DML_KEYS], rel=1e-9):
        raise AssertionError(
            f"Checkpoint 3: on the app data your result {_fmt(got)} differs from the reference"
            f" {_fmt(ref)}, although the toy passed. Check that you use the arrays you are given"
            " (no globals) and the same formulas." + _recover(3))
    if not ref["ci_low"] <= target <= ref["ci_high"]:
        raise AssertionError(
            f"Checkpoint 3: the true overlap-weighted average effect, ${target:.2f}, lies outside"
            f" the 95% interval [${ref['ci_low']:.2f}, ${ref['ci_high']:.2f}] (θ̂ ="
            f" ${ref['theta']:.2f}; the ATE, a different average, is ${ate:.2f}). Your formula is"
            " right, so this compares the provided predictions with the truth: tell the"
            " instructor.")
    return got


def _fmt(d: dict, digits: int = 4) -> str:
    return "{" + ", ".join(f"{k}: {float(v):.{digits}g}" for k, v in d.items()) + "}"


def check_no_cross_fitting(se_in_sample: float, se_cross_fitted: float,
                           min_ratio: float = 2.0) -> float:
    """The labeled `no cross-fitting` check of the provided overfit demonstration."""
    ratio = se_in_sample / se_cross_fitted
    if not ratio >= min_ratio:
        raise AssertionError(
            "This compares two provided fits, not your code: without cross-fitting the standard"
            f" error should be several times larger, and it was only {ratio:.1f} times the"
            f" cross-fitted one (at least {min_ratio:g} expected)." + INSTRUCTOR)
    return ratio


# The toy of Checkpoint 4: rows 2 to 5 (of 6) have t_hat inside [0.05, 0.95].
TOY4 = {"y": np.array([0.0, 1.0, 4.0, 2.0, 5.0, 9.0]), "t": np.array([0, 0, 1, 0, 1, 1]),
        "y_hat": np.array([0.5, 1.0, 2.0, 2.0, 3.0, 6.0]),
        "t_hat": np.array([0.02, 0.10, 0.30, 0.50, 0.70, 0.97]), "low": 0.05, "high": 0.95}
TRIM_KEYS = DML_KEYS + ["share_kept", "weight_share_dropped"]


def check_trimmed_estimate(fn: Callable, dml_fn: Callable, y, t, y_hat, t_hat, low: float,
                           high: float, target_fn: Callable) -> dict:
    """Checkpoint 4: the toy exactly, then on the app data: the share kept, the estimate on the
    kept rows, less weight than customers dropped, and the truth on the kept rows inside the
    95% interval. ``target_fn(mask)`` is the true overlap-weighted effect on the kept rows."""
    toy = {k: v for k, v in TOY4.items()}
    got = fn(**toy)
    if not isinstance(got, dict) or not set(TRIM_KEYS) <= set(got):
        have = sorted(got) if isinstance(got, dict) else type(got).__name__
        raise AssertionError(
            f"Checkpoint 4: trimmed_estimate returned {have}; it must return dml_estimate's dict"
            " plus `share_kept` and `weight_share_dropped`." + _recover(4))
    got = {k: float(np.ravel(got[k])[0]) for k in TRIM_KEYS}
    keep = (TOY4["t_hat"] >= TOY4["low"]) & (TOY4["t_hat"] <= TOY4["high"])
    want = trimmed_estimate(**toy)
    all_rows = dml_estimate(TOY4["y"], TOY4["t"], TOY4["y_hat"], TOY4["t_hat"])
    if _close(got["theta"], all_rows["theta"], abs=1e-9) and not _close(
            got["theta"], want["theta"], abs=1e-9):
        raise AssertionError(
            f"Checkpoint 4: on the toy your θ̂ is {got['theta']:.4f}, the estimate on all six"
            f" rows; on the four kept rows it is {want['theta']:.4f}. Subset `y`, `t`, `y_hat` and"
            " `t_hat` with the same mask before calling `dml_estimate`." + _recover(4))
    if not _close(got["theta"], want["theta"], abs=1e-12):
        raise AssertionError(
            f"Checkpoint 4: on the toy your θ̂ is {got['theta']:.4f}; dml_estimate on the kept rows"
            f" (t_hat inside [{TOY4['low']}, {TOY4['high']}]: rows 2 to 5) gives"
            f" {want['theta']:.4f}. Keep the rows with `low <= t_hat <= high`." + _recover(4))
    if _close(got["share_kept"], keep.sum(), abs=1e-9):
        raise AssertionError(
            f"Checkpoint 4: on the toy your share_kept is {got['share_kept']:g}; return a share,"
            " `mask.mean()`, not a count." + _recover(4))
    if not _close(got["share_kept"], keep.mean(), abs=1e-12):
        raise AssertionError(
            f"Checkpoint 4: on the toy your share_kept is {got['share_kept']:.4f}; 4 of 6 rows are"
            " kept, 0.6667. Use the mask `(t_hat >= low) & (t_hat <= high)`." + _recover(4))
    w = TOY4["t_hat"] * (1 - TOY4["t_hat"])
    wrong = {
        float(1 - keep.mean()): "that is the share of customers dropped; weight each customer by"
                                " `t_hat * (1 - t_hat)`.",
        float(w[keep].sum() / w.sum()): "that is the share of weight kept; return the share"
                                        " dropped.",
        0.0: "the weights come from `t_hat`, not `t`: `t * (1 - t)` is 0 for every customer.",
    }
    wsd = got["weight_share_dropped"]
    if not np.isfinite(wsd):
        raise AssertionError(
            f"Checkpoint 4: on the toy your weight_share_dropped is {wsd}; the weights come from"
            " `t_hat`, not `t`: `t * (1 - t)` is 0 for every customer." + _recover(4))
    if not _close(wsd, want["weight_share_dropped"], abs=1e-12):
        cause = next((m for v, m in wrong.items() if _close(wsd, v, abs=1e-6)),
                     "divide the weight t_hat (1 - t_hat) of the dropped rows by the weight of"
                     " all rows.")
        raise AssertionError(
            f"Checkpoint 4: on the toy your weight_share_dropped is {wsd:.4f}; by hand it is"
            f" 0.0487 / 0.8087 ≈ {want['weight_share_dropped']:.4f}. Likely cause: " + cause
            + _recover(4))
    got = fn(y, t, y_hat, t_hat, low, high)
    got = {k: float(np.ravel(got[k])[0]) for k in TRIM_KEYS}
    t_hat = np.asarray(t_hat, dtype=float)
    mask = (t_hat >= low) & (t_hat <= high)
    on_kept = dml_fn(np.asarray(y)[mask], np.asarray(t)[mask], np.asarray(y_hat)[mask],
                     t_hat[mask])
    if not _close(got["share_kept"], mask.mean(), abs=1e-12) or not _close(
            got["theta"], float(on_kept["theta"]), rel=1e-9):
        raise AssertionError(
            f"Checkpoint 4: on the app data your share_kept is {got['share_kept']:.4f} and θ̂"
            f" {got['theta']:.4f}; the mask keeps {mask.mean():.4f} and dml_estimate on the kept"
            f" rows gives {float(on_kept['theta']):.4f}. Use the same mask for every array."
            + _recover(4))
    if not got["weight_share_dropped"] < 1 - got["share_kept"]:
        raise AssertionError(
            f"Checkpoint 4: on the app data the share of weight dropped,"
            f" {got['weight_share_dropped']:.4f}, is not below the share of customers dropped,"
            f" {1 - got['share_kept']:.4f}; customers near 0 or 1 carry little weight"
            " t_hat (1 - t_hat). Check the weights." + _recover(4))
    truth = target_fn(mask)
    if not got["ci_low"] <= truth <= got["ci_high"]:
        raise AssertionError(
            f"Checkpoint 4: the true overlap-weighted effect among the kept customers,"
            f" ${truth:.2f}, lies outside your 95% interval [${got['ci_low']:.2f},"
            f" ${got['ci_high']:.2f}]. Your function is right, so this compares the provided"
            " predictions with the truth: tell the instructor.")
    got["mask"] = mask
    return got


def check_linear_dml(fn: Callable, y, t, W, folds, model_y, model_t, seed: int, mine: dict,
                     target: float):
    """Checkpoint 5: LinearDML with your folds and models reproduces your Exercise 3 numbers,
    and its interval contains the true overlap-weighted effect. EconML's likely exceptions
    become messages. Returns (est, ate, ci_low, ci_high)."""
    try:
        est = fn(y, t, W, folds, model_y, model_t, seed)
    except AttributeError as exc:
        if "Cannot use a classifier as a first stage model" in str(exc):
            raise AssertionError(
                "Checkpoint 5: EconML says 'Cannot use a classifier as a first stage model when"
                " the target is continuous!'. Set `discrete_treatment=True`: install is 0 or 1,"
                " and `model_t` is a classifier." + _recover(5)) from None
        raise
    if est is None or not hasattr(est, "ate_interval"):
        raise AssertionError(
            f"Checkpoint 5: fit_linear_dml returned {type(est).__name__}; return the fitted"
            " estimator (`LinearDML(...).fit(...)` returns it)." + _recover(5))
    try:
        ate = float(np.ravel(est.ate())[0])
        lo, hi = (float(np.ravel(v)[0]) for v in est.ate_interval(alpha=0.05))
    except AssertionError as exc:
        if "X was not None" in str(exc):
            raise AssertionError(
                "Checkpoint 5: EconML says 'X was not None when fitting'. Pass the covariates as"
                " controls, `W=W`, with `X=None`. `X` is for covariates the effect varies with,"
                " and then EconML wants one effect per customer." + _recover(5)) from None
        raise
    if not _close(ate, mine["theta"], rel=1e-6):
        raise AssertionError(
            f"Checkpoint 5: EconML's estimate is ${ate:.4f}; yours is ${mine['theta']:.4f}. Pass"
            " `cv=folds`, the folds your `cross_fit` used; EconML's default is `cv=2`."
            + _recover(5))
    # EconML's default standard error is HC1 = HC0 x sqrt(n / (n - 1)); yours is HC0.
    hc1 = np.sqrt(len(y) / (len(y) - 1))
    half = (mine["ci_high"] - mine["ci_low"]) / 2 * hc1
    if not _close([lo, hi], [mine["theta"] - half, mine["theta"] + half], rel=1e-4):
        raise AssertionError(
            f"Checkpoint 5: EconML's interval is [${lo:.4f}, ${hi:.4f}]; yours is"
            f" [${mine['ci_low']:.4f}, ${mine['ci_high']:.4f}]. Use `est.ate_interval(alpha=0.05)`"
            " for a 95% interval." + _recover(5))
    if not lo <= target <= hi:
        raise AssertionError(
            f"Checkpoint 5: the true overlap-weighted average effect, ${target:.2f}, lies outside"
            f" EconML's 95% interval [${lo:.2f}, ${hi:.2f}]. Your call matches your Exercise 3"
            " numbers, so this compares the provided models with the truth: tell the instructor.")
    return est, ate, lo, hi


def check_hillstrom(kept: dict, naive: tuple[float, float], dml_ci: tuple[float, float],
                    benchmark: float, tol: float = KEPT_TOLERANCE) -> None:
    """The labeled `5 on Hillstrom` check: the spoiling rule did what it says, the naive 95%
    interval misses the benchmark and LinearDML's contains it (all in proportions)."""
    problems = []
    if kept["control"] != 1.0:
        problems.append(f"only {kept['control']:.3f} of the No E-Mail customers were kept (all"
                        " should be)")
    for group, want in (("favored", KEEP_FAVORED), ("other", KEEP_OTHER)):
        if abs(kept[group] - want) > tol:
            problems.append(f"{kept[group]:.3f} of the {group} e-mailed customers were kept,"
                            f" expected {want:.2f} ± {tol:.2f}")
    if naive[0] <= benchmark <= naive[1]:
        problems.append(f"the naive 95% CI [{100 * naive[0]:.2f}, {100 * naive[1]:.2f}] pp"
                        f" contains the benchmark {100 * benchmark:.2f} pp")
    if not dml_ci[0] <= benchmark <= dml_ci[1]:
        problems.append(f"LinearDML's 95% CI [{100 * dml_ci[0]:.2f}, {100 * dml_ci[1]:.2f}] pp"
                        f" misses the benchmark {100 * benchmark:.2f} pp")
    if problems:
        raise AssertionError(
            "Your functions passed Checkpoints 1 and 5, so this compares the provided spoiled"
            " sample with the experiment: " + "; ".join(problems) + "." + INSTRUCTOR)


def check_hidden_confounder(shift: float, min_shift: float = 3.0) -> None:
    """The labeled `hidden confounder` check of the provided demonstration."""
    if not shift > min_shift:
        raise AssertionError(
            "This compares two provided fits, not your code: leaving out `mobile_share`, which"
            " drives both install and spend, should raise the estimate by several dollars, and it"
            f" moved it by {shift:+.2f}." + INSTRUCTOR)


# -------------------------------------------------------------- measurements (slow tests)
def measure_app(seed: int, quick: bool, *, n: int = synth.N, overfit: bool = True,
                linear: bool = True, hidden: bool = True) -> dict:
    """Run the lab's provided fits and reference solutions on the generator's data with
    ``seed`` (data, QUICK subsample, folds and models all from ``seed``) and report every
    quantity a checkpoint or a printed claim depends on."""
    from sklearn.metrics import r2_score, roc_auc_score

    clock = {}
    t0 = time.perf_counter()
    df = synth.app_install(seed, n=n).customers
    if quick:
        df = df.iloc[quick_rows(len(df), seed, min(N_QUICK, len(df) // 2))].reset_index(drop=True)
    y = df["spend_next_26w"].to_numpy(dtype=float)
    t = df["app_installed"].to_numpy()
    W = features(df)
    e = df["true_propensity"].to_numpy(dtype=float)
    tau = df["true_effect"].to_numpy(dtype=float)
    truths = true_averages(e, tau)
    base = baseline_estimates(y, t, W)
    folds = folds_for(W, t, seed)
    model_y, model_t = boosted(seed)
    y_hat = cross_fit(model_y, W, y, folds)
    t_hat = cross_fit(model_t, W, t, folds)
    clock["cross_fit"] = time.perf_counter() - t0
    mine = dml_estimate(y, t, y_hat, t_hat)
    out = {
        "seed": seed, "quick": quick, "n": len(df), "truth": truths,
        "naive": base.loc["naive"].to_dict(), "regression": base.loc["regression"].to_dict(),
        "r2": float(r2_score(y, y_hat)), "auc": float(roc_auc_score(t, t_hat)),
        "dml": mine,
        "dml_covers_overlap": mine["ci_low"] <= truths["overlap"] <= mine["ci_high"],
        "dml_covers_ate": mine["ci_low"] <= truths["ATE"] <= mine["ci_high"],
        "baselines_above_att": bool((base["ci_low"] > truths["ATT"]).all()),
    }
    trims = {}
    for low, high in ((0.05, 0.95), (0.1, 0.9), (0.2, 0.8)):
        tr = trimmed_estimate(y, t, y_hat, t_hat, low, high)
        mask = (t_hat >= low) & (t_hat <= high)
        tr["truth_kept"] = overlap_truth(e, tau, mask)
        tr["covers"] = tr["ci_low"] <= tr["truth_kept"] <= tr["ci_high"]
        trims[f"{low}-{high}"] = tr
    out["trimmed"] = trims
    out["share_outside_0.05_0.95"] = float(np.mean((t_hat < 0.05) | (t_hat > 0.95)))
    if overfit:
        t1 = time.perf_counter()
        oy, ot = boosted(seed, OVERFIT)
        # EconML and scikit-learn draw no progress bar, so there is no progressbar=False to pass.
        y_in = oy.fit(W, y).predict(W)
        t_in = ot.fit(W, t).predict_proba(W)[:, 1]
        ins = dml_estimate(y, t, y_in, t_in)
        out["overfit"] = {**ins, "se_ratio": ins["se"] / mine["se"],
                          "mean_abs_t_in": float(np.mean(np.abs(t - t_in))),
                          "mean_abs_t_oof": float(np.mean(np.abs(t - t_hat)))}
        clock["overfit"] = time.perf_counter() - t1
    if linear:
        t1 = time.perf_counter()
        est = fit_linear_dml(y, t, W, folds, model_y, model_t, seed)
        lo, hi = (float(np.ravel(v)[0]) for v in est.ate_interval(alpha=0.05))
        ate = float(np.ravel(est.ate())[0])
        out["linear"] = {"ate": ate, "ci_low": lo, "ci_high": hi,
                         "rel_diff": abs(ate - mine["theta"]) / abs(mine["theta"]),
                         "bound_rel_diff": max(abs(lo - mine["ci_low"]) / abs(mine["ci_low"]),
                                               abs(hi - mine["ci_high"]) / abs(mine["ci_high"])),
                         "covers": lo <= truths["overlap"] <= hi}
        clock["linear"] = time.perf_counter() - t1
    if hidden:
        t1 = time.perf_counter()
        keep_cols = [i for i, c in enumerate(FEATURES) if c != "mobile_share"]
        W2 = W[:, keep_cols]
        h = dml_estimate(y, t, cross_fit(model_y, W2, y, folds), cross_fit(model_t, W2, t, folds))
        out["hidden"] = {**h, "shift": h["theta"] - mine["theta"],
                         "covers_overlap": h["ci_low"] <= truths["overlap"] <= h["ci_high"]}
        clock["hidden"] = time.perf_counter() - t1
        be = synth.CREDIT / synth.MARGIN
        out["decision"] = {
            "offer": mine["ci_low"] > be,
            "offer_after_shift": mine["ci_low"] - out["hidden"]["shift"] > be,
            "point_after_shift_above_break_even": mine["theta"] - out["hidden"]["shift"] > be,
        }
    out["seconds"] = {k: round(v, 2) for k, v in clock.items()}
    return out


def measure_hillstrom(hill_arms: pd.DataFrame, seed: int, quick: bool,
                      half_seed: int | None = None) -> dict:
    """The provided Hillstrom cells with ``seed`` for the spoiling, the folds and the models,
    and ``half_seed`` (default ``seed``) for the QUICK half."""
    hill = hill_arms
    if quick:
        rows = quick_rows(len(hill), seed if half_seed is None else half_seed, len(hill) // 2)
        hill = hill.iloc[rows].reset_index(drop=True)
    bench, bench_se = visit_benchmark(hill)
    spoiled = confound(hill, seed)
    kept = kept_shares(hill, spoiled)
    ys = spoiled["visit"].to_numpy(dtype=float)
    ts = spoiled["treatment"].to_numpy()
    Ws = hillstrom_features(spoiled)
    base = baseline_estimates(ys, ts, Ws)
    model_y, model_t = boosted(seed)
    est = fit_linear_dml(ys, ts, Ws, folds_for(Ws, ts, seed), model_y, model_t, seed)
    lo, hi = (float(np.ravel(v)[0]) for v in est.ate_interval(alpha=0.05))
    naive = base.loc["naive"]
    return {
        "seed": seed, "quick": quick, "half_seed": half_seed, "n": len(hill),
        "n_spoiled": len(spoiled),
        "share_emailed": float(ts.mean()), "benchmark": bench, "benchmark_se": bench_se,
        "kept": kept, "naive": naive.to_dict(), "regression": base.loc["regression"].to_dict(),
        "linear": {"ate": float(np.ravel(est.ate())[0]), "ci_low": lo, "ci_high": hi},
        "naive_excludes": not naive["ci_low"] <= bench <= naive["ci_high"],
        "linear_covers": lo <= bench <= hi,
        "kept_ok": (kept["control"] == 1.0
                    and abs(kept["favored"] - KEEP_FAVORED) <= KEPT_TOLERANCE
                    and abs(kept["other"] - KEEP_OTHER) <= KEPT_TOLERANCE),
    }
