"""Module 15's double machine learning helpers (mktstats.dml): the reference implementations on
the toys worked by hand, and every checkpoint check passing on the reference and failing on each
wrong answer the lab brief lists (briefs/15-double-ml.md, "Failure messages"), with that
answer's message."""

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone

from mktstats import dml
from mktstats.synth import app_install as synth

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"
LAB = ROOT / "labs" / "src" / "15-double-ml.py"
SEED = 2035


@pytest.fixture(scope="module")
def committed():
    customers = pd.read_csv(SYNTH / "app_install_customers.csv")
    truth = json.loads((SYNTH / "truth_app_install.json").read_text())
    return customers, truth


@pytest.fixture(scope="module")
def app(committed):
    """The committed data with oracle predictions (the true q(x) and e(x)): fast, no fits."""
    df = committed[0]
    e = df["true_propensity"].to_numpy()
    tau = df["true_effect"].to_numpy()
    return {
        "y": df["spend_next_26w"].to_numpy(dtype=float),
        "t": df["app_installed"].to_numpy(),
        "W": dml.features(df),
        "e": e,
        "tau": tau,
        "y_hat": df["true_baseline"].to_numpy() + e * tau,
        "t_hat": e,
    }


def fails(fn, *args, match, **kwargs):
    with pytest.raises(AssertionError) as err:
        fn(*args, **kwargs)
    assert re.search(match, str(err.value)), str(err.value)
    return str(err.value)


# ------------------------------------------------------------------ references by hand
def test_baseline_estimates_on_the_toy():
    out = dml.baseline_estimates(dml.TOY1_Y, dml.TOY1_T, dml.TOY1_W)
    assert list(out.index) == ["naive", "regression"]
    assert list(out.columns) == ["estimate", "se", "ci_low", "ci_high"]
    assert out.loc["naive", "estimate"] == pytest.approx(8)
    assert out.loc["naive", "se"] == pytest.approx(np.sqrt(50 / 3))
    assert out.loc["regression", "estimate"] == pytest.approx(3, abs=1e-9)
    assert out.loc["regression", "se"] == pytest.approx(0, abs=1e-9)


def test_dml_estimate_on_the_toy():
    out = dml.dml_estimate(**dml.TOY3)
    assert out["theta"] == pytest.approx(2, abs=1e-12)
    assert out["se"] == pytest.approx(np.sqrt(10) / 3, abs=1e-12)
    assert out["ci_low"] == pytest.approx(-0.066, abs=1e-3)
    assert out["ci_high"] == pytest.approx(4.066, abs=1e-3)


def test_trimmed_estimate_on_the_toy():
    out = dml.trimmed_estimate(**dml.TOY4)
    kept = slice(1, 5)
    on_kept = dml.dml_estimate(dml.TOY4["y"][kept], dml.TOY4["t"][kept],
                               dml.TOY4["y_hat"][kept], dml.TOY4["t_hat"][kept])
    assert out["theta"] == pytest.approx(on_kept["theta"], abs=1e-12)
    assert out["theta"] == pytest.approx(2.381, abs=1e-3)
    assert out["share_kept"] == pytest.approx(4 / 6)
    assert out["weight_share_dropped"] == pytest.approx(0.0487 / 0.8087, abs=1e-12)


def test_the_overlap_truth_is_the_e_1_minus_e_weighted_mean(app):
    e, tau = app["e"], app["tau"]
    w = e * (1 - e)
    assert dml.overlap_truth(e, tau) == pytest.approx(np.sum(w * tau) / np.sum(w))
    mask = (e > 0.1) & (e < 0.9)
    assert dml.overlap_truth(e, tau, mask) == pytest.approx(
        np.sum(w[mask] * tau[mask]) / np.sum(w[mask]))


def test_the_lab_uses_the_modules_model_settings():
    """The lab writes its settings out for the learner; they must stay those the tolerances
    were measured with (mktstats.dml.BOOST and OVERFIT)."""
    src = LAB.read_text()
    for key, value in dml.BOOST.items():
        assert re.search(rf"^\s*{key}={value!r},", src, re.M), key
    assert ("OVERFIT = dict(BOOST, max_iter=600, learning_rate=0.1, max_leaf_nodes=255,"
            " min_samples_leaf=2)") in src
    assert dict(dml.BOOST, max_iter=600, learning_rate=0.1, max_leaf_nodes=255,
                min_samples_leaf=2) == dml.OVERFIT
    assert re.search(r"^SEED = 2035\b", src, re.M)
    block = re.search(r"^FEATURES = (\[.*?\])", src, re.M | re.S).group(1)
    assert ast.literal_eval(block) == list(dml.FEATURES)


# --------------------------------------------------------------------------- the data check
def test_check_data_passes_and_names_what_differs(committed):
    customers, truth = committed
    dml.check_data(customers, customers, truth, dml.features, dml.FEATURES, len(customers))
    half = customers.iloc[dml.quick_rows(len(customers), SEED)].reset_index(drop=True)
    dml.check_data(half, customers, truth, dml.features, dml.FEATURES, dml.N_QUICK)
    fails(dml.check_data, half, customers, truth, dml.features, dml.FEATURES, 20_000,
          match=r"10,000 customers in use, expected 20,000.*Rerun the install cell")
    leaky = [*dml.FEATURES[:-1], "true_effect"]
    fails(dml.check_data, customers, customers, truth,
          lambda df: np.column_stack([dml.features(df)[:, :-1], df["true_effect"]]), leaky,
          len(customers), match=r"features include \['true_effect'\]")
    stale = json.loads(json.dumps(truth))
    stale["estimands"]["ate"]["value"] += 1
    fails(dml.check_data, customers, customers, stale, dml.features, dml.FEATURES,
          len(customers), match=r"different versions.*tell the instructor")


# ------------------------------------------------------------------------- Checkpoint 1
def _naive(y, t, ddof=1, pooled=False):
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    if pooled:
        s2 = ((len(y1) - 1) * y1.var(ddof=1) + (len(y0) - 1) * y0.var(ddof=1)) / (len(y) - 2)
        se = np.sqrt(s2 * (1 / len(y1) + 1 / len(y0)))
    else:
        se = np.sqrt(np.var(y1, ddof=ddof) / len(y1) + np.var(y0, ddof=ddof) / len(y0))
    return est, se


def _baseline(param=1, constant=True, covariates=True, ddof=1, pooled=False, cov="HC1", z=1.96,
              index=("naive", "regression"), naive_cov=None, scale=(1.0, 1.0)):
    """A baseline_estimates with one mistake. ``naive_cov``: the naive standard error from a
    regression of y on t alone with that covariance type. ``scale``: multiply the (naive,
    regression) standard errors on the app data only, a near miss the toy cannot see."""
    def fn(y, t, W):
        n_est, n_se = _naive(y, t, ddof, pooled)
        if naive_cov:
            n_se = sm.OLS(y, sm.add_constant(t.astype(float))).fit(cov_type=naive_cov).bse[1]
        cols = np.column_stack([t, W]) if covariates else t
        X = sm.add_constant(cols) if constant else cols
        fit = sm.OLS(y, X).fit(cov_type=cov)
        r_est = np.ravel(fit.params)[param - (0 if constant else 1)]
        r_se = np.ravel(fit.bse)[param - (0 if constant else 1)]
        if len(y) > len(dml.TOY1_Y):
            n_se, r_se = n_se * scale[0], r_se * scale[1]
        est, se = np.array([n_est, r_est]), np.array([n_se, r_se])
        return pd.DataFrame({"estimate": est, "se": se, "ci_low": est - z * se,
                             "ci_high": est + z * se}, index=list(index))
    return fn


def test_checkpoint_1_passes_on_the_reference(app, committed):
    att = synth.estimands(app["e"], app["tau"], committed[0]["true_baseline"])["att"]
    out = dml.check_baseline_estimates(dml.baseline_estimates, app["y"], app["t"], app["W"], att)
    assert (out["ci_low"] > att).all()
    assert dml.check_baseline_estimates(_baseline(), app["y"], app["t"], app["W"], att).equals(out)


@pytest.mark.parametrize("wrong, message", [
    (dict(param=0), r"regression estimate is 2\b.*you took the constant.*params\[1\]"),
    (dict(param=2), r"regression estimate is 5\b.*coefficient on `w`"),
    (dict(constant=False), r"regression estimate is 3\.286.*no constant.*sm.add_constant"),
    (dict(covariates=False), r"regression estimate is 8\b.*left out the covariates"),
    (dict(ddof=0), r"standard error is 3\.333.*`np.var\(x, ddof=1\)`"),
    (dict(z=1.645), r"95% interval is estimate ± 1.96 × se"),
    (dict(index=("regression", "naive")), r"index \['naive', 'regression'\] and columns"),
])
def test_checkpoint_1_names_each_wrong_answer_on_the_toy(app, wrong, message):
    msg = fails(dml.check_baseline_estimates, _baseline(**wrong), app["y"], app["t"], app["W"],
                0.0, match=message)
    assert "workshop.use_reference(1)" in msg


@pytest.mark.parametrize("wrong, message", [
    (dict(pooled=True), r"naive standard error.*unpooled formula"),
    (dict(cov="nonrobust"), r"regression standard error.*`cov_type=\"HC1\"`"),
])
def test_checkpoint_1_checks_the_standard_errors_on_the_app_data(app, wrong, message):
    msg = fails(dml.check_baseline_estimates, _baseline(**wrong), app["y"], app["t"], app["W"],
                0.0, match=message)
    assert "on the app data" in msg and "workshop.use_reference(1)" in msg


@pytest.mark.parametrize("wrong, message", [
    # Other robust covariance types are within 0.06% of HC1 on the app data; the message used to
    # say that the default assumes equal noise, which is not what these did.
    (dict(cov="HC0"), r"regression standard error is 1\.7115.*`cov_type=\"HC0\"`, another robust"
                      r".*asks for `cov_type=\"HC1\"`"),
    (dict(cov="HC2"), r"regression standard error.*`cov_type=\"HC2\"`.*asks for"
                      r" `cov_type=\"HC1\"`"),
    (dict(cov="HC3"), r"regression standard error is 1\.7130.*`cov_type=\"HC3\"`.*asks for"
                      r" `cov_type=\"HC1\"`"),
    # A regression of y on t alone with HC1 equals the formula on the balanced toy and is 0.003%
    # off on the app data; it used to be told "not a pooled variance".
    (dict(naive_cov="HC1"), r"naive standard error is 2\.2949.*regression of y on t alone with"
                            r" `cov_type=\"HC1\"`.*unpooled formula.*ddof=1"),
    (dict(scale=(1.0005, 1.0)), r"naive standard error.*within 0\.1% of the reference \(0\.05%"
                                r" away\).*unpooled formula"),
    (dict(scale=(1.0, 1.0005)), r"regression standard error.*within 0\.1% of the reference"
                                r" \(0\.05% away\).*`cov_type=\"HC1\"`"),
])
def test_checkpoint_1_names_a_near_miss_in_the_standard_error(app, wrong, message):
    """On the committed data, so the numbers are those a learner sees (1.7115 and 1.7130 against
    1.7120; 2.2949 against 2.2950)."""
    msg = fails(dml.check_baseline_estimates, _baseline(**wrong), app["y"], app["t"], app["W"],
                0.0, match=message)
    assert "equally noisy" not in msg and "not a pooled variance" not in msg
    assert msg.endswith("and come back to this later.") and "workshop.use_reference(1)" in msg


def test_checkpoint_1_says_when_the_data_not_the_code_disagree(app):
    fails(dml.check_baseline_estimates, dml.baseline_estimates, app["y"], app["t"], app["W"],
          att=1000.0, match=r"matches the reference.*tell the instructor")


# ------------------------------------------------------------------------- Checkpoint 2
def _cross_fit(kind):
    def fn(model, X, target, folds):
        if kind == "in sample":
            fitted = clone(model).fit(X, target)
            pred = (fitted.predict_proba(X)[:, 1] if hasattr(fitted, "predict_proba")
                    else fitted.predict(X))
            return pred
        parts, out = [], np.full(len(target), np.nan)
        for train, test in folds:
            fitted = clone(model).fit(X[train], target[train])
            if kind == "two columns" and hasattr(fitted, "predict_proba"):
                out[test] = fitted.predict_proba(X[test])  # both columns into one: ValueError
                continue
            if kind == "labels":
                p = fitted.predict(X[test])
            elif kind == "column 0" and hasattr(fitted, "predict_proba"):
                p = fitted.predict_proba(X[test])[:, 0]
            elif hasattr(fitted, "predict_proba"):
                p = fitted.predict_proba(X[test])[:, 1]
            else:
                p = fitted.predict(X[test])
            parts.append(p)
            out[test] = p
        if kind == "appended":
            return np.concatenate(parts)
        if kind == "short":
            return out[:-1]
        return out
    return fn


def test_checkpoint_2_passes_on_the_reference_and_counts_the_toy():
    assert dml.check_cross_fit(dml.cross_fit, SEED) == {"n": 40, "in_sample": 40, "out_of_fold": 0}


@pytest.mark.parametrize("kind, message", [
    ("in sample", r"40 of 40 out-of-fold predictions equal the customer's own spend.*training"),
    ("appended", r"cross_val_predict.*`out\[test\] = \.\.\.`.*`clone\(model\)`"),
    ("labels", r"`predict_proba\(X\[test\]\)\[:, 1\]`.*not `predict`'s 0/1 labels"),
    ("column 0", r"column 1 is the probability of installing; column 0"),
    ("short", r"return 40 numbers.*one prediction per row of `X`"),
    # `predict_proba(X[test])` without `[:, 1]` used to end in a raw ValueError.
    ("two columns", r"raised `ValueError: shape mismatch: value array of shape \(8,2\).*two"
                    r" columns.*column 1 is P\(install\).*`predict_proba\(X\[test\]\)\[:, 1\]`"),
])
def test_checkpoint_2_names_each_wrong_answer(kind, message):
    msg = fails(dml.check_cross_fit, _cross_fit(kind), SEED, match=message)
    assert "workshop.use_reference(2)" in msg


# ------------------------------------------------------------------------- Checkpoint 3
def _dml(kind):
    def fn(y, t, y_hat, t_hat):
        u, v = y - y_hat, t - t_hat
        if kind == "t":
            v2 = t
            theta = np.sum(v2 * u) / np.sum(v2 * v2)
        elif kind == "y":
            theta = np.sum(v * y) / np.sum(v * v)
        elif kind == "intercept":
            theta = np.polyfit(v, u, 1)[0]
        elif kind == "sign":
            theta = np.sum(-v * u) / np.sum(v * v)
        else:
            theta = np.sum(v * u) / np.sum(v * v)
        r = u - theta * v
        n = len(y)
        se = np.sqrt(np.sum(v**2 * r**2)) / np.sum(v * v)
        se = {
            "classical": np.sqrt(np.sum(r**2) / (n - 1) / np.sum(v * v)),
            "hc1": se * np.sqrt(n / (n - 1)),
            "squared": se**2,
            "sqrt denominator": np.sqrt(np.sum(v**2 * r**2)) / np.sqrt(np.sum(v * v)),
            "residual": np.sqrt(np.sum(v**2 * u**2)) / np.sum(v * v),
            "means": np.sqrt(np.mean(v**2 * r**2)) / np.mean(v * v),
        }.get(kind, se)
        z = 1.645 if kind == "1.645" else 1.96
        return {"theta": theta, "se": se, "ci_low": theta - z * se, "ci_high": theta + z * se}
    return fn


def test_checkpoint_3_passes_on_the_reference(app):
    target = dml.overlap_truth(app["e"], app["tau"])
    out = dml.check_dml_estimate(dml.dml_estimate, app["y"], app["t"], app["y_hat"],
                                 app["t_hat"], target, ate=app["tau"].mean())
    assert out["ci_low"] <= target <= out["ci_high"]


@pytest.mark.parametrize("kind, message", [
    ("t", r"θ̂ is 0\.5\b.*you used `t`, not the install residual"),
    ("y", r"θ̂ is 1\.556.*you used `y`, not the spend residual"),
    ("intercept", r"θ̂ is 2\.353.*no intercept.*`np.polyfit`"),
    ("sign", r"θ̂ is -2\b.*sign flipped"),
    ("classical", r"standard error is 1\.277.*classical standard error"),
    ("hc1", r"standard error is 1\.217.*HC1.*HC0"),
    ("squared", r"standard error is 1\.111.*squared standard error"),
    ("sqrt denominator", r"standard error is 1\.118.*not by its square root"),
    ("residual", r"standard error is 1\.678.*Ỹ − θ̂T̃, not Ỹ"),
    ("means", r"standard error is 2\.108.*means where the formula has sums.*√n"),
    ("1.645", r"\[0\.266, 3\.734\].*uses 1\.96, not 1\.645"),
])
def test_checkpoint_3_names_each_wrong_answer(app, kind, message):
    msg = fails(dml.check_dml_estimate, _dml(kind), app["y"], app["t"], app["y_hat"],
                app["t_hat"], 0.0, 0.0, match=message)
    assert "workshop.use_reference(3)" in msg


def test_checkpoint_3_says_use_the_arguments(app):
    """A function that reads the notebook's 20,000-customer arrays: mixed with its arguments it
    used to end in a raw broadcast ValueError on the 4-customer toy; reading only the notebook's
    arrays, it was told to check the formula."""
    def mixed(y, t, y_hat_, t_hat_):  # residuals from the notebook's predictions
        return dml.dml_estimate(y, t, app["y_hat"], app["t_hat"])

    def only_globals(y_, t_, y_hat_, t_hat_):
        return dml.dml_estimate(app["y"], app["t"], app["y_hat"], app["t_hat"])

    for fn, match in [
        (mixed, r"raised `ValueError: operands could not be broadcast together with shapes"
                r" \(4,\) \(20000,\)`: it mixes its 4-customer arguments with the notebook's"
                r" 20,000-customer arrays\. Use the arguments"),
        (only_globals, r"θ̂ is [\d.]+, the estimate on the notebook's 20,000 customers; by hand"
                       r" it is 2: your function reads the notebook's arrays\. Use the arguments"),
    ]:
        msg = fails(dml.check_dml_estimate, fn, app["y"], app["t"], app["y_hat"], app["t_hat"],
                    0.0, 0.0, match=match)
        assert "workshop.use_reference(3)" in msg


def test_checkpoint_3_names_the_target_and_the_ate_when_the_truth_is_outside(app):
    msg = fails(dml.check_dml_estimate, dml.dml_estimate, app["y"], app["t"], app["y_hat"],
                app["t_hat"], 100.0, ate=13.58,
                match=r"overlap-weighted average effect, \$100\.00.*ATE.*\$13\.58.*instructor")
    assert "use_reference" not in msg


# ------------------------------------------------------------------------- Checkpoint 4
LOW, HIGH = 0.05, 0.95  # the notebook's constants, which the checkpoint passes as low and high


def _trim(kind):
    def fn(y, t, y_hat, t_hat, low, high):
        keep = (t_hat >= low) & (t_hat <= high)
        if kind == "constants":  # reads the notebook's LOW and HIGH, not its arguments
            keep = (t_hat >= LOW) & (t_hat <= HIGH)
        rows = slice(None) if kind == "all rows" else keep
        if kind == "y and t only":
            out = dict(dml.dml_estimate(y[keep], t[keep], y_hat, t_hat))
        else:
            out = dict(dml.dml_estimate(y[rows], t[rows], y_hat[rows], t_hat[rows]))
        w = (t * (1 - t)) if kind == "weights from t" else t_hat * (1 - t_hat)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["weight_share_dropped"] = {
                "share dropped": float(1 - keep.mean()),
                "weight kept": float(w[keep].sum() / w.sum()),
            }.get(kind, float(w[~keep].sum() / w.sum()))
        out["share_kept"] = float(keep.sum()) if kind == "count" else float(keep.mean())
        if kind == "share from constants":  # the estimate uses the arguments, the share does not
            out["share_kept"] = float(np.mean((t_hat >= LOW) & (t_hat <= HIGH)))
        return out
    return fn


def test_checkpoint_4_passes_on_the_reference(app):
    out = dml.check_trimmed_estimate(
        dml.trimmed_estimate, dml.dml_estimate, app["y"], app["t"], app["y_hat"], app["t_hat"],
        0.05, 0.95, lambda m: dml.overlap_truth(app["e"], app["tau"], m))
    assert out["weight_share_dropped"] < 1 - out["share_kept"]


@pytest.mark.parametrize("kind, message", [
    ("all rows", r"θ̂ is 2\.496.*estimate on all six rows.*same mask"),
    ("share dropped", r"weight_share_dropped is 0\.3333.*share of customers dropped"),
    ("weight kept", r"weight_share_dropped is 0\.9398.*share of weight kept"),
    ("weights from t", r"weights come from `t_hat`, not `t`"),
    ("count", r"share_kept is 4\b.*not a count"),
    # Passed silently: the toy and the app data are both called with 0.05 and 0.95.
    ("constants", r"`low, high = 0\.25, 0\.75` \(3 of 6 rows kept, share_kept 0\.5\) you get the"
                  r" same result as with 0\.05, 0\.95 \(share_kept 0\.6667\): your function"
                  r" ignores `low` and `high`: use the arguments, not `LOW` and `HIGH`\."),
    ("share from constants", r"`low, high = 0\.25, 0\.75` your result is .*share_kept: 0\.6667.*by"
                             r" hand it is .*share_kept: 0\.5\b.*using the arguments"),
    # Used to end in a raw ValueError.
    ("y and t only", r"raised `ValueError: operands could not be broadcast together with shapes"
                     r" \(4,\) \(6,\)`.*Subset all four arrays with the same mask"),
])
def test_checkpoint_4_names_each_wrong_answer(app, kind, message):
    msg = fails(dml.check_trimmed_estimate, _trim(kind), dml.dml_estimate, app["y"], app["t"],
                app["y_hat"], app["t_hat"], 0.05, 0.95, lambda m: 0.0, match=message)
    assert "workshop.use_reference(4)" in msg


def test_checkpoint_4_says_when_the_truth_is_outside(app):
    fails(dml.check_trimmed_estimate, dml.trimmed_estimate, dml.dml_estimate, app["y"],
          app["t"], app["y_hat"], app["t_hat"], 0.05, 0.95, lambda m: 100.0,
          match=r"among the kept customers, \$100\.00.*Your function is right.*instructor")


# ------------------------------------------------------------------------- Checkpoint 5
FAST = dict(max_iter=30, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=50,
            early_stopping=False)


@pytest.fixture(scope="module")
def small():
    df = synth.app_install(7, n=3000).customers
    y = df["spend_next_26w"].to_numpy(dtype=float)
    t = df["app_installed"].to_numpy()
    W = dml.features(df)
    folds = dml.folds_for(W, t, 7)
    model_y, model_t = dml.boosted(7, FAST)
    mine = dml.dml_estimate(y, t, dml.cross_fit(model_y, W, y, folds),
                            dml.cross_fit(model_t, W, t, folds))
    return {"y": y, "t": t, "W": W, "folds": folds, "model_y": model_y, "model_t": model_t,
            "mine": mine}


def _linear(kind):
    from econml.dml import LinearDML

    def fn(y, t, W, folds, model_y, model_t, seed):
        kw = dict(model_y=model_y, model_t=model_t, discrete_treatment=kind != "continuous",
                  random_state=seed)
        if kind != "default cv":
            kw["cv"] = folds
        est = LinearDML(**kw)
        if kind == "unfitted":
            return est
        if kind == "as X":
            return est.fit(y, t, X=W, W=None)
        est.fit(y, t, X=None, W=W)
        if kind == "none":
            return None
        if kind == "alpha":
            est.ate_interval = lambda alpha=0.05: LinearDML.ate_interval(est, alpha=0.10)
        return est
    return fn


def test_checkpoint_5_passes_on_the_reference(small):
    s = small
    est, ate, lo, hi = dml.check_linear_dml(dml.fit_linear_dml, s["y"], s["t"], s["W"],
                                            s["folds"], s["model_y"], s["model_t"], 7,
                                            s["mine"], target=s["mine"]["theta"])
    assert ate == pytest.approx(s["mine"]["theta"], rel=1e-9)
    # EconML's width = yours x sqrt(n / (n - 1)) (HC1 vs HC0) x 1.959964 / 1.96 (it uses the
    # exact normal quantile; the lab uses 1.96).
    assert (hi - lo) / (s["mine"]["ci_high"] - s["mine"]["ci_low"]) == pytest.approx(
        np.sqrt(3000 / 2999) * stats.norm.ppf(0.975) / 1.96, rel=1e-9)


@pytest.mark.parametrize("kind, message", [
    ("continuous", r"Cannot use a classifier.*`discrete_treatment=True`"),
    ("as X", r"X was not None when fitting.*`W=W`, with `X=None`"),
    ("default cv", r"Pass `cv=folds`.*default is `cv=2`"),
    ("none", r"returned NoneType.*return the fitted estimator"),
    ("alpha", r"`est.ate_interval\(alpha=0.05\)`"),
    # Used to end in a raw AttributeError from est.ate().
    ("unfitted", r"raised `AttributeError: 'LinearDML' object has no attribute '_d_t_in'`.*never"
                 r" fitted.*fit it, `\.fit\(y, t, X=None, W=W\)`, and return the result\."),
])
def test_checkpoint_5_turns_each_mistake_into_a_message(small, kind, message):
    s = small
    msg = fails(dml.check_linear_dml, _linear(kind), s["y"], s["t"], s["W"], s["folds"],
                s["model_y"], s["model_t"], 7, s["mine"], s["mine"]["theta"], match=message)
    assert "workshop.use_reference(5)" in msg


def test_checkpoint_5_says_when_the_truth_is_outside(small):
    s = small
    fails(dml.check_linear_dml, dml.fit_linear_dml, s["y"], s["t"], s["W"], s["folds"],
          s["model_y"], s["model_t"], 7, s["mine"], target=1000.0,
          match=r"outside EconML's 95% interval.*instructor")


# ------------------------------------------------------- labeled checks of provided code
def test_the_hillstrom_check():
    ok = {"control": 1.0, "favored": 0.905, "other": 0.198}
    dml.check_hillstrom(ok, (0.054, 0.069), (0.038, 0.055), 0.045)
    for kept, naive, ci, match in [
        ({**ok, "control": 0.99}, (0.054, 0.069), (0.038, 0.055), r"No E-Mail"),
        ({**ok, "favored": 0.85}, (0.054, 0.069), (0.038, 0.055), r"favored.*0\.90 ± 0\.02"),
        ({**ok, "other": 0.25}, (0.054, 0.069), (0.038, 0.055), r"other.*0\.20 ± 0\.02"),
        (ok, (0.040, 0.069), (0.038, 0.055), r"naive 95% CI.*contains the benchmark"),
        (ok, (0.054, 0.069), (0.050, 0.060), r"LinearDML's 95% CI.*misses the benchmark"),
    ]:
        msg = fails(dml.check_hillstrom, kept, naive, ci, 0.045, match=match)
        assert msg.startswith("Your functions passed Checkpoints 1 and 5")
        assert msg.endswith("Tell the instructor.")


def test_the_demonstration_checks():
    assert dml.check_no_cross_fitting(9.0, 2.0) == pytest.approx(4.5)
    fails(dml.check_no_cross_fitting, 3.0, 2.0, match=r"provided fits, not your code.*only 1\.5")
    dml.check_hidden_confounder(6.4)
    fails(dml.check_hidden_confounder, 2.0, match=r"`mobile_share`.*moved it by \+2\.00")
    fails(dml.check_hidden_confounder, -1.0, match=r"moved it by -1\.00.*Tell the instructor")


def test_confound_keeps_every_control_and_the_stated_shares():
    rng = np.random.default_rng(1)
    n = 40_000
    df = pd.DataFrame({"recency": rng.integers(1, 13, n), "history": rng.gamma(2, 120, n),
                       "treatment": rng.integers(0, 2, n)})
    spoiled = dml.confound(df, 3)
    assert spoiled.index.equals(pd.RangeIndex(len(spoiled)))
    kept = dml.kept_shares(df, spoiled)
    assert kept["control"] == 1.0
    assert kept["favored"] == pytest.approx(0.9, abs=0.02)
    assert kept["other"] == pytest.approx(0.2, abs=0.02)
    assert dml.confound(df, 3).equals(spoiled) and not dml.confound(df, 4).equals(spoiled)
