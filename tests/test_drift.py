"""Module 17's helpers (mktstats.drift): the reference implementations on the hand-worked toys of
the lab brief, and every checkpoint check passing on the reference and failing on each wrong
answer the brief lists (briefs/17-mmm-time-varying.md, "Failure messages"), with that answer's
message. Model fits are faked with draws built around the truth, so nothing here samples."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mktstats import checks
from mktstats import drift as dr
from mktstats.synth.mmm_drift import mmm_drift

CH = dr.CHANNELS


@pytest.fixture(scope="module")
def res():
    return mmm_drift()


@pytest.fixture(scope="module")
def arrays(res):
    spend = dr.spend_array(res.weekly)
    tc = dr.true_contrib_array(res.latent)
    return spend, tc, dr.windows_of(res.truth)


def fails(fn, *args, match):
    with pytest.raises(AssertionError, match=match) as info:
        fn(*args)
    msg = str(info.value)
    return msg


def recovers(msg, n):
    assert f"workshop.use_reference({n})" in msg, msg


# --------------------------------------------------------------- Exercise 1: window_roas
def test_window_roas_toy_by_hand():
    d4 = pd.date_range("2023-01-02", periods=4, freq="7D")
    s = xr.DataArray([[4.0], [3.0], [2.0], [2.0]], dims=("date", "channel"),
                     coords={"date": d4, "channel": ["x"]})
    c = xr.DataArray([[1.0], [1.0], [1.0], [5.0]], dims=("date", "channel"),
                     coords={"date": d4, "channel": ["x"]})
    assert float(dr.window_roas(c, s, "2023-01-09", "2023-01-23").squeeze()) == pytest.approx(1.0)


def test_check_window_roas_passes_and_reproduces_truth(arrays, res):
    spend, tc, W = arrays
    out = dr.check_window_roas(dr.window_roas, tc, spend, W, res.truth)
    assert float(out["last_quarter"].sel(channel="social")) == pytest.approx(2.108, abs=1e-3)


def _sel(a, start, end, lo_open=False, hi_open=False):
    d = a.coords["date"]
    keep = (d > np.datetime64(start)) if lo_open else (d >= np.datetime64(start))
    keep &= (d < np.datetime64(end)) if hi_open else (d <= np.datetime64(end))
    return a.where(keep, drop=True)


WRONG_WINDOW = {
    "mean of weekly ratios": (lambda c, s, a, b: (_sel(c, a, b) / _sel(s, a, b)).mean("date"),
                              "averaged the weekly ratios"),
    "last week dropped": (lambda c, s, a, b: _sel(c, a, b, hi_open=True).sum("date")
                          / _sel(s, a, b, hi_open=True).sum("date"), "last week is missing"),
    "first week dropped": (lambda c, s, a, b: _sel(c, a, b, lo_open=True).sum("date")
                           / _sel(s, a, b, lo_open=True).sum("date"), "first week is missing"),
    "window ignored": (lambda c, s, a, b: c.sum("date") / s.sum("date"), "ignores start and end"),
    "spend over all weeks": (lambda c, s, a, b: _sel(c, a, b).sum("date") / s.sum("date"),
                             "spend over all weeks"),
    "axis number": (lambda c, s, a, b: _sel(c, a, b).sum(dim=c.dims[0])
                    / _sel(s, a, b).sum(dim=s.dims[0]), "by name"),
    "numpy result": (lambda c, s, a, b: (_sel(c, a, b).sum("date") / _sel(s, a, b).sum("date"))
                     .values, "not an xarray DataArray"),
    "inverted": (lambda c, s, a, b: _sel(s, a, b).sum("date") / _sel(c, a, b).sum("date"),
                 "divided spend by contribution"),
    # passes toy 1 (one channel); before the fix toy 2 said "must be 2 for every channel; your
    # function gives 2.0"
    "sum over every dim": (lambda c, s, a, b: c.sel(date=slice(a, b)).sum()
                           / s.sel(date=slice(a, b)).sum(), r"returned one number \(2\) instead"
                           r" of one per channel.*sum over \"date\" only"),
    # passes toys 1 and 2 (date is the first dim); raised a bare TypeError on toy 3
    ".loc": (lambda c, s, a, b: c.loc[a:b].sum("date") / s.loc[a:b].sum("date"),
             r"raised TypeError.*\.loc\[start:end\] selects along the first dim, which here is"
             " chain"),
}


@pytest.mark.parametrize("name", list(WRONG_WINDOW))
def test_check_window_roas_wrong_answers(arrays, res, name):
    spend, tc, W = arrays
    fn, match = WRONG_WINDOW[name]
    recovers(fails(dr.check_window_roas, fn, tc, spend, W, res.truth, match=match), 1)


def test_check_window_roas_wrong_dates_on_truth(arrays, res):
    spend, tc, W = arrays
    shifted = {k: (str(pd.Timestamp(a) + pd.Timedelta(days=7))[:10], b) for k, (a, b) in W.items()}
    msg = fails(dr.check_window_roas, dr.window_roas, tc, spend, shifted, res.truth,
                match="WINDOWS")
    recovers(msg, 1)


# ------------------------------------------------- labeled check "static model" (faked)
def _draws_like(tc, rng, n=(2, 50), sd=0.03, factor=None):
    noise = rng.lognormal(0, sd, size=(*n, tc.sizes["channel"]))
    out = xr.DataArray(noise, dims=("chain", "draw", "channel"),
                       coords={"channel": tc.coords["channel"].values}) * tc
    out = out.transpose("chain", "draw", "date", "channel")
    if factor is not None:
        out = out * factor
    return out


def test_check_static_model(arrays, res):
    spend, tc, W = arrays
    rng = np.random.default_rng(0)
    # a static fit gives social about its average ROAS in every week: scale the last quarter up
    lq = xr.where((tc.date >= np.datetime64(W["last_quarter"][0]))
                  & (tc.channel == "social"), 1.8, 1.0)
    fake = _draws_like(tc, rng, factor=lq)
    cos = dr.window_roas(fake, spend, *W["all"])
    out = dr.check_static_model(dr.window_roas, fake, spend, cos, W, res.truth)
    assert set(out) == set(W)
    msg = fails(dr.check_static_model, dr.window_roas, fake, spend, cos + 0.01, W, res.truth,
                match="contribution_over_spend")
    assert "use_reference(1)" in msg
    honest = _draws_like(tc, rng)  # draws around the truth: the HDI covers it
    msg = fails(dr.check_static_model, dr.window_roas, honest, spend,
                dr.window_roas(honest, spend, *W["all"]), W, res.truth, match="not your code")


# ------------------------------------------------------- Exercise 2: residual_spend_corr
def test_corr_toy_by_hand():
    r, s, y = dr._toy_corr_inputs()
    got = dr.residual_spend_corr(r, s, y)
    np.testing.assert_allclose(got.loc[2023], [0.8854, -0.3162], atol=1e-4)
    np.testing.assert_allclose(got.loc[2024], [-1.0, -0.5963], atol=1e-4)
    # the wrong answers the brief lists, computed from their definitions
    np.testing.assert_allclose(s.corrwith(r), [0.369, -0.312], atol=1e-3)
    sp = s[y == 2023].corrwith(r[y == 2023], method="spearman")
    np.testing.assert_allclose(sp, [1.0, -0.4], atol=1e-12)
    assert np.cov(s.loc[y == 2023, "a"], r[y == 2023])[0, 1] == pytest.approx(14 / 3)


def _corr_inputs(res, social_signal=True):
    """Residuals that, like a static fit's, correlate with social's spend positively early and
    negatively late (or pure noise)."""
    w, lat = res.weekly, res.latent
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 3_000, len(w))
    if social_signal:
        noise = noise + (lat["multiplier_rel_social"] - 1) * w["social"] * 2.0
    resid = pd.Series(noise, index=w.index)
    return resid, w[CH], w["date_week"].dt.year


def _spearman(r, s, y):
    return pd.DataFrame({yr: s[y == yr].corrwith(r[y == yr], method="spearman")
                         for yr in sorted(y.unique())}).T


def _cov(r, s, y):
    out = {}
    for yr in sorted(y.unique()):
        part = r[y == yr]
        out[yr] = s[y == yr].apply(lambda col, part=part: np.cov(col, part)[0, 1])
    return pd.DataFrame(out).T


WRONG_CORR = {
    "all weeks": (lambda r, s, y: s.corrwith(r).to_frame().T, "all weeks at once"),
    "spearman": (_spearman, "Spearman"),
    "covariance": (_cov, "covariance"),
    "transposed": (lambda r, s, y: dr.residual_spend_corr(r, s, y).T, "transpose"),
    "series": (lambda r, s, y: s.corrwith(r), "not a pandas"),
}


@pytest.mark.parametrize("name", list(WRONG_CORR))
def test_check_residual_spend_corr_wrong_answers(res, name):
    fn, match = WRONG_CORR[name]
    recovers(fails(dr.check_residual_spend_corr, fn, *_corr_inputs(res), match=match), 2)


def test_check_residual_spend_corr_on_a_fit(res):
    table = dr.check_residual_spend_corr(dr.residual_spend_corr, *_corr_inputs(res))
    assert table.loc[2023, "social"] > 0 > table.loc[2025, "social"]
    fails(dr.check_residual_spend_corr, dr.residual_spend_corr,
          *_corr_inputs(res, social_signal=False), match="not your code")


# ---------------------------------------------------------------- Exercise 3: drift_ratio
def _prior_like(res, rng):
    T = len(res.weekly)
    vals = rng.lognormal(0, 0.3, size=(1, 40, T, len(CH)))
    return xr.DataArray(vals, dims=("chain", "draw", "date", "channel"),
                        coords={"date": res.weekly["date_week"].to_numpy(), "channel": CH})


def test_linear_toy_ratio(res):
    W = dr.windows_of(res.truth)
    toy = dr.linear_toy(res.weekly["date_week"])
    assert float(dr.drift_ratio(toy, W["first_year"], W["last_quarter"]).squeeze()) == (
        pytest.approx(0.56591, abs=1e-5))


def _all_but_channel(m):
    """Average over every dim but channel: right on the toys, wrong on prior draws."""
    return m.mean([d for d in m.dims if d != "channel"])


WRONG_RATIO = {
    "swapped": (lambda m, e, z: m.sel(date=slice(*e)).mean("date") / m.sel(date=slice(*z))
                .mean("date"), "swapped"),
    "sums": (lambda m, e, z: m.sel(date=slice(*z)).sum("date") / m.sel(date=slice(*e))
             .sum("date"), "used sums"),
    "last week lost": (lambda m, e, z: _sel(m, *z, hi_open=True).mean("date")
                       / _sel(m, *e, hi_open=True).mean("date"), "lost its last week"),
    "all dims": (lambda m, e, z: _all_but_channel(m.sel(date=slice(*z)))
                 / _all_but_channel(m.sel(date=slice(*e))), "every prior draw keeps"),
    "empty window": (lambda m, e, z: m.sel(date=slice(z[1], z[0])).mean("date")
                     / m.sel(date=slice(*e)).mean("date"), "selected no weeks"),
    # equal to the mean on the constant and the linear toy; passed every check before the fix
    "median": (lambda m, e, z: m.sel(date=slice(*z)).median("date")
               / m.sel(date=slice(*e)).median("date"), "ratio of medians"),
}


def test_spike_toy_tells_mean_from_median(res):
    W = dr.windows_of(res.truth)
    toy = dr.spike_toy(res.weekly["date_week"], W["last_quarter"])
    late = toy.sel(date=slice(*W["last_quarter"]))
    assert late.sizes["date"] == 13 and float(late.max()) == 14.0
    assert float(dr.drift_ratio(toy, W["first_year"], W["last_quarter"]).squeeze()) == 2.0
    assert float(late.median("date").squeeze()) == 1.0


@pytest.mark.parametrize("name", list(WRONG_RATIO))
def test_check_drift_ratio_wrong_answers(res, name):
    fn, match = WRONG_RATIO[name]
    prior = _prior_like(res, np.random.default_rng(2))
    msg = fails(dr.check_drift_ratio, fn, res.weekly["date_week"], dr.windows_of(res.truth),
                prior, match=match)
    recovers(msg, 3)


def test_check_drift_ratio_passes(res):
    prior = _prior_like(res, np.random.default_rng(2))
    got = dr.check_drift_ratio(dr.drift_ratio, res.weekly["date_week"], dr.windows_of(res.truth),
                               prior)
    assert set(got.dims) == {"chain", "draw", "channel"}


# -------------------------------------------------------------- Exercise 4: path_coverage
def test_coverage_toys(res):
    toys = dr.coverage_toys(res.weekly["date_week"])
    assert dr.path_coverage(*toys["a"]) == 1.0
    assert dr.path_coverage(*toys["b"]) == 0.0
    alt = toys["c"][1]
    assert [float(np.mean(alt == v)) for v in dr.TOY_C_TRUE] == [pytest.approx(1 / 3)] * 3
    assert dr.path_coverage(*toys["c"]) == pytest.approx(2 / 3, abs=1e-12)
    draws_c = toys["c"][0].isel(date=0).values
    assert checks.interval(draws_c, 0.94, "hdi") == pytest.approx((0.0005, 2.8051), abs=1e-4)
    # each wrong interval misses one of 0.01 and 2.5, and -0.5 as well
    for prob, kind, want in [(0.94, "eti", (0.031, 3.491)), (0.89, "hdi", (0.0005, 2.203)),
                             (0.89, "eti", (0.057, 2.892))]:
        assert checks.interval(draws_c, prob, kind) == pytest.approx(want, abs=1e-3)
        lo, hi = checks.interval(draws_c, prob, kind)
        assert np.mean((alt >= lo) & (alt <= hi)) == pytest.approx(1 / 3)
    # mean +/- 1.88 sd reaches below 0, so it covers all three
    m, s = draws_c.mean(), draws_c.std()
    assert m - 1.88 * s < -0.5 < 0 < draws_c.min()
    # one chain gives 1/3 on toy (c) and the right answer when both chains hold the same draws
    one = toys["c"][0].isel(chain=[0])
    assert dr.path_coverage(one, alt) == pytest.approx(1 / 3)
    assert dr.path_coverage(toys["c_same"][0].isel(chain=[0]), alt) == (
        dr.path_coverage(*toys["c_same"]))
    assert dr.path_coverage(*toys["c_same"]) == pytest.approx(2 / 3, abs=1e-12)
    d = dr.path_coverage(*toys["d"])
    assert 0 < d < 0.1
    pooled = checks.interval(toys["d"][0].values, 0.94, "hdi")
    assert np.mean((toys["d"][1] >= pooled[0]) & (toys["d"][1] <= pooled[1])) > 0.85


def _cov_fn(kind="hdi", prob=None, outside=False, count=False, pooled=False, normal=False,
            chain=None, mean_chains=False, percent=False, per_week=False):
    def fn(mult_draws, true_path, p=0.94):
        p = prob or p
        t = np.asarray(true_path, dtype=float)
        if chain is not None:
            mult_draws = mult_draws.isel(chain=chain)
        if mean_chains:
            mult_draws = mult_draws.mean("chain")
        if pooled:
            lo, hi = checks.interval(mult_draws.values, p, kind)
            inside = (t >= lo) & (t <= hi)
        else:
            def bounds(d):
                if normal:  # 94% of a Normal lies within 1.88 sd of its mean
                    return d.mean() - 1.88 * d.std(), d.mean() + 1.88 * d.std()
                return checks.interval(d, p, kind)
            inside = np.array([lo <= t[k] <= hi for k, (lo, hi) in enumerate(
                bounds(mult_draws.isel(date=k).values)
                for k in range(mult_draws.sizes["date"]))])
        if outside:
            inside = ~inside
        if per_week:
            return pd.Series(inside)
        if percent:
            return 100 * float(inside.mean())
        return float(inside.sum()) if count else float(inside.mean())
    return fn


def _mult_like(res, rng, social_shift=0.0):
    rel = np.stack([res.latent[f"multiplier_rel_{c}"].to_numpy() for c in CH], axis=-1)
    rel = rel + np.array([social_shift if c == "social" else 0.0 for c in CH])
    vals = rel[None, None] * rng.lognormal(0, 0.02, size=(2, 100, *rel.shape))
    return xr.DataArray(vals, dims=("chain", "draw", "date", "channel"),
                        coords={"date": res.weekly["date_week"].to_numpy(), "channel": CH})


WRONG_COVERAGE = {
    "outside share": (_cov_fn(outside=True), "share of weeks outside"),
    "count": (_cov_fn(count=True), "count of weeks"),
    "equal-tailed": (_cov_fn(kind="eti"), "not the 94% HDI"),
    "89% HDI": (_cov_fn(prob=0.89), "not the 94% HDI"),
    "pooled": (_cov_fn(pooled=True), "one interval for all weeks"),
    # passed all four toys before the fix and failed on the fit with a message about prob and
    # chains
    "Normal approximation": (_cov_fn(normal=True), r"contains -0\.5.*Normal approximation"),
    # raised a bare TypeError ("cannot convert the series to <class 'float'>")
    "per-week values": (_cov_fn(per_week=True), "returned a Series of 156 values, one per week,"
                        " not their share"),
    # got the generic message
    "percentage": (_cov_fn(percent=True), "you returned a percentage. A share lies between 0"
                   " and 1"),
    # got the equal-tailed / 89% message
    "chain 0 only": (_cov_fn(chain=[0]), "does not use every draw of both chains"),
    "chain 1 only": (_cov_fn(chain=[1]), "does not use every draw of both chains"),
    "chains averaged": (_cov_fn(mean_chains=True), "does not use every draw of both chains"),
}


@pytest.mark.parametrize("name", list(WRONG_COVERAGE))
def test_check_path_coverage_wrong_answers(res, name):
    fn, match = WRONG_COVERAGE[name]
    mult = _mult_like(res, np.random.default_rng(3))
    recovers(fails(dr.check_path_coverage, fn, mult, res.latent, match=match), 4)


def test_check_path_coverage_on_a_fit(res):
    rng = np.random.default_rng(3)
    out = dr.check_path_coverage(dr.path_coverage, _mult_like(res, rng), res.latent)
    assert out == {c: 1.0 for c in CH}
    fails(dr.check_path_coverage, dr.path_coverage, _mult_like(res, rng, social_shift=0.5),
          res.latent, match="not your code")


# ---------------------------------------------------------- labeled check "drifting model"
def _ratio_like(centers, rng):
    vals = np.stack([rng.normal(m, 0.05, size=(2, 200)) for m in centers], axis=-1)
    return xr.DataArray(vals, dims=("chain", "draw", "channel"), coords={"channel": CH})


def test_check_drifting_model(res):
    rng = np.random.default_rng(4)
    table = dr.check_drifting_model(_ratio_like([1.0, 1.0, 0.55, 1.0], rng), res.truth)
    assert table.loc["social", "true_ratio"] == pytest.approx(0.4195, abs=1e-4)
    fails(dr.check_drifting_model, _ratio_like([1.0, 1.0, 0.95, 1.0], rng), res.truth,
          match="social's HDI")
    fails(dr.check_drifting_model, _ratio_like([0.7, 1.0, 0.55, 1.0], rng), res.truth,
          match="tv's HDI")


# ------------------------------------------------------------------ labeled check "prophet"
def test_check_prophet_on_the_committed_data(res):
    w = res.weekly
    comp = dr.prophet_components(w, dr.holidays_table(w))
    dr.check_prophet(comp, w)
    bad = comp.copy()
    bad.loc[w.index[w["holiday"] == 0][0], "holiday"] = 5.0
    fails(dr.check_prophet, bad, w, match="not a holiday week")
    fails(dr.check_prophet, comp.drop(columns="season"), w, match="provided code")
    off = comp.assign(yhat=comp["yhat"] + 1.0)
    fails(dr.check_prophet, off, w, match="add up")


# ------------------------------------------------------- Exercise 5: baseline_comparison
def test_comparison_toy_by_hand():
    own, pro, t = dr.comparison_toys()
    got = dr.baseline_comparison(own, pro, t)
    assert got.loc[("a", "own"), "mean"] == pytest.approx(2.0919, abs=1e-4)
    assert got.loc[("a", "prophet"), "mean"] == pytest.approx(3.1379, abs=1e-4)
    assert (got.loc[("a", "own"), "hdi_low"], got.loc[("a", "own"), "hdi_high"]) == (
        pytest.approx((1.0209, 3.2904), abs=1e-4))
    assert (got.loc[("a", "prophet"), "hdi_low"], got.loc[("a", "prophet"), "hdi_high"]) == (
        pytest.approx((1.5314, 4.9357), abs=1e-4))
    assert got.loc[("a", "own"), "abs_error"] == pytest.approx(0.4081, abs=1e-4)
    assert got.loc[("b", "own"), "abs_error"] == pytest.approx(0.9919, abs=1e-4)
    assert got.loc[("b", "prophet"), "abs_error"] == pytest.approx(2.0379, abs=1e-4)
    assert got["covered"].tolist() == [True, True, True, False]
    eti = checks.interval(own.sel(channel="b").values, 0.94, "eti")
    assert eti == pytest.approx((1.1399, 3.5090), abs=1e-4)


def _comparison(kind="hdi", signed=False, swapped=False, flat=False, labels=("own", "prophet"),
                prob=None, stat=np.mean, relative=False, covered_by_mean=False):
    def fn(roas_own, roas_prophet, true_roas, p=0.94):
        p = prob or p
        draws_in_order = (roas_prophet, roas_own) if swapped else (roas_own, roas_prophet)
        pairs = dict(zip(labels, draws_in_order, strict=True))
        rows = []
        for b, draws in pairs.items():
            for c in draws.coords["channel"].values:
                d = draws.sel(channel=c).values.ravel()
                lo, hi = checks.interval(d, p, kind)
                t = true_roas[str(c)]
                m = stat(d)
                err = m - t
                err = err if signed else abs(err) / t if relative else abs(err)
                covered = abs(m - t) < 0.5 if covered_by_mean else lo <= t <= hi
                rows.append({"channel": str(c), "baseline": b, "mean": m, "hdi_low": lo,
                             "hdi_high": hi, "true_roas": t, "abs_error": err,
                             "covered": covered})
        df = pd.DataFrame(rows)
        return df if flat else df.set_index(["channel", "baseline"])
    return fn


WRONG_COMPARISON = {
    "signed error": (_comparison(signed=True), "signed error"),
    "equal-tailed": (_comparison(kind="eti"), "That is the equal-tailed interval"),
    "swapped": (_comparison(swapped=True), "swapped"),
    "no index": (_comparison(flat=True), "index"),
    # before the fix: the index-and-columns message, which printed the required names
    "capitalized labels": (_comparison(labels=("Own", "Prophet")),
                           r"baseline labels in your index are \['Own', 'Prophet'\]; they must be"
                           ' exactly "own" and "prophet"'),
    # before the fix: "probably the equal-tailed interval"
    "89% HDI": (_comparison(prob=0.89), "or an interval at a different probability.*pass"
                " prob=0.94"),
    # before the fix: the equal-tailed message, which contradicted the right interval it printed
    "covered from the mean": (_comparison(covered_by_mean=True), "your interval for own b is"
                              " right.*covered is hdi_low <= true_roas <= hdi_high"),
    # before the fix: two dicts of unrounded floats
    "median": (_comparison(stat=np.median), r"row \(a, own\) differs from the reference in mean,"
               r" abs_error: mean 2\.0000, the reference 2\.0919.*the median of the draws"),
    "relative error": (_comparison(relative=True), r"in abs_error: abs_error 0\.1632, the"
                       r" reference 0\.4081.*relative error"),
}


def _roas_like(means, rng):
    vals = np.stack([rng.normal(m, 0.15, size=(2, 300)) for m in means], axis=-1)
    return xr.DataArray(vals, dims=("chain", "draw", "channel"), coords={"channel": CH})


@pytest.mark.parametrize("name", list(WRONG_COMPARISON))
def test_check_baseline_comparison_wrong_answers(res, name):
    fn, match = WRONG_COMPARISON[name]
    rng = np.random.default_rng(5)
    true = {c: res.truth["channels"][c]["roas"] for c in CH}
    msg = fails(dr.check_baseline_comparison, fn, _roas_like([1.6, 3.0, 3.9, 0.6], rng),
                _roas_like([1.4, 3.0, 2.9, 0.6], rng), true, match=match)
    recovers(msg, 5)


def test_check_baseline_comparison_on_fits(res):
    rng = np.random.default_rng(5)
    true = {c: res.truth["channels"][c]["roas"] for c in CH}
    table = dr.check_baseline_comparison(dr.baseline_comparison,
                                         _roas_like([1.6, 3.0, 3.9, 0.6], rng),
                                         _roas_like([1.4, 3.0, 2.9, 0.6], rng), true)
    assert list(table.index.names) == ["channel", "baseline"] and len(table) == 8
    fails(dr.check_baseline_comparison, dr.baseline_comparison,
          _roas_like([1.6, 3.0, 3.9, 0.6], rng), _roas_like([1.4, 3.0, 4.0, 0.6], rng), true,
          match="not your code")


def test_counterfactual_formula_matches_the_truth(res):
    """The truth's roas_last_quarter_counterfactual is counterfactual_window_roas with the true
    parameters (pymc-marketing's definition; the slow test checks it against
    contribution_over_spend on posterior draws)."""
    from mktstats.synth.mmm_drift import ADSTOCK_ALPHA, SATURATION_LAM

    t = res.truth["channels"]["social"]
    n = len(res.weekly)
    got = dr.counterfactual_window_roas(
        res.weekly["social"].to_numpy(dtype=float), [ADSTOCK_ALPHA["social"]],
        [SATURATION_LAM["social"]], [t["saturation_beta_sales_units"]],
        res.latent["multiplier_social"].to_numpy()[None, :], n - 13, n - 1)
    assert got[0] == pytest.approx(t["roas_last_quarter_counterfactual"], rel=1e-5)
    assert t["roas_last_quarter_counterfactual"] < t["roas_last_quarter"]


def test_lognormal_toy_has_median_2():
    own, _, _ = dr.comparison_toys()
    q = np.sort(own.sel(channel="a").values.ravel())
    assert q[499] < 2 < q[500]


def test_drifting_model_cell_gives_the_recovery_line_without_exercise_3(capsys):
    """The lab's "drifting model" cell calls the learner's drift_ratio inside the labeled check,
    so a learner who skipped Exercise 3 gets the harness's recovery line, not only a bare
    NotImplementedError (before the fix the call came before the `with` block)."""
    from mktstats.harness import Workshop

    jupytext = pytest.importorskip("jupytext")
    lab = Path(__file__).resolve().parent.parent / "labs" / "src" / "17-mmm-time-varying.py"
    cell = next(c.source for c in jupytext.read(lab, fmt="py:percent").cells
                if 'checkpoint(label="drifting model")' in c.source)

    def drift_ratio(mult, early, late):
        raise NotImplementedError("TODO 3")

    ns = {"drift_ratio": drift_ratio, "mult": None, "drift": dr, "truth": {}, "CHANNELS": CH,
          "WINDOWS": {"first_year": ("a", "b"), "last_quarter": ("c", "d")}}
    ns["workshop"] = Workshop("python/17-mmm-time-varying", "-", "-", "-",
                              {3: ("drift_ratio",)}, ns)
    with pytest.raises(NotImplementedError, match="TODO 3"):
        exec(compile(cell, str(lab), "exec"), ns)  # noqa: S102
    out = capsys.readouterr().out
    assert "Checkpoint drifting model uses TODO 3, which is not written yet" in out, out
    assert "workshop.use_reference(3)" in out, out
