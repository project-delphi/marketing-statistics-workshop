"""mktstats.mle (Module 13's maximum-likelihood helper) against the published CDNOW estimates,
PyMC-Marketing's flat-prior MAP and PyMC-Marketing's own likelihood and prediction code.

Published values: Fader, Hardie & Lee, note 004 (BG/NBD on CDNOW: r 0.243, alpha 4.414, a 0.793,
b 2.426) and Fader & Hardie, note 025 (Gamma-Gamma on CDNOW's 946 repeat buyers: p 6.25, q 3.74,
v 15.44), as cited in references.qmd. The notes print three and two decimals, so "reproduces"
means equal at that precision; against the four-digit values Modules 2 and 4 printed from
PyMC-Marketing's flat-prior MAP in their FULL runs (modules/02-btyd.qmd and
modules/04-monetary-clv.qmd) the helper must agree within 0.1%.
"""

import numpy as np
import pandas as pd
import pytest

from mktstats import data, mle
from mktstats.synth import messy

NOTE_004 = {"r": 0.243, "alpha": 4.414, "a": 0.793, "b": 2.426}
NOTE_025 = {"p": 6.25, "q": 3.74, "v": 15.44}
MODULE_2_FULL = {"r": 0.2426, "alpha": 4.4136, "a": 0.793, "b": 2.4262}
MODULE_4_FULL = {"p": 6.250, "q": 3.744, "v": 15.444}


@pytest.fixture(scope="module")
def cdnow():
    return data.cdnow_rfm(observation_period_end="1997-09-30")


@pytest.fixture(scope="module")
def messy_reference():
    """The messy log's reference-pipeline calibration RFM (what the lab's Part B fits)."""
    res = messy.messy_retailer()
    lines = res.lines.assign(invoice_date=pd.to_datetime(res.lines["invoice_date"]))
    tx = messy.reference_purchases(lines, messy.reference_flags(lines))
    return data.rfm_summary(tx, "customer_id", "date", "amount",
                            observation_period_end=res.truth["calibration_end"],
                            time_unit="D", time_scaler=7)


def assert_reproduces(estimate, published, decimals, printed):
    for k, value in published.items():
        assert abs(estimate[k] - value) <= 0.5 * 10.0**-decimals + 1e-12, (k, estimate[k], value)
        assert abs(estimate[k] / printed[k] - 1) <= 1e-3, (k, estimate[k], printed[k])


def test_bgnbd_reproduces_note_004(cdnow):
    assert_reproduces(mle.bgnbd_mle(cdnow), NOTE_004, 3, MODULE_2_FULL)


def test_gamma_gamma_reproduces_note_025(cdnow):
    est = mle.gamma_gamma_mle(cdnow[cdnow["frequency"] > 0])
    assert_reproduces(est, NOTE_025, 2, MODULE_4_FULL)
    assert int((cdnow["frequency"] > 0).sum()) == 946


@pytest.mark.parametrize("start, published, decimals, printed", [
    (mle.BGNBD_START, NOTE_004, 3, MODULE_2_FULL),
    (mle.GAMMA_GAMMA_START, NOTE_025, 2, MODULE_4_FULL),
])
def test_a_stub_that_returns_the_start_values_fails(start, published, decimals, printed):
    with pytest.raises(AssertionError):
        assert_reproduces(start, published, decimals, printed)


def test_the_optimum_does_not_depend_on_the_start(cdnow):
    a = mle.bgnbd_mle(cdnow)
    b = mle.bgnbd_mle(cdnow, start={"r": 0.05, "alpha": 50.0, "a": 3.0, "b": 0.5})
    assert max(abs(a[k] / b[k] - 1) for k in a) < 1e-9
    rep = cdnow[cdnow["frequency"] > 0]
    c = mle.gamma_gamma_mle(rep)
    d = mle.gamma_gamma_mle(rep, start={"p": 20.0, "q": 1.5, "v": 500.0})
    assert max(abs(c[k] / d[k] - 1) for k in c) < 1e-9


def test_gamma_gamma_asks_for_repeat_buyers(cdnow):
    with pytest.raises(ValueError, match=r"rfm\[rfm\['frequency'\] > 0\]"):
        mle.gamma_gamma_mle(cdnow)


def test_a_likelihood_without_a_maximum_raises():
    # Nobody ever repeats: the BG/NBD likelihood grows as the purchase rate goes to 0.
    rfm = pd.DataFrame({"frequency": 0.0, "recency": 0.0, "T": np.linspace(5, 40, 200)})
    with pytest.raises(mle.MLEError):
        mle.bgnbd_mle(rfm)


def test_loglik_equals_pymc_marketing_logp(messy_reference):
    import pymc as pm
    from pymc_extras.prior import Prior
    from pymc_marketing import clv
    from pymc_marketing.clv.distributions import BetaGeoNBD

    rfm = messy_reference
    bg = {"r": 0.4, "alpha": 2.5, "a": 0.6, "b": 2.0}
    # np.float64: PyTensor makes a plain Python float that float32 holds exactly (2.0, 2.5) a
    # float32 constant, and gammaln(b) would then be computed in single precision
    dist = BetaGeoNBD.dist(**{k: np.float64(v) for k, v in bg.items()}, T=rfm["T"].to_numpy())
    value = np.column_stack([rfm["recency"], rfm["frequency"]])
    theirs = pm.logp(dist, value).eval()
    np.testing.assert_allclose(mle.bgnbd_loglik(bg, rfm), theirs, rtol=1e-10, atol=1e-10)

    rep = rfm[rfm["frequency"] > 0]
    gg = {"p": 5.0, "q": 3.0, "v": 12.0}
    model = clv.GammaGammaModel(model_config={k: Prior("HalfFlat") for k in gg})
    model.build_model(rep[["customer_id", "frequency", "monetary_value"]])
    logp = model.model.compile_logp(jacobian=False)
    point = {f"{k}_log__": np.log(v) for k, v in gg.items()}
    np.testing.assert_allclose(mle.gamma_gamma_loglik(gg, rep).sum(), logp(point), rtol=1e-10)


def _pmm_map(rfm):
    from pymc_extras.prior import Prior
    from pymc_marketing import clv

    bg = clv.BetaGeoModel(model_config={k: Prior("HalfFlat") for k in mle.BGNBD_PARAMS})
    bg.fit(data=rfm[["customer_id", "frequency", "recency", "T"]], method="map",
           progressbar=False)
    rep = rfm.loc[rfm["frequency"] > 0, ["customer_id", "frequency", "monetary_value"]]
    gg = clv.GammaGammaModel(model_config={k: Prior("HalfFlat") for k in mle.GAMMA_GAMMA_PARAMS})
    gg.fit(data=rep, method="map", progressbar=False)
    values = {k: float(bg.fit_result[k].mean()) for k in mle.BGNBD_PARAMS}
    values.update({k: float(gg.fit_result[k].mean()) for k in mle.GAMMA_GAMMA_PARAMS})
    return bg, gg, values


def _assert_agree(rfm, rel):
    _, _, pmm = _pmm_map(rfm)
    ours = {**mle.bgnbd_mle(rfm), **mle.gamma_gamma_mle(rfm[rfm["frequency"] > 0])}
    worst = max(abs(ours[k] / pmm[k] - 1) for k in ours)
    assert worst <= rel, (worst, ours, pmm)
    return worst


def test_agrees_with_pymc_marketing_map_on_the_messy_reference_rfm(messy_reference):
    _assert_agree(messy_reference, 1e-3)


def test_agrees_with_pymc_marketing_map_on_the_retailer():
    _assert_agree(data.retailer_rfm(), 1e-3)


def test_predictions_equal_pymc_marketing_at_the_same_parameters(messy_reference):
    rfm = messy_reference
    bg_model, gg_model, values = _pmm_map(rfm)
    theirs = bg_model.expected_purchases(data=rfm, future_t=26).mean(("chain", "draw")).values
    ours = mle.bgnbd_expected_purchases(values, rfm, 26)
    np.testing.assert_allclose(ours, theirs, rtol=1e-9)
    rep = rfm[rfm["frequency"] > 0]
    theirs = gg_model.expected_customer_spend(rep).mean(("chain", "draw")).values
    np.testing.assert_allclose(mle.gamma_gamma_expected_spend(values, rep), theirs, rtol=1e-9)
    # frequency 0: the population mean p v / (q - 1)
    first = rfm[rfm["frequency"] == 0].head(1)
    assert mle.gamma_gamma_expected_spend(values, first)[0] == pytest.approx(
        mle.population_mean_spend(values), rel=1e-12)


@pytest.mark.slow
def test_agrees_with_pymc_marketing_map_on_cleaned_online_retail_ii():
    """Downloads Online Retail II (7 MB parquet mirror, sha256-checked) on first use."""
    raw = data.load_online_retail_ii()
    tx = messy.reference_purchases(raw, messy.reference_flags(raw))
    rfm = data.rfm_summary(tx, "customer_id", "date", "amount",
                           observation_period_end="2011-06-30", time_unit="D", time_scaler=7)
    assert len(rfm) == 5_008
    _assert_agree(rfm, 1e-3)
