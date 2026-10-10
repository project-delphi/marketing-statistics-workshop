"""Loaders: cache, sha256 check, fallbacks, offline mode, committed files, RFM helpers and the
MKTSTATS_SABOTAGE hook. No test touches the network: sources are pointed at local file:// URLs."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from mktstats import data, synth


@pytest.fixture
def cache(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    monkeypatch.setenv("MKTSTATS_CACHE", str(d))
    monkeypatch.delenv("MKTSTATS_OFFLINE", raising=False)
    monkeypatch.delenv("MKTSTATS_SABOTAGE", raising=False)
    return d


@pytest.fixture
def fake_cdnow(tmp_path, monkeypatch):
    """A tiny CDNOW-format file served from a file:// URL, registered as the 'cdnow' dataset."""
    src = tmp_path / "src"
    src.mkdir()
    f = src / "cdnow_transactions.csv"
    f.write_text("_id,id,date,cds_bought,spent\n"
                 "4,1,19970101,2,29.33\n4,1,19970118,2,29.73\n4,1,19970802,1,14.96\n"
                 "18,2,19970101,1,11.77\n21,3,19970101,2,26.48\n21,3,19970113,3,40.00\n"
                 # customer 4 buys on Monday 6 and Wednesday 8 January 1997: one calendar week
                 "30,4,19970106,1,10.00\n30,4,19970108,1,12.00\n")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    entry = {"description": "fake", "sources": [
        {"url": (src / "missing.csv").as_uri(), "file": "cdnow_transactions.csv", "sha256": sha},
        {"url": f.as_uri(), "file": "cdnow_transactions.csv", "sha256": sha},
    ]}
    monkeypatch.setitem(data.DATASETS, "cdnow", entry)
    return f, sha


def test_fetch_falls_back_verifies_and_caches(cache, fake_cdnow, monkeypatch):
    f, sha = fake_cdnow
    path = data.fetch("cdnow", retries=1)
    assert path.parent == cache and data.sha256_file(path) == sha
    # second call is served from the cache, even offline
    monkeypatch.setenv("MKTSTATS_OFFLINE", "1")
    assert data.fetch("cdnow") == path


def test_fetch_rejects_a_bad_checksum(cache, fake_cdnow, monkeypatch):
    f, _ = fake_cdnow
    bad = {"description": "fake", "sources": [
        {"url": f.as_uri(), "file": "x.csv", "sha256": "0" * 64}]}
    monkeypatch.setitem(data.DATASETS, "bad", bad)
    with pytest.raises(OSError, match="sha256 mismatch"):
        data.fetch("bad", retries=1)
    assert not (cache / "x.csv").exists()


def test_offline_without_cache_says_what_to_download(cache, fake_cdnow, monkeypatch):
    monkeypatch.setenv("MKTSTATS_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        data.fetch("cdnow")


def test_unknown_dataset():
    with pytest.raises(KeyError):
        data.fetch("nope")


def test_load_cdnow_and_rfm(cache, fake_cdnow):
    tx = data.load_cdnow()
    assert pd.api.types.is_datetime64_any_dtype(tx["date"])
    rfm = data.cdnow_rfm(time_unit="D", time_scaler=1).set_index("customer_id")
    # customer 1: purchases on day 0, 17, 213 of 1997; observation ends at the last date (213)
    assert rfm.loc[1, "frequency"] == 2
    assert rfm.loc[1, "recency"] == 213 and rfm.loc[1, "T"] == 213
    assert rfm.loc[1, "monetary_value"] == pytest.approx((29.73 + 14.96) / 2)
    assert rfm.loc[2, "frequency"] == 0 and rfm.loc[2, "monetary_value"] == 0


def test_cdnow_rfm_defaults_to_days_over_seven(cache, fake_cdnow):
    """The default counts purchase days and reports weeks as days / 7, like
    rfm_summary(time_unit="D", time_scaler=7); weekly periods merge same-week purchases."""
    rfm = data.cdnow_rfm().set_index("customer_id")
    assert rfm.loc[1, "frequency"] == 2
    assert rfm.loc[1, "recency"] == pytest.approx(213 / 7)
    assert rfm.loc[1, "T"] == pytest.approx(213 / 7)
    # customer 4: Monday and Wednesday of one week are two purchase days, 2 days apart
    assert rfm.loc[4, "frequency"] == 1
    assert rfm.loc[4, "recency"] == pytest.approx(2 / 7)
    weekly = data.cdnow_rfm(time_unit="W").set_index("customer_id")
    assert weekly.loc[4, "frequency"] == 0 and weekly.loc[4, "recency"] == 0
    pd.testing.assert_frame_equal(data.cdnow_rfm(), data.cdnow_rfm("D", None, 7))


def test_sabotage_swaps_recency_and_frequency(cache, fake_cdnow, monkeypatch):
    clean = data.cdnow_rfm(time_unit="D", time_scaler=1)
    monkeypatch.setenv("MKTSTATS_SABOTAGE", "swap_rf")
    bad = data.cdnow_rfm(time_unit="D", time_scaler=1)
    np.testing.assert_array_equal(bad["recency"], clean["frequency"])
    np.testing.assert_array_equal(bad["frequency"], clean["recency"])
    monkeypatch.setenv("MKTSTATS_SABOTAGE", "something_else")
    pd.testing.assert_frame_equal(data.cdnow_rfm(time_unit="D", time_scaler=1), clean)


@pytest.mark.parametrize("unit,scaler", [("D", 7), ("W", 1)])
def test_rfm_summary_matches_pymc_marketing(unit, scaler, monkeypatch):
    monkeypatch.delenv("MKTSTATS_SABOTAGE", raising=False)
    clv_utils = pytest.importorskip("pymc_marketing.clv.utils")
    res = synth.retailer(seed=5, n_customers=600)
    end = res.truth["calibration_end"]
    ours = data.rfm_summary(res.transactions, "customer_id", "date", "amount",
                            observation_period_end=end, time_unit=unit, time_scaler=scaler)
    ref = clv_utils.rfm_summary(res.transactions, "customer_id", "date", "amount",
                                observation_period_end=end, time_unit=unit, time_scaler=scaler)
    ref = ref.sort_values("customer_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(ours, ref, check_dtype=False)


def test_committed_real_files():
    p = data.load_prop99()
    assert p.shape == (1209, 7) and p["state"].nunique() == 39
    assert p["year"].min() == 1970 and p["year"].max() == 2000
    ca = p.set_index(["state", "year"]).loc[("California", 1988), "cigsale"]
    assert ca == pytest.approx(90.1)
    s = data.load_sbg_retention()
    assert list(s.columns) == ["year", "regular", "highend"] and len(s) == 13
    assert s.loc[0, "regular"] == 1000 and s.loc[7, "regular"] == 241
    assert s.loc[12, "highend"] == 394
    assert (s[["regular", "highend"]].diff().dropna() < 0).all().all()


def test_repo_file_downloads_at_mktstats_ref(cache, monkeypatch):
    # On Colab there is no checkout: committed files come from raw GitHub at the install
    # cell's ref, so a notebook pinned to a tag never reads data/ from main.
    monkeypatch.setattr(data, "_local_data_dir", lambda: None)
    monkeypatch.setenv("MKTSTATS_REF", "v9.9.9")
    urls = []

    def fake_download(url, dest, sha256, timeout, retries):
        urls.append(url)
        dest.write_text("{}")

    monkeypatch.setattr(data, "_download", fake_download)
    path = data.repo_file("synthetic/truth.json")
    assert urls == [f"{data.REPO_RAW}/v9.9.9/data/synthetic/truth.json"]
    assert path == cache / "repo" / "v9.9.9" / "synthetic" / "truth.json"


def test_committed_synthetic_files_load():
    truth = data.load_truth()
    for key in ["retailer", "btyd_bgnbd", "mmm", "geo_panel", "email_experiment", "tolerances"]:
        assert key in truth
    for name in data.SYNTHETIC_TABLES:
        df = data.load_synthetic(name)
        assert len(df) > 0, name
    tx = data.load_synthetic("retailer_transactions")
    assert pd.api.types.is_datetime64_any_dtype(tx["date"])


def test_retailer_rfm_from_committed_files(monkeypatch):
    monkeypatch.delenv("MKTSTATS_SABOTAGE", raising=False)
    rfm = data.retailer_rfm()
    for col in ["customer_id", "frequency", "recency", "T", "monetary_value", "test_frequency",
                "test_T", "acquisition_channel", "channel_social", "channel_referral"]:
        assert col in rfm.columns
    assert (rfm["recency"] <= rfm["T"]).all() and (rfm["frequency"] % 1 == 0).all()
    monkeypatch.setenv("MKTSTATS_SABOTAGE", "swap_rf")
    assert not (data.retailer_rfm()["frequency"] % 1 == 0).all()


def test_online_retail_sampling_uses_cached_parquet(cache, monkeypatch):
    tag = data.DATASETS["online_retail_ii"]["sources"][0]["sha256"][:12]
    cache.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"invoice": ["1", "1", "2", "3", "4"], "customer_id": [10, 10, 11, None, 12],
                       "quantity": [1, 2, 3, 4, 5]})
    df["customer_id"] = df["customer_id"].astype("Int64")
    df.to_parquet(cache / f"online_retail_ii-{tag}.parquet", index=False)
    monkeypatch.setattr(data, "fetch", lambda name, **kw: cache / "unused.zip")
    assert len(data.load_online_retail_ii()) == 5
    two = data.load_online_retail_ii(sample=2, seed=0)
    assert two["customer_id"].nunique() == 2 and two["customer_id"].notna().all()
    assert data.load_online_retail_ii(sample=2, seed=0).equals(two)
    with pytest.raises(ValueError):
        data.load_online_retail_ii(sample=1.5)
