"""Cached, checksummed loaders for the workshop's real datasets and committed synthetic data.

Downloads land in ``$MKTSTATS_CACHE`` (default ``~/.cache/mktstats``) and are verified with sha256
before use; each dataset lists fallback URLs that are tried in order. Once a file is cached no
network is needed. ``MKTSTATS_OFFLINE=1`` forbids downloads (a missing file then raises with the URL
to fetch by hand and the path to put it at).

Files committed to the repository (``data/synthetic/*``, ``data/real/*``) are read from the local
checkout when there is one (``$MKTSTATS_DATA_DIR`` or the ``data/`` folder next to ``src/``), and
otherwise from raw GitHub at ``$MKTSTATS_REF`` (default ``main``), cached like any download.

Sabotage hook (CI): with ``MKTSTATS_SABOTAGE=swap_rf`` the RFM helpers (:func:`rfm_summary`,
:func:`cdnow_rfm`, :func:`retailer_rfm`) return recency and frequency swapped, so a checkpoint
that validates the RFM table must fail. Any other value leaves the data alone.

Sources, licences and checksums are listed in ``data/README.md``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from mktstats.synth._rfm import rfm_core, swap_recency_frequency

REPO_RAW = "https://raw.githubusercontent.com/project-delphi/marketing-statistics-workshop"
_PMM = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing"
_MIRROR = "https://github.com/project-delphi/marketing-statistics-workshop/releases/download/data-2026-10-09"

DATASETS: dict[str, dict] = {
    "cdnow": {
        "description": "CDNOW sample: purchases of 2,357 customers, Jan 1997 - Jun 1998",
        "sources": [
            {"url": f"{_PMM}/1.2.0/data/cdnow_transactions.csv", "file": "cdnow_transactions.csv",
             "sha256": "7311ae0fb676e9f454bba310b10bc769b1a6838aeec09bd00175eb49481dddac"},
            {"url": f"{_PMM}/main/data/cdnow_transactions.csv", "file": "cdnow_transactions.csv",
             "sha256": "7311ae0fb676e9f454bba310b10bc769b1a6838aeec09bd00175eb49481dddac"},
        ],
    },
    "mmm_example": {
        "description": "pymc-marketing's simulated weekly MMM example (two channels)",
        "sources": [
            {"url": f"{_PMM}/1.2.0/data/mmm_example.csv", "file": "mmm_example.csv",
             "sha256": "320d299569b989c37facac5bda6a4ee2d661cb12e158ac5d80689c8cc3d37955"},
            {"url": f"{_PMM}/main/data/mmm_example.csv", "file": "mmm_example.csv",
             "sha256": "320d299569b989c37facac5bda6a4ee2d661cb12e158ac5d80689c8cc3d37955"},
        ],
    },
    "online_retail_ii": {
        "description": "UCI Online Retail II (dataset 502), Dec 2009 - Dec 2011, CC BY 4.0",
        "sources": [
            {"url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
             "file": "online_retail_ii.zip",
             "sha256": "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"},
            # The same archive, unchanged, mirrored as a release asset of this repository
            # (CC BY 4.0 allows redistribution with attribution; see the release notes).
            {"url": f"{_MIRROR}/online_retail_ii.zip",
             "file": "online_retail_ii.zip",
             "sha256": "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"},
        ],
    },
    "online_retail_ii_parquet": {
        "description": "Online Retail II as load_online_retail_ii() returns it (both sheets, "
                       "overlap removed, snake_case), mirrored as parquet; CC BY 4.0, from UCI 502",
        "sources": [
            {"url": f"{_MIRROR}/online_retail_ii.parquet",
             "file": "online_retail_ii.parquet",
             "sha256": "ae7d5f4552906134bdf016819aaa3bb93d1a626c6f3c56ea99d55a74210437b3"},
        ],
    },
    "hillstrom": {
        "description": "MineThatData e-mail analytics challenge (Hillstrom 2008), 64,000 rows",
        "sources": [
            {"url": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_"
                    "DataMiningChallenge_2008.03.20.csv",
             "file": "hillstrom.csv",
             "sha256": "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"},
            {"url": "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
             "file": "hillstrom.csv.gz",
             "sha256": "bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f"},
        ],
    },
}

SABOTAGE_ENV = "MKTSTATS_SABOTAGE"


# ---------------------------------------------------------------------------------------------
# cache and download
# ---------------------------------------------------------------------------------------------
def cache_dir() -> Path:
    """The download cache: ``$MKTSTATS_CACHE`` or ``~/.cache/mktstats`` (created if needed)."""
    d = Path(os.environ.get("MKTSTATS_CACHE") or Path.home() / ".cache" / "mktstats")
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _offline() -> bool:
    return os.environ.get("MKTSTATS_OFFLINE", "").strip() not in ("", "0", "false", "False")


def _download(url: str, dest: Path, sha256: str | None, timeout: float, retries: int) -> None:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mktstats (workshop loader)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            got = hashlib.sha256(data).hexdigest()
            if sha256 is not None and got != sha256:
                raise OSError(f"sha256 mismatch for {url}: expected {sha256}, got {got}")
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - retry any network or checksum failure
            last = e
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise OSError(f"could not download {url}: {last}")


def fetch(name: str, *, timeout: float = 60, retries: int = 3) -> Path:
    """Return the path of a verified local copy of dataset ``name`` (see ``DATASETS``).

    Tries the cache first, then each source URL in order.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    sources = DATASETS[name]["sources"]
    cdir = cache_dir()
    for src in sources:
        path = cdir / src["file"]
        if path.exists() and sha256_file(path) == src["sha256"]:
            return path
    if _offline():
        first = sources[0]
        raise FileNotFoundError(
            f"{name} is not in the cache and MKTSTATS_OFFLINE is set. Download {first['url']} "
            f"and save it as {cdir / first['file']} (sha256 {first['sha256']})."
        )
    errors = []
    for src in sources:
        path = cdir / src["file"]
        try:
            _download(src["url"], path, src["sha256"], timeout, retries)
            return path
        except OSError as e:
            errors.append(str(e))
    raise OSError(f"all sources failed for {name}:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------------------------
# committed repository files (synthetic data, small real data)
# ---------------------------------------------------------------------------------------------
def _local_data_dir() -> Path | None:
    env = os.environ.get("MKTSTATS_DATA_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "data"
        if (cand / "synthetic").is_dir() or (cand / "real").is_dir():
            return cand
    return None


def repo_file(relpath: str, *, timeout: float = 60, retries: int = 3) -> Path:
    """Local path of a committed file under ``data/`` (e.g. ``"synthetic/truth.json"``).

    Uses the checkout when there is one; otherwise downloads it from raw GitHub at
    ``$MKTSTATS_REF`` (default ``main``) into the cache.
    """
    local = _local_data_dir()
    if local is not None and (local / relpath).exists():
        return local / relpath
    ref = os.environ.get("MKTSTATS_REF", "main")
    dest = cache_dir() / "repo" / ref / relpath
    if dest.exists():
        return dest
    if _offline():
        raise FileNotFoundError(f"data/{relpath} not found locally and MKTSTATS_OFFLINE is set")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download(f"{REPO_RAW}/{ref}/data/{relpath}", dest, None, timeout, retries)
    return dest


def load_truth() -> dict:
    """``data/synthetic/truth.json``: true parameters of every committed synthetic dataset."""
    return json.loads(repo_file("synthetic/truth.json").read_text())


SYNTHETIC_TABLES = {
    "retailer_transactions": ["date"],
    "retailer_customers": ["first_date"],
    "bgnbd_transactions": ["date"],
    "bgnbd_rfm": [],
    "mmm_weekly": ["date_week"],
    "mmm_lift_tests": [],
    "geo_panel": ["date"],
    "email_experiment": [],
    # Module 9: confounded MMM (the latent file is truth only; never give it to a model)
    "mmm_confounded_weekly": ["date_week"],
    "mmm_confounded_lift_tests": ["date", "test_start"],
    "mmm_confounded_latent": ["date_week"],
    # Module 12: the capstone scenario
    "capstone_transactions": ["date"],
    "capstone_customers": ["first_date"],
    "capstone_geo_panel": ["date"],
    "capstone_mmm_weekly": ["date_week"],
    "capstone_email_experiment": [],
}


def load_synthetic(name: str) -> pd.DataFrame:
    """A committed synthetic table, e.g. ``load_synthetic("retailer_transactions")``."""
    if name not in SYNTHETIC_TABLES:
        raise KeyError(f"unknown synthetic table {name!r}; known: {sorted(SYNTHETIC_TABLES)}")
    return pd.read_csv(repo_file(f"synthetic/{name}.csv"), parse_dates=SYNTHETIC_TABLES[name])


# ---------------------------------------------------------------------------------------------
# RFM helpers (with the sabotage hook)
# ---------------------------------------------------------------------------------------------
def _sabotage(rfm: pd.DataFrame) -> pd.DataFrame:
    if os.environ.get(SABOTAGE_ENV, "").strip() == "swap_rf":
        return swap_recency_frequency(rfm)
    return rfm


def rfm_summary(
    transactions: pd.DataFrame,
    customer_id_col: str = "customer_id",
    datetime_col: str = "date",
    monetary_value_col: str | None = None,
    observation_period_end=None,
    time_unit: str = "D",
    time_scaler: float = 1,
) -> pd.DataFrame:
    """RFM table with pymc-marketing 1.2.0 ``rfm_summary`` conventions (repeat-purchase frequency,
    recency = last minus first purchase period, T = end minus first period, monetary value = mean of
    repeat purchases). Honors ``MKTSTATS_SABOTAGE=swap_rf``.
    """
    rfm = rfm_core(transactions, customer_id_col, datetime_col, monetary_value_col,
                   observation_period_end, time_unit, time_scaler)
    return _sabotage(rfm)


def cdnow_rfm(time_unit: str = "D", observation_period_end: str | None = None,
              time_scaler: float | None = None) -> pd.DataFrame:
    """CDNOW RFM table (frequency, recency, T, monetary_value).

    The default counts purchase **days** and reports recency and T in weeks as days / 7, the
    convention of the workshop's BTYD labs (``rfm_summary(..., time_unit="D", time_scaler=7)``):
    ``time_scaler`` defaults to 7 for ``time_unit="D"`` and to 1 otherwise. Weekly periods
    (``time_unit="W"``) are not the same: they merge purchases made in the same calendar week and
    floor the times (1,422 customers without a repeat purchase at 1997-09-30 instead of the
    1,411 that Fader, Hardie & Lee report; measured 2026-10-09).

    ``observation_period_end`` defaults to the last transaction date (1998-06-30).
    """
    if time_scaler is None:
        time_scaler = 7 if time_unit == "D" else 1
    tx = load_cdnow()
    rfm = rfm_core(tx, "id", "date", "spent", observation_period_end, time_unit, time_scaler)
    return _sabotage(rfm)


def retailer_rfm(transactions: pd.DataFrame | None = None,
                 customers: pd.DataFrame | None = None,
                 truth: dict | None = None) -> pd.DataFrame:
    """Calibration-period RFM of the synthetic retailer, in weeks, with holdout counts.

    Columns: customer_id, frequency, recency, T, monetary_value (calibration, daily resolution,
    ``time_scaler=7``), test_frequency (repeat purchase days in the holdout), test_T (weeks),
    acquisition_channel and the ``channel_<name>`` indicators. Defaults to the committed files.
    """
    if transactions is None:
        transactions = load_synthetic("retailer_transactions")
    if customers is None:
        customers = load_synthetic("retailer_customers")
    if truth is None:
        truth = load_truth()["retailer"]
    tx = transactions.assign(date=pd.to_datetime(transactions["date"]))
    cal_end = pd.Timestamp(truth["calibration_end"])
    rfm = rfm_core(tx, "customer_id", "date", "amount", cal_end, "D", 7)
    hold = tx.loc[tx["date"] > cal_end, ["customer_id", "date"]].drop_duplicates()
    rfm["test_frequency"] = (rfm["customer_id"].map(hold.groupby("customer_id").size())
                             .fillna(0).astype(float))
    rfm["test_T"] = float(truth["holdout_weeks"])
    keep = ["customer_id", "acquisition_channel"] + [c for c in customers.columns
                                                     if c.startswith("channel_")]
    rfm = rfm.merge(customers[keep], on="customer_id", how="left")
    return _sabotage(rfm)


# ---------------------------------------------------------------------------------------------
# real datasets
# ---------------------------------------------------------------------------------------------
def load_cdnow() -> pd.DataFrame:
    """CDNOW transactions as distributed by pymc-marketing: id, date, cds_bought, spent
    (plus the original ``_id`` column). ``date`` is parsed to datetime."""
    df = pd.read_csv(fetch("cdnow"))
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df


def load_mmm_example() -> pd.DataFrame:
    """pymc-marketing's simulated MMM example: date_week, y, x1, x2, event_1, event_2,
    dayofyear, t. Simulated data, not a real advertiser."""
    return pd.read_csv(fetch("mmm_example"), parse_dates=["date_week"])


ONLINE_RETAIL_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_online_retail_ii(sample: int | float | None = None, seed: int = 0) -> pd.DataFrame:
    """UCI Online Retail II, both sheets stacked, snake_case columns (invoice, stock_code,
    description, quantity, invoice_date, price, customer_id, country).

    The two sheets of the workbook overlap on 1-9 Dec 2010; invoices of the second sheet that are
    already in the first are dropped. Duplicate lines inside a sheet are kept as published.
    Cancellations (invoice starting with "C") and negative quantities are kept too.

    The 45 MB xlsx is parsed once with openpyxl (minutes) and cached as parquet. ``sample``: an
    int keeps that many randomly chosen customers, a float in (0, 1) that fraction of customers
    (rows without a customer id are dropped when sampling); ``seed`` fixes the choice.
    """
    tag = DATASETS["online_retail_ii"]["sources"][0]["sha256"][:12]
    cached = cache_dir() / f"online_retail_ii-{tag}.parquet"
    if not cached.exists():
        # Fast path: the parsed table, mirrored (7 MB, sha256-checked), instead of downloading the
        # 45 MB workbook and parsing it with openpyxl. Falls back to the original on any failure.
        try:
            fast = fetch("online_retail_ii_parquet", timeout=30, retries=2)
            shutil.copyfile(fast, cached)
        except OSError:
            pass
    if cached.exists():
        df = pd.read_parquet(cached)
    else:
        zpath = fetch("online_retail_ii")
        with zipfile.ZipFile(zpath) as z, z.open("online_retail_II.xlsx") as f:
            sheets = pd.read_excel(io.BytesIO(f.read()), sheet_name=None, engine="openpyxl")
        first, second = (s.assign(Invoice=s["Invoice"].astype(str)) for s in sheets.values())
        # The sheets overlap on 1-9 Dec 2010 (1,088 invoices appear in both): keep one copy.
        second = second.loc[~second["Invoice"].isin(set(first["Invoice"]))]
        df = pd.concat([first, second], ignore_index=True).rename(columns=ONLINE_RETAIL_COLUMNS)
        df["stock_code"] = df["stock_code"].astype(str)
        df["description"] = df["description"].astype("string")
        df["customer_id"] = df["customer_id"].astype("Int64")
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        tmp = cached.with_name(cached.name + ".part")
        df.to_parquet(tmp, index=False)
        tmp.replace(cached)
    if sample is not None:
        ids = pd.Series(df["customer_id"].dropna().unique())
        if isinstance(sample, float):
            if not 0 < sample < 1:
                raise ValueError("a float sample is a fraction in (0, 1)")
            k = max(1, round(sample * len(ids)))
        else:
            k = min(int(sample), len(ids))
        chosen = ids.sample(n=k, random_state=seed)
        df = df.loc[df["customer_id"].isin(chosen)].reset_index(drop=True)
    return df


def load_hillstrom() -> pd.DataFrame:
    """MineThatData e-mail challenge (Hillstrom 2008): recency, history_segment, history, mens,
    womens, zip_code, newbie, channel, segment, visit, conversion, spend (64,000 rows)."""
    return pd.read_csv(fetch("hillstrom"))


def load_prop99() -> pd.DataFrame:
    """California Proposition 99 panel (Abadie, Diamond & Hainmueller 2010), 39 states,
    1970-2000: state, year, cigsale (per-capita cigarette sales, packs), lnincome, beer,
    age15to24, retprice. Committed copy of tidysynth 0.2.1's ``smoking`` data (MIT)."""
    return pd.read_csv(repo_file("real/prop99.csv"))


def load_sbg_retention() -> pd.DataFrame:
    """Fader & Hardie (2007) survivor counts per 1,000 customers for the 'regular' and 'highend'
    segments, years 0-12 (columns: year, regular, highend)."""
    return pd.read_csv(repo_file("real/sbg_retention.csv"))


__all__ = [
    "DATASETS",
    "cache_dir",
    "cdnow_rfm",
    "fetch",
    "load_cdnow",
    "load_hillstrom",
    "load_mmm_example",
    "load_online_retail_ii",
    "load_prop99",
    "load_sbg_retention",
    "load_synthetic",
    "load_truth",
    "repo_file",
    "retailer_rfm",
    "rfm_summary",
    "sha256_file",
]
