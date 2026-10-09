# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Spike S1/S2 · Colab Python: install time and model timings
#
# Throwaway notebook (see DECISIONS.md). It measures, on a fresh Colab runtime: the install of the
# pinned stack, then BG/NBD, Pareto/NBD, Gamma-Gamma, a small MMM and a causal forest. The last
# cell prints a JSON record between markers and also offers it as a download.

# %%
import json, os, platform, sys, time

T0 = time.time()
REC = {
    "spike": "s2-colab-python",
    "colab_release": os.environ.get("COLAB_RELEASE_TAG"),
    "python": platform.python_version(),
    "cpus": os.cpu_count(),
    "preloaded_pymc": "pymc" in sys.modules,
    "timings": {},
    "errors": {},
}
try:
    REC["mem_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
except Exception:
    pass


def timed(name):
    class _T:
        def __enter__(self):
            self.t = time.time()

        def __exit__(self, et, ev, tb):
            REC["timings"][name] = round(time.time() - self.t, 1)
            if et is not None:
                REC["errors"][name] = f"{et.__name__}: {ev}"[:300]
                print(f"[{name}] FAILED: {et.__name__}: {ev}")
                return True  # keep going: a spike records failures
            print(f"[{name}] {REC['timings'][name]} s")

    return _T()


print(REC)

# %%
CONSTRAINTS = (
    "https://raw.githubusercontent.com/project-delphi/marketing-statistics-workshop/main/"
    "environment/requirements.txt"
)
with timed("install"):
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pymc-marketing==1.2.0", "econml==0.17.0",
         "nutpie", "-c", CONSTRAINTS],
        capture_output=True, text=True,
    )
    print(out.stdout[-2000:], out.stderr[-2000:])
    if out.returncode:
        raise RuntimeError("pip failed")

# %%
import importlib.metadata as md

REC["versions"] = {
    p: md.version(p)
    for p in ["pymc-marketing", "pymc", "pytensor", "arviz", "numpy", "pandas", "scipy",
              "scikit-learn", "numba", "econml", "nutpie"]
}
print(REC["versions"])

# %%
import numpy as np
import pandas as pd

with timed("import_pymc_marketing"):
    from pymc_marketing import clv

URL = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/cdnow_transactions.csv"
tx = pd.read_csv(URL)
rfm = clv.utils.rfm_summary(tx, customer_id_col="id", datetime_col="date",
                            monetary_value_col="spent", datetime_format="%Y%m%d", time_unit="W")
print(rfm.describe().T)

# %%
with timed("bgnbd_map"):
    bg = clv.BetaGeoModel()
    bg.fit(data=rfm, method="map")
    print(bg.fit_summary())

with timed("bgnbd_mcmc_default_2x500"):
    bg2 = clv.BetaGeoModel()
    bg2.fit(data=rfm, chains=2, draws=500, tune=500, random_seed=1, progressbar=False)

with timed("bgnbd_mcmc_nutpie_2x500"):
    bg3 = clv.BetaGeoModel()
    bg3.fit(data=rfm, chains=2, draws=500, tune=500, random_seed=1, progressbar=False,
            nuts_sampler="nutpie")

# %%
with timed("paretonbd_map"):
    pn = clv.ParetoNBDModel()
    pn.fit(data=rfm, method="map")
    print(pn.fit_summary())

with timed("paretonbd_mcmc_2x300"):
    pn2 = clv.ParetoNBDModel()
    pn2.fit(data=rfm, method="mcmc", chains=2, draws=300, tune=300, random_seed=1,
            progressbar=False)

# %%
with timed("gammagamma_mcmc_2x500"):
    gg = clv.GammaGammaModel()
    gg.fit(data=rfm.query("frequency > 0"), chains=2, draws=500, tune=500, random_seed=1,
           progressbar=False)
with timed("clv_expected"):
    v = gg.expected_customer_lifetime_value(transaction_model=bg2, data=rfm, future_t=12,
                                            discount_rate=0.01, time_unit="W")
    print(float(v.mean()))

# %%
with timed("mmm_fit_2x500"):
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    df = pd.read_csv("https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_example.csv",
                     parse_dates=["date_week"])
    X = df.drop(columns=["y"])
    y = df["y"]
    mmm = MMM(date_column="date_week", channel_columns=["x1", "x2"],
              control_columns=["event_1", "event_2", "t"], adstock=GeometricAdstock(l_max=8),
              saturation=LogisticSaturation(), yearly_seasonality=2)
    mmm.fit(X, y, chains=2, draws=500, tune=500, random_seed=1, progressbar=False)

with timed("mmm_fit_nutpie_2x500"):
    mmm2 = MMM(date_column="date_week", channel_columns=["x1", "x2"],
               control_columns=["event_1", "event_2", "t"], adstock=GeometricAdstock(l_max=8),
               saturation=LogisticSaturation(), yearly_seasonality=2)
    mmm2.fit(X, y, chains=2, draws=500, tune=500, random_seed=1, progressbar=False,
             nuts_sampler="nutpie")

# %%
with timed("causal_forest_20k"):
    from econml.dml import CausalForestDML
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    n = 20_000
    Xc = rng.normal(size=(n, 5))
    T = rng.integers(0, 2, n)
    Y = Xc[:, 0] + T * (0.5 + Xc[:, 1]) + rng.normal(size=n)
    cf = CausalForestDML(model_y=HistGradientBoostingRegressor(),
                         model_t=HistGradientBoostingClassifier(), discrete_treatment=True,
                         n_estimators=200, random_state=0)
    cf.fit(Y, T, X=Xc)
    print("ATE", cf.ate(Xc))

# %%
REC["total_seconds"] = round(time.time() - T0, 1)
text = json.dumps(REC)
print("----- spike record -----")
print(text)
print("----- end -----")
with open("spike_record.json", "w") as f:
    f.write(text)
try:
    from google.colab import files

    files.download("spike_record.json")
except Exception as e:
    print("no download:", e)
