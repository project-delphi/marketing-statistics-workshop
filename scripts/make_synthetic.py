"""Write the default-seed synthetic datasets to data/synthetic/ (CSV) and their truth to truth.json.

    python scripts/make_synthetic.py            # (re)write the files
    python scripts/make_synthetic.py --check    # exit 1 if regenerating would change any file

The generators live in src/mktstats/synth/; never edit data/synthetic/* by hand. Recovery
tolerances (how close a fit must get to the truth in tests and checkpoints) are kept in
TOLERANCES below and written into truth.json so pages and notebooks can cite them.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mktstats import synth  # noqa: E402

OUT = ROOT / "data" / "synthetic"

# Measured on 2026-10-09 with numpy 2.1.3 and pymc-marketing 1.2.0 (see data/README.md): each
# tolerance is set above the largest error seen across independent seeds of the same generator,
# so a correct fit on the committed data passes and a wrong one (swapped columns, wrong time unit,
# missing covariates) fails.
TOLERANCES = {
    "basis": (
        "MAP fits with pymc-marketing 1.2.0 on default-size data; bounds sit above the largest "
        "error measured over independent seeds (n_seeds per block). rel = |est - truth| / |truth|, "
        "abs = |est - truth|."
    ),
    "btyd_bgnbd_map": {
        "n_seeds": 30, "priors": "pymc-marketing defaults",
        "r": {"rel": 0.15}, "alpha": {"rel": 0.2}, "a": {"rel": 0.3}, "b": {"rel": 0.35},
    },
    "retailer_pnbd_covariates_map": {
        "n_seeds": 30, "priors": "pymc-marketing defaults",
        "covariates": "channel_social and channel_referral on purchase and dropout",
        "r": {"rel": 0.25}, "alpha": {"rel": 0.25}, "s": {"rel": 0.25}, "beta": {"rel": 0.45},
        "gamma_purchase": {"social": {"abs": 0.2}, "referral": {"abs": 0.15}},
        "gamma_dropout": {"social": {"abs": 0.35}, "referral": {"abs": 0.4}},
    },
    "retailer_pnbd_nocovariates_map": {
        "n_seeds": 30, "priors": "pymc-marketing defaults",
        "note": "without covariates the fitted r, alpha, s, beta describe the mixture over "
                "channels; compare the implied population means r/alpha and s/beta with the "
                "sample means of true_lambda and true_mu",
        "mean_purchase_rate": {"rel": 0.1}, "mean_dropout_rate": {"rel": 0.4},
    },
    "retailer_gamma_gamma_map": {
        "n_seeds": 30,
        "priors": "weakly informative: p, q ~ HalfNormal(10), v ~ HalfNormal(100) "
                  "(pymc-marketing's default Weibull priors pull p to about 4.6)",
        "p": {"rel": 0.35}, "q": {"rel": 0.12}, "v": {"rel": 0.5},
        "population_mean_spend": {"rel": 0.07},
    },
    "geo_did": {
        "n_seeds": 41, "estimator": "difference in differences of mean log sales (two-way FE)",
        "lift_pct": {"abs": 2.5},
    },
    "mmm_mcmc": {
        "sampler": "nutpie, 2 chains x 500 draws (500 tune), default priors",
        "interval": "94% HDI",
        "roas_inside_min_channels": 3,
        "tv_adstock_alpha_inside": True,
    },
    "email_uplift": {
        "check": "Qini coefficient of true_cate as the score exceeds that of a random score",
    },
    "mmm_confounded_mcmc": {
        "sampler": "nutpie, 2 chains, draws = tune = 300 (QUICK) and 1000 (FULL), random_seed=1, "
                   "default priors, mktstats.recovery.fit_mmm",
        "interval": "94% HDI",
        "uncalibrated": {"search_hdi_above_truth": True, "other_channels_inside_min": 2},
        "calibrated": {"lift_rows": "both search rows of mmm_confounded_lift_tests.csv",
                       "search_inside": True, "search_hdi_narrower": True},
        "measured_2026_10_09": {
            "search_roas_truth": 3.0,
            "uncalibrated_search_mean_hdi": {"300": [5.418, 4.135, 6.719],
                                             "1000": [5.354, 4.013, 6.530]},
            "calibrated_search_mean_hdi": {"300": [3.055, 2.783, 3.344],
                                           "1000": [3.056, 2.772, 3.368]},
            "other_channels_inside": "3 of 3 in all four fits; calibration raises the social and "
                                     "display means (to about 2.8 and 1.4) with wide HDIs",
            "other_lift_rows_500_draws": {
                "search_plus_30pct_test_only": [3.241, 2.694, 4.025],
                "search_switch_off_test_only": [3.507, 3.128, 3.813],
                "all_five_rows": [3.070, 2.800, 3.383],
                "note": "search ROAS mean and 94% HDI; the switch-off test alone leaves the "
                        "truth outside, the +30% test alone or both search tests bring it "
                        "inside",
            },
        },
    },
}


def build() -> tuple[dict[str, object], dict]:
    """Run every generator with its default seed; return {filename: DataFrame} and the truth."""
    retail = synth.retailer()
    bg = synth.btyd_bgnbd()
    mm = synth.mmm()
    mmc = synth.mmm_confounded()
    geo = synth.geo_panel()
    email = synth.email_experiment()
    frames = {
        "retailer_transactions.csv": retail.transactions,
        "retailer_customers.csv": retail.customers,
        "bgnbd_transactions.csv": bg.transactions,
        "bgnbd_rfm.csv": bg.rfm,
        "mmm_weekly.csv": mm.weekly,
        "mmm_lift_tests.csv": mm.lift_tests,
        "geo_panel.csv": geo.panel,
        "email_experiment.csv": email.experiment,
        "mmm_confounded_weekly.csv": mmc.weekly,
        "mmm_confounded_lift_tests.csv": mmc.lift_tests,
        "mmm_confounded_latent.csv": mmc.latent,
    }
    truth = {
        "about": (
            "True parameters of the committed synthetic datasets in data/synthetic/, written by "
            "scripts/make_synthetic.py from src/mktstats/synth/. Do not edit by hand."
        ),
        "files": {
            "retailer": ["retailer_transactions.csv", "retailer_customers.csv"],
            "btyd_bgnbd": ["bgnbd_transactions.csv", "bgnbd_rfm.csv"],
            "mmm": ["mmm_weekly.csv", "mmm_lift_tests.csv"],
            "geo_panel": ["geo_panel.csv"],
            "email_experiment": ["email_experiment.csv"],
            "mmm_confounded": ["mmm_confounded_weekly.csv", "mmm_confounded_lift_tests.csv",
                               "mmm_confounded_latent.csv"],
        },
        "retailer": retail.truth,
        "btyd_bgnbd": bg.truth,
        "mmm": mm.truth,
        "geo_panel": geo.truth,
        "email_experiment": email.truth,
        "tolerances": TOLERANCES,
        "mmm_confounded": mmc.truth,
        "channel_value": synth.channel_value(retail.truth, {"mmm": mm.truth,
                                                            "mmm_confounded": mmc.truth}),
    }
    return frames, truth


def write(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    frames, truth = build()
    paths = []
    for name, df in frames.items():
        p = outdir / name
        df.to_csv(p, index=False, date_format="%Y-%m-%d", lineterminator="\n")
        paths.append(p)
    p = outdir / "truth.json"
    p.write_text(json.dumps(truth, indent=2, sort_keys=False) + "\n")
    paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if regenerating changes any file")
    args = ap.parse_args(argv)
    if not args.check:
        for p in write(OUT):
            print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        fresh = write(Path(tmp))
        changed = []
        for p in fresh:
            committed = OUT / p.name
            if not committed.exists() or committed.read_bytes() != p.read_bytes():
                changed.append(p.name)
        extra = sorted({q.name for q in OUT.glob("*")} - {p.name for p in fresh} - {".gitkeep"})
    if changed:
        print("synthetic data drift: regenerate with `python scripts/make_synthetic.py`; changed: "
              + ", ".join(changed))
        return 1
    if extra:
        print("unexpected files in data/synthetic/: " + ", ".join(extra))
        return 1
    print("data/synthetic is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
