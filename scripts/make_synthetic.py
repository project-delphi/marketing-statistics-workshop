"""Write the default-seed synthetic datasets to data/synthetic/ (CSV) and their truth to truth.json.

The extension modules (13-17) each write their own CSVs and their own truth file,
truth_<name>.json (EXTENSIONS below), so adding or changing an extension dataset leaves
truth.json, which the core modules' run records hash, unchanged (DECISIONS.md).

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
from collections.abc import Callable
from pathlib import Path

import pandas as pd

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
    "capstone": {
        "note": "Checkpoint tolerances are in capstone.answers.*.checkpoint; these are the "
                "reference results measured on the committed capstone data (2026-10-09), so "
                "each tolerance can be read against a correct pipeline.",
        "stage1_bgnbd_map": {
            "method": "BetaGeoModel MAP with channel_social and channel_referral on purchase and "
                      "dropout; GammaGamma MAP with WEAK_GAMMA_GAMMA_PRIORS; Module 5's monthly "
                      "value from expected_purchases_new_customer",
            "relative_error": {"search": 0.086, "social": 0.148, "referral": -0.007},
            "top_channel_right": True,
        },
        "stage2_synthetic_control": {
            "method": "mktstats.recovery.synthetic_control on the treated mean, 32 placebos",
            "relative_error": 0.018, "p_value": 0.030,
            "sigma_relative_to_effect": 0.086,
            "sigma_rule": "sd of placebo relative effects (placebos with pre-RMSPE at most twice "
                          "the median) times the treated synthetic total",
        },
        "stage3_mcmc": {
            "sampler": "nutpie, 2 chains, random_seed=1, default priors, lift row from stage 2",
            "search_truth": 2.6,
            "uncalibrated_search_mean_hdi": {"300": [4.058, 2.738, 5.465],
                                             "1000": [3.928, 2.514, 5.426]},
            "calibrated_search_mean_hdi": {"300": [2.875, 2.510, 3.186],
                                           "1000": [2.863, 2.536, 3.202]},
            "calibrated_inside": "4 of 4 at 300 and 1000 draws",
            "spend_increase_instead": "with a +100% search test (spend doubled) in place of the "
                                      "switch-off, calibration moved search ROAS up, away from "
                                      "the truth (500 draws: 3.92 -> 4.71 with a milder shock, "
                                      "5.63 -> 7.82 with Module 9's), because the test probes "
                                      "the curve above the observed spend, not its level",
        },
        "stage5_targeting": {
            "test_split_true_value": {"t_learner_hist_gb_200_iter": 0.316,
                                      "causal_forest_dml_200_trees": 0.383,
                                      "treat_all": 0.126, "oracle": 0.512},
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
    cap = synth.capstone()
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
        "capstone_transactions.csv": cap.transactions,
        "capstone_customers.csv": cap.customers,
        "capstone_mmm_weekly.csv": cap.weekly,
        "capstone_geo_panel.csv": cap.panel,
        "capstone_email_experiment.csv": cap.experiment,
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
            "capstone": ["capstone_transactions.csv", "capstone_customers.csv",
                         "capstone_mmm_weekly.csv", "capstone_geo_panel.csv",
                         "capstone_email_experiment.csv"],
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
        "capstone": cap.truth,
    }
    return frames, truth


# Extension datasets: {name: builder}. A builder runs its generator with the default seed and
# returns ({filename: DataFrame}, truth); the truth goes to truth_<name>.json, with the CSV names
# under "files". Every filename starts with the dataset's name. One entry per extension module.
# Module 14's checkpoint rules, applied with the reference solutions (mktstats.survival
# .checkpoint_outcomes) to datasets from other generator seeds at the default settings.
# Measured 2026-10-10 with lifelines 0.30.3, numpy 2.1.3, pandas 2.2.3 on macOS arm64;
# tests/test_synth_subscriptions.py re-measures the first block (pytest -m slow).
SUBSCRIPTIONS_TOLERANCES = {
    "basis": (
        "How often each checkpoint rule of Module 14 holds when the reference solutions are run "
        "on data from other seeds of the same generator (n = 8,000, default settings). A rule "
        "that holds in nearly every seed passes on a correct solution and still fails on the "
        "wrong ones it targets. Intervals are named: 95% pointwise (exponential Greenwood) for "
        "Kaplan-Meier, 95% Wald confidence intervals for Cox hazard ratios, and the 95% "
        "bootstrap percentile interval (stratified by plan x onboarding, seed 14) for the RMST "
        "gain."
    ),
    "measured": "2026-10-10, lifelines 0.30.3, numpy 2.1.3, pandas 2.2.3",
    "seed_sets": {
        "1000-1039": {
            "n_seeds": 40,
            "km_truth_3_of_4": 39,
            "km_truth_all_4": 34,
            "km_s12_inside": 40,
            "dashboards_outside": 40,
            "cox_onboarded_inside_and_2_of_3": 39,
            "cox_all_3_inside": 34,
            "cox_unadjusted_outside_and_lower": 40,
            "ph_offer_flagged_p_below_0_01": 40,
            "ph_other_covariate_flagged": 1,
            "landmark_3_of_4": 40,
            "landmark_all_4": 33,
            "landmark_each": {"offer_before": 38, "offer_after": 40, "onboarded_before": 37,
                              "onboarded_after": 38},
            "rmst_truth_inside_B100": 38,
            "rmst_truth_inside_B200": 38,
            "rmst_unadjusted_outside_B100": 40,
            "rmst_unadjusted_outside_B200": 40,
            "means": {"hr_standard": 0.7822, "hr_premium": 0.5493, "hr_onboarded": 0.6694,
                      "hr_offer_full_period": 1.3823, "hr_onboarded_unadjusted": 0.5698,
                      "rmst_gain": 2.8974, "rmst_unadjusted": 4.0525,
                      "dashboard_a": 0.5341, "dashboard_b": 0.4285},
            "note": "the seeds of the brief's prototype; this generator reproduces its draws",
        },
        "3000-3039": {
            "n_seeds": 40,
            "km_truth_3_of_4": 39,
            "km_truth_all_4": 34,
            "km_s12_inside": 39,
            "dashboards_outside": 40,
            "cox_onboarded_inside_and_2_of_3": 34,
            "cox_all_3_inside": 32,
            "cox_unadjusted_outside_and_lower": 40,
            "ph_offer_flagged_p_below_0_01": 40,
            "ph_other_covariate_flagged": 0,
            "landmark_3_of_4": 40,
            "landmark_all_4": 32,
            "landmark_each": {"offer_before": 40, "offer_after": 37, "onboarded_before": 37,
                              "onboarded_after": 38},
            "rmst_truth_inside_B100": 34,
            "rmst_truth_inside_B200": 35,
            "rmst_unadjusted_outside_B100": 40,
            "rmst_unadjusted_outside_B200": 40,
            "means": {"hr_standard": 0.7793, "hr_premium": 0.5493, "hr_onboarded": 0.6618,
                      "hr_offer_full_period": 1.3932, "hr_onboarded_unadjusted": 0.5640,
                      "rmst_gain": 2.9712, "rmst_unadjusted": 4.1146,
                      "dashboard_a": 0.5343, "dashboard_b": 0.4294},
            "note": "an independent check; checkpoints 3 and 5 failed together on the same "
                    "seeds (3000, 3028, 3034, 3035, 3037), where the onboarding effect came "
                    "out stronger than the truth by chance",
        },
    },
    "summary": (
        "Over 80 seeds: Kaplan-Meier truth 3 of 4 in 78, dashboards outside in 80; checkpoint 3 "
        "in 73 and its unadjusted check in 80; checkpoint 4 (3 of 4) in 80; checkpoint 5's "
        "truth inside in 72 (B = 100) and 73 (B = 200), the unadjusted gain outside in 80. "
        "Checkpoints 3 and 5 rest on one 95% interval each, so about 1 seed in 10 misses: "
        "roughly the nominal 5% plus some undercoverage of the percentile bootstrap. On the "
        "committed seed (2034) every rule holds with the reference in QUICK and FULL, and a "
        "correct solution reproduces the reference's numbers exactly."
    ),
    "rules": {
        "km_vs_truth": "true S(t) at 6, 12, 18 and 24 months inside the 95% pointwise interval "
                       "for at least 3 of 4; dashboards A and B outside it at 12 months",
        "checkpoint_3": "true onboarding HR (0.670) inside its 95% CI and at least 2 of the 3 "
                        "true HRs (standard, premium, onboarded) inside theirs; with onboarding "
                        "alone, the truth outside the CI and the HR below the adjusted one",
        "checkpoint_4": "at least 3 of 4 inside their 95% CIs: offer HR 2.014 before month 3, "
                        "1.0 after, onboarding HR 0.670 before and after",
        "checkpoint_5": "true target-population gain (2.907 months) inside the 95% bootstrap "
                        "percentile interval; the unadjusted Kaplan-Meier gain outside it",
    },
}


def _subscriptions() -> tuple[dict[str, pd.DataFrame], dict]:
    """Module 14: the subscription program (seed 2034) and its checkpoint coverage."""
    from mktstats.synth.subscriptions import subscriptions

    res = subscriptions()
    return ({"subscriptions_subscribers.csv": res.subscribers},
            {**res.truth, "tolerances": SUBSCRIPTIONS_TOLERANCES})


def _mta() -> tuple[dict[str, pd.DataFrame], dict]:
    """Module 16: user journeys with targeted exposure (seed 2036, 100,000 users). The CSV
    writes one letter per touch (truth "path_codes"), so it stays under 1.5 MB."""
    from mktstats.synth import mta

    res = mta.mta_journeys()
    return {"mta_journeys.csv": mta.encode_paths(res.journeys)}, res.truth


def _messy() -> tuple[dict[str, pd.DataFrame], dict]:
    """Module 13: the synthetic retailer as a messy line log (mktstats.synth.messy, seed 2033).
    Only the lines are written; the per-line labels stay in memory for the tests."""
    from mktstats.synth import messy

    res = messy.messy_retailer()
    return {"messy_lines.csv": res.lines}, res.truth


def _mmm_drift() -> tuple[dict[str, pd.DataFrame], dict]:
    """Module 17: weekly MMM data in which social wears out (seed 2037), with its noiseless
    latent paths and one dated social lift test. The truth carries the measured reference
    results of the lab's checkpoints (MMM_DRIFT_TOLERANCES)."""
    from mktstats.synth import mmm_drift

    res = mmm_drift.mmm_drift()
    frames = {"mmm_drift_weekly.csv": res.weekly, "mmm_drift_latent.csv": res.latent,
              "mmm_drift_lift_tests.csv": res.lift_tests}
    return frames, {**res.truth, "tolerances": MMM_DRIFT_TOLERANCES}


# Module 17: measured with the reference solutions on the committed data (mktstats.drift.
# reference_results, pymc-marketing 1.2.0, nutpie 0.16.11, prophet 1.4.0, 2 chains on 2 cores,
# random_seed 2037, draws = tune = 300 (QUICK) and 1000 (FULL), target_accept 0.98), on an
# Apple Silicon laptop, 2026-10-10. Over more seeds: tests/test_synth_mmm_drift.py (slow).
MMM_DRIFT_TOLERANCES: dict = {
    "sampler": "nutpie, 2 chains on 2 cores, draws = tune = 300 (QUICK) and 1000 (FULL), "
               "target_accept 0.98, random_seed 2037, Module 8's spend-share priors",
    "interval": "94% HDI (mktstats.checks.interval), mean and [low, high]",
    "checks": {
        "checkpoint_1": "window_roas on the latent contributions equals roas, "
                        "roas_first_year and roas_last_quarter to a relative 1e-9",
        "static_model": "window ROAS over all weeks equals contribution_over_spend to 1e-8; "
                        "social's last-quarter HDI lies above roas_last_quarter",
        "checkpoint_2": "social's residual-spend correlation falls most from the first year to "
                        "the last and ends below -0.3",
        "checkpoint_4": "coverage of the true relative path: social >= 0.5, others >= 0.9",
        "drifting_model": "social's drift-ratio HDI lies below 1; the others' contain 1",
        "prophet": "trend + season + holiday = yhat to 1e-6",
        "checkpoint_5": "social: Prophet-baseline mean below own-baseline mean, its HDI "
                        "excludes roas, the own-baseline HDI covers it",
        "decision": "cut if P(last-quarter ROAS < break-even) >= 0.9 under the drifting model",
    },
    "measured_2026_10_10": {
        "static_social_roas_all": {"300": [3.892, 3.439, 4.313], "1000": [3.895, 3.522, 4.317]},
        "static_social_roas_last_quarter": {"300": [3.875, 3.417, 4.309],
                                            "1000": [3.879, 3.477, 4.282]},
        "static_vs_contribution_over_spend_max_gap": {"300": 4.3e-14, "1000": 3.2e-14},
        "residual_corr_social_2023_2024_2025": {"300": [0.405, 0.196, -0.569],
                                                "1000": [0.404, 0.195, -0.574]},
        "prior_social_drift_ratio": {"p_le_0.5": 0.088, "p_le_truth": 0.074,
                                     "eti94": [0.217, 5.119], "draws": 1000},
        "hsgp_m_L": [56, 426.4],
        "coverage": {"300": {"tv": 1.0, "search": 1.0, "social": 0.647, "display": 1.0},
                     "1000": {"tv": 1.0, "search": 1.0, "social": 0.686, "display": 1.0}},
        "drift_ratio_social": {"300": [0.599, 0.455, 0.752], "1000": [0.570, 0.429, 0.706]},
        "drift_ratio_hdi_others_1000": {"tv": [0.904, 1.130], "search": [0.488, 1.438],
                                        "display": [0.505, 1.763]},
        "drifting_social_roas_last_quarter": {"300": [2.669, 1.975, 3.285],
                                              "1000": [2.533, 1.942, 3.151]},
        "drifting_social_p_below_break_even": {"300": 0.963, "1000": 0.992},
        "drifting_social_counterfactual_last_quarter_1000": [2.449, 1.889, 3.061],
        "counterfactual_definition_max_gap_on_draws": 1.8e-15,
        "prophet": {"trend_per_week": 148.5, "holiday_effect": 53641,
                    "trend_gap_corr_social": 0.563, "sum_gap": 1.2e-10},
        "prophet_baseline_social_roas_all": {"300": [2.889, 2.303, 3.654],
                                             "1000": [2.901, 2.245, 3.577]},
        "prophet_baseline_tv_roas_all_1000": [1.443, 1.249, 1.629],
        "stretch_a_social_last_quarter_1000": [2.219, 1.916, 2.504],
        "stretch_b_roas_all_1000": {"tv": [1.511, 1.401, 1.628],
                                    "social": [3.153, 2.667, 3.717]},
        "divergences": "0 in every fit, QUICK and FULL, stretch fits included",
    },
}


EXTENSIONS: dict[str, Callable[[], tuple[dict[str, pd.DataFrame], dict]]] = {
    "messy": _messy,
    "subscriptions": _subscriptions,
    "mta": _mta,
    "mmm_drift": _mmm_drift,
}


def build_extension(name: str) -> tuple[dict[str, pd.DataFrame], dict]:
    frames, truth = EXTENSIONS[name]()
    bad = [f for f in frames if not f.startswith(name)]
    if bad:
        raise ValueError(f"extension {name}: file names must start with {name!r}: {bad}")
    about = (f"True parameters of the {name} extension dataset in data/synthetic/, written by "
             "scripts/make_synthetic.py from src/mktstats/synth/. Do not edit by hand.")
    return frames, {"about": about, "files": sorted(frames), **truth}


def _write_frames(outdir: Path, frames: dict[str, pd.DataFrame]) -> list[Path]:
    paths = []
    for name, df in frames.items():
        p = outdir / name
        df.to_csv(p, index=False, date_format="%Y-%m-%d", lineterminator="\n")
        paths.append(p)
    return paths


def _write_json(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    return path


def write(outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    frames, truth = build()
    paths = _write_frames(outdir, frames)
    paths.append(_write_json(outdir / "truth.json", truth))
    for name in EXTENSIONS:
        ext_frames, ext_truth = build_extension(name)
        paths += _write_frames(outdir, ext_frames)
        paths.append(_write_json(outdir / f"truth_{name}.json", ext_truth))
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
