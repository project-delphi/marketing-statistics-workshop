# Lab brief · Module 8 · Bayesian marketing mix models

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-09, Academic Director. Written before the lab was built; the lab has since passed on Colab and in the workshop's Docker image (`runs/`; the Readiness page). Review: Pedagogy Expert. |
| Notebook | `labs/src/08-bayesian-mmm.py` → `labs/python/08-bayesian-mmm.ipynb` |
| Lab slot | 55 min (`modules.m08.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Day | Day 4, first module |

## Question and decision

**Question.** How much did each media channel contribute to sales, and what did each dollar return?

**Decision.** Which channels pay back: channels whose ROAS exceeds the break-even ROAS (1 ÷ gross margin)
with posterior probability ≥ 0.9, channels below it with probability ≥ 0.9, and channels the data cannot
yet decide (`modules.m08.decision`).

## What this lab fixes

Module 7 measured one campaign in one window; it cannot compare channels or say what the next dollar
returns.

## Notation

Weekly sales y_t; spend x_{c,t} per channel c. Geometric adstock with decay α_c ∈ (0, 1) and maximum lag
L = 8: a_{c,t} = Σ_{l=0}^{L−1} α_c^l x_{c,t−l} / Σ_l α_c^l (normalized). Logistic saturation with λ_c:
s(a) = (1 − e^{−λa}) / (1 + e^{−λa}). Channel effect β_c · s(a_{c,t}). ROAS_c over a window = incremental
sales caused by c's spend in the window ÷ c's spend in the window (the counterfactual sets that spend to 0
and keeps carryover). PyMC-Marketing scales y and x internally, so priors are on the scaled problem.

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic weekly MMM (the retailer's media) | `mktstats.synth.mmm(seed=<default>)` (on main: 156 weeks, columns `date_week, tv, search, social, display, price_index, holiday, t, y`); truth per channel `adstock_alpha`, `saturation_lam`, `saturation_beta_model_units`, `roas`, `roas_with_carryover`, `contribution_share`, the ROAS window (all weeks) and `tolerances.mmm_mcmc` (true ROAS inside the 94% HDI for at least 3 of 4 channels with nutpie 2×500 and default priors) | everything with truth |
| PyMC-Marketing `mmm_example.csv` (demo data in the PyMC-Marketing repository; the pages that use it point to a "simulated example", no real source is given; we treat it as simulated demo data with unknown true parameters) | `mktstats.data.load_mmm_example()`: `date_week, y, x1, x2, event_1, event_2, dayofyear, t` (179 weeks) | a dataset without truth (Exercise 6) |

The truth's ROAS must use the same definition as `mmm.incrementality.contribution_over_spend(frequency=
"all_time", start_date=..., end_date=..., include_carryover=True)` (counterfactual spend factor 0); the
Technical Expert checks this on noiseless data.

## Model fits (once each, provided cells)

```python
mmm = MMM(date_column="date_week", channel_columns=["tv", "search", "social", "display"],
          control_columns=["price_index", "holiday", "t"], target_column="y",
          adstock=GeometricAdstock(l_max=8), saturation=LogisticSaturation(), yearly_seasonality=2,
          model_config={"saturation_beta": Prior("HalfNormal", sigma=sigma_from_ex3, dims="channel")})
mmm.build_model(X, y)
prior = mmm.sample_prior_predictive(X, y, samples=500)
mmm.fit(X, y, nuts_sampler="nutpie", chains=2, draws=D, tune=D, random_seed=SEED, progressbar=False)
mmm.sample_posterior_predictive(X, extend_idata=True)
```

QUICK: D = 300; FULL: D = 1000. Timings we have: 2-channel `mmm_example` 2×300 took 6.1 s on a laptop
(2026-10-09); the lead measured 2×500 at 9.7 s on Colab after a 35–45 s first-fit compile. The 4-channel fit
is estimated under a minute FULL on Colab until a run record exists. The `mmm_example` fit (Exercise 6) uses
`channel_columns=["x1", "x2"]`, `control_columns=["event_1", "event_2", "t"]`, `l_max=8`,
`yearly_seasonality=2`, the same sampler settings.

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · The two transforms | 1 · Geometric adstock | 7 |
| | 2 · Logistic saturation | 5 |
| B · Priors before data | 3 · Priors from spend shares | 7 |
| | *Run: prior predictive check, then the fit (provided)* | — |
| C · Can we trust the fit? | 4 · Diagnostics before results | 7 |
| D · What each dollar returned | 5 · ROAS against the truth | 9 |
| E · A dataset without truth | 6 · Contribution shares on `mmm_example` | 5 |
| **Exercises** | | **40** |
| Decision | Which channels pay back? | 5 |

### Exercise 1 · Geometric adstock (7 minutes)

- **Predict.** With α = 0.5 and normalized weights over 8 weeks, what share of one week's TV effect lands in
  the week of spend: about a quarter, about a half, or about three quarters?
- **Function.** `geometric_adstock(x: np.ndarray, alpha: float, l_max: int = 8, normalize: bool = True) ->
  np.ndarray` (convolution with weights α^l, l = 0…l_max − 1, divided by their sum when normalized).
- **Checkpoint.** On an impulse, equals α^l / Σα^l; on random spend, equals PyMC-Marketing's
  `transformers.geometric_adstock(as_xtensor(x, dims=("date",)), alpha=alpha, l_max=8, dim="date",
  normalize=True).eval()` to 1e-8 (the reference call lives in the checkpoint, not in the learner's code).
  First checkpoint, before any fit.
- **Explain.** Normalized adstock moves an effect later in time without changing its total. For α = 0.5 the
  weights are 0.502, 0.251, 0.126, … (checked 2026-10-09).

### Exercise 2 · Logistic saturation (5 minutes)

- **Predict.** At λ·x = 1, what share of the maximum effect does logistic saturation give: under 50%, about
  50%, or over 50%?
- **Function.** `logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray`.
- **Checkpoint.** Equals PyMC-Marketing's `transformers.logistic_saturation` (same xtensor call pattern) to
  1e-8; increasing and in [0, 1) (`checks.monotone`).
- **Explain.** 0.462 at λ·x = 1 (checked): each extra dollar returns less. Together with adstock this makes
  the response curve Module 10 optimizes.

### Exercise 3 · Priors from spend shares (7 minutes)

- **Predict.** Before seeing sales, should the channel with the largest spend get a wider or narrower prior
  on its effect than the smallest channel?
- **Function.** `beta_prior_sigma(spend: pd.DataFrame) -> np.ndarray`: per channel, the HalfNormal scale for
  `saturation_beta` proportional to the channel's share of total spend, scaled so the scales average 1
  (σ_c = C × share_c for C channels). This is our design choice, stated in the notebook: without other
  information, expect channels to matter in proportion to what the business spends on them.
- **Checkpoint.** Scales are positive, sum to C, and have the same order as total spend.
- **Explain.** The default prior gives every channel the same scale (HalfNormal σ = 2 in
  `default_model_config`), which says a tiny channel could matter as much as TV.

*Provided after this exercise:* `sample_prior_predictive`, plotted with one sentence: "Look at whether the
observed sales lie inside the prior's 94% band, and whether the band allows absurd values (negative or ten
times the observed)." Then the fit.

### Exercise 4 · Diagnostics before results (7 minutes)

- **Predict.** Will the fit have no divergences, a few (under 10), or many?
- **Function.** `diagnose(idata) -> dict` with `divergences` (sum of `idata["/sample_stats"]["diverging"]`),
  `max_rhat` (`az.rhat`) and `min_ess_bulk` (`az.ess(..., method="bulk")`) over the model's parameters
  (`adstock_alpha`, `saturation_lam`, `saturation_beta`, controls, intercept, noise).
- **Checkpoint.** Matches `az.summary(idata, var_names=..., ci_prob=0.94, ci_kind="hdi")` on the real fit; on
  a provided toy DataTree with 3 divergences and a stuck chain, returns 3 and an R-hat above 1.1.
- **Explain.** Read results only if divergences are 0, R-hat ≤ 1.01 and bulk ESS ≥ 400 (FULL; QUICK will
  often miss the ESS bar, and the notebook says so). The posterior predictive plot (provided) is the second
  check: point to weeks the model misses.

### Exercise 5 · ROAS against the truth (9 minutes)

- **Predict.** Which channel will have the widest ROAS interval: tv, search, social or display? Pick one.
- **Function.** `roas_table(roas_draws: xr.DataArray, truth: dict, prob: float = 0.94) -> pd.DataFrame` with
  per channel the posterior mean, the 94% HDI, the true ROAS and whether it is covered. Provided:
  `roas_draws = mmm.incrementality.contribution_over_spend(frequency="all_time", start_date=w0,
  end_date=w1)` (dims `(chain, draw, channel)`).
- **Checkpoint.** `checks.k_of_K_in_interval`: the true ROAS lies inside the 94% HDI for at least 3 of the 4
  channels (four 94% intervals all cover about 78% of the time); the Technical Expert confirms on the
  committed seed in QUICK and FULL.
- **Explain.** Channels with spend that barely varies, or that moves with another channel, get wide
  intervals: the data cannot separate their effect. Point to the spend plot.

### Exercise 6 · Contribution shares on `mmm_example` (5 minutes)

*Provided before this exercise: the `mmm_example` fit and `diagnose(idata_example)`.*

- **Predict.** Which channel contributes more to sales in `mmm_example`, x1 or x2?
- **Function.** `contribution_share(idata) -> pd.DataFrame`: each channel's share of total media
  contribution (sum `posterior["channel_contribution"]` over dates, divide by the total across channels, per
  draw) with posterior mean and 94% HDI.
- **Checkpoint.** Shares sum to 1 in every draw (to 1e-9) and lie in [0, 1]; values match the reference.
- **Explain.** Without truth, only diagnostics, predictive checks and outside evidence (Module 9's lift
  tests) protect you. Shares, not ROAS: in this file x1 and x2 are not dollar spend, so
  `contribution_over_spend` returned values in the thousands (about 5,000 and 2,400 per unit on 2026-10-09),
  which is not a return on dollars.

### Decision · Which channels pay back? (5 minutes)

Provided cell, in the standard order:
- **Number:** per channel, ROAS posterior mean and 94% HDI (Exercise 5) and the posterior probability that
  ROAS exceeds the break-even ROAS, computed draw by draw.
- **Rule:** "A channel pays back if gross margin × ROAS > 1, so the break-even ROAS is 1 ÷ margin (1 ÷ 0.3 ≈
  3.3 at the stated 30% margin). Classify a channel as paying back if P(ROAS > break-even) ≥ 0.9, as not
  paying back if P(ROAS < break-even) ≥ 0.9, otherwise undecided."
- **Recommendation:** the learner's sentence naming the three groups and what would change them (a lift test
  for an undecided channel: Module 9). Then the cell shows the true ROAS. The cell prints the mode.

## Stretch (optional)

Refit with the default equal priors and compare the ROAS intervals with Exercise 3's priors; or try
`HillSaturation` instead of `LogisticSaturation` and compare predictive fit.

## Known pitfalls

- The transformer functions in pymc-marketing 1.2.0 take xtensors with a named `dim`; calling them on a plain
  NumPy array fails with "missing 1 required keyword-only argument: 'dim'" (seen 2026-10-09).
- `mmm.summary.roas(hdi_probs=[0.94], method="incremental", frequency="all_time")` labels its interval
  columns `abs_error_94_lower` / `abs_error_94_upper`; say in the notebook that these are the 94% HDI.
- ArviZ 1.x defaults to 89% ETI: always pass `ci_prob` and `ci_kind` (or `prob` to `az.hdi`).
- `mmm.plot` warns that the legacy plot suite will be removed in 2.0; prefer the notebook's own matplotlib
  plots from posterior arrays.
- `contribution_over_spend` is NaN where a channel has zero spend in the window.

## APIs verified (and how)

Installed pymc-marketing 1.2.0, run on `mmm_example.csv` on a laptop, 2026-10-09: `MMM(...)` signature,
`default_model_config` (`saturation_beta` HalfNormal σ = 2, `adstock_alpha` Beta(1, 3), `saturation_lam`
Gamma(3, 1)), `build_model`, `sample_prior_predictive(X, y, samples=)`, `fit(..., nuts_sampler="nutpie")`,
posterior variables (`adstock_alpha`, `saturation_lam`, `saturation_beta`, `channel_contribution`, …),
`sample_stats.diverging`, `incrementality.contribution_over_spend` (dims `(chain, draw, channel)` with
`frequency="all_time"`), `summary.roas`; `transformers.geometric_adstock` and `logistic_saturation` with
xtensor inputs. ArviZ 1.3.0 `az.summary(ci_prob=, ci_kind=)`, `az.rhat`, `az.ess`.

## Open items for the Technical Expert

- Truth ROAS computed with the same counterfactual as `contribution_over_spend` (see Data).
- `load_mmm_example()` with a docstring saying what the source does and does not say about the data.
