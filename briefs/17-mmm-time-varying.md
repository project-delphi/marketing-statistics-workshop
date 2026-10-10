# Lab brief · Module 17 · MMM with drifting effects and a Prophet baseline

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. Written before the lab exists. Every number below marked *prototype* comes from a throwaway prototype on a laptop (a stand-in generator with the design below, pymc-marketing 1.2.0, nutpie, 2 chains on 2 cores, a machine shared with other jobs); the committed generator will draw different numbers, so the Technical Expert re-measures every checkpoint on the committed data in QUICK and FULL. Nothing here has run in Docker or on Colab. Review: Pedagogy Expert. |
| Notebook | `labs/src/17-mmm-time-varying.py` → `labs/python/17-mmm-time-varying.ipynb` |
| Lab slot | 55 min (`modules.m17.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
| Track | Extensions (`days.d6`), optional, outside the five timed days. It keeps the core format (45 min briefing, 55 min lab, 10 min debrief) so an instructor can swap it into a custom schedule after Day 4. |

## Question and decision

**Question.** Has a channel's effect per dollar changed, and what is it worth now?

**Decision** (`modules.m17.decision`). Decide whether to cut the fading channel's budget next quarter
(social, in the lab): cut if its current-quarter ROAS is below break-even (1 ÷ gross margin) with
posterior probability ≥ 0.9, keep if above with probability ≥ 0.9, otherwise run a lift test first; and
say how this differs from the answer the all-period average gives.

## What this lab fixes

Modules 8 to 10 assume a channel's effect per dollar is the same in every week of the history. When a
channel wears out (the same creative shown too often, an audience used up, a platform change), a static
MMM still reports one ROAS for the whole history: a weighted average of a good past and a poor present.
Module 9 showed that an MMM can be wrong while fitting well; this lab shows a second way, and a model that
lets the effect move.

The lab also tests an idea practitioners use, in the form Robyn uses it: decompose sales with Prophet
into trend, seasonality and holidays, and give those components to the MMM as its baseline. We measure
against the truth whether that helps or biases channel ROAS.

## Notation (same symbols as Module 8)

- Weekly sales $y_t$, spend $x_{c,t}$ for channel $c$ in week $t = 1, \dots, T$ ($T = 156$).
- Geometric adstock $a_{c,t}$ with decay $\alpha_c$ and $L = 8$, normalized; logistic saturation
  $s(z; \lambda) = (1 - e^{-\lambda z}) / (1 + e^{-\lambda z})$ applied to $a_{c,t} / S_c$, where
  $S_c = \max_t x_{c,t}$ (PyMC-Marketing's default scaling).
- **Effectiveness multiplier** $m_c(t) > 0$: how strongly channel $c$ works in week $t$ relative to its
  average. The channel's contribution is
  $$\text{contrib}_{c,t} = \beta_c \, m_c(t) \, s(a_{c,t} / S_c;\ \lambda_c),$$
  with $\beta_c$ the average effect size. A static MMM has $m_c(t) = 1$ for every week.
- **Relative path** $\tilde m_c(t) = m_c(t) / \bar m_c$, where $\bar m_c$ is the mean of $m_c(t)$ over
  the $T$ fitted weeks. PyMC-Marketing's multiplier is normalized this way (its mean over the fitted
  weeks is 1), so the model's multiplier and the truth's relative path are directly comparable.
- **ROAS over a window** $W$ (a set of weeks):
  $$\text{ROAS}_c(W) = \frac{\sum_{t \in W} \text{contrib}_{c,t}}{\sum_{t \in W} x_{c,t}},$$
  the sales the channel produced in those weeks per dollar spent in those weeks (carry-in from earlier
  spend counts, carry-out after the window does not). Over all weeks it is Module 8's ROAS. Windows used:
  all weeks (2023-01-02 to 2025-12-22), the first year (weeks 1 to 52, 2023-01-02 to 2023-12-25) and the
  last quarter (weeks 144 to 156, 2025-09-29 to 2025-12-22). PyMC-Marketing's
  `contribution_over_spend(start_date=, end_date=)` uses a counterfactual instead (spend set to zero in the
  window, carry-in kept); the two differ slightly for a window shorter than the history (prototype, social,
  last quarter: truth 2.11 by the window definition, 2.04 by the counterfactual one; the drifting model
  2.53 and 2.45). The lab uses the window definition, which one function computes for the truth and for
  posterior draws alike, and says so in a Markdown cell.
- **Drift ratio** $r_c = \text{mean}_{t \in \text{last quarter}}\ m_c(t) \,/\, \text{mean}_{t \in \text{first year}}\ m_c(t)$:
  how much of its first-year effect the channel keeps in the last quarter.
- Gross margin 30%, so the **break-even ROAS** is $1 / 0.30 \approx 3.33$ (Modules 8 and 9).

## Data

| Source | Loader | Use |
|---|---|---|
| Synthetic weekly MMM with a fatiguing channel: `mmm_drift_weekly.csv` (`date_week, tv, search, social, display, price_index, holiday, t, y`, 156 rows) | `pd.read_csv(repo_file("synthetic/mmm_drift_weekly.csv"), parse_dates=["date_week"])` (`mktstats.data.repo_file` returns a local path: the checkout, or a download from raw GitHub at `MKTSTATS_REF`) | every fit |
| Hidden truth by week: `mmm_drift_latent.csv` (`date_week, baseline, baseline_trend, seasonality`, and per channel `multiplier_<c>`, `multiplier_rel_<c>`, `contribution_<c>`, all noiseless, dollars) | the same, `"synthetic/mmm_drift_latent.csv"` | checkpoints, plots, never a model input |
| One dated lift test on social in the last quarter: `mmm_drift_lift_tests.csv` (Module 9's columns: `channel, x, delta_x, delta_y, sigma, date, test_start, test_weeks, true_delta_y`) | the same, `"synthetic/mmm_drift_lift_tests.csv"` | stretch A only |
| Truth: `truth_mmm_drift.json` | `json.loads(repo_file("synthetic/truth_mmm_drift.json").read_text())` | checkpoints, decision |

### Generator specification (`src/mktstats/synth/mmm_drift.py`, function `mmm_drift(seed=2037, ...)`)

A new file, not exported from `mktstats.synth.__init__`; the lab imports nothing from it (it reads the
committed files). It may import `geometric_adstock` and `logistic_saturation` from `mktstats.synth.mmm`
and `make_rng`, `SynthResult`, `sig` from `mktstats.synth._common`; no existing file changes. Registered in
`EXTENSIONS` in `scripts/make_synthetic.py` under the name `mmm_drift`, writing the three CSVs above and
`truth_mmm_drift.json`. Seed **2037** (the lead's numbering for the extension generators).

Data-generating process (the prototype's; every value is a default argument):

| Part | Specification |
|---|---|
| Calendar | 156 weeks starting Monday 2023-01-02 (`freq="7D"`), as Module 8. |
| Channels | tv, search, social, display. Adstock $\alpha$ = 0.6, 0.2, 0.4, 0.3; saturation $\lambda$ = 1.5, 2.5, 2.0, 3.0 (Module 8's values); $L = 8$, normalized, zero spend before week 1. |
| All-period ROAS targets | tv 1.6, search 3.0, social **4.0**, display 0.6. $\beta_c$ (sales units) is solved so that $\sum_t \text{contrib}_{c,t} / \sum_t x_{c,t}$ equals the target, with the multiplier included. |
| Fatigue | Social only: $m_\text{social}(t) = 1 - 0.65 / (1 + e^{-(t - 104)/20})$ for $t = 0, \dots, 155$ (about 1.0 in the first year, 0.40 in the last week); the other channels have $m_c(t) = 1$. The multiplier acts on the week the effect lands, as PyMC-Marketing's time-varying media does. Smooth by design: the point is a slow decline whose true path is known, not a test of how rough a path a prior can follow. |
| Baseline | $250{,}000 + 500\,t$ + yearly Fourier terms (15,000 sin, 25,000 cos, 5,000 sin 2, −8,000 cos 2 on day of year / 365.25, as Module 8) + price effect $-400{,}000\,(p_t - 1)$ (price index: AR(1) with coefficient 0.7 and sd 0.012 around 1, ten promotion weeks 4 to 10% lower, rounded to 4 decimals) + holiday effect 40,000 in the weeks containing Black Friday, December 18 and December 24 (Module 8's rule). |
| Spend (dollars, rounded) | **tv:** flights on for 4 to 8 weeks and off for 3 to 8, 60,000 × Uniform(0.7, 1.3) when on. **search:** follows demand, $20{,}000 \times d_t \times \text{LogNormal}(0, 0.12)$ with $d_t = (\text{intercept} + \text{trend}_t + \text{seasonality}_t) / \text{intercept}$; this is the channel whose spend trends with demand. **social:** $10{,}000 \times \text{LogNormal}(0, 0.3)$, multiplied by 2.5 in a four-week push starting every 13 weeks (weeks 5, 18, 31, … with a start jitter of −2 to +2 weeks, so every quarter has a push, the last one included). The pushes are what let a model see the effect at different times. **display:** $6{,}000 \times \text{LogNormal}(0, 0.3)$. |
| Noise | Normal(0, 8,000) added to sales; sales rounded to dollars. |
| Lift test (stretch A) | One randomized social test: +30% weekly spend for 8 weeks ending 2025-11-17 (week 151); $x$ = mean social spend over the 26 weeks before the test (rounded to $100), $\Delta x = 0.3x$ (rounded to $100); true $\Delta y = \beta_\text{social}\,\bar m\,[s((x+\Delta x)/S) - s(x/S)]$ with $\bar m$ the mean multiplier over the test weeks; $\sigma$ = 8% of the true lift (Module 9's relative error); $\Delta y$ = truth + Normal(0, $\sigma$); `date` = last week of the test. |

`truth_mmm_drift.json` (besides `about` and `files`, which `build_extension` adds):

- `description`, `seed`, `n_weeks`, `start_date`, `end_date`, `l_max`, the adstock and saturation forms,
  units (as `truth["mmm"]` in `truth.json`).
- `windows`: `all`, `first_year`, `last_quarter`, each `{start, end}` as dates.
- `channels.<c>`: `adstock_alpha`, `saturation_lam`, `saturation_beta_sales_units` ($\beta_c$),
  `channel_scale` ($S_c$), `roas` (all weeks), `roas_first_year`, `roas_last_quarter` (window definition),
  `roas_last_quarter_counterfactual` (PyMC-Marketing's definition: spend zeroed in the window, carry-in
  kept, summed to the end of the data), `spend` per window, `multiplier_mean` ($\bar m_c$),
  `drift_ratio` ($r_c$ of the true multiplier).
- `fatigue`: channel, functional form and its parameters (`end` 0.35, `mid` 104, `width` 20).
- `baseline`: every baseline parameter; `noise_sd`.
- `economics`: `gross_margin` 0.30, `break_even_roas`, and for social `pays_back_all_period` (true),
  `pays_back_last_quarter` (false), `spend_last_quarter`.
- `lift_tests`: definition and the row (as Module 9's truth).
- `tolerances`: the Technical Expert's measured reference results for each checkpoint on the committed
  data in QUICK and FULL (as `tolerances.mmm_confounded_mcmc` in `truth.json`).

Prototype values on seed 2037 of the stand-in generator: mean weekly spend tv $32,101, search $23,084,
social $15,533, display $6,417; mean weekly sales $478,188; media 39% of sales. Social's true ROAS: 4.00
over all weeks, 5.00 in the first year, 2.11 in the last quarter (drift ratio 0.42); social spend in the
last quarter $192,407.

## Model fits (once each, provided cells)

All three: Module 8's specification (`GeometricAdstock(l_max=8)`, `LogisticSaturation()`, spend-share
HalfNormal priors on `saturation_beta`), `nuts_sampler="nutpie"`, `chains=2`, `draws=tune=D`,
`target_accept=0.98`, `random_seed=SEED` (2037), `progressbar=False`. QUICK: D = 300; FULL: D = 1000.
With 0.98 the prototype had 0 divergences in all three fits, FULL and QUICK; at 0.95 the static FULL fit
had 1, and drifting fits of earlier designs had 3 to 4 (Module 9 made the same choice for the same
reason).

1. **Static.** Controls `price_index, holiday, t`, `yearly_seasonality=2` (Module 8).
2. **Drifting.** The static specification plus
   `time_varying_media=SoftPlusHSGP.parameterize_from_data(X=np.arange(156), dims=("date", "channel"),
   ls_lower=20, ls_upper=104, eta_upper=2.0, cov_func="matern52")`. That call gave $m = 56$ basis
   functions and $L = 426.4$ (measured). `sample_prior_predictive(X, y, samples=1000,
   random_seed=SEED)` comes first (Exercise 3); the prior draws are in `mmm.idata["prior"]`.
3. **Prophet baseline (static media).** Controls `price_index` and Prophet's `trend`, `season` and
   `holiday` components, each divided by the maximum of `y` (PyMC-Marketing scales the target by its
   maximum and does not scale controls, so a coefficient of 1 means "use Prophet's component as it is");
   `yearly_seasonality=None`; no `t` or `holiday` column. Informed prior:
   `Prior("Normal", mu=[0, 1, 1, 1], sigma=[2, 0.1, 0.1, 0.1], dims="control")` (DataArrays over the
   `control` coordinate): the MMM may rescale each Prophet component by about ±20% (94% of a Normal lies
   within ±1.88σ). The media part stays static on purpose: comparing it with fit 1 changes only the
   baseline.

**Why PyMC-Marketing and not a custom model.** In pymc-marketing 1.2.0, `time_varying_media=True`
creates one multiplier shared by all channels (dims `("date", *self.dims)`, `mmm/mmm.py`), which cannot
represent one channel wearing out while the others hold. But the argument also accepts an `HSGPBase`
instance with its own dims, and with `dims=("date", "channel")` each channel gets its own path: the HSGP
coefficients have dims `(channel, m)` (`HSGP.create_variable`, `mmm/hsgp.py`). The lengthscale and
amplitude priors are shared across channels (the class rejects non-scalar priors). The prototype built,
sampled and passed every check below with it, so neither a custom PyMC model nor a data-generating
process bent to fit a shared multiplier is needed. The cost: every channel gets a multiplier, so the
model must find that three of them stay near 1, which it did (Exercise 4). A Gaussian random walk on the
log multiplier, the other common choice, would need a custom model; the briefing mentions it, the lab
does not use it.

**Terms the notebook defines.** A *Gaussian process* (GP) is a prior over smooth functions; its
*lengthscale* $\ell$ is roughly how many weeks it takes the function to change appreciably, and its
*amplitude* $\eta$ how far it can move. An *HSGP* (Hilbert-space GP) approximates it with a fixed set of
basis functions so it samples fast (Riutort-Mayol et al.). `SoftPlusHSGP` passes the GP through a
softplus (always positive) and divides by its mean over the fitted weeks, so the multiplier is positive
with mean 1. The priors set here: $\ell$ InverseGamma with 90% of its mass between 20 and 104 weeks
(slow drift, not week-to-week noise); $\eta$ Exponential with a 5% chance of exceeding 2.

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · ROAS depends on the window | 1 · ROAS over a window | 7 |
| | *Run: static fit (provided) and its ROAS by window* | — |
| B · What the static model says about now | 2 · Residuals that drift | 7 |
| C · Letting the effect drift | 3 · How much drift does the prior allow? | 5 |
| | *Run: drifting fit (provided)* | — |
| | 4 · Does the band cover the true path? | 8 |
| D · A Prophet baseline | 5 · Prophet's decomposition of sales | 6 |
| | *Run: MMM with Prophet's components as an informed prior (provided)* | — |
| | 6 · Did the Prophet baseline help? | 7 |
| **Exercises** | | **40** |
| Decision | Cut or keep social next quarter? | 5 |

Names the lab uses: `weekly`, `latent`, `truth`, `CHANNELS`, `SEED = 2037`, `X`, `y`, `spend` (DataArray
`(date, channel)`), `WINDOWS` (dict of `(start, end)` date strings), `GROSS_MARGIN`, `BREAK_EVEN`,
`mmm_static`/`idata_static`, `mmm_drift`/`idata_drift`, `mmm_prophet`/`idata_prophet`, `roas_draws_*`.

### Exercise 1 · ROAS over a window (7 minutes)

*Provided before it:* the spend and sales plot, then the true relative paths from `latent` with one
sentence: "Look at social's line: its effect per dollar holds for a year, then falls to under half."

- **Predict.** Social's true ROAS over all three years is 4.0, above the break-even ROAS of 3.3. Will its
  ROAS in the last quarter be above break-even, between 2.5 and 3.3, or below 2.5? Write your guess.
- **Function.** `window_roas(contrib: xr.DataArray, spend: xr.DataArray, start: str, end: str) ->
  xr.DataArray`. `contrib` has dims `(..., date, channel)` in dollars (any leading dims, such as `chain`
  and `draw`); `spend` has dims `(date, channel)`; weeks from `start` to `end` inclusive
  (`.sel(date=slice(start, end))`); returns dims `(..., channel)`. The same function is used on the truth
  now and on posterior draws later.
- **Checkpoint.** First checkpoint, before any fit. (1) A toy with contribution = 2 × spend gives 2 for
  every channel. (2) A toy with leading dims `(2, 3)` returns shape `(2, 3, 4)`. (3) On the truth
  (`contribution_<c>` columns of `latent`), all three windows equal `truth_mmm_drift.json`'s `roas`,
  `roas_first_year` and `roas_last_quarter` to a relative 1e-9.
- **Explain.** Compare with your guess (prototype: 2.11, below 2.5). Which weeks dominate an all-period
  average: the many good weeks or the few recent ones? Point to the plot.
  `<details>`: the average weights each week by its spend; two good years outweigh one bad quarter, so
  the average stays above break-even long after the channel stopped paying back. A decision about next
  quarter needs the recent window.

*Provided after it (labelled checkpoint "static model", no exercise):* the static fit (fit 1), then
`window_roas` on `idata_static.posterior["channel_contribution"] * target_scale` (`target_scale` from
`idata_static.constant_data`). Checks: (a) over all weeks it equals
`mmm_static.incrementality.contribution_over_spend(frequency="all_time")` to 1e-8 (prototype: largest
difference 5e-14), which confirms the function on draws; (b) social's last-quarter ROAS 94% HDI lies
entirely above the true last-quarter ROAS. Prototype, seed 2037: all weeks 3.90, 94% HDI [3.53, 4.32]
(truth 4.00); last quarter 3.88 [3.48, 4.28] (truth 2.11), FULL; QUICK [3.42, 4.32]. In 5 of 5 seeds
(2037 to 2041) the truth lay below the static model's last-quarter HDI. A table shows each channel's
all-period and last-quarter ROAS with 94% HDIs next to the truth; the sentence before it: "Look at
social's two rows: the static model gives the same number for both windows."

### Exercise 2 · Residuals that drift (7 minutes)

On real data there is no truth. This exercise finds the drift from the static model alone.

- **Predict.** If a channel's effect has faded, the static model over-credits it late and under-credits it
  early. In which channel will the correlation between the model's residuals and the channel's spend fall
  most from 2023 to 2025: tv, search, social or display?
- **Function.** `residual_spend_corr(resid: pd.Series, spend: pd.DataFrame, year: pd.Series) ->
  pd.DataFrame`: index year, one column per channel, each the Pearson correlation between the residuals
  and that channel's weekly spend within the year. Provided: `resid = y − (posterior mean of
  idata_static.posterior["mu"]) × target_scale`, `year = weekly["date_week"].dt.year`.
- **Checkpoint.** Exact against `np.corrcoef` on a toy. On the static fit: social's correlation falls
  most from the first year to the last, and its last-year correlation is below −0.3. Prototype (QUICK,
  300 draws): social +0.40 in 2023, +0.19 in 2024, −0.58 in 2025; the other channels' changes from 2023
  to 2025 were at most +0.28. Over seeds 2037 to 2041 social fell by 0.75 to 1.19 and ended at −0.51 to
  −0.67, the largest fall in 5 of 5; the largest change for another channel was +0.49. The Technical
  Expert confirms in FULL.
- **Explain.** Compare with your pick. Why does a positive correlation in 2023 mean the model gives
  social too little credit then, and a negative one in 2025 too much? Point to the yearly table.
  `<details>`: with one effect size for all weeks, the model fits the average; in early weeks sales jump
  more than the model predicts when social spends (positive residuals in push weeks), in late weeks less
  (negative). A sign change in a channel's residual correlation is a cheap warning to run before
  believing an all-period ROAS. What would change it: spend that never varies within a year leaves
  nothing to correlate.

### Exercise 3 · How much drift does the prior allow? (5 minutes)

*Provided before it:* the drifting model built and `sample_prior_predictive` (1,000 draws), with the
terms GP, lengthscale, amplitude, HSGP and the mean-1 normalization defined in one Markdown cell
(Notation above).

- **Predict.** Before seeing any data, what probability does this prior give to a channel keeping half
  or less of its first-year effect in the last quarter: under 1%, about 5 to 10%, or over 25%?
- **Function.** `drift_ratio(mult: xr.DataArray, early: tuple[str, str], late: tuple[str, str]) ->
  xr.DataArray`: the mean of `mult` over the `late` weeks divided by its mean over the `early` weeks;
  dims `(..., channel)`.
- **Checkpoint.** A constant multiplier gives 1; a toy linear path gives its known ratio; on the prior
  draws (`idata["prior"]["media_temporal_latent_multiplier"]`) the result has dims `(chain, draw,
  channel)` and is positive and finite. *Provided after it:* the prior probability that social's ratio
  is at most the true one, and the prior's 94% equal-tailed interval (prototype, social: P(ratio ≤ 0.5)
  = 8.8%, P(ratio ≤ 0.42) = 7.4%, 94% equal-tailed interval 0.22 to 5.1; with `eta_upper=1.0`, P(ratio ≤
  0.42) was 2.4%).
- **Explain.** A prior that almost never allows a large fall cannot find one; a prior that allows
  anything follows noise. Is 7% enough room? Answer after Exercise 4.
  `<details>`: the lengthscale prior keeps the path smooth (changes over months, not weeks); the
  amplitude prior keeps it near 1 unless the data insist. This is Module 8's prior predictive check
  applied to the drift; it is cheaper than a refit and catches a prior that rules the answer out.

*Then the drifting fit (fit 2), the longest cell; the notebook asks the learner to read Exercise 4's
Predict prompt while it runs.*

### Exercise 4 · Does the band cover the true path? (8 minutes)

- **Predict.** In what share of the 156 weeks will social's true relative path lie inside the model's
  94% HDI: about 94%, between 60 and 90%, or under 50%?
- **Function.** `path_coverage(mult_draws: xr.DataArray, true_path: np.ndarray, prob: float = 0.94) ->
  float`: `mult_draws` has dims `(chain, draw, date)` for one channel; per week, the HDI of the draws
  (`checks.interval(draws, prob, "hdi")`); returns the share of weeks whose true value lies inside.
- **Checkpoint.** Toy: draws tightly around a known path give 1.0, shifted far away give 0.0. On the fit:
  social ≥ 0.5 and tv, search and display ≥ 0.9 (`latent["multiplier_rel_<c>"]` is the truth). With
  Exercise 3's function on the posterior: social's drift ratio 94% HDI lies entirely below 1, and the
  other three channels' HDIs contain 1. Prototype, seed 2037: social 0.69 (FULL) and 0.65 (QUICK); the
  other channels 1.0; drift ratio social 0.57 [0.43, 0.71] (truth 0.42), tv [0.90, 1.13], search
  [0.48, 1.43], display [0.50, 1.76]. Over seeds 2037 to 2041 (500 draws) social's coverage was 0.57 to
  1.0, every constant channel's at least 0.99, and social's ratio HDI below 1 in 5 of 5.
- **Explain.** Compare with your guess. Where does the band miss: at the two ends of the history, in
  the middle, or at random? Point to the plot of the band, the true path and the model's mean path.
  `<details>`: the prior pulls the path toward its average, so the band sits below the truth in the
  first months and above it in the last eight (prototype FULL: weeks 5 to 23 and 122 to 152; at week 144
  the model's mean is 0.68 against a true 0.55). The model sees the decline (the ratio's HDI is below 1)
  but understates it (0.57 against 0.42), which makes its current ROAS too high, not too low. The
  effect is identified by the quarterly pushes: the sales jump after each push shrinks over time.
  Without the pushes a fading effect would look like a falling baseline, and the model could not tell
  them apart. What would change it: more pushes, a longer history after the decline, or a lift test now
  (stretch A).

*Provided after it:* the current-quarter ROAS table, static against drifting, next to the truth
(prototype FULL: drifting 2.53 [1.94, 3.15], static 3.88 [3.48, 4.28], truth 2.11), with the sentence
"Look at which interval contains the truth."

### Exercise 5 · Prophet's decomposition of sales (6 minutes)

*Provided before it:* one Markdown cell on Prophet (Taylor & Letham 2018: an additive model of a
piecewise-linear trend with automatic changepoints, Fourier seasonality and holiday effects) and on how
Robyn uses it: Robyn fits Prophet to the dependent variable and adds the trend, season and holiday
components to its regression as baseline variables; paid media are not Prophet regressors (only one-hot
factor variables are). Then the holidays table: `pd.DataFrame({"holiday": "peak_week", "ds": <date_week
of the rows with holiday == 1>})`.

- **Predict.** Prophet sees only dates and sales. The true baseline grows by $500 a week. Over the three
  years, will Prophet's trend grow faster, about the same, or slower?
- **Function.** `prophet_components(weekly: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame` with
  columns `trend`, `season`, `holiday` (dollars), one row per week in the order of `weekly`. Fit
  `Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
  holidays=holidays, uncertainty_samples=0)` on `ds = date_week`, `y = y`; `predict` on the same dates;
  `season` is Prophet's `yearly` column, `holiday` its `holidays` column.
- **Checkpoint.** Columns, 156 rows, finite; `holiday` is 0 in non-holiday weeks; the three components
  add up to Prophet's `yhat` from the reference fit inside the checkpoint to 1e-6 (prototype: 1.2e-10;
  three repeated fits were identical, Prophet's MAP fit starts from deterministic initial values).
- **Explain.** Compare with your guess (prototype: 149 a week against the true 500). *Provided:* a plot
  of Prophet's trend minus the true baseline trend next to social's true contribution, with the
  sentence "Look at whether the gap falls when social's contribution falls." Prototype: correlation 0.56
  (0.49 to 0.57 over seeds 2037 to 2041). Which part of sales did Prophet put into its trend?
  `<details>`: a fading channel lowers sales slowly, and a slow change is what a trend with changepoints
  is built to follow. Prophet cannot know that the fall came from social. Its holiday effect is also
  larger than the true 40,000 (prototype 53,600), because media spent in holiday weeks shows up there.

*Then fit 3 (Prophet baseline).*

### Exercise 6 · Did the Prophet baseline help? (7 minutes)

- **Predict.** With Prophet's components as an informed prior on the baseline, will social's all-period
  ROAS move toward the truth (4.0), away from it, or stay where the static model put it?
- **Function.** `baseline_comparison(roas_own: xr.DataArray, roas_prophet: xr.DataArray, true_roas:
  dict, prob: float = 0.94) -> pd.DataFrame`: per channel the posterior mean and 94% HDI under each
  baseline, the absolute error of each mean, and whether each HDI covers the truth. Both inputs are
  all-period ROAS draws from Exercise 1's function.
- **Checkpoint.** On the fits: for social, the Prophet-baseline mean is below the own-baseline mean, its
  94% HDI excludes the true all-period ROAS, and the own-baseline HDI covers it. Prototype, seed 2037
  (FULL): own 3.90 [3.53, 4.32], Prophet 2.90 [2.25, 3.58], truth 4.00; QUICK own [3.43, 4.31], Prophet
  [2.30, 3.66]. Over seeds 2037 to 2041 the Prophet baseline lowered social's ROAS in 5 of 5 and its HDI
  excluded the truth in 4 of 5.
- **Explain.** Compare with your prediction. Which channel lost credit, and where did that credit go?
  Point to Exercise 5's plot.
  `<details>`: Prophet's trend already contains the slow part of social's effect, so with a coefficient
  held near 1 the baseline keeps it and the MMM gives social less. Two more problems follow from
  fitting Prophet to the same sales the MMM explains: the MMM treats Prophet's components as known
  data, so the uncertainty in Prophet's own fit never reaches the ROAS intervals; and the intervals
  look reasonable while excluding the truth. In the prototype the intervals did not become narrower;
  they moved. What would change it: a baseline the MMM estimates jointly with the media (Module 8's
  trend and Fourier terms, or PyMC-Marketing's time-varying intercept), or Prophet told about spend
  (stretch B, which helped tv but not social in the prototype).

What the prototype supports, and only that, for the Markdown and the module page:

- On this retailer (seed 2037), the Prophet baseline moved social's all-period ROAS from 3.90 to 2.90
  (truth 4.00) and tv's from 1.73 to 1.44 (truth 1.60); search's interval widened (own 5.43 [0.99,
  12.98], Prophet 7.54 [1.96, 18.29], truth 3.0, FULL, target_accept 0.98).
- Over five seeds (MCMC, 500 to 1,000 draws), the MMM's own baseline was closer to the truth than the
  Prophet baseline for social in 5 of 5 seeds, for tv in 4 of 5 and for search in 4 of 5.
- In 30 seeds of a cheaper check (least squares with the true adstock, saturation and multiplier, so
  only the baseline differs): with Prophet's components fixed at coefficient 1 (the limit of a tight
  informed prior), ROAS was biased on average by −32% for tv, −62% for social and −21% for search; with
  free coefficients by −15%, −31% and +107%; with the true baseline terms by 0%, 0% and −2%. With
  spend added to Prophet as extra regressors and the components fixed, −12%, −43% and +50%. With a
  stiffer trend (`changepoint_prior_scale=0.005`), −32%, −57% and −31%.
- Without fatigue (all multipliers 1) the same check still gave −31% (tv), −57% (social) and −70%
  (search) with fixed components, so the absorption is not only the fatigue: Prophet's trend and its
  10-term yearly seasonality also take up part of the flights, pushes and demand-following spend.
- With a trend break added to the true baseline (growth stopping at week 78), which Module 8's linear
  `t` cannot follow, the linear trend left search at +29% and tv and social within 3%. Two Prophet
  variants brought search closer (spend as Prophet regressors +21%, stiffer trend −10%, both with fixed
  components), but every Prophet variant moved tv by −7 to −33% and social by −19 to −64% (least
  squares, 30 seeds). We did not find a case in this simulation where the Prophet baseline helped every
  channel; we tested only the cases listed here.

## Decision · Cut or keep social next quarter? (5 minutes)

Provided cell, in the standard order:

- **Number:** per channel, the drifting model's last-quarter ROAS (mean and 94% HDI) and the posterior
  probability that it is below break-even, computed draw by draw; beside it the static model's
  all-period ROAS with the same probability. Prototype FULL, social: drifting last quarter 2.53 [1.94,
  3.15], P(ROAS < 3.33) = 0.99 (QUICK 2.67 [1.97, 3.29], 0.96); static all period 3.90 [3.53, 4.32],
  P(ROAS > 3.33) = 0.997. Over seeds 2037 to 2041 the drifting model gave P(below) ≥ 0.96 in 4 of 5
  seeds and 0.84 in the fifth (which the rule calls undecided).
- **Rule:** "A dollar of sales earns the 30% gross margin, so break-even ROAS is 1 ÷ 0.30 ≈ 3.33. Cut a
  channel's budget next quarter if P(current-quarter ROAS < 3.33) ≥ 0.9; keep it if P(> 3.33) ≥ 0.9;
  otherwise run a lift test first (Module 9). We assume next quarter's effect equals last quarter's: the
  model's multiplier is not extrapolated, because a GP reverts to its prior mean outside the data
  (PyMC-Marketing's TVP notebook says so). Under saturation the last dollar returns less than the
  average dollar, so an average below break-even means the marginal dollar is below it too."
- **Recommendation:** the learner's sentence: what to do with social's budget (last quarter's spend in
  the data, prototype $192,407), how sure, how the answer differs from the static model's (keep), and
  what would change it (a creative refresh tested before spend returns; a lift test now, stretch A).
  Then the cell shows the truth (social's true last-quarter ROAS 2.11, below break-even). The cell
  prints the mode.

## Stretch (optional)

- **A · A lift test now.** Add `mmm_drift_lift_tests.csv` (one dated social test ending 2025-11-17) to
  the drifting model: `build_model`, then `add_lift_test_measurements(df)`, then fit. With a
  time-varying media model, PyMC-Marketing multiplies the saturation curve by the multiplier at the
  test's `date` (`create_time_varying_saturation`, `mmm/lift_test.py`), so a recent test pins the current
  level. Prototype (1000 draws, 29.6 s on the laptop, 0 divergences): social's last-quarter ROAS 2.23
  [1.96, 2.53] against 2.53 [1.94, 3.15] without the test (truth 2.11). The Technical Expert checks that
  the `date` column must equal a model date (it does in the prototype).
- **B · Tell Prophet about spend.** Refit Prophet with `add_regressor(c)` for each channel, rebuild the
  components and refit fit 3. Prototype, seed 2037: tv 1.51 [1.40, 1.64] (truth 1.60), social 3.16
  [2.64, 3.71] (truth 4.00): better for tv, still wrong for social, because a linear spend regressor
  cannot represent an effect that fades.

Nothing later depends on either. A third option was measured and left out: refitting with
`eta_upper=1.0` barely moved the answer (social 2.61 [1.99, 3.27], coverage 0.60).

## Expected run time (estimate, not a run record)

Measured on the laptop with the prototype (Apple Silicon, 2 chains on 2 cores, load average about 5 from
other jobs). The Colab column multiplies by 3.8, the ratio of Colab to laptop times for the MMM fits in
spike S2 (45.0 s against 11.9 s first fit, 9.7 s against 2.6 s again; DECISIONS.md). It is an estimate
until a Colab record exists.

| Step | Laptop FULL (s) | Laptop QUICK (s) | Colab FULL, estimated (s) |
|---|---|---|---|
| Install pymc-marketing and nutpie (Prophet 1.4.0 is preinstalled on Colab) | — | — | 35 to 53 (Modules 8 and 9 Colab records) |
| `import pymc_marketing` | 2 | 2 | 17 (S2) |
| Static fit | 22 | 11 | 85 |
| Prior predictive, 1,000 draws | 2 | 1 | 6 |
| Drifting fit (the longest cell) | 47 to 54 | 18 to 21 | 180 to 205 |
| Prophet import and fit | 0.2 | 0.2 | 1 to 5 |
| Prophet-baseline fit | 18 | 7 | 70 |
| Whole notebook, compute | about 90 | about 35 | about 6 to 7 minutes |
| Stretch A + B | 47 | — | about 3 minutes more |

Against the targets (install ≤ 3 min, any cell ≤ 4 min, compute ≤ 15 min): inside, with the drifting fit
the closest (estimated 3.0 to 3.4 minutes). Cut rule if a Colab record shows that cell above 210 s:
lower FULL draws for the drifting fit to 700 (500 draws took 16 to 29 s on the laptop over five seeds),
never `target_accept`.

## Known pitfalls

- `time_varying_media=True` is one multiplier for all channels; per-channel drift needs the `SoftPlusHSGP`
  instance with `dims=("date", "channel")`.
- The default `media_tvp_config` (HSGPKwargs m=200, InverseGamma lengthscale with mean 5 weeks) is far
  wigglier than a fatigue path; the lab sets its own priors through `parameterize_from_data`.
- The multiplier is normalized to mean 1 over the fitted weeks: compare it with the truth's relative path
  (`multiplier_rel_<c>`), not the raw multiplier.
- Prophet logs every fit through `cmdstanpy` at INFO level: set
  `logging.getLogger("cmdstanpy").setLevel(logging.WARNING)`. Off Colab, importing Prophet without plotly
  logs "Importing plotly failed"; harmless.
- Prophet's components are in dollars and PyMC-Marketing does not scale controls: divide them by the
  maximum of `y` before passing them, or the informed prior means nothing.
- ArviZ 1.x defaults to 89% ETI; always pass the interval type and probability.
- Search's interval is wide in every fit (its spend follows demand, so the data separate its effect from
  the trend poorly); display's is wide because it is small. The decision is about social; the notebook
  says why the other intervals are wide instead of hiding them.

## APIs verified (and how)

Installed source in `.venv/lib/python3.13/site-packages`, 2026-10-10, and run in the prototype:

- pymc-marketing 1.2.0 `mmm/mmm.py`: `MMM(..., time_varying_media: bool | HSGPBase)`; with `True` the
  multiplier is built by `create_hsgp_from_config(X=time_index, dims=("date", *self.dims),
  config=model_config["media_tvp_config"])` and named `media_temporal_latent_multiplier`; the default
  `media_tvp_config` is `HSGPKwargs(m=200, L=None, eta_lam=1, ls_mu=5, ls_sigma=10, cov_func=None)`;
  with an `HSGPBase` instance, `register_data(time_index)` then `create_variable(...)`, and the
  contribution is `baseline_channel_contribution * media_latent_process`. The time index is
  `0, …, T−1`. `frozen_deterministics` lists `media_temporal_latent_multiplier_f_mean`.
- `mmm/hsgp.py`: `SoftPlusHSGP.create_variable` divides softplus(f) by its mean over the first dim;
  `HSGP.parameterize_from_data(X, dims, ls_lower, ls_upper, eta_upper, cov_func, drop_first=True, ...)`
  (InverseGamma lengthscale constrained to `[ls_lower, ls_upper]` with mass 0.9; Exponential amplitude
  with `P(eta > eta_upper) = 0.05`); scalar `ls` and `eta` are enforced by validators.
- `mmm/tvp.py`: `_create_hsgp_instance` uses Matérn 5/2 by default and `L = 2 × X_mid` when `L` is None.
- `mmm/incrementality.py`: windowed counterfactual definition (module docstring) and
  `contribution_over_spend(frequency, start_date, end_date, include_carryover=True, ...)`; works with the
  drifting model (prototype).
- `mmm/mmm.py` `add_lift_test_measurements` passes `time_varying_var_name="media_temporal_latent_multiplier"`
  when the media vary; `mmm/lift_test.py` `create_time_varying_saturation` multiplies the saturation
  curve by it.
- `sample_prior_predictive(X, y, samples=, random_seed=)` stores the multiplier in `idata["prior"]`
  (prototype). `idata.constant_data["target_scale"]` and `posterior["mu"]` exist on the fits.
- prophet 1.4.0 `forecaster.py`: `Prophet(growth="linear", n_changepoints=25, changepoint_range=0.8,
  yearly_seasonality="auto", ..., holidays=None, changepoint_prior_scale=0.05, uncertainty_samples=1000,
  ...)`; `predict()` returns `trend`, `yearly`, `holidays`, `additive_terms`, `yhat` with `yhat =
  trend * (1 + multiplicative_terms) + additive_terms`; `add_regressor`. Yearly seasonality with
  `True` uses 10 Fourier terms (prototype). Colab preinstalls prophet 1.4.0, cmdstanpy 1.3.0 and holidays
  0.105 (`environment/colab-constraints.txt`).

## Prototype results (for the record)

All on the stand-in generator, laptop, 2026-10-10; not run records.

| Seed | Static: social last quarter (HDI, truth) | Drifting: social last quarter, P(< 3.33) | Path coverage social | Drift ratio HDI (truth 0.42) |
|---|---|---|---|---|
| 2037 FULL | 3.88 [3.48, 4.28], 2.11 | 2.53 [1.94, 3.15], 0.99 | 0.69 | [0.43, 0.71] |
| 2037 QUICK | 3.88 [3.42, 4.32], 2.11 | 2.67 [1.97, 3.29], 0.96 | 0.65 | [0.45, 0.75] |
| 2038 (500 draws) | 4.35 [3.85, 4.87], 2.07 | 2.56 [2.03, 3.18], 0.99 | 0.97 | [0.38, 0.61] |
| 2039 | 3.80 [3.20, 4.57], 2.16 | 2.22 [1.60, 2.83], 1.00 | 1.00 | [0.32, 0.57] |
| 2040 | 4.62 [4.08, 5.18], 2.11 | 2.93 [2.23, 3.66], 0.84 | 0.57 | [0.46, 0.72] |
| 2041 | 4.05 [3.49, 4.54], 2.16 | 2.49 [1.86, 3.22], 0.99 | 1.00 | [0.34, 0.59] |

Seed 2037: target_accept 0.98 for both models. Seeds 2038 to 2041: 500 draws, the static model at
target_accept 0.95, the drifting model at 0.98.

Earlier designs that were dropped: with social pushes every 18 weeks at ×2.0 (spend noise sd 0.2,
amplitude prior `eta_upper=1`), the drifting model was undecided on social (P(below) 0.51, 0.62 and 0.77)
in 3 of 5 seeds. Quarterly pushes at ×2.5, spend noise sd 0.3 and `eta_upper=2` gave the results above.
Sharper declines (logistic width 10 or 15 weeks instead of 20) were also tried on that earlier design;
the band covered the true path in 73 to 86% of weeks, and the slow decline was kept because fatigue is
meant to be slow.

## References (verified)

- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.
  <https://doi.org/10.1080/00031305.2017.1380080>\
  *Verified (2026-10-10)* through the Crossref record and the Semantic Scholar abstract; the publisher
  page refused automated access (403) and the PeerJ preprint page also returned 403. Supports: Prophet is
  a modular, interpretable regression model for forecasting with analyst input (the abstract); Prophet's
  README cites this paper.
- Prophet documentation, *Trend changepoints*.
  <https://facebook.github.io/prophet/docs/trend_changepoints.html>\
  *Verified (2026-10-10).* "By default, Prophet specifies 25 potential changepoints which are uniformly
  placed in the first 80% of the time series"; `changepoint_prior_scale` defaults to 0.05.
- Prophet documentation, *Seasonality, holiday effects, and regressors*.
  <https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html>\
  *Verified (2026-10-10).* Holidays are a table with columns `holiday` and `ds`; `add_regressor` adds a
  regressor to the linear part, which "must be known for both the history and for future dates".
- Robyn source, `R/R/inputs.R` on `main`, which is version 3.12.1 (`R/DESCRIPTION`, read the same day).
  <https://raw.githubusercontent.com/facebookexperimental/Robyn/main/R/R/inputs.R>\
  *Verified (2026-10-10).* `prophet_decomp()` fits Prophet to the dependent variable (`rename("y" =
  "dep_var")`) and returns the `trend`, `season`, `monthly`, `weekday` and `holiday` components it was
  asked for; `add_regressor` is called only for one-hot-encoded `factor_vars`, so paid media are not
  Prophet regressors; `robyn_inputs()` sets `all_ind_vars <- c(tolower(prophet_vars), context_vars,
  all_media)`, so the components enter the model as independent variables. The `prophet_vars` argument
  documentation recommends "trend", "season", "holiday" for weekly data.
- Robyn documentation, *Features*. <https://facebookexperimental.github.io/Robyn/docs/features/>\
  *Verified (2026-10-10).* "Robyn provides trend, season, holiday & weekday decomposition out of the box
  using Prophet"; trend, season and holiday are treated as baseline ("non-steerable") variables.
- PyMC-Marketing documentation, *MMM with time-varying media baseline*.
  <https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_time_varying_media_example.html>\
  *Verified (2026-10-10).* `time_varying_media=True`; the notebook assumes "all channels share the same
  time-dependent fluctuations" and models the shared multiplier with an HSGP. (The page's metadata shows
  version 1.0.0.dev0; the lab follows the installed 1.2.0 source.)
- PyMC-Marketing documentation, *MMM with time-varying parameters (TVP)*.
  <https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_tvp_example.html>\
  *Verified (2026-10-10).* A time-varying intercept as a GP; "Out-of-sample, the GP reverts to its prior
  mean"; for trends and seasonality the page recommends a linear or Fourier basis instead of a GP.
- Riutort-Mayol, G., Bürkner, P.-C., Andersen, M. R., Solin, A., & Vehtari, A. (2023). Practical Hilbert
  space approximate Bayesian Gaussian processes for probabilistic programming. *Statistics and
  Computing*, 33, 17. <https://doi.org/10.1007/s11222-022-10167-2>\
  *Verified (2026-10-10)* through the Crossref record (published online 2022-12-14, issue dated February
  2023). The HSGP approximation PyMC-Marketing uses; its basis-size heuristics are what
  `parameterize_from_data` applies (`approx_hsgp_hyperparams` docstring).

## Open items for the Technical Expert

- Write `src/mktstats/synth/mmm_drift.py` to the specification above, register it in `EXTENSIONS`,
  write the three CSVs and `truth_mmm_drift.json`, and add the measured `tolerances`. The prototype
  generator (`gen.py` in the Academic Director's session scratchpad, temporary) can be read for
  reference but is not to be copied as is: fix the draw order and document it.
- Re-measure every checkpoint above on the committed data in QUICK and FULL, and change a threshold only
  with the measured number written into the brief and the truth file's `tolerances`.
- Check that `mktstats.data.repo_file` finds the extension files in a checkout, offline from the cache,
  and on Colab at `repo.ref` (they exist at a tag only after the next release).
- Window ROAS for the truth is computed from the noiseless contributions; confirm `roas_last_quarter`
  equals `window_roas` on `latent`, and the counterfactual variant equals
  `contribution_over_spend(frequency="all_time", start_date=, end_date=)` on noiseless data with the true
  parameters, as Module 8 did.
- Confirm the HSGP prior settings (`m = 56`, `L = 426.4`) and the drifting fit's time on Colab; apply the
  cut rule above if needed.
- `modules.m17.deps` (yours) should list the three CSVs and `truth_mmm_drift.json`, not `truth.json`.
