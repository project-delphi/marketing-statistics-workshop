# Lab brief · Module 17 · MMM with drifting effects and a Prophet baseline

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. Written before the lab existed, from a throwaway prototype on a laptop (a stand-in generator with the design below, pymc-marketing 1.2.0, nutpie, 2 chains on 2 cores, a machine shared with other jobs). Reviewed by the Pedagogy Expert on 2026-10-10 against `briefs/_lab-standard.md` ("Pedagogy review" at the end). Reconciled with the build on 2026-10-10 ("Build reconciliation" at the end): the committed generator reproduces the prototype's data exactly at seed 2037, so numbers the build re-measured on the committed data are now labeled *committed data* (FULL unless QUICK is said), and numbers the build did not re-run keep the label *prototype*. The lab has run on a laptop and in Docker (run records of 2026-10-10), not on Colab. |
| Notebook | `labs/src/17-mmm-time-varying.py` → `labs/python/17-mmm-time-varying.ipynb` |
| Lab slot | 55 min (`modules.m17.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (7 + 8 + 6 + 9 + 10; limit 40) + decision 5 + slack 5 = 55. Each exercise's minutes include reading the provided cells immediately before it and waiting for any fit among them (Colab FULL estimates, "Parts and exercises"). Minutes are estimates until a pilot. |
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
  window, carry-in kept); the two differ slightly for a window shorter than the history (committed data,
  social, last quarter: truth 2.11 by the window definition, 2.04 by the counterfactual one; the drifting
  model, FULL, 2.53 and 2.45). The lab uses the window definition, which one function computes for the truth and for
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

Committed data (seed 2037, `truth_mmm_drift.json`; the prototype drew the same values): mean weekly spend
tv $32,101, search $23,084, social $15,533, display $6,417; mean weekly sales $478,188; media 39% of
sales. Social's true ROAS: 4.00 over all weeks, 5.00 in the first year, 2.11 in the last quarter (drift
ratio 0.42); social spend in the last quarter $192,407.

## Model fits (once each, provided cells)

All three: Module 8's specification (`GeometricAdstock(l_max=8)`, `LogisticSaturation()`, spend-share
HalfNormal priors on `saturation_beta`), `nuts_sampler="nutpie"`, `chains=2`, `draws=tune=D`,
`target_accept=0.98`, `random_seed=SEED` (2037), `progressbar=False`. QUICK: D = 300; FULL: D = 1000.
With 0.98 the build had 0 divergences in all three fits and both stretch fits, FULL and QUICK, on the
committed data, and 1 in the 45 fits of its 15-seed sweep (seed 2050, drifting fit). The prototype's
reason for 0.98 was not re-measured: at 0.95 its static FULL fit had 1 divergence, and drifting fits of
earlier designs had 3 to 4 (Module 9 made the same choice for the same reason).

1. **Static.** Controls `price_index, holiday, t`, `yearly_seasonality=2` (Module 8).
2. **Drifting.** The static specification plus
   `time_varying_media=SoftPlusHSGP.parameterize_from_data(X=np.arange(156), dims=("date", "channel"),
   ls_lower=20, ls_upper=104, eta_upper=2.0, cov_func="matern52")`. That call gives $m = 56$ basis
   functions and $L = 426.4$ (measured in the prototype and the build), of which PyMC-Marketing keeps 55
   per channel (`drop_first=True` drops the first basis function; checked on a FULL refit, 2026-10-10). `sample_prior_predictive(X, y, samples=1000,
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

## Terms the notebook defines, where they are first used

The Notation above is for the Technical Expert. The notebook defines each term in one plain sentence in
the provided cell or exercise that first uses it, never in an opening glossary; the briefing has defined
them first. Suggested wording:

| Where | Term | One sentence for the notebook |
|---|---|---|
| Part A, first cell | ad fatigue (wear-out) | A channel's effect per dollar falls over time: the same creative shown too often, an audience used up, a platform change. |
| Part A, first cell | effectiveness multiplier, relative path | How strongly a channel works in a week compared with its own average over the three years: 1 is average, 0.5 is half as well. The plot of the true relative paths shows it. |
| Before Exercise 1 | adstock carryover (recall, Module 8) | Part of a week's effect lands in later weeks. |
| Before Exercise 1 | ROAS over a window | The sales a channel produced in those weeks per dollar it spent in those weeks; carryover from earlier spend counts, carryover after the window does not. Over all weeks it is Module 8's ROAS. |
| Before Exercise 1 | break-even ROAS | The ROAS at which the gross margin on the sales a channel produced pays for its spend: 1 ÷ 0.30 ≈ 3.33 at the retailer's 30% margin. |
| Run · static fit | static MMM | Module 8's model: one effect size per channel, the same in every week. |
| Run · static fit | 94% HDI (recall, Module 8) | The narrowest interval that holds 94% of the posterior draws. |
| Exercise 2 | residual | Actual sales minus the model's fitted sales (posterior mean), in dollars. |
| Part C, first cell | drifting MMM | The same model with each channel's effect multiplied by its own multiplier $m_c(t)$ in every week. |
| Part C, first cell | Gaussian process prior | A prior over smooth curves: each draw from it is a whole curve. The plot of prior paths right after shows 20 of them. |
| Part C, first cell | lengthscale | Roughly how many weeks a curve takes to change appreciably. Set here so it changes over months, not weeks. |
| Part C, first cell | amplitude | How far a curve can move away from its average. |
| Part C, first cell | mean 1 | The model's multiplier averages 1 over the three years, like the true relative path, so the two can be compared directly. |
| Exercise 3 | prior draws (recall, Module 8's prior predictive check) | Draws from the model before it has seen any sales. |
| Exercise 3 | drift ratio | The multiplier's average in the last quarter divided by its average in the first year: the share of its first-year effect a channel keeps. |
| Exercise 4 | coverage of a band | The share of weeks in which the true value lies inside that week's 94% HDI. |
| Part D, first cell | Prophet | A forecasting model that adds a trend, a yearly seasonal pattern and holiday effects, fitted to dates and sales only. |
| Part D, first cell | changepoint | A week where Prophet's trend may change slope; Prophet places 25 candidates in the first 80% of the history and keeps the slope changes small unless the data insist. |
| Part D, first cell | baseline | The part of sales the MMM credits to everything except media: trend, season, holidays, price. |
| Before Exercise 5 | informed prior | A prior centered on what we already believe: here each Prophet component enters at its own size (coefficient 1), give or take about 20%. |
| Decision | lift test (recall, Module 9) | A randomized test that changes a channel's spend for some weeks or regions and measures the sales that change with it. |

**What stays out of learner-facing text.** The learner needs three things of the drifting model: each
channel gets a multiplier per week with average 1; it can only change slowly; and how far it can move is a
prior, checked before the fit in the decision's own units (the share of first-year effect kept). Everything
else goes in code comments (one line each, as the reason for a setting), a `<details>` block or the
briefing, and no Predict or Explain prompt asks about it. The Academic Director's definitions, for those
places:

- A *Gaussian process* (GP) is a prior over smooth functions; its *lengthscale* $\ell$ is roughly how many
  weeks it takes the function to change appreciably, and its *amplitude* $\eta$ how far it can move. An
  *HSGP* (Hilbert-space GP) approximates it with a fixed set of basis functions so it samples fast
  (Riutort-Mayol et al.). `SoftPlusHSGP` passes the GP through a softplus (always positive) and divides by
  its mean over the fitted weeks, so the multiplier is positive with mean 1. The priors set here: $\ell$
  InverseGamma with 90% of its mass between 20 and 104 weeks (slow drift, not week-to-week noise); $\eta$
  Exponential with a 5% chance of exceeding 2.
- Code comments for the drifting model's cell (one line each): `SoftPlusHSGP` "a fast approximation of a
  Gaussian process, kept positive and scaled to average 1 over the fitted weeks"; `dims=("date",
  "channel")` "one curve per channel (`time_varying_media=True` would give one curve shared by all
  channels)"; `ls_lower=20, ls_upper=104` "the curve changes over months, not weeks: 90% prior probability
  that the lengthscale is 20 to 104 weeks"; `eta_upper=2.0` "how far a curve can move: a 5% prior
  probability that the amplitude exceeds 2"; `cov_func="matern52"` "how smooth the curve is; the choice
  PyMC-Marketing makes for its own `time_varying_media=True`" (`mmm/tvp.py` falls back to Matérn 5/2;
  `parameterize_from_data`'s own default is `ExpQuad`, `mmm/hsgp.py` line 780, pymc-marketing 1.2.0,
  checked 2026-10-10). The basis size and boundary ($m = 56$, $L = 426.4$) are printed in the maintainer
  header, not shown to the learner.
- The window definition of ROAS against PyMC-Marketing's counterfactual one (Notation) gets one sentence in
  a `<details>` block under the static model's check: over all weeks the two agree, which that check
  confirms; for one quarter they differ slightly, and the lab uses the window definition throughout.

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · ROAS depends on the window | *Run: data, spend and sales, the true relative paths (provided)* | — |
| | 1 · ROAS over a window | 7 |
| B · What the static model says about now | *Run: static fit (provided), its ROAS by window, labeled check* | — |
| | 2 · Residuals that drift | 8 |
| C · Letting the effect drift | *Run: the drifting model, 1,000 prior draws, 20 prior paths plotted (provided)* | — |
| | 3 · How much drift does the prior allow? | 6 |
| | *Run: drifting fit (provided; the longest cell)* | — |
| | 4 · Does the band cover the true path? | 9 |
| | *Run: drift ratios against the truth (labeled check); current-quarter ROAS, static against drifting (provided)* | — |
| D · A Prophet baseline | *Run: Prophet's decomposition of sales (provided function, labeled check, two plots)* | — |
| | *Run: MMM with Prophet's components as an informed prior (provided)* | — |
| | 5 · Did the Prophet baseline help? | 10 |
| **Exercises** | | **40** |
| Decision | Cut or keep social next quarter? | 5 |

The first draft had six exercises (7 + 7 + 5 + 8 + 6 + 7 = 40) and a Prophet exercise between Exercises 4
and 5 above. With the provided cells read and the fits waited for, a realistic count was 46 to 50 minutes
(Pedagogy review, item 1), so Prophet's decomposition is now a provided cell and the hands-on Prophet work is
stretch B.

**Minute marks for the room** (Colab FULL, compute estimated): open and install 0–5; Exercise 1 5–12;
Exercise 2 12–20; Exercise 3 20–26; Exercise 4 26–35; Exercise 5 35–45; Decision 45–50; slack 50–55.

**Where learners wait, and what they do meanwhile.** Compute counts against the exercise whose minutes it
falls in. The Colab times are the brief's estimates (laptop × 3.8, "Expected run time"); QUICK times are the
laptop QUICK times × 3.8, also estimates. Each fit cell's Markdown says what to do while it runs:

| Wait (Colab FULL, estimated; QUICK in brackets) | In the minutes of | What the notebook asks the learner to do meanwhile |
|---|---|---|
| Install and `import pymc_marketing`, about 1 minute | Open | Read the header and Part A's first cell. |
| Static fit, about 1.5 minutes (0.7) | Exercise 2 | Read Exercise 2's Predict, write the two signs, and start writing `residual_spend_corr` (its toy check needs no fit). |
| 1,000 prior draws, seconds | Exercise 3 | Nothing; read the prior paths plot. |
| Drifting fit, about 3 to 3.5 minutes (1.1 to 1.3) | Exercise 4 | Read Exercise 4's Predict, write the guess, and write `path_coverage` (its toy checks need no fit). Run its cells when the fit has finished. |
| Prophet's fit, seconds; Prophet-baseline fit, about 1.2 minutes (0.5) | Exercise 5 | Read Exercise 5's Predict, write the guess, and write `baseline_comparison`. |

Total compute about 6 to 7 minutes of the 40 (estimate), none of it idle if learners follow these notes.

Names the lab uses: `weekly`, `latent`, `truth`, `CHANNELS`, `SEED = 2037`, `X`, `y`, `spend` (DataArray
`(date, channel)`), `true_contrib` (DataArray `(date, channel)` of the true contributions in dollars, built
from `latent` in a provided cell), `WINDOWS` (dict of `(start, end)` date strings), `GROSS_MARGIN`,
`BREAK_EVEN`, `mmm_static`/`idata_static`, `mmm_drift`/`idata_drift`, `prior_mult` (the prior draws of the
multiplier), `mmm_prophet`/`idata_prophet`, `holidays`, `components` (Prophet's, from the provided
`prophet_components`), `roas_draws_*`, `fading` (the channels whose drift ratio's 94% HDI lies below 1;
Decision). Every Predict ends with "Write your guess in the cell below." Every plot with more than one line
tells them apart by line style or marker as well as color.

### Exercise 1 · ROAS over a window (7 minutes)

*Provided before it* (read inside the 7 minutes): load the data and build `spend` and `true_contrib`; the
spend and sales plot; then the true relative paths from `latent` with one sentence: "Look at social's line:
its effect per dollar holds for a year, then falls to under half." One Markdown cell defines ROAS over a
window and the break-even ROAS (Terms) and names the three windows in `WINDOWS`.

- **Predict.** Social's true ROAS over all three years is 4.0, above the break-even ROAS of 3.3. Will its
  ROAS in the last quarter be above break-even, between 2.5 and 3.3, or below 2.5? Write your guess in the
  cell below.
- **Function.** `window_roas(contrib: xr.DataArray, spend: xr.DataArray, start: str, end: str) ->
  xr.DataArray`. `contrib` has dims `(..., date, channel)` in dollars (any leading dims, such as `chain`
  and `draw`); `spend` has dims `(date, channel)`; weeks from `start` to `end` inclusive
  (`.sel(date=slice(start, end))`); returns dims `(..., channel)`. The same function is used on the truth
  now and on posterior draws later. Hints in the stub: select the window's weeks in both arrays with
  `.sel(date=slice(start, end))` (both ends included); sum each over `"date"` by name (`.sum("date")`);
  divide. xarray matches the remaining dims by name, so leading dims survive. The reference is three lines.
- **Checkpoint.** First checkpoint, before any fit (reading CSVs only). (1) A toy with contribution = 2 ×
  spend gives 2 for every channel. (2) A toy with leading dims `(2, 3)` returns shape `(2, 3, 4)`.
  (3) A one-channel toy that tells the likely mistakes apart (computed by the Pedagogy Expert,
  2026-10-10): four weeks from 2023-01-02, spend 4, 3, 2, 2, contribution 1, 1, 1, 5, window 2023-01-09 to
  2023-01-23 gives (1 + 1 + 5) ÷ (3 + 2 + 2) = 1.0 exactly. Toy (1) cannot catch the commonest mistake, a
  mean of weekly ratios, because every weekly ratio there is 2; this one can. (4) On the truth
  (`true_contrib`), all three windows equal `truth_mmm_drift.json`'s `roas`, `roas_first_year` and
  `roas_last_quarter` to a relative 1e-9.
- **Failure messages** (the four parts of `_lab-standard.md` §5, each ending with "To move on now, run
  `workshop.use_reference(1)`"):
  - Toy (3) gives 1.111: "you averaged the weekly ratios. ROAS over a window is total sales over total
    spend: sum each over the window's weeks, then divide. (On the real data the weekly ratio is infinite or
    undefined in tv's weeks off air, when its spend is 0.)"
  - Toy (3) gives 0.400: "the window's last week is missing. `.sel(date=slice(start, end))` includes both
    ends; a `<` comparison on the end date drops it." 1.500: the same for the first week.
  - Toy (3) gives 0.727: "your function ignores `start` and `end`; select the window's weeks first."
    0.636: "you divided the window's sales by spend over all weeks; select the same weeks in `spend`."
  - Toy (2) has the wrong shape, or a scalar: "sum over `"date"` by name, not over an axis number, so that
    `chain` and `draw` stay."
  - Toys pass but the truth differs (the message prints the window, the channel, both values): "check
    that you pass `WINDOWS[...]`'s dates unchanged, as strings."
- **Explain.** Compare with your guess (committed data: 2.11, below 2.5). Which weeks dominate an all-period
  average: the many good weeks or the few recent ones? Point to the plot.
  `<details>`: the average weights each week by its spend; two good years outweigh one bad quarter, so
  the average stays above break-even long after the channel stopped paying back. A decision about next
  quarter needs the recent window.

*Provided after it · Run: the static fit* (fit 1), read and waited for inside Exercise 2's minutes. Its
Markdown says, before the code: what the model is (Module 8's static MMM, Terms); that it takes about a
minute and a half on Colab (an estimate until a Colab record exists); "While it runs, read Exercise 2's
Predict, write your two signs, and start writing `residual_spend_corr`: its toy check needs no fit"; and the
standard QUICK sentence ("With QUICK on, intervals are wider and may not match the module page; a decision
near its threshold can flip. That is a lesson about sample size, not a bug.").

*Then the labeled check "static model" (no exercise):* `window_roas` on
`idata_static.posterior["channel_contribution"] * target_scale` (`target_scale` from
`idata_static.constant_data`). Checks: (a) over all weeks it equals
`mmm_static.incrementality.contribution_over_spend(frequency="all_time")` to 1e-8 (build: largest
difference 3.2e-14 FULL, 4.3e-14 QUICK), which confirms the function on draws; (b) social's last-quarter
ROAS 94% HDI lies entirely above the true last-quarter ROAS. Committed data, FULL: all weeks 3.90, 94%
HDI [3.52, 4.32] (truth 4.00); last quarter 3.88 [3.48, 4.28] (truth 2.11); QUICK last quarter [3.42,
4.31]. In 15 of 15 generator seeds (2037 to 2051, QUICK) the truth lay below the static model's
last-quarter HDI (prototype: 5 of 5, seeds 2037 to 2041). A table shows each channel's
all-period and last-quarter ROAS with 94% HDIs next to the truth; the sentence before it: "Look at
social's two rows: the static model gives about the same number for both windows." Under it, the
`<details>` sentence on the counterfactual definition (Terms, "What stays out").
Messages: (a) "Your `window_roas` passed Checkpoint 1 but differs from PyMC-Marketing's
`contribution_over_spend` on the posterior draws by [largest gap]. Check that it sums over `"date"` by name
and keeps `chain` and `draw`. To move on now, run `workshop.use_reference(1)` and rerun from Checkpoint 1."
(b) "This compares the fit with the truth, not your code. With QUICK on the interval is wider; if it fails
in a FULL run, note the message and tell the instructor."

### Exercise 2 · Residuals that drift (8 minutes)

On real data there is no truth. This exercise finds the drift from the static model alone. The 8 minutes
include the static fit and its table before it (about 1.5 minutes of fit on Colab, estimated, during which
the learner reads this Predict and starts the function). One sentence defines the residual (Terms); a
provided cell computes `resid` and `year` before the Predict.

- **Predict.** The static model gives social one effect size for all three years. In 2023, when social
  worked better than its three-year average, will the residuals be positively or negatively correlated
  with social's weekly spend? And in 2025? Write two signs in the cell below. (Committed data, FULL: + in
  2023, +0.40; − in 2025, −0.57. Which channel fades is no longer the question: the true paths in Part A
  show it.)
- **Function.** `residual_spend_corr(resid: pd.Series, spend: pd.DataFrame, year: pd.Series) ->
  pd.DataFrame`: index year, one column per channel, each the Pearson correlation between the residuals
  and that channel's weekly spend within the year. Provided: `resid = y − (posterior mean of
  idata_static.posterior["mu"]) × target_scale`, `year = weekly["date_week"].dt.year`. Hint in the stub:
  `spend_part.corrwith(resid_part)` gives one Pearson correlation per column (pandas 2.2.3, `method`
  defaults to `"pearson"`); select one year's rows with `year == y`. The three inputs share `weekly`'s
  index.
- **Checkpoint.** (1) Exact against `np.corrcoef` on a toy (computed by the Pedagogy Expert, 2026-10-10):
  `year` = 2023 × 4, 2024 × 4; `spend` a = 1, 2, 3, 4, 1, 2, 3, 4 and b = 4, 1, 3, 2, 2, 2, 1, 5; `resid` =
  1, 2, 3, 10, 4, 3, 2, 1. Right answer, to 1e-12: 2023 a 0.8854, b −0.3162; 2024 a −1.0, b −0.5963.
  (2) On the static fit, equals the reference to 1e-12. (3) Social's correlation falls most from the first
  year to the last, and its last-year correlation is below −0.3. Committed data, FULL: social +0.40 in
  2023, +0.19 in 2024, −0.57 in 2025 (QUICK +0.405, +0.196, −0.569); the other channels' changes from
  2023 to 2025 were at most +0.28 (tv). Over 15 generator seeds (QUICK) social fell most and ended below
  −0.3 in 15 of 15, and its 2023 correlation was above +0.1 in 15 of 15, so the Predict keeps the two
  signs. (Prototype, seeds 2037 to 2041: social fell by 0.75 to 1.19 and ended at −0.51 to −0.67; the
  largest change for another channel was +0.49.)
- **Failure messages** (four parts, each ending with `workshop.use_reference(2)`):
  - One row (a 0.369, b −0.312): "you correlated over all weeks at once. Compute one correlation per year,
    from that year's rows only."
  - 2023 a = 1.0 and b = −0.4: "that is a rank (Spearman) correlation. Use Pearson, the default of
    `corrwith` and of `np.corrcoef`."
  - 2023 a = 4.667: "that is a covariance. A correlation divides it by both standard deviations;
    `corrwith` does that for you."
  - Index a, b instead of 2023, 2024: "rows are years and columns are channels; transpose."
  - (3) fails after (1) and (2) pass: "Your function is right (it matched `np.corrcoef` and the reference).
    This compares the static fit with the drift built into the data, not your code. With QUICK on the
    numbers move; if it fails in a FULL run, note the message and tell the instructor."
- **Explain.** Compare with your two signs. Why does a positive correlation in 2023 mean the model gives
  social too little credit then, and a negative one in 2025 too much? Point to social's column in the
  yearly table, and to one other channel's column for contrast.
  `<details>`: with one effect size for all weeks, the model fits the average; in early weeks sales jump
  more than the model predicts when social spends (positive residuals in push weeks), in late weeks less
  (negative). A sign change in a channel's residual correlation is a cheap warning to run before
  believing an all-period ROAS. What would change it: spend that never varies within a year leaves
  nothing to correlate.

*Provided before it (Part C's first cells, read inside Exercise 3's 6 minutes):*

1. One Markdown cell, at most eight sentences: the drifting MMM multiplies each channel's effect by its
   own multiplier in every week, the quantity whose true value Part A plotted; the multiplier's prior is a
   Gaussian process, a prior over smooth curves; its lengthscale and amplitude (one sentence each, Terms);
   it averages 1 over the three years, like the true relative path. No other GP term (Terms, "What stays
   out").
2. The drifting model built, with the code comments listed under Terms, and `sample_prior_predictive`
   (1,000 draws, seconds).
3. A plot of 20 prior draws of social's multiplier over the 156 weeks, the true relative path dashed,
   a horizontal line at 1. Sentence before it: "Look at how far the prior's curves wander from 1, and
   how fast they change." It is the definition of a Gaussian process prior for this audience, and the
   reference the Predict needs.

### Exercise 3 · How much drift does the prior allow? (6 minutes)

- **Predict.** Over all 1,000 prior draws, in what share will a channel keep half or less of its
  first-year effect in the last quarter (a drift ratio of 0.5 or less): under 3%, 3 to 20%, or over 20%?
  Use the plot of 20 curves to estimate it. Write your guess in the cell below. (Committed data: 8.8%, as
  in the prototype; over 15 generator seeds 8% to 12%, never within 2 points of a boundary.)
- **Function.** `drift_ratio(mult: xr.DataArray, early: tuple[str, str], late: tuple[str, str]) ->
  xr.DataArray`: the mean of `mult` over the `late` weeks divided by its mean over the `early` weeks;
  dims `(..., channel)`. Hint in the stub: the same window selection as Exercise 1, with
  `.mean("date")`.
- **Checkpoint.** (1) A constant multiplier gives 1. (2) A toy linear path gives its known ratio:
  $m(t) = 2 - t/155$, $t = 0, \dots, 155$, on `weekly`'s dates, with the windows `WINDOWS["first_year"]`
  (52 weeks) and `WINDOWS["last_quarter"]` (13 weeks), gives 0.56591 (computed by the Pedagogy Expert,
  2026-10-10). (3) On the prior draws (`idata["prior"]["media_temporal_latent_multiplier"]`, `prior_mult`)
  the result has dims `(chain, draw, channel)` and is positive and finite. *Build (checkpoint fixes,
  2026-10-10):* (4) a spike toy (1 every week, 14 in the last week: mean ratio 2.0, median ratio 1.0)
  catches a ratio of medians.
- **Failure messages** (four parts, each ending with `workshop.use_reference(3)`):
  - Toy (2) gives 1.767: "early and late are swapped: divide the late mean by the early mean."
  - Toy (1) gives 0.25 (and toy (2) 0.141): "you used sums. The windows have 52 and 13 weeks, so their
    sums differ by a factor of 4 even when nothing changes; use means."
  - Toy (2) gives 0.56667: "each window lost its last week. `.sel(date=slice(start, end))` includes both
    ends."
  - (3) has no `chain` or `draw`: "average over `"date"` only, so that every prior draw keeps its own
    ratio." NaN: "a window selected no weeks; pass the dates in `WINDOWS` unchanged."
- *Provided after it:* the share of prior draws with social's ratio at most 0.5, at most the true ratio
  (the truth's `drift_ratio`, printed beside it), and the prior's 94% equal-tailed interval (committed data,
  social: P(ratio ≤ 0.5) = 8.8%, P(ratio ≤ 0.42) = 7.4%, 94% equal-tailed interval 0.22 to 5.12;
  prototype only, not re-run: with `eta_upper=1.0`, P(ratio ≤ 0.42) was 2.4%).
- **Explain.** Compare with your guess. Social's true drift ratio is printed above. Does this prior rule
  out a fall that large, allow it but treat it as unusual, or expect it? Point to the printed share.
  `<details>`: it allows the fall but treats it as unusual (committed data: 7.4% of prior draws at or below
  the true 0.42), so the data have to show it. A prior that almost never allows a large fall cannot find one;
  a prior that allows anything follows noise. The lengthscale prior keeps the path smooth (changes over
  months, not weeks); the amplitude prior keeps it near 1 unless the data insist. This is Module 8's prior
  predictive check applied to the drift; it takes seconds and catches a prior that rules the answer out
  before the longest fit of the lab. What would change it: a smaller amplitude; in the prototype (not
  re-run in the build), half the amplitude bound (`eta_upper=1.0`) gave 2.4%, and the refit barely moved
  social's answer (2.61 [1.99, 3.27] against 2.53 [1.94, 3.15]). The lab's `<details>` says only that a
  smaller bound puts less prior mass on large falls.

*Then the drifting fit (fit 2), the longest cell, read and waited for inside Exercise 4's minutes. Its
Markdown says it takes about three minutes on Colab (an estimate until a Colab record exists) and: "While it
runs, read Exercise 4's Predict, write your guess, and write `path_coverage`: its toy checks need no fit.
Run its cells when the fit has finished."*

### Exercise 4 · Does the band cover the true path? (9 minutes)

The 9 minutes include the drifting fit before it (about three minutes on Colab, estimated), during which
the learner reads this Predict and writes the function.

- **Predict.** A 94% band that is right covers the truth in about 94% of weeks. In what share of the 156
  weeks will social's true relative path lie inside the model's 94% HDI: 90% or more, 50 to 90%, or under
  50%? Write your guess in the cell below. (Committed data: 0.69 FULL, 0.65 QUICK, more than 5 points from
  either boundary; over 15 generator seeds, QUICK, 0.65 to 1.00.)
- **Function.** `path_coverage(mult_draws: xr.DataArray, true_path: np.ndarray, prob: float = 0.94) ->
  float`: `mult_draws` has dims `(chain, draw, date)` for one channel; per week, the HDI of the draws
  (`checks.interval(draws, prob, "hdi")`); returns the share of weeks whose true value lies inside. Hints
  in the stub: `mult_draws.isel(date=k).values` holds week k's draws, from both chains;
  `checks.interval` returns `(low, high)`; the reference is about five lines.
- **Checkpoint.** Toys first (they need no fit; computed by the Pedagogy Expert, 2026-10-10, with 2
  chains × 500 draws × 156 weeks):
  (a) draws tightly around a known path (Normal, sd 0.01, around a line from 1.0 to 0.4) give 1.0;
  (b) the same draws shifted by +5 give 0.0;
  (c) skewed draws: every week the same 1,000 exact quantiles of an Exponential(1), and a true value of
  0.01 and 2.5 in alternate weeks, give 1.0 with the 94% HDI ([0.0005, 2.805]) and 0.5 with any other
  interval (94% equal-tailed [0.031, 3.491], 89% HDI [0.0005, 2.203], 89% equal-tailed, ArviZ's default,
  [0.057, 2.892]); deterministic, so it cannot pass or fail by chance. *Build (checkpoint fixes,
  2026-10-10):* the true values now cycle 0.01, 2.5 and −0.5 (`TOY_C_TRUE`), so the correct coverage is
  2/3, an equal-tailed or 89% interval gives 1/3, and a Normal approximation (mean ± 1.88 sd), which
  contains −0.5 although every draw is above 0, gives 1.0 with its own message; a copy with identical
  chains tells a one-chain answer apart from a wrong interval;
  (d) toy (a)'s draws with the true path reversed in time give about 0.03 per week (4 of 156 weeks with
  seed 2037), but 0.94 with one interval pooled over all weeks.
  Then on the fit: (e) for each channel, equals the reference function's coverage exactly; (f) social
  ≥ 0.5 and tv, search and display ≥ 0.9 (`latent["multiplier_rel_<c>"]` is the truth). Committed data:
  social 0.69 (FULL) and 0.65 (QUICK); the other channels 1.0. The rule held in 15 of 15 generator seeds
  (QUICK; social 0.65 to 1.00). (Prototype, seeds 2037 to 2041, 500 draws: social 0.57 to 1.0, every
  constant channel at least 0.99.)
- **Failure messages** (four parts, each ending with `workshop.use_reference(4)`):
  - (a) gives 0.0: "you returned the share of weeks outside the band; return the share inside."
  - (a) gives 156: "you returned a count of weeks; divide by the number of weeks."
  - (c) gives 0.5: "the interval is not the 94% HDI. An equal-tailed interval (3rd to 97th percentile), or
    89% (ArviZ's default), misses one of the two true values in this skewed toy. Use
    `checks.interval(draws, prob, "hdi")` and pass `prob`."
  - (d) gives about 0.94: "you used one interval for all weeks together. Compute an interval for each week
    from that week's draws only."
  - (f) fails after the toys and (e) pass: "Your function is right. This compares the fit with the truth,
    not your code. With QUICK on, the band is rougher; if it fails in a FULL run, note the message and
    tell the instructor."
- *Provided after the checkpoint:* the plot of social's band, the true path and the model's mean path,
  with the sentence "Look at where the band misses the true path: which weeks, and on which side." The
  Explain needs this plot.
- **Explain.** Compare with your guess. Where does the band miss: at the two ends of the history, in
  the middle, or at random? Point to the plot of the band, the true path and the model's mean path.
  `<details>`: the prior pulls the path toward its average, so the band sits below the truth in the
  first months and above it in the last eight (committed data, FULL: the band misses 49 of 156 weeks, the
  truth lying above it in weeks 5 and 7 to 23 and below it in weeks 122 to 152; at week 144 the model's
  mean is 0.68 against a true 0.55, from a refit with the lab's settings that reproduced its printed
  numbers). The lab's `<details>` says "weeks 5 to 23"; week 6 is inside the band. The model sees the decline (the ratio's HDI is below 1)
  but understates it (0.57 against 0.42), which makes its current ROAS too high, not too low. The
  effect is identified by the quarterly pushes: the sales jump after each push shrinks over time.
  Without the pushes a fading effect would look like a falling baseline, and the model could not tell
  them apart. What would change it: more pushes, a longer history after the decline, or a lift test now
  (stretch A).

*Provided after it · labeled check "drifting model" (no exercise).* With Exercise 3's function on the
posterior: social's drift ratio 94% HDI lies entirely below 1, and the other three channels' HDIs contain
1. Committed data, FULL: drift ratio social 0.57 [0.43, 0.71] (truth 0.42), tv 1.03 [0.90, 1.13], search
0.96 [0.49, 1.44], display 1.12 [0.51, 1.76]; the check held in 15 of 15 generator seeds (QUICK;
prototype: 5 of 5). It
was part of Checkpoint 4 in the first draft; it moves here because it uses Exercise 3's function, which
already passed, so it can fail only through the fit. Message: "This compares the fit with the truth, not
your code. If it fails in a FULL run, note the message and tell the instructor." The cell prints the four
ratios with their HDIs and the true ratios, and sets `fading` (the Decision uses it).

*Then:* the current-quarter ROAS table, static against drifting, next to the truth
(committed data, FULL: drifting 2.53 [1.94, 3.15], static 3.88 [3.48, 4.28], truth 2.11), with the sentence
"Look at which interval contains the truth."

### Part D · Run · Prophet's decomposition of sales (provided, read inside Exercise 5's minutes)

The first draft's Exercise 5 (`prophet_components`, 6 minutes) becomes this provided cell: its function was
API recall (construct Prophet, fit, pick three columns of `predict()`), while its idea, that Prophet's trend
takes up part of a fading channel's effect, lives in the plot below and is what the new Exercise 5 tests.
Hands-on Prophet is stretch B.

1. One Markdown cell on Prophet (Taylor & Letham 2018: an additive model of a piecewise-linear trend with
   automatic changepoints, Fourier seasonality and holiday effects) and on how Robyn uses it: Robyn fits
   Prophet to the dependent variable and adds the trend, season and holiday components to its regression
   as baseline variables; paid media are not Prophet regressors (only one-hot factor variables are). It
   defines Prophet, changepoint and baseline (Terms) and stays under ten sentences.
2. The holidays table: `pd.DataFrame({"holiday": "peak_week", "ds": <date_week of the rows with holiday ==
   1>})`.
3. `prophet_components(weekly: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame`, provided, with
   columns `trend`, `season`, `holiday` (dollars), one row per week in the order of `weekly`. It fits
   `Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
   holidays=holidays, uncertainty_samples=0)` on `ds = date_week`, `y = y`; `predict` on the same dates;
   `season` is Prophet's `yearly` column, `holiday` its `holidays` column. `components =
   prophet_components(weekly, holidays)`.
4. Labeled check "prophet" (the first draft's Checkpoint 5): columns, 156 rows, finite; `holiday` is 0 in
   non-holiday weeks; the three components add up to Prophet's `yhat` from the same fit to 1e-6
   (build: 1.2e-10 on the committed data; prototype: three repeated fits were identical, Prophet's MAP fit
   starts from deterministic initial values). Message: "This checks the provided code, not yours. If it
   fails, note the message and tell the instructor."
5. Printed, with the sentence before it "Look at the trend's growth: the true baseline grows by $500 a
   week.": Prophet's trend growth per week over the three years (committed data: $149 a week) and its
   holiday effect against the true 40,000 (committed data: $53,641).
6. A plot of Prophet's trend minus the true baseline trend next to social's true contribution, with the
   sentence "Look at whether the gap falls when social's contribution falls." and the printed correlation
   (committed data: 0.56; prototype: 0.49 to 0.57 over seeds 2037 to 2041). Two panels or two line styles,
   not color alone.

*Then fit 3 (Prophet baseline).* Its Markdown first defines the informed prior (Terms) in two sentences:
each Prophet component enters with a coefficient whose prior is centered on 1 with sd 0.1, so the MMM may
rescale it by about ±20%; the media part stays static, so the only difference from the static fit is the
baseline. Then: about a minute on Colab (estimate), "While it runs, read Exercise 5's Predict, write your
guess, and write `baseline_comparison`."

### Exercise 5 · Did the Prophet baseline help? (10 minutes)

The 10 minutes include reading Part D's provided cells (about 3 minutes) and waiting for fit 3 (about a
minute on Colab, estimated), during which the learner reads the Predict and writes the function.

- **Predict.** With Prophet's components as an informed prior on the baseline, will social's all-period
  ROAS rise, stay within 0.3 of the static model's value (the table after Exercise 1), or fall by more
  than 0.3? The truth is 4.0. Write your guess in the cell below. (Committed data, FULL: it fell by 1.0,
  from 3.90 to 2.90; over 15 generator seeds, QUICK, it fell in 15 of 15, by 0.58 to 2.39, never within
  0.15 of a boundary.)
- **Function.** `baseline_comparison(roas_own: xr.DataArray, roas_prophet: xr.DataArray, true_roas:
  dict, prob: float = 0.94) -> pd.DataFrame`: per channel the posterior mean and 94% HDI under each
  baseline, the absolute error of each mean, and whether each HDI covers the truth. Both inputs are
  all-period ROAS draws from Exercise 1's function, dims `(chain, draw, channel)`. Output in long form,
  one row per channel and baseline: index `(channel, baseline)` with `baseline` in `{"own", "prophet"}`,
  columns `mean`, `hdi_low`, `hdi_high`, `true_roas`, `abs_error`, `covered`. Long form keeps the
  function to two loops (Module 8's `roas_table` was the inner one); a provided cell pivots it for
  display. Hint in the stub: loop over `{"own": roas_own, "prophet": roas_prophet}.items()`, then over
  channels.
- **Checkpoint.** A toy first (computed by the Pedagogy Expert, 2026-10-10): for two channels a and b,
  own draws are the 1,000 exact quantiles of a lognormal with median 2 and log-sd 0.3 (dims `(2, 500)` per
  channel), Prophet draws the same with median 3; true ROAS a 2.5, b 1.10. Reference: own mean 2.0919,
  94% HDI [1.0209, 3.2904]; Prophet mean 3.1379, [1.5314, 4.9357]; a: absolute errors 0.4081 (own) and
  0.6379 (Prophet), both covered; b: 0.9919 (own, covered: 1.10 lies just inside the HDI) and 2.0379
  (Prophet, not covered). Means to 1e-9, interval bounds within 2% of the interval's width (Module 8's
  tolerance). Then on the fits: for social, the Prophet-baseline mean is below the own-baseline mean, its
  94% HDI excludes the true all-period ROAS, and the own-baseline HDI covers it. Committed data, FULL: own
  3.90 [3.52, 4.32], Prophet 2.90 [2.25, 3.58], truth 4.00; QUICK own [3.44, 4.31], Prophet [2.30, 3.65].
  Over 15 generator seeds (QUICK) the Prophet baseline lowered social's ROAS in 15 of 15, but the whole
  fit check held in 12 of 15, the weakest of the lab's fit checks: in seeds 2038 and 2040 the own
  baseline's HDI lay above 4.0, and in 2044 the Prophet baseline's HDI still covered 4.0. (Prototype,
  seeds 2037 to 2041: lowered in 5 of 5, HDI excluded the truth in 4 of 5.)
- **Failure messages** (four parts, each ending with `workshop.use_reference(5)`):
  - Own a's `abs_error` is −0.408: "that is a signed error (mean minus truth); take its absolute value."
  - Own b's interval is [1.1399, 3.5090], or b is not covered under "own": "that is the equal-tailed
    interval. Use `checks.interval(draws, prob, "hdi")`: in these skewed draws the HDI reaches down to
    1.02 and covers 1.10; the equal-tailed interval does not."
  - The "own" rows hold means near 3.14: "the two baselines are swapped; check the order of the
    arguments."
  - Wrong index or columns: "one row per channel and baseline, index `(channel, baseline)`, with the six
    columns named in the task."
  - The fit checks fail after the toy passes: "Your function is right. This compares the fits with the
    truth, not your code. If it fails in a FULL run, note the message and tell the instructor."
- **Explain.** Compare with your prediction. Which channel lost credit, and where did that credit go?
  Point to the trend-gap plot in Part D.
  `<details>`: Prophet's trend already contains the slow part of social's effect, so with a coefficient
  held near 1 the baseline keeps it and the MMM gives social less. A fading channel lowers sales slowly,
  and a slow change is what a trend with changepoints is built to follow; Prophet cannot know that the
  fall came from social. The fade is not the only thing it takes: in the prototype's least-squares check
  without any fade (not re-run in the build), the Prophet baseline still took credit from tv, social and
  search, because its trend and yearly seasonality also follow parts of the flights, the pushes and
  search's demand-following spend. Two more problems follow from fitting Prophet to the same sales the
  MMM explains: the MMM treats Prophet's components as known data, so the uncertainty in Prophet's own fit
  never reaches the ROAS intervals; and the intervals look reasonable while excluding the truth. On the
  committed data (FULL) social's interval did not become narrower (0.80 wide with the own baseline, 1.33
  with Prophet's); it moved below the truth. What would change it: a baseline the MMM estimates jointly
  with the media (Module 8's trend and Fourier terms, or PyMC-Marketing's time-varying intercept), or
  Prophet told about spend (stretch B, which on the committed data helped tv but not social).

What the runs support, and only that, for the Markdown and the module page. First, the build's runs on the
committed data (FULL, target_accept 0.98, the notebook's printed output):

- The Prophet baseline moved social's all-period ROAS from 3.90 [3.52, 4.32] to 2.90 [2.25, 3.58] (truth
  4.00) and tv's from 1.73 [1.60, 1.86] to 1.44 [1.25, 1.63] (truth 1.60); search's interval widened and
  moved up (own 5.43 [1.01, 12.98], Prophet 7.53 [2.05, 18.34], truth 3.0), and display's moved up and no
  longer covered the truth (own 3.99 [0.19, 9.00], Prophet 4.94 [0.93, 9.45], truth 0.60). Every
  channel's posterior mean was further from the truth with Prophet's baseline than with the model's own
  (absolute errors tv 0.16 against 0.13, search 4.53 against 2.43, social 1.10 against 0.10, display 4.34
  against 3.39).
- Stretch B (spend as Prophet regressors, components still fixed near 1): tv 1.51 [1.40, 1.63], social
  3.15 [2.67, 3.72] (still excluding 4.00), search 5.59 [0.41, 15.25], display 3.52 [0.35, 6.67]. Against
  the model's own baseline, closer for tv and display, further for social and search.
- Over 15 generator seeds (QUICK), the Prophet baseline lowered social's all-period ROAS in 15 of 15.

Second, the prototype only (not re-run in the build):

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

Markdown first, then one provided code cell, in the standard order. The Markdown states the number and the
rule before any result is printed (as in Modules 14 and 16), so the learner reads the rule before the
verdict:

- **Number:** per channel, the drifting model's last-quarter ROAS (mean and 94% HDI) and the posterior
  probability that it is below break-even, computed draw by draw; beside it the static model's
  all-period ROAS with the same probability. Committed data, FULL, social: drifting last quarter 2.53
  [1.94, 3.15], P(ROAS < 3.33) = 0.99 (QUICK 2.67 [1.97, 3.29], 0.96); static all period 3.90 [3.52,
  4.32], P(ROAS > 3.33) = 0.997. Over 15 generator seeds (QUICK) the drifting model said "cut" in 14 of
  15; in seed 2040 P(below) was 0.81, which the rule calls "lift test first". (Prototype, seeds 2037 to
  2041: P(below) ≥ 0.96 in 4 of 5 and 0.84 in the fifth.)
- **Rule:** "A dollar of sales earns the 30% gross margin, so break-even ROAS is 1 ÷ 0.30 ≈ 3.33. Cut a
  channel's budget next quarter if P(current-quarter ROAS < 3.33) ≥ 0.9; keep it if P(> 3.33) ≥ 0.9;
  otherwise run a lift test first (Module 9). We assume next quarter's effect equals last quarter's: the
  model's multiplier is not extrapolated, because a GP reverts to its prior mean outside the data
  (PyMC-Marketing's TVP notebook says so). Under saturation the last dollar returns less than the
  average dollar, so an average below break-even means the marginal dollar is below it too." In the
  notebook, "a GP reverts to its prior mean" reads "beyond the last week the model's multiplier drifts
  back toward its average of 1", which needs no GP vocabulary.

The code cell computes everything; nothing in it is typed by hand:

- **Which channel.** The rule is applied to the channels in `fading`, set by the labeled check after
  Exercise 4 (drift ratio 94% HDI entirely below 1; social on the committed data), not to the string
  `"social"`, so the same cell works on data where another channel fades. The other channels' ROAS are in
  the table after Exercise 4; the cell says in one sentence that this decision concerns the fading
  channel.
- **Verdicts.** The rule applied in code to the drifting model's last-quarter draws and, for contrast, to
  the static model's all-period draws: "cut", "keep" or "lift test first", with the probability beside each.
- **Scale.** Last quarter's spend on the channel, summed from `weekly` ($192,407), and what that spend
  returned net of its cost, per draw: spend × (`GROSS_MARGIN` × ROAS − 1), with its mean and 94% HDI.
  Committed data, FULL: −$46,199 (94% HDI −$80,289 to −$10,515); the truth file's value is −$70,731.5.
  It is the average return of last quarter's spend, not the return of the next dollar.
- **The mode.** "FULL run", or "QUICK run: treat this as a rough answer".

Printed layout (the FULL run's output on the committed data; the notebook fills the numbers by code):

```text
Number          : social, last quarter, drifting model: ROAS 2.53 (94% HDI 1.94 to 3.15),
                  P(ROAS < 3.33) = 0.99.
                  social, all weeks, static model: ROAS 3.90 (94% HDI 3.52 to 4.32), P(ROAS > 3.33) = 0.997.
Scale           : social spent $192,407 last quarter; net of that spend, the margin on the sales it
                  produced was about -$46,199 (94% HDI -$80,289 to -$10,515).
Rule            : cut if P(current-quarter ROAS < 3.33) >= 0.9; keep if P(> 3.33) >= 0.9;
                  otherwise run a lift test first.
Verdict         : drifting model, last quarter: cut. Static model, all weeks: keep.
Assumes         : next quarter's effect per dollar equals last quarter's.
What would change it: a creative refresh that a test shows has brought the effect back above
                  break-even; a lift test now (stretch A).
Your sentence   : ________________________________________________
FULL run.
```

- **Recommendation:** the learner's sentence: what to do with social's budget (last quarter's spend in
  the data, $192,407), how sure, how the answer differs from the static model's (keep), and what would
  change it (a creative refresh tested before spend returns; a lift test now, stretch A).
  One sentence under the cell links Exercise 4: there the drifting model understated the fall, and its
  current-quarter ROAS lay above the truth on the committed data and in 14 of 15 generator seeds (QUICK;
  in seed 2050 it was 1.91 against a true 2.08; prototype: 5 of 5); so a "cut" from it is not the model
  exaggerating the fade.
  Then the cell shows the truth (social's true last-quarter ROAS 2.11, below break-even). The cell
  prints the mode.

## Stretch (optional)

- **A · A lift test now.** Add `mmm_drift_lift_tests.csv` (one dated social test ending 2025-11-17) to
  the drifting model: `build_model`, then `add_lift_test_measurements(df)`, then fit. With a
  time-varying media model, PyMC-Marketing multiplies the saturation curve by the multiplier at the
  test's `date` (`create_time_varying_saturation`, `mmm/lift_test.py`), so a recent test pins the current
  level. Committed data, FULL (0 divergences): social's last-quarter ROAS 2.22 [1.92, 2.50], P(< 3.33) =
  1.00, against 2.53 [1.94, 3.15] without the test (truth 2.11); prototype 2.23 [1.96, 2.53]. The `date`
  column must equal one of the model's weeks (`lift_test.py`, `exact_row_indices`); the lab says so.
- **B · Tell Prophet about spend.** Refit Prophet with `add_regressor(c)` for each channel, rebuild the
  components and refit fit 3. Committed data, FULL: tv 1.51 [1.40, 1.63] (truth 1.60), social 3.15
  [2.67, 3.72] (truth 4.00), search 5.59 [0.41, 15.25] (truth 3.0), display 3.52 [0.35, 6.67] (truth
  0.60): better for tv, still wrong for social, because a linear spend regressor cannot represent an
  effect that fades. This is now the lab's hands-on Prophet work: the learner copies
  the provided `prophet_components` into a new function with the regressors added (the stub names
  `add_regressor` and quotes Prophet's documentation: a regressor "must be known for both the history and
  for future dates", so each spend column goes into the frame passed to `predict` as well), and compares the
  result with `baseline_comparison`.

Nothing later depends on either. A third option was measured in the prototype (not re-run in the build)
and left out: refitting with `eta_upper=1.0` barely moved the answer (social 2.61 [1.99, 3.27], coverage
0.60).

## Expected run time (estimate, not a run record)

**Measured since (run records, worked mode, FULL, the whole notebook including both stretch fits;
re-recorded after the checkpoint-message fixes):** 140.2 s on the Apple Silicon laptop in the native
environment (shared with other build jobs; the slowest cell was the drifting fit, 52.61 s) and 169.7 s in
the workshop's Docker image with `--cpus 2` (the slowest cell was the static fit, 36.64 s, which includes
the first compilation), both on 2026-10-10
(`runs/2026-10-10-macos-arm64-python-17-mmm-time-varying.json`,
`runs/2026-10-10-docker-arm64-python-17-mmm-time-varying.json`). Neither is a Colab time; no Colab record
exists, so the Colab column below is still an estimate.

The table below is the prototype's, measured on the laptop (Apple Silicon, 2 chains on 2 cores, load
average about 5 from other jobs). The Colab column multiplies by 3.8, the ratio of Colab to laptop times for the MMM fits in
spike S2 (45.0 s against 11.9 s first fit, 9.7 s against 2.6 s again; DECISIONS.md). It is an estimate
until a Colab record exists.

| Step | Laptop FULL (s) | Laptop QUICK (s) | Colab FULL, estimated (s) |
|---|---|---|---|
| Install pymc-marketing and nutpie (Prophet 1.4.0 is preinstalled on Colab) | — | — | 35 to 53 (Modules 8 and 9 Colab records) |
| `import pymc_marketing` | 2 | 2 | 17 (S2) |
| Static fit | 22 | 11 | 85 |
| Prior predictive, 1,000 draws | 2 | 1 | 6 |
| Drifting fit (the longest cell) | 47 to 54 | 18 to 21 | 180 to 205 |
| Prophet import and fit (provided cell, Part D) | 0.2 | 0.2 | 1 to 5 |
| Prophet-baseline fit | 18 | 7 | 70 |
| Whole notebook, compute | about 90 | about 35 | about 6 to 7 minutes |
| Stretch A + B | 47 | — | about 3 minutes more |

Against the targets (install ≤ 3 min, any cell ≤ 4 min, compute ≤ 15 min): inside, with the drifting fit
the closest (estimated 3.0 to 3.4 minutes). Cut rule if a Colab record shows that cell above 210 s:
lower FULL draws for the drifting fit to 700 (500 draws took 16 to 29 s on the laptop over five seeds),
never `target_accept`.

**If a pilot runs long** (room time, not compute). First, Exercise 3's function becomes a provided cell:
keep the prior paths plot, the Predict, the printed share and the Explain, and provide `drift_ratio`; that
frees about 3 minutes, and the labeled drift-ratio check after Exercise 4 still runs. Second, switch the
room to QUICK, which roughly halves the fits (laptop prototype: about 35 s of compute against about 90 s). Keep
Exercises 2 and 4 hands-on: the residual check is the one learners can run on their own MMM tomorrow, and
the band is the evidence for the decision.

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

All on the stand-in generator, laptop, 2026-10-10; not run records. The seed-2037 rows agree within 0.01
with what the build measured on the committed data (`truth_mmm_drift.json` `tolerances`). For other seeds the build's 15-seed sweep (seeds 2037 to 2051, QUICK;
`tests/test_synth_mmm_drift.py`, `data/README.md`) replaces this table as evidence; see "Build
reconciliation".

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
- From the pedagogy review: each checkpoint lists its likely wrong answers, with the values the toys give
  for them, and what its message says. Add a verify-mode run, or a `mktstats.checks`-style test, that feeds
  each listed wrong answer and expects its message. Re-check every Predict's boundaries on the committed
  data (Exercises 3, 4 and 5 say how far from a boundary the answer must be), and Exercise 2's 2023 sign.
- Replace the prototype numbers in Explain `<details>` blocks and "Look at" sentences with the committed
  data's FULL numbers, or phrase them without numbers (`_lab-standard.md` §11, Honesty).
- Each fit cell's Markdown says how long it takes on Colab as an estimate until a Colab record exists, and
  what to do while it runs ("Parts and exercises", the waits table).

## Pedagogy review (2026-10-10)

Checked against `briefs/_lab-standard.md` §11 before any code exists. What changed, and why:

1. **Budget: 7 + 8 + 6 + 9 + 10 = 40 (limit 40), with reading and waiting counted.** The draft's six
   exercises added up to 40 on paper, but its minutes left out about 6 to 7 minutes of Colab compute (three
   fits, estimated) and about 5 minutes of provided reading (the static ROAS table, the GP cell, the
   current-quarter table, the Prophet and Robyn cell, the trend-gap plot). Counted the way Modules 14 and
   16 count, it came to 46 to 50 minutes against 40: past the slack, for an audience meeting Gaussian
   processes and Prophet for the first time. The cut: **Prophet's decomposition becomes a provided cell**
   with a labeled check (the draft's Checkpoint 5) and the two outputs its Explain used. Its function was
   API recall (construct Prophet, fit, pick three columns); its idea, that Prophet's trend takes up part of
   a fading channel's effect, survives as the trend-gap plot and is what the new Exercise 5 tests. Hands-on
   Prophet moves to stretch B. Objective 4 in `modules.m17` still holds (the learner measures how each
   channel's ROAS moves; the fits were always provided), so `_variables.yml` is unchanged. The other
   exercises get their real minutes: Exercise 2 from 7 to 8 (the static fit and its table come first),
   Exercise 3 from 5 to 6 (a Part C cell and a plot to read), Exercise 4 from 8 to 9 (the longest fit),
   Exercise 5 (was 6) from 7 to 10 (Part D's provided cells and fit 3).
2. **No idle waits.** Each fit cell tells the learner what to do while it runs: read the next Predict,
   write the guess, and write the next function, whose toy checks need no fit (the waits table under
   "Parts and exercises"). The drifting fit, about 3 to 3.5 minutes on Colab (estimated), sits inside
   Exercise 4, whose function the learner writes meanwhile.
3. **The drifting model at the depth the decision needs (question 2 of the review request).** The draft
   defined GP, lengthscale, amplitude, HSGP, softplus, the mean-1 normalization and two prior families in
   one Markdown cell before a 5-minute exercise. The decision needs three ideas: a multiplier per channel
   and week with average 1, that changes slowly, and a prior on how far it moves, checked in the decision's
   units before the fit. The notebook now shows the GP prior as 20 curves (the plot is the definition)
   and names lengthscale and amplitude in one plain sentence each; HSGP, softplus, the basis, the kernel
   and the prior families go into one-line code comments, `<details>` and the briefing (Terms, "What stays
   out"). No Predict or Explain asks about them.
4. **Predicts ask what has not been shown, with a scale.** Exercise 2 asked which channel's correlation
   falls most, which the true paths in Part A already answer; it now asks for the signs in 2023 and 2025,
   the mechanism its Explain uses. Exercise 3's bins (under 1%, 5 to 10%, over 25%) had gaps and no
   reference; they are now contiguous (under 3%, 3 to 20%, over 20%), and the plot of 20 prior curves is
   the reference. Exercise 4's bins had gaps (90 to 94%, 50 to 60%) and are now contiguous with a stated
   reference (a right band covers about 94%). Exercise 5's "toward, away, stay" had no size; it now has
   one (within 0.3, or further). Each Predict says how far from a boundary the answer must be on the
   committed data.
5. **Explains are answerable from what the learner just saw.** Exercise 3's Explain deferred its answer to
   after Exercise 4 ("Is 7% enough room?"); it now asks whether the prior rules out, allows or expects a
   fall as large as the true one, which the printed share answers, and its `<details>` cites the
   prototype's half-amplitude refit. Exercise 4 gets its plot before the Explain that points at it.
6. **Checkpoints check the learner's code first, with toys that tell mistakes apart.** Exercise 1's toy
   (contribution = 2 × spend) could not catch a mean of weekly ratios; a four-week toy now separates six
   mistakes. Exercise 4's toys add a skewed one (exact quantiles, so it cannot flake) that catches an
   equal-tailed or 89% interval, and one that catches a band pooled over all weeks. The drift-ratio HDIs
   moved from Checkpoint 4 to a labeled check, because they use Exercise 3's function and can fail only
   through the fit. Every checkpoint has failure messages with the likely wrong values, computed by the
   Pedagogy Expert on 2026-10-10 (numpy 2.1.3, pandas 2.2.3, xarray 2025.12.0, `mktstats.checks.interval`);
   the fit-against-truth parts say "not your code" and how to report it.
7. **Decision.** Number and rule in Markdown before any result; the verdict computed by the rule in code
   for both models; the fading channel taken from the drift ratio, not typed; a Scale line (last quarter's
   spend and what it returned net of its cost, per draw, with its 94% HDI); an Assumes line; the template's
   "What would change it" and "Your sentence" lines; one sentence linking Exercise 4 (the model errs high,
   so "cut" is not an exaggeration).
8. **Wording.** Terms defined where first used (new section); every Predict ends with "Write your guess in
   the cell below"; the static model's two windows give "about the same number" (3.90 and 3.88 in the
   prototype), not "the same number". One code-comment claim was corrected against the installed source:
   Matérn 5/2 is PyMC-Marketing's choice for `time_varying_media=True` (`mmm/tvp.py`), while
   `parameterize_from_data` defaults to `ExpQuad` (`mmm/hsgp.py`). The changepoint sentence in Terms was
   checked against Prophet's *Trend changepoints* page on 2026-10-10 ("puts a sparse prior on the
   magnitudes of the rate changes").

**What the briefing carries, so that the lab can stay short** (for the Academic Director's module page): a
picture of a GP prior as random smooth curves and what lengthscale and amplitude do to them; why one
multiplier per channel and not one shared (`time_varying_media=True`); identification by the quarterly
pushes, with the picture of a shrinking post-push jump; the window and counterfactual definitions of ROAS;
how Robyn uses Prophet, and why a baseline fitted to the same sales can take media credit (the
prototype's least-squares results under "What the prototype supports"); the random-walk alternative,
which the lab does not use.

**For the Academic Director to reconsider.** (a) The cut of Prophet's decomposition to a provided cell:
the draft said no cut was needed; the estimate in item 1 says otherwise. If a pilot shows room, the
function can come back as an exercise before Exercise 5, with Exercise 3's function provided instead to
keep the budget. (b) The Scale line's net figure, spend × (margin × ROAS − 1), is a new computed
quantity in the Decision, not a new fact; check that it reads as the average return, not the marginal one,
next to the rule's sentence on saturation. (c) Exercise 2's Predict now asks for signs; if the committed
data's 2023 correlation is near 0, the brief says what to ask instead.

## Build reconciliation (2026-10-10)

Checked by the Academic Director against the build (`labs/src/17-mmm-time-varying.py`,
`src/mktstats/drift.py`, `src/mktstats/synth/mmm_drift.py`, `truth_mmm_drift.json`), its two run records,
and a worked, FULL run of the notebook on the laptop on 2026-10-10 (not recorded; it passed, and its printed
numbers agree with the truth file's `tolerances` to the printed precision).

- **Data.** The committed generator reproduces the prototype exactly at seed 2037, so the prototype's
  seed-2037 numbers carried over and are now labeled *committed data*. Numbers from analyses the build did
  not repeat keep the label *prototype*: the five-seed MCMC comparisons, the 30-seed least-squares Prophet
  results under "What the runs support", the `eta_upper=1.0` refit, the divergence at `target_accept` 0.95,
  and the run-time table.
- **Numbers that changed when re-measured** (prototype, then committed data; FULL unless said): static
  all-period HDI [3.53, 4.32], then [3.52, 4.32]; QUICK static last-quarter upper bound 4.32, then 4.31;
  social's 2025 residual correlation −0.58, then −0.57; drift-ratio HDIs of search [0.48, 1.43], then
  [0.49, 1.44], and display [0.50, 1.76], then [0.51, 1.76]; search with the Prophet baseline 7.54 [1.96,
  18.29], then 7.53 [2.05, 18.34], and with its own baseline lower bound 0.99, then 1.01; Checkpoint 5's
  QUICK intervals; stretch A 2.23 [1.96, 2.53], then 2.22 [1.92, 2.50]; stretch B tv [1.40, 1.64], then
  [1.40, 1.63], and social 3.16 [2.64, 3.71], then 3.15 [2.67, 3.72]; the Decision's net figure, estimated
  as about −$46,000 (−$80,000 to −$11,000), then −$46,199 (−$80,289 to −$10,515).
- **Seeds.** The build's slow sweep (`tests/test_synth_mmm_drift.py`; 15 generator seeds, 2037 to 2051,
  QUICK; summarized in `data/README.md`) replaces the prototype's five seeds as evidence. Every
  fit-against-truth check held in 15 of 15 seeds except Checkpoint 5's fit check (12 of 15); the Decision
  said "cut" in 14 of 15 (seed 2040: P(below) 0.81, "lift test first"); the drifting model's
  current-quarter ROAS lay above the truth in 14 of 15; 1 of 45 fits had a divergence (seed 2050, drifting
  fit). The test holds each claim to its measured count minus one.
- **Divergences.** 0 in every fit on the committed data at `target_accept=0.98`, FULL and QUICK, the two
  stretch fits included.
- **Kernel.** `parameterize_from_data` defaults to `ExpQuad` (`mmm/hsgp.py`), so the lab sets
  `cov_func="matern52"` explicitly, the kernel PyMC-Marketing uses for its own `time_varying_media=True`.
  With it, $m = 56$ and $L = 426.4$; 55 coefficients per channel remain after `drop_first`.
- **Prophet.** As the Pedagogy review decided, Prophet's decomposition is a provided cell with the labeled
  check "prophet", and hands-on Prophet is stretch B. Objective 4 still holds.
- **One wording slip in the lab**, for the next change to `labs/src` (which stales the run records and so
  waits for a Colab re-sweep): Exercise 4's `<details>` says the truth lies above the band in "weeks 5 to
  23"; the printed output says weeks 5 and 7 to 23 (week 6 lies inside). The week-144 numbers in the same
  block (model 0.68, truth 0.55) were confirmed by a FULL refit with the lab's settings, which reproduced
  the lab's last-quarter ROAS for social (2.53 [1.94, 3.15]).
- **Run time.** Measured totals from the run records now stand beside the prototype's estimates ("Expected
  run time"). The Colab column is still an estimate: the lab has not run on Colab, because no release tag
  contains it yet.
- **"For the Academic Director to reconsider."** (a) Keep the cut: Prophet's decomposition stays provided
  until a pilot shows room. (b) The Scale line is the average return of last quarter's spend; the brief's
  Decision and the module page say so beside the saturation sentence. (c) Social's 2023 correlation is
  +0.40 on the committed data and above +0.1 in 15 of 15 seeds, so the Predict keeps the two signs.
- **Open items.** Done, except the two that need a release tag: `repo_file` on Colab at `repo.ref`, and the
  drifting fit's Colab time (with the 210 s cut rule). `tests/test_drift.py` feeds each listed wrong answer
  to its check and expects its message; `tests/test_synth_mmm_drift.py` checks the Predict boundaries at
  FULL.
