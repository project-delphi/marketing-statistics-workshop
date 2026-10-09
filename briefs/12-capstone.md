# Lab brief · Module 12 · Capstone: one decision pipeline for the retailer

| | |
|---|---|
| Status | Brief by the Academic Director, 2026-10-09; updated the same day by the Technical Expert to match the built lab (`labs/src/12-capstone.py`) and the synthetic-data measurements (Stage 2 question, Stage 4 checkpoint, CLV without the first purchase, relative placebo spread). Built and run on a laptop and in Docker; Colab not yet run. Review: Pedagogy Expert (who writes the rubric, `briefs/12-capstone-rubric.md`). |
| Notebook | `labs/src/12-capstone.py` → `labs/python/12-capstone.ipynb` |
| Slot | Day 5 from 11:25 (lead's schedule): briefing 15, pair work 80, lunch 50, pair work 100, break 10, presentations 60, wrap-up 20. `modules.m12.minutes`: briefing 15, lab 180 (80 + 100 of pair work; lunch and the break are separate blocks in `days.d5.blocks`), debrief 60. |
| Budget | Pair work 180 min: open 5 + stages 125 + brief 25 + slack 25 (15 in block 1, 10 in block 2). Minutes are estimates until a pilot. |
| Work mode | Pairs, one notebook per pair, one runtime. |

## Question and decision

**Question.** Putting the week together: what should the retailer spend next quarter, where, and on whom?

**Decision.** Recommend next quarter's budget by channel and the customers to target with the retention
offer, with expected value, a 94% interval and the main risk to the recommendation
(`modules.m12.decision`), presented as a five-slide brief.

## What this lab fixes

Modules 1–11 each answered one question with its own data. Here each stage's output feeds the next, so an
error early on shows up in the final number, and every stage is checked against the truth before the next
one uses it.

## Data: one coherent synthetic scenario

`mktstats.synth.capstone(seed=2032)`, committed as `data/synthetic/capstone_*.csv` and read with
`mktstats.data.load_synthetic("capstone_transactions" | "capstone_customers" | "capstone_geo_panel" |
"capstone_mmm_weekly" | "capstone_email_experiment")`. Truth in `truth.json` → `capstone`: `scenario` is
what pairs are told (dates, treated regions, base and changed spend, the treated-share and
national-equivalent rules, budget, bounds, current plan, the media-to-acquisition link and cost per new
customer, offer economics); `answers` holds each stage's true value and checkpoint tolerance and is read only
by checkpoints and the final reveal cell. The capstone's seeds and parameters differ from the module labs'
so answers cannot be copied.

- 4,000 customers with acquisition channels search / social / referral (Stage 1).
- A 40-region geo panel in which **paid search was switched off in 8 regions for 10 weeks** (Stage 2). Each
  region responds as a scaled copy of the national curve, so the national-equivalent lift row lies on the
  MMM's true search curve.
- A weekly MMM (tv, search, social, display) whose search spend follows a hidden demand shock (Module 9's
  confounding), ending the week before the geo test (Stages 3–4).
- The e-mail experiment with true CATE and a `split` column (train / test halves; Stage 5).

**Why a switch-off and not a spend increase.** The synthetic-data build measured that a test which doubles
search spend made the calibrated MMM's search ROAS worse (it probes the curve above the observed spend and
leaves the level, which the demand shock inflates, free), so the scenario's test is a holdout: sales lost
$85,310 for $31,200 of spend saved, incremental ROAS 2.73, below the 3.33 break-even at 30% margin.

## Provided toolkit (folded cell)

`geo_readout` (Module 7's synthetic control with in-space placebos, through
`mktstats.recovery.synthetic_control`), `response_curve` and `allocate` (Module 10, SLSQP inside
`threadpool_limits(1, "blas")`), `features` and `targeting_profit` (Module 11; profit per customer from the
experiment's outcomes with a 94% normal-approximation interval), `show(fig)`. Pairs may paste their own
versions. The stage exercises are the integration functions.

## Model fits (provided cells)

| Stage | Fit | QUICK | FULL | Measured, FULL (laptop) |
|---|---|---|---|---|
| 1 | `BetaGeoModel` with `channel_social`, `channel_referral` on purchase and dropout, MCMC nutpie; `GammaGammaModel` MCMC with weak half-normal priors; every purchase to date | 2 chains × 200 draws (tune 200) | 2 × 500 | 20–21 s for both (Docker, 2 CPUs: 39 s, the slowest cell) |
| 3 | MMM without the test, then calibrated (`build_model` → `add_lift_test_measurements` → `fit`), Module 8's specification, default priors, `target_accept=0.9` | 2 × 200 | 2 × 500 | 6 s and 7 s |
| 5 | `CausalForestDML(discrete_treatment=True)` on the train split | 200 trees | 1,000 trees | 8 s |

Fewer draws than the module labs (1,000–2,000) so that the whole notebook fits Colab. Recorded
(`runs/2026-10-09-m12-*.json`): the whole notebook in worked mode with verification took 50.4 s FULL and
33.3 s QUICK on the laptop and 91.5 s FULL in Docker with 2 CPUs; no cell took more than 21 s on the laptop
or 39 s in Docker. Colab is expected to be two to five times slower than the laptop (Module 8's MMM fit:
110 s on Colab against about 20 s on a laptop) and has not been timed yet. No Pareto/NBD MCMC (231 s for
2×300 on Colab, lead's measurement). All fits `progressbar=False`.

## Stages

| Block | Stage | Minutes |
|---|---|---|
| Block 1 (80) | Open, install, read the scenario | 5 |
| | 1 · What is a new customer worth, by channel? (Exercise 1, checkpoint A) | 30 |
| | 2 · Is search incremental? (Exercise 2, checkpoint B) | 30 |
| | *slack* | 15 |
| Lunch (50) | | |
| Block 2 (100) | 3 · Calibrate the MMM with the geo test (Exercise 3, checkpoint C) | 25 |
| | 4 · Next quarter's budget (Exercise 4, checkpoint D) | 20 |
| | 5 · Whom to send the retention offer (Exercise 5, checkpoint E) | 20 |
| | The five-slide brief (provided cell) | 25 |
| | *slack* | 10 |
| Break (10) | | |

Each stage is one short function with a checkpoint against `capstone.answers` (a stage is longer than a lab
exercise because pairs also read the scenario, run the fits and discuss). Checkpoints 1–5 are the rubric's
A–E. The numbers below are from the FULL laptop run (QUICK in brackets where it differs).

### Stage 1 · What is a new customer worth, by channel? (30 minutes)

- **Predict.** Rank the acquisition channels by 36-month new-customer value; is the top worth less than 1.5×,
  1.5–3× or more than 3× the bottom?
- **Function.** `new_customer_value(cum_purchases, mean_spend, monthly_rate, margin, first_purchase=True)`:
  Module 5's convention, $m \bar s (1 + \sum_k \Delta N_k / (1 + d)^k)$, with the first axis of
  `cum_purchases` the month 0..36 and any further axes (draws, channels) carried through.
- **Checkpoint 1 (before any fit).** A toy curve with and without the first purchase; the shape rule; and
  the **generator's** Pareto/NBD curve with the true parameters, on which the function must reproduce
  `answers.stage1_customer_value` (margin with and without the first purchase) to 1e-4.
- **Checkpoint "1 · fitted values".** With the BG/NBD and Gamma-Gamma draws, each channel's posterior mean
  within 20% of `margin_including_first_purchase` (BG/NBD approximates the Pareto/NBD generator) and the
  top channel right. Measured: search $88.85 (+2.4%), social $55.87 (+4.8%), referral $122.83 (−12.2%) vs
  $86.75 / $53.30 / $139.86; top channel referral. (The synthetic-data build measured a MAP fit on the
  calibration year at +8.6% / +14.8% / −0.7%; the lab fits every purchase to date, which a planner would.)
- **Run.** Values with 94% HDIs and CAC caps (20th percentile of value).

### Stage 2 · Is search incremental? (30 minutes)

- **Predict.** The 8 regions saved $31,200; at 30% margin the cut paid for itself if they lost less than
  $104,000 of sales. More or less? And will the placebo p-value be below 0.1?
- **Function.** `lift_test_row(readout, treated_share, test_weeks, base_weekly_spend, spend_change_total)`
  → one row `channel, x, delta_x, delta_y, sigma` on the national weekly scale: Δx = spend change / (f W),
  Δy = sales change / (f W), σ = (sd of the placebos' **relative** effects, keeping placebos whose
  pre-period RMSPE is at most twice the median) × the treated synthetic total / (f W). The readout
  (`geo_readout`) and the treated share f (computed from the panel) are provided.
- **Checkpoint 2.** A toy readout whose fifth placebo must be dropped; then `answers.stage2_geo_test`: the
  readout's sales change within 15% and placebo p ≤ 0.1; the row's Δx within 1% and Δy within 15% of the
  national-equivalent lift test; 0 < σ ≤ 25% of |Δy| (a σ from raw or unfiltered placebo spread is larger
  than the effect and fails). Measured: −$86,857 (truth −$85,310, 1.8% off), p = 0.03, row x = 18,000,
  Δx = −17,996, Δy = −50,099 (truth −49,207), σ = 4,310 (8.6% of |Δy|); 2 of 32 placebos dropped.
- **Run.** Sales lost with a 94% interval ($72,803 to $100,910) and sales per search dollar 2.78 (2.33 to
  3.23) against the 3.33 break-even: search is incremental, and below break-even in short-run margin alone.

### Stage 3 · Calibrate the MMM with the geo test (25 minutes)

- **Predict.** Will calibration move search's ROAS up or down, and its 94% HDI narrower or wider?
- **Function.** `roas_summary(roas_draws, break_even, prob=0.94) -> pd.DataFrame` from
  `mmm.incrementality.contribution_over_spend(frequency="all_time", ...)`: mean, 94% HDI and
  P(ROAS > break-even) per channel. (Both fits provided; the calibrated one uses Stage 2's row.)
- **Checkpoint 3.** Toy draws (mean, strict share above, the HDI from `checks.interval`); then
  `checks.k_of_K_in_interval`: true ROAS inside the calibrated 94% HDI for at least 3 of 4 channels.
  Measured: search 3.96 (2.51 to 5.31) uncalibrated → 2.87 (2.54 to 3.24) calibrated, truth 2.6; all four
  inside. Fit checks printed: 0 divergences; max R-hat 1.005 calibrated, 1.035 uncalibrated (the notebook
  says the uncalibrated chains disagree because the data cannot pin search's curve).

### Stage 4 · Next quarter's budget (20 minutes)

- **Predict.** Which channel gains the most budget once new customers' future margin counts?
- **Function.** `recommend_budget(curves, clv_media, new_customers_per_dollar, bounds, weekly_total,
  margin) -> pd.Series`: per channel s ↦ m r_c(s) + s n_c CLV_c, passed to the toolkit `allocate`. Inputs
  (provided): posterior-mean curves from the calibrated MMM; CLV by media channel = Stage 1's value
  **without the first purchase** (the MMM's short-run sales already count it), weighted by the scenario's
  media-to-acquisition shares; new customers per dollar from the scenario.
- **Checkpoint 4.** On the **true** curves and true CLV the function must land on
  `answers.stage4_allocation.optimal_allocation`: spend by channel within 5% of the budget and long-run
  value within 3% of the optimum. The value test alone does not discriminate (the short-run optimum reaches
  98.9% of the optimal value, the current plan 92.4%); the spend test does (ignoring future margin moves
  $8,272 a week, 11% of the budget). Then the pipeline plan keeps the budget and the bounds.
- **Why the spend test is not applied to the pipeline plan.** Measured on the reference run, the plan from
  the calibrated MMM's curves (tv $34,376, search $29,995, social and display at their lower bounds) is
  $15,540 a week (21% of the budget) from the true optimum (tv $49,916, search $14,455) and reaches 96.2% of
  the optimal value (current plan 92.4%). The reveal cell decomposes it: true curves with the pipeline's
  CLV reach 99.9%, the pipeline's curves with the true CLV 98.7%. The error is Stage 3's: the switch-off
  test pins search's level at the current spend (calibrated ROAS 2.87 vs 2.6) but not its slope above it,
  so the posterior-mean curve saturates too slowly (marginal ROAS at the plan 1.69 to 2.92, 94% HDI). That
  is the capstone's main risk and the next test to run (a search spend increase in other regions), and the
  brief prints it on slide 4. Applying the 5% test to the pipeline plan would fail on the reference.
- **Run.** Four plans scored in every posterior draw (MMM draw i paired with Stage 1 draw i): current,
  short-run (CLV 0), long-run, risk-averse (maximize the 10% quantile, SLSQP from the long-run plan). Long-run
  gain over current $6,237 of margin a week (94% HDI $4,532 to $7,972), P(beats current) 1.00; the
  risk-averse plan barely moves.

### Stage 5 · Whom to send the retention offer (20 minutes)

- **Predict.** What share of held-out customers will the rule target: under 20%, 20–50%, over 50%?
- **Function.** `retention_targets(cate_hat, margin, offer_cost) -> pd.Series[bool]`: margin × uplift >
  cost, strictly.
- **Checkpoint 5.** A toy case at the break-even uplift; then on the test split, the rule's true value
  (from `true_cate`) is positive and at least `answers.stage5_targeting.test_split.treat_all`. Measured:
  the rule targets 39% and earns $0.367 per held-out customer [QUICK, 200 trees: $0.342] against $0.126 for
  mailing everyone and $0.512 for perfect targeting.
- **Run.** Held-out profit estimated from the outcomes alone (`targeting_profit`): $0.31 per customer
  (94% interval $0.07 to $0.55) vs $0.06 for mailing everyone.

### The five-slide brief (25 minutes, provided cell)

A provided cell collects every number the slides quote from the stages (no hand-typed numbers) and prints a
five-slide outline that pairs copy into their slide tool, in the rubric's order: 1 · the recommendation
(budget by channel, gain with 94% HDI, P(beats current), the offer list and its profit); 2 · value by
channel with 94% HDIs and CAC caps; 3 · the geo test and what it changed in the MMM; 4 · the plans and their
risk (largest move, the channel with the widest marginal-ROAS interval); 5 · targeting, the assumptions, the
disclosure of reference stages and QUICK. (The earlier `brief_numbers` exercise became this provided cell to
keep the exercise minutes at 125.)

## Decision · The recommendation (in the presentations)

Provided cell in the standard order: the **numbers** from the stages; the **rules** (budget: the long-run
plan unless its 10% quantile of weekly value is below the current plan's, then the risk-averse plan; CAC
caps at the 20th percentile of value; offer where 0.30 × uplift > $0.75); the pair's **recommendation**
sentence. After the presentations a provided cell reveals the truth for every stage, scores every plan on the
true curves (including the true optimum and the two decomposition plans) and prints the true targeting
profit. The cell prints the mode.

## Stretch (optional)

Rerun Stage 4 with the uncalibrated MMM's curves and score the plan on the true curves (what one geo test was
worth); or value new customers with the first purchase in Stage 4 and see whether double counting changes
the plan (on the true curves it does not: `value_sensitivity.plan_if_clv_includes_first_purchase` equals
the optimum, because tv and search gain the same $32 × 0.30 / $150 per dollar).

## Known pitfalls

- Channel names: media channels (tv, search, social, display) and acquisition channels (search, social,
  referral) share two names but are different lists; the scenario's media-to-acquisition table links them.
- Signs: a holdout gives negative Δx and Δy; `add_lift_test_measurements` accepts them (Δx·Δy ≥ 0).
- Placebo spread in dollars, or with the two placebos that do not fit, gives σ larger than the effect.
- Lambdas in a dict comprehension must bind the channel (`lambda s, c=c: ...`).
- Per-period budgets in PyMC-Marketing's optimizer (Module 10) if pairs use it instead of the toolkit.
- Compute: QUICK is the fallback for a slow room; the notebook says QUICK intervals are wider and may flip a
  close decision. In QUICK the MMMs' R-hat exceeds 1.01 and the notebook says so.
- Pairs should not "tune toward the truth": the answers are read only by checkpoints and the reveal cell,
  and failure messages say which way an estimate is off, not the true value.

## APIs verified (and how)

Installed source in `.venv/lib/python3.13/site-packages`, 2026-10-09, then by running every call
(the notebook's first cell lists them): pymc-marketing 1.2.0 `BetaGeoModel` covariates through
`model_config` and `expected_purchases_new_customer(data=, t=)` (clv/models/beta_geo.py),
`GammaGammaModel(model_config=)` (clv/models/gamma_gamma.py), `MMM.add_lift_test_measurements`
(mmm/mmm.py; `lift_test.assert_monotonic` needs only Δx·Δy ≥ 0, the likelihood uses absolute values),
`incrementality.contribution_over_spend(frequency, start_date, end_date)` (mmm/incrementality.py);
econml 0.17.0 `CausalForestDML`; SciPy 1.16.3 SLSQP; threadpoolctl 3.7.0 `threadpool_limits(limits,
user_api)`; ArviZ 1.3.0 `rhat`.

## Open items

- Colab FULL run to record (the lead).
- The rubric's stage letters A–E map to checkpoints 1–5 (Pedagogy Expert: align `12-capstone-rubric.md`).
- `days/_day-5-notes.md` and the rubric's timing table put Stage 3 at 40 minutes, 4a and 4b at 25 each and
  lunch at 12:40, while this brief and `days.d5.blocks` give 80 + 100 minutes of pair work (Stage 3 at 25,
  Stages 4 and 5 at 20, lunch at 13:00): Pedagogy Expert to reconcile.
- Stage 4's pipeline plan is far from the true optimum because one switch-off test cannot pin search's
  slope; if the Academic Director wants pairs to get close, the scenario needs a second search test (a spend
  increase) or a prior on saturation, not a looser checkpoint.
