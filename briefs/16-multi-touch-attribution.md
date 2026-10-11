# Lab brief · Module 16 · Multi-touch attribution: Markov chains, Shapley values and incrementality

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. Written before the lab or its generator existed, from a throwaway prototype script (Apple M1 Pro, Python 3.13.2, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3). Reviewed by the Pedagogy Expert on 2026-10-10 against `briefs/_lab-standard.md`; the changes and their reasons are under "Pedagogy review" below. **Built 2026-10-10 by the Technical Expert:** numbers marked *committed* were re-measured on the committed data (`mta_journeys()`, seed 2036, 100,000 users) with the reference solution; the lab's printed numbers come from the FULL run recorded in `runs/2026-10-10-macos-arm64-python-16-multi-touch-attribution.json`. Where the committed value differs from the prototype, the prototype value follows in parentheses; the changes are listed under "Build notes" at the end. |
| Notebook | `labs/src/16-multi-touch-attribution.py` → `labs/python/16-multi-touch-attribution.ipynb` |
| Lab slot | 55 min (`modules.m16.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Each exercise's minutes include reading the provided cells immediately before it. Minutes are estimates until a pilot. |
| Day | Extensions track (`d6`), optional; outside the five timed days. Builds on Modules 6 and 8. |

## Question and decision

**Question.** When a customer sees several channels before buying, how much of each conversion should
each channel get, and is that credit the same as the conversions the channel caused?

**Decision.** Move budget toward the paid channel that data-driven attribution says is cheapest per
conversion, or run a holdout test first; name the channel where attributed credit and true incrementality
disagree most (`modules.m16.decision`).

## What this lab fixes

Module 6 (Exercise 6) showed with one randomized email test that last-touch credit counts conversions that
would have happened anyway. Multi-touch attribution (MTA) starts from a different complaint: one touch should
not get all the credit. It shares each conversion across the touches in the journey, by rules learned from the
data (a Markov chain, Shapley values) rather than fixed by position (Shao & Li 2011). Whether data-driven
credit equals the conversions a channel causes is a separate question, and nothing in the core week tested it.
This lab tests it in a simulation where each channel's true incremental conversions are known.

Module 6 had one channel and a randomized control group. Here there are five channels, no control group, and
exposure that is **targeted**: some channels reach users who were already likely to buy. Attribution
reports are built from such journeys: nobody randomized who saw what.

## Notation

Five channels C = {display, social, search, email, retargeting}, K = 5. Journey i is the time-ordered list of
touches (c_i1, …, c_iL_i) a user received in one month, with outcome y_i = 1 if the user converted (bought) at
the end of the month, else 0. N journeys, Y = Σ y_i conversions among them. A **touch** is one ad impression,
click or email that the tracking records for a user; a **journey** (or path) is all of one user's touches.

- **Heuristic credit.** Last-click: credit_c = Σ_i y_i · 1[c_iL_i = c]. First-click: 1[c_i1 = c]. Linear:
  Σ_i y_i · (1/L_i) · #{l : c_il = c} (each touch gets an equal share, so a channel touched twice gets two shares).
- **First-order Markov chain.** States start, the K channels, conv, null. n_ab = number of times state a is
  followed by state b over all journeys (start before the first touch, conv or null after the last).
  Transition probability p_ab = n_ab / Σ_b′ n_ab′. conv and null are **absorbing** (p_conv,conv = p_null,null = 1).
  Transient states T = {start, channels}; Q = P restricted to T × T; r = P[T, conv]. The probability of
  eventually reaching conv from each transient state is x = (I − Q)⁻¹ r, and the chain's **conversion
  probability** is p = x_start.
- **Removal effect.** Remove channel c: every transition into c ends the journey without conversion
  (equivalently x_c = 0: zero row c of Q and entry c of r), giving p₋c. RE_c = 1 − p₋c / p. **Markov credit**
  credit_c = Y · RE_c / Σ_k RE_k (the lab's normalization, stated: the removal effects add up to more than 1).
- **Shapley value.** For a set of players S and a value function v on subsets of S with v(∅) = 0,
  φ_c(v; S) = Σ_{T ⊆ S∖{c}} |T|! (|S| − |T| − 1)! / |S|! · [v(T ∪ {c}) − v(T)], the average of c's marginal
  contribution over all orders in which the players could join. Efficiency: Σ_{c∈S} φ_c = v(S) − v(∅).
- **The value function used for journeys.** v(S) = conversion rate among journeys whose *set* of channels is
  exactly S; v(∅) = 0 (no journey has no touch). **Shapley credit** splits each converting journey with channel
  set S in proportion to φ_c(v; S): credit_c = Σ_{S ∋ c} Y_S · φ_c(v; S) / v(S), with Y_S the conversions among
  journeys with set S. By efficiency each conversion is split exactly once.
- **True incremental conversions** of channel c (simulation only): Δ_c = Σ_{i=1}^{n} [Y_i − Y_i^(−c)], over all
  n users in the market (tracked or not), where Y_i^(−c) is the user's outcome had c not reached them, with
  intent, other channels and the user's uniform draw U_i unchanged (common random numbers).
- **Share error** of a method m: D_m = ½ Σ_c |credit_c^m / Σ_k credit_k^m − Δ_c / Σ_k Δ_k|, the share of credit
  that would have to move between channels to match the true incremental shares (total variation distance;
  0 = same shares, 1 = no overlap).

## Data: the synthetic generator (new; the Technical Expert builds it)

| Item | Specification |
|---|---|
| File | `src/mktstats/synth/mta.py` (not exported from `synth/__init__.py`); function `mta_journeys(seed=2036, *, n_users=100_000, random_exposure=False, holdout=None) -> SynthResult` (`name="mta"`) |
| Registration | `EXTENSIONS["mta"]` in `scripts/make_synthetic.py`: writes `data/synthetic/mta_journeys.csv` and `data/synthetic/truth_mta.json` |
| Lab reads | `pd.read_csv(repo_file("synthetic/mta_journeys.csv"))`, then spells out each path with `truth["path_codes"]` (one line in the load cell); `json.loads(repo_file("synthetic/truth_mta.json").read_text())` (`mktstats.data.repo_file`; no change to `data.py`) |
| Table `journeys` | `user_id` (int, 1 to n_users), `path` (str, touches in time order joined by `" > "`), `converted` (0/1), `true_intent` (float, 4 decimals; truth, hidden in real data). One row per user with at least one touch, sorted by `user_id`. **The committed CSV writes each path as one letter per touch with no separator** (`truth["path_codes"]`: D display, S social, P paid search, E email, R retargeting; `mta.encode_paths` / `mta.decode_paths`), so it stays small; the generator's table has the readable paths. |
| Size (committed, seed 2036) | 100,000 users → 69,410 journeys, 3,020 converting (4.35%); 8,441 distinct paths (prototype 8,473); mean 3.57 touches; CSV 1.34 MB with one letter per touch (3.5 MB with the names written out) |

**Data-generating process** (one month, one market of n users; the retailer's online journeys):

1. Purchase intent z_i ~ Normal(0, 1), independent across users; U_i ~ Uniform(0, 1) (the conversion draw).
2. Reach: user i is reached by channel c with probability expit(a_c + g_c · z_i), independently across
   channels given z_i. g_c > 0 means the channel **targets** high-intent users.
3. Frequency: a reached user gets 1 + Poisson(m_c) touches from c.
4. Timing: each touch gets a time in [0, 1] drawn from Beta(e_c, f_c); the path lists touches by time.
5. Conversion at the end of the month: P(y_i = 1) = expit(θ₀ + δ · z_i + Σ_c θ_c · R_ic), with R_ic = 1 if c
   reached i; y_i = 1[U_i < P(y_i = 1)]. **Only which channels reached the user matters: the order and number of
   touches do not change conversion.** That is a simplification, stated in the notebook.
6. Users with no touch are not in `journeys` (no tracked path), but their conversions count in the truth.

| Channel | Story (stated assumption) | a_c | g_c | m_c | Beta(e_c, f_c) | θ_c (log-odds) | Spend / month |
|---|---|---|---|---|---|---|---|
| display | prospecting display, broad reach | −0.6 | 0.0 | 1.5 | (1.5, 4.0), early | 0.30 | $4,000 |
| social | paid social prospecting | −0.9 | 0.2 | 1.0 | (2.0, 3.0) | 0.40 | $3,500 |
| search | paid search; people search when in the market | −2.2 | 1.2 | 0.5 | (3.0, 2.0) | 0.35 | $2,500 |
| email | sent to subscribers, mostly past buyers | −2.0 | 0.8 | 1.0 | (1.0, 1.0) | 0.15 | $300 (list-limited, cannot scale) |
| retargeting | display ads to recent site visitors | −2.4 | 1.8 | 2.0 | (4.0, 1.5), late | **0.00** | $2,500 |

θ₀ = −4.2, δ = 1.0. Retargeting has **no effect by construction**: a stylized assumption chosen to make the
mechanism visible, not a claim about retargeting in general. Margin per conversion: one order at the retailer's
population mean spend of $30 (`truth.json` → `retailer.population_mean_spend`) times the workshop's 30% gross
margin (`channel_value.margin`) = $9; repeat purchases are ignored (Modules 4 and 5 value them). Spend levels are
assumptions of the synthetic world. TV, which Module 8 models, leaves no user-level touch, so it is absent here;
that is one reason MTA and MMM answer different questions.

**Options.** `random_exposure=True` keeps each channel's population reach share but makes reach independent
of intent: reach probability = E_z[expit(a_c + g_c z)] for every user (prototype: display 0.354, social 0.291,
search 0.147, email 0.143, retargeting 0.168, by Gauss–Hermite quadrature); it reuses the same draws
(intent, conversion draw, reach and frequency draws) and changes only the reach probabilities, so the two
worlds are compared user by user. `holdout={"retargeting": 0.2}` randomly withholds the channel from that share
of the users it would reach (their touches from it are removed and their outcome uses the counterfactual with
the same U_i) and adds a table `holdout` (`user_id`, `holdout` 0/1, `converted`) with every user the channel
would reach. The holdout assignment uses its own random stream, so the default dataset does not depend on it.

**`truth_mta.json`** (written through `build_extension`, which adds `about` and `files`):
`seed`, `n_users`, `channels`, `path_separator`, `path_codes`, `csv_encoding`; `dgp` (every parameter above, with a one-paragraph
description); `n_journeys`, `conversions_total`, `conversions_in_journeys`, `conversions_untracked`;
`incremental_conversions` (Δ_c, common random numbers, integers), `incremental_conversions_expected`
(Σ_i [P_i − P_i^(−c)], floats), `incremental_conversions_all_channels` and its expected version;
`definition` (the Δ_c sentence above); diagnostics for the Explain cells: `reach_share`,
`mean_intent_reached`, `conversion_rate_reached`, `conversion_rate_not_reached`; `economics` (`spend`,
`mean_order_value` 30, `gross_margin` 0.30, `margin_per_conversion` 9, `scalable` false for email);
`true_cost_per_incremental_conversion` (spend ÷ Δ_c, `null` where Δ_c = 0); `random_exposure_reach_prob`;
with a holdout, also `holdout` (channel, share, users eligible, held out and served, conversion rates, and
the true incremental conversions among served users).

## What the committed data show (committed numbers, seed 2036, 100,000 users)

The generator reproduces the prototype's truth exactly (same draws for intent, conversion, reach and
frequency); only the touch times are drawn differently (for every potential touch, so that
`random_exposure` reuses them), which moves the positional methods slightly. Linear and Shapley credit do not
depend on touch order and are unchanged.

Truth: Δ = display 358, social 448, search 340, email 114, retargeting 0 (expected values 339, 430, 330, 116,
0). Switching off every channel would lose 1,077 of 3,425 conversions; the single-channel Δ add up to 1,260,
not 1,077, because effects add on the log-odds scale, not on conversions. 405 conversions came from users with
no tracked touch.

| Channel | Last | First | Linear | Markov | Shapley | True Δ | True share |
|---|---|---|---|---|---|---|---|
| display | 450 (465) | 1,183 (1,159) | 797 | 738 (745) | 624 | 358 | 0.284 |
| social | 506 (508) | 804 (815) | 685 | 654 (655) | 654 | 448 | 0.356 |
| search | 532 (503) | 312 (303) | 441 | 472 (460) | 644 | 340 | 0.270 |
| email | 418 (409) | 431 (452) | 369 | 456 (453) | 354 | 114 | 0.090 |
| retargeting | 1,114 (1,135) | 290 (291) | 728 | 700 (708) | 744 | 0 | 0 |
| Share error D_m | 0.42 | 0.26 | 0.27 | 0.29 | 0.27 | | |

What the brief may promise, because the committed data show it (and, where stated, seven more seeds, 11 to
17, with the reference solution; `tests/test_synth_mta.py`, the multi-seed part marked slow):

1. Every method hands out all 3,020 tracked conversions; only about a third of conversions are incremental
   (1,077 of 3,425).
2. Retargeting, with no effect, gets 23–25% of the credit from both data-driven methods (committed: Markov
   0.232, Shapley 0.246; over the eight seeds Markov 0.226–0.238, Shapley 0.225–0.255) and is the channel with
   the largest share gap for last-click, linear, Markov and Shapley on every seed (first-click's largest gap is
   search, on every seed).
3. The data-driven methods beat last-click (committed D: Markov 0.29, Shapley 0.27, last-click 0.42; over the
   eight seeds 0.28–0.31, 0.25–0.29 and 0.40–0.44). They do not beat first-click (committed 0.26; 0.23–0.28) or
   linear (committed 0.27; 0.26–0.28) on the committed data, where Shapley ties linear (0.273 and 0.273). Over
   the eight seeds Markov is further from the truth than both on every seed; Shapley is within 0.03 of both and
   beats linear on 2 seeds (14, 16) and first-click on 2 (16, 17). Say "no better than", not "worse than".
   Markov credit is close to linear credit.
4. The gap is bias, not noise: on the committed data none of the 25 method × channel true shares lies inside
   its 95% bootstrap percentile interval (200 resamples, the FULL run; also 50 resamples, QUICK). Over the
   eight seeds, 1 of 200 intervals covers its true share (seed 17, display under linear: 0.244–0.266 around
   0.263).
5. Two mechanisms, separated by `random_exposure=True`: with the same reach but exposure independent of
   intent, Shapley's share error falls to 0.143 and Markov's to 0.227 on the committed seed (eight seeds:
   0.115–0.147 and 0.209–0.236), and retargeting keeps 10.6% (Shapley) and 14.9% (Markov) of the credit
   (eight seeds: 8.6–10.6% and 14.4–15.2%). The rest is **efficiency**: each method must give every
   conversion, including those that would have happened anyway, to some channel on its path.

The lesson stated in the lab, no further: *attribution methods share credit by their own rules; when exposure
is targeted, neither the Markov removal effect nor Shapley values measure the conversions a channel causes.
Only a comparison with a counterfactual (a holdout, a geo test, a lift-calibrated MMM) does.*

## Model fits

None. Every method is counting, one small linear solve, or exact enumeration over 32 coalitions.

## Terms the notebook defines, where they are first used

The Notation above is for the Technical Expert. The notebook introduces each definition in the part that
uses it, in one plain sentence, never as an opening glossary: Part A (touch, journey, credit, the three
rules), Part B (Markov chain, transition probability, absorbing state, removal effect), Part C (Shapley
value, coalition, Shapley's axioms (symmetry, efficiency, additivity) and the null-player property, the journey value function v), Part D (incremental conversions,
share error, bootstrap interval, holdout, MDE). Suggested wording:

| Term | One sentence for the notebook |
|---|---|
| touch | One ad impression, click or email that the tracking records for a user. |
| journey (path) | All of one user's touches in the month, in time order, for example `display > search > retargeting`. |
| credit | The conversions a method assigns to a channel. Every method hands out all the tracked conversions. |
| last-click, first-click | Give each conversion to the last (first) touch of its journey. |
| linear | Split each conversion equally over the touches of its journey, so a channel touched twice gets two shares. |
| first-order Markov chain | A model of the journey in which the next state depends only on the current one. |
| transition probability | The share of departures from state a that go next to state b. |
| absorbing state | A state the journey never leaves: conversion (`conv`) or the end without conversion (`null`). |
| removal effect | The share of the chain's conversion probability that is lost when every journey that reaches channel c stops there without converting. |
| Shapley value | A channel's added value, averaged over every order in which the channels could have joined. |
| coalition | A set of channels. |
| efficiency, symmetry, additivity, null player | A set's values add up to its value; two channels that add the same to every coalition get the same; the value of a sum of two games is the sum of their values. These three are Shapley's axioms and fix the value uniquely; a channel that adds nothing to any coalition gets 0 follows from them (the null-player property, Shapley's Lemma 1). |
| incremental conversions | Conversions that would not have happened had the channel not reached the user. Known here only because the data are simulated. |
| share error | The share of credit that would have to move between channels for a method's shares to match the true incremental shares (0 = same shares, 1 = no overlap). |
| targeted exposure (selection) | A channel reaches users who were already more likely to buy, so its users convert more often whether or not it works. |
| 95% bootstrap percentile interval | Resample journeys with replacement, recompute, and take the 2.5th and 97.5th percentiles. |
| holdout | A randomly chosen share of the users a channel would reach, from whom it is withheld. |
| minimum detectable effect (MDE) | The smallest true effect a test of this size detects with 80% power at α = 0.05, two-sided (Module 6). |

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · Rules of thumb | 1 · Last-click, first-click and linear credit | 8 |
| B · The journey as a Markov chain | 2 · The transition matrix | 8 |
| | *Run: the probability of reaching conversion (provided solver and check)* | — |
| | 3 · Removal effects | 8 |
| | *Run: Markov credit (provided)* | — |
| C · Shapley values | 4 · Exact Shapley values | 9 |
| | *Run: coalition conversion rates and Shapley credit (provided)* | — |
| D · Credit against the truth | *Run: the credit table and who each channel reaches (provided)* | — |
| | 5 · Score the methods | 7 |
| | *Run: is the gap noise? 95% bootstrap intervals (provided)* | — |
| **Exercises** | | **40** |
| Decision | Move budget or test first? | 5 |

The minutes of each exercise include reading the provided cells immediately before it, so those cells print
short summaries, not long tables (the full tables stay in variables for anyone who wants them).

Provided before Exercise 1 (nothing in it, or anywhere before Exercise 5's Predict, prints
`truth["incremental_conversions"]`): load `journeys` and `truth`; print counts (journeys, conversions,
conversion share) and ten example paths; a table with, for each channel, the share of journeys it appears in
and how often it is the first and the last touch in the journeys with two or more channels that it appears in
(Exercise 1's Predict and Explain use it); define `CHANNELS` and `TINY`, a five-journey example used by
Exercises 1–3:

| Journey | Path | Converted |
|---|---|---|
| 1 | A > B | 1 |
| 2 | A | 0 |
| 3 | B | 1 |
| 4 | B > A | 0 |
| 5 | A > A > B | 1 |

### Exercise 1 · Last-click, first-click and linear credit (8 minutes)

- **Predict.** Retargeting reaches about one user in six, appears in about a quarter of the tracked journeys
  (the table above; committed 24.1%), and its touches come late in the month (the same table: how often each
  channel is the first and the last touch). Under last-click, what share of the conversions will retargeting
  get: under 20%, 20 to 30%, or over 30%? Write your guess.
  (A guess from reach alone lands in the first two bins. The committed answer, 36.9% (prototype 38%), is 6.9
  points above the top boundary, so the bins stay; over eight seeds it is 35.6–38.6%. A test guards it.)
- **Function.** `heuristic_credit(journeys: pd.DataFrame, rule: str, channels: list[str]) -> pd.Series`:
  credited conversions per channel (index `channels`, zeros included) for `rule` in `{"last", "first",
  "linear"}`; only converting journeys give credit; linear gives each touch 1/L_i. Hints in the stub: split
  paths with `.str.split(" > ")`; for linear, `explode` the touches and give each 1/L_i.
- **Checkpoint.** On `TINY` (worked by hand): last {A: 0, B: 3}; first {A: 2, B: 1}; linear {A: 7/6, B: 11/6}
  to 1e-12. On `journeys`, for each rule: credit adds up to the number of converting journeys
  (`truth["conversions_in_journeys"]`) to 1e-9 and no value is negative. First checkpoint; runs in under a
  second (prototype: all three rules 0.10 s; the FULL laptop run's checkpoint cells took 0.23 s in all).
- **Failure messages** (the four parts of `briefs/_lab-standard.md` §5, each ending with
  `workshop.use_reference(1)`). Credit that adds up to the number of journeys (prototype 69,410) instead of
  the converting journeys: "only converting journeys give credit; keep the rows with `converted == 1`".
  Linear on `TINY` = {A: 1, B: 2}: "linear here is per touch, not per distinct channel; journey 5 (A > A > B)
  gives A two shares of 1/3". A channel missing or NaN: "return every channel in `channels`, with 0 where it
  got no credit (`reindex(channels, fill_value=0)`)".
- **Explain.** Compare with your guess (committed: 36.9%). Last-click and first-click disagree most about
  retargeting and display. Which feature of the journeys decides it: when each channel's touches happen, how
  many users each channel reaches, or how often reached users convert? Point to the table of opening and
  closing touches (committed: display opens 77% and closes 17% of its multi-channel journeys; retargeting
  opens 12% and closes 82% (prototype 83%)).
  *Why:* a positional rule credits whatever sits at that position; display is early, retargeting late. Neither
  rule asks whether the touch changed the outcome.

### Exercise 2 · The transition matrix (8 minutes)

*Provided before it:* `journey_states(path: str, converted: int) -> list[str]`, which turns one journey into
its states, shown on `TINY`'s journey 1: `["start", "A", "B", "conv"]`. The string handling is not the lesson;
counting transitions is.

- **Predict.** In the display row, which next state will be most likely: another display touch, a touch from
  another channel (the four others together), conversion, or the end of the journey without conversion
  (null)? (Committed display row: display → display 0.508, the other channels together 0.265, null 0.222,
  conv 0.005; the answer leads the next option by 0.24 on every one of eight seeds, so the prompt stays; the
  notebook says "the four others together" so that the options do not overlap. A test guards it.)
- **Function.** `transition_matrix(journeys: pd.DataFrame, channels: list[str]) -> pd.DataFrame`: the
  first-order transition probabilities with index and columns `["start", *channels, "conv", "null"]`; each
  journey adds start → first touch, touch → next touch, last touch → conv or null; conv and null map to
  themselves with probability 1. Hints in the stub: use `journey_states`; count consecutive pairs
  (`collections.Counter` or `pd.crosstab`); add the conv → conv and null → null entries; divide each row by
  its total.
- **Checkpoint.** On `TINY` (by hand): start → (A 3/5, B 2/5); A → (A 1/5, B 2/5, null 2/5); B → (A 1/4,
  conv 3/4); conv and null absorbing; to 1e-12. On `journeys`: every row sums to 1 (1e-12), the start column is
  zero, labels in the stated order, conv and null rows are unit rows.
- **Failure messages** (four parts, ending with `workshop.use_reference(2)`). A row that does not sum to 1:
  "divide each row by its own total (`M.div(M.sum(axis=1), axis=0)`), not by column totals or the grand
  total". NaN in the conv or null row: "a row of zeros divided by its total of 0 gives NaN; set conv → conv
  and null → null to 1 before dividing". `TINY`'s start row wrong: "each journey adds exactly one start →
  first-touch transition". Labels out of order: "reindex rows and columns to `["start", *channels, "conv",
  "null"]`".
- **Explain.** Compare with your guess (committed: display → display 0.508). A run of display touches shows up
  as the display → display entry, a **self-loop**. Divide the other entries of the display row by
  1 − p(display → display): these are the chances of each next state once the journey leaves display. Does the
  self-loop change where the journey goes after display, or only how long it stays there?
  *Why:* once the journey leaves c, it goes to d with probability p_cd / (1 − p_cc), whatever p_cc is; a
  self-loop only delays the next step. So the probability of eventually converting from c (x_c, Exercise 3)
  does not depend on repeated touches in a row: x_c = p_cc x_c + Σ_{d≠c} p_cd x_d gives
  x_c = Σ_{d≠c} p_cd x_d / (1 − p_cc). On the committed data, collapsing repeated touches leaves every removal
  effect unchanged (largest difference 3e-16; tested to 1e-9): Markov credit does not reward a channel for
  repeating itself.

*Provided after Exercise 2 · The probability of reaching conversion.* One paragraph and a provided function,
read inside Exercise 3's minutes. x_a, the probability that a journey now in state a eventually converts,
satisfies x_a = p_a,conv + Σ_b p_ab x_b over the transient states b: take one step, then carry on from where
you land. In matrix form x = Q x + r, so x = (I − Q)⁻¹ r (Notation). `conversion_probability(P: pd.DataFrame)
-> float` returns x_start with `np.linalg.solve(np.eye(len(T)) - Q, r)`, with T = start and the channels,
Q = P[T × T] and r = P[T, conv]. A labeled check (`workshop.checkpoint(label="chain")`): on `TINY`, p = 3/5
(worked by hand in a folded block); on `journeys`, p equals the share of converting journeys to 1e-9. One
sentence states the identity ("a chain estimated by counting reproduces the observed conversion share
exactly"); a `<details>` block gives the reason: with n_a = Σ_b n_ab departures from state a, the chain's
equations are n_a x_a = Σ_b n_ab x_b. Summed over the transient states, every channel term cancels (each
channel state is entered as often as it is left), leaving n_start · x_start = Y with n_start = N. The check
can fail only through the learner's `transition_matrix`; its message: "The chain's conversion probability is
0.071, but 4.35% of journeys converted, and a chain estimated by counting must reproduce that share exactly.
Your transition_matrix passed Checkpoint 2, so it counts transitions differently from the reference: check
that every journey adds exactly one start → first-touch transition and exactly one last-touch → conv or null
transition. To move on now, run `workshop.use_reference(2)` and rerun from Exercise 2's checkpoint."

### Exercise 3 · Removal effects (8 minutes)

The Markdown before the Predict defines the removal effect (Terms) and RE_c = 1 − p₋c / p, where p₋c is the
chain's conversion probability with c removed.

- **Predict.** Will the five channels' removal effects add up to less than 1, about 1, or more than 1? Write
  your guess.
- **Function.** `removal_effects(P: pd.DataFrame, channels: list[str]) -> pd.Series`: RE_c for each channel
  (index `channels`), using the provided `conversion_probability`. For channel c, work on a copy of P, set row
  c to 0 and `P.loc[c, "null"] = 1` (every journey that reaches c ends there without converting; the same
  counterfactual as x_c = 0 in Notation), compute p₋c, and return 1 − p₋c / p. The function must not change
  `P`. Provided after it: Markov credit Y · RE_c / Σ_k RE_k, printed per channel.
- **Checkpoint.** On `TINY` (by hand): p = 3/5, p₋A = 3/10, p₋B = 0, so RE = {A: 1/2, B: 1}, to 1e-12; `P` is
  unchanged after the call. On `journeys`: every RE between 0 and 1, and equal to the reference to 1e-9. After
  the provided Markov credit cell, a labeled check: `TINY` credit {A: 1, B: 2} (Y = 3, removal effects adding
  up to 1.5).
- **Failure messages** (four parts, ending with `workshop.use_reference(3)`; each wrong value below was
  computed on `TINY` during review). RE_A = −0.67: "a negative removal effect usually means the channel was
  deleted and the other transitions renormalized, which reroutes journeys around it instead of stopping them;
  set row c to 0 and `P.loc[c, 'null'] = 1`". RE_B = 0 instead of 1: "the removal effect is 1 − p₋c / p, not
  p₋c / p; removing B loses every conversion in TINY". `P` changed by the call: "your function changed P, so
  each channel after the first was removed from an already changed chain; work on `P.copy()` for each
  channel".
- **Explain.** Compare with your guess (committed: 1.97). In `TINY` the removal effects are 1/2 and 1, adding
  up to 1.5. Which converting journeys pass through both A and B, and what happens to them when A is removed,
  and when B is removed? Then: what does dividing each removal effect by their sum (the Markov credit above)
  assume?
  *Why:* removing one channel breaks every path through it, and a converting path through three channels is
  "lost" in three removal scenarios, so the effects overlap; dividing by their sum is a choice, not a law. The
  removal effect asks what the *fitted chain* predicts if journeys stopped at c, which treats "on the path to
  conversion" as "caused the conversion".

### Exercise 4 · Exact Shapley values (9 minutes)

The Markdown before the Predict defines the Shapley value, coalition, Shapley's three axioms and the null-player property (Terms), and
says in words what the game will be for journeys (v(S) = the conversion rate of journeys whose set of
channels is exactly S, v(∅) = 0), so that the Explain can use it.

- **Predict.** In the three-player game below, C adds nothing to any coalition. Write C's Shapley value (0,
  between 0 and 0.10, or above 0.10), and say whether A and B will split v(ABC) = 0.40 equally (yes or no).

  | S | ∅ | A | B | C | AB | AC | BC | ABC |
  |---|---|---|---|---|---|---|---|---|
  | v(S) | 0 | 0.10 | 0.20 | 0 | 0.40 | 0.10 | 0.20 | 0.40 |

- **Function.** `shapley_values(players: list[str], v: Callable[[frozenset], float]) -> dict[str, float]`:
  exact enumeration of the formula in Notation (`itertools.combinations`, `math.factorial`).
- **Checkpoint.** Two players with v(A) = 0.10, v(B) = 0.05, v(AB) = 0.20: φ = {A: 0.125, B: 0.075}. The game
  above: φ = {A: 0.15, B: 0.25, C: 0} (by hand, averaging over the six orders). On a provided random five-player
  game (seed shown): efficiency Σφ = v(N) − v(∅) to 1e-12; equal to a provided brute-force average over all
  120 orders to 1e-12; swapping two players in v swaps their values.
- **Failure messages** (four parts, ending with `workshop.use_reference(4)`; the wrong values were computed
  during review). The two-player example giving {A: 0.175, B: 0.100}, which adds up to 0.275 instead of
  v(AB) − v(∅) = 0.20: "the weight is |T|! (n − |T| − 1)! / n! for a coalition T that leaves c out;
  (n − |T|)! in place of (n − |T| − 1)! gives these numbers". C ≠ 0 in the three-player game: "sum over the
  coalitions T that do not contain c, and compare v(T ∪ {c}) with v(T)". A `KeyError` or `TypeError` from v:
  "v takes a frozenset; call `v(frozenset(T) | {c})`" (the checkpoint catches these and prints the message
  instead of a bare traceback).
- **Explain.** Compare with your guess. Your function passed the efficiency check: a set's Shapley values add
  up to v(S) − v(∅). For journeys, v(S) is the conversion rate of journeys whose set of channels is exactly S,
  and v(∅) = 0 because users with no touch are not in the data. Suppose users would convert at 1% with no
  marketing at all. Under Shapley, who gets the credit for those conversions: nobody, the channels on each
  converting journey, or only its last channel?
  *Why:* the channels on the journey. With v(∅) = 0 the whole conversion rate of each set, baseline included,
  is shared out. Shapley's properties (efficiency, symmetry, a player who adds nothing gets nothing) are rules
  for sharing a value; they say nothing about whether v measures what a channel causes. That depends on v, and
  v here is an observed conversion rate.

*Provided after Exercise 4:* the table of v(S) for the 31 non-empty channel sets (rate, conversions,
journeys; the cell stops with a clear message if any set needed by a converting journey has no journeys,
instead of filling in 0) and the Shapley credit, calling the learner's `shapley_values` once per set. It
prints a summary, not the 31 rows: the number of sets, the fewest journeys in a set, the smallest φ over all
sets, and that each set's shares sum to 1; the table stays in `coalitions` for anyone who wants it.
Committed: all 31 sets occur (smallest 177 journeys); the smallest φ over all sets is +0.0118, so no
negative shares (the prototype found −0.004 on one of five seeds at 60,000 users: report negative values,
never clip them). All 31 sets occur on all eight seeds at 100,000 users.

### Exercise 5 · Score the methods against the truth (7 minutes)

*Provided before it (Part D's opening, read inside this exercise's minutes):* `credit`, a DataFrame (index
`CHANNELS`, columns `last, first, linear, markov, shapley`) of credited conversions, shown; and
`incremental = pd.Series(truth["incremental_conversions"])`, loaded but not printed until the checkpoint.
One sentence on why the lab compares shares: credit always adds up to the tracked conversions, while
incremental conversions are fewer and do not add up across channels, so each method's credit shares are
compared with the true incremental shares. Then one sentence and a plot: "Look at the conversion rate of
users each channel reached against those it did not reach" (from `truth`, because users with no touch are not
in `journeys`; committed retargeting 8.3% against 2.4%).

- **Predict.** Under Markov and Shapley, retargeting gets about as much credit as display or social (the
  table above; committed 700 (prototype 708) and 744 conversions), and the users it reached converted far more often than
  those it did not reach (the plot). How many conversions did retargeting truly cause: about as many as its
  Markov credit, about half of it, or close to none? Write your guess.
- **Function.** `credit_vs_truth(credit: pd.DataFrame, incremental: pd.Series) -> pd.DataFrame`, one row per
  method with `tvd` (D_m), `largest_gap_channel` and `largest_gap` (credit share − true share for that channel,
  signed).
- **Checkpoint.** Hand example: credit m1 {A: 5, B: 3, C: 2}, m2 {A: 2, B: 2, C: 6}; incremental {A: 6, B: 2,
  C: 0}: m1 → tvd 0.25, largest gap A, −0.25; m2 → tvd 0.60, largest gap C, +0.60. On the data: equals the
  reference to 1e-9, and `largest_gap_channel` is retargeting for `markov` and `shapley` (true on the committed
  seed and on seeds 11 to 17). After it passes, the cell prints `incremental` beside each method's credit.
- **Failure messages** (four parts, ending with `workshop.use_reference(5)`; the wrong values were computed
  during review). m1's tvd = 2.0: "compare shares, not counts; divide each column of `credit` by its total
  and `incremental` by its total". m1's tvd = 0.50: "the share error has a factor ½, because each misplaced
  share is counted once where it is missing and once where it is extra". m1's largest gap reported as C
  (+0.20) instead of A (−0.25): "pick the channel with the largest absolute gap, then report its gap with
  its sign".
- **Explain.** Compare with your guess (truth: 0 for retargeting). Look at the provided plot of mean
  true intent among the users each channel reaches (committed: display 0.00, social 0.14, email 0.64, search
  0.86, retargeting 1.05 standard deviations). Which mechanism earns retargeting its credit: it sits late in
  journeys, it reaches users who would buy anyway, or it has the most touches? Rule out what you can first:
  Shapley's v uses only the set of channels on a journey, not their order or number, and Exercise 2 showed
  that repeated touches in a row do not change where a journey goes next.
  *Why:* mostly selection. Position and touch count cannot explain Shapley's credit, and repeated touches in
  a row cannot explain Markov's. Retargeting reaches users who already intend to buy, so journeys that contain it
  convert more often, and every method reads that as credit. The stretch removes the targeting and shows how
  much remains: the efficiency rule still shares out conversions that would have happened anyway. Module 6's
  experiment avoided both problems by comparing with a randomized control group.

*Provided after Exercise 5 · Is the gap noise?* Resample journeys with replacement (`N_BOOT` = 200 FULL, 50
QUICK, seed shown), recompute every method's credit shares (counts vectorized over distinct paths, calling the
provided `conversion_probability` and the learner's `removal_effects` and `shapley_values`), and plot each
method's share with its 95% bootstrap percentile interval and the true share marked. Sentence before the
plot: "Look at whether any true share (black mark) falls inside a method's interval." Sentence after it:
these intervals show sampling noise only; a method that is wrong in the same direction on every resample
(bias) stays wrong however many journeys you have. Committed: 0 of 25 inside at 200 resamples (FULL) and at 50
(QUICK); the cell, with its plot, took 1.22 s in the recorded laptop FULL run (prototype: 200 resamples in
0.71 s).

### Decision · Move budget or test first? (5 minutes)

The Markdown before the cell states the rule (below) before any number is printed, as Module 6's lab does,
and adds: "Decide as you would at work, where there is no `truth`: use only the credit, the spend, the
margin and the test's minimum detectable effect. The reveal comes after your sentence."

Provided cell, in the standard order:

- **Number:** for the four paid channels that can scale (display, social, search, retargeting; email is
  list-limited), cost per credited conversion = spend ÷ credit under Markov and under Shapley, each with its 95%
  bootstrap percentile interval; the channel each method says is cheapest. Committed (FULL, 200 resamples):
  retargeting under both, $3.57 (95% interval $3.40–3.73) Markov (prototype $3.53, $3.39–3.71) and $3.36
  ($3.13–3.68) Shapley (prototype $3.03–3.70), against $5.30–5.42 (Markov; prototype $5.35–5.44) and
  $3.88–6.42 (Shapley) for the others. Margin per conversion $9, so by attribution every paid channel pays back.
  Then the power of a holdout, with Module 6's MDE formula for unequal arms and σ² = p(1 − p) for a conversion
  rate (α = 0.05 two-sided, 80% power): withholding retargeting from 20% of the users it reaches for one month detects a change of
  (z₀.₉₇₅ + z₀.₈) · √(p(1 − p)(1/n₁ + 1/n₀)) in conversion rate. Committed: 16,741 users reached (13,393 served,
  3,348 held out), conversion rate 8.30%, MDE 1.49 points = about 200 conversions among served users, about 250
  scaled to all reached users, against 700 credited (prototype 708); margin at risk if the credit were real:
  20% × 700 × $9 = $1,260 (prototype $1,270). Also per channel, the break-even: spend ÷ $9, the
  incremental conversions a month the channel must cause to pay back (retargeting: $2,500 ÷ $9 ≈ 278). Beside
  the bootstrap intervals, one line: they cover sampling noise in the credit, not the bias Exercise 5 found.
- **Rule** (stated before the numbers): "Attribution credit cannot separate a channel's effect from the intent
  of the users it reaches. Move budget toward a channel only when an experiment (Modules 6 and 7) or a
  lift-calibrated MMM (Module 9) shows that its cost per incremental conversion is below the $9 margin per
  conversion. Run a holdout first when it can tell a channel that pays back from one that does not: its MDE,
  scaled to all the users the channel reaches, must be below the break-even (spend ÷ margin per conversion).
  If it is not, make the holdout larger or longer before running it."
- **Recommendation:** the cell prints what the rule gives (committed: test retargeting first, because the
  holdout's MDE scaled to all reached users, about 250 conversions a month, is below its break-even of about
  278; comparing the MDE with the 700 credited would be a weaker bar that a test with an MDE of 600 would also
  pass, though it could not tell a break-even channel from a useless one), the margin at risk, what would
  change it (a holdout showing a cost per incremental conversion below $9, that is more incremental
  conversions a month than the break-even), and the mode ("QUICK changes only the bootstrap intervals").
  Then the learner writes one sentence: test or move, which channel, the holdout share and its MDE, the
  margin at risk, and what result would justify moving budget. The Markdown gives a template to fill in:

  ```text
  Your sentence: Before moving budget to ____ (cheapest by attribution, $____ per credited conversion),
  withhold it from ____% of the users it reaches for one month. The test can detect ____ points (about
  ____ conversions) and puts about $____ of margin at risk. Move budget only if it shows more than ____
  incremental conversions a month (a cost per incremental conversion below $9).
  ```

Then a reveal cell: true incremental conversions, true cost per incremental conversion against the $9 margin
(committed, as in the prototype: display $11.17, social $7.81, search $7.35, email $2.63, retargeting none, because Δ = 0), the
channel where credit and truth disagree most (retargeting), and what moving budget to it would have bought
(nothing).

## Stretch (optional)

1. **Same reach, random exposure.** `from mktstats.synth.mta import mta_journeys`;
   `res = mta_journeys(random_exposure=True)`; rerun the five methods and `credit_vs_truth` against
   `res.truth["incremental_conversions"]`. Committed: Shapley D 0.143, Markov 0.227; retargeting keeps 10.6%
   (Shapley) and 14.9% (Markov) of credit (eight seeds: see claim 5). Explain what remains (efficiency). Ask why
   Shapley gains more than Markov here; a likely reason, to check rather than assert: its v compares conversion
   rates of channel sets, and random exposure makes those sets comparable.
2. **Run the holdout.** `res = mta_journeys(holdout={"retargeting": 0.2})`; with `res.holdout`, estimate
   retargeting's incremental conversions among served users with a 95% CI as in Module 6's Exercise 6; compare
   with its Markov credit and its truth. Committed (the holdout has its own random stream, so these differ from
   the prototype's 37 (−100 to 174) and 186 (−8 to 380)): 20% holdout → −131 (95% CI −275 to 13); 10% → −60
   (−271 to 151). Both intervals include 0 and exclude the 700 credited; the 20% interval only just includes 0
   from below, a sampling fluctuation around a true 0, which the stretch can discuss.
3. **A count-based value function.** With v(S) = conversions of journeys that use only channels in S, show that
   the Shapley values equal "split each conversion equally among the distinct channels on its path" (the game
   is a sum of unanimity games; committed: equal to 1e-13).

## Run time (measured on a laptop and in Docker; no Colab record yet)

FULL runs recorded at commit `ba1b3e8` (worked mode, including the five verification copies of the
checkpoints; Run all of the notebook, not a Colab time):

| | macOS, native (`runs/2026-10-10-macos-arm64-python-16-multi-touch-attribution.json`) | Docker image `mktstats-env:extensions`, `--cpus 2` (`runs/2026-10-10-docker-arm64-python-16-multi-touch-attribution.json`) | Colab target |
|---|---|---|---|
| Install | 0.0 s (a checkout installs nothing) | 0.0 s (the image has everything) | ≤ 3 min. Not measured: same install list as Module 6 (`[numpy, pandas, scipy, matplotlib]`), whose three Colab records at `v2026.10.1` show 14.8–17.0 s (`runs/2026-10-10-colab-py-python-06-experiments.json`) |
| Slowest cell | 1.29 s (the bootstrap, cell-055) | 2.33 s (imports, cell-001; the bootstrap took 1.4 s in an unrecorded run of the same commit) | ≤ 4 min |
| Whole notebook | 5.7 s | 7.6 s | ≤ 15 min |

QUICK changes only `N_BOOT` (50 instead of 200); every credit number, checkpoint and the decision are the
same in both modes, and the notebook says so. QUICK runs were made (macOS, 4.5 s) but not recorded.

## Known pitfalls

- Credit only converting journeys. Credit sums must equal the converting journeys, not all journeys.
- Linear credit here is per touch; a per-distinct-channel version gives different numbers. State which.
- Removal means "transitions into c end without conversion" (x_c = 0). Deleting c and renormalizing the other
  transitions is a different counterfactual; do not mix the two.
- conv and null must be absorbing, or be left out of Q; start has no incoming transitions.
- v(S) must not silently become 0 for a set with no journeys (see Exercise 4's provided cell).
- Exact Shapley enumeration grows as 2^K: 16 coalitions per player for 5 channels; 20 channels would need
  sampling of orders.
- The truth counts all 100,000 users, including the converters with no tracked touch (prototype 405), while
  credit spreads only the tracked conversions. Incremental conversions do not add up across channels.

## APIs verified (and how)

No attribution library is used: `numpy.linalg.solve`, `itertools.combinations`, `math.factorial`, pandas
grouping and `scipy.stats.norm.ppf` (SciPy 1.16.3, pinned; the same calls as Module 6) are enough, and the
prototype ran them in the workshop's venv on 2026-10-10. ChannelAttribution was checked and not adopted
(References): on CRAN, 2.2.5 offers first-, last- and linear-touch heuristics and k-order Markov models
(`markov_model(Data, var_path, var_conv, var_value=NULL, var_null=NULL, order=1, nsim_start=1e5, …,
out_more=FALSE, …)`; `out_more=TRUE` returns transition probabilities and removal effects) and documents no
Shapley function; on PyPI, 2.2.5 is a source distribution only (Cython and C++ sources, no wheels), so it would
compile during the install cell, which the workshop avoids for Colab labs (DECISIONS.md: causalml was not
used for the same reason). The lab teaches the methods by writing them; a comparison with ChannelAttribution
would fit an R notebook, with the package added to the R image.

## Names used in the lab

`journeys`, `truth` (from `truth_mta.json`), `CHANNELS`, `TINY`, `P` (transition matrix), `credit`
(channels × methods), `incremental`, `coalitions` (the v(S) table), `SEED`, `N_BOOT`, `QUICK`.
Learner functions: `heuristic_credit`, `transition_matrix`, `removal_effects`, `shapley_values`,
`credit_vs_truth`. Provided helpers: `journey_states`, `conversion_probability`.

## References (verified)

Read on 2026-10-10 unless stated. For `references.qmd` (proposed entries are in the Academic Director's report).

- Shao, X., & Li, L. (2011). Data-driven multi-touch attribution models. *Proceedings of the 17th ACM SIGKDD
  International Conference on Knowledge Discovery and Data Mining (KDD '11)*, 258–264.
  <https://doi.org/10.1145/2020408.2020453>. *Verified (2026-10-10)*: Crossref record; abstract via OpenAlex
  (the ACM page refused automated access). Supports: multi-touch attribution gives credit to more than one ad
  according to its contribution, instead of all to the last ad; the authors frame it as interpreting each
  interaction's influence accurately and stably, not only classifying users.
- Dalessandro, B., Perlich, C., Stitelman, O., & Provost, F. (2012). Causally motivated attribution for online
  advertising. *Proceedings of the Sixth International Workshop on Data Mining for Online Advertising and
  Internet Economy (ADKDD '12)*, 1–9. <https://doi.org/10.1145/2351356.2351363>. *Verified (2026-10-10)*:
  Crossref; abstract via OpenAlex. Supports: attribution posed as a causal estimation problem; approximate
  methods that draw on cooperative game theory; where causal assumptions are violated, these approximations are
  variable-importance measures.
- Anderl, E., Becker, I., von Wangenheim, F., & Schumann, J. H. (2016). Mapping the customer journey: Lessons
  learned from graph-based online attribution modeling. *International Journal of Research in Marketing*,
  33(3), 457–474. <https://doi.org/10.1016/j.ijresmar.2016.03.001>. *Verified (2026-10-10)*: Crossref; abstract
  on RePEc/IDEAS (<https://ideas.repec.org/a/eee/ijrema/v33y2016i3p457-474.html>); ScienceDirect refused
  automated access, full text not read. Supports: customer paths modeled as first- and higher-order Markov
  walks, applied to four large data sets with at least seven online channels each; results differ
  substantially from last-click. The removal-effect formula in Notation is this brief's own statement.
- Shapley, L. S. (1953). A value for n-person games. In *Contributions to the Theory of Games (AM-28), Volume
  II* (pp. 307–318 as Crossref lists them). Princeton University Press.
  <https://doi.org/10.1515/9781400881970-018>. Earlier as RAND paper P-295 (1952, 15 pages),
  <https://www.rand.org/pubs/papers/P295.html>. *Verified (2026-10-10)*: Crossref record and the RAND page,
  whose abstract says the value "is deduced from a set of three axioms, having simple intuitive
  interpretations." Full text not read.
- Zhao, K., Mahboobi, S. H., & Bagheri, S. R. (2018). Shapley value methods for attribution modeling in online
  advertising. arXiv:1804.05327. <https://arxiv.org/abs/1804.05327>. *Verified (2026-10-10)*, abstract only:
  Shapley-value attribution across channels, a simplified formula, and an "ordered" variant that uses the order
  of touches. (Briefing mention; full text not read, so its value function is not described here.)
- ChannelAttribution (R package) 2.2.5, published 2026-06-08, GPL-3, maintainer Davide Altomare.
  <https://CRAN.R-project.org/package=ChannelAttribution>; reference manual
  <https://cran.r-project.org/web/packages/ChannelAttribution/refman/ChannelAttribution.html>.
  *Verified (2026-10-10)*: k-order Markov model, first-/last-/linear-touch heuristics, implemented in C++;
  functions `markov_model`, `auto_markov_model`, `choose_order`, `heuristic_models`, `transition_matrix`,
  `install_pro`; removal effects with `out_more = TRUE`; no Shapley function.
- ChannelAttribution (Python package) 2.2.5, uploaded to PyPI 2026-06-08. <https://pypi.org/project/ChannelAttribution/>.
  *Verified (2026-10-10)* through the PyPI JSON API and the source archive's file list: only
  `channelattribution-2.2.5.tar.gz` (Cython `.pyx` and C++ sources), `Requires-Python >=3.8`,
  GPL-3.0-or-later; defines `heuristic_models`, `markov_model`, `transition_matrix`, `choose_order`,
  `auto_markov_model`; no Shapley function.
- Gordon, B. R., Zettelmeyer, F., Bhargava, N., & Chapsky, D. (2019), already in `references.qmd`
  (Verified 2026-10-09): observational methods often missed the lift measured by randomized experiments
  (Module 6's evidence; cite it again in this module's briefing).

## Open items for the Technical Expert (all done 2026-10-10; see "Build notes")

- Build `mta_journeys` from the specification and re-measure every prototype number on the committed seed; the
  brief's claims 2–5 are prototype results until then. If a claim fails on the committed data, tell the
  Academic Director before changing the parameters. *Done: every claim holds on the committed data; claim 3
  is worded more carefully above.*
- CSV size: 100,000 users give about 3.1 MB (2.6 MB without `true_intent`). At 60,000 users (1.85 MB) the
  lesson held on five seeds, but one seed gave a negative Shapley value; prefer 100,000 unless size is a
  problem. *Done: 100,000 users, one letter per touch in the CSV, 1.34 MB.*
- Seed 2036 follows the core pattern (2026–2032); check it does not collide with another extension's seed
  (harmless if it does, but confusing). *Done: the extensions use 2033–2037 (Modules 13–17); no collision.*
- `modules.m16.deps`: add `src/mktstats/synth/mta.py`, `data/synthetic/mta_journeys.csv` and
  `data/synthetic/truth_mta.json`; install list `[numpy, pandas, scipy, matplotlib]` (SciPy for the MDE).
- Write the bootstrap and the provided Shapley-credit cell so they call the learner's functions
  (`removal_effects`, `shapley_values`; after their checkpoints passed or `use_reference`), with counts
  vectorized over distinct paths.
- Checkpoint failure messages need the four parts of `briefs/_lab-standard.md` §5. Each exercise above lists
  its likely wrong answers and what the message says about them; the labeled `chain` check after Exercise 2
  has its full message written out. Add a `mktstats.checks`-style test, or a verify-mode run, that feeds each
  listed wrong answer and expects the message.

## Pedagogy review (2026-10-10)

Checked against `briefs/_lab-standard.md` §11 before any code exists. What changed, and why:

1. **The linear solve is provided; the learner writes the removal effect.** Exercise 3 asked for two new
   ideas at once (absorption probabilities by a linear solve, and the removal counterfactual) and its
   Explain asked two questions, one of them a proof. For analysts who know regression but not Markov chains,
   the solve is machinery; the removal counterfactual is the attribution idea, and its common mistake
   (delete the channel and renormalize, which gives a negative removal effect on `TINY`) is worth meeting at
   a checkpoint. `conversion_probability(P)` is now a provided cell with a labeled check (p equals the
   observed share; the proof moved into its folded block). Exercise 3 is `removal_effects(P, channels)`.
2. **Predicts ask about what has not been shown yet.** Exercise 5 asked for retargeting's Markov and Shapley
   credit share, which the provided cells after Exercises 3 and 4 already print. It now asks for
   retargeting's true incremental conversions, which nothing shows before the checkpoint. Exercise 3's
   Predict now asks about the sum of removal effects, which its Explain then uses.
3. **Explains use only what the learner has seen.** Exercise 2's Explain needed x (defined only in Exercise
   3); it now works from the display row. Exercise 4's Explain needed the journey value function before the
   notebook defined it; Part C now defines v(S) in words before Exercise 4, and the question has three
   options.
4. **Budget.** 8 + 8 + 8 + 9 + 7 = 40 (limit 40). Exercise 1 goes from 7 to 8 minutes (first exercise,
   three rules, two provided tables to read); Exercise 3 from 9 to 8 (one idea instead of two, with the
   solver cell read inside it). Provided cells between exercises print summaries, not long tables, because
   their reading time comes out of the next exercise's minutes.
5. **Failure messages** for every checkpoint, with the likely wrong answers computed on the hand examples.
6. **Decision.** The rule comes first in Markdown; "what would change it" gets a number (break-even =
   spend ÷ margin); the learner fills in a one-sentence template; the cell says the bootstrap intervals
   cover noise, not bias.
7. **Exercise 1's bins** moved from 20 / 35% to 20 / 30%, so that the prototype's 38% is not near a
   boundary and a guess from reach alone falls in a different bin from the answer.

## Build notes (Technical Expert, 2026-10-10)

What the build changed in this brief, and why. Everything else above stands as written.

1. **CSV encoding.** The committed `mta_journeys.csv` writes one letter per touch (`truth["path_codes"]`),
   1.34 MB instead of 3.5 MB; the lab spells the paths out in its load cell. Fewer users would also have
   shrunk the file, but the decision compares a holdout MDE, which in conversions grows with the square root
   of the market, with a break-even fixed by spend. Measured at 60,000 users (seeds 2036, 11, 12): with the
   same spend, display costs $9.03–9.09 (Markov) and $10.39–12.48 (Shapley) per credited conversion, above
   the $9 margin, so "by attribution every paid channel pays back" fails; with spend scaled to the market
   (break-even 167), the 20% holdout's scaled MDE (194–198) is above it, so the recommendation flips to
   "make the holdout larger".
2. **Touch times** are drawn for every potential touch, so `random_exposure=True` reuses them. Intent,
   conversion, reach and frequency draws are those of the prototype, so the truth is identical; positional
   credit moved by a few conversions (table above).
3. **Numbers changed to the committed data:** distinct paths 8,473 → 8,441; last-click 465/508/503/409/1,135
   → 450/506/532/418/1,114; first-click 1,159/815/303/452/291 → 1,183/804/312/431/290; Markov
   745/655/460/453/708 → 738/654/472/456/700; retargeting's last-click share 38% → 36.9%; retargeting closes
   83% → 82%; collapsing repeats: "six decimals" → "3e-16"; Markov cost per credited conversion for
   retargeting $3.53 ($3.39–3.71) → $3.57 ($3.40–3.73), Shapley interval $3.03–3.70 → $3.13–3.68, other
   channels under Markov $5.35–5.44 → $5.30–5.42; credited retargeting 708 → 700 and margin at risk $1,270 →
   $1,260; holdout stretch 37 (−100 to 174) and 186 (−8 to 380) → −131 (−275 to 13) and −60 (−271 to 151);
   claims 2–5 now quote the committed seed and seeds 11–17 instead of the prototype's seeds; claim 3 now says
   "no better than" (Shapley beats linear or first-click on some seeds); count-based Shapley equality
   "six decimals" → "1e-13"; the run-time section now quotes the two run records. Unchanged on the committed
   data: the truth (Δ, expected Δ, 1,077, 3,425, 405, 1,260), linear and Shapley credit, the share errors to
   two decimals, 24% of journeys with retargeting, display opens 77% and closes 17%, retargeting opens 12%,
   display → display 0.51, removal effects adding up to 1.97, 31 sets with at least 177 journeys, smallest φ
   +0.012, 8.3% against 2.4%, the mean intents, 0 of 25 intervals, 16,741 users reached at 8.3%, MDE 1.5
   points (about 200 among served users, about 250 scaled), break-even 278, the true costs per incremental
   conversion and the random-exposure reach probabilities.
4. **Prompts.** Exercise 1's bins stay (36.9% is 6.9 points above 30%). Exercise 2's prompt stays: display →
   display (0.508) leads the four other channels together (0.265) by 0.24 on all eight seeds; the notebook
   adds "(the four others together)". A test in `tests/test_synth_mta.py` guards both.
5. **Checks** live in `src/mktstats/attribution.py` (`check_*`), so `tests/test_attribution.py` feeds every
   wrong answer listed above and expects its message. Two orderings were needed for that: the
   transition-matrix check tests an empty start row before row sums, and the Shapley check tests the null
   player before the two-player example (otherwise those messages were unreachable).
6. **Bootstrap** counts with `np.bincount`, not matrix products: under `docker run --cpus 2` on a 10-core
   laptop, OpenBLAS started ten threads per product and the bootstrap cell took 13.5 s instead of 1.4 s.
