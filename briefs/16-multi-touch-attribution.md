# Lab brief · Module 16 · Multi-touch attribution: Markov chains, Shapley values and incrementality

| | |
|---|---|
| Status | Brief for the Technical Expert, 2026-10-10, Academic Director. Written before the lab or its generator exists. Every number below marked *prototype* comes from a throwaway script run on a laptop while writing this brief (Apple M1 Pro, Python 3.13.2, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3); it is not a run record, and the Technical Expert re-measures everything after building the generator. Review: Pedagogy Expert. |
| Notebook | `labs/src/16-multi-touch-attribution.py` → `labs/python/16-multi-touch-attribution.ipynb` |
| Lab slot | 55 min (`modules.m16.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 40 (limit 40) + decision 5 + slack 5 = 55. Minutes are estimates until a pilot. |
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
| Lab reads | `pd.read_csv(repo_file("synthetic/mta_journeys.csv"))`, `json.loads(repo_file("synthetic/truth_mta.json").read_text())` (`mktstats.data.repo_file`; no change to `data.py`) |
| Table `journeys` | `user_id` (int), `path` (str, touches in time order joined by `" > "`), `converted` (0/1), `true_intent` (float, 4 decimals; truth, hidden in real data). One row per user with at least one touch, sorted by `user_id`. |
| Size (prototype, seed 2036) | 100,000 users → 69,410 journeys, 3,020 converting (4.35%); 8,473 distinct paths; mean 3.57 touches; CSV about 3.1 MB (the largest committed CSV today is 1.4 MB; see Open items) |

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
`seed`, `n_users`, `channels`, `path_separator`; `dgp` (every parameter above, with a one-paragraph
description); `n_journeys`, `conversions_total`, `conversions_in_journeys`, `conversions_untracked`;
`incremental_conversions` (Δ_c, common random numbers, integers), `incremental_conversions_expected`
(Σ_i [P_i − P_i^(−c)], floats), `incremental_conversions_all_channels` and its expected version;
`definition` (the Δ_c sentence above); diagnostics for the Explain cells: `reach_share`,
`mean_intent_reached`, `conversion_rate_reached`, `conversion_rate_not_reached`; `economics` (`spend`,
`mean_order_value` 30, `gross_margin` 0.30, `margin_per_conversion` 9, `scalable` false for email);
`true_cost_per_incremental_conversion` (spend ÷ Δ_c, `null` where Δ_c = 0); `random_exposure_reach_prob`.

## What the prototype showed (prototype numbers, seed 2036, 100,000 users)

Truth: Δ = display 358, social 448, search 340, email 114, retargeting 0 (expected values 339, 430, 330, 116,
0). Switching off every channel would lose 1,077 of 3,425 conversions; the single-channel Δ add up to 1,260,
not 1,077, because effects add on the log-odds scale, not on conversions. 405 conversions came from users with
no tracked touch.

| Channel | Last | First | Linear | Markov | Shapley | True Δ | True share |
|---|---|---|---|---|---|---|---|
| display | 465 | 1,159 | 797 | 745 | 624 | 358 | 0.284 |
| social | 508 | 815 | 685 | 655 | 654 | 448 | 0.356 |
| search | 503 | 303 | 441 | 460 | 644 | 340 | 0.270 |
| email | 409 | 452 | 369 | 453 | 354 | 114 | 0.090 |
| retargeting | 1,135 | 291 | 728 | 708 | 744 | 0 | 0 |
| Share error D_m | 0.42 | 0.26 | 0.27 | 0.29 | 0.27 | | |

What the brief may promise, because the prototype shows it (also on four more seeds, 11 to 14, and with
search given the largest effect instead):

1. Every method hands out all 3,020 tracked conversions; only about a third of conversions are incremental.
2. Retargeting, with no effect, gets 23–25% of the credit from both data-driven methods (Markov 0.231–0.236,
   Shapley 0.236–0.254 over five seeds) and is the channel with the largest share gap for last-click, linear,
   Markov and Shapley on every seed (first-click's largest gap is search).
3. The data-driven methods beat last-click (D 0.28–0.31 and 0.26–0.29 against 0.40–0.43) but not first-click
   (0.24–0.28) or linear (0.26–0.28). Markov credit is close to linear credit.
4. The gap is bias, not noise: none of the 25 method × channel true shares lies inside its 95% bootstrap
   percentile interval (200 resamples of journeys).
5. Two mechanisms, separated by `random_exposure=True` (seeds 2036, 11, 12): with the same reach but exposure
   independent of intent, Shapley's share error falls to 0.11–0.14 and Markov's to 0.21–0.23, and retargeting
   keeps 9–11% (Shapley) and 15% (Markov) of the credit. The rest is **efficiency**: each method must give every
   conversion, including those that would have happened anyway, to some channel on its path.

The lesson stated in the lab, no further: *attribution methods share credit by their own rules; when exposure
is targeted, neither the Markov removal effect nor Shapley values measure the conversions a channel causes.
Only a comparison with a counterfactual (a holdout, a geo test, a lift-calibrated MMM) does.*

## Model fits

None. Every method is counting, one small linear solve, or exact enumeration over 32 coalitions.

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · Rules of thumb | 1 · Last-click, first-click and linear credit | 7 |
| B · The journey as a Markov chain | 2 · The transition matrix | 8 |
| | 3 · Conversion probability and removal effects | 9 |
| C · Shapley values | 4 · Exact Shapley values | 9 |
| | *Run: coalition conversion rates and Shapley credit (provided)* | — |
| D · Credit against the truth | 5 · Score the methods | 7 |
| | *Run: is the gap noise? 95% bootstrap intervals (provided)* | — |
| **Exercises** | | **40** |
| Decision | Move budget or test first? | 5 |

Provided before Exercise 1: load `journeys` and `truth`; print counts (journeys, conversions, conversion share)
and ten example paths; define `CHANNELS` and `TINY`, a five-journey example used by Exercises 1–3:

| Journey | Path | Converted |
|---|---|---|
| 1 | A > B | 1 |
| 2 | A | 0 |
| 3 | B | 1 |
| 4 | B > A | 0 |
| 5 | A > A > B | 1 |

### Exercise 1 · Last-click, first-click and linear credit (7 minutes)

- **Predict.** Retargeting reaches about one user in six and its touches come late in the month (the provided
  table shows, for each channel, how often it opens and how often it closes the journeys with more than one
  channel that it appears in). Under last-click, what share of the conversions will retargeting get: under
  20%, 20 to 35%, or over 35%?
- **Function.** `heuristic_credit(journeys: pd.DataFrame, rule: str, channels: list[str]) -> pd.Series`:
  credited conversions per channel (index `channels`, zeros included) for `rule` in `{"last", "first",
  "linear"}`; only converting journeys give credit; linear gives each touch 1/L_i.
- **Checkpoint.** On `TINY` (worked by hand): last {A: 0, B: 3}; first {A: 2, B: 1}; linear {A: 7/6, B: 11/6}
  to 1e-12. On `journeys`, for each rule: credit adds up to the number of converting journeys
  (`truth["conversions_in_journeys"]`) to 1e-9 and no value is negative. First checkpoint; runs in under a
  second (prototype: all three rules 0.10 s).
- **Explain.** Compare with your guess (prototype: 38%). Last-click and first-click disagree most about
  retargeting and display. Which feature of the journeys decides it: when each channel's touches happen, how
  many users each channel reaches, or how often reached users convert? Point to the table of opening and
  closing touches (prototype: display opens 77% and closes 17% of its multi-channel journeys; retargeting
  opens 12% and closes 83%).
  *Why:* a positional rule credits whatever sits at that position; display is early, retargeting late. Neither
  rule asks whether the touch changed the outcome.

### Exercise 2 · The transition matrix (8 minutes)

- **Predict.** In the display row, which next state will be most likely: another display touch, a touch from
  another channel, conversion, or the end of the journey without conversion (null)?
- **Function.** `transition_matrix(journeys: pd.DataFrame, channels: list[str]) -> pd.DataFrame`: the
  first-order transition probabilities with index and columns `["start", *channels, "conv", "null"]`; each
  journey adds start → first touch, touch → next touch, last touch → conv or null; conv and null map to
  themselves with probability 1.
- **Checkpoint.** On `TINY` (by hand): start → (A 3/5, B 2/5); A → (A 1/5, B 2/5, null 2/5); B → (A 1/4,
  conv 3/4); conv and null absorbing; to 1e-12. On `journeys`: every row sums to 1 (1e-12), the start column is
  zero, labels in the stated order, conv and null rows are unit rows.
- **Explain.** Compare with your guess (prototype: display → display 0.51). A long run of display touches is a
  self-loop. Does a self-loop change the chance of eventually reaching conversion from display? Write x_display
  in terms of the other states to check. *Why:* x_c = p_cc x_c + Σ_{d≠c} p_cd x_d gives
  x_c = Σ_{d≠c} p_cd x_d / (1 − p_cc), so self-loops cancel; in the prototype, collapsing repeated touches left
  every removal effect unchanged to six decimals.

### Exercise 3 · Conversion probability and removal effects (9 minutes)

- **Predict.** Before solving: will the chain's probability of reaching conversion from start be lower than,
  about equal to, or exactly equal to the share of journeys that converted (printed above)?
- **Function.** `conversion_probability(P: pd.DataFrame, removed: str | None = None) -> float`: x_start from
  x = (I − Q)⁻¹ r (`np.linalg.solve`); with `removed=c`, row c of Q and entry c of r are set to 0 first.
  Provided after it: RE_c = 1 − p₋c / p for each channel and Markov credit Y · RE_c / Σ RE.
- **Checkpoint.** On `TINY` (by hand): p = 3/5, p₋A = 3/10, p₋B = 0, so RE = {A: 1/2, B: 1} and Markov credit
  {A: 1, B: 2}; to 1e-12. On `journeys`: p equals the share of converting journeys to 1e-9 (an exact identity,
  below), and p₋c ≤ p for every channel.
- **Explain.** Compare with your guess. Why must a chain estimated by counting reproduce the observed conversion
  share exactly? Then: the removal effects add up to about 2 (prototype 1.97), not 1. Why?
  *Why:* with n_a = Σ_b n_ab departures from state a, the chain's equations are n_a x_a = Σ_b n_ab x_b. Summed
  over the transient states, every channel term cancels (each channel state is entered as often as it is left),
  leaving n_start · x_start = Y with n_start = N. Removing one channel
  breaks every path through it, and a converting path through three channels is "lost" in three removal
  scenarios, so the effects overlap; dividing by their sum is a choice, not a law. The removal effect asks what
  the *fitted chain* predicts if journeys stopped at c, which treats "on the path to conversion" as "caused the
  conversion".

### Exercise 4 · Exact Shapley values (9 minutes)

- **Predict.** In the three-player game below, C adds nothing to any coalition. What Shapley value will C get,
  and will A and B split v(ABC) = 0.40 equally?

  | S | ∅ | A | B | C | AB | AC | BC | ABC |
  |---|---|---|---|---|---|---|---|---|
  | v(S) | 0 | 0.10 | 0.20 | 0 | 0.40 | 0.10 | 0.20 | 0.40 |

- **Function.** `shapley_values(players: list[str], v: Callable[[frozenset], float]) -> dict[str, float]`:
  exact enumeration of the formula in Notation (`itertools.combinations`, `math.factorial`).
- **Checkpoint.** Two players with v(A) = 0.10, v(B) = 0.05, v(AB) = 0.20: φ = {A: 0.125, B: 0.075}. The game
  above: φ = {A: 0.15, B: 0.25, C: 0} (by hand, averaging over the six orders). On a provided random five-player
  game (seed shown): efficiency Σφ = v(N) − v(∅) to 1e-12; equal to a provided brute-force average over all
  120 orders to 1e-12; swapping two players in v swaps their values.
- **Explain.** Compare with your guess. Efficiency says a path's values add up to v(S) − v(∅), and here
  v(∅) = 0. What does that force onto the channels of a converting journey, including a conversion that would
  have happened with no marketing at all?
  *Why:* with v(∅) = 0 the whole conversion rate of each set, baseline included, is shared out. Shapley's
  properties (efficiency, symmetry, a player who adds nothing gets nothing) are rules for sharing a value; they
  say nothing about whether v measures what a channel causes. That depends on v, and v here is an observed
  conversion rate.

*Provided after Exercise 4:* the table of v(S) for the 31 non-empty channel sets (rate, conversions,
journeys; the cell stops with a clear message if any set needed by a converting journey has no journeys,
instead of filling in 0) and the Shapley credit, calling the learner's `shapley_values` once per set; it prints
that each set's shares sum to 1. Prototype: all 31 sets occur (smallest 177 journeys); the smallest φ over
all sets was +0.012, so no negative shares (at 60,000 users one of five seeds gave −0.004: report negative
values, never clip them). Then one sentence and a plot: "Look at the conversion rate of users each channel
reached against those it did not reach" (from `truth`, because users with no touch are not in `journeys`;
prototype retargeting 8.3% against 2.4%).

### Exercise 5 · Score the methods against the truth (7 minutes)

*Provided before it:* `credit`, a DataFrame (index `CHANNELS`, columns `last, first, linear, markov,
shapley`) of credited conversions, and `incremental = pd.Series(truth["incremental_conversions"])`.

- **Predict.** In this simulation retargeting changes nobody's decision to buy: its true incremental
  conversions are 0. What share of the credit will the Markov and Shapley methods give it: under 5%, 5 to
  15%, or over 15%?
- **Function.** `credit_vs_truth(credit: pd.DataFrame, incremental: pd.Series) -> pd.DataFrame`, one row per
  method with `tvd` (D_m), `largest_gap_channel` and `largest_gap` (credit share − true share for that channel,
  signed).
- **Checkpoint.** Hand example: credit m1 {A: 5, B: 3, C: 2}, m2 {A: 2, B: 2, C: 6}; incremental {A: 6, B: 2,
  C: 0}: m1 → tvd 0.25, largest gap A, −0.25; m2 → tvd 0.60, largest gap C, +0.60. On the data: equals the
  reference to 1e-9, and `largest_gap_channel` is retargeting for `markov` and `shapley` (prototype: true on all
  five seeds).
- **Explain.** Compare with your guess (prototype: 23% and 25%). Look at the provided plot of mean true intent
  among the users each channel reaches (prototype: display 0.00, social 0.14, email 0.64, search 0.86,
  retargeting 1.05 standard deviations). Which mechanism earns retargeting its credit: it sits late in journeys,
  it reaches users who would buy anyway, or it has the most touches?
  *Why:* mostly selection. Retargeting reaches users who already intend to buy, so journeys that contain it
  convert more often, and every method reads that as credit. The stretch removes the targeting and shows how
  much remains: the efficiency rule still shares out conversions that would have happened anyway. Module 6's
  experiment avoided both problems by comparing with a randomized control group.

*Provided after Exercise 5 · Is the gap noise?* Resample journeys with replacement (`N_BOOT` = 200 FULL, 50
QUICK, seed shown), recompute every method's credit shares (counts vectorized over distinct paths, calling the
learner's `conversion_probability` and `shapley_values`), and plot each method's share with its 95% bootstrap
percentile interval and the true share marked. Sentence before the plot: "Look at whether any true share
(black mark) falls inside a method's interval." Prototype: 200 resamples in 0.71 s; 0 of 25 inside.

### Decision · Move budget or test first? (5 minutes)

Provided cell, in the standard order:

- **Number:** for the four paid channels that can scale (display, social, search, retargeting; email is
  list-limited), cost per credited conversion = spend ÷ credit under Markov and under Shapley, each with its 95%
  bootstrap percentile interval; the channel each method says is cheapest. Prototype: retargeting under both,
  $3.53 (95% interval $3.39–3.71) Markov and $3.36 ($3.03–3.70) Shapley, against $5.35–5.44 (Markov) and
  $3.88–6.42 (Shapley) for the others. Margin per conversion $9, so by attribution every paid channel pays back.
  Then the power of a holdout, with Module 6's MDE formula for unequal arms and σ² = p(1 − p) for a conversion
  rate (α = 0.05 two-sided, 80% power): withholding retargeting from 20% of the users it reaches for one month detects a change of
  (z₀.₉₇₅ + z₀.₈) · √(p(1 − p)(1/n₁ + 1/n₀)) in conversion rate. Prototype: 16,741 users reached, conversion
  rate 8.3%, MDE 1.5 points = about 200 conversions among served users, against about 708 credited; margin at
  risk if the credit were real: 20% × 708 × $9 ≈ $1,270.
- **Rule** (stated before the numbers): "Attribution credit cannot separate a channel's effect from the intent
  of the users it reaches. Move budget toward a channel only when an experiment (Modules 6 and 7) or a
  lift-calibrated MMM (Module 9) shows that its cost per incremental conversion is below the $9 margin per
  conversion. If a holdout can detect an effect as large as the credit (MDE below the credited conversions),
  run it first."
- **Recommendation:** the learner's sentence: test or move, which channel, the holdout share and its MDE, the
  margin at risk, and what result would justify moving budget. The cell prints the mode.

Then a reveal cell: true incremental conversions, true cost per incremental conversion against the $9 margin
(prototype: display $11.17, social $7.81, search $7.35, email $2.63, retargeting none, because Δ = 0), the
channel where credit and truth disagree most (retargeting), and what moving budget to it would have bought
(nothing).

## Stretch (optional)

1. **Same reach, random exposure.** `from mktstats.synth.mta import mta_journeys`;
   `res = mta_journeys(random_exposure=True)`; rerun the five methods and `credit_vs_truth` against
   `res.truth["incremental_conversions"]`. Prototype (three seeds): Shapley D 0.11–0.14, Markov 0.21–0.23;
   retargeting keeps 9–11% (Shapley) and 15% (Markov) of credit. Explain what remains (efficiency). Ask why
   Shapley gains more than Markov here; a likely reason, to check rather than assert: its v compares conversion
   rates of channel sets, and random exposure makes those sets comparable.
2. **Run the holdout.** `res = mta_journeys(holdout={"retargeting": 0.2})`; with `res.holdout`, estimate
   retargeting's incremental conversions among served users with a 95% CI as in Module 6's Exercise 6; compare
   with its Markov credit and its truth. Prototype: 20% holdout → 37 (95% CI −100 to 174); 10% → 186 (−8 to
   380). Both intervals include 0 and exclude the roughly 708 credited.
3. **A count-based value function.** With v(S) = conversions of journeys that use only channels in S, show that
   the Shapley values equal "split each conversion equally among the distinct channels on its path" (the game
   is a sum of unanimity games; prototype: equal to six decimals).

## Expected run time (estimates; no run record exists)

| | QUICK | FULL | Colab target |
|---|---|---|---|
| Install | same as FULL | estimate under 30 s: same install list as Module 6, whose Colab record at `v2026.10.1` shows 17.0 s (`runs/2026-10-10-colab-py-python-06-experiments.json`) | ≤ 3 min |
| Slowest cell | estimate under 5 s | estimate under 10 s (bootstrap) | ≤ 4 min |
| All compute | estimate under 20 s | estimate under 60 s | ≤ 15 min |

Prototype on the laptop (not Colab): generate 0.22 s; heuristic credit 0.10 s; transition matrix 0.12 s;
removal effects 0.002 s; Shapley credit 0.09 s; 200 bootstrap resamples 0.71 s; everything 1.6 s. QUICK
changes only `N_BOOT`; every credit number and checkpoint is the same in both modes. The notebook says so.

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
(channels × methods), `incremental`, `SEED`, `N_BOOT`, `QUICK`.

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

## Open items for the Technical Expert

- Build `mta_journeys` from the specification and re-measure every prototype number on the committed seed; the
  brief's claims 2–5 are prototype results until then. If a claim fails on the committed data, tell the
  Academic Director before changing the parameters.
- CSV size: 100,000 users give about 3.1 MB (2.6 MB without `true_intent`). At 60,000 users (1.85 MB) the
  lesson held on five seeds, but one seed gave a negative Shapley value; prefer 100,000 unless size is a
  problem.
- Seed 2036 follows the core pattern (2026–2032); check it does not collide with another extension's seed
  (harmless if it does, but confusing).
- `modules.m16.deps`: add `src/mktstats/synth/mta.py`, `data/synthetic/mta_journeys.csv` and
  `data/synthetic/truth_mta.json`; install list `[numpy, pandas, scipy, matplotlib]` (SciPy for the MDE).
- Write the bootstrap and the provided Shapley-credit cell so they call the learner's functions (after their
  checkpoints passed or `use_reference`), with counts vectorized over distinct paths.
- Checkpoint failure messages need the four parts of `briefs/_lab-standard.md` §5, for example for Exercise 3:
  "Checkpoint 3: your conversion probability is 0.071; the share of journeys that converted is 0.0435, and a
  chain estimated by counting must reproduce it exactly. This usually means conv or null is still in Q, or the
  start row was left out. Check that Q has only start and the channels, and that r is the conv column. To move
  on now, run `workshop.use_reference(3)`."
