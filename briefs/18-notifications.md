# Lab brief · Module 18 · Notifications: volume, targeting and a policy for the long run

| | |
|---|---|
| Status | **Written 2026-10-16 by the Academic Director from the built lab and its recorded FULL run** (the lab was prototyped and built before this brief, unlike the core modules, whose briefs came first). Every number below is from the FULL run of `labs/python/18-notifications.ipynb` in worked mode on 20,000 users (Apple Silicon laptop, native Python 3.13.2, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3; run record `runs/2026-10-16-macos-arm64-python-18-notifications.json`, Run all 11.8 s) or from `data/synthetic/truth_notifications.json`. Nothing here is a prototype number. Not yet reviewed by the Pedagogy Expert. |
| Notebook | `labs/src/18-notifications.py` → `labs/python/18-notifications.ipynb` |
| Lab slot | 55 min (`modules.m18.minutes.lab`). Budget (`briefs/_lab-standard.md`): open 5 + exercises 8 + 7 + 8 + 9 + 8 = 40 (limit 40) + decision 5 + slack 5 = 55. Each exercise's minutes include reading the provided cells immediately before it. Minutes are estimates until a pilot. |
| Day | Extensions track (`d6`), optional; outside the five timed days. Builds on Modules 6 and 11; Module 5 helps. |

## Question and decision

**Question.** The retailer's app can send a push notification to any user on any day at no price per
send. What does one send do, what does a cadence do over a quarter, and which policy (a fixed cadence, a
per-segment rule, a contextual bandit with an opt-out penalty, or a value-iteration policy) should run
next quarter?

**Decision.** Choose the policy with the highest 90-day margin per 1,000 users whose quarterly opt-out
rate is below the reachable-base guardrail (10%), with the paired 95% interval of its margin against the
best fixed cadence and what a longer horizon would change (`modules.m18.decision`).

## What this lab fixes

Module 6 measured one campaign against a randomized control group, and Module 11 chose whom to treat by
the estimated uplift against the offer's cost. Neither asked what happens when the same channel fires
every day. A notification's cost is not its price: it is the users who turn notifications off, for good.
And its value is not only the open: a session today keeps a habit alive and raises the chance of an
organic visit later. Both effects are invisible in a same-day or seven-day metric. The lab makes them
measurable by simulating the population, so that the true same-day effect of a send is known for every
user-day and any policy can be run on the same users with the same random numbers.

The lab writes no model-fitting code. Each exercise is a few lines of NumPy or pandas: a difference in
means, a grouped table, a send rule, one Thompson draw, one Bellman sweep. The simulator and the loops
around the learner's functions are provided (`mktstats.synth.notifications`, `mktstats.notifications`).

## Data: the simulated population and its truth

| Item | Specification |
|---|---|
| File | `src/mktstats/synth/notifications.py`; `notification_users(seed=2038, *, n_users=20_000) -> SynthResult` (`name="notifications"`) |
| Registration | `EXTENSIONS["notifications"]` in `scripts/make_synthetic.py`: writes `data/synthetic/notifications_users.csv` and `data/synthetic/truth_notifications.json` |
| Lab reads | `pd.read_csv(repo_file("synthetic/notifications_users.csv"))`; QUICK keeps the first 5,000 rows (`notif.QUICK_USERS`); the rows are in random order, so any prefix is a random subsample |
| Table `users` | `user_id` (1 to 20,000), `tenure_days`, `sessions_last_30`, `purchases_last_90`, `segment` (low 0–1, mid 2–4, high 5+ sessions in the last 30 days), `days_since_session` at the start, and the hidden `true_theta` (engagement) and `true_kappa` (fatigue sensitivity), 4 decimals; no policy reads them |
| Committed shares | low 0.290, mid 0.460, high 0.250 (`truth["segment_share"]`); mean engagement by segment −0.78, −0.05, +0.97 |
| Generated live | the send experiment (28 days, 560,000 user-days) and the volume experiment (90 days) are generated in the lab from the committed users with the seeds in `mktstats.notifications` (`SEED_SEND_SIM` 2041, `SEED_SEND_POLICY` 2042, `SEED_VOLUME_ARMS` 2039, `SEED_VOLUME_SIM` 2040, `SEED_EVAL` 2043, `SEED_BANDIT` 2044), because a 28-day log of 20,000 users is 560,000 rows |

**Data-generating process** (one user-day at a time; every parameter is in `truth["dgp"]`):

1. **Engagement** θ_i ~ Normal(0, 1), hidden. Observed readings of it: `sessions_last_30` ~ Poisson(exp(1 + 0.6 θ)), `purchases_last_90` ~ Poisson(exp(−0.3 + 0.5 θ)); `tenure_days` is independent of θ. **Fatigue sensitivity** κ_i = exp(−0.4 θ_i + Normal(0, 0.5)), hidden: engaged users tolerate more.
2. **Fatigue** F_{i,t+1} = 0.7 F_{i,t} + s_{i,t}, where s_{i,t} = 1 if a notification was sent. After a long run of daily sends F settles at 1/(1 − 0.7) = 3.33 just after a send; every other day 1.96; weekly 1.09 (`truth["steady_state_fatigue"]`).
3. **A send** of template k is opened with probability expit(α_k + 0.8 θ − 0.5 F), with α = −2.2 (reminder), −1.8 (offer), −2.0 (social). An open is a session. Fresh-user open rates (F = 0, population mean): reminder 12.2%, offer 16.7%, social 14.3%.
4. **Opt-out.** After every send the user opts out with probability expit(−8 + 0.8 κ F), F measured before the send; absorbing. Half of the opt-outs are uninstalls, after which the user never visits again. Population mean opt-out probability per send at F = 0, 1, 2, 3: 0.0003, 0.0016, 0.0139, 0.0487 (`truth["optout_per_send_at_fatigue"]`).
5. **Habit.** Without an open the user visits with probability expit(−2.2 + 0.9 θ − 0.08 min(d, 30)), d = days since the last session; any session resets d to 0.
6. **Purchase.** In a session the user buys with probability expit(−1.6 + 0.4 θ + 0.6·[opened an offer]); an order is \$30 at 30% gross margin = \$9, \$3 less when bought from an offer.
7. **Common random numbers.** The five uniforms (open, opt-out, uninstall, organic visit, purchase) are drawn for every user every day whatever the policy does, so two policies run at one seed are compared user by user. The potential outcomes of today's action for the reminder (`margin_if_send`, `margin_if_skip`, `session_if_send`, `session_if_skip`) are computed from the same uniforms and logged; this is what makes the true same-day effect known.

**`truth_notifications.json`**: `seed`, `n_users`, `templates`, `actions` (`skip`, `reminder`, `offer`, `social`), `segments`, `segment_edges`; `dgp` (every parameter above with a one-paragraph description); `segment_share`, `mean_theta_by_segment`, `open_rate_fresh_by_template`, `optout_per_send_at_fatigue`, `steady_state_fatigue`; `send_experiment` (28 days, p_send 0.3, seeds, and the **true same-day effect** on the logged user-days: sessions 0.0741 and margin \$0.1394 per send, defined as the mean over eligible user-days of the potential-outcome difference, reminder sent against skipped); `volume_experiment` (90 days, cadences 0, 1, 2, 3, 5, 7 a week, seeds); `quick_users` 5,000.

**The two experiments the lab analyzes** (both logged by the simulator; "real logs do not have the potential-outcome columns, and nothing uses them except the checkpoint that reveals the truth"):

- **Send experiment** (`notif.send_experiment(users)`): for 28 days, every user who had not opted out is sent the reminder with probability 0.3, independently of everything. FULL: 560,000 user-days; 29.9% of eligible user-days got a send; open rate 8.8%. Among sent user-days, openers had 1.000 sessions and \$1.970 of margin against 0.079 and \$0.161 for non-openers: a gap of \$1.81 in same-day margin, about 13 times the true effect.
- **Volume experiment** (`notif.volume_experiment(users)`): each user assigned at random to a cadence of 0, 1, 2, 3, 5 or 7 reminders a week on fixed weekdays (3,307 / 3,395 / 3,335 / 3,311 / 3,367 / 3,285 users) for 90 days. `opens_week_1` replays the same run for 7 days.

## Notation

- i users, t days, g ∈ {low, mid, high} segments. s_{i,t} ∈ {0, 1} a send; a ∈ {skip, reminder, offer, social} the action.
- **Same-day effect** of a send: τ = E[Y(send) − Y(skip)] over eligible user-days, for Y = session and Y = margin. Estimate from the send experiment by the difference in means, τ̂ = ȳ_1 − ȳ_0, se = √(s_1²/n_1 + s_0²/n_0) with sample variances (ddof = 1), 95% normal interval τ̂ ± 1.96 se. Eligible = not opted out before today.
- **Volume curve**: per cadence k, sends per user, open rate = Σ opens / Σ sends, opens per user in week 1, opt-out share, margin per user over 7 and 90 days.
- **7-day effect** per segment: τ̂_7(g), the difference in mean margin over days t to t+6 between sent and not-sent eligible user-days with a complete window. **Opt-out probability per send** p̂_out(g) = mean of `opted_out_today` over the segment's sent rows. **Value at stake** V_g = 90-day margin per user in the 3-a-week arm of the volume experiment. **Send rule**: send to g when τ̂_7(g) > p̂_out(g) V_g + c, with c the channel's cost per send.
- **Context** x: segment × fatigue bucket (<0.5, 0.5–1.0, 1.0–1.5, 1.5–2.0, 2.0–3.0, ≥3.0) × days since the last session (<2, 2–6, ≥7): 54 contexts. **State** s: the 54 contexts plus the absorbing opted-out state (55).
- **Thompson step**: S_{x,a} and n_{x,a} the summed reward and count for context x and action a; draw μ̃_{x,a} ~ Normal(S_{x,a}/(n_{x,a}+1), σ²/(n_{x,a}+1)) (one pseudo-observation of 0 reward), play argmax_a μ̃_{x,a}. σ = 3 by default. Bandit reward of a user-day: margin − λ_out · 1[opted out today], λ_out per segment (0, V_g, 3V_g, 10V_g).
- **Transition model** from the volume log: P[s, a, s′] empirical next-state probabilities, R[s, a] mean margin, N[s, a] visits, actions skip (0) and reminder (1). **Send penalty** (Horizon's device): R[s, send] − λ. **Bellman update**: Q ← R + γ (P @ max_a Q), γ = 0.98. **Greedy policy within the data**: argmax over actions with N[s, a] ≥ 30.
- **Paired difference** of policy A against B: d_i = margin_i(A) − margin_i(B) on common random numbers; mean, se = sd(d)/√n, 95% normal interval. **Guardrail**: opt-out share over 90 days ≤ 0.10.

## Terms the notebook defines, where they are first used

| Term | One sentence for the notebook |
|---|---|
| policy | The rule that decides, each day and for each user who has not opted out, whether to send and which template. |
| fatigue | A running count of recent sends that decays by 30% a day; it lowers opens and raises opt-outs. |
| opt-out | The user turns notifications off, for good; half of them also uninstall and never visit again. |
| eligible user-day | A user-day on which a send was possible: the user had not opted out before that day. |
| same-day effect | What a send changes on the day it is sent, against the same user-day without one. |
| cadence | A fixed number of reminders a week on fixed weekdays. |
| volume curve | Sends, open rate, opens, opt-outs and margin per user as a function of the cadence. |
| value at stake | What an opt-out puts at risk: the segment's 90-day margin per user under the best uniform cadence. |
| contextual bandit | A learner that picks an action for each context and updates from the reward it gets, one day at a time, with no model of what comes next. |
| Thompson sampling | Draw one plausible mean reward per action from its posterior and play the largest. |
| state, absorbing state | The user's situation that actions move (segment, fatigue, days since a session); the opted-out state, which nothing leaves. |
| value iteration | Repeat the Bellman update from zero until the action values stop changing. |
| common random numbers | The same random draws for every policy, so two policies are compared user by user. |
| paired interval | The 95% interval of the mean per-user difference between two policies run on common random numbers. |
| reachable-base guardrail | The largest share of users a policy may lose to opt-outs in a quarter. |

## Parts and exercises

| Part | Exercise | Minutes |
|---|---|---|
| A · One send, measured properly | *Run: the send experiment; openers against non-openers (provided)* | — |
| | 1 · The same-day effect of a send | 8 |
| B · Volume: the trade-off over a quarter | *Run: the volume experiment (provided)* | — |
| | 2 · The volume curve | 7 |
| | *Run: the three-panel plot; the curve by segment (provided)* | — |
| C · Who should get it | *Run: 7-day forward margin and the value at stake (provided)* | — |
| | 3 · A send rule per segment | 8 |
| | *Run: the rule with three per-send costs (provided)* | — |
| D · A contextual bandit | *Run: the 54 contexts (provided)* | — |
| | 4 · The Thompson-sampling step | 9 |
| | *Run: four bandits for 90 days; what one learned (provided)* | — |
| E · A policy for the long run | *Run: the transition model from the volume log (provided)* | — |
| | 5 · Value iteration | 8 |
| | *Run: four send penalties; the learned rule (provided)* | — |
| **Exercises** | | **40** |
| Decision | Which policy next quarter? | 5 |

### Exercise 1 · The same-day effect of a send (8 minutes)

*Provided before it:* the send experiment's log (`notif.send_experiment(users)`), one row per user-day,
and the openers-against-non-openers table among sent user-days: session rate 1.000 against 0.079, margin
\$1.970 against \$0.161, gap \$1.81.

- **Predict.** The openers-versus-non-openers gap is about \$1.80 of margin. The true same-day effect of a
  send on margin per send: under \$0.05, \$0.05 to \$0.25, or over \$0.25? *Answer: \$0.05 to \$0.25 (the
  estimate is \$0.139, the truth \$0.139).*
- **Function.** `incremental_per_send(log) -> pd.DataFrame` with rows `sessions_per_send` and
  `margin_per_send` and columns `estimate`, `se`, `lo`, `hi`: the difference in means between sent and
  not-sent **eligible** user-days, se = √(s_1²/n_1 + s_0²/n_0) with `ddof=1`, and estimate ± 1.96 se.
- **Checkpoint 1** (`notif.check_incremental_per_send`): structure (DataFrame, the two rows, the four
  columns); estimate and se equal to the reference to `rtol=1e-5, atol=1e-6` (the computation is
  deterministic, so the tolerance only absorbs floating-point order); `lo`/`hi` equal to estimate ∓ 1.96 se
  to 1e-6; and the 95% interval covers the true same-day effect from the potential-outcome columns
  (`env.true_same_day_effect(log)`). The last part is the only statistical check; on the committed users and
  seeds it holds (sessions 0.0745 [0.0725, 0.0765] against 0.0741; margin 0.1386 [0.1297, 0.1476] against
  0.1394). First checkpoint; no model is fitted anywhere in the lab.
- **Explain.** Compare with your guess and with the \$1.80 gap. Which users open: the ones a send changes, or
  the ones who would have visited anyway? Point to the session rate of the not-sent user-days.
  *Why:* opening is a choice engaged users make more often, and they buy more whether or not they are
  notified; the gap measures who the engaged users are. The coin flip removes that selection. The effect is
  small in dollars because most sends are not opened and most sessions do not end in an order.

### Exercise 2 · The volume curve (7 minutes)

*Provided before it:* the volume experiment's per-user table (`arm`, `sends`, `opens`, `sessions`,
`purchases`, `margin`, `margin_7d`, `opted_out`, `uninstalled`) and `opens_w1`.

- **Predict.** Which cadence will have the highest 90-day margin per user: 2, 3, 5 or 7 a week? Which will
  have the most opens per user in the first week? *Answers: 3 a week (\$21.88); 7 a week (0.419 opens).*
- **Function.** `volume_curve(per_user, opens_w1) -> pd.DataFrame`, one row per cadence (sorted), columns
  `users`, `sends` (mean per user), `open_rate` (Σ opens ÷ Σ sends; 0 where nothing was sent),
  `opens_week_1` (mean of `opens_w1` over the arm's users), `opt_out` (share with `opted_out`), `margin_7d`,
  `margin_90d` (means).
- **Checkpoint 2** (`notif.check_volume_curve`): structure (the seven columns, a row per cadence); every
  column equal to the reference to `atol=1e-6, rtol=1e-5`; then the two facts the committed data show: the
  daily arm has the most week-1 opens and the 3-a-week arm the highest 90-day margin. The two facts hold in
  FULL and, per the Technical Expert's QUICK runs, in QUICK; they are facts about the committed seeds, and
  the message says so if the data changed.

FULL run:

| Cadence | Users | Sends per user | Open rate | Opens, week 1 | Opt-out share | Margin 7 d | Margin 90 d |
|---|---|---|---|---|---|---|---|
| 0 | 3,307 | 0.0 | 0.000 | 0.000 | 0.000 | \$1.28 | \$13.34 |
| 1 | 3,395 | 13.0 | 0.116 | 0.118 | 0.007 | \$1.51 | \$17.68 |
| 2 | 3,335 | 25.8 | 0.094 | 0.215 | 0.019 | \$1.63 | \$19.88 |
| 3 | 3,311 | 38.0 | 0.080 | 0.300 | 0.054 | \$1.83 | **\$21.88** |
| 5 | 3,367 | 54.6 | 0.051 | 0.352 | 0.258 | \$1.95 | \$20.41 |
| 7 | 3,285 | 62.4 | 0.035 | **0.419** | 0.458 | \$1.95 | \$17.93 |

By segment (90-day margin per user; share who opted out):

| Segment | 0 | 1 | 2 | 3 | 5 | 7 |
|---|---|---|---|---|---|---|
| low | \$2.17 (0.000) | \$3.95 (0.006) | \$5.13 (0.028) | \$5.37 (0.089) | **\$5.77** (0.388) | \$3.88 (0.627) |
| mid | \$8.06 (0.000) | \$12.00 (0.007) | \$14.74 (0.017) | **\$15.11** (0.045) | \$14.08 (0.246) | \$11.77 (0.443) |
| high | \$35.77 (0.000) | \$43.45 (0.007) | \$48.47 (0.012) | **\$51.14** (0.032) | \$49.28 (0.130) | \$46.59 (0.284) |

- **Explain.** The week-1 metric and the quarter metric choose different cadences: which two mechanisms,
  both absent on day 1, separate them? Point to `opt_out` and to the never-send arm's 90-day margin.
  *Why:* the habit effect (an opened push is a session, and a session raises tomorrow's organic visit,
  which is why one a week lifts the quarter well above never) and fatigue (opt-outs rise steeply after three
  a week, and an opted-out user contributes nothing more from notifications). The 90-day window still
  understates the cost: a user lost on day 60 is lost next quarter too.

### Exercise 3 · A send rule per segment (8 minutes)

*Provided before it:* `margin_fwd = notif.forward_margin(log)` (margin over days t to t+6 per user-day,
NaN where the window runs past day 28) and `value = notif.value_at_stake(per_user)` (low \$5.37, mid
\$15.11, high \$51.14).

- **Predict.** Which segment will have the largest 7-day effect of a send in dollars? Will any segment fail
  the rule `effect > opt-out probability × value at stake` when a send costs nothing? *Answers: high
  (\$0.197); no, all three pass.*
- **Function.** `segment_uplift(log, margin_fwd, value_at_stake, cost_per_send=0.0) -> pd.DataFrame`
  indexed `low`, `mid`, `high` with `cate` (difference in mean `margin_fwd` between sent and not-sent
  rows), `se`, `lo`, `hi` (as in Exercise 1), `p_optout` (mean of `opted_out_today` over the segment's sent
  rows), `cost` (`p_optout × value_at_stake[segment] + cost_per_send`) and `send` (`cate > cost`); only
  eligible user-days with a complete window, for every column.
- **Checkpoint 3** (`notif.check_segment_uplift`): structure; `cate` equal to the reference to
  `rtol=1e-5, atol=1e-6`; `se`, `p_optout`, `cost` likewise; `send` equal to the reference; every `cate`
  positive (holds on the committed data in both modes).

FULL run (cost at \$0 per send):

| Segment | 7-day effect | se | 95% interval | p_optout per send | Cost | Send? |
|---|---|---|---|---|---|---|
| low | \$0.0593 | 0.0128 | \$0.034 to \$0.084 | 0.00499 | \$0.0268 | yes |
| mid | \$0.0983 | 0.0166 | \$0.066 to \$0.131 | 0.00212 | \$0.0321 | yes |
| high | \$0.1973 | 0.0429 | \$0.113 to \$0.281 | 0.00076 | \$0.0388 | yes |

With the provided per-send costs (assumptions: push \$0, email \$0.002, SMS \$0.05), the low segment is
the only one that drops out, and only for SMS (\$0.059 against \$0.027 + \$0.05).

- **Explain.** The high segment has the largest effect and the smallest opt-out probability: what about
  those users makes both true? Then say why a rule built from one send's 7-day effect cannot choose a
  cadence (what happens to `p_optout` after the third send of the week?).
  *Why:* engaged users open more, buy more per session and tolerate more sends. The rule prices one send at
  the fatigue level the experiment happened to have (0.3 sends a day); a daily policy moves every user up
  the fatigue curve, where opt-outs per send are several times higher. Choosing how often needs the state
  the user is in (Parts D and E).

### Exercise 4 · The Thompson-sampling step (9 minutes)

*Provided before it:* the 54 contexts (`notif.context_labels()`), and the Markdown that defines the
bandit, the Thompson draw and the reward (margin minus a penalty if the user opted out that day).

- **Predict.** With penalty 0, will the bandit send more or fewer notifications per user over the quarter
  than three a week (about 38)? Will its opt-out share be above or below the daily cadence's? *Answers:
  more (61.2); below (41.2% against 45.8%).*
- **Function.** `thompson_choose(sums, counts, sigma, rng) -> np.ndarray`: for arrays of shape (contexts,
  actions), draw `rng.normal(sums / (counts + 1), sigma / sqrt(counts + 1))` and return the row-wise
  argmax (integer array of length `contexts`).
- **Checkpoint 4** (`notif.check_thompson`), on a fixed 3-context table: one integer arm per context
  (shape and dtype); with 1,000 tries per arm and one arm's mean 1 against 0, that arm is chosen at least
  99% of 2,000 draws (at σ = 3 the posterior spread is 3/√1001 ≈ 0.095, so a correct function chooses it
  essentially always: the 99% bar leaves room for nothing but a wrong spread); with no data every arm is
  drawn at least once in 2,000 draws; and two generators with the same seed give the same 50 choices while
  different seeds differ (the function must draw from `rng`).
- **Explain.** The bandit sees each day's margin and, with a penalty, each day's opt-outs; what does it
  still not see about a send? Why is `margin_7d` almost the same for every penalty while `margin_90d` is
  not? *Why:* a bandit optimizes the reward it is given, one day at a time; the habit effect arrives later
  and is credited to whatever action is played then. Opt-outs are rare in week 1, so week-1 margins agree;
  by day 90 the penalties have produced different bases.

FULL run, 90 days, common random numbers (`SEED_EVAL`), bandit draws from `SEED_BANDIT`, σ = 3:

| Policy | Sends per user | Open rate | Opt-out share | Margin 7 d | Margin 90 d |
|---|---|---|---|---|---|
| bandit, penalty 0 | 61.2 | 0.051 | 0.412 | \$2.12 | \$20.65 |
| bandit, penalty V | 52.1 | 0.072 | 0.214 | \$2.10 | \$22.20 |
| bandit, penalty 3V | 43.7 | 0.085 | 0.136 | \$2.10 | **\$22.51** |
| bandit, penalty 10V | 33.6 | 0.103 | 0.085 | \$2.11 | \$21.96 |
| 3 a week (volume experiment arm) | 38.0 | 0.080 | 0.054 | \$1.83 | \$21.88 |

What the penalty-10V bandit learned for the mid segment among users who visited within a day (mean
reward per action; n = visits): at fatigue <0.5 every template beats skip (social 0.629, reminder 0.413,
offer 0.359, skip 0.178; n = 2,884); at 1.0–1.5 the offer still wins (0.377 against skip 0.245; n = 54,643);
at 2.0–3.0 every send is negative (−0.60 to −0.67 against skip 0.283; n = 13,821).

### Exercise 5 · Value iteration (8 minutes)

*Provided before it:* the volume experiment's log and `P, R, N = notif.transition_model(volume_log)`:
55 states × 2 actions; 1,780,000 transitions (20,000 users × 89 day pairs); 0 state-action pairs seen
fewer than 30 times in FULL; mean margin per user-day \$0.241 after a skip and \$0.354 after a send;
\$0.031 in the opted-out state.

- **Predict.** With a per-send penalty of \$0.06 and γ = 0.98, will the learned policy send more or fewer
  notifications per user over the quarter than three a week (about 38)? Write a number. *Answer: fewer,
  31.1.*
- **Function.** `q_iteration(P, R, gamma, n_iter=1000) -> np.ndarray` of shape (states, actions): from
  zeros, `n_iter` sweeps of `Q = R + gamma * (P @ Q.max(axis=1))`.
- **Checkpoint 5** (`notif.check_q_iteration`): on a two-state example worked by hand (from state 0 a send
  earns 1 and leads to state 1; there a send earns −1 and a skip returns to state 0 for free; γ = 0.5), Q
  after 200 sweeps equals [[2/3, 4/3], [2/3, −2/3]] (V = [4/3, 2/3]) to `atol=1e-6` (the error after 200
  sweeps is at most 0.5^200); and 200 and 201 sweeps give the same Q (convergence; catches a sweep that
  restarts from zeros).
- **Explain.** The value-iteration policies send less than three a week and lose far fewer users, but their
  90-day margin is at or below the 3-a-week arm's. Which explains it best: coarse states, a model that only
  saw the cadences the experiment ran, or the discount? Which fatigue bucket flips to skip? *Why:* value
  iteration is exact for the model it is given; the model is the approximation (users with different
  hidden engagement share a state; high-fatigue states were seen only through the 5- and 7-a-week arms;
  `greedy_policy` refuses actions tried fewer than 30 times in a state). At \$0.06 the mid segment's rule
  flips to skip at fatigue 1.0 whatever the days since a session.

FULL run, 90 days, common random numbers:

| Policy | Sends per user | Open rate | Opt-out share | Margin 7 d | Margin 90 d |
|---|---|---|---|---|---|
| value iteration, send penalty \$0.00 | 36.1 | 0.078 | 0.084 | \$1.87 | \$20.74 |
| value iteration, send penalty \$0.03 | 35.2 | 0.080 | 0.076 | \$1.86 | \$20.71 |
| value iteration, send penalty \$0.06 | 31.1 | 0.089 | 0.028 | \$1.82 | \$20.61 |
| value iteration, send penalty \$0.10 | 29.0 | 0.094 | 0.023 | \$1.77 | \$20.47 |

### Decision · Which policy next quarter? (5 minutes)

The rule is stated in Markdown before any number: a policy may lose at most 10% of its reachable users to
opt-outs in a quarter (a user lost is lost for every later quarter); among the policies within the
guardrail, choose the highest 90-day margin per 1,000 users and report whether its paired interval
against three a week excludes zero. "Decide as you would at work, where there is no second quarter to
look at yet." The reveal comes after the learner's sentence.

Provided cell: `notif.evaluate` runs the six fixed cadences and the four value-iteration policies at
`SEED_EVAL`; the four bandit runs (same seed) are added; `notif.paired_difference` against "fixed, 3 a
week". Note that the fixed 3-a-week policy evaluated on all 20,000 users at `SEED_EVAL` gives \$21,099 per
1,000 (37.9 sends, 5.4% opt-outs), not the volume arm's \$21.88 per user on 3,311 users at `SEED_VOLUME_SIM`.

FULL run:

| Policy | Sends per user | Opt-outs | Margin 90 d per 1,000 | vs 3 a week, \$ per 1,000 (95% paired) | Within guardrail |
|---|---|---|---|---|---|
| fixed, 0 a week | 0.0 | 0.0% | \$13,662 | −7,437 (−7,593 to −7,280) | yes |
| fixed, 1 a week | 13.0 | 0.5% | \$17,876 | −3,223 (−3,334 to −3,112) | yes |
| fixed, 2 a week | 25.8 | 1.6% | \$20,149 | −950 (−1,072 to −828) | yes |
| fixed, 3 a week | 37.9 | 5.4% | \$21,099 | 0 | yes |
| fixed, 5 a week | 54.2 | 27.2% | \$20,302 | −797 (−912 to −683) | no |
| fixed, 7 a week | 62.4 | 46.2% | \$18,459 | −2,640 (−2,784 to −2,495) | no |
| value iteration, \$0.00 | 36.1 | 8.4% | \$20,742 | −357 (−482 to −233) | yes |
| value iteration, \$0.03 | 35.2 | 7.6% | \$20,707 | −392 (−517 to −267) | yes |
| value iteration, \$0.06 | 31.1 | 2.8% | \$20,605 | −494 (−620 to −368) | yes |
| value iteration, \$0.10 | 29.0 | 2.3% | \$20,473 | −626 (−755 to −498) | yes |
| bandit, penalty 0 | 61.2 | 41.2% | \$20,646 | −454 (−596 to −311) | no |
| bandit, penalty V | 52.1 | 21.4% | \$22,197 | +1,098 (+956 to +1,239) | no |
| bandit, penalty 3V | 43.7 | 13.6% | \$22,507 | +1,408 (+1,274 to +1,542) | no |
| **bandit, penalty 10V** | 33.6 | 8.5% | **\$21,964** | **+865 (+730 to +999)** | **yes** |

- **Number:** best within the guardrail, bandit with penalty 10V: \$21,964 per 1,000 users over 90 days,
  opt-outs 8.5%, +\$865 per 1,000 against three a week (95% paired interval +730 to +999). Best ignoring
  the guardrail: bandit with penalty 3V, \$22,507, opt-outs 13.6%.
- **Rule:** highest 90-day margin among policies that lose at most 10% of the reachable base.
- **Recommendation** (printed): run the penalty-10V bandit next quarter; +865 per 1,000 against three a
  week, interval excludes zero. What would change it: a guardrail above 14% would admit the 3V bandit
  (+\$1,408 per 1,000 this quarter), or evidence that the users it loses were worth less than the value at
  stake assumes. Mode printed (FULL, or the QUICK warning that intervals are about twice as wide and a
  choice near the guardrail can flip).
- **The learner's sentence** (template in Markdown): the policy, its margin per 1,000 and opt-out share,
  the paired difference and interval against three a week, and what would change the choice. The learner
  writes it in `MY_SENTENCE`; the reveal cell raises `RuntimeError` while it is empty (not in worked mode),
  so the reveal only teaches after the decision is made. (This gate, and the caption of the "what the
  bandit learned" table, were edited into `labs/src/18-notifications.py` at 12:54 on 2026-10-16, after the
  run records of 12:47 were written; the numbers are unaffected, and the regenerated notebook needs a new
  record.)
- **Reveal** (180 days, same users and random numbers; the bandits rerun with their penalty):

| Policy | Margin 90 d per 1,000 | Margin 180 d per 1,000 | Second quarter | Opt-outs, 180 d |
|---|---|---|---|---|
| fixed, 3 a week | \$21,099 | \$41,557 | \$20,458 | 10.1% |
| bandit, penalty 10V | \$21,964 | \$43,567 | \$21,603 | 10.6% |
| bandit, penalty 3V | \$22,507 | \$44,401 | \$21,893 | 19.7% |

Over 180 days the best policy is still the 3V bandit: its second quarter (\$21,893) beats the 10V bandit's
(\$21,603) although it has lost a fifth of the base. The guardrail is a business choice about what the
base is worth beyond the simulation's horizon and its reward; the simulation does not prove it right. The
notebook says so in its closing note ("the penalty a bandit is given, and the guardrail, are decisions, not
parameters the data return").

## Failure messages

Every message ends with "To move on now, run `workshop.use_reference(N)` and come back to this later."
The wrong answers below are the branches in `src/mktstats/notifications.py` (`check_*`), each exercised
by `tests/test_notifications.py`.

| Checkpoint | Wrong answer caught | What the message says |
|---|---|---|
| 1 | not a DataFrame; a column or a row missing | names the type, or the missing column(s) / row(s) and what is needed |
| 1 | difference over all user-days instead of eligible ones | "This is the difference over all user-days: users who had already opted out are in the not-sent group but could never be sent. Keep only rows with eligible == 1" |
| 1 | estimate differs for another reason | your value against the reference; check the filter, the groups and the column |
| 1 | se differs | your se against the reference; "Use sqrt(var1 / n1 + var0 / n0) with the sample variance (ddof=1)" |
| 1 | interval not estimate ± 1.96 se | your interval against the correct one |
| 1 | truth outside the 95% interval although the estimate matches | "this is sampling chance: on the committed users and seeds it does not happen; if you changed the seed, users or days, read the notebook's note" |
| 2 | not a DataFrame; a column missing; a cadence missing | names them; "group by `arm` so every cadence gets a row" |
| 2 | a column differs from the reference | the first differing cadence, yours against the reference, and the cause per column: open rate as a mean of per-user rates ("would weight a user with one send like one with sixty"), `sends` not the mean, `opt_out` not the share, `margin_7d`/`margin_90d` the wrong column, `opens_week_1` not grouped by arm, `users` not the row count |
| 2 | daily arm not first in week-1 opens; 3-a-week not first in 90-day margin | "Your numbers match the reference, so the data changed: check the users, the seeds and the number of days" |
| 3 | not a DataFrame; a column or a segment missing | names them |
| 3 | `cate` differs | yours against the reference; "Keep eligible user-days with a complete window (margin_fwd not NaN), then take the difference in means of margin_fwd between sent and not-sent rows within the segment" |
| 3 | `se`, `p_optout` or `cost` differs | yours against the reference and the definition of that column |
| 3 | `send` flipped | the cate and cost that give the opposite |
| 3 | a non-positive `cate` although the estimates match | "the data changed: check the users, the seeds and the number of days" |
| 4 | wrong shape or non-integer dtype | "one arm per context (argmax over axis=1)"; "integer arm indexes (the argmax), not the draws" |
| 4 | best arm chosen under 99% of the time with 1,000 tries per arm | "The posterior standard deviation is sigma / sqrt(n + 1): check that counts scale the spread, and that the mean is sums / (counts + 1)" |
| 4 | some arm never tried in a context with no data | "Draw from Normal(mean, sigma / sqrt(n + 1)) with rng.normal, not the mean itself" |
| 4 | same seed gives different choices, or different seeds the same | "The function must draw from `rng`, not from a generator of its own or numpy's global state" |
| 5 | wrong shape | "Q with one row per state and one column per action" |
| 5 | Q equals a single sweep from zeros | "repeat the update n_iter times" |
| 5 | Q equals R | "add gamma times the expected value of the next state" |
| 5 | Q wrong for another reason | your Q against the hand-worked one, and the update written out |
| 5 | 200 and 201 sweeps differ | "Check that each sweep uses the previous sweep's Q, not the initial zeros" |

## Stretch (optional) · Exploration, a referral loop, and real data

1. **Exploration.** Rerun the penalty-3V bandit with σ = 1 and 10. FULL: σ = 1 sends 5.42 per user in
   week 1 and earns \$22.24 over 90 days with 13.2% opt-outs; σ = 3 (the default) 5.47, \$22.51, 13.6%;
   σ = 10 5.45, \$22.30, 15.8%. A small σ risks settling on the wrong action in a context it has barely
   tried; a large one keeps paying for exploration.
2. **A referral loop.** With the 3-a-week open rate (0.080), an invitation rate of 0.1 per open and an
   acceptance rate of 0.2 per invitation (assumptions), the viral factor K = 3 × 0.080 × 0.1 × 0.2 = 0.0048
   new users per user per week against a weekly loss of 0.054 ÷ 13 = 0.0041: the base grows. At 7 a week
   K = 0.0049 (opens per week barely move: 7 × 0.035) against a loss of 0.0352: it shrinks.
3. **Real data.** Duolingo's replication data for its KDD 2020 bandit paper (about 200 million practice
   reminders, Harvard Dataverse, CC BY-NC 4.0; `references.qmd`). Exercises 1 and 3 need a control group of
   eligible users who were sent nothing that day, which the file does not have; Exercise 2 needs opt-outs,
   which it does not record; Exercise 4's bandit is what the paper ran, on template choice, with "practiced
   within two hours" as the reward.

Nothing later depends on the stretch.

## Run settings: QUICK and FULL

QUICK (`MKTSTATS_QUICK=1`, or the `QUICK` switch in the harness cell) keeps the first 5,000 of the 20,000
users. It is the only thing QUICK changes: the experiments, the policies and the checkpoints are the same.
With QUICK on, intervals are about twice as wide and a decision near its guardrail can flip; the header
and the Decision cell say so. The Technical Expert's QUICK worked, verify and learner runs pass
(PLAN.md, Phase 9); no QUICK run is recorded, so no QUICK number is quoted anywhere.

## Run time (measured on the laptop; a Docker record exists; no Colab record)

| | macOS, native (`runs/2026-10-16-macos-arm64-python-18-notifications.json`) | Docker image, linux/arm64, `--cpus 2` (`runs/2026-10-16-docker-arm64-python-18-notifications.json`) | Colab target |
|---|---|---|---|
| Install | 0.0 s (a checkout installs nothing) | 0.0 s (the image has everything) | ≤ 3 min; the install list is `[numpy, pandas, scipy, matplotlib]`, the same as Modules 6 and 16 |
| Slowest cell | 1.8 s (the 180-day reveal; the decision table and the four bandit runs are next) | 3.0 s (the imports) | ≤ 4 min |
| Whole notebook, worked, FULL | **11.8 s** | 14.0 s (learner mode 3.8 s) | ≤ 15 min |

Neither is a Colab time. The notebook is not in `v2026.10.2`, the version Colab opens, so it carries the
"Colab after the next release" chip until the next tag. Earlier runs of the same code on the same laptop took 10.0 to 11.1 s:
run-to-run variation. The recorded figure is the one quoted.

## Known pitfalls

- Compare eligible user-days only: opted-out users sit in the not-sent group forever and bias the
  difference toward zero.
- Open rate is Σ opens ÷ Σ sends per arm, not a mean of per-user rates.
- The 7-day effect needs a complete window: `margin_fwd` is NaN for days 22 to 27.
- `p_optout` is the opt-out probability per send at the experiment's fatigue level (0.3 sends a day); it
  does not transfer to a daily cadence.
- Thompson sampling: the spread shrinks with the count, the mean has one pseudo-observation of zero, and
  the draws come from the generator passed in (reproducibility and the checkpoint depend on it).
- `P @ Q.max(axis=1)` relies on `P` having shape (S, A, S); a transposed `P` runs and gives wrong values.
- `greedy_policy` never chooses an action tried fewer than 30 times in a state. In FULL every pair has at
  least 30 visits, so the restriction does not bind on the committed data; it is there for smaller logs
  and real ones.
- The fixed 3-a-week policy's margin in the decision table (\$21,099 per 1,000 at `SEED_EVAL`, all users)
  is not the volume arm's (\$21.88 per user on 3,311 users at `SEED_VOLUME_SIM`).
- The bandit's opt-out share is reported over all users; its "within guardrail" test uses the same 90-day
  share as the fixed policies.

## APIs verified (and how)

No bandit or RL library. The lab's header records the calls checked against the installed source on
2026-10-16: NumPy 2.1.3 `Generator.normal(loc, scale)` with array arguments, `np.digitize`,
`ndarray.argmax(axis=1)`, `np.add.at`; pandas 2.2.3 `groupby(...).mean()`, `Series.var(ddof=1)`,
`DataFrame.pivot`; SciPy 1.16.3 `stats.norm.ppf`.

## Names used in the lab

`users`, `log` (send experiment), `live`, `per_user` and `volume_log` (volume experiment), `opens_w1`,
`curve`, `margin_fwd`, `value`, `labels` (contexts), `P`, `R`, `N`, `Q`, `rule_06`, `table`, `margins`,
`reveal`, `SEED`, `HORIZON`, `GAMMA`, `SEND_PENALTIES`, `PENALTIES`, `GUARDRAIL`, `CHANNEL_COST`, `QUICK`.
Learner functions: `incremental_per_send`, `volume_curve`, `segment_uplift`, `thompson_choose`,
`q_iteration`. Provided: everything in `mktstats.notifications` (`send_experiment`, `volume_experiment`,
`opens_week_1`, `forward_margin`, `value_at_stake`, `context_labels`, `run_bandit`, `transition_model`,
`with_send_penalty`, `greedy_policy`, `evaluate`, `summarize`, `paired_difference`) and
`mktstats.synth.notifications` (`Simulator`, `fixed_policy`, `true_same_day_effect`).

## References

The entries for `references.qmd` ("Notifications (Module 18)"), each fetched on 2026-10-16, with what
was read and what it supports: Gauci et al. (2019, Horizon, arXiv 1811.00260 v5; section 9.1 read on
ar5iv); Prabhakar et al. (2022, arXiv 2207.03029; full text on ar5iv); Yuan et al. (2022, arXiv
2202.03867; full text on ar5iv); Ocejo et al. (2025, arXiv 2509.02458; full text on arxiv.org/html); Gao
et al. (KDD 2018, DOI 10.1145/3219819.3219880; Crossref record and abstract through OpenAlex and Semantic
Scholar; the ACM page and the videolectures page refused automated access); Zhao et al. (KDD 2018, DOI
10.1145/3219819.3219906; Crossref record and abstract through OpenAlex); the Pinterest Engineering post
(2020; not read: Medium refused automated access); Yancey and Settles (KDD 2020, DOI
10.1145/3394486.3403351; PDF read from research.duolingo.com) and its Harvard Dataverse replication data
(doi:10.7910/DVN/23ZWVI, read through the Dataverse API); Thompson (1933, Crossref record); ReAgent on
PyPI; Apple, Android, MDN and FTC documentation for the channel table.
