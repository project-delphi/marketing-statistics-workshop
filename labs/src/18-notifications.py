# %% [markdown]
# <!--
# Module 18 lab; brief: briefs/18-notifications.md. No bandit or RL library: the learner writes
# the experiment analysis, the volume curve, the uplift rule, the Thompson-sampling step and
# value iteration with NumPy and pandas; the simulated population and the loops around those
# functions are in mktstats.notifications and mktstats.synth.notifications. APIs checked
# 2026-10-16 against the installed source in the workshop's environment: numpy 2.1.3
# `Generator.normal(loc, scale)` (array loc and scale broadcast), `np.digitize`, `ndarray.argmax(axis=1)`,
# `np.add.at`; pandas 2.2.3 `DataFrame.groupby(...).mean()`, `Series.var(ddof=1)`,
# `DataFrame.pivot`; SciPy 1.16.3 `stats.norm.ppf`.
# -->
#
# **How this lab runs.** Nothing here fits a model that takes more than a second: every policy
# is a few lines of NumPy run day by day over the users, so every cell runs in seconds. `QUICK`
# (in the harness cell) keeps the first 5,000 of the 20,000 simulated users; the experiments,
# the policies and the checkpoints are the same. With QUICK on, intervals are about twice as
# wide and a decision near its guardrail can flip. That is a lesson about sample size, not a bug.
#
# The data are simulated: the retailer's app has 20,000 users, each with a hidden engagement
# level and a hidden tolerance for being notified. Each day a **policy** decides, for every
# user who has not opted out, whether to send a push notification and which of three
# templates. A sent notification may be opened (an open is a session); every send raises the
# user's **fatigue**, which lowers later opens and raises the chance that the user opts out, for
# good; half the opt-outs are uninstalls. Without a notification a user still visits on their
# own, less often the longer it has been since the last session: a session today keeps the
# habit alive. In a session the user may buy; an order is \$30 at a 30% margin, \$9. Because the
# world is simulated, the same-day effect of a send is known exactly for every user-day, and
# any policy can be run on the same users with the same random numbers.

# %%
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mktstats import notifications as notif
from mktstats.data import repo_file
from mktstats.synth import notifications as env

SEED = 2038  # the users' seed; the experiments' and runs' seeds are constants in `notif`
HORIZON = notif.HORIZON  # days a policy is judged over: one quarter


def show(fig):
    """Show a figure as a PNG on Colab, in Jupyter and in a headless run alike."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    from IPython.display import Image, display
    display(Image(buf.getvalue()))


users_all = pd.read_csv(repo_file("synthetic/notifications_users.csv"))
users = (users_all.iloc[: notif.QUICK_USERS] if QUICK else users_all).reset_index(drop=True)
print(f"QUICK = {QUICK}: {len(users):,} of {len(users_all):,} users (the only thing QUICK changes)")
print("Segment shares:", users["segment"].value_counts(normalize=True).round(2).to_dict())
users.head()

# %% [markdown]
# The observed columns are `tenure_days`, `sessions_last_30`, `purchases_last_90` and `segment`
# (a bucket of `sessions_last_30`: low 0–1, mid 2–4, high 5 or more). `true_theta` (engagement)
# and `true_kappa` (fatigue sensitivity) are the simulation's hidden truth; no policy below
# reads them.
#
# # Part A · One send, measured properly
#
# The **send experiment**: for 28 days, each day, every user who had not opted out was sent the
# reminder with probability 0.3, at random and independently of everything else. The log has
# one row per user-day. `eligible` says whether a send was possible that day (the user had not
# opted out), `sent` whether one went out, `opened`, `session` (opened, or visited on their own),
# `margin` (dollars that day). The four `*_if_send` / `*_if_skip` columns are the simulation's
# potential outcomes of that day's decision, from the same random draws: real logs do not have
# them, and nothing below uses them except the checkpoint that reveals the truth.

# %%
log = notif.send_experiment(users)  # seeds notif.SEED_SEND_SIM and notif.SEED_SEND_POLICY
live = log[log["eligible"] == 1]
print(f"{len(log):,} user-days over {notif.SEND_DAYS} days; {live['sent'].mean():.1%} of eligible"
      f" user-days got a send; open rate {live.loc[live['sent'] == 1, 'opened'].mean():.1%}")
log.head()

# %% [markdown]
# The report a dashboard gives first compares **openers with non-openers** among the users who
# were sent a notification. The next cell computes it. Look at the gap in same-day margin.

# %%
sent = live[live["sent"] == 1]
by_open = sent.groupby("opened")[["session", "margin"]].mean()
by_open.index = ["did not open", "opened"]
print(by_open.round(3).to_string())
print(f"Gap in same-day margin, openers minus non-openers: ${by_open.loc['opened', 'margin'] - by_open.loc['did not open', 'margin']:.2f}")

# %% [markdown]
# ## Exercise 1 · The same-day effect of a send (8 minutes)
#
# A notification's **same-day effect** is the difference between what happens to a user-day
# with a send and what would have happened to the same user-day without one. The experiment
# makes that measurable: among eligible user-days, which ones got a send was decided by a coin
# flip, so the sent and the not-sent user-days are alike in everything else.
#
# **Predict.** The openers-versus-non-openers gap above is about \$1.80 of margin. The true
# same-day effect of a send on margin per send: under \$0.05, \$0.05 to \$0.25, or over \$0.25?
# Write your guess.
#
# **Task.** Write `incremental_per_send(log)` returning a `pd.DataFrame` with rows
# `sessions_per_send` and `margin_per_send` and columns `estimate`, `se`, `lo`, `hi`: for each
# outcome (`session`, `margin`), the difference in means between sent and not-sent **eligible**
# user-days, its standard error $\sqrt{s_1^2/n_1 + s_0^2/n_0}$ with the sample variances
# (`ddof=1`), and the 95% normal interval `estimate ± 1.96 se`.

# %% tags=["exercise"]
def incremental_per_send(log):
    # TODO 1: keep log[log["eligible"] == 1]; split it by sent == 1 / sent == 0; for each of
    # "session" and "margin": estimate = mean(sent) - mean(not sent), se = sqrt(var1/n1 + var0/n0)
    # with .var(ddof=1), lo/hi = estimate -/+ 1.96 * se; build the two-row DataFrame
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def incremental_per_send(log):
    live = log[log["eligible"] == 1]
    sent, skip = live[live["sent"] == 1], live[live["sent"] == 0]
    rows = {}
    for name, col in (("sessions_per_send", "session"), ("margin_per_send", "margin")):
        est = sent[col].mean() - skip[col].mean()
        se = np.sqrt(sent[col].var(ddof=1) / len(sent) + skip[col].var(ddof=1) / len(skip))
        rows[name] = {"estimate": est, "se": se, "lo": est - 1.96 * se, "hi": est + 1.96 * se}
    return pd.DataFrame(rows).T[["estimate", "se", "lo", "hi"]]


# %% [markdown]
# The checkpoint compares your estimate with the reference computation and then, because this
# is a simulation, with the true same-day effect: the mean over eligible user-days of
# `margin_if_send − margin_if_skip` (and the same for sessions). Your 95% interval must cover it.

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    truth_same_day = env.true_same_day_effect(log)  # from the potential-outcome columns
    effect = notif.check_incremental_per_send(incremental_per_send, log, truth_same_day)
    print(effect.round(4).to_string())
    print("True same-day effect:", {k: round(v, 4) for k, v in truth_same_day.items()})

# %% [markdown]
# **Explain.** Compare the margin effect with your guess and with the \$1.80 gap between openers
# and non-openers. Which users open a notification: the ones a send changes, or the ones who
# would have visited anyway? Point to the session rate of the not-sent user-days in `by_open`
# and in `live`.
#
# <details><summary>Why this solution works</summary>
#
# Opening is a choice the user makes, and engaged users make it more often; they also buy more
# whether or not they are notified. Comparing openers with non-openers therefore measures mostly
# who the engaged users are, not what the notification did: the gap is about ten times the true
# effect. The experiment's coin flip is what removes that selection: the not-sent user-days are a
# fair picture of what the sent ones would have done. The same-day effect is small in dollars
# because most sends are not opened and most sessions do not end in an order; the rest of the
# lab asks what a send does beyond the day, in both directions.
# </details>
#
# # Part B · Volume: the trade-off over a quarter
#
# A single send's effect does not say how often to send. The **volume experiment** answers that:
# every user was assigned at random to a **cadence** (0, 1, 2, 3, 5 or 7 reminders a week, on
# fixed weekdays) and kept on it for 90 days. The per-user table below has each user's `arm`
# (the cadence), `sends`, `opens`, `sessions`, `purchases`, `margin` over the 90 days,
# `margin_7d` over the first week, and whether the user `opted_out` (and `uninstalled`).

# %%
per_user, _ = notif.volume_experiment(users)  # seeds notif.SEED_VOLUME_ARMS and SEED_VOLUME_SIM
opens_w1 = notif.opens_week_1(users)  # opens per user in the first 7 days of the same run
print(per_user["arm"].value_counts().sort_index().rename("users per cadence").to_string())
per_user.head()

# %% [markdown]
# ## Exercise 2 · The volume curve (7 minutes)
#
# **Predict.** Which cadence will have the highest margin per user over the 90 days: 2, 3, 5 or
# 7 reminders a week? And which will have the most opens per user in the first week? Write both.
#
# **Task.** Write `volume_curve(per_user, opens_w1)` returning a `pd.DataFrame` with one row per
# cadence (`arm`, sorted) and columns `users` (rows in the arm), `sends` (mean per user),
# `open_rate` (total opens ÷ total sends in the arm; 0 where nothing was sent), `opens_week_1`
# (mean of `opens_w1` over the arm's users), `opt_out` (share of users with `opted_out`),
# `margin_7d` and `margin_90d` (means of `margin_7d` and `margin`).

# %% tags=["exercise"]
def volume_curve(per_user, opens_w1):
    # TODO 2: g = per_user.groupby("arm"); users = g.size(); sends = g["sends"].mean();
    # open_rate = g["opens"].sum() / g["sends"].sum() (0 where sends are 0);
    # opens_week_1 = opens_w1.groupby(per_user["arm"].to_numpy()).mean();
    # opt_out = g["opted_out"].mean(); margin_7d = g["margin_7d"].mean(); margin_90d = g["margin"].mean()
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def volume_curve(per_user, opens_w1):
    g = per_user.groupby("arm")
    sends_total = g["sends"].sum()
    return pd.DataFrame({
        "users": g.size(),
        "sends": g["sends"].mean(),
        "open_rate": (g["opens"].sum() / sends_total.replace(0, np.nan)).fillna(0.0),
        "opens_week_1": opens_w1.groupby(per_user["arm"].to_numpy()).mean(),
        "opt_out": g["opted_out"].mean(),
        "margin_7d": g["margin_7d"].mean(),
        "margin_90d": g["margin"].mean(),
    })


# %% [markdown]
# The checkpoint compares every column with the reference and then checks the two facts the
# committed data show: the daily cadence has the most week-1 opens, and three a week the
# highest 90-day margin.

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    curve = notif.check_volume_curve(volume_curve, per_user, opens_w1)
    print(curve.round(3).to_string())

# %% [markdown]
# Look at where each panel peaks: week-1 opens rise with every extra send, 90-day margin peaks
# at three a week, and the share of users lost rises steeply after it.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
x = curve.index.to_numpy()
for ax, col, label in zip(axes, ("opens_week_1", "margin_90d", "opt_out"),
                          ("Opens per user, week 1", "Margin per user, 90 days ($)",
                           "Users who opted out, 90 days"), strict=True):
    ax.plot(x, curve[col], marker="o", color="#0e3a44")
    ax.set_xlabel("Reminders a week (cadence)")
    ax.set_title(label, fontsize=10)
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
axes[1].axvline(curve["margin_90d"].idxmax(), color="#b8860b", ls="--", lw=1)
show(fig)

# %% [markdown]
# The same curve by segment (90-day margin per user, then the share who opted out):

# %%
by_seg = per_user.pivot_table(index="segment", columns="arm", values="margin", aggfunc="mean")
by_seg_out = per_user.pivot_table(index="segment", columns="arm", values="opted_out", aggfunc="mean")
print("Margin per user over 90 days, by segment and cadence:")
print(by_seg.loc[list(env.SEGMENTS)].round(2).to_string())
print("\nShare of users who opted out, by segment and cadence:")
print(by_seg_out.loc[list(env.SEGMENTS)].round(3).to_string())

# %% [markdown]
# **Explain.** Compare the two peaks with your predictions. The week-1 metric and the quarter
# metric choose different cadences: which two mechanisms, both absent on day 1, separate them?
# Point to the `opt_out` column and to the never-send arm's 90-day margin (what a user is
# worth with no notifications at all).
#
# <details><summary>Why this solution works</summary>
#
# Opens count sends that were read; they rise with volume and never see what volume costs. Two
# things happen only with time. Sends keep the habit alive: an opened notification is a session,
# and a session today raises the chance of visiting tomorrow, which is why even one reminder a
# week lifts the quarter's margin well above the never-send arm. And sends accumulate fatigue:
# at five or seven a week the opt-out share rises steeply, and an opted-out user contributes
# nothing more from notifications, or nothing at all after an uninstall. Three a week is where
# the second curve starts to overtake the first. The 90-day window still understates the cost:
# a user lost on day 60 is also lost next quarter, which Part C prices and the Decision's reveal
# measures.
# </details>
#
# # Part C · Who should get it: uplift with a cost per send
#
# Part B chose one cadence for everyone. Module 11 chose whom to treat by the estimated effect of
# the treatment against its cost. Here the treatment is a send, its benefit is the margin it
# brings over the next week, and its cost is not its price: it is the chance that the user opts
# out, times what that user was worth. The send experiment gives both per segment.
#
# The next cell adds to each user-day of the send experiment the margin over the next seven
# days (`margin_fwd`, NaN where the window runs past day 28), and takes the **value at stake**
# per segment from the volume experiment: the 90-day margin per user under the best uniform
# cadence, three a week.

# %%
margin_fwd = notif.forward_margin(log)  # margin over days t .. t+6 for each user-day
value = notif.value_at_stake(per_user)  # 90-day margin per user in the 3-a-week arm, by segment
print("Value at stake per segment ($, 90-day margin in the 3-a-week arm):", value.round(2).to_dict())

# %% [markdown]
# ## Exercise 3 · A send rule per segment (8 minutes)
#
# **Predict.** Which segment will have the largest 7-day effect of a send in dollars: low, mid
# or high? And will any segment fail the rule `effect > opt-out probability × value at stake`
# when a send costs nothing? Write both.
#
# **Task.** Write `segment_uplift(log, margin_fwd, value_at_stake, cost_per_send)` returning a
# `pd.DataFrame` indexed by segment (`low`, `mid`, `high`) with columns `cate` (the difference in
# mean `margin_fwd` between sent and not-sent rows), `se`, `lo`, `hi` (as in Exercise 1),
# `p_optout` (mean of `opted_out_today` over the segment's sent rows), `cost`
# (`p_optout × value_at_stake[segment] + cost_per_send`) and `send` (`cate > cost`). Use only
# eligible user-days whose window is complete (`margin_fwd` not NaN), for every column.

# %% tags=["exercise"]
def segment_uplift(log, margin_fwd, value_at_stake, cost_per_send=0.0):
    # TODO 3: rows = log[(log["eligible"] == 1) & margin_fwd.notna()]; y = margin_fwd.loc[rows.index];
    # for each segment s: sent = rows["sent"] == 1 within the segment; cate = mean(y sent) - mean(y not);
    # se as in Exercise 1; p_optout = rows.loc[segment & sent, "opted_out_today"].mean();
    # cost = p_optout * value_at_stake[s] + cost_per_send; send = cate > cost
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def segment_uplift(log, margin_fwd, value_at_stake, cost_per_send=0.0):
    rows = log[(log["eligible"] == 1) & margin_fwd.notna()]
    y = margin_fwd.loc[rows.index]
    out = {}
    for s in env.SEGMENTS:
        seg = rows["segment"] == s
        sent = seg & (rows["sent"] == 1)
        skip = seg & (rows["sent"] == 0)
        cate = y[sent].mean() - y[skip].mean()
        se = np.sqrt(y[sent].var(ddof=1) / sent.sum() + y[skip].var(ddof=1) / skip.sum())
        p_out = rows.loc[sent, "opted_out_today"].mean()
        cost = p_out * value_at_stake[s] + cost_per_send
        out[s] = {"cate": cate, "se": se, "lo": cate - 1.96 * se, "hi": cate + 1.96 * se,
                  "p_optout": p_out, "cost": cost, "send": cate > cost}
    return pd.DataFrame(out).T


# %% [markdown]
# The checkpoint compares each column with the reference and checks that every segment's effect
# is positive on the committed data.

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    uplift = notif.check_segment_uplift(segment_uplift, log, margin_fwd, value)
    print(uplift.round(4).to_string())

# %% [markdown]
# A push costs nothing to send. Other channels do, and tolerate less: an e-mail costs a fraction
# of a cent, an SMS several cents, and both have their own opt-out curves (the briefing's channel
# table). The next cell applies your rule with three per-send costs. Look at which segment drops
# out first.

# %%
CHANNEL_COST = {"push": 0.0, "email": 0.002, "sms": 0.05}  # dollars per send (assumptions)
rule = pd.DataFrame({ch: segment_uplift(log, margin_fwd, value, c)["send"]
                     for ch, c in CHANNEL_COST.items()})
print("Send to the segment? (effect over 7 days > opt-out probability x value at stake + cost per send)")
print(rule.to_string())

# %% [markdown]
# **Explain.** Compare with your prediction. The high segment has the largest effect in dollars
# and the smallest opt-out probability per send: what about those users makes both true? Then
# say why a rule built from a single send's 7-day effect cannot choose a cadence (hint: what
# happens to `p_optout` after the third send of the week?).
#
# <details><summary>Why this solution works</summary>
#
# Engaged users open more often, buy more per session and tolerate more sends, so a send to
# them earns more and risks less; low-engagement users earn little per send and opt out more
# readily, so they are the first to fail once a send has a price. The rule is a screen, not a
# schedule: it prices one send at the fatigue level the experiment happened to have (0.3 sends a
# day on average). A policy that sends daily moves every user up the fatigue curve, where the
# opt-out probability is several times higher, so the same rule would say less. Choosing how
# often needs the state the user is in, which is what Parts D and E add.
# </details>
#
# # Part D · A contextual bandit
#
# A **contextual bandit** chooses an action for each **context** and learns from the reward it
# gets, one day at a time, without a model of what comes next. Here a context is the user's
# segment, fatigue bucket and days since the last session (54 contexts; the table below), and
# the actions are `skip`, `reminder`, `offer` and `social`. **Thompson sampling** keeps, for each
# context and action, the sum of rewards and the count of tries; each day it draws one value per
# action from Normal(mean, σ / √(n + 1)) with mean = sum / (n + 1) and plays the action with the
# largest draw. Actions with few tries have wide draws, so they keep being tried; actions that
# have lost often enough stop being drawn on top.
#
# The reward is the design decision. The loop in `notif.run_bandit` gives a user-day the margin
# earned that day, **minus a penalty if the user opted out that day**. With penalty 0 the bandit
# sees opt-outs as free. The penalty is a number you choose, and the lab tries several.

# %%
labels = notif.context_labels()
print(f"{len(labels)} contexts, for example:")
print(labels.iloc[[0, 10, 25, 40, 53]].to_string())

# %% [markdown]
# ## Exercise 4 · The Thompson-sampling step (9 minutes)
#
# **Predict.** With penalty 0, will the bandit end the quarter having sent more or fewer
# notifications per user than the 3-a-week cadence (about 38)? And will its opt-out share be
# above or below the daily cadence's? Write both.
#
# **Task.** Write `thompson_choose(sums, counts, sigma, rng)`: `sums` and `counts` are arrays of
# shape (contexts, actions). Draw, for every entry, `rng.normal(mean, sigma / sqrt(counts + 1))`
# with `mean = sums / (counts + 1)`, and return the index of the largest draw in each row (an
# integer array of length `contexts`).

# %% tags=["exercise"]
def thompson_choose(sums, counts, sigma, rng):
    # TODO 4: mean = sums / (counts + 1); spread = sigma / np.sqrt(counts + 1);
    # draws = rng.normal(mean, spread); return draws.argmax(axis=1)
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def thompson_choose(sums, counts, sigma, rng):
    mean = sums / (counts + 1)
    spread = sigma / np.sqrt(counts + 1)
    return rng.normal(mean, spread).argmax(axis=1)


# %% [markdown]
# The checkpoint runs your function on a fixed table: with 1,000 tries per action and one
# action far ahead, it must pick that action almost always; with no data, it must try every
# action some of the time; and it must draw from the generator it is given.

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    notif.check_thompson(thompson_choose)
    print("Thompson step: exploits where the data are clear, explores where they are not.")

# %% [markdown]
# Now the bandit runs for 90 days with four penalties: 0, the value at stake `V` (per segment,
# from Part C), 3 V and 10 V. Each run starts from the same users and the same random numbers
# (the simulator's seed), so the runs are comparable pair by pair. Look at how sends and
# opt-outs move with the penalty.

# %%
PENALTIES = {"bandit, penalty 0": 0.0, "bandit, penalty V": value,
             "bandit, penalty 3V": 3 * value, "bandit, penalty 10V": 10 * value}
bandit_runs, bandit_margins = {}, {}
for name, pen in PENALTIES.items():
    out, means = notif.run_bandit(users, pen, choose=thompson_choose)
    bandit_runs[name] = notif.summarize(out.per_user)
    bandit_margins[name] = out.per_user["margin"].to_numpy()
bandit_table = pd.DataFrame(bandit_runs).T
print(bandit_table[["sends_per_user", "open_rate", "opt_out", "margin_7d", "margin_90d"]]
      .round(3).to_string())
print("For comparison, the 3-a-week arm of the volume experiment:",
      curve.loc[3, ["sends", "open_rate", "opt_out", "margin_7d", "margin_90d"]].round(3).to_dict())

# %% [markdown]
# What the last bandit (penalty 10V) learned, for the mid segment: its mean reward per action by
# fatigue bucket among users who visited within the last day.

# %%
mid_fresh = labels[(labels["segment"] == "mid") & (labels["days_since"] == "<2")].index
learned = means.loc[mid_fresh].copy()  # `means` is the last run's: penalty 10V
learned.insert(0, "fatigue", labels.loc[mid_fresh, "fatigue"])
print("Mean reward per action (last run, penalty 10V), mid segment, visited within a day:")
print(learned.round(3).to_string())

# %% [markdown]
# **Explain.** Compare the penalty-0 row with your predictions. The bandit sees each day's
# margin and, with a penalty, each day's opt-outs; what does it still not see about a send
# (think of what a session does to tomorrow's organic visit)? Point to the `margin_7d` column:
# why is it almost the same for every penalty while `margin_90d` is not?
#
# <details><summary>Why this solution works</summary>
#
# A bandit optimizes the reward it is given, one day at a time. With penalty 0 the reward is the
# day's margin, which a send raises a little and never lowers, so the bandit sends to nearly
# everyone nearly every day and loses close to half the base, like the daily cadence. The
# penalty makes an opt-out cost something today, so in high-fatigue contexts the skip action
# wins and sends fall with the penalty. What the bandit never sees is the habit effect: a session
# today raises tomorrow's organic visit, a reward that arrives later and is credited to whatever
# action is played then. The week-1 margin is nearly the same for every penalty because opt-outs
# are still rare in week 1; by day 90 the penalties have produced different bases.
# </details>
#
# # Part E · A policy for the long run: value iteration on the volume log
#
# Reinforcement learning treats the user as a **state** that actions move: fatigue goes up with a
# send and decays, days since the last session reset with a session, and opting out is an
# **absorbing state** nothing leaves. The value of an action is today's reward plus the
# discounted value of the state it leads to, so a send is credited with the sessions it keeps
# alive and charged with the opt-outs it causes, whenever they arrive.
#
# The volume experiment's log is the training data: 90 days of every user's state before each
# day, the action (skip or reminder) and the margin that day. `notif.transition_model` counts,
# for each state and action, where users went next (`P[s, a, s']`), the mean margin (`R[s, a]`)
# and the number of visits (`N[s, a]`). The states are the 54 contexts of Part D plus the
# opted-out state. Meta's Horizon paper trains a send-or-drop policy with "rewards for
# interactions and activity, with a penalty for sending": `notif.with_send_penalty(R, λ)` takes
# λ dollars off every send's reward, so a send must earn it back in margin or in later sessions.

# %%
per_user_again, volume_log = notif.volume_experiment(users, log=True)  # the same run, logged
assert per_user_again["margin"].equals(per_user["margin"])
P, R, N = notif.transition_model(volume_log)
S, A = R.shape
n_transitions = int(N[:notif.OUT_STATE].sum() + N[notif.OUT_STATE, 0])
print(f"{S} states x {A} actions; {n_transitions:,} transitions; "
      f"{int((N[:notif.OUT_STATE] < 30).sum())} state-action pairs seen fewer than 30 times")
print(f"Mean margin per user-day: skip ${R[:notif.OUT_STATE, 0].mean():.3f}, "
      f"send ${R[:notif.OUT_STATE, 1].mean():.3f}; opted-out state ${R[notif.OUT_STATE, 0]:.3f}")

# %% [markdown]
# ## Exercise 5 · Value iteration (8 minutes)
#
# With the model known, the optimal action values satisfy, for every state $s$ and action $a$,
#
# $$ Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, \max_{a'} Q(s', a') , $$
#
# where $\gamma$ (0.98 below, a horizon of about 50 days) discounts tomorrow. **Value iteration**
# starts from $Q = 0$ and applies the right-hand side as an update until nothing changes.
#
# **Predict.** With a per-send penalty of \$0.06 and γ = 0.98, will the learned policy send more
# or fewer notifications per user over the quarter than three a week (about 38)? Write a number.
#
# **Task.** Write `q_iteration(P, R, gamma, n_iter)` returning the array `Q` of shape
# (states, actions) after `n_iter` sweeps of `Q ← R + gamma * (P @ Q.max(axis=1))`, starting
# from zeros. `P @ v` with `P` of shape (S, A, S) and `v` of length S gives, for each state and
# action, the expected `v` of the next state.

# %% tags=["exercise"]
def q_iteration(P, R, gamma, n_iter=1000):
    # TODO 5: Q = np.zeros(R.shape); repeat n_iter times: Q = R + gamma * (P @ Q.max(axis=1)); return Q
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def q_iteration(P, R, gamma, n_iter=1000):
    Q = np.zeros(R.shape)
    for _ in range(n_iter):
        Q = R + gamma * (P @ Q.max(axis=1))
    return Q


# %% [markdown]
# The checkpoint runs your function on a two-state example worked by hand: from state 0 a send
# earns 1 and leads to state 1, where a send earns −1 and a skip returns to state 0 for free;
# with γ = 0.5 the values are V = [4/3, 2/3].

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    notif.check_q_iteration(q_iteration)
    print("Value iteration matches the hand-worked example and has converged.")

# %% [markdown]
# Now the policies. `notif.greedy_policy(Q, N)` takes the best action by `Q` in each state **among
# the actions the data tried at least 30 times there**: an action never tried in a state has a
# `Q` built from nothing, and choosing it is how offline reinforcement learning goes wrong (the
# LinkedIn papers in the briefing use conservative Q-learning for the same reason). Four per-send
# penalties are tried. Look at how the number of sends and the opt-outs move.

# %%
GAMMA = 0.98
SEND_PENALTIES = [0.0, 0.03, 0.06, 0.1]
rl_policies = {}
for lam in SEND_PENALTIES:
    Q = q_iteration(P, notif.with_send_penalty(R, lam), GAMMA)
    policy, best = notif.greedy_policy(Q, N)
    rl_policies[f"value iteration, send penalty ${lam:.2f}"] = policy
    if lam == 0.06:
        rule_06 = pd.Series(np.array(env.ACTIONS[:2])[best[:notif.OUT_STATE]], index=labels.index)
rl_table, rl_margins = notif.evaluate(users, rl_policies)
print(rl_table[["sends_per_user", "open_rate", "opt_out", "margin_7d", "margin_90d"]].round(3).to_string())

# %% [markdown]
# The learned rule at penalty \$0.06, for the mid segment: send or skip by fatigue bucket (rows)
# and days since the last session (columns).

# %%
mid = labels[labels["segment"] == "mid"].copy()
mid["action"] = rule_06.loc[mid.index]
print(mid.pivot(index="fatigue", columns="days_since", values="action")
      .loc[labels["fatigue"].unique(), labels["days_since"].unique()].to_string())

# %% [markdown]
# **Explain.** Compare the sends per user at penalty \$0.06 with your number. The value-iteration
# policies send less than three a week and lose far fewer users, but their 90-day margin is at
# or below the 3-a-week arm's. Which of these explains it best: the model's states are coarse
# (two users in one state can differ), the model only saw the cadences the experiment ran, or
# the discount shortens the horizon? Point to the rule table above: which fatigue bucket flips
# to skip?
#
# <details><summary>Why this solution works</summary>
#
# Value iteration is exact for the model it is given; the model is the approximation. The state
# is three buckets of fatigue and days since a session within a segment, so users with
# different hidden engagement share a row, and the mean reward and transitions are blends. The
# volume experiment ran six cadences, so the model saw high-fatigue states only through the
# daily and five-a-week arms, and `greedy_policy` refuses actions the data never tried. Within
# those limits the policy does what the model says: send while fatigue is low, stop once it is
# not, which protects the base at a small cost in quarter margin. A richer state (the observed
# features, recency of opens) and a logging policy with more variety are what the production
# systems in the briefing add.
# </details>
#
# ## Decision · Which policy next quarter?
#
# **The number.** Every policy's margin per 1,000 users over 90 days and its opt-out share, on the
# same users and random numbers, with the paired 95% interval of its margin against the best
# fixed cadence (three a week).
#
# **The rule**, fixed before any number is read: the retailer's growth team keeps a **reachable-
# base guardrail**: a policy may lose at most 10% of its reachable users to opt-outs in a quarter,
# because a user lost is lost for every later quarter too. Among the policies within the
# guardrail, choose the one with the highest 90-day margin per 1,000 users, and report whether
# its paired interval against three a week excludes zero.
#
# Decide as you would at work, where there is no second quarter to look at yet: use the table.
# The reveal, after your sentence, runs the top candidates for 180 days.

# %%
GUARDRAIL = 0.10  # at most a tenth of the reachable base lost to opt-outs in a quarter
fixed_policies = {f"fixed, {k} a week": env.fixed_policy(k) for k in env.CADENCES}
table, margins = notif.evaluate(users, {**fixed_policies, **rl_policies})
for name in PENALTIES:
    table.loc[name] = bandit_runs[name]
    margins[name] = bandit_margins[name]
base = "fixed, 3 a week"
paired = pd.DataFrame({n: notif.paired_difference(margins, n, base) for n in table.index}).T
table["vs 3/week, $ per 1,000"] = 1000 * paired["difference"]
table["95% lo"] = 1000 * paired["lo"]
table["95% hi"] = 1000 * paired["hi"]
table["within guardrail"] = table["opt_out"] <= GUARDRAIL
table["opt-outs %"] = 100 * table["opt_out"]
cols = ["sends_per_user", "opt-outs %", "margin_90d_per_1000", "vs 3/week, $ per 1,000", "95% lo",
        "95% hi", "within guardrail"]
print(table[cols].round(1).to_string())
eligible = table[table["within guardrail"]]
choice = eligible["margin_90d_per_1000"].idxmax()
best_any = table["margin_90d_per_1000"].idxmax()
print("\nNumber")
print(f"  Best within the guardrail: {choice}: ${table.loc[choice, 'margin_90d_per_1000']:,.0f} per"
      f" 1,000 users over 90 days, opt-outs {table.loc[choice, 'opt_out']:.1%}; against three a"
      f" week {table.loc[choice, 'vs 3/week, $ per 1,000']:+,.0f} (95% paired interval"
      f" {table.loc[choice, '95% lo']:+,.0f} to {table.loc[choice, '95% hi']:+,.0f}).")
print(f"  Best ignoring the guardrail: {best_any}: ${table.loc[best_any, 'margin_90d_per_1000']:,.0f}"
      f" per 1,000 users, opt-outs {table.loc[best_any, 'opt_out']:.1%}.")
print("Rule")
print(f"  Highest 90-day margin among policies that lose at most {GUARDRAIL:.0%} of the reachable"
      " base in the quarter.")
print("Recommendation")
excludes = table.loc[choice, "95% lo"] > 0 or table.loc[choice, "95% hi"] < 0
print(f"  Run {choice} next quarter. It earns ${table.loc[choice, 'margin_90d_per_1000']:,.0f} per"
      f" 1,000 users and loses {table.loc[choice, 'opt_out']:.1%} of them; its margin differs from"
      f" three a week by {table.loc[choice, 'vs 3/week, $ per 1,000']:+,.0f} per 1,000"
      f" ({'interval excludes zero' if excludes else 'interval includes zero: no measured difference'}).")
print(f"  What would change it: a guardrail above {table.loc[best_any, 'opt_out']:.0%} would admit"
      f" {best_any} (+${table.loc[best_any, 'vs 3/week, $ per 1,000']:,.0f} per 1,000 this quarter),"
      " or evidence that the users it loses were worth less than the value at stake assumes.")
print(f"  Mode: {'QUICK run: intervals are about twice as wide; a choice near the guardrail can flip' if QUICK else 'FULL run'}.")

# %% [markdown]
# **Decide.** Write one sentence: the policy, its margin per 1,000 users and opt-out share, how it
# compares with three a week (with the interval), and what would change your choice.
#
# ```text
# Your sentence: Next quarter, run ____ : about $____ per 1,000 users over 90 days, losing ____%
# of the reachable base (guardrail 10%), ____ per 1,000 against three a week (95% interval
# ____ to ____). I would change it if ____.
# ```
#
# Write it in `MY_SENTENCE` in the next cell. **Stop here until it is written:** the cell after
# it does what a quarter cannot, and refuses to run while the sentence is empty. It runs the best
# policy within the guardrail, the best policy ignoring it, and three a week for 180 days, with
# the same users and random numbers.

# %%
MY_SENTENCE = ""  # your decision sentence, in the form above

# %%
if not MY_SENTENCE.strip() and not WORKED_EXAMPLE:
    raise RuntimeError("Write your decision sentence in MY_SENTENCE in the cell above, then run"
                       " this cell: the reveal only teaches after you have decided.")
candidates = {n: ({**fixed_policies, **rl_policies}.get(n)) for n in {choice, best_any, base}}
long_run = {}
for name, policy in candidates.items():
    if policy is None:  # a bandit: rerun it for 180 days with its penalty
        out, _ = notif.run_bandit(users, PENALTIES[name], days=180, choose=thompson_choose)
    else:
        out = env.Simulator(users, notif.SEED_EVAL).run(policy, 180)
    long_run[name] = {"opt-outs % (180 days)": 100 * out.per_user["opted_out"].mean(),
                      "margin_180d_per_1000": 1000 * out.per_user["margin"].mean()}
reveal = pd.DataFrame(long_run).T
reveal["margin_90d_per_1000"] = table.loc[reveal.index, "margin_90d_per_1000"]
reveal["second quarter, $ per 1,000"] = reveal["margin_180d_per_1000"] - reveal["margin_90d_per_1000"]
print(reveal[["margin_90d_per_1000", "margin_180d_per_1000", "second quarter, $ per 1,000",
              "opt-outs % (180 days)"]].round(1).to_string())
print(f"\nOver 180 days the best policy is {reveal['margin_180d_per_1000'].idxmax()}"
      f"{' (QUICK run: 5,000 users)' if QUICK else ''}.")

# %% [markdown]
# **About this simulation.** The opt-out curve, the habit effect and the value of a session are
# assumptions of this simulated world, chosen so that every mechanism a growth team argues
# about is present and measurable. On a real app the same two experiments (a randomized send
# experiment and a volume experiment) give the numbers; the penalty a bandit is given, and the
# guardrail, are decisions, not parameters the data return.
#
# ## Stretch (optional) · Exploration, a referral loop, and real data
#
# 1. **Exploration.** Rerun the penalty-3V bandit with `sigma` 1 and 10 (`notif.run_bandit(...,
#    sigma=...)`). Which setting sends more in the first week, and which earns more by day 90?
#    Why does a small σ risk settling on the wrong action in a context it has barely tried?
# 2. **A referral loop.** A "your friend joined" notification sends an invitation when opened.
#    With the social template's open rate from the bandit's learned means, an invitation rate of
#    0.1 per open and an acceptance rate of 0.2 per invitation (assumptions), compute the
#    **viral factor** K = sends per user per week × open rate × 0.1 × 0.2: the new users each
#    user brings a week. Compare it with the weekly loss rate under the same cadence (the
#    quarter's opt-out share ÷ 13). The loop grows the base only while K exceeds the loss, and
#    every extra send moves both.
# 3. **Real data.** Duolingo released the 200 million practice reminders behind its KDD 2020
#    bandit paper (the briefing's references): sends with their template, time and whether the
#    user practiced within two hours. Which of this lab's exercises could run on it as it is,
#    and which need something the file does not have (a control group; opt-outs)?

# %% tags=["solution"]
# @title Stretch solutions — try them yourself first { display-mode: "form" }
# 1. Exploration
for sigma in (1.0, 3.0, 10.0):
    out, _ = notif.run_bandit(users, 3 * value, sigma=sigma, choose=thompson_choose)
    s = notif.summarize(out.per_user)
    print(f"sigma {sigma:>4}: week-1 sends per user {out.per_day['sends'][:7].sum() / len(users):.2f},"
          f" 90-day margin per user ${s['margin_90d']:.2f}, opt-outs {s['opt_out']:.1%}")

# 2. A referral loop, at three a week
open_rate_3 = curve.loc[3, "open_rate"]
K = 3 * open_rate_3 * 0.1 * 0.2
loss = curve.loc[3, "opt_out"] / 13
print(f"\nReferral loop at 3 a week: K = 3 x {open_rate_3:.3f} x 0.1 x 0.2 = {K:.4f} new users per"
      f" user per week; weekly loss {loss:.4f}; the base {'grows' if K > loss else 'shrinks'}.")
K7 = 7 * curve.loc[7, "open_rate"] * 0.1 * 0.2
loss7 = curve.loc[7, "opt_out"] / 13
print(f"At 7 a week: K = {K7:.4f}, weekly loss {loss7:.4f}; the base {'grows' if K7 > loss7 else 'shrinks'}.")

# 3. Real data: Exercises 1 and 3 need a control group of eligible users who were not sent
# anything on the day, which the Duolingo file does not have (every row is a send); Exercise 2
# needs opt-outs, which it does not record; Exercise 4's bandit is what the paper ran, on the
# template choice, with "practiced within two hours" as the reward.
