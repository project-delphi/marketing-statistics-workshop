# %% [markdown]
# <!--
# Module 16 lab; brief: briefs/16-multi-touch-attribution.md. No attribution library: the
# learner writes every method with NumPy and pandas (ChannelAttribution 2.2.5 has no Shapley
# function and ships as source only on PyPI; see the brief). APIs checked 2026-10-10 against the
# installed source in the workshop's environment: numpy 2.1.3 `np.linalg.solve(a, b)`,
# `Generator.multinomial(n, pvals)`, `np.quantile(a, q, axis=0)`; pandas 2.2.3 `Series.str.split`,
# `Series.str[i]`, `DataFrame.explode(column)`, `Series.unstack(fill_value=0)`,
# `DataFrame.reindex(index=, columns=, fill_value=)`; SciPy 1.16.3 `stats.norm.ppf`.
# Checks and the bootstrap machinery: src/mktstats/attribution.py. Data: src/mktstats/synth/mta.py
# (seed 2036), committed as data/synthetic/mta_journeys.csv and truth_mta.json.
# -->
#
# **How this lab runs.** Nothing here fits a model: every method is counting, one small linear
# solve, or an exact sum over at most 32 sets of channels, so every cell runs in seconds. `QUICK`
# (in the harness cell) changes only the number of bootstrap resamples in Part D: every credit
# number, every checkpoint and the decision are the same in both modes. With QUICK on, the
# bootstrap intervals are rougher; that is a lesson about the number of resamples, not a bug.
#
# The data are simulated: one month of a retailer's online marketing to 100,000 users through
# five channels: **display** (prospecting banner ads), **social** (paid social ads), **search**
# (paid search ads), **email** (sent to the retailer's subscribers) and **retargeting** (display
# ads shown to recent visitors of the retailer's site). Because the simulation knows which
# conversions each channel caused, you can score every attribution method against the truth,
# which real tracking data never allow.

# %%
import io
import itertools
import json
import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy import stats

from mktstats import attribution
from mktstats.data import repo_file

SEED = 2036  # the data's seed; every random step below starts from it
N_BOOT = 50 if QUICK else 200  # Part D: bootstrap resamples of the journeys


def show(fig):
    """Show a figure as a PNG on Colab, in Jupyter and in a headless run alike."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    display(Image(buf.getvalue()))


print(f"QUICK = {QUICK}: {N_BOOT} bootstrap resamples (the only thing QUICK changes)")

# %% [markdown]
# # Part A · Rules of thumb
#
# A **touch** is one ad impression, click or email that the tracking records for a user. A
# **journey** (or path) is all of one user's touches in the month, in time order, for example
# `display > search > retargeting`. A journey **converts** if the user bought at the end of the
# month. **Credit** is the conversions a method assigns to a channel; every method hands out all
# the tracked conversions.
#
# The next cell loads the journeys. The file stores one letter per touch to stay small; the cell
# spells each path out. Users who saw no touch have no journey: the tracking never saw them. The
# column `true_intent` is the simulation's purchase intent, which real tracking never records; no
# method below uses it.

# %%
truth = json.loads(repo_file("synthetic/truth_mta.json").read_text())
CHANNELS = truth["channels"]
journeys = pd.read_csv(repo_file("synthetic/mta_journeys.csv"))
journeys["path"] = journeys["path"].map(lambda s: " > ".join(truth["path_codes"][k] for k in s))
n_conv = truth["conversions_in_journeys"]

print(f"{len(journeys):,} journeys from {truth['n_users']:,} users; {n_conv:,} converted"
      f" ({n_conv / len(journeys):.2%})")
print(f"Touches per journey: {journeys['path'].str.count(' > ').add(1).mean():.2f} on average")
rng_show = np.random.default_rng(SEED)  # five converting and five other journeys, at random
examples = np.sort(np.concatenate([
    rng_show.choice(np.flatnonzero(journeys["converted"] == y), size=5, replace=False)
    for y in (1, 0)]))
with pd.option_context("display.max_colwidth", None):  # show every path in full
    display(journeys.iloc[examples][["user_id", "path", "converted"]])

# %% [markdown]
# The next table shows, for each channel, the share of journeys it appears in and, among the
# journeys with two or more channels that contain it, how often it **opens** (is the first touch)
# and how often it **closes** (is the last touch). Look at which channels open and which close.

# %%
touch_lists = journeys["path"].str.split(" > ")
multi = touch_lists.map(lambda t: len(set(t)) >= 2)
position = {}
for c in CHANNELS:
    has_c = journeys["path"].str.contains(c, regex=False)  # no channel name contains another
    both = has_c & multi
    position[c] = {
        "share of journeys": has_c.mean(),
        "opens (first touch)": (touch_lists[both].str[0] == c).mean(),
        "closes (last touch)": (touch_lists[both].str[-1] == c).mean(),
    }
position = pd.DataFrame(position).T
position.round(2)

# %% [markdown]
# `TINY` is a five-journey example over two channels, A and B, worked by hand in Exercises 1 to 3.

# %%
TINY = pd.DataFrame({
    "path": ["A > B", "A", "B", "B > A", "A > A > B"],
    "converted": [1, 0, 1, 0, 1],
})
TINY

# %% [markdown]
# ## Exercise 1 · Last-click, first-click and linear credit (8 minutes)
#
# Three rules of thumb, each fixed by position in the journey:
#
# - **last-click** gives each conversion to the last touch of its journey;
# - **first-click** gives it to the first touch;
# - **linear** splits each conversion equally over the touches of its journey, so a channel
#   touched twice gets two shares.
#
# **Predict.** Retargeting reaches about one user in six, appears in about a quarter of the
# journeys (the table above), and its touches come late in the month (the same table: how often
# each channel opens and closes). Under last-click, what share of the conversions will
# retargeting get: under 20%, 20 to 30%, or over 30%? Write your guess.
#
# **Task.** Write `heuristic_credit(journeys, rule, channels)` returning a `pd.Series` of credited
# conversions per channel (index `channels`, in that order, with 0 where a channel got none) for
# `rule` in `"last"`, `"first"` or `"linear"`. Only converting journeys give credit; under linear,
# each touch of a journey with L touches gets 1/L.

# %% tags=["exercise"]
def heuristic_credit(journeys, rule, channels):
    # TODO 1: keep the converting journeys and split their paths with .str.split(" > ");
    # last: .str[-1], first: .str[0], then .value_counts(); linear: give each journey's touches
    # 1 / (its number of touches), .explode() the touches and add up by channel;
    # finish with .reindex(channels, fill_value=0)
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def heuristic_credit(journeys, rule, channels):
    touches = journeys.loc[journeys["converted"] == 1, "path"].str.split(" > ")
    if rule == "last":
        credit = touches.str[-1].value_counts()
    elif rule == "first":
        credit = touches.str[0].value_counts()
    elif rule == "linear":
        each = pd.DataFrame({"channel": touches, "share": 1 / touches.str.len()})
        credit = each.explode("channel").groupby("channel")["share"].sum()
    else:
        raise ValueError(f"rule must be 'last', 'first' or 'linear', not {rule!r}")
    return credit.reindex(channels, fill_value=0).astype(float)


# %% [markdown]
# The checkpoint runs your function on `TINY`, worked by hand (last {A: 0, B: 3}; first
# {A: 2, B: 1}; linear {A: 7/6, B: 11/6}), then checks on the journeys that each rule's credit
# adds up to the converting journeys, with nothing negative.

# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    heuristic = attribution.check_heuristic_credit(heuristic_credit, journeys, CHANNELS, n_conv)
    print("Share of the conversions each rule gives each channel:")
    print((heuristic / n_conv).round(3).to_string())

# %% [markdown]
# **Explain.** Compare retargeting's last-click share with your guess. Last-click and first-click
# disagree most about retargeting and display. Which feature of the journeys decides it: when each
# channel's touches happen, how many users each channel reaches, or how often reached users
# convert? Point to the opens and closes columns of the table above.
#
# <details><summary>Why this solution works</summary>
#
# A positional rule credits whatever sits at that position. Display's touches come early in the
# month, so it opens most of the journeys it is in and collects first-click credit; retargeting's
# come late, so it closes most of its journeys and collects last-click credit. Neither rule asks
# whether the touch changed the outcome: move retargeting's touches earlier and its credit would
# follow them, with no change in what anyone bought. Linear spreads each conversion over the
# journey, but every touch of a converting journey still gets a share, whatever it did.
# </details>

# %% [markdown]
# # Part B · The journey as a Markov chain
#
# A data-driven alternative to fixed positions learns from all journeys, converting or not, how
# users move between channels. A **first-order Markov chain** models the journey as a sequence of
# **states** in which the next state depends only on the current one. The states are `start`, the
# five channels, `conv` (the journey ends in a conversion) and `null` (it ends without one). The
# **transition probability** p(a → b) is the share of departures from state a that go next to
# state b. `conv` and `null` are **absorbing states**: a journey that reaches one never leaves it.
#
# The next cell turns one journey into its states (provided: the string handling is not the
# lesson; counting transitions is).

# %%
def journey_states(path, converted):
    """One journey as its chain states: ("A > B", 1) -> ["start", "A", "B", "conv"]."""
    return ["start", *path.split(" > "), "conv" if converted else "null"]


print(journey_states(TINY.loc[0, "path"], TINY.loc[0, "converted"]))
print(journey_states(TINY.loc[4, "path"], TINY.loc[4, "converted"]))

# %% [markdown]
# ## Exercise 2 · The transition matrix (8 minutes)
#
# **Predict.** In the display row of the matrix, which next state will be most likely: another
# display touch, a touch from another channel (the four others together), conversion, or the end
# of the journey without conversion (null)? Write your guess.
#
# **Task.** Write `transition_matrix(journeys, channels)` returning a `pd.DataFrame` of transition
# probabilities with index and columns `["start", *channels, "conv", "null"]`. Each journey adds
# start → first touch, touch → next touch, and last touch → conv or null; conv and null go to
# themselves with probability 1; each row adds up to 1. Tip:
# `pd.Series(pairs).unstack(fill_value=0)` turns a `Counter` of (from, to) pairs into a table of
# counts, with a row per "from" state and a column per "to" state.

# %% tags=["exercise"]
def transition_matrix(journeys, channels):
    # TODO 2: count the consecutive pairs of journey_states(path, converted) over all journeys
    # (collections.Counter, or pd.crosstab); put the counts in a DataFrame with index and columns
    # ["start", *channels, "conv", "null"] (0 where a pair never occurs); set conv -> conv and
    # null -> null to 1; divide each row by its total
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def transition_matrix(journeys, channels):
    pairs = Counter()
    for path, converted in zip(journeys["path"], journeys["converted"], strict=True):
        states = journey_states(path, converted)
        pairs.update(zip(states[:-1], states[1:], strict=True))
    labels = ["start", *channels, "conv", "null"]
    M = pd.Series(pairs).unstack(fill_value=0)
    M = M.reindex(index=labels, columns=labels, fill_value=0).astype(float)
    M.loc["conv", "conv"] = M.loc["null", "null"] = 1.0
    return M.div(M.sum(axis=1), axis=0)


# %% [markdown]
# The checkpoint compares your matrix on `TINY` with the one worked by hand (start → A 3/5,
# start → B 2/5; A → A 1/5, A → B 2/5, A → null 2/5; B → A 1/4, B → conv 3/4), then checks the
# journeys' matrix: rows add up to 1, nothing returns to start, conv and null are absorbing, and
# the labels are in order.

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    attribution.check_transition_matrix(transition_matrix, journeys, CHANNELS)
    P = transition_matrix(journeys, CHANNELS)
    print(P.round(3).to_string())

# %% [markdown]
# **Explain.** Compare the display row with your guess. A run of display touches shows up as the
# display → display entry, a **self-loop**. The next cell divides the other entries of the display
# row by 1 − p(display → display): the chance of each next state once the journey leaves display.
# Does the self-loop change where the journey goes after display, or only how long it stays there?

# %%
p_stay = P.loc["display", "display"]
leaves = P.loc["display"].drop(["start", "display"]) / (1 - p_stay)
print(f"p(display -> display) = {p_stay:.3f}. Once the journey leaves display, it goes to:")
print(leaves.round(3).to_string())

# %% [markdown]
# <details><summary>Why this solution works</summary>
#
# Once the journey leaves channel c, it goes to d with probability p(c → d) / (1 − p(c → c)),
# whatever p(c → c) is: a self-loop only delays the next step. So the probability of eventually
# converting from c (x_c, used next) does not depend on repeated touches in a row:
# x_c = p(c → c) x_c + Σ_{d≠c} p(c → d) x_d gives x_c = Σ_{d≠c} p(c → d) x_d / (1 − p(c → c)).
# Collapsing every run of repeated touches into one touch leaves every removal effect of
# Exercise 3 unchanged (checked on these journeys when the lab was built): Markov credit does not
# reward a channel for repeating itself.
# </details>
#
# **Provided: the probability of reaching conversion.** Let x_a be the probability that a journey
# now in state a eventually converts. Take one step, then carry on from where you land:
# x_a = p(a → conv) + Σ_b p(a → b) x_b, summing over the **transient** states b (start and the
# channels). In matrix form x = Q x + r, so
#
# $$ x = (I - Q)^{-1} r, $$
#
# where Q is P restricted to the transient states and r is the column of P for conv on those
# rows. The chain's **conversion probability** is p = x_start. `conversion_probability(P)` below
# solves this linear system. A chain estimated by counting reproduces the observed conversion
# share exactly; the check after it confirms that on `TINY` (p = 3/5) and on the journeys.
#
# <details><summary>Why the chain reproduces the observed share (and TINY by hand)</summary>
#
# TINY: x_A = x_A/5 + 2x_B/5 and x_B = x_A/4 + 3/4, so x_A = x_B/2, x_B = 6/7, x_A = 3/7, and
# p = x_start = (3/5)(3/7) + (2/5)(6/7) = 3/5: three of TINY's five journeys converted.
#
# In general, let n_ab be the number of a → b transitions and n_a = Σ_b n_ab the departures from
# a, so p(a → b) = n_ab / n_a. The chain's equations are n_a x_a = Σ_b n_ab x_b, with x_conv = 1
# and x_null = 0. Add them up over the transient states: each channel state is entered exactly as
# often as it is left, so every channel term cancels, leaving n_start x_start = (transitions into
# conv) = Y, the converting journeys. With n_start = N journeys, x_start = Y / N.
# </details>

# %%
def conversion_probability(P):
    """x_start, the probability that a journey eventually reaches conv: x = (I - Q)^-1 r."""
    T = [s for s in P.index if s not in ("conv", "null")]  # the transient states
    Q = P.loc[T, T].to_numpy()
    r = P.loc[T, "conv"].to_numpy()
    x = np.linalg.solve(np.eye(len(T)) - Q, r)
    return float(x[T.index("start")])


# %% tags=["checkpoint"]
with workshop.checkpoint(label="chain"):
    P_TINY = transition_matrix(TINY, ["A", "B"])
    p_chain = attribution.check_chain(conversion_probability, P_TINY, P, n_conv / len(journeys))
    print(f"TINY: p = {conversion_probability(P_TINY):.4f} (3/5 by hand)")
    print(f"Journeys: p = {p_chain:.6f}; share of journeys that converted ="
          f" {n_conv / len(journeys):.6f}")

# %% [markdown]
# ## Exercise 3 · Removal effects (8 minutes)
#
# The **removal effect** of channel c is the share of the chain's conversion probability that is
# lost when every journey that reaches c stops there without converting:
#
# $$ \text{RE}_c = 1 - \frac{p_{-c}}{p}, $$
#
# where p is the chain's conversion probability and p₋c the same with c removed. To remove c, set
# row c of P to 0 and p(c → null) to 1: a journey that reaches c ends there, unconverted.
#
# **Predict.** Will the five channels' removal effects add up to less than 1, about 1, or more
# than 1? Write your guess.
#
# **Task.** Write `removal_effects(P, channels)` returning a `pd.Series` of RE_c (index
# `channels`), using the provided `conversion_probability`. Work on a copy of P for each channel
# (`P.copy()`): your function must not change `P`.

# %% tags=["exercise"]
def removal_effects(P, channels):
    # TODO 3: p from the unchanged P; for each channel c, remove c from a copy of P (row c to 0,
    # then c -> null to 1) and compare the copy's conversion probability with p
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def removal_effects(P, channels):
    p = conversion_probability(P)
    effects = {}
    for c in channels:
        removed = P.copy()
        removed.loc[c, :] = 0.0
        removed.loc[c, "null"] = 1.0
        effects[c] = 1 - conversion_probability(removed) / p
    return pd.Series(effects)


# %% [markdown]
# The checkpoint uses `TINY`, worked by hand: p = 3/5, p₋A = 3/10 (journeys that reach A stop
# there), p₋B = 0, so RE = {A: 1/2, B: 1}. It also checks that `P` is unchanged after the call,
# and that on the journeys every removal effect lies between 0 and 1 and equals the reference.

# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    removal = attribution.check_removal_effects(removal_effects, P, CHANNELS)
    print(removal.round(3).to_string())
    print(f"Sum of the removal effects: {removal.sum():.2f}")

# %% [markdown]
# **Markov credit** shares the Y converting journeys' conversions in proportion to the removal
# effects: credit_c = Y · RE_c / Σ_k RE_k. The check after it uses `TINY`: Y = 3 and removal
# effects 1/2 and 1 (sum 1.5) give A 1 and B 2.

# %%
def markov_credit(removal, conversions):
    """credit_c = Y * RE_c / sum_k RE_k (the lab's normalization)."""
    removal = pd.Series(removal, dtype=float)  # a dict {channel: RE_c} works too
    return conversions * removal / removal.sum()


# %% tags=["checkpoint"]
with workshop.checkpoint(label="Markov credit"):
    tiny_markov = markov_credit(removal_effects(transition_matrix(TINY, ["A", "B"]), ["A", "B"]), 3)
    attribution.check_markov_credit(tiny_markov)
    markov = markov_credit(removal, n_conv)
    print("TINY:", tiny_markov.round(3).to_dict())
    print("Markov credit on the journeys:", markov.round(0).to_dict())

# %% [markdown]
# **Explain.** Compare the sum with your guess. In `TINY` the removal effects are 1/2 and 1,
# adding up to 1.5. Which converting journeys pass through both A and B, and what happens to them
# when A is removed, and when B is removed? Then: the Markov credit above divides each removal
# effect by their sum. Without that division, what would the credit on the journeys add up to
# (the sum printed above times the 3,020 conversions), and is the division measured from the data
# or chosen?
#
# <details><summary>Why this solution works</summary>
#
# Removing one channel breaks every path through it, so a converting journey through three
# channels is lost in three removal scenarios: the effects overlap and add up to more than 1. In
# `TINY`, journeys 1 and 5 (A > B and A > A > B) pass through both channels and are lost whichever
# channel is removed. Without the division the credit would add up to about 1.97 × 3,020 ≈ 5,950,
# twice the conversions there are. Dividing by the sum is a convention: it shares the overlap in
# proportion to each channel's removal effect, and nothing in the data says it should be shared
# that way. And the removal effect asks what the *fitted chain* predicts if journeys stopped at
# c: it treats "on the path to a conversion" as "caused the conversion".
# </details>

# %% [markdown]
# # Part C · Shapley values
#
# Cooperative game theory asks how to share a value that players create together. A
# **coalition** is a set of players (here: channels), and a value function v(S) gives the value
# that coalition S creates, with v(∅) = 0. The **Shapley value** of player c is c's added value,
# v(T ∪ {c}) − v(T), averaged over every order in which the players could have joined:
#
# $$ \varphi_c = \sum_{T \subseteq N \setminus \{c\}} \frac{|T|!\,(n - |T| - 1)!}{n!}
#    \big[v(T \cup \{c\}) - v(T)\big], $$
#
# where N is the set of all n players and the sum runs over the coalitions T that leave c out
# (the weight is the share of join orders in which exactly the players of T come before c).
# Shapley (1953) derived it from three properties:
#
# - **efficiency**: the players' values add up to v(N) − v(∅);
# - **symmetry**: two players that add the same to every coalition get the same value;
# - **null player**: a player that adds nothing to any coalition gets 0.
#
# For journeys, the players are the channels on a journey, and v(S) is the **conversion rate of
# the journeys whose set of channels is exactly S** (in any order, with any number of touches),
# with v(∅) = 0 because users with no touch are not in the data. Each converting journey is split
# among its channels in proportion to their Shapley values in the game on its own set S.

# %% [markdown]
# ## Exercise 4 · Exact Shapley values (9 minutes)
#
# **Predict.** In the three-player game below, C adds nothing to any coalition, so the null-player
# property gives it 0, and A and B share v(ABC) = 0.40. Alone, A makes 0.10 and B makes 0.20;
# together they make 0.10 more than that. What will B's Shapley value be: 0.20 (an equal split),
# 0.25 (each keeps what it makes alone and the extra 0.10 is split equally), about 0.27 (a split
# in proportion to what each makes alone) or 0.30 (B keeps the extra)? Write your choice.
#
# | S | ∅ | A | B | C | AB | AC | BC | ABC |
# |---|---|---|---|---|---|---|---|---|
# | v(S) | 0 | 0.10 | 0.20 | 0 | 0.40 | 0.10 | 0.20 | 0.40 |
#
# **Task.** Write `shapley_values(players, v)` returning a dict {player: Shapley value} by exact
# enumeration of the formula above. `v` takes a `frozenset` of players and returns its value.

# %% tags=["exercise"]
def shapley_values(players, v):
    # TODO 4: for each player c, loop over k = 0 .. n - 1 and over the coalitions T of k other
    # players (itertools.combinations); add the weight math.factorial(k) *
    # math.factorial(n - k - 1) / math.factorial(n) times v(frozenset(T) | {c}) - v(frozenset(T))
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def shapley_values(players, v):
    n = len(players)
    values = {}
    for c in players:
        others = [p for p in players if p != c]
        total = 0.0
        for k in range(n):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for T in itertools.combinations(others, k):
                total += weight * (v(frozenset(T) | {c}) - v(frozenset(T)))
        values[c] = total
    return values


# %% [markdown]
# The checkpoint uses a two-player game (v(A) = 0.10, v(B) = 0.05, v(AB) = 0.20, so φ = {A: 0.125,
# B: 0.075}), the game above (φ = {A: 0.15, B: 0.25, C: 0}, averaging over the six orders), and a
# random five-player game (seed shown): your values must add up to v(N) − v(∅) (efficiency), equal
# the average over all 120 join orders, and swap when two players are swapped in v (symmetry).

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    phi_random = attribution.check_shapley_values(shapley_values, seed=SEED + 4)
    three = shapley_values(["A", "B", "C"], attribution.THREE_PLAYER_GAME.get)
    print("The three-player game:", {c: round(x, 4) for c, x in three.items()})
    print(f"A random five-player game (seed {SEED + 4}):",
          {c: round(x, 4) for c, x in phi_random.items()})

# %% [markdown]
# **Explain.** Compare with your guess. Your function passed the efficiency check: a set's Shapley
# values add up to v(S) − v(∅). For journeys, v(S) is the conversion rate of journeys whose set of
# channels is exactly S, and v(∅) = 0 because users with no touch are not in the data. Suppose
# users would convert at 1% with no marketing at all. Under Shapley, who gets the credit for those
# conversions: nobody, the channels on each converting journey, or only its last channel?
#
# <details><summary>Why this solution works</summary>
#
# The channels on the journey. With v(∅) = 0, the whole conversion rate of each set, the 1%
# baseline included, is shared out among its channels (efficiency). Shapley's properties are rules
# for sharing a value; they say nothing about whether v measures what a channel causes. That
# depends on v, and v here is an observed conversion rate. In the game above B gets 0.25 and A
# 0.15: averaged over the join orders, each keeps what it makes alone (0.20 and 0.10), and the
# 0.10 they make only together is split equally, not in proportion to what each makes alone.
# </details>
#
# The next cell builds v(S) for every set of channels that occurs (`coalitions`: journeys,
# conversions and rate = v(S) per set) and the Shapley credit, calling your `shapley_values` once
# per set. It stops with a message if a set needed by a converting journey has no journeys,
# instead of treating its rate as 0. It prints a summary; the table stays in `coalitions`. Look at
# the smallest Shapley value: a negative one would mean that adding a channel lowered a set's
# conversion rate (it is reported, never clipped).

# %%
coalitions = attribution.coalition_table(journeys)  # index: the set, e.g. "display + search"
v_rate = dict(zip(coalitions["channels"], coalitions["rate"], strict=True))


def v(S):
    """The conversion rate of the journeys whose set of channels is exactly S; v(empty) = 0."""
    if not S:
        return 0.0
    if S not in v_rate:
        raise ValueError(f"No journey has exactly the channels {sorted(S)}: v(S) is unknown.")
    return v_rate[S]


shapley = pd.Series(0.0, index=CHANNELS)
smallest_phi, worst_sum = np.inf, 0.0
converting_sets = coalitions[coalitions["conversions"] > 0]
for S, conversions, rate in converting_sets[["channels", "conversions", "rate"]].itertuples(
        index=False):
    phi = shapley_values(sorted(S), v)  # the game on this set's own channels
    smallest_phi = min(smallest_phi, min(phi.values()))
    worst_sum = max(worst_sum, abs(sum(phi[c] / rate for c in S) - 1))
    for c in S:
        shapley[c] += conversions * phi[c] / rate  # each conversion split in shares phi_c / v(S)

print(f"{len(coalitions)} channel sets occur; the smallest has {coalitions['journeys'].min():,}"
      " journeys.")
print(f"Smallest Shapley value over all sets: {smallest_phi:+.4f}; each set's shares add up to 1"
      f" (largest deviation {worst_sum:.1e}).")
print("Shapley credit:", shapley.round(0).to_dict())

# %% [markdown]
# # Part D · Credit against the truth
#
# Five methods, five answers. Because the data are simulated, we also know each channel's
# **incremental conversions**: the conversions that would not have happened had the channel not
# reached the user, counted over all 100,000 users (tracked or not), with everything else about
# each user unchanged. Real data never show this.
#
# Credit always adds up to the tracked conversions, while incremental conversions are fewer and do
# not add up across channels, so the lab compares **shares**: each method's share of the credit
# per channel against each channel's share of the true incremental conversions.

# %%
credit = pd.concat([heuristic, markov.rename("markov"), shapley.rename("shapley")], axis=1)
incremental = pd.Series(truth["incremental_conversions"])  # the truth; shown at the checkpoint
print("Credited conversions by method:")
credit.round(0)

# %% [markdown]
# Look at the conversion rate of the users each channel reached against those it did not reach.
# The rates come from the simulation's truth file, because users with no touch are not in the
# journeys.

# %%
rates = pd.DataFrame({"reached": truth["conversion_rate_reached"],
                      "not reached": truth["conversion_rate_not_reached"]}).loc[CHANNELS]
x = np.arange(len(CHANNELS))
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.bar(x - 0.2, 100 * rates["reached"], 0.4, color="#4c72b0", label="users the channel reached")
ax.bar(x + 0.2, 100 * rates["not reached"], 0.4, color="#dd8452", hatch="//",
       label="users it did not reach")
for i, (a, b) in enumerate(100 * rates.to_numpy()):
    ax.text(i - 0.2, a + 0.15, f"{a:.1f}", ha="center", fontsize=8)
    ax.text(i + 0.2, b + 0.15, f"{b:.1f}", ha="center", fontsize=8)
ax.set_xticks(x, CHANNELS)
ax.set_ylim(0, 1.3 * 100 * rates.to_numpy().max())
ax.set_ylabel("Conversion rate in the month (%)")
ax.set_title("Conversion rate of the users each channel reached and did not reach\n"
             f"(all {truth['n_users']:,} users)")
ax.legend(loc="upper left", ncol=2)
show(fig)

# %% [markdown]
# ## Exercise 5 · Score the methods against the truth (7 minutes)
#
# **Predict.** Under Markov and Shapley, retargeting gets about as much credit as display or
# social (the credit table above), and the users it reached converted far more often than those
# it did not reach (the plot). How many conversions did retargeting truly cause: about as many as
# its Markov credit, about half of it, or close to none? Write your guess.
#
# The **share error** of a method m is the share of credit that would have to move between
# channels for its shares to match the true incremental shares (the total variation distance):
#
# $$ D_m = \frac{1}{2} \sum_c \left| \frac{\text{credit}^m_c}{\sum_k \text{credit}^m_k}
#    - \frac{\Delta_c}{\sum_k \Delta_k} \right|, $$
#
# where Δ_c are channel c's true incremental conversions; D = 0 means the same shares, 1 no
# overlap.
#
# **Task.** Write `credit_vs_truth(credit, incremental)` returning a `pd.DataFrame` with one row
# per method (the columns of `credit`) and the columns `tvd` (D_m), `largest_gap_channel` (the
# channel whose credit share is furthest from its true share) and `largest_gap` (credit share −
# true share for that channel, with its sign). Tip: with `gap` a table of channels × methods,
# `gap.abs().idxmax()` gives each method's channel, and `gap.loc[channel, method]` its signed gap.

# %% tags=["exercise"]
def credit_vs_truth(credit, incremental):
    # TODO 5: shares of each method (divide each column by its total) and true shares
    # (incremental / its total); gap = shares - true shares (per channel, every method);
    # tvd = half the sum of |gap|; the channel with the largest |gap| (.abs().idxmax()) and its
    # signed gap
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def credit_vs_truth(credit, incremental):
    shares = credit / credit.sum()
    true = incremental.reindex(credit.index) / incremental.sum()
    gap = shares.sub(true, axis=0)
    worst = gap.abs().idxmax()
    return pd.DataFrame({
        "tvd": 0.5 * gap.abs().sum(),
        "largest_gap_channel": worst,
        "largest_gap": [gap.loc[worst[m], m] for m in gap.columns],
    })


# %% [markdown]
# The checkpoint uses a hand example (credit m1 = {A: 5, B: 3, C: 2} and m2 = {A: 2, B: 2, C: 6};
# incremental {A: 6, B: 2, C: 0}: m1 has D = 0.25 and its largest gap is A, −0.25; m2 has D = 0.60,
# largest gap C, +0.60), then compares your result on `credit` with the reference. After it
# passes, it prints the true incremental conversions beside each method's credit.

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    scores = attribution.check_credit_vs_truth(credit_vs_truth, credit, incremental)
    print(scores.round(3).to_string())
    print()
    print(credit.round(0).assign(true_incremental=incremental).to_string())

# %% [markdown]
# Look at how far above the average user (0) the users each channel reaches sit in purchase
# intent: the simulation's intent is a standard normal score, so 1.0 is one standard deviation
# above the average user.

# %%
intent = pd.Series(truth["mean_intent_reached"]).loc[CHANNELS]
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.barh(CHANNELS[::-1], intent[::-1], color="#4c72b0")
for i, val in enumerate(intent[::-1]):
    ax.text(val + 0.02, i, f"{val:.2f}", va="center", fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Mean purchase intent of the users reached (standard deviations)")
ax.set_title("Who each channel reaches: intent to buy before any ad")
show(fig)

# %% [markdown]
# **Explain.** Compare with your guess: the truth is the `true_incremental` column of the
# checkpoint's output above. Which mechanism earns retargeting its credit: it sits late in
# journeys, it reaches users who would buy anyway, or it has the most touches? Rule out what you
# can first: Shapley's v uses only the set of channels on a journey, not their order or number,
# and Exercise 2 showed that repeated touches in a row do not change where a journey goes next.
# Point to the plot of intent.
#
# <details><summary>Why this solution works</summary>
#
# Mostly **selection** (targeted exposure): a channel that reaches users who were already more
# likely to buy has users who convert more often whether or not it works. Position and touch count
# cannot explain Shapley's credit, and repeated touches in a row cannot explain Markov's.
# Retargeting reaches users who already intend to buy, so journeys that contain it convert more
# often, and every method reads that as credit. The stretch removes the targeting and shows how
# much remains: the efficiency property still shares out conversions that would have happened
# anyway. Module 6's experiment avoided both problems by comparing with a randomized control group.
# </details>
#
# **Is the gap noise?** A **95% bootstrap percentile interval** resamples the journeys with
# replacement, recomputes every method's credit shares, and takes the 2.5th and 97.5th percentiles
# over the resamples. The next cell does this `N_BOOT` times (seed shown). It counts the methods on
# the distinct (path, converted) rows, so each resample is fast, and it calls your
# `removal_effects` and `shapley_values`. Look at whether any true share (the black line) falls
# inside a method's interval.

# %%
boot = attribution.bootstrap_credit(journeys, CHANNELS, N_BOOT, np.random.default_rng(SEED + 5),
                                    removal_fn=removal_effects, shapley_fn=shapley_values)
METHODS = list(credit.columns)  # the methods in `boot`'s last axis: last, first, linear, ...
assert METHODS == list(attribution.METHODS)
boot_shares = boot / boot.sum(axis=1, keepdims=True)  # (resample, channel, method)
share_lo, share_hi = np.quantile(boot_shares, [0.025, 0.975], axis=0)
true_share = (incremental / incremental.sum()).loc[CHANNELS].to_numpy()
point = (credit / credit.sum()).to_numpy()
inside = (share_lo <= true_share[:, None]) & (true_share[:, None] <= share_hi)

fig, ax = plt.subplots(figsize=(8.5, 3.8))
markers = ["o", "s", "^", "D", "v"]
for k, m in enumerate(METHODS):
    xs = np.arange(len(CHANNELS)) + (k - 2) * 0.14
    ax.errorbar(xs, point[:, k], yerr=[point[:, k] - share_lo[:, k], share_hi[:, k] - point[:, k]],
                fmt=markers[k], ms=5, capsize=2, label=m)
ax.hlines(true_share, np.arange(len(CHANNELS)) - 0.4, np.arange(len(CHANNELS)) + 0.4,
          color="black", lw=2.5, label="true incremental share")
ax.set_xticks(np.arange(len(CHANNELS)), CHANNELS)
ax.set_ylabel("Share of credit")
ax.set_title(f"Credit shares with 95% bootstrap percentile intervals ({N_BOOT} resamples)")
ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
show(fig)
print(f"{int(inside.sum())} of {inside.size} true shares lie inside their method's 95% bootstrap"
      " percentile interval.")

# %% [markdown]
# These intervals show sampling noise only. A method that is wrong in the same direction on every
# resample (bias) stays wrong however many journeys you have.

# %% [markdown]
# ## Decision · Move budget or test first?
#
# The attribution report says one paid channel is the cheapest per conversion. Move budget toward
# it, or run a holdout test first?
#
# A **holdout** withholds the channel from a randomly chosen share of the users it would reach.
# The **minimum detectable effect (MDE)** is the smallest true effect a test of this size detects
# with 80% power at α = 0.05, two-sided (Module 6). For a conversion rate p with n₁ users served
# and n₀ held out, σ² = p(1 − p) and
#
# $$ \text{MDE} = (z_{0.975} + z_{0.8}) \sqrt{p(1-p)\left(\frac{1}{n_1} + \frac{1}{n_0}\right)}. $$
#
# The margin per conversion is one order at the retailer's mean spend of \$30 times its 30% gross
# margin, \$9 (repeat purchases ignored). The monthly spend per channel is an assumption of the
# simulated world. Email is sent to a list and cannot scale, so it is left out. A channel's
# **break-even** is its spend ÷ \$9: the incremental conversions a month it must cause to pay back.
#
# **The rule**, fixed before any number: attribution credit cannot separate a channel's effect
# from the intent of the users it reaches. Move budget toward a channel only when an experiment
# (Modules 6 and 7) or a lift-calibrated MMM (Module 9) shows that its cost per incremental
# conversion is below the \$9 margin per conversion. Run a holdout first when it can tell a channel
# that pays back from one that does not: its MDE, scaled to all the users the channel reaches,
# must be below the break-even. If it is not, make the holdout larger or longer before running it.
#
# Decide as you would at work, where there is no `truth`: use only the credit, the spend, the
# margin and the test's MDE. The reveal comes after your sentence.

# %%
ECON = truth["economics"]  # business inputs (spend, margin), not truth about effects
SPEND = pd.Series(ECON["spend"]).loc[CHANNELS]
MARGIN = ECON["margin_per_conversion"]  # $30 order x 30% gross margin = $9
PAID = [c for c in CHANNELS if ECON["scalable"][c]]  # email is list-limited
HOLDOUT_SHARE = 0.20  # withhold the channel from 20% of the users it reaches, for one month
DATA_DRIVEN = ["markov", "shapley"]

k = [METHODS.index(m) for m in DATA_DRIVEN]
cost = credit[DATA_DRIVEN].rdiv(SPEND, axis=0)  # $ per credited conversion
boot_cost = SPEND.to_numpy()[None, :, None] / boot[:, :, k]
cost_lo, cost_hi = np.quantile(boot_cost, [0.025, 0.975], axis=0)
cheapest = {m: cost.loc[PAID, m].idxmin() for m in DATA_DRIVEN}
channel = cheapest["markov"]

reached = journeys["path"].str.contains(channel, regex=False)  # every reached user has a touch
n_reached, p_reached = int(reached.sum()), journeys.loc[reached, "converted"].mean()
n0 = round(HOLDOUT_SHARE * n_reached)
n1 = n_reached - n0
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
mde = z * np.sqrt(p_reached * (1 - p_reached) * (1 / n1 + 1 / n0))
break_even = SPEND[channel] / MARGIN
at_risk = HOLDOUT_SHARE * credit.loc[channel, "markov"] * MARGIN
test_first = mde * n_reached < break_even

print("Number")
print("  Cost per credited conversion (spend / credit), 95% bootstrap percentile interval:")
for i, c in enumerate(CHANNELS):
    if c in PAID:
        print(f"    {c:<12} spend ${SPEND[c]:>5,.0f}   "
              + "   ".join(f"{m} ${cost.loc[c, m]:.2f} (${cost_lo[i, j]:.2f} to ${cost_hi[i, j]:.2f})"
                           for j, m in enumerate(DATA_DRIVEN)))
print(f"  Cheapest by attribution: {cheapest['markov']} (Markov), {cheapest['shapley']} (Shapley)."
      f" Margin per conversion ${MARGIN:.0f}: by attribution, every paid channel pays back.")
print("  The bootstrap intervals cover sampling noise in the credit, not the bias Exercise 5 found.")
print(f"  Holdout: withhold {channel} from {HOLDOUT_SHARE:.0%} of the {n_reached:,} users it reaches"
      f" ({n1:,} served, {n0:,} held out); their conversion rate is {p_reached:.2%}.")
print(f"  MDE: {100 * mde:.2f} points = about {mde * n1:.0f} conversions among served users ="
      f" about {mde * n_reached:.0f} a month scaled to all {n_reached:,} users reached.")
print(f"  Break-even: ${SPEND[channel]:,.0f} / ${MARGIN:.0f} = {break_even:.0f} incremental"
      " conversions a month.")
print(f"  Margin at risk if the credit were real: {HOLDOUT_SHARE:.0%} x"
      f" {credit.loc[channel, 'markov']:.0f} credited x ${MARGIN:.0f} = ${at_risk:,.0f}.")
print("Rule")
print("  Move budget only on a measured cost per incremental conversion below the margin; test"
      " first when the holdout's MDE, scaled to all users reached, is below the break-even.")
print("Recommendation")
if test_first:
    print(f"  Test {channel} first: its holdout's MDE scaled to all users reached, about"
          f" {mde * n_reached:.0f} conversions a month, is below its break-even of"
          f" {break_even:.0f}, so the test can tell a channel that pays back from one that does"
          " not.")
else:
    print(f"  Make the holdout larger or longer before running it: its MDE scaled to all users"
          f" reached, about {mde * n_reached:.0f} conversions a month, is not below the"
          f" break-even of {break_even:.0f}.")
print(f"  Margin at risk during the test: about ${at_risk:,.0f}, if the credit were real.")
print(f"  What would change it: a holdout showing more than {break_even:.0f} incremental"
      f" conversions a month (a cost per incremental conversion below ${MARGIN:.0f}) would justify"
      " moving budget toward it.")
print(f"  Mode: {'QUICK run: treat the intervals as rough' if QUICK else 'FULL run'} (QUICK changes"
      " only the bootstrap intervals; the credit, the MDE and this recommendation do not change).")

# %% [markdown]
# **Decide.** Write one sentence: test or move, which channel, the holdout share and its MDE, the
# margin at risk, and what result would justify moving budget.
#
# ```text
# Your sentence: Before moving budget to ____ (cheapest by Markov credit, $____ per credited
# conversion), withhold it from ____% of the users it reaches for one month. The test can detect
# ____ points (about ____ conversions a month, scaled to all the users it reaches) and puts about
# $____ of margin at risk. Move budget only if it shows more than ____ incremental conversions a
# month (a cost per incremental conversion below $9).
# ```
#
# Then run the next cell, which reveals the truth the data were generated from.

# %%
true_cost = pd.Series(truth["true_cost_per_incremental_conversion"], dtype=float).loc[CHANNELS]
reveal = pd.DataFrame({
    "credited (Markov)": credit["markov"].round(0),
    "credited (Shapley)": credit["shapley"].round(0),
    "true incremental": incremental.loc[CHANNELS],
    "$ per credited (Markov)": cost["markov"].round(2),
    "true $ per incremental": true_cost.round(2),
    "truly pays back": true_cost < MARGIN,
})
print(reveal.to_string())
print(f"Largest gap between credit and truth: {scores.loc['markov', 'largest_gap_channel']}"
      f" (Markov), {scores.loc['shapley', 'largest_gap_channel']} (Shapley).")
if incremental[channel] == 0:
    print(f"Moving budget to {channel} would have bought nothing: it caused no conversions.")
else:
    print(f"Each extra ${true_cost[channel]:.2f} on {channel} buys about one more conversion.")

# %% [markdown]
# **About this simulation.** Retargeting's zero effect is an assumption of this simulated world,
# chosen to make the mechanism visible; it is not a claim about retargeting in general. The
# simulation also simplifies: whether a user buys depends on their intent and on which channels
# reached them, not on the order or the number of their touches. In your own data, only an
# experiment (Modules 6 and 7) or a lift-calibrated MMM (Module 9) tells you what a channel causes.
#
# ## Stretch (optional) · Random exposure, a real holdout, and a counting game
#
# 1. **Same reach, random exposure.** `mta_journeys(SEED, random_exposure=True)` simulates the
#    same users with every random draw unchanged, but each channel now reaches users at random
#    (its reach share stays the same, and reach no longer depends on intent). Rerun the five
#    methods and `credit_vs_truth` against that world's own truth. What share error remains, and
#    which property explains it? Why might Shapley gain more than Markov here? A likely reason, to
#    check rather than assert: its v compares the conversion rates of channel sets, and random
#    exposure makes those sets comparable.
# 2. **Run the holdout.** `mta_journeys(SEED, holdout={"retargeting": 0.2})` withholds retargeting
#    from 20% of the users it would reach. With its `holdout` table, estimate retargeting's
#    incremental conversions among served users with a 95% confidence interval, as in Module 6's
#    Exercise 6, and compare with its Markov credit and its truth.
# 3. **A count-based value function.** With v(S) = conversions of the journeys that use only
#    channels in S, show that the Shapley values equal "split each conversion equally among the
#    distinct channels on its path" (the game is a sum of unanimity games).

# %% tags=["solution"]
# @title Stretch solutions — try them yourself first { display-mode: "form" }
from mktstats.synth.mta import mta_journeys


def all_credit(j):
    """The five methods' credited conversions on journeys j, with your functions."""
    out = pd.DataFrame({r: heuristic_credit(j, r, CHANNELS) for r in ("last", "first", "linear")})
    out["markov"] = markov_credit(removal_effects(transition_matrix(j, CHANNELS), CHANNELS),
                                  j["converted"].sum())
    out["shapley"] = attribution.shapley_credit(attribution.coalition_table(j), CHANNELS,
                                                shapley_values)[0]
    return out


# 1. Same reach, random exposure
flat = mta_journeys(SEED, random_exposure=True)
flat_credit = all_credit(flat.journeys)
flat_scores = credit_vs_truth(flat_credit, pd.Series(flat.truth["incremental_conversions"]))
print("Share error D, targeted vs random exposure:")
print(pd.concat({"targeted": scores["tvd"], "random exposure": flat_scores["tvd"]}, axis=1)
      .round(3).to_string())
print("Retargeting's credit share with random exposure:",
      (flat_credit.loc["retargeting"] / flat_credit.sum()).round(3).to_dict())

# 2. Run the holdout (Module 6, Exercise 6)
held = mta_journeys(SEED, holdout={"retargeting": HOLDOUT_SHARE})
ho = held.holdout
served = ho.loc[ho["holdout"] == 0, "converted"]
withheld = ho.loc[ho["holdout"] == 1, "converted"]
p1, p0, n1_h, n0_h = served.mean(), withheld.mean(), len(served), len(withheld)
se = np.sqrt(p1 * (1 - p1) / n1_h + p0 * (1 - p0) / n0_h)
z975 = stats.norm.ppf(0.975)
print(f"Holdout of {HOLDOUT_SHARE:.0%}: {n1_h:,} served convert at {p1:.2%}, {n0_h:,} held out at"
      f" {p0:.2%}. Incremental conversions among served users: {(p1 - p0) * n1_h:.0f}"
      f" (95% CI {(p1 - p0 - z975 * se) * n1_h:.0f} to {(p1 - p0 + z975 * se) * n1_h:.0f});"
      f" Markov credit {credit.loc['retargeting', 'markov']:.0f};"
      f" truth {held.truth['holdout']['incremental_conversions_served']}.")

# 3. A count-based value function
conv_by_set = dict(zip(coalitions["channels"], coalitions["conversions"], strict=True))


def v_count(S):
    """Conversions of the journeys whose channels all lie in S."""
    return sum(n for T, n in conv_by_set.items() if T <= S)


equal_split = pd.Series(0.0, index=CHANNELS)
for T, n in conv_by_set.items():
    for c in T:
        equal_split[c] += n / len(T)
print(pd.DataFrame({"Shapley, count-based v": pd.Series(shapley_values(CHANNELS, v_count)),
                    "equal split over distinct channels": equal_split}).round(6).to_string())
