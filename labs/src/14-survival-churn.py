# ---
# jupyter:
#   mktstats:
#     api_checks: |
#       Technical Expert, 2026-10-10. Checked against the installed source of lifelines 0.30.3
#       (.venv/lib/python3.13/site-packages/lifelines):
#       - fitters/kaplan_meier_fitter.py: KaplanMeierFitter(alpha=0.05); fit(durations,
#         event_observed, ...); survival_function_ (column KM_estimate, index 0 and every
#         distinct duration, censoring times included) = exp(cumsum(log(1 - d / n)));
#         confidence_interval_ (KM_estimate_lower_0.95, KM_estimate_upper_0.95) from _bounds,
#         the exponential Greenwood formula: pointwise intervals.
#       - fitters/__init__.py: UnivariateFitter.predict(times) = estimate.asof(times), so the
#         last value is carried flat past the data without a warning.
#       - utils/__init__.py: restricted_mean_survival_time(kmf, t) integrates the Kaplan-Meier
#         step function exactly up to t (predict(t) appended); return_variance=True gives the
#         variance of min(T, t), not a standard error of the RMST.
#       - fitters/coxph_fitter.py: CoxPHFitter(alpha=...) through BaseFitter; Breslow baseline,
#         Efron ties; fit(df, duration_col, event_col) treats every other column of df as a
#         covariate; confidence_intervals_ on the coefficient scale with columns
#         "95% lower-bound", "95% upper-bound" (the level changes with alpha);
#         fitters/mixins.py: hazard_ratios_ = exp(params_).
#       - statistics.py: proportional_hazard_test(cph, df, time_transform="rank").summary
#         (test_statistic, p, -log2(p)), from scaled Schoenfeld residuals.
# ---

# %% [markdown]
# # Part A · Censoring and the Kaplan-Meier curve
#
# In January 2024 the retailer launched a monthly replenishment subscription in three plans
# (basic, standard, premium). Each subscriber starts with a 14-day free trial, during which they
# complete onboarding or not (`onboarded` = 1 or 0); the **first billing date** ends the trial,
# and a subscriber's **tenure** is counted in months from that date. Some subscribers joined
# through an introductory offer that discounts their first three paid months
# (`intro_offer` = 1). Sign-ups ran to the end of 2025 and grew over time. The data stop on
# 2026-06-30, when almost half of the subscribers are still active.
#
# A tenure is **right-censored** when the subscriber was still active when the data stopped: we
# know the subscriber stayed at least this long, not when they will leave. The **survival
# (retention) curve** S(t) is the share of subscribers still subscribed t months after their
# first billing date.
#
# The data are synthetic: 8,000 subscribers made by a generator whose churn process is known
# (`mktstats.synth.subscriptions`, seed 2034), so every estimate can be checked against the
# truth. The file has two truth columns that real data never has, `true_tenure_months` and
# `true_rmst24_gain`; the next cell leaves them out of `subs` (they stay in `raw`), and only the
# stretch reads a truth column.
#
# **QUICK.** The data are the same with QUICK on or off. QUICK changes only the number of
# bootstrap resamples in Exercise 5 (100 instead of 200), so every number is the same in both
# modes except Exercise 5's interval and the Decision built on it. Interval ends can move by a
# few hundredths, and the share of resamples in which a call pays by several points, so a
# decision near its break-even can flip. That is a lesson about resampling, not a bug.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import restricted_mean_survival_time
from scipy.special import betaln

from mktstats import data, survival
from mktstats.synth.subscriptions import subscriptions, true_survival

OBS_END = "2026-06-30"       # the data stop here: a subscriber still active then is censored
DAYS_PER_MONTH = 30.4375     # PyMC-Marketing's month, as in Module 4
ALPHA = 0.05                 # every interval in this lab is a 95% interval (1 - ALPHA)
LANDMARK = 3.0               # months: the intro offer discounts the first three paid months
HORIZON = 24                 # months: Exercise 2 shows why the decision does not look further
MARGIN = 12.0                # dollars of margin per subscriber-month (a scenario assumption)
CALL_COST = 15.0             # dollars per onboarding call (a scenario assumption)
Q_GRID = (0.20, 0.35, 0.50)  # share of called subscribers who then onboard: unknown, so a range
BOOT_SEED = 14               # the bootstrap's seed (Exercise 5)
B = 100 if QUICK else 200    # bootstrap resamples (Exercise 5); QUICK changes nothing else
COVARIATES = ["plan_standard", "plan_premium", "onboarded", "intro_offer"]  # Exercises 3 and 4

# %% [markdown]
# ### Run · Tenure, churn and two dashboard numbers
#
# The next cell reads the subscribers and adds `tenure_months` (from the first billing date to
# the cancellation, or to 2026-06-30 for a subscriber still active, in months of 30.4375 days)
# and `churned` (1 if the cancellation was observed, 0 if the tenure is censored). Then it prints
# two 12-month retention rates a dashboard might show:
#
# - **A**: the share of all subscribers who did not cancel within 12 months of starting;
# - **B**: the same share among subscribers whose 12-month status is known: they cancelled
#   within 12 months, or were observed for more than 12.

# %%
truth = json.loads(data.repo_file("synthetic/truth_subscriptions.json").read_text())
raw = pd.read_csv(data.repo_file("synthetic/subscriptions_subscribers.csv"),
                  parse_dates=["start_date", "cancel_date"])


def add_tenure(subscribers, obs_end=OBS_END):
    """A copy with tenure_months and churned, computed from the dates."""
    end = pd.Timestamp(obs_end)
    days = (subscribers["cancel_date"].fillna(end) - subscribers["start_date"]).dt.days
    return subscribers.assign(tenure_months=days / DAYS_PER_MONTH,
                              churned=subscribers["cancel_date"].notna().astype(int))


subs = add_tenure(raw.drop(columns=["true_tenure_months", "true_rmst24_gain"]))
cancelled_12 = (subs["churned"] == 1) & (subs["tenure_months"] <= 12)
beyond_12 = subs["tenure_months"] > 12
dash_a = 1 - cancelled_12.mean()
dash_b = beyond_12.sum() / (cancelled_12 | beyond_12).sum()

print(f"{len(subs):,} subscribers started between {subs['start_date'].min():%Y-%m-%d} and"
      f" {subs['start_date'].max():%Y-%m-%d}; {subs['churned'].sum():,} cancellations were"
      f" observed ({subs['churned'].mean():.1%}); the rest are censored on {OBS_END}.")
print(f"Longest follow-up: {subs['tenure_months'].max():.1f} months.")
print(f"Dashboard A, 12-month retention of all subscribers:        {dash_a:.1%}")
print(f"Dashboard B, 12-month retention where the status is known: {dash_b:.1%}"
      f" ({(cancelled_12 | beyond_12).sum():,} of {len(subs):,} subscribers)")
subs.head()

# %% [markdown]
# ## Exercise 1 · Write the Kaplan-Meier estimator (10 minutes)
#
# A subscriber is **at risk at t** if they are still subscribed and still observed just before
# t; a subscriber censored at t counts as at risk at t. The **Kaplan-Meier estimator** Ŝ(t) is
# the product, over the times up to t with at least one cancellation, of the share of those at
# risk who did not cancel:
#
# Ŝ(t) = ∏<sub>t<sub>j</sub> ≤ t</sub> (1 − d<sub>j</sub> / n<sub>j</sub>),
#
# where t<sub>j</sub> are the distinct cancellation times, d<sub>j</sub> the number of
# cancellations at t<sub>j</sub>, and n<sub>j</sub> the number at risk just before
# t<sub>j</sub> (everyone whose observed tenure is t<sub>j</sub> or longer, censored or not).
#
# **Predict.** Neither dashboard number is the true 12-month retention. Where is the truth:
# above A, between A and B but nearer A, between them but nearer B, or below B? Write your guess
# in the cell below.
#
# **Task.** Write `kaplan_meier(duration, event)` returning a `pd.Series` named `"survival"`:
# its index is the distinct times with at least one cancellation, ascending, and its values are
# Ŝ(t) just after each. `duration` and `event` are array-likes of the same length (lists, NumPy
# arrays or Series); `event` is 1 for a cancellation and 0 for censored. Tenures are whole days,
# so several subscribers share a time: group them.
#
# The checkpoint runs your function on the toy data below, on a second toy with two
# cancellations at the same time, and on `subs`, where it must match lifelines'
# `KaplanMeierFitter` (lifelines is the survival-analysis library this lab uses).
#
# A worked start on the checkpoint's toy data (durations 2, 3, 3, 5, 6, 6, 8, 9; events 1, 1, 0,
# 1, 0, 1, 1, 0): at t = 2, 8 are at risk and 1 cancels, so Ŝ = 7/8 = 0.875; at t = 3, 7 are at
# risk (the subscriber censored at 3 still counts) and 1 cancels, so Ŝ = 0.875 × 6/7 = 0.75.

# %% [markdown]
# *Your guess (Exercise 1):*

# %% tags=["exercise"]
def kaplan_meier(duration, event):
    # TODO 1: put duration and event in one DataFrame; group by duration to get, at each time,
    # the cancellations d_j (the sum of event) and the number of rows; the number at risk n_j
    # is the number of rows at that time or later (a reversed cumulative sum); take the
    # cumulative product of 1 - d_j / n_j; keep the times with d_j > 0.
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def kaplan_meier(duration, event):
    df = pd.DataFrame({"t": np.asarray(duration, dtype=float),
                       "d": np.asarray(event, dtype=float)})
    by_time = df.groupby("t")["d"].agg(["sum", "size"])  # d_j and rows at each time
    at_risk = by_time["size"][::-1].cumsum()[::-1]  # n_j: duration >= t_j, censored included
    s = (1 - by_time["sum"] / at_risk).cumprod()
    return s[by_time["sum"] > 0].rename("survival")


# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    recover = " To move on, run workshop.use_reference(1)."
    # lifelines fits the same curve (it draws no progress bar, so there is no progressbar=False).
    kmf_subs = KaplanMeierFitter(alpha=ALPHA).fit(subs["tenure_months"], subs["churned"])

    def as_dict(km, what):
        assert isinstance(km, pd.Series), (
            f"Checkpoint 1: on {what} your function returned a {type(km).__name__}. Return a"
            " pandas Series indexed by time, with Ŝ(t) as its values." + recover
        )
        return {float(t): float(s) for t, s in km.items()}

    toy = as_dict(kaplan_meier([2, 3, 3, 5, 6, 6, 8, 9], [1, 1, 0, 1, 0, 1, 1, 0]), "the toy")
    tie = as_dict(kaplan_meier([1, 1, 2], [1, 1, 0]), "the tie toy")
    want = {2.0: 0.875, 3.0: 0.75, 5.0: 0.6, 6.0: 0.45, 8.0: 0.225}
    values = list(toy.values())
    if 0.0 in toy or 9.0 in toy or 2.0 in tie:
        cause = ("Keep only times with at least one cancellation; a censoring-only time lowers"
                 " n_j for later times but adds no factor.")
    elif list(toy) != sorted(toy):
        cause = "Sort the index ascending (`.sort_index()`)."
    elif any(b > a + 1e-12 for a, b in zip(values, values[1:], strict=False)):
        cause = "Your values rise: you returned the share cancelled, 1 − Ŝ(t). Return Ŝ(t)."
    elif abs(toy.get(3.0, np.nan) - 0.875 * 5 / 6) < 1e-3:
        cause = ("A subscriber censored at t is still at risk at t. n_j counts everyone whose"
                 " duration is t_j or longer, censored or not; they leave the risk set after"
                 " t_j.")
    elif abs(toy.get(2.0, np.nan) - 6 / 7) < 1e-3:
        cause = ("Subscribers who cancel at t_j are at risk at t_j: count duration ≥ t_j, not"
                 " duration > t_j.")
    elif abs(tie.get(1.0, np.nan) - 2 / 3) < 1e-3:
        cause = ("Two cancellations at t = 1 are d_j = 2 in one factor, 1 − 2/3; you counted one"
                 " per distinct time. Sum `event` at each time.")
    elif abs(tie.get(1.0, np.nan) - 4 / 9) < 1e-3:
        cause = ("You applied 1 − 1/3 twice with the same n_j; use one factor per time,"
                 " 1 − d_j / n_j.")
    elif abs(toy.get(2.0, np.nan) - 1) < 1e-12:
        cause = ("Your values are Ŝ just before each time: Ŝ(t_j) includes the factor of t_j"
                 " itself, so Ŝ(2) = 1 − 1/8 = 0.875. Do not shift the cumulative product.")
    else:
        cause = "Compare your n_j and d_j with the worked start above, one time at a time."
    shown = {t: round(s, 3) for t, s in toy.items()}
    assert list(toy) == list(want) and all(abs(toy[t] - want[t]) < 1e-12 for t in want), (
        f"Checkpoint 1: on the toy data your function returned {shown}; expected {want}."
        f" {cause}" + recover
    )
    assert list(tie) == [1.0] and abs(tie[1.0] - 1 / 3) < 1e-12, (
        f"Checkpoint 1: on the tie toy (durations 1, 1, 2; events 1, 1, 0) your function"
        f" returned { {t: round(s, 3) for t, s in tie.items()} }; expected {{1: 0.333}}: 3 at"
        f" risk at t = 1, 2 cancel, 1 − 2/3. {cause}" + recover
    )
    km_mine = kaplan_meier(subs["tenure_months"], subs["churned"])
    event_times = np.sort(subs.loc[subs["churned"] == 1, "tenure_months"].unique())
    mine_times = np.asarray(km_mine.index, dtype=float)
    # The toys' durations are whole numbers in ascending order; the tenures in subs are not.
    no_rounding = ("Tenures are fractions of a month (days ÷ 30.4375), so several of them round"
                   " to the same number: your function must use the durations as they are,"
                   " without rounding them or casting them (no round(), no astype(int)).")
    if len(mine_times) < len(event_times):
        whole = np.all(mine_times == np.round(mine_times))
        data_cause = ("Your times are whole numbers. " if whole else "") + no_rounding
    elif len(mine_times) == len(event_times) and np.array_equal(np.sort(mine_times),
                                                                event_times):
        data_cause = ("The times are all there but not in ascending order: sort by time"
                      " before the cumulative product.")
    else:
        data_cause = ("Return one row per distinct cancellation time, ascending, keeping each"
                      " duration with its own event. " + no_rounding)
    assert len(km_mine) == len(event_times) and np.array_equal(mine_times, event_times), (
        f"Checkpoint 1: on `subs` your curve has {len(km_mine)} times; there are"
        f" {len(event_times)} distinct cancellation times. {data_cause}" + recover
    )
    gap = np.abs(km_mine.to_numpy() - kmf_subs.survival_function_["KM_estimate"]
                 .loc[event_times].to_numpy())
    assert gap.max() <= 1e-9, (
        f"Checkpoint 1: your curve has the right times but differs from lifelines' by"
        f" {gap.max():.2g} at t = {event_times[int(np.argmax(gap))]:.3f} months. Take the"
        " cumulative product over the times in ascending order: sort before .cumprod(), not"
        " after (the toys' durations were already in order; the tenures in subs are not)."
        + recover
    )
print(f"Your curve: {len(km_mine):,} cancellation times among"
      f" {subs['tenure_months'].nunique():,} distinct tenures; largest gap to lifelines"
      f" {gap.max():.1g}.")

# %% [markdown]
# ### Run · Your curve against the truth and the dashboards
#
# Your curve equals `kmf_subs`, the curve lifelines fitted in the checkpoint, so the next cell
# uses `kmf_subs` and its **95% pointwise interval**: an interval for S(t) at one t at a time, not
# a band that covers the whole curve with 95% probability. The check compares the data with the truth, not your code. The
# cell also prints how many subscribers were censored before month 12, and when the earliest of
# them started.

# %%
def plot_km(ax, kmf, label, linestyle="-"):
    """A Kaplan-Meier curve as a step line with its 95% pointwise band."""
    curve = kmf.survival_function_.iloc[:, 0]
    band = kmf.confidence_interval_
    (line,) = ax.step(curve.index, curve.to_numpy(), where="post", linestyle=linestyle,
                      label=label)
    ax.fill_between(band.index, band.iloc[:, 0], band.iloc[:, 1], step="post", alpha=0.2,
                    color=line.get_color())
    return line


times = [6, 12, 18, 24]
band = kmf_subs.confidence_interval_
km_table = pd.DataFrame(
    {
        "Kaplan-Meier": [float(kmf_subs.predict(t)) for t in times],
        "95% lower": [float(band.asof(t).iloc[0]) for t in times],
        "95% upper": [float(band.asof(t).iloc[1]) for t in times],
        "truth": [truth["survival"]["overall"][str(t)] for t in times],
    },
    index=pd.Index(times, name="month"),
)
lo12, hi12 = km_table.loc[12, ["95% lower", "95% upper"]]

with workshop.checkpoint(label="km-vs-truth"):
    data_check = ("This compares the data with the truth, not your code; if it fails, tell the"
                  " instructor.")
    survival.k_of_K_ci_contains(
        dict(zip(times, km_table["truth"], strict=True)),
        {t: (km_table.loc[t, "95% lower"], km_table.loc[t, "95% upper"]) for t in times},
        k=3, level=1 - ALPHA, kind="pointwise confidence interval",
        name="S(t) at 6, 12, 18 and 24 months", hint=data_check, label="Check km-vs-truth",
    )
    for name, value in (("A", dash_a), ("B", dash_b)):
        assert not lo12 <= value <= hi12, (
            f"Check km-vs-truth: dashboard {name} ({value:.3f}) lies inside the 95% pointwise interval at 12 months"
            f" [{lo12:.3f}, {hi12:.3f}]. {data_check}"
        )

censored_early = subs[(subs["churned"] == 0) & (subs["tenure_months"] <= 12)]
print(km_table.round(3).to_string())
print(f"At 12 months: dashboard A {dash_a:.3f}, dashboard B {dash_b:.3f}.")
print(f"{len(censored_early):,} subscribers were censored before month 12; the earliest of them"
      f" started on {censored_early['start_date'].min():%Y-%m-%d}.")

# %% [markdown]
# Look at where A and B sit against the curve at 12 months. The band is too narrow to see at
# this scale; the table above gives its ends.

# %%
grid = np.linspace(0, 30, 301)
fig, ax = plt.subplots(figsize=(7, 4))
plot_km(ax, kmf_subs, "Kaplan-Meier, 95% pointwise band")
ax.plot(grid, true_survival(grid), linestyle="--", color="black", label="true S(t)")
ax.plot([12], [dash_a], marker="^", markersize=9, linestyle="none", label=f"A: {dash_a:.1%}")
ax.plot([12], [dash_b], marker="v", markersize=9, linestyle="none", label=f"B: {dash_b:.1%}")
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Months since the first billing date")
ax.set_ylabel("Share still subscribed, S(t)")
ax.set_title("Retention of 8,000 subscribers")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Explain.** Compare with your guess. Which subscribers does A count as "retained" who may
# still cancel before month 12? Which does B drop, and why does that make B too low? Point to the
# earliest start date of the subscribers censored before month 12, printed above.
#
# <details><summary>Why this solution works</summary>
#
# A treats recent starters as retained although they have not had 12 months to cancel; B keeps
# the recent starters who already cancelled but drops those who did not, so it over-counts
# churn. Kaplan-Meier uses each subscriber for exactly as long as they were observed: each
# month's churn rate comes only from subscribers still at risk then. It assumes that censoring
# is unrelated to churn risk; here censoring comes only from the observation end, and the plan
# mix does not change with the start date. It would fail if, say, the newest cohorts came from
# a promotion with different churn.
# </details>

# %% [markdown]
# ### Run · The Fader-Hardie curves, years 0-7
#
# Module 4's real data: of 1,000 customers in each of two segments, Regular and High End, how
# many were still subscribed at each annual renewal, years 0 to 12 (Fader & Hardie 2007). Fader
# and Hardie fit years 0 to 7 and kept years 8 to 12 to test the projection. In Module 4's
# encoding, a customer lost in year t has tenure t and event 1, and one still active after year
# 7 has tenure 7 and event 0. `kmf_regular` is Kaplan-Meier on Regular's years 0 to 7.
#
# For an annual subscription paid at each renewal, the **expected paid years in the first H
# years** are S(0) + S(1) + … + S(H − 1): how many of the first H years an average customer
# pays for, the area under the survival curve. Times the annual margin, it is the undiscounted
# customer lifetime value (CLV) of a contract like this. `sbg_survival` is the survival curve of
# Module 4's shifted beta-geometric (sBG) model, S(t) = B(α, β + t) / B(α, β), where B is the
# beta function, with Fader and Hardie's estimates for Regular, α = 0.704 and β = 1.182 (Module 4
# reproduced them). The cell builds four NumPy arrays for Exercise 2.

# %%
retention = data.load_sbg_retention()  # year, regular, highend: survivors of 1,000
alive = retention.loc[retention["year"] <= 7, "regular"].to_numpy()
lost = alive[:-1] - alive[1:]  # customers lost during years 1 to 7
regular_tenure = np.concatenate([np.repeat(np.arange(1, 8), lost), np.full(alive[-1], 7)])
regular_event = np.concatenate([np.ones(lost.sum()), np.zeros(alive[-1])])
# lifelines draws no progress bar, so there is no progressbar=False to pass.
kmf_regular = KaplanMeierFitter(alpha=ALPHA).fit(regular_tenure, regular_event)


def sbg_survival(alpha, beta, t):
    """Module 4's sBG survival curve, S(t) = B(alpha, beta + t) / B(alpha, beta)."""
    return np.exp(betaln(alpha, beta + np.asarray(t)) - betaln(alpha, beta))


SBG_REGULAR = {"alpha": 0.704, "beta": 1.182}  # Fader & Hardie (2007), Regular segment
km_years = kmf_regular.survival_function_["KM_estimate"].to_numpy()  # years 0-7: 8 values
km_flat = kmf_regular.predict(np.arange(13)).to_numpy()  # years 0-12: flat after year 7
sbg_years = sbg_survival(SBG_REGULAR["alpha"], SBG_REGULAR["beta"], np.arange(13))
actual_years = retention["regular"].to_numpy() / 1000  # Table 1, years 0-12
pd.DataFrame({"survivors of 1,000": alive, "Kaplan-Meier S(t)": km_years},
             index=pd.Index(range(8), name="year")).round(3)

# %% [markdown]
# ## Exercise 2 · Where Kaplan-Meier stops: expected years on the Fader-Hardie data (7 minutes)
#
# **Predict.** From the actual counts, a Regular customer pays for 4.41 of the first 12 years on
# average. Kaplan-Meier knows years 0 to 7 and ends at 241 of 1,000 at year 7. **Carried flat**
# means using the last known S(t) for every later t, which assumes nobody leaves after the data
# end. Carry Kaplan-Meier flat to year 12: will its expected paid years in the first 12 be above
# or below 4.41, and by under 1%, 1 to 5%, or over 5%? Write your guess in the cell below.
#
# **Task.** Write `expected_periods(survival, horizon)` returning a float: `survival` is a 1-D
# list or array S(0), S(1), … with S(0) = 1; return the sum of `survival[0]` to
# `survival[horizon − 1]`. When the curve has fewer than `horizon` values, raise `ValueError`
# with a sentence saying how far the curve is known.

# %% [markdown]
# *Your guess (Exercise 2):*

# %% tags=["exercise"]
def expected_periods(survival, horizon):
    # TODO 2: check the length first (raise ValueError if the curve is too short), then sum
    # the first `horizon` values, starting with S(0)
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def expected_periods(survival, horizon):
    s = np.asarray(survival, dtype=float)
    if len(s) < horizon:
        raise ValueError(f"the curve is known for periods 0 to {len(s) - 1} only; a horizon of"
                         f" {horizon} needs S(0) to S({horizon - 1})")
    return float(s[:horizon].sum())


# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    recover = " To move on, run workshop.use_reference(2)."

    def refused(curve, horizon, what):
        """None if expected_periods raised ValueError, else the value it returned."""
        try:
            return expected_periods(curve, horizon)
        except ValueError:
            return None
        except AssertionError as exc:
            raise AssertionError(
                f"Checkpoint 2: {what} raised AssertionError, not ValueError: an assert always"
                " raises AssertionError. Check the length with an if and raise ValueError"
                " yourself: if len(survival) < horizon: raise ValueError(\"...\")." + recover
            ) from exc
        except (IndexError, KeyError) as exc:
            raise AssertionError(
                f"Checkpoint 2: {what} raised {type(exc).__name__}, not ValueError. Check the"
                " length and raise ValueError yourself, before summing." + recover
            ) from exc

    def summed(curve, horizon, what):
        """The value of expected_periods on a curve long enough for the horizon."""
        try:
            return expected_periods(curve, horizon)
        except (ValueError, AssertionError) as exc:
            raise AssertionError(
                f"Checkpoint 2: {what} raised {type(exc).__name__}, but its {len(curve)} values,"
                f" S(0) to S({len(curve) - 1}), are enough for a horizon of {horizon}: raise"
                " only when len(survival) < horizon."
                + (" And raise ValueError, not an assert." if isinstance(exc, AssertionError)
                   else "") + recover
            ) from exc

    got3 = summed([1, 0.5, 0.25], 3, "expected_periods([1, 0.5, 0.25], 3)")
    assert abs(got3 - 0.75) > 1e-9, (
        "Checkpoint 2: expected_periods([1, 0.5, 0.25], 3) is 0.75; expected 1.75. You started"
        " at survival[1]. Everyone pays the first period, so S(0) = 1 counts: sum survival[0]"
        " to survival[horizon − 1]." + recover
    )
    assert abs(got3 - 1.75) < 1e-12, (
        f"Checkpoint 2: expected_periods([1, 0.5, 0.25], 3) is {got3}; expected"
        " 1 + 0.5 + 0.25 = 1.75." + recover
    )
    got2 = summed([1, 0.5, 0.25], 2, "expected_periods([1, 0.5, 0.25], 2)")
    assert abs(got2 - 1.5) < 1e-12, (
        f"Checkpoint 2: expected_periods([1, 0.5, 0.25], 2) is {got2}; expected 1 + 0.5 = 1.5."
        + (" The function ignores `horizon`; sum only the first `horizon` values."
           if abs(got2 - 1.75) < 1e-12 else "") + recover
    )
    short = refused([1, 0.5, 0.25], 4, "expected_periods([1, 0.5, 0.25], 4)")
    assert short is None, (
        f"Checkpoint 2: expected_periods([1, 0.5, 0.25], 4) returned {short} without an error."
        " Slicing past the end of a list or array returns what is there without complaint"
        " (s[:4] of three values is three values); that is how a curve gets used past its"
        " data. Check len(survival) < horizon and raise ValueError." + recover
    )
    km_short = refused(km_years, 12, "Kaplan-Meier's own curve with horizon 12")
    assert km_short is None, (
        f"Checkpoint 2: Kaplan-Meier's curve (years 0-7, {len(km_years)} values) with horizon 12"
        f" returned {km_short}; it should raise ValueError." + recover
    )
    actual_12 = summed(actual_years, 12, "the actual Regular curve with horizon 12")
    assert abs(actual_12 - retention["regular"].iloc[:12].sum() / 1000) < 1e-12, (
        f"Checkpoint 2: on the actual Regular curve your total is {actual_12:.4f}; the sum of"
        " S(0) to S(11) of the actual curve is 4.406. Sum 12 values starting at year 0."
        + recover
    )
    sbg_12 = summed(sbg_years, 12, "the sBG curve with horizon 12")
    flat_12 = summed(km_flat, 12, "Kaplan-Meier carried flat with horizon 12")
    # Data checks once the function is right: published sBG estimates, and the flat tail.
    assert abs(sbg_12 / actual_12 - 1) < 0.01, (
        f"Checkpoint 2: the sBG gives {sbg_12:.3f} expected paid years against {actual_12:.3f}"
        " actual (allowed 1%). This compares the data with the published model; tell the"
        " instructor."
    )
    assert flat_12 / actual_12 - 1 > 0.03, (
        f"Checkpoint 2: Kaplan-Meier carried flat gives {flat_12:.3f} against {actual_12:.3f}"
        " actual; it should overstate it by more than 3%. Tell the instructor."
    )
print(f"Expected paid years in the first 12: actual {actual_12:.3f}.")

# %% [markdown]
# ### Run · Years 8-12
#
# The table gives the survivors of 1,000 in years 8 to 12 by each method, and the expected
# paid years in the first 12 with their difference from the actual.

# %%
years_8_12 = pd.DataFrame(
    {"Kaplan-Meier, flat": 1000 * km_flat[8:], "sBG": 1000 * sbg_years[8:],
     "actual": 1000 * actual_years[8:]},
    index=pd.Index(range(8, 13), name="year"),
)
print(years_8_12.round(1).to_string())
print()
for name, total in (("Kaplan-Meier, flat", flat_12), ("sBG", sbg_12), ("actual", actual_12)):
    print(f"Expected paid years in the first 12, {name:<18}: {total:.3f}"
          f" ({total / actual_12 - 1:+.1%} against the actual)")

# %% [markdown]
# Look at what each curve does after year 7, the last year either method saw.

# %%
years = np.arange(13)
fig, ax = plt.subplots(figsize=(7, 4))
(km_line,) = ax.step(years[:8], 1000 * km_flat[:8], where="post",
                     label="Kaplan-Meier, years 0-7")
ax.step(years[7:], 1000 * km_flat[7:], where="post", linestyle="--", color=km_line.get_color(),
        label="Kaplan-Meier carried flat")
ax.plot(years, 1000 * sbg_years, marker="o", linestyle="-.", label="sBG fitted on years 0-7")
ax.plot(years, 1000 * actual_years, marker="x", markersize=8, linestyle="none", color="black",
        label="actual")
ax.axvline(7.5, color="grey", linestyle=":")
ax.text(7.6, 900, "after the data", color="grey")
ax.set_xlabel("Year")
ax.set_ylabel("Regular customers still active (of 1,000)")
ax.set_title("Fader-Hardie Regular segment")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Explain.** Compare with your guess. Kaplan-Meier matches years 0 to 7 exactly. What does
# carrying it flat assume about churn after year 7, and what lets the sBG go further? Point to
# year 12 in the table.
#
# <details><summary>Why this solution works</summary>
#
# Kaplan-Meier describes the observed churn rates and has no rate for a year it has not seen;
# carrying it flat assumes nobody churns after the last observation. The sBG extrapolates because
# it assumes a form, a Beta mix of constant churn probabilities, which fits these data (Fader &
# Hardie 2007); on High End the same comparison gives 7.166 actual, 7.156 from the sBG and 7.381
# carried flat. That is why the decision below uses a 24-month horizon: the subscription data
# follow subscribers for at most 30 months, and beyond that we would need a parametric model and
# its assumption.
# </details>

# %% [markdown]
# # Part B · Who churns faster: the Cox model
#
# **Time zero** is when each subscriber's clock starts, here the first billing date. A group
# must be defined by what is known then: a status reached later makes its group look better only
# because its members stayed long enough to reach it (immortal-time bias). Onboarding is
# completed or not during the trial, before the first billing date, so it is known at time zero.
#
# The **hazard** h(t) is, among subscribers still active at t, the rate at which they cancel,
# per month.
#
# The next cell adds the dummy columns `plan_standard` and `plan_premium` (basic is the
# reference) and shows the share of each plan's subscribers who completed onboarding.

# %%
PLANS = ["basic", "standard", "premium"]


def add_plan_dummies(frame):
    """A copy with plan_standard and plan_premium (0/1); basic is the reference."""
    return frame.assign(plan_standard=(frame["plan"] == "standard").astype(int),
                        plan_premium=(frame["plan"] == "premium").astype(int))


def fit_km(frame):
    """Kaplan-Meier on some of the subscribers (lifelines draws no progress bar, so there is
    no progressbar=False to pass)."""
    return KaplanMeierFitter(alpha=ALPHA).fit(frame["tenure_months"], frame["churned"])


subs = add_plan_dummies(subs)
onboarding_by_plan = (subs.groupby("plan")["onboarded"].agg(["mean", "size"]).reindex(PLANS)
                      .rename(columns={"mean": "share onboarded", "size": "subscribers"}))
onboarding_by_plan.round(3)

# %% [markdown]
# Look at how far apart the three plans are at 12 months, and whether their bands overlap.

# %%
kmf_plan = {plan: fit_km(subs[subs["plan"] == plan]) for plan in PLANS}
fig, ax = plt.subplots(figsize=(7, 4))
for plan, style in zip(PLANS, ["-", "--", ":"], strict=True):
    plot_km(ax, kmf_plan[plan], plan, style)
ax.axvline(12, color="grey", linewidth=0.8)
ax.set_xlabel("Months since the first billing date")
ax.set_ylabel("Share still subscribed, S(t)")
ax.set_title("Kaplan-Meier by plan, 95% pointwise bands")
ax.set_ylim(0, 1.02)
ax.legend(title="Plan")
plt.tight_layout()
plt.show()

# %% [markdown]
# Look at the gap between onboarded and not onboarded at 12 months: Exercise 3 asks how much of
# it belongs to onboarding and how much to plan.

# %%
kmf_onboarded = {onb: fit_km(subs[subs["onboarded"] == onb]) for onb in (1, 0)}
fig, ax = plt.subplots(figsize=(7, 4))
plot_km(ax, kmf_onboarded[1], "onboarded", "-")
plot_km(ax, kmf_onboarded[0], "not onboarded", "--")
ax.axvline(12, color="grey", linewidth=0.8)
ax.set_xlabel("Months since the first billing date")
ax.set_ylabel("Share still subscribed, S(t)")
ax.set_title("Kaplan-Meier by onboarding, 95% pointwise bands")
ax.set_ylim(0, 1.02)
ax.legend()
plt.tight_layout()
plt.show()
print("Still subscribed at 12 months: onboarded"
      f" {float(kmf_onboarded[1].predict(12)):.1%}, not onboarded"
      f" {float(kmf_onboarded[0].predict(12)):.1%}.")

# %% [markdown]
# ## Exercise 3 · Hazard ratios adjusted for plan (7 minutes)
#
# The **Cox proportional-hazards model** is a regression for the hazard,
# h(t | z) = h<sub>0</sub>(t) exp(γ′z), that leaves the baseline hazard h<sub>0</sub>(t)
# unspecified; z holds a subscriber's covariates and γ their coefficients. The **hazard ratio**
# exp(γ<sub>k</sub>) is the factor by which covariate k multiplies the hazard: 0.67 means
# "cancels at 67% of the rate", not "67% as likely to cancel". A **confounder** is a variable
# that affects both who onboards and who churns, here plan; **adjusting** for it compares
# subscribers on the same plan.
#
# **Predict.** Premium subscribers onboard far more often than basic ones (the table at the
# start of Part B) and also churn less. In a model with onboarding alone, its hazard ratio is
# about 0.57: onboarded subscribers cancel at 57% of the rate of the rest (the checkpoint prints
# it). Once plan is in the model, will onboarding's hazard ratio be closer to 1, further from 1,
# or the same? Write your guess, as a number, in the cell below.
#
# **Task.** Write `cox_hazard_ratios(subs, covariates, alpha=0.05)` returning a DataFrame with
# one row per covariate, in the order given, and the columns `hr`, `lower` and `upper`: the
# hazard ratio and its (1 − alpha) confidence interval. Fit `CoxPHFitter(alpha=alpha)` with
# `duration_col="tenure_months"` and `event_col="churned"`.
#
# Hints. `CoxPHFitter` treats every column of the frame it is given as a covariate, so pass only
# `subs[covariates + ["tenure_months", "churned"]]`. All of `subs` stops inside lifelines with
# `TypeError: Cannot cast DatetimeArray to dtype float64` (the date columns) or
# `ValueError: could not convert string to float: 'basic'` (the plan column).
# `cph.hazard_ratios_` holds the hazard ratios. `np.exp(cph.confidence_intervals_)` gives the
# interval as two columns, lower then upper, whose names change with `alpha`
# (`"95% lower-bound"`), so take them by position.

# %% [markdown]
# *Your guess (Exercise 3):*

# %% tags=["exercise"]
def cox_hazard_ratios(subs, covariates, alpha=0.05):
    # TODO 3: fit CoxPHFitter(alpha=alpha) on the covariates and the two outcome columns only;
    # return hr, lower, upper per covariate (exponentiate the interval; take it by position)
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def cox_hazard_ratios(subs, covariates, alpha=0.05):
    # lifelines draws no progress bar, so there is no progressbar=False to pass.
    cph = CoxPHFitter(alpha=alpha).fit(subs[covariates + ["tenure_months", "churned"]],
                                       duration_col="tenure_months", event_col="churned")
    ci = np.exp(cph.confidence_intervals_)
    out = pd.DataFrame({"hr": cph.hazard_ratios_, "lower": ci.iloc[:, 0], "upper": ci.iloc[:, 1]})
    return out.loc[covariates]


# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    recover = " To move on, run workshop.use_reference(3)."
    hr_true = truth["model"]["hazard_ratios"]
    adjusted = cox_hazard_ratios(subs, COVARIATES, alpha=ALPHA)
    unadjusted = cox_hazard_ratios(subs, ["onboarded"], alpha=ALPHA)
    for covs, table in ((COVARIATES, adjusted), (["onboarded"], unadjusted)):
        assert isinstance(table, pd.DataFrame), (
            f"Checkpoint 3: cox_hazard_ratios returned a {type(table).__name__}; return a"
            " DataFrame with the columns hr, lower and upper." + recover
        )
        ignores = len(covs) == 1 and len(table) > 1
        assert list(table.index) == covs, (
            f"Checkpoint 3: with covariates {covs} your table's rows are {list(table.index)}."
            + (" Your function ignores `covariates`; fit on"
               " subs[covariates + ['tenure_months', 'churned']]." if ignores else
               " Index the result by covariate, in the order given (.loc[covariates]).")
            + recover
        )
        checks.columns(table, ["hr", "lower", "upper"], name="cox_hazard_ratios(...)")
        assert (table["hr"] > 0).all(), (
            f"Checkpoint 3: some hazard ratios are 0 or negative ({table['hr'].round(3).tolist()})."
            " You returned the coefficient γ, which is on the log scale; the hazard ratio is"
            " exp(γ), cph.hazard_ratios_." + recover
        )
        assert ((0 < table["lower"]) & (table["lower"] < table["hr"])
                & (table["hr"] < table["upper"])).all(), (
            "Checkpoint 3: some intervals do not surround their hazard ratio:"
            f" {table.round(3).to_dict('index')}. confidence_intervals_ is on the log scale;"
            " exponentiate it, and take lower then upper by position." + recover
        )
    assert abs(unadjusted.loc["onboarded", "hr"] - adjusted.loc["onboarded", "hr"]) > 1e-6, (
        "Checkpoint 3: with ['onboarded'] alone the ratio equals the adjusted one. Your function"
        " ignores `covariates`; fit on subs[covariates + ['tenure_months', 'churned']]."
        + recover
    )
    dummies = ("Use the dummy columns plan_standard and plan_premium (basic is the reference),"
               " not a text plan column or dummies with another plan as the reference." + recover)
    survival.ci_contains(hr_true["onboarded"], adjusted.loc["onboarded", "lower"],
                         adjusted.loc["onboarded", "upper"], level=1 - ALPHA,
                         name="the onboarding hazard ratio, adjusted for plan", hint=dummies,
                         label="Checkpoint 3")
    three = ["plan_standard", "plan_premium", "onboarded"]
    survival.k_of_K_ci_contains(
        {c: hr_true[c] for c in three},
        {c: (adjusted.loc[c, "lower"], adjusted.loc[c, "upper"]) for c in three},
        k=2, level=1 - ALPHA, name="standard, premium and onboarded", hint=dummies,
        label="Checkpoint 3",
    )
    u = unadjusted.loc["onboarded"]
    assert not u["lower"] <= hr_true["onboarded"] <= u["upper"] and u["hr"] < adjusted.loc[
        "onboarded", "hr"], (
        f"Checkpoint 3: without plan the onboarding ratio is {u['hr']:.3f} [{u['lower']:.3f},"
        f" {u['upper']:.3f}]; with plan {adjusted.loc['onboarded', 'hr']:.3f}. Without plan it"
        f" should be lower, and its interval should miss the truth {hr_true['onboarded']:.3f}."
        " Check that the function fits only the covariates it is given." + recover
    )
hr_table = pd.concat(
    [adjusted.assign(model="with plan"), unadjusted.assign(model="onboarding only")]
).reset_index(names="covariate")
hr_table["truth"] = hr_table["covariate"].map(hr_true)
print("Hazard ratios with 95% confidence intervals. The offer has no single true hazard ratio"
      " (NaN in the truth column); Exercise 4 shows why.")
hr_table[["model", "covariate", "hr", "lower", "upper", "truth"]].round(3)

# %% [markdown]
# **Explain.** Compare with your guess. Which way did plan bias the unadjusted ratio, and why?
# Point to the share onboarded per plan and the plan hazard ratios.
#
# <details><summary>Why this solution works</summary>
#
# Plan affects both onboarding and churn, so the unadjusted comparison credits onboarding with
# part of the premium plan's lower churn. Holding plan fixed removes that part. This works here
# because plan is the only thing that drives onboarding in this simulation; real onboarding also
# depends on motivation, which is not in the data (the stretch). The `intro_offer` ratio of about
# 1.36 is an average over tenure of an effect that is not constant (Exercise 4); hazard ratios
# are ratios of rates, not probabilities of churning.
# </details>

# %% [markdown]
# ### Run · The proportional-hazards test
#
# The Cox model assumes **proportional hazards**, as its name says: each hazard ratio is the same
# at every tenure. The **PH test** asks, covariate by covariate, whether its hazard ratio changes
# with tenure; a small p-value says it does. The next cell refits your Exercise 3 model as `cph`
# (your function returns a table, not the model) and runs lifelines' `proportional_hazard_test`.
# It flags p < 0.01 rather than 0.05 because it runs four tests at once. With 8,000 subscribers
# the test can also flag a change too small to matter, so Exercise 4 measures how large it is.

# %%
ph_frame = subs[COVARIATES + ["tenure_months", "churned"]]
# lifelines draws no progress bar, so there is no progressbar=False to pass.
cph = CoxPHFitter(alpha=ALPHA).fit(ph_frame, duration_col="tenure_months", event_col="churned")
ph_test = proportional_hazard_test(cph, ph_frame, time_transform="rank").summary
flagged = ph_test.index[ph_test["p"] < 0.01].tolist()
print("Covariates whose hazard ratio changes with tenure (p < 0.01):", ", ".join(flagged) or "none")
ph_test.assign(p=ph_test["p"].map("{:.2g}".format)).round(2)

# %% [markdown]
# ## Exercise 4 · An effect that fades: hazard ratios before and after month 3 (8 minutes)
#
# A **landmark analysis** picks a time (the landmark), keeps those still at risk then, and
# starts the clock again (Giobbie-Hurder et al. 2013).
#
# **Predict.** The full-period model gave `intro_offer` a hazard ratio of about 1.36, and the
# test above says the offer's effect changes with tenure. The offer discounts the first three
# paid months. In months 0 to 3, will its hazard ratio be higher than 1.36, lower, or about the
# same? After month 3? Write both guesses in the cell below.
#
# **Task.** Write `landmark_split(subs, landmark=3.0)` returning `(before, after)`, two new
# DataFrames, each with the columns of `subs`:
#
# - **before**: every subscriber, followed only to the landmark: tenure = min(tenure, landmark),
#   and `churned` = 1 only if the cancellation happened by the landmark;
# - **after**: the subscribers still active after the landmark (tenure > landmark), with the
#   clock restarted: tenure − landmark.
#
# Leave `subs` unchanged: build new frames with `subs.assign(...)`. Assigning into `subs`, or
# into a frame that is `subs` under another name, changes the data every later cell uses. The
# checkpoint then runs your Exercise 3 function on each part and prints the offer's and
# onboarding's hazard ratios in each period.

# %% [markdown]
# *Your guesses (Exercise 4), months 0-3 and after month 3:*

# %% tags=["exercise"]
def landmark_split(subs, landmark=3.0):
    # TODO 4: before = everyone, tenure clipped at the landmark, churned only if by the
    # landmark; after = tenure > landmark, tenure - landmark. Use subs.assign(...), not
    # assignment into subs.
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def landmark_split(subs, landmark=3.0):
    late = subs["tenure_months"] > landmark
    before = subs.assign(tenure_months=subs["tenure_months"].clip(upper=landmark),
                         churned=subs["churned"].where(~late, 0))
    after = subs[late].assign(tenure_months=subs.loc[late, "tenure_months"] - landmark)
    return before, after


# %% tags=["checkpoint"]
rebuild = ("rerun the cell under \"Run · Tenure, churn and two dashboard numbers\" in Part A,"
           " then Part B's first cell (it adds plan_standard and plan_premium), then this"
           " checkpoint.")
# Before any check: a landmark_split that changed `subs` in place on an earlier run leaves it
# changed, even after the function is fixed or the reference is chosen.
assert subs.equals(add_plan_dummies(add_tenure(raw.drop(columns=["true_tenure_months",
                                                                 "true_rmst24_gain"])))), (
    "Checkpoint 4: `subs` is no longer the table Parts A and B built: a function changed it in"
    " place, probably an earlier run of your landmark_split. Rebuild it before anything else: "
    + rebuild
)
with workshop.checkpoint(4):
    recover = " To move on, run workshop.use_reference(4)."
    subs_copy = subs.copy()
    before, after = landmark_split(subs, LANDMARK)
    assert subs.equals(subs_copy), (
        "Checkpoint 4: your function changed `subs` in place, which would make Exercise 5 and"
        " the Decision wrong without an error. Fix it to build new frames (subs.assign(...) or"
        " subs.copy()) and rerun its cell, or run workshop.use_reference(4). Then rebuild"
        " `subs`: " + rebuild
    )
    for name, part in (("before", before), ("after", after)):
        checks.columns(part, list(subs.columns), name=f"'{name}'")
    assert len(before) == len(subs), (
        f"Checkpoint 4: 'before' has {len(before):,} rows; subs has {len(subs):,}. 'before'"
        " keeps every subscriber and stops their clock at the landmark"
        " (clip(upper=landmark)); do not filter." + recover
    )
    assert before["tenure_months"].max() <= LANDMARK + 1e-12, (
        f"Checkpoint 4: 'before' has tenures up to {before['tenure_months'].max():.2f} months."
        " Stop every clock at the landmark: tenure = min(tenure, landmark)." + recover
    )
    late_cancels = int(((before["churned"].to_numpy() == 1)
                        & (subs["tenure_months"].to_numpy() > LANDMARK)).sum())
    assert late_cancels == 0, (
        f"Checkpoint 4: 'before' counts {late_cancels:,} cancellations that happened after"
        " month 3. These subscribers were still active at month 3, so in 'before' they are"
        " censored there: churned = 0." + recover
    )
    want_after = subs.loc[subs["tenure_months"] > LANDMARK, ["customer_id", "tenure_months"]]
    merged = want_after.merge(after[["customer_id", "tenure_months"]], on="customer_id",
                              how="outer", suffixes=("", "_after"), indicator=True)
    restart = ("'after' keeps those with tenure > landmark and restarts their clock:"
               " tenure − landmark.")
    assert (after["tenure_months"] > 0).all() and (merged["_merge"] == "both").all(), (
        f"Checkpoint 4: 'after' has {len(after):,} rows; {len(want_after):,} subscribers were"
        f" still active after month 3. {restart}" + recover
    )
    shift = merged["tenure_months"] - merged["tenure_months_after"]
    assert np.allclose(shift, LANDMARK, rtol=0, atol=1e-9), (
        f"Checkpoint 4: the shortest tenure in 'after' is {after['tenure_months'].min():.2f}"
        f" months; with the clock restarted it would be {(want_after['tenure_months'] - LANDMARK).min():.2f}."
        f" {restart}" + recover
    )
    parts = {"before": cox_hazard_ratios(before, COVARIATES, alpha=ALPHA),
             "after": cox_hazard_ratios(after, COVARIATES, alpha=ALPHA)}
    landmark_table = pd.concat(
        [t.loc[["intro_offer", "onboarded"]].assign(period=p) for p, t in parts.items()]
    ).reset_index(names="covariate")[["period", "covariate", "hr", "lower", "upper"]]
    offer_true = truth["model"]["intro_offer"]
    landmark_table["truth"] = [offer_true["hr_before"], hr_true["onboarded"],
                               offer_true["hr_after"], hr_true["onboarded"]]
    keys = [f"{r.covariate}, {r.period} month 3" for r in landmark_table.itertuples()]
    survival.k_of_K_ci_contains(
        dict(zip(keys, landmark_table["truth"], strict=True)),
        {k: (r.lower, r.upper)
         for k, r in zip(keys, landmark_table.itertuples(), strict=True)},
        k=3, level=1 - ALPHA, name="the offer and onboarding hazard ratios before and after"
        " month 3",
        hint=("On the committed data this does not happen with the reference. All four:"
              f" {landmark_table.round(3).to_dict('records')}. Compare with"
              " workshop.use_reference(4)." + recover),
        label="Checkpoint 4",
    )
print("Hazard ratios with 95% confidence intervals, before and after the 3-month landmark:")
landmark_table.round(3)

# %% [markdown]
# Look at the month where the two lines meet. The monthly churn rate of month k is the
# cancellations in month k divided by the subscribers active at the start of month k.

# %%
rates = survival.monthly_churn_rate(subs, months=HORIZON, by="intro_offer")
fig, ax = plt.subplots(figsize=(7, 4))
for offer, label, marker, style in ((1, "intro offer", "o", "-"), (0, "no offer", "s", "--")):
    r = rates[rates["intro_offer"] == offer]
    ax.plot(r["month"], 100 * r["rate"], marker=marker, linestyle=style, label=label)
ax.axvline(LANDMARK + 0.5, color="grey", linestyle=":")
ax.text(LANDMARK + 0.7, ax.get_ylim()[1] * 0.9, "discount ends", color="grey")
ax.set_xlabel("Month of tenure")
ax.set_ylabel("Monthly churn rate (%)")
ax.set_title("Cancellations in month k per subscriber active at its start")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Explain.** Compare with your guesses. Why was 1.36 a misleading summary of the offer, and
# what should a report say instead? Point to the two periods' intervals and the churn-rate plot.
#
# <details><summary>Why this solution works</summary>
#
# The full-period ratio averages a doubled churn rate in the first three months with no
# difference afterwards, weighted by when cancellations happen. Splitting at the landmark
# recovers both. Deal-seekers leave when the discount ends; those who stay are like everyone
# else. Alternatives: `strata=["intro_offer"]` gives the offer its own baseline and keeps the
# other ratios but reports no offer effect; a time-varying coefficient (`CoxTimeVaryingFitter`
# on episode data) is the general tool. A caution: here subscribers differ only in observed ways,
# so the survivors at month 3 are comparable; in real data the survivors are a selected group,
# and an after-landmark ratio carries that selection (Hernán 2010).
# </details>

# %% [markdown]
# # Part C · From hazard ratios to months and dollars
#
# **RMST(t\*)**, the restricted mean survival time, is the expected months subscribed within the
# first t\* months: the area under S(t) up to t\*. It suits a decision: it is in months, needs no
# proportional hazards, and multiplies by a monthly margin (Royston & Parmar 2013).
#
# The next cell computes the unadjusted gain: RMST(24) of the onboarded subscribers minus that of
# the rest, each from its own Kaplan-Meier curve with `restricted_mean_survival_time(kmf, t=24)`.
# Exercise 5 repeats this call within each plan. The **target population** is the subscribers a
# call would reach, those who did not complete onboarding; the cell computes their plan mix,
# `weights`.

# %%
rmst_by_group = {onb: restricted_mean_survival_time(kmf_onboarded[onb], t=HORIZON)
                 for onb in (1, 0)}
rmst_unadjusted = rmst_by_group[1] - rmst_by_group[0]
target = subs[subs["onboarded"] == 0]
weights = target["plan"].value_counts(normalize=True).reindex(PLANS)
print(f"RMST({HORIZON}): onboarded {rmst_by_group[1]:.2f} months, not onboarded"
      f" {rmst_by_group[0]:.2f}: unadjusted gain {rmst_unadjusted:.2f} months.")
print(f"Target population: {len(target):,} subscribers who did not onboard. Plan mix"
      " (weights): " + ", ".join(f"{p} {w:.3f}" for p, w in weights.items()))

# %% [markdown]
# ## Exercise 5 · Months gained from onboarding (8 minutes)
#
# **Standardized by plan** means: compare within each plan, then add up the differences weighted
# by the target population's plan mix. A **95% bootstrap percentile interval** resamples
# subscribers with replacement within each plan × onboarding group, recomputes the estimate, and
# takes the 2.5th and 97.5th percentiles of the results.
#
# **Predict.** Unadjusted, onboarded subscribers stay about 4.1 more months within the first 24
# (printed above). Standardized by plan, will the gain be larger, about the same or smaller? By
# how much: under 0.5 month, 0.5 to 2, or over 2? Write your guess in the cell below.
#
# **Task.** Write `rmst_gain(subs, horizon, weights)` returning a float: for each plan p in
# `weights`, RMST(horizon) of its onboarded subscribers minus that of its not-onboarded ones,
# each from its own Kaplan-Meier fit (`fit_km(frame)`, defined at the start of Part B, fits one)
# and `restricted_mean_survival_time(kmf, t=horizon)`; return
# Σ<sub>p</sub> weights[p] × difference. `weights` is a dict or Series over plans; assert that
# it sums to 1.

# %% [markdown]
# *Your guess (Exercise 5):*

# %% tags=["exercise"]
def rmst_gain(subs, horizon, weights):
    # TODO 5: assert the weights sum to 1; for each plan, fit_km on its onboarded and its
    # not-onboarded subscribers, take restricted_mean_survival_time(kmf, t=horizon) of each,
    # and add weights[plan] * (onboarded - not onboarded)
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def rmst_gain(subs, horizon, weights):
    w = pd.Series(weights, dtype=float)
    assert abs(w.sum() - 1) < 1e-9, f"weights must sum to 1, not {w.sum():.4g}"
    total = 0.0
    for plan, weight in w.items():
        area = {}
        for onb in (1, 0):
            group = subs[(subs["plan"] == plan) & (subs["onboarded"] == onb)]
            area[onb] = restricted_mean_survival_time(fit_km(group), t=horizon)
        total += weight * (area[1] - area[0])
    return float(total)


# %% [markdown]
# The checkpoint runs the fast checks first, then a bootstrap of your function: B resamples (200,
# or 100 with QUICK), each refitting six Kaplan-Meier curves. While it runs, read the Explain
# prompt below.

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    recover = " To move on, run workshop.use_reference(5)."

    def direct_gain(plan):
        area = {onb: restricted_mean_survival_time(
            fit_km(subs[(subs["plan"] == plan) & (subs["onboarded"] == onb)]), t=HORIZON)
            for onb in (1, 0)}
        return area[1] - area[0]

    direct = {plan: direct_gain(plan) for plan in ("basic", "premium")}
    got_basic = rmst_gain(subs, HORIZON, {"basic": 1.0})
    if not np.isfinite(got_basic):
        cause = ("A gain that is not a number comes from restricted_mean_survival_time(kmf)"
                 " without t: it then integrates each curve to infinity, and infinity minus"
                 " infinity is nan. Pass t=horizon.")
    elif abs(got_basic + direct["basic"]) < 1e-9:
        cause = "You subtracted the other way: onboarded minus not onboarded."
    else:
        cause = ("For each plan, fit one KaplanMeierFitter to its onboarded subscribers and one"
                 " to its not-onboarded subscribers, and take restricted_mean_survival_time(kmf,"
                 " t=horizon) of each, as the provided cell above does for everyone.")
    assert abs(got_basic - direct["basic"]) < 1e-9, (
        f"Checkpoint 5: with weights {{'basic': 1.0}} your gain is {got_basic:.4f} months; for"
        f" basic alone it is {direct['basic']:.4f}. {cause}" + recover
    )
    # Unequal weights: with 0.5 and 0.5 a plain average of the plans would pass.
    got_mix = rmst_gain(subs, HORIZON, {"basic": 0.75, "premium": 0.25})
    want_mix = 0.75 * direct["basic"] + 0.25 * direct["premium"]
    assert abs(got_mix - want_mix) < 1e-9, (
        f"Checkpoint 5: with weights {{'basic': 0.75, 'premium': 0.25}} your gain is"
        f" {got_mix:.4f}; 0.75 × basic's difference + 0.25 × premium's is {want_mix:.4f}."
        " Weight each plan's difference by weights[plan] and add them up; a plain average"
        " ignores the weights." + recover
    )
    try:
        rmst_gain(subs, HORIZON, {"basic": 0.25, "premium": 0.25})
        raised = False
    except (AssertionError, ValueError):
        raised = True
    assert raised, (
        "Checkpoint 5: weights that sum to 0.5 gave no error. Assert that the weights sum to 1"
        " (within 1e-9), so that counts passed by mistake are caught." + recover
    )
    boot = survival.bootstrap_interval(lambda d: rmst_gain(d, HORIZON, weights), subs,
                                       by=["plan", "onboarded"], B=B, seed=BOOT_SEED,
                                       prob=1 - ALPHA)
    assert not boot.lower <= rmst_unadjusted <= boot.upper, (
        f"Checkpoint 5: the unadjusted gain {rmst_unadjusted:.2f} lies inside your"
        f" {boot.label} [{boot.lower:.2f}, {boot.upper:.2f}]. You compared onboarded with not"
        " onboarded across all plans; compare within each plan, then weight by the target"
        " population's plan mix." + recover
    )
    survival.ci_contains(
        truth["rmst"]["gain_target"], boot.lower, boot.upper, level=1 - ALPHA,
        kind="bootstrap percentile interval",
        name="the months onboarding adds within 24 months for the target population",
        label="Checkpoint 5",
        hint=("A 95% interval misses now and then (on other seeds of this generator the"
              " reference missed in about 1 in 10), but on the committed data the reference"
              " passes; compare your estimate with workshop.use_reference(5)." + recover),
    )
print(f"Months onboarding adds within {HORIZON}, standardized to the target population:"
      f" {boot.estimate:.2f} ({boot.label} {boot.lower:.2f} to {boot.upper:.2f},"
      f" B = {boot.B}). Unadjusted: {rmst_unadjusted:.2f}.")

# %% [markdown]
# ### Run · The gain plan by plan
#
# One row per plan: RMST(24) of the onboarded and the not onboarded, the difference, the
# target population's weight, and weight × difference. The last row adds them up: your estimate.

# %%
rows = []
for plan in PLANS:
    area = {onb: restricted_mean_survival_time(
        fit_km(subs[(subs["plan"] == plan) & (subs["onboarded"] == onb)]), t=HORIZON)
        for onb in (1, 0)}
    rows.append({"plan": plan, "RMST onboarded": area[1], "RMST not onboarded": area[0],
                 "difference": area[1] - area[0], "weight": weights[plan],
                 "weight × difference": weights[plan] * (area[1] - area[0])})
by_plan = pd.DataFrame(rows).set_index("plan")
by_plan.loc["sum"] = [np.nan, np.nan, np.nan, by_plan["weight"].sum(),
                      by_plan["weight × difference"].sum()]
by_plan.round(3)

# %% [markdown]
# **Explain.** Compare with your guess. In every plan, onboarding adds less than the unadjusted
# 4.06 months (the `difference` column). Where does the rest of the 4.06 come from? Point to the
# RMSTs of basic and premium subscribers in the table, and to the share of each plan that
# onboards (the table at the start of Part B).
#
# <details><summary>Why this solution works</summary>
#
# The unadjusted difference compares onboarded subscribers (about a third of them premium) with
# the rest (two thirds basic); much of its 4.1 months is plan, not onboarding. Comparing within
# plan and weighting by the plan mix of the people a call would reach answers the decision's
# question. Here the three plans' differences are close (2.5 to 3.0 months), so the plan mix
# moves the answer little; comparing within plan is what moves it. If the plans' differences
# were far apart, the target population's mix would matter as well. Stratifying fully adjusts
# here because onboarding depends on plan alone. lifelines'
# `restricted_mean_survival_time(..., return_variance=True)` returns the variance of min(T, 24)
# across subscribers, not the uncertainty of the estimate; that is why the interval comes from
# the bootstrap.
# </details>

# %% [markdown]
# ## Decision · Is an onboarding call worth its cost?
#
# **Rule** (before the numbers): call a subscriber who did not onboard when
# q × ΔRMST × m > c. Here ΔRMST is the months of subscription that onboarding adds within 24
# months (Exercise 5), m = $12 of margin a month, c = $15 a call, and q the share of called
# subscribers whom the call gets to complete onboarding. Nobody knows q yet, so the cell shows
# three values. The **break-even q\*** is the share of called subscribers who must complete
# onboarding for a call to pay: c ÷ (ΔRMST × m).
#
# The cell also reports, for each q, the share of bootstrap resamples in which a call pays. That
# share is not a probability that calls pay: the bootstrap measures sampling noise only, and
# confounding, if any, would shift every resample the same way.
#
# Run the cell, then write your sentence.

# %%
NUDGE_COST = 3.0  # dollars: an illustrative cheaper nudge (an e-mail sequence), not a quote
gain, draws = boot.estimate, boot.draws
q_mid = Q_GRID[1]  # the middle scenario
discount_cut = 1 - truth["rmst"]["gain_target_discounted"] / truth["rmst"]["gain_target"]


def interval(x):
    lo, hi = np.quantile(x, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lo), float(hi)


print(f"Number          : onboarding adds {gain:.2f} months within {HORIZON} (95% bootstrap"
      f" interval {boot.lower:.2f} to {boot.upper:.2f})")
print("                  for subscribers who did not onboard, standardized by plan."
      f" Unadjusted: {rmst_unadjusted:.2f}.")
print(f"Rule            : call when q × ΔRMST × m > c, with m = ${MARGIN:.0f} a month and"
      f" c = ${CALL_COST:.0f} a call; q is not known.")
for i, q in enumerate(Q_GRID):
    lo, hi = interval(q * MARGIN * draws)
    pays = float(np.mean(q * MARGIN * draws > CALL_COST))
    head = "Value per call  : " if i == 0 else " " * 18
    print(f"{head}q = {q:.2f}: ${q * MARGIN * gain:.2f} (95% interval ${lo:.2f} to ${hi:.2f});"
          f" calls pay in {pays:.0%} of bootstrap resamples")
q_lo, q_hi = interval(CALL_COST / (MARGIN * draws))
print(f"Break-even q    : {CALL_COST / (gain * MARGIN):.3f} ({q_lo:.3f} to {q_hi:.3f}); from the"
      f" unadjusted gain it would be {CALL_COST / (rmst_unadjusted * MARGIN):.3f}.")
print(f"Scale           : {len(target):,} subscribers did not onboard; calling all of them costs"
      f" ${len(target) * CALL_COST:,.0f} and returns ${len(target) * q_mid * MARGIN * gain:,.0f}"
      f" at q = {q_mid:.2f}.")
print("Assumes         : onboarding prompted by a call works like onboarding chosen during the"
      " trial.")
print("                  Onboarding was not randomized, so the gain is an association, not a"
      " measured effect.")
print("Recommendation  : (yours) what to do, for how many subscribers, how sure, and what would"
      " change it.")
print("What would change it: a pilot's measured q and retention; a cheaper nudge (at"
      f" ${NUDGE_COST:.0f} the break-even q is {NUDGE_COST / (gain * MARGIN):.0%});"
      f" discounting at 0.8% a month ({discount_cut:.0%} less in the simulation's truth).")
print("Your sentence   : ________________________________________________")
print(f"Mode            : {f'QUICK run (B = {B}): treat this as a rough answer.' if QUICK else 'FULL run.'}")

# %% [markdown]
# **Decide.** Write your sentence in the cell below: what to do, for how many subscribers, how
# sure you are, and what would change it.

# %% [markdown]
# *Your sentence:*

# %% [markdown]
# Only a simulation can do this: compare with the truth. The cell prints the true gain for the
# target population and what a call is truly worth at each q.

# %%
true_gain = truth["rmst"]["gain_target"]
inside = boot.lower <= true_gain <= boot.upper
print(f"True gain for the target population: {true_gain:.3f} months (your interval"
      f" {boot.lower:.2f} to {boot.upper:.2f} {'covers' if inside else 'misses'} it);"
      f" unadjusted truth {truth['rmst']['observed_groups']['unadjusted_gain']:.3f}.")
for q in Q_GRID:
    print(f"  q = {q:.2f}: a call is truly worth ${q * MARGIN * true_gain:.2f} against"
          f" ${CALL_COST:.0f}.")
print(f"True break-even q: {truth['decision']['true_break_even_q']:.3f}.")

# %% [markdown]
# ## Stretch (optional) · A confounder that is not in the data
#
# The next cell regenerates the subscribers with hidden motivation: `hidden_confounding=0.5`
# makes motivated subscribers both more likely to onboard and less likely to churn, and
# motivation is not a column. It then reruns your `cox_hazard_ratios` and `rmst_gain` on the new
# data and prints them beside the truth: the onboarding hazard ratio is still 0.670 for
# subscribers of the same motivation, and the true gain is the mean of `true_rmst24_gain` among
# the subscribers who did not onboard.
#
# **Predict.** With motivation hidden, will the plan-adjusted onboarding hazard ratio's 95%
# interval still cover 0.670? Will the standardized gain be above or below the true gain? Write
# your guesses before you run the cell.
#
# Two shorter options: run `rmst_gain` for one plan at a time (a weight of 1.0) with
# `survival.bootstrap_interval`, to see whether the data can tell the plans' gains apart; or
# redo the Decision with Module 4's 0.8% monthly discount, counting a month subscribed at time t
# as worth 1.008<sup>−t</sup> of a month now (the truth file's convention; the Decision's "What
# would change it" line gives the truth's answer, about 10% less).

# %%
hidden = subscriptions(seed=2034, hidden_confounding=0.5).subscribers
subs_hidden = add_plan_dummies(add_tenure(hidden.drop(columns=["true_tenure_months",
                                                               "true_rmst24_gain"])))
cox_hidden = cox_hazard_ratios(subs_hidden, COVARIATES, alpha=ALPHA)
weights_hidden = (subs_hidden.loc[subs_hidden["onboarded"] == 0, "plan"]
                  .value_counts(normalize=True).reindex(PLANS))
gain_hidden = rmst_gain(subs_hidden, HORIZON, weights_hidden)
true_gain_hidden = hidden.loc[hidden["onboarded"] == 0, "true_rmst24_gain"].mean()
o = cox_hidden.loc["onboarded"]
print(f"Onboarding hazard ratio, adjusted for plan: {o['hr']:.3f} [{o['lower']:.3f},"
      f" {o['upper']:.3f}]; truth for subscribers of the same motivation"
      f" {truth['model']['hazard_ratios']['onboarded']:.3f}.")
print(f"Months onboarding adds within {HORIZON}, standardized by plan: {gain_hidden:.2f};"
      f" true {true_gain_hidden:.2f}.")

# %% [markdown]
# **Explain.** Compare with your guesses. Why does adjusting for plan no longer remove the bias,
# and what in this output would have warned you if you did not know the truth? Point to the
# hazard ratio's interval and the two gains.
#
# <details><summary>Why this happens</summary>
#
# Motivated subscribers onboard more and churn less, so part of the onboarded group's lower churn
# is motivation, not onboarding. Plan does not measure motivation, so comparing within plan
# leaves that part in: the hazard ratio is further from 1 than the truth and the gain is too
# large. Nothing in the output flags it; the interval is about as narrow as before. Only a design
# that breaks the link between motivation and onboarding removes it, such as a randomized pilot
# of the calls (Module 6). Methods for observational data (Module 15) still need the confounders
# to be measured.
# </details>
