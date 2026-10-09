# %% [markdown]
# <!--
# APIs checked for this lab (2026-10-09) against the installed source of SciPy 1.16.3 in the
# workshop's environment: `scipy.stats.chisquare(f_obs, f_exp, ddof=0, axis=0, *, sum_check=True)`
# (counts, not shares; `sum_check` requires the expected counts to add up to the observed total)
# and `scipy.stats.norm.ppf` / `norm.sf`. Everything else is NumPy and pandas.
# -->
#
# **How this lab runs.** Nothing here fits a model: every method is a formula or a small
# simulation, so every cell runs in seconds. `QUICK` (in the harness cell) shrinks the
# simulations; the counts are printed below. With QUICK on, simulated rates are noisier and the
# checkpoints use wider tolerances (stated beside each); a decision near its threshold can flip.
# That is a lesson about sample size, not a bug.
#
# Terms used throughout:
#
# - A **randomized experiment** (an A/B test) assigns each customer at random to **treatment**
#   (here: gets the email) or **control** (does not). Randomization makes the two groups alike in
#   everything except the email, so the difference in their outcomes estimates the email's effect.
# - The **average treatment effect (ATE)** is the mean outcome with the email minus the mean outcome
#   without it, over the same customers. Here the outcome is spend in dollars per customer.
# - A **95% confidence interval (CI)** is a range computed from the data by a procedure that covers
#   the true value in 95% of repeated experiments. All intervals in this lab are 95% normal
#   approximation intervals: estimate ± 1.96 standard errors.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mktstats.data import load_hillstrom, load_synthetic, load_truth

SEED = 2026  # every simulation starts from np.random.default_rng(SEED + exercise number)
N_POWER_SIMS = 500 if QUICK else 2000  # Exercise 1: simulated experiments
POWER_TOL = 0.06 if QUICK else 0.03  # about 3.4 binomial standard errors of a power of 0.8
N_PEEK_SIMS = 1000 if QUICK else 4000  # Exercise 3: simulated A/A tests
PEEK_TOL = 0.03 if QUICK else 0.015  # about 4.4 binomial standard errors of a rate of 0.05
N_BOOT = 500 if QUICK else 2000  # Exercise 5: bootstrap resamples
BOOT_TOL = 0.20 if QUICK else 0.10  # how close the delta method must be to the bootstrap

print(f"QUICK = {QUICK}: {N_POWER_SIMS:,} power simulations, {N_PEEK_SIMS:,} A/A tests,"
      f" {N_BOOT:,} bootstrap resamples")

# %% [markdown]
# # Part A · Before the test
#
# A test must be big enough to see the effect you care about. Two numbers fix that before any
# data arrive:
#
# - the **significance level** α: the false-positive rate you accept when there is no effect
#   (0.05, two-sided);
# - the **power** 1 − β: the chance the test detects an effect of a given size when it is real
#   (0.8 is the usual target).
#
# The **minimum detectable effect (MDE)** is the smallest true effect that a test of a given size
# detects with that power. For a difference between two means with outcome standard deviation σ
# and n customers per arm, the standard error of the difference is σ√(2/n), and
#
# $$ n = \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\,\sigma^2}{\text{MDE}^2}, \qquad
#    \text{MDE} = (z_{1-\alpha/2} + z_{1-\beta})\,\sigma\,\sqrt{2/n}, $$
#
# where $z_q$ is the q-quantile of the standard normal distribution (`stats.norm.ppf(q)`):
# $z_{0.975} = 1.96$ and $z_{0.8} = 0.84$.

# %% [markdown]
# ## Exercise 1 · Sample size and minimum detectable effect (7 minutes)
#
# **Predict.** In the real Hillstrom email test you will meet below, two-week spend per customer
# has a standard deviation of about \$15. To detect a \$0.50 lift in mean spend per customer
# (α = 0.05 two-sided, 80% power), how many customers do you need **per arm**: under 10,000,
# 10,000 to 100,000, or over 100,000? Write your guess down.
#
# **Task.** Write two functions from the formulas above:
#
# - `sample_size(sd, mde, alpha=0.05, power=0.8)` returning the number of customers per arm as an
#   `int`, rounded **up** (`np.ceil`);
# - `mde(n_per_arm, sd, alpha=0.05, power=0.8)` returning the minimum detectable effect as a
#   `float`, in the units of `sd`.

# %% tags=["exercise"]
def sample_size(sd, mde, alpha=0.05, power=0.8):
    # TODO 1: z = norm.ppf(1 - alpha / 2) + norm.ppf(power); n = 2 * (z * sd / mde) ** 2, rounded up
    raise NotImplementedError("TODO 1")


def mde(n_per_arm, sd, alpha=0.05, power=0.8):
    # TODO 1: the same z, times sd * sqrt(2 / n_per_arm)
    raise NotImplementedError("TODO 1")


# %% tags=["solution"]
# @title Solution 1 — try it yourself first { display-mode: "form" }
@workshop.solution(1)
def sample_size(sd, mde, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sd / mde) ** 2))


@workshop.solution(1)
def mde(n_per_arm, sd, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2 / n_per_arm))


# %% [markdown]
# The checkpoint checks your formula against a simulation: it runs many experiments of the size
# your `sample_size` returns, each with a true lift of \$0.50, and counts how often a two-sided
# z-test at α = 0.05 detects it. If the formula is right, that share is close to 0.8.

# %%
def simulate_power(n_per_arm, sd, lift, n_sims, rng, alpha=0.05, chunk=100):
    """Share of simulated experiments whose two-sided z-test rejects 'no effect' at alpha.

    Each experiment draws n_per_arm normal outcomes per arm with standard deviation sd; the
    treated mean is higher by `lift`. Runs `chunk` experiments at a time to keep memory small.
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)
    rejections = 0
    for start in range(0, n_sims, chunk):
        k = min(chunk, n_sims - start)
        control = rng.normal(0.0, sd, size=(k, n_per_arm))
        treated = rng.normal(lift, sd, size=(k, n_per_arm))
        diff = treated.mean(axis=1) - control.mean(axis=1)
        se = np.sqrt(treated.var(axis=1, ddof=1) / n_per_arm + control.var(axis=1, ddof=1) / n_per_arm)
        rejections += int(np.sum(np.abs(diff / se) > z_crit))
    return rejections / n_sims


# %% tags=["checkpoint"]
with workshop.checkpoint(1):
    n_needed = sample_size(15, 0.5)
    assert isinstance(n_needed, (int, np.integer)), (
        f"sample_size should return a whole number of customers (an int), not {type(n_needed).__name__}."
        " Round up with int(np.ceil(...))."
    )
    checks.close(mde(n_needed, 15), 0.5, rel=0.01, name="mde(sample_size(15, 0.5), 15)")
    assert 1_000 <= n_needed <= 200_000, (
        f"sample_size(15, 0.5) is {n_needed:,} customers per arm, far from what the formula gives."
        " Check the factor 2 (two arms), the square, and that sd and mde are not swapped."
        " To move on, run workshop.use_reference(1)."
    )
    power_sim = simulate_power(n_needed, 15, 0.5, N_POWER_SIMS, np.random.default_rng(SEED + 1))
    assert abs(power_sim - 0.8) <= POWER_TOL, (
        f"At n = {n_needed:,} per arm, {power_sim:.1%} of {N_POWER_SIMS:,} simulated experiments"
        f" detected the $0.50 lift; the target is 80% (± {POWER_TOL:.0%}). Your sample size is"
        f" {'too small' if power_sim < 0.8 else 'too large'}. Common causes: a one-sided z"
        " (norm.ppf(1 - alpha) instead of 1 - alpha / 2), a missing factor 2 for the two arms, or"
        " z_power taken as norm.ppf(1 - power). Fix TODO 1, or run workshop.use_reference(1)."
    )
    print(f"n per arm = {n_needed:,}; simulated power = {power_sim:.3f} ({N_POWER_SIMS:,} experiments)")

# %% [markdown]
# **Explain.** Compare the number with your guess. Then answer: if you wanted to detect a
# \$0.25 lift instead of \$0.50, how many customers per arm would you need, and why that many?
#
# <details><summary>Why this solution works</summary>
#
# The standard error of a difference in means falls with √n, so halving the effect you want to
# see needs four times the sample. The formula and the simulation agree because, at these sample
# sizes, the difference in means is close to normal even though individual spend is not
# (the central limit theorem). A wider spend distribution (larger σ) or a stricter α raises n;
# a covariate that explains part of the outcome lowers the effective σ (Exercise 4).
# </details>

# %% [markdown]
# The synthetic email experiment you analyze below has about 10,000 customers per arm. Run the
# next cell to see the smallest effect it can detect, in dollars of spend per customer.

# %%
experiment = load_synthetic("email_experiment")
truth = load_truth()["email_experiment"]
n_arm = int(experiment["treatment"].value_counts().min())
spend_sd = experiment["spend"].std()
print(f"Synthetic experiment: {len(experiment):,} customers, spend SD ${spend_sd:.2f}")
print(f"MDE with {n_arm:,} per arm: ${mde(n_arm, spend_sd):.2f} of spend per customer")

# %% [markdown]
# ## Exercise 2 · Sample-ratio mismatch (5 minutes)
#
# A **sample-ratio mismatch (SRM)** is a gap between the share of customers each arm was meant to
# get and the share it got. It means the randomization or the logging is broken, and then the
# comparison is not trustworthy, whatever the effect looks like (Fabijan et al. 2019; Kohavi, Tang
# & Xu 2020). The check is a chi-square goodness-of-fit test of the observed arm counts against
# the designed shares.
#
# The next cell downloads the **Hillstrom** email test once (a real test from 2008: 64,000
# customers randomized to a men's email, a women's email or no email; outcomes over the next two
# weeks) and shows its arm counts.

# %%
hill = load_hillstrom()
HILL_ARMS = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
hill_counts = hill["segment"].value_counts().reindex(HILL_ARMS).to_numpy()
print(f"Hillstrom: {len(hill):,} customers")
print(pd.Series(hill_counts, index=HILL_ARMS).to_string())

# %% [markdown]
# **Predict.** Hillstrom assigned one third of customers to each arm. Will the chi-square test of
# these counts give a p-value above or below 0.05?
#
# **Task.** Write `srm_pvalue(counts, expected_shares)` returning the p-value of
# `stats.chisquare(f_obs, f_exp)`. `chisquare` wants expected **counts**, not shares: turn the
# shares into counts that add up to the observed total.

# %% tags=["exercise"]
def srm_pvalue(counts, expected_shares):
    # TODO 2: expected counts = shares / sum(shares) * sum(counts); return the chisquare p-value
    raise NotImplementedError("TODO 2")


# %% tags=["solution"]
# @title Solution 2 — try it yourself first { display-mode: "form" }
@workshop.solution(2)
def srm_pvalue(counts, expected_shares):
    counts = np.asarray(counts, dtype=float)
    shares = np.asarray(expected_shares, dtype=float)
    expected = shares / shares.sum() * counts.sum()
    return float(stats.chisquare(counts, expected).pvalue)


# %% [markdown]
# The checkpoint also uses a deliberately broken copy of the synthetic experiment: a logging bug
# that lost the records of 10% of the treated customers (chosen at random, seed shown).

# %%
def arm_counts(df):
    """Customers in control (treatment == 0) and treatment (treatment == 1), in that order."""
    return df["treatment"].value_counts().reindex([0, 1], fill_value=0).to_numpy()


def lose_treated(df, share, rng):
    """A copy of df without a random `share` of its treated rows (a logging bug)."""
    treated_rows = np.flatnonzero(df["treatment"].to_numpy() == 1)
    lost = rng.choice(treated_rows, size=round(share * len(treated_rows)), replace=False)
    return df.drop(index=df.index[lost])


broken = lose_treated(experiment, 0.10, np.random.default_rng(SEED + 2))
print("Intact split (control, treated):", arm_counts(experiment))
print("Broken split (control, treated):", arm_counts(broken))

# %% tags=["checkpoint"]
with workshop.checkpoint(2):
    p_hill = srm_pvalue(hill_counts, [1, 1, 1])
    reference = stats.chisquare(hill_counts, np.full(3, hill_counts.sum() / 3)).pvalue
    checks.close(p_hill, reference, rel=1e-9, name="srm_pvalue on Hillstrom's arm counts")
    checks.close(srm_pvalue([900, 100], [0.9, 0.1]), 1.0, abs=1e-9,
                 name="srm_pvalue([900, 100], [0.9, 0.1]) (a 90/10 design that came out exactly 90/10)")
    p_broken = srm_pvalue(arm_counts(broken), [0.5, 0.5])
    assert p_broken < 0.001, (
        f"On the broken copy (10% of treated records lost) the SRM p-value is {p_broken:.3g}; it"
        " should be below 0.001. Check that you pass expected COUNTS (shares times the observed"
        " total) to stats.chisquare and return .pvalue, not the statistic. Fix TODO 2, or run"
        " workshop.use_reference(2)."
    )
    print(f"Hillstrom SRM p-value: {p_hill:.3f}; broken copy: {p_broken:.2g}")

# %% [markdown]
# **Explain.** Compare with your prediction. Then run the next cell: it loses only 2% of the
# treated records instead of 10%. Does the check still see it? What does that tell you about
# how large a logging loss a test of 20,000 customers can detect?

# %%
slightly_broken = lose_treated(experiment, 0.02, np.random.default_rng(SEED + 2))
print(f"2% of treated records lost: SRM p-value {srm_pvalue(arm_counts(slightly_broken), [0.5, 0.5]):.2f}")
print(f"Intact synthetic split:     SRM p-value {srm_pvalue(arm_counts(experiment), [0.5, 0.5]):.2f}")

# %% [markdown]
# <details><summary>Why this solution works</summary>
#
# The chi-square statistic adds up (observed − expected)² / expected over the arms. Hillstrom's
# three counts differ from a third of the total by well under 1%, well within chance. The SRM
# check is itself a test with limited power: losing 10% of one arm in 20,000 customers is
# obvious, while losing 2% is not distinguishable from chance at this size.
# Large platforms with millions of users per test detect much smaller losses. Run the check before
# reading any effect: a passing check does not prove the logging is perfect, but a failing one
# means the comparison cannot be trusted.
# </details>

# %% [markdown]
# # Part B · During the test
#
# An **A/A test** gives both arms the same experience, so the true effect is zero and every
# "significant" result is a **false positive**. **Peeking** means checking the p-value while data
# arrive and stopping as soon as it drops below 0.05.

# %% [markdown]
# ## Exercise 3 · What peeking does to false positives (9 minutes)
#
# **Predict.** An A/A test is checked 10 times as data arrive and stopped at the first p < 0.05.
# What share of A/A tests will be declared "significant": about 5%, 10%, 20% or 40%?
#
# **Task.** Write `peeking_fpr(n_per_arm, looks, n_sims, alpha, rng)` returning the share of
# simulated A/A tests that are declared significant at **any** of `looks` equally spaced
# interim looks. For each of `n_sims` tests:
#
# 1. draw `n_per_arm` standard normal outcomes per arm (`rng.standard_normal((n_sims, n_per_arm))`
#    draws all tests at once, one row per test);
# 2. at the look sizes `k = np.linspace(n_per_arm / looks, n_per_arm, looks).round().astype(int)`,
#    compute the difference in means of the first k outcomes (`cumsum` along axis 1, then divide
#    by k) and the z statistic, difference / √(2/k) (the outcomes have variance 1);
# 3. count the test as positive if any look has two-sided p = 2·`stats.norm.sf(|z|)` below `alpha`.
#
# No loops over tests: work on whole arrays.

# %% tags=["exercise"]
def peeking_fpr(n_per_arm, looks, n_sims, alpha, rng):
    # TODO 3: simulate (n_sims, n_per_arm) outcomes per arm, z at each look, any p < alpha
    raise NotImplementedError("TODO 3")


# %% tags=["solution"]
# @title Solution 3 — try it yourself first { display-mode: "form" }
@workshop.solution(3)
def peeking_fpr(n_per_arm, looks, n_sims, alpha, rng):
    sizes = np.linspace(n_per_arm / looks, n_per_arm, looks).round().astype(int)
    sum_a = rng.standard_normal((n_sims, n_per_arm)).cumsum(axis=1)[:, sizes - 1]
    sum_b = rng.standard_normal((n_sims, n_per_arm)).cumsum(axis=1)[:, sizes - 1]
    z = (sum_b - sum_a) / sizes / np.sqrt(2 / sizes)
    p = 2 * stats.norm.sf(np.abs(z))
    return float((p < alpha).any(axis=1).mean())


# %% tags=["checkpoint"]
with workshop.checkpoint(3):
    looks_grid = [1, 2, 5, 10]
    fpr = [peeking_fpr(500, k, N_PEEK_SIMS, 0.05, np.random.default_rng(SEED + 3)) for k in looks_grid]
    checks.probability(fpr, name="peeking_fpr results")
    assert abs(fpr[0] - 0.05) <= PEEK_TOL, (
        f"With one look the false-positive rate is {fpr[0]:.3f}; it should be 0.05 ± {PEEK_TOL}"
        " (no peeking means an ordinary test). Check the z statistic: the difference in MEANS"
        " divided by sqrt(2 / k), and the two-sided p-value 2 * norm.sf(abs(z))."
        " Fix TODO 3, or run workshop.use_reference(3)."
    )
    assert fpr[3] >= 0.10 and fpr[3] > fpr[0], (
        f"With 10 looks the rate is {fpr[3]:.3f}; it should be at least 0.10 and above the"
        f" one-look rate {fpr[0]:.3f}. Check that a test counts as positive if ANY look has"
        " p < alpha (.any(axis=1)), not only the last one."
    )
    checks.monotone(fpr, increasing=True, name="the false-positive rate for 1, 2, 5 and 10 looks")
    print("False-positive rate by number of looks:",
          ", ".join(f"{k}: {f:.3f}" for k, f in zip(looks_grid, fpr)))

# %% [markdown]
# Look at how far the bars climb above the dashed 5% line as the number of looks grows.

# %%
looks_plot = [1, 2, 5, 10, 20]
fpr_plot = [peeking_fpr(500, k, N_PEEK_SIMS, 0.05, np.random.default_rng(SEED + 3)) for k in looks_plot]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar([str(k) for k in looks_plot], fpr_plot, color="#4c72b0")
ax.axhline(0.05, color="black", linestyle="--", label="α = 0.05 (one look)")
for i, v in enumerate(fpr_plot):
    ax.text(i, v + 0.005, f"{v:.0%}", ha="center")
ax.set_xlabel("Number of looks at the data (stop at the first p < 0.05)")
ax.set_ylabel("Share of A/A tests\ndeclared significant")
ax.set_title(f"Peeking inflates false positives ({N_PEEK_SIMS:,} simulated A/A tests)")
ax.legend(loc="upper left")
fig.tight_layout()

# %% [markdown]
# **Explain.** Compare with your guess. Why does each extra look add false positives, even though
# every single look uses α = 0.05? Point to the bars.
#
# <details><summary>Why this solution works</summary>
#
# Each look is another chance for noise to cross the line, and a test that stops at the first
# crossing keeps every lucky crossing. The looks are correlated (later looks reuse earlier data),
# so the rate grows more slowly than 1 − 0.95^looks, but it keeps growing. Three fixes: fix the
# sample size in advance and look once; use a group-sequential design with a stricter threshold
# at each look (the stretch finds one by simulation); or use always-valid inference, built for
# continuous monitoring (Johari et al. 2022).
# </details>

# %% [markdown]
# # Part C · Reading the test
#
# **CUPED** (controlled experiments using pre-experiment data; Deng et al. 2013) subtracts from
# each customer's outcome the part that was predictable before the test. With pre-period
# covariate x (here: last year's spend, `history`) and outcome y:
#
# $$ \theta = \frac{\operatorname{cov}(y, x)}{\operatorname{var}(x)}, \qquad
#    y^{\text{adj}} = y - \theta\,(x - \bar{x}), $$
#
# with θ computed on both arms pooled. Because x was fixed before randomization, it is balanced
# across arms on average, so the adjustment cannot bias the comparison; it can only remove noise.
# The share of variance removed is ρ², where ρ is the correlation between y and x.

# %%
rho = experiment["spend"].corr(experiment["history"])
print(f"Correlation between last year's spend and spend in the test: rho = {rho:.3f}")

# %% [markdown]
# ## Exercise 4 · CUPED (7 minutes)
#
# **Predict.** The correlation ρ is printed above. Will CUPED cut the variance of the spend
# outcome by about ρ, by about ρ², or not at all? Write the percentage you expect.
#
# **Task.** Write `cuped(y, x, treatment)` returning a dict with
#
# - `"theta"`: θ on the pooled data (`np.cov(y, x)[0, 1] / np.var(x, ddof=1)`);
# - `"raw"` and `"adjusted"`: each a dict with `"effect"` (mean of treated minus mean of control),
#   `"se"` (√(s₁²/n₁ + s₀²/n₀)) and `"ci"` (a tuple: effect ± `stats.norm.ppf(0.975)` × se), for
#   y and for y_adj;
# - `"variance_reduction"`: 1 − var(y_adj) / var(y).

# %% tags=["exercise"]
def cuped(y, x, treatment):
    # TODO 4: theta, y_adj, then effect / se / 95% CI for y and for y_adj
    raise NotImplementedError("TODO 4")


# %% tags=["solution"]
# @title Solution 4 — try it yourself first { display-mode: "form" }
@workshop.solution(4)
def cuped(y, x, treatment):
    y, x, t = (np.asarray(v, dtype=float) for v in (y, x, treatment))
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    z = stats.norm.ppf(0.975)
    out = {"theta": float(theta),
           "variance_reduction": float(1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1))}
    for name, v in (("raw", y), ("adjusted", y_adj)):
        effect = v[t == 1].mean() - v[t == 0].mean()
        se = np.sqrt(v[t == 1].var(ddof=1) / (t == 1).sum() + v[t == 0].var(ddof=1) / (t == 0).sum())
        out[name] = {"effect": float(effect), "se": float(se), "ci": (effect - z * se, effect + z * se)}
    return out


# %% [markdown]
# The checkpoint runs `cuped` on three datasets: the synthetic experiment (true ATE known), a
# simulated metric whose pre-period value predicts it well (ρ = 0.8, seed shown; think of weekly
# site visits, which customers repeat), and Hillstrom (women's email against no email; the true
# effect is unknown, so only structure is checked).

# %%
rng_toy = np.random.default_rng(SEED + 4)
toy_x = rng_toy.normal(100, 20, 20_000)  # last period's value of the metric
toy_t = rng_toy.integers(0, 2, 20_000)  # randomized 50/50
toy_y = 0.4 * toy_x + rng_toy.normal(0, 6, 20_000) + 1.0 * toy_t  # rho = 0.8, true effect 1.0
hill_wc = hill[hill["segment"].isin(["Womens E-Mail", "No E-Mail"])]
hill_wc_t = (hill_wc["segment"] == "Womens E-Mail").astype(int)

# %% tags=["checkpoint"]
with workshop.checkpoint(4):
    res_syn = cuped(experiment["spend"], experiment["history"], experiment["treatment"])
    checks.columns(pd.DataFrame([res_syn]), ["theta", "raw", "adjusted", "variance_reduction"],
                   name="the dict cuped returns")
    lo, hi = res_syn["adjusted"]["ci"]
    assert lo <= truth["ate"] <= hi, (
        f"The true ATE, ${truth['ate']:.2f}, is outside your CUPED-adjusted 95% CI"
        f" [{lo:.2f}, {hi:.2f}]. Check that the effect is treated minus control on y_adj, that"
        " theta uses both arms pooled, and that x is centered (x - x.mean()). Fix TODO 4, or run"
        " workshop.use_reference(4)."
    )
    raw_w = np.ptp(res_syn["raw"]["ci"])
    adj_w = np.ptp(res_syn["adjusted"]["ci"])
    assert adj_w < raw_w, (
        f"The adjusted CI ({adj_w:.3f} wide) is not narrower than the raw CI ({raw_w:.3f})."
        " Check the sign: y_adj = y - theta * (x - x.mean())."
    )
    for label, (yy, xx, tt) in {
        "the synthetic experiment": (experiment["spend"], experiment["history"], experiment["treatment"]),
        "the rho = 0.8 metric": (toy_y, toy_x, toy_t),
    }.items():
        got = cuped(yy, xx, tt)["variance_reduction"]
        checks.close(got, np.corrcoef(yy, xx)[0, 1] ** 2, abs=0.01,
                     name=f"the variance reduction on {label} (should equal rho squared)")
    red_hill = cuped(hill_wc["spend"], hill_wc["history"], hill_wc_t)["variance_reduction"]
    checks.probability(red_hill, name="the variance reduction on Hillstrom")

# %% [markdown]
# Compare the interval widths without and with CUPED on the three datasets.

# %%
rows = []
for label, (yy, xx, tt) in {
    "synthetic experiment (spend)": (experiment["spend"], experiment["history"], experiment["treatment"]),
    "metric with rho = 0.8 (simulated)": (toy_y, toy_x, toy_t),
    "Hillstrom: women's email vs none (spend)": (hill_wc["spend"], hill_wc["history"], hill_wc_t),
}.items():
    r = cuped(yy, xx, tt)
    rows.append({"data": label, "rho": np.corrcoef(yy, xx)[0, 1],
                 "raw effect": r["raw"]["effect"], "raw 95% CI width": np.ptp(r["raw"]["ci"]),
                 "CUPED effect": r["adjusted"]["effect"], "CUPED 95% CI width": np.ptp(r["adjusted"]["ci"]),
                 "variance reduction": r["variance_reduction"]})
cuped_table = pd.DataFrame(rows).set_index("data")
cuped_table.round(3)

# %% [markdown]
# **Explain.** Compare with your prediction. Why does CUPED barely help on spend here, when Deng
# et al. report variance reductions of about 50% at Bing? Point to the ρ column.
#
# <details><summary>Why this solution works</summary>
#
# CUPED removes the share ρ² of the outcome's variance that the pre-period covariate predicts, and
# interval widths shrink by the factor √(1 − ρ²). Two-week spend is mostly zeros with a few large
# orders: whether a customer buys in these two weeks is mostly chance, so last year's spend
# predicts little of it (small ρ, tiny ρ²). Metrics that customers repeat steadily, such as visits
# or sessions, have large ρ and gain a lot. A better covariate (for example, spend in the same
# weeks last year, or a model's prediction from several pre-period variables) raises ρ; a
# post-period variable must never be used, because the treatment can move it.
# </details>

# %% [markdown]
# A **ratio metric** divides two sums, such as revenue per session = total revenue ÷ total
# sessions. We randomize customers, but one customer contributes many sessions, and sessions of the
# same customer resemble each other. The next cell simulates the treated arm of such a test
# (seed shown): 10,000 customers, each with a random number of sessions and a typical revenue per
# session of their own.

# %%
rng_sessions = np.random.default_rng(SEED + 5)
n_customers = 10_000
sessions = 1 + rng_sessions.poisson(rng_sessions.gamma(0.8, 3.0, n_customers))  # sessions per customer
typical = rng_sessions.lognormal(1.0, 0.8, n_customers)  # customer's typical revenue per session
session_customer = np.repeat(np.arange(n_customers), sessions)
session_revenue = typical[session_customer] * rng_sessions.gamma(2.0, 0.5, session_customer.size)
revenue = np.bincount(session_customer, weights=session_revenue, minlength=n_customers)
naive_se = session_revenue.std(ddof=1) / np.sqrt(session_revenue.size)  # sessions as independent
print(f"{n_customers:,} customers, {session_revenue.size:,} sessions;"
      f" revenue per session ${revenue.sum() / sessions.sum():.3f}")
print(f"Naive standard error (every session independent): {naive_se:.4f}")

# %% [markdown]
# ## Exercise 5 · Delta-method standard error for a ratio metric (7 minutes)
#
# **Predict.** Will the correct standard error of revenue per session be larger or smaller than
# the naive one printed above, which treats every session as an independent observation?
#
# **Task.** Write `ratio_delta_se(num, den)` returning a tuple `(R, se)`: the ratio
# R = Σnum / Σden and its **delta-method** standard error from per-customer numerators (revenue)
# and denominators (sessions). The *delta method* approximates the variance of a function of
# averages, here a ratio, by replacing the function with a straight line around the means.
# With n customers, means $\bar{n}$ and $\bar{d}$, and
# Var($\bar{n}$) = var(num)/n, Var($\bar{d}$) = var(den)/n, Cov($\bar{n}$, $\bar{d}$) = cov(num, den)/n:
#
# $$ \operatorname{Var}(R) \approx \frac{\operatorname{Var}(\bar{n}) - 2R\,\operatorname{Cov}(\bar{n}, \bar{d})
#    + R^2 \operatorname{Var}(\bar{d})}{\bar{d}^{\,2}}. $$

# %% tags=["exercise"]
def ratio_delta_se(num, den):
    # TODO 5: R = sum(num) / sum(den); the variances and covariance of the means; the formula above
    raise NotImplementedError("TODO 5")


# %% tags=["solution"]
# @title Solution 5 — try it yourself first { display-mode: "form" }
@workshop.solution(5)
def ratio_delta_se(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    n = len(num)
    ratio = num.sum() / den.sum()
    var_n, var_d = num.var(ddof=1) / n, den.var(ddof=1) / n
    cov_nd = np.cov(num, den)[0, 1] / n
    var_ratio = (var_n - 2 * ratio * cov_nd + ratio**2 * var_d) / den.mean() ** 2
    return float(ratio), float(np.sqrt(var_ratio))


# %% [markdown]
# The checkpoint compares your standard error with a **bootstrap** one: resample customers with
# replacement many times, recompute the ratio each time, and take the standard deviation of the
# ratios. The bootstrap needs no formula but takes much more computation.

# %%
def bootstrap_ratio_se(num, den, n_boot, rng, chunk=250):
    """Standard deviation of Σnum/Σden over n_boot resamples of customers (rows)."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    ratios = []
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, len(num), size=(min(chunk, n_boot - start), len(num)))
        ratios.append(num[idx].sum(axis=1) / den[idx].sum(axis=1))
    return float(np.concatenate(ratios).std(ddof=1))


boot_se = bootstrap_ratio_se(revenue, sessions, N_BOOT, np.random.default_rng(SEED + 5))
print(f"Bootstrap standard error ({N_BOOT:,} resamples of customers): {boot_se:.4f}")

# %% tags=["checkpoint"]
with workshop.checkpoint(5):
    ratio, delta_se = ratio_delta_se(revenue, sessions)
    checks.close(ratio, revenue.sum() / sessions.sum(), rel=1e-9, name="the ratio R")
    assert abs(delta_se / boot_se - 1) <= BOOT_TOL, (
        f"Your delta-method SE is {delta_se:.4f}; the bootstrap SE is {boot_se:.4f}. They should"
        f" agree within {BOOT_TOL:.0%}. Check the minus sign on the covariance term, that the"
        " variances are of the MEANS (divide by n), and the division by mean(den) squared."
        " Fix TODO 5, or run workshop.use_reference(5)."
    )
    toy_num = np.array([3.0, 5.0, 4.0, 8.0, 6.0])
    toy_den = np.full(5, 2.0)  # every customer has 2 sessions
    _, toy_se = ratio_delta_se(toy_num, toy_den)
    checks.close(toy_se, (toy_num / toy_den).std(ddof=1) / np.sqrt(5), rel=1e-9,
                 name="the delta-method SE when every denominator is 2 (should equal the ordinary SE)")
    print(f"R = {ratio:.3f}; delta-method SE {delta_se:.4f}; bootstrap {boot_se:.4f}; naive {naive_se:.4f}")

# %% [markdown]
# Now the same function on Hillstrom: spend per visit in the women's-email arm.

# %%
womens = hill[hill["segment"] == "Womens E-Mail"]
r_hill, se_hill = ratio_delta_se(womens["spend"], womens["visit"])
visits = womens.loc[womens["visit"] > 0, "spend"]
print(f"Hillstrom spend per visit: ${r_hill:.2f}; delta-method SE {se_hill:.3f};"
      f" naive SE {visits.std(ddof=1) / np.sqrt(len(visits)):.3f}")
print(f"Visits per customer in the data: {sorted(womens['visit'].unique().tolist())}")

# %% [markdown]
# **Explain.** Compare with your prediction. In the simulated test the correct standard error is
# far from the naive one, while on Hillstrom the two agree. What is different about Hillstrom's
# `visit` column? Point to the last printed line.
#
# <details><summary>Why this solution works</summary>
#
# Sessions of the same customer are correlated (a big spender is big in every session), so the
# sessions carry less information than as many independent observations; the naive standard error
# pretends otherwise and is too small, which makes intervals too narrow and false positives more
# common. The delta method linearizes the ratio around the means and uses the
# customer-level variances and covariance, so it respects the unit of randomization (Deng,
# Knoblich & Lu 2018). Hillstrom records only whether a customer visited (0 or 1), so each
# customer contributes at most one visit and the naive and delta-method errors coincide.
# </details>

# %% [markdown]
# # Part D · What the test measures
#
# **Last-touch attribution** credits each conversion to the last marketing touch before it. In
# the email test, every conversion of an emailed customer would be credited to the email.
# **Incremental conversions** are the ones the email caused: conversions among emailed customers
# minus the conversions they would have made without it, estimated from the control arm.

# %% [markdown]
# ## Exercise 6 · Attribution is not incrementality (5 minutes)
#
# **Predict.** A last-touch report credits every conversion by an emailed customer to the email.
# Will that be more than, fewer than, or about the number of conversions the email caused?
#
# **Task.** Write `attributed_vs_incremental(df)` for the synthetic experiment (columns
# `treatment`, `conversion`) returning a dict with
#
# - `"attributed"`: conversions among treated customers;
# - `"incremental"`: (conversion rate treated − conversion rate control) × number treated;
# - `"ci"`: the 95% CI of `"incremental"` as a tuple, from the standard error of the difference in
#   rates, √(p₁(1 − p₁)/n₁ + p₀(1 − p₀)/n₀), times the number treated.

# %% tags=["exercise"]
def attributed_vs_incremental(df):
    # TODO 6: attributed conversions, incremental conversions and its 95% CI
    raise NotImplementedError("TODO 6")


# %% tags=["solution"]
# @title Solution 6 — try it yourself first { display-mode: "form" }
@workshop.solution(6)
def attributed_vs_incremental(df):
    treated = df.loc[df["treatment"] == 1, "conversion"]
    control = df.loc[df["treatment"] == 0, "conversion"]
    p1, p0, n1, n0 = treated.mean(), control.mean(), len(treated), len(control)
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    z = stats.norm.ppf(0.975)
    return {"attributed": int(treated.sum()),
            "incremental": float((p1 - p0) * n1),
            "ci": ((p1 - p0 - z * se) * n1, (p1 - p0 + z * se) * n1)}


# %% tags=["checkpoint"]
with workshop.checkpoint(6):
    att = attributed_vs_incremental(experiment)
    n_treated = int((experiment["treatment"] == 1).sum())
    true_incremental = (truth["conversion_treated"] - truth["base_conversion_control"]) * n_treated
    lo, hi = att["ci"]
    assert lo <= true_incremental <= hi, (
        f"The true number of incremental conversions, {true_incremental:.0f}, is outside your 95% CI"
        f" [{lo:.0f}, {hi:.0f}]. Check that the rates are means of `conversion` within each arm and"
        " that the difference is multiplied by the number TREATED. Fix TODO 6, or run"
        " workshop.use_reference(6)."
    )
    checks.close(att["attributed"], int(experiment.loc[experiment["treatment"] == 1, "conversion"].sum()),
                 abs=0, name="attributed conversions (all conversions among treated customers)")
    assert att["attributed"] > hi, (
        f"Attributed conversions ({att['attributed']}) should exceed the upper end of the CI for"
        f" incremental conversions ({hi:.0f}). Check the two definitions."
    )
    print(f"Attributed: {att['attributed']}; incremental: {att['incremental']:.0f}"
          f" (95% CI {lo:.0f} to {hi:.0f}); true incremental {true_incremental:.0f}")

# %% [markdown]
# **Explain.** Compare with your prediction. What share of the conversions the last-touch report
# credits to the email would have happened anyway? Which customers are they?
#
# <details><summary>Why this solution works</summary>
#
# Attribution counts who was touched before converting; incrementality counts who converted
# because of the touch. Customers who would have bought anyway still get the email, so
# touch-based credit includes them. Here the control arm converts at a few percent without any
# email, so a large share of the credited conversions are not caused by it. Large field
# experiments at Facebook found that observational methods often failed to reproduce the lift that
# randomized tests measured (Gordon et al. 2019). Module 7 measures incrementality when customers
# cannot be randomized one by one.
# </details>

# %% [markdown]
# ## Decision · Roll out the email?
#
# The rule, fixed before reading the result: **roll out if the SRM check passes (p > 0.001) and
# the lower end of the 95% CI of incremental margin per email is above 0.** That is stricter than
# "the point estimate is positive", because a rollout is hard to reverse and a wrong rollout
# costs money on every email. Incremental margin per email = gross margin × incremental revenue
# per customer − offer cost per email. The margin and offer cost below are assumptions recorded
# with the synthetic data.

# %%
MARGIN = truth["margin"]  # gross margin on spend (assumption)
OFFER_COST = truth["offer_cost"]  # cost per email sent, $ (assumption)
readout = cuped(experiment["spend"], experiment["history"], experiment["treatment"])
effect = readout["adjusted"]["effect"]
effect_lo, effect_hi = readout["adjusted"]["ci"]
srm_p = srm_pvalue(arm_counts(experiment), [0.5, 0.5])
per_email = MARGIN * effect - OFFER_COST
per_email_lo, per_email_hi = MARGIN * effect_lo - OFFER_COST, MARGIN * effect_hi - OFFER_COST
roll_out = srm_p > 0.001 and per_email_lo > 0

print("Number")
print(f"  SRM check p-value: {srm_p:.2f}")
print(f"  Incremental revenue per customer (CUPED): ${effect:.2f}, 95% CI ${effect_lo:.2f} to ${effect_hi:.2f}")
print(f"  Incremental margin per email: {MARGIN:.0%} x ${effect:.2f} - ${OFFER_COST:.2f} = ${per_email:.2f}"
      f" (95% CI ${per_email_lo:.2f} to ${per_email_hi:.2f})")
print("Rule")
print("  Roll out if SRM p > 0.001 and the lower end of the 95% CI of margin per email > $0.")
print("Recommendation")
if roll_out:
    print(f"  Roll out. Per 100,000 emails: about ${100_000 * per_email:,.0f} of incremental margin"
          f" (95% CI ${100_000 * per_email_lo:,.0f} to ${100_000 * per_email_hi:,.0f}).")
else:
    print("  Do not roll out yet: the check failed or the interval reaches $0 or below.")
print(f"  Mode: {'QUICK run: treat this as a rough answer' if QUICK else 'FULL run'}"
      " (this readout has no simulation, so QUICK does not change it).")

# %% [markdown]
# **Decide.** Write one sentence: roll out or not, the money per 100,000 emails with its range,
# and what would change your mind (a cheaper or more expensive offer; a segment where the effect
# is much larger or negative, which Module 11 finds).
#
# Then run the next cell, which reveals the truth the data were generated from.

# %%
true_per_email = MARGIN * truth["ate"] - OFFER_COST
print(f"True ATE: ${truth['ate']:.2f} per customer; inside your 95% CI: {effect_lo <= truth['ate'] <= effect_hi}")
print(f"True incremental margin per email: ${true_per_email:.2f}"
      f" (${100_000 * true_per_email:,.0f} per 100,000 emails)")

# %% [markdown]
# ## Stretch (optional) · A stricter threshold for ten looks
#
# Exercise 3 showed that ten looks at |z| > 1.96 give far more than 5% false positives. Find by
# simulation the constant threshold c such that stopping at the first look with |z| > c keeps the
# false-positive rate at 5%: simulate A/A tests, take for each test the largest |z| over the ten
# looks, and set c to the 95th percentile of those maxima. Then rerun the readout on Hillstrom,
# men's against women's email, with the SRM check, CUPED and the delta method.

# %% tags=["solution"]
# @title Stretch solution — try it yourself first { display-mode: "form" }
rng_stretch = np.random.default_rng(SEED + 7)
sizes = np.linspace(50, 500, 10).round().astype(int)
sum_a = rng_stretch.standard_normal((N_PEEK_SIMS, 500)).cumsum(axis=1)[:, sizes - 1]
sum_b = rng_stretch.standard_normal((N_PEEK_SIMS, 500)).cumsum(axis=1)[:, sizes - 1]
max_abs_z = np.abs((sum_b - sum_a) / sizes / np.sqrt(2 / sizes)).max(axis=1)
c = np.quantile(max_abs_z, 0.95)
print(f"Constant threshold for 10 looks: |z| > {c:.2f} (instead of 1.96);"
      f" nominal p-value per look {2 * stats.norm.sf(c):.4f}")
