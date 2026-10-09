"""One channel vocabulary for the retailer story: media channels, acquisition channels, the link.

The retailer's customers carry an **acquisition channel** (``retailer``: search, social, referral —
how the customer first arrived, as the CRM records it). The marketing-mix model has **media
channels** (``mmm``: tv, search, social, display — where the money is spent). The two lists share
two names but are different things: "search" spend buys clicks; a "search" acquisition is a
customer whose first visit came from a search engine, whichever ad made them search. (The e-mail
experiment's ``channel`` column is a third thing: Hillstrom's shopping channel, Phone / Web /
Multichannel.)

The link is an explicit assumption of the synthetic world, stated here once and written to
``truth.json`` (block ``channel_value``):

* ``MEDIA_TO_ACQUISITION[m][a]``: of the new customers that media channel ``m`` brings, the share
  recorded under acquisition channel ``a`` (rows sum to 1). TV has no click: viewers arrive by
  searching for the brand or through word of mouth (referral). Paid search clicks are recorded as
  search. Social ads bring social sign-ups and some shares to friends (referral). Display
  impressions lead to later brand searches or to social profiles.
* ``COST_PER_NEW_CUSTOMER[m]``: dollars of media spend per new customer acquired, constant over the
  allocation bounds (so new customers per dollar = 1 / cost, the ``new_customers_per_dollar`` the
  Module 10 and capstone briefs ask for).

A new customer's long-run value is the discounted **margin on their repeat purchases** (the
retailer truth's ``discounted_clv_excluding_first_purchase`` times the gross margin): the first
purchase is part of the media channel's short-run sales, which the MMM already counts, so adding it
again would count it twice. The value of a media channel's new customer is the
``MEDIA_TO_ACQUISITION``-weighted average over acquisition channels.

The long-run weekly objective of a steady weekly plan ``x`` is

    margin * sum_m response_m(x_m)  +  sum_m x_m / COST_PER_NEW_CUSTOMER[m] * clv_margin[m]

(short-run margin on media-driven sales plus the future margin of the customers acquired). The
first term is concave and the second linear, so the optimum within bounds is unique and is found
exactly by :func:`mktstats.synth.mmm.optimal_allocation_value`.
"""

from __future__ import annotations

import numpy as np

from mktstats.synth._common import sig
from mktstats.synth.btyd import RETAILER_CHANNELS
from mktstats.synth.mmm import (
    MMM_CHANNELS,
    long_run_value,
    optimal_allocation_value,
    steady_state_response,
)

MEDIA_CHANNELS = MMM_CHANNELS
ACQUISITION_CHANNELS = RETAILER_CHANNELS
MEDIA_TO_ACQUISITION = {
    "tv": {"search": 0.3, "social": 0.0, "referral": 0.7},
    "search": {"search": 1.0, "social": 0.0, "referral": 0.0},
    "social": {"search": 0.0, "social": 0.9, "referral": 0.1},
    "display": {"search": 0.6, "social": 0.4, "referral": 0.0},
}
COST_PER_NEW_CUSTOMER = {"tv": 150.0, "search": 150.0, "social": 100.0, "display": 300.0}
GROSS_MARGIN = 0.30
WEEKS_PER_MONTH = 30.4375 / 7  # Module 5's month (365.25 / 12 days) in weeks


def pnbd_expected_purchases_new(t, r: float, alpha: float, s: float, beta: float):
    """Expected repeat purchases in ``[0, t]`` of a new Pareto/NBD customer:
    ``(r / alpha) * int_0^t (beta / (beta + u))**s du`` (purchase rate r / alpha times the
    probability of being alive at u), in closed form ``(r / alpha) * beta / (s - 1) *
    (1 - (beta / (beta + t))**(s - 1))``, or ``(r / alpha) * beta * log(1 + t / beta)`` for
    ``s = 1``. Rates per week, ``t`` in weeks."""
    t = np.asarray(t, dtype=float)
    if np.isclose(s, 1.0):
        return r / alpha * beta * np.log1p(t / beta)
    return r / alpha * beta / (s - 1) * (1 - (beta / (beta + t)) ** (s - 1))


def new_customer_value_monthly(r: float, alpha: float, s: float, beta: float, mean_spend: float,
                               months: int, monthly_rate: float,
                               weeks_per_month: float = WEEKS_PER_MONTH) -> dict[str, float]:
    """Revenue value of a newly acquired customer in Module 5's convention: the acquisition purchase
    (month 0, not discounted) plus, for k = 1..months, mean spend times the expected repeat
    purchases in month k discounted by ``(1 + monthly_rate)**k``. Rates are per week."""
    t = np.arange(months + 1) * weeks_per_month
    cum = pnbd_expected_purchases_new(t, r, alpha, s, beta)
    per_month = np.diff(cum)
    disc = (1 + monthly_rate) ** -np.arange(1, months + 1)
    repeat_value = float(mean_spend * (per_month * disc).sum())
    return {"expected_repeat_purchases": float(cum[-1]),
            "value_excluding_first_purchase": repeat_value,
            "value_including_first_purchase": float(mean_spend) + repeat_value}


def media_clv(clv_by_acquisition: dict[str, float],
              mapping: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    """Value of a media channel's new customer: the mapping-weighted average over acquisition
    channels."""
    mapping = MEDIA_TO_ACQUISITION if mapping is None else mapping
    return {m: float(sum(w * clv_by_acquisition[a] for a, w in shares.items()))
            for m, shares in mapping.items()}


def long_run_allocation(mmm_truth: dict, clv_margin_by_media: dict[str, float], *,
                        margin: float = GROSS_MARGIN,
                        cost_per_new_customer: dict[str, float] | None = None) -> dict:
    """True optimal steady weekly allocation under the long-run objective (module docstring), with
    the budget, bounds and current plan of ``mmm_truth["true_optimal_allocation"]``."""
    cost = COST_PER_NEW_CUSTOMER if cost_per_new_customer is None else cost_per_new_customer
    ch = list(mmm_truth["channels"])
    al = mmm_truth["true_optimal_allocation"]
    bb = np.array([mmm_truth["channels"][c]["saturation_beta_sales_units"] for c in ch])
    lam = np.array([mmm_truth["channels"][c]["saturation_lam"] for c in ch])
    sc = np.array([mmm_truth["channels"][c]["channel_scale"] for c in ch])
    lo = np.array([al["bounds"][c][0] for c in ch])
    hi = np.array([al["bounds"][c][1] for c in ch])
    cur = np.array([al["current_allocation"][c] for c in ch])
    short = np.array([al["optimal_allocation"][c] for c in ch])
    budget = float(al["weekly_budget"])
    ncpd = np.array([1.0 / cost[c] for c in ch])
    lin = ncpd * np.array([clv_margin_by_media[c] for c in ch])
    x, eta = optimal_allocation_value(budget, bb, lam, sc, lo, hi, margin=margin, linear=lin)

    def value(plan):
        return float(long_run_value(plan, bb, lam, sc, margin=margin, linear=lin).sum())

    marginal = (margin * bb * lam / (2 * sc) * (1 - np.tanh(lam * x / (2 * sc)) ** 2) + lin)
    v_cur, v_short, v_opt = value(cur), value(short), value(x)
    return {
        "definition": (
            "Steady weekly plan maximizing margin * sum_c B_c * logistic(lam_c x_c / S_c) + "
            "sum_c x_c * new_customers_per_dollar_c * clv_margin_c (one week's short-run margin "
            "plus the discounted future margin of the customers that week's spend acquires), "
            "subject to sum(x_c) = weekly_budget and the bounds of true_optimal_allocation. "
            "Concave, so unique; solved exactly by bisection on the common marginal value "
            "(mktstats.synth.mmm.optimal_allocation_value)."
        ),
        "margin": margin,
        "weekly_budget": budget,
        "bounds": al["bounds"],
        "current_allocation": al["current_allocation"],
        "short_run_optimal_allocation": al["optimal_allocation"],
        "long_run_value_per_dollar_of_new_customers": {c: sig(lin[j]) for j, c in enumerate(ch)},
        "optimal_allocation": {c: sig(x[j]) for j, c in enumerate(ch)},
        "optimal_share": {c: sig(x[j] / budget) for j, c in enumerate(ch)},
        "change_vs_short_run_optimum": {c: sig(x[j] - short[j]) for j, c in enumerate(ch)},
        "weekly_value_current": sig(v_cur),
        "weekly_value_short_run_optimum": sig(v_short),
        "weekly_value_optimal": sig(v_opt),
        "weekly_short_run_margin_at_optimum": sig(
            margin * steady_state_response(x, bb, lam, sc).sum()),
        "weekly_new_customers_at_optimum": sig(float((x * ncpd).sum())),
        "marginal_value_at_optimum": {c: sig(marginal[j]) for j, c in enumerate(ch)},
        "lagrange_multiplier": sig(eta),
        "at_bound": {c: ("lower" if np.isclose(x[j], lo[j]) else
                         "upper" if np.isclose(x[j], hi[j]) else "interior")
                     for j, c in enumerate(ch)},
        "value_sensitivity": {
            "note": (
                "How flat the objective is near its optimum: values of other plans as a share of "
                "the optimal value, and the share of the optimal gain over the current plan that "
                "each captures. A checkpoint on value alone cannot tell these plans apart unless "
                "its tolerance is below these gaps."
            ),
            "short_run_optimum_share_of_optimal_value": sig(v_short / v_opt),
            "current_share_of_optimal_value": sig(v_cur / v_opt),
            "short_run_optimum_share_of_optimal_gain": sig((v_short - v_cur) / (v_opt - v_cur)),
        },
    }


def channel_value(retailer_truth: dict, mmm_truths: dict[str, dict], *,
                  margin: float = GROSS_MARGIN,
                  clv_key: str = "discounted_clv_excluding_first_purchase") -> dict:
    """The ``channel_value`` block of ``truth.json``: the vocabulary, the mapping, the cost per new
    customer, the value of a new customer by acquisition and media channel (margin, from
    ``retailer_truth["value_by_channel"]["new_customer"]``) and, for each MMM truth in
    ``mmm_truths`` (e.g. ``{"mmm": ..., "mmm_confounded": ...}``), the long-run optimal allocation.
    """
    vbc = retailer_truth["value_by_channel"]
    new = vbc["new_customer"]
    clv_acq = {a: margin * new[a][clv_key] for a in ACQUISITION_CHANNELS}
    clv_media = media_clv(clv_acq)
    return {
        "definition": (
            "Shared channel vocabulary of the retailer story. Media channels (where money is "
            "spent, the MMM's columns) and acquisition channels (how a customer first arrived, the "
            "retailer's acquisition_channel) are different lists; media_to_acquisition gives, for "
            "each media channel, the shares of its new customers recorded under each acquisition "
            "channel. cost_per_new_customer is media dollars per new customer, constant within "
            "the allocation bounds (an assumption of the synthetic world). clv_margin = margin * "
            f"retailer value_by_channel.new_customer.{clv_key} ({vbc['horizon_weeks']} weeks, "
            f"{vbc['annual_discount_rate']:.0%} a year): the first purchase is excluded because "
            "it is already in the MMM's short-run sales. See mktstats.synth.channels."
        ),
        "media_channels": list(MEDIA_CHANNELS),
        "acquisition_channels": list(ACQUISITION_CHANNELS),
        "email_channel_column": (
            "the e-mail experiment's `channel` (Phone / Web / Multichannel) is the shopping "
            "channel, unrelated to both lists"
        ),
        "media_to_acquisition": MEDIA_TO_ACQUISITION,
        "cost_per_new_customer": COST_PER_NEW_CUSTOMER,
        "new_customers_per_dollar": {m: sig(1 / c) for m, c in COST_PER_NEW_CUSTOMER.items()},
        "margin": margin,
        "clv_source": f"retailer.value_by_channel.new_customer.*.{clv_key}",
        "clv_margin_by_acquisition_channel": {a: sig(v) for a, v in clv_acq.items()},
        "clv_margin_by_media_channel": {m: sig(v) for m, v in clv_media.items()},
        "long_run_optimal_allocation": {
            name: long_run_allocation(t, clv_media, margin=margin) for name, t in mmm_truths.items()
        },
    }
