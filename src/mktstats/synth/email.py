"""Hillstrom-like randomized e-mail offer with heterogeneous effects and a known per-row CATE.

Covariates mimic the MineThatData e-mail challenge (Hillstrom 2008): recency (months since last
purchase, 1-12), history (dollars spent last year) and its segment label, mens / womens (bought in
that category), zip_code (Urban / Surburban / Rural, Hillstrom's spelling kept so code ports to the
real file), newbie, channel (Phone / Web / Multichannel). Marginals are close to the real file's.
Conversion rates are scaled up (about 3% without the e-mail instead of Hillstrom's 0.6%) so that
heterogeneity is learnable from 20,000 rows.

Outcome (zero-inflated spend):

    P(purchase | x, T) = expit(f(x) + T * tau(x))
    spend | purchase, x ~ Gamma(shape=k, mean=m(x))         (independent of T)

so the true conditional average treatment effect on expected spend is

    true_cate(x) = (expit(f(x) + tau(x)) - expit(f(x))) * m(x).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from mktstats.synth._common import SynthResult, make_rng, sig

RECENCY_PROBS = np.array([0.140, 0.118, 0.092, 0.079, 0.070, 0.072,
                          0.064, 0.055, 0.101, 0.118, 0.055, 0.036])
HISTORY_BINS = [0, 100, 200, 350, 500, 750, 1000, np.inf]
HISTORY_LABELS = ["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
                  "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"]


def _f_tau_m(recency, history, mens, womens, zip_code, newbie, channel):
    lh = np.log(history / 150.0)
    web = channel == "Web"
    multi = channel == "Multichannel"
    rural = zip_code == "Rural"
    f = (-3.45 - 0.08 * (recency - 6) + 0.3 * lh + 0.2 * womens + 0.1 * mens
         - 0.3 * newbie + 0.15 * multi + 0.1 * rural)
    tau = (0.25 + 0.45 * womens - 0.35 * mens * (1 - womens) - 0.06 * (recency - 6)
           + 0.25 * web - 0.4 * newbie * (recency > 8))
    m = 80.0 * (history / 150.0) ** 0.3 * (1 + 0.2 * womens)
    return f, tau, m


def email_experiment(
    seed: int = 2030,
    *,
    n: int = 20_000,
    treat_prob: float = 0.5,
    spend_shape: float = 2.0,
    margin: float = 0.35,
    offer_cost: float = 0.5,
) -> SynthResult:
    """Randomized e-mail offer; spend outcome with a known per-row ``true_cate``.

    Tables
    ------
    experiment : customer_id, recency, history_segment, history, mens, womens, zip_code, newbie,
        channel, treatment (1 = got the e-mail offer), conversion, spend, true_cate.

    ``margin`` (gross margin on spend) and ``offer_cost`` (cost per customer sent the offer) are the
    assumptions of the targeting decision: treat when ``cate * margin > offer_cost``.
    """
    rng = make_rng(seed)
    recency = rng.choice(np.arange(1, 13), size=n, p=RECENCY_PROBS / RECENCY_PROBS.sum())
    history = np.round(np.maximum(29.99, np.exp(rng.normal(5.0, 1.0, n))), 2)
    cat = rng.choice(3, size=n, p=[0.45, 0.45, 0.10])  # mens only, womens only, both
    mens = ((cat == 0) | (cat == 2)).astype(np.int64)
    womens = ((cat == 1) | (cat == 2)).astype(np.int64)
    zip_code = rng.choice(np.array(["Urban", "Surburban", "Rural"], dtype=object), size=n,
                          p=[0.40, 0.45, 0.15])
    newbie = rng.binomial(1, 0.5, n).astype(np.int64)
    channel = rng.choice(np.array(["Phone", "Web", "Multichannel"], dtype=object), size=n,
                         p=[0.44, 0.44, 0.12])
    treatment = rng.binomial(1, treat_prob, n).astype(np.int64)

    f, tau, m = _f_tau_m(recency, history, mens, womens, zip_code, newbie, channel)
    p0, p1 = expit(f), expit(f + tau)
    buy = rng.uniform(size=n) < np.where(treatment == 1, p1, p0)
    amount = rng.gamma(spend_shape, m / spend_shape)
    spend = np.where(buy, np.round(amount, 2), 0.0)
    true_cate = (p1 - p0) * m

    segment = pd.cut(history, HISTORY_BINS, labels=HISTORY_LABELS, right=False).astype(str)
    experiment = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "recency": recency.astype(np.int64),
            "history_segment": segment,
            "history": history,
            "mens": mens,
            "womens": womens,
            "zip_code": zip_code,
            "newbie": newbie,
            "channel": channel,
            "treatment": treatment,
            "conversion": buy.astype(np.int64),
            "spend": spend,
            "true_cate": np.round(true_cate, 4),
        }
    )
    cate = experiment["true_cate"].to_numpy()
    gain = margin * cate - offer_cost
    truth = {
        "description": (
            "Randomized e-mail offer (Bernoulli treat_prob). P(buy) = expit(f(x) + T*tau(x)); "
            "spend | buy ~ Gamma(shape, mean m(x)) independent of T; true_cate = "
            "(expit(f+tau) - expit(f)) * m(x), the effect on expected spend."
        ),
        "seed": seed,
        "n": n,
        "treat_prob": treat_prob,
        "ate": sig(cate.mean()),
        "ate_definition": "mean of true_cate over the n rows (sample average treatment effect)",
        "margin": margin,
        "offer_cost": offer_cost,
        "decision_rule": "treat when cate * margin > offer_cost",
        "share_negative_cate": sig((cate < 0).mean()),
        "share_should_treat": sig((gain > 0).mean()),
        "policy_value_per_customer": {
            "treat_none": 0.0,
            "treat_all": sig(gain.mean()),
            "oracle": sig(np.maximum(gain, 0).mean()),
            "definition": "mean over rows of policy * (margin * true_cate - offer_cost), relative "
                          "to sending nobody the offer",
        },
        "base_conversion_control": sig(p0.mean()),
        "conversion_treated": sig(p1.mean()),
        "spend_shape": spend_shape,
    }
    return SynthResult("email_experiment", {"experiment": experiment}, truth)
