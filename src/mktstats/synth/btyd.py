"""Buy-till-you-die generators: the running retailer (Pareto/NBD with acquisition-channel covariates
and Gamma-Gamma spend) and a plain BG/NBD cohort for the recovery exercise.

Parameterizations (checked against installed source, 2026-10-09):

* pymc-marketing 1.2.0 ``clv/distributions.py``: ``lambda ~ Gamma(shape=r, rate=alpha)``,
  ``mu ~ Gamma(shape=s, rate=beta)`` (Pareto/NBD), dropout ``p ~ Beta(a, b)`` applied after each
  *repeat* purchase (BG/NBD); time is measured in the unit of ``recency`` and ``T``.
* Static covariates, CLVTools 0.12.1 (``pnbd_staticcov_alpha_i`` / ``_beta_i`` evaluated in the
  workshop Docker image) and pymc-marketing 1.2.0 ``ParetoNBDModel.build_model`` agree:
  ``alpha_i = alpha * exp(-gamma_purchase' x_i)`` and ``beta_i = beta * exp(-gamma_dropout' x_i)``.
  So a positive ``gamma_purchase`` raises the purchase rate and a positive ``gamma_dropout`` raises
  the dropout rate (shortens lifetimes).
* Gamma-Gamma (pymc-marketing 1.2.0 ``GammaGammaModel``): each transaction value
  ``z ~ Gamma(shape=p, rate=nu_i)``, ``nu_i ~ Gamma(shape=q, rate=v)``; a customer's mean spend is
  ``p / nu_i`` and the population mean is ``p * v / (q - 1)``.

All rates are per **week**. Transactions are stamped with calendar dates (one row per purchase);
summarise them with ``time_unit="D", time_scaler=7`` to get recency and T in weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mktstats.synth._common import SynthResult, make_rng, sig, sig_dict
from mktstats.synth._rfm import rfm_core

RETAILER_CHANNELS = ("search", "social", "referral")
RETAILER_CHANNEL_PROBS = (0.45, 0.35, 0.20)
RETAILER_GAMMA_PURCHASE = {"search": 0.0, "social": -0.4, "referral": 0.3}
RETAILER_GAMMA_DROPOUT = {"search": 0.0, "social": 0.6, "referral": -0.6}


def _poisson_purchases(rng, rate, window):
    """Purchase times of a Poisson process with ``rate`` per week on ``[0, window)`` weeks.

    Returns (customer index, time) arrays sorted by customer then time, and the counts.
    Exact: given N ~ Poisson(rate * window), the N times are i.i.d. uniform on the window.
    """
    n = rng.poisson(rate * window)
    idx = np.repeat(np.arange(len(rate)), n)
    t = rng.uniform(0.0, 1.0, size=int(n.sum())) * window[idx]
    order = np.lexsort((t, idx))
    return idx[order], t[order], n


def _rank_within(idx):
    """0-based position of each element within its (sorted, contiguous) group."""
    starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    sizes = np.diff(np.r_[starts, len(idx)])
    return np.arange(len(idx)) - np.repeat(starts, sizes)


def retailer(
    seed: int = 2026,
    *,
    n_customers: int = 5000,
    start: str = "2024-01-01",
    acquisition_weeks: int = 26,
    calibration_weeks: int = 52,
    holdout_weeks: int = 52,
    channels: tuple[str, ...] = RETAILER_CHANNELS,
    channel_probs: tuple[float, ...] = RETAILER_CHANNEL_PROBS,
    r: float = 0.8,
    alpha: float = 5.0,
    s: float = 0.5,
    beta: float = 10.0,
    gamma_purchase: dict[str, float] | None = None,
    gamma_dropout: dict[str, float] | None = None,
    p: float = 6.0,
    q: float = 4.0,
    v: float = 15.0,
    annual_discount_rate: float = 0.10,
) -> SynthResult:
    """The running retailer: Pareto/NBD purchases and dropout with acquisition-channel effects.

    Customers are acquired (make their first purchase) on a uniformly random day in the first
    ``acquisition_weeks``. The first channel in ``channels`` is the reference level: its
    coefficients are 0 and ``r, alpha, s, beta`` are its population parameters. Every other channel
    gets a 0/1 indicator column ``channel_<name>`` in ``customers``; ``gamma_purchase[name]`` and
    ``gamma_dropout[name]`` are the coefficients on that column in the CLVTools / pymc-marketing
    parameterization (see the module docstring).

    Tables
    ------
    transactions : customer_id, date, amount (every purchase, including the first one).
    customers : customer_id, acquisition_channel, first_date, channel_<name> indicators, and truth
        columns true_lambda, true_mu (per week), true_mean_spend (p / nu_i),
        true_alive_at_cal_end, true_alive_at_end.

    Truth also holds ``value_by_channel``: true expected purchases, spend per transaction and
    discounted CLV per acquisition channel over the holdout horizon (``annual_discount_rate``),
    for the existing customers and for a newly acquired customer (see ``_value_by_channel``).
    """
    gamma_purchase = dict(RETAILER_GAMMA_PURCHASE if gamma_purchase is None else gamma_purchase)
    gamma_dropout = dict(RETAILER_GAMMA_DROPOUT if gamma_dropout is None else gamma_dropout)
    channels = tuple(channels)
    if len(channel_probs) != len(channels):
        raise ValueError("channel_probs must have one entry per channel")
    ref = channels[0]
    for g in (gamma_purchase, gamma_dropout):
        g.setdefault(ref, 0.0)
        if g[ref] != 0.0:
            raise ValueError(f"the reference channel {ref!r} must have coefficient 0")
        missing = set(channels) - set(g)
        if missing:
            raise ValueError(f"missing coefficients for channels {sorted(missing)}")

    rng = make_rng(seed)
    n = n_customers
    cal_days = 7 * calibration_weeks
    total_days = 7 * (calibration_weeks + holdout_weeks)
    if acquisition_weeks <= 0 or 7 * acquisition_weeks > cal_days:
        raise ValueError("acquisition must happen inside the calibration period")

    ch = rng.choice(len(channels), size=n, p=np.asarray(channel_probs, float))
    d0 = rng.integers(0, 7 * acquisition_weeks, size=n)
    gp = np.array([gamma_purchase[c] for c in channels])[ch]
    gd = np.array([gamma_dropout[c] for c in channels])[ch]
    alpha_i = alpha * np.exp(-gp)
    beta_i = beta * np.exp(-gd)
    lam = rng.gamma(r, 1.0 / alpha_i)
    mu = rng.gamma(s, 1.0 / beta_i)
    tau = rng.exponential(1.0 / mu)  # weeks from first purchase to (unobserved) dropout
    horizon = (total_days - d0) / 7.0
    active = np.minimum(tau, horizon)
    idx, t, _ = _poisson_purchases(rng, lam, active)
    nu = rng.gamma(q, 1.0 / v, size=n)

    # first purchase on day d0, repeats on day d0 + floor(7 t)
    cust = np.r_[np.arange(n), idx]
    day = np.r_[d0, d0[idx] + np.floor(7.0 * t).astype(np.int64)]
    amount = rng.gamma(p, 1.0 / nu[cust])
    amount = np.maximum(np.round(amount, 2), 0.01)
    order = np.lexsort((day, cust))  # stable: first purchase stays first on a tie
    cust, day, amount = cust[order], day[order], amount[order]

    start_ts = pd.Timestamp(start)
    dates = start_ts + pd.to_timedelta(day, unit="D")
    ids = np.arange(1, n + 1)
    transactions = pd.DataFrame({"customer_id": ids[cust], "date": dates, "amount": amount})

    dropout_day = d0 + 7.0 * tau
    customers = pd.DataFrame(
        {
            "customer_id": ids,
            "acquisition_channel": np.array(channels, dtype=object)[ch],
            "first_date": start_ts + pd.to_timedelta(d0, unit="D"),
        }
    )
    for j, c in enumerate(channels[1:], start=1):
        customers[f"channel_{c}"] = (ch == j).astype(np.int64)
    customers["true_lambda"] = np.round(lam, 6)
    customers["true_mu"] = np.round(mu, 6)
    customers["true_mean_spend"] = np.round(p / nu, 4)
    customers["true_alive_at_cal_end"] = (dropout_day > cal_days).astype(np.int64)
    customers["true_alive_at_end"] = (dropout_day > total_days).astype(np.int64)

    cal_end = start_ts + pd.Timedelta(days=cal_days - 1)
    end = start_ts + pd.Timedelta(days=total_days - 1)
    cal = transactions["date"] <= cal_end
    truth = {
        "description": (
            "Pareto/NBD purchases and dropout with static acquisition-channel covariates "
            "(CLVTools parameterization), Gamma-Gamma spend per transaction. Rates per week."
        ),
        "seed": seed,
        "n_customers": n,
        "time_unit": "week",
        "rfm_settings": {"time_unit": "D", "time_scaler": 7},
        "start_date": str(start_ts.date()),
        "acquisition_end": str((start_ts + pd.Timedelta(days=7 * acquisition_weeks - 1)).date()),
        "calibration_end": str(cal_end.date()),
        "holdout_end": str(end.date()),
        "calibration_weeks": calibration_weeks,
        "holdout_weeks": holdout_weeks,
        "pnbd": sig_dict({"r": r, "alpha": alpha, "s": s, "beta": beta}),
        "reference_channel": ref,
        "covariates": {
            "convention": (
                "alpha_i = alpha * exp(-gamma_purchase * x_i), beta_i = beta * exp(-gamma_dropout "
                "* x_i); x_i = 0/1 indicator column channel_<name>; positive gamma_purchase = more "
                "purchases, positive gamma_dropout = faster dropout (CLVTools and pymc-marketing "
                "agree)."
            ),
            "columns": {c: f"channel_{c}" for c in channels[1:]},
            "gamma_purchase": sig_dict(gamma_purchase),
            "gamma_dropout": sig_dict(gamma_dropout),
        },
        "channel_probs": dict(zip(channels, map(float, channel_probs), strict=True)),
        "gamma_gamma": sig_dict({"p": p, "q": q, "v": v}),
        "population_mean_spend": sig(p * v / (q - 1)),
        "sample": {
            "n_transactions": int(len(transactions)),
            "n_transactions_calibration": int(cal.sum()),
            "mean_true_lambda": sig(lam.mean()),
            "mean_true_mu": sig(mu.mean()),
            "share_alive_at_cal_end": sig(customers["true_alive_at_cal_end"].mean()),
            "share_alive_at_end": sig(customers["true_alive_at_end"].mean()),
            "channel_counts": {c: int((ch == j).sum()) for j, c in enumerate(channels)},
        },
        "value_by_channel": _value_by_channel(
            channels, ch, lam, mu, p / nu, dropout_day > cal_days, transactions, cal_end,
            r=r, alpha_c={c: alpha * np.exp(-gamma_purchase[c]) for c in channels},
            s=s, beta_c={c: beta * np.exp(-gamma_dropout[c]) for c in channels},
            mean_spend=p * v / (q - 1), horizon_weeks=holdout_weeks,
            annual_discount_rate=annual_discount_rate,
        ),
    }
    return SynthResult("retailer", {"transactions": transactions, "customers": customers}, truth)


def _value_by_channel(channels, ch, lam, mu, mean_spend_i, alive_cal, transactions, cal_end, *,
                      r, alpha_c, s, beta_c, mean_spend, horizon_weeks, annual_discount_rate):
    """True customer value per acquisition channel (answer key for the CLV and capstone labs).

    Two views, both in revenue (spend, not margin), horizon ``H = horizon_weeks`` (the holdout),
    continuous discounting at weekly rate ``d = ln(1 + annual_discount_rate) / 52``:

    * existing customers at the calibration end (closed form from each customer's true parameters):
      expected purchases ``alive * lam / mu * (1 - exp(-mu H))``; discounted value
      ``alive * m * lam * (1 - exp(-(mu + d) H)) / (mu + d)`` with ``m = p / nu`` the customer's
      true mean spend per transaction; means are over all customers of the channel;
    * a newly acquired customer (population expectation over lam ~ Gamma(r, alpha_c),
      mu ~ Gamma(s, beta_c), nu ~ Gamma(q, v)): expected repeat purchases in the first H weeks
      ``(r / alpha_c) * int_0^H (beta_c / (beta_c + t))**s dt`` and the discounted value
      ``E[spend] * (r / alpha_c) * int_0^H exp(-d t) (beta_c / (beta_c + t))**s dt``, by numerical
      quadrature (scipy.integrate.quad); "including first purchase" adds ``E[spend]`` at t = 0.
    """
    from scipy.integrate import quad

    d = float(np.log1p(annual_discount_rate) / 52.0)
    h = float(horizon_weeks)
    exp_purch = alive_cal * lam / mu * (1 - np.exp(-mu * h))
    clv = alive_cal * mean_spend_i * lam * (1 - np.exp(-(mu + d) * h)) / (mu + d)
    hold = transactions.loc[transactions["date"] > cal_end]
    ids = np.arange(1, len(ch) + 1)
    n_hold = hold.groupby("customer_id").size().reindex(ids, fill_value=0).to_numpy()
    s_hold = hold.groupby("customer_id")["amount"].sum().reindex(ids, fill_value=0.0).to_numpy()
    existing, new = {}, {}
    for j, c in enumerate(channels):
        m = ch == j
        existing[c] = {
            "n_customers": int(m.sum()),
            "share_alive_at_cal_end": sig(alive_cal[m].mean()),
            "mean_expected_purchases": sig(exp_purch[m].mean()),
            "mean_spend_per_transaction": sig(mean_spend_i[m].mean()),
            "mean_discounted_clv": sig(clv[m].mean()),
            "total_discounted_clv": sig(clv[m].sum()),
            "observed_holdout_mean_purchases": sig(n_hold[m].mean()),
            "observed_holdout_mean_spend": sig(s_hold[m].mean()),
        }
        a_c, b_c = alpha_c[c], beta_c[c]
        undisc = quad(lambda t, b_c=b_c: (b_c / (b_c + t)) ** s, 0, h)[0]
        disc = quad(lambda t, b_c=b_c: np.exp(-d * t) * (b_c / (b_c + t)) ** s, 0, h)[0]
        new[c] = {
            "expected_repeat_purchases": sig(r / a_c * undisc),
            "mean_spend_per_transaction": sig(mean_spend),
            "discounted_clv_excluding_first_purchase": sig(mean_spend * r / a_c * disc),
            "discounted_clv_including_first_purchase": sig(mean_spend * (1 + r / a_c * disc)),
        }
    return {
        "definition": (
            "Revenue (spend), not margin. Horizon = holdout_weeks after the calibration end "
            "(existing customers) or after acquisition (new customer). Continuous discounting at "
            "weekly rate ln(1 + annual_discount_rate) / 52. existing_customers_at_cal_end: closed "
            "form from each customer's true lambda, mu, mean spend and true alive status, averaged "
            "over all customers of the channel. new_customer: expectation over the channel's "
            "population distributions, by numerical quadrature. observed_holdout_*: what the "
            "holdout transactions actually show (one realization)."
        ),
        "horizon_weeks": int(horizon_weeks),
        "annual_discount_rate": annual_discount_rate,
        "weekly_discount_rate": sig(d),
        "existing_customers_at_cal_end": existing,
        "new_customer": new,
    }


def btyd_bgnbd(
    seed: int = 2027,
    *,
    n_customers: int = 4000,
    start: str = "2024-01-01",
    acquisition_weeks: int = 12,
    calibration_weeks: int = 52,
    holdout_weeks: int = 26,
    r: float = 0.6,
    alpha: float = 6.0,
    a: float = 0.8,
    b: float = 3.0,
) -> SynthResult:
    """Plain BG/NBD customers with known r, alpha, a, b (rates per week).

    First purchases fall on a uniformly random day of the first ``acquisition_weeks`` (all on the
    start date if 0). While active a customer buys as a Poisson process with rate
    ``lambda ~ Gamma(r, rate=alpha)``; after each repeat purchase they drop out with probability
    ``p ~ Beta(a, b)`` (the convention of pymc-marketing's ``BetaGeoNBD``).

    Tables
    ------
    transactions : customer_id, date (one row per purchase, first purchase included).
    rfm : customer_id, frequency, recency, T (calibration period, weeks, exactly what
        ``pymc_marketing.clv.utils.rfm_summary(tx, "customer_id", "date",
        observation_period_end=calibration_end, time_unit="D", time_scaler=7)`` returns),
        test_frequency (repeat purchases in the holdout), test_T (holdout length, weeks), and truth
        columns true_lambda, true_p, true_alive_at_cal_end, true_alive_at_end.
    """
    rng = make_rng(seed)
    n = n_customers
    cal_days = 7 * calibration_weeks
    total_days = 7 * (calibration_weeks + holdout_weeks)
    if 7 * acquisition_weeks > cal_days:
        raise ValueError("acquisition must happen inside the calibration period")
    d0 = (rng.integers(0, 7 * acquisition_weeks, size=n) if acquisition_weeks > 0
          else np.zeros(n, dtype=np.int64))
    lam = rng.gamma(r, 1.0 / alpha, size=n)
    pdrop = rng.beta(a, b, size=n)
    k = rng.geometric(pdrop)  # dropout right after the k-th repeat purchase (k >= 1)
    horizon = (total_days - d0) / 7.0
    idx, t, cnt = _poisson_purchases(rng, lam, horizon)
    pos = _rank_within(idx) if len(idx) else idx
    keep = pos < k[idx]
    idx, t, pos = idx[keep], t[keep], pos[keep]

    # dropout day: day of the k-th repeat purchase if it happened inside the horizon
    dead = cnt >= k
    dropout_day = np.full(n, np.inf)
    last_of_k = pos == (k[idx] - 1)
    dropout_day[idx[last_of_k]] = d0[idx[last_of_k]] + 7.0 * t[last_of_k]
    dropout_day[~dead] = np.inf

    cust = np.r_[np.arange(n), idx]
    day = np.r_[d0, d0[idx] + np.floor(7.0 * t).astype(np.int64)]
    order = np.lexsort((day, cust))
    cust, day = cust[order], day[order]
    start_ts = pd.Timestamp(start)
    ids = np.arange(1, n + 1)
    transactions = pd.DataFrame(
        {"customer_id": ids[cust], "date": start_ts + pd.to_timedelta(day, unit="D")}
    )

    cal_end = start_ts + pd.Timedelta(days=cal_days - 1)
    end = start_ts + pd.Timedelta(days=total_days - 1)
    rfm = rfm_core(transactions, observation_period_end=cal_end, time_unit="D", time_scaler=7)
    hold = transactions.loc[transactions["date"] > cal_end]
    test_freq = hold.drop_duplicates().groupby("customer_id").size()
    rfm["test_frequency"] = rfm["customer_id"].map(test_freq).fillna(0).astype(float)
    rfm["test_T"] = float(holdout_weeks)
    rfm["true_lambda"] = np.round(lam, 6)
    rfm["true_p"] = np.round(pdrop, 6)
    rfm["true_alive_at_cal_end"] = (dropout_day > cal_days).astype(np.int64)
    rfm["true_alive_at_end"] = (dropout_day > total_days).astype(np.int64)

    truth = {
        "description": "Plain BG/NBD (Fader, Hardie & Lee 2005) cohort; rates per week.",
        "seed": seed,
        "n_customers": n,
        "time_unit": "week",
        "rfm_settings": {"time_unit": "D", "time_scaler": 7},
        "start_date": str(start_ts.date()),
        "calibration_end": str(cal_end.date()),
        "holdout_end": str(end.date()),
        "calibration_weeks": calibration_weeks,
        "holdout_weeks": holdout_weeks,
        "bgnbd": sig_dict({"r": r, "alpha": alpha, "a": a, "b": b}),
        "sample": {
            "n_transactions": int(len(transactions)),
            "share_with_repeat_in_calibration": sig((rfm["frequency"] > 0).mean()),
            "mean_frequency": sig(rfm["frequency"].mean()),
            "share_alive_at_cal_end": sig(rfm["true_alive_at_cal_end"].mean()),
        },
    }
    return SynthResult("btyd_bgnbd", {"transactions": transactions, "rfm": rfm}, truth)
