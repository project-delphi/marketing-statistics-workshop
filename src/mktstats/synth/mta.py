"""User journeys across five channels with targeted exposure, and each channel's true
incremental conversions (Module 16, multi-touch attribution; briefs/16-multi-touch-attribution.md).

One month, one market of ``n_users`` users (the retailer's online journeys):

1. purchase intent z_i ~ Normal(0, 1); conversion draw U_i ~ Uniform(0, 1);
2. channel c reaches user i with probability expit(a_c + g_c z_i), independently across channels
   given z_i (g_c > 0: the channel targets high-intent users);
3. a reached user gets 1 + Poisson(m_c) touches from c, each at a time in [0, 1] drawn from
   Beta(e_c, f_c); the path lists the user's touches in time order;
4. y_i = 1[U_i < expit(theta_0 + delta z_i + sum_c theta_c R_ic)], R_ic = 1 if c reached i.
   Only which channels reached the user matters: the order and number of touches do not
   change conversion (a stated simplification);
5. users with no touch have no tracked journey, but their conversions count in the truth.

Retargeting has theta = 0 by construction: a stylized assumption that makes the mechanism
visible, not a claim about retargeting in general.

The true incremental conversions of channel c are Delta_c = sum_i [Y_i - Y_i^(-c)] over all
users, where Y_i^(-c) is the outcome had c not reached the user, with intent, the other channels
and U_i unchanged (common random numbers); ``incremental_conversions_expected`` is the same sum
of probabilities.

Draws come from one stream in a fixed order (z, U, then per channel the reach uniforms and the
touch counts of every user, then per channel the times of every user's potential touches), so
``random_exposure=True`` reuses every draw and changes only the reach probabilities: a user
reached by c in both worlds gets the same touches from c. A holdout uses its own stream
(``SeedSequence(seed, spawn_key=(1,))``), so it never changes the default dataset's draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit

from mktstats.synth._common import SynthResult, make_rng, sig, sig_dict

CHANNELS = ("display", "social", "search", "email", "retargeting")
PATH_SEPARATOR = " > "
# The committed CSV stores each touch as one letter with no separator, to stay small
# (encode_paths / decode_paths); the generator's table has the readable paths.
PATH_CODES = {"D": "display", "S": "social", "P": "search", "E": "email", "R": "retargeting"}
CSV_ENCODING = (
    "In mta_journeys.csv each touch of `path` is one letter (path_codes: P is paid search), in "
    "time order with no separator, so the file stays small; decode a path with "
    "' > '.join(path_codes[k] for k in path). The generator's journeys table has decoded paths."
)

# Reach: expit(a_c + g_c * z). g_c > 0 means the channel reaches users who already intend to buy.
REACH_A = {"display": -0.6, "social": -0.9, "search": -2.2, "email": -2.0, "retargeting": -2.4}
REACH_G = {"display": 0.0, "social": 0.2, "search": 1.2, "email": 0.8, "retargeting": 1.8}
# A reached user gets 1 + Poisson(m_c) touches.
EXTRA_TOUCHES = {"display": 1.5, "social": 1.0, "search": 0.5, "email": 1.0, "retargeting": 2.0}
# Touch times on [0, 1] (one month): Beta(e_c, f_c). Display early, retargeting late.
TIMING = {"display": (1.5, 4.0), "social": (2.0, 3.0), "search": (3.0, 2.0), "email": (1.0, 1.0),
          "retargeting": (4.0, 1.5)}
# Conversion: logit P(y = 1) = THETA0 + DELTA * z + sum_c THETA_c * R_c.
THETA0 = -4.2
DELTA = 1.0
THETA = {"display": 0.30, "social": 0.40, "search": 0.35, "email": 0.15, "retargeting": 0.0}

# Economics (assumptions of the synthetic world). One order at the retailer's population mean
# spend (truth.json retailer.population_mean_spend = 30) times the workshop's gross margin
# (truth.json channel_value.margin = 0.30); repeat purchases are ignored.
SPEND = {"display": 4000.0, "social": 3500.0, "search": 2500.0, "email": 300.0,
         "retargeting": 2500.0}
SCALABLE = {"display": True, "social": True, "search": True, "email": False, "retargeting": True}
MEAN_ORDER_VALUE = 30.0
GROSS_MARGIN = 0.30

DESCRIPTION = (
    "One month, one market of n_users users. Intent z ~ Normal(0, 1). Channel c reaches a user "
    "with probability expit(a_c + g_c z), independently across channels given z (g_c > 0: the "
    "channel targets users who already intend to buy; random_exposure replaces this with the "
    "constant E_z[expit(a_c + g_c z)]). A reached user gets 1 + Poisson(m_c) touches from c at "
    "times drawn from Beta(e_c, f_c) on [0, 1]; the path lists touches in time order. The user "
    "converts at the end of the month when U < expit(theta0 + delta z + sum_c theta_c R_c), with "
    "U ~ Uniform(0, 1) and R_c = 1 if c reached the user: only which channels reached the user "
    "matters, not the order or number of touches. Retargeting has theta = 0 by construction. "
    "Users with no touch have no tracked journey, but their conversions count in the truth."
)
DEFINITION = (
    "incremental_conversions[c] = sum over all n_users users of [Y_i - Y_i^(-c)], where Y_i^(-c) "
    "is the user's outcome had channel c not reached them, with intent, the other channels and "
    "the user's uniform draw U_i unchanged (common random numbers). "
    "incremental_conversions_expected[c] is the same sum of conversion probabilities."
)


def encode_paths(journeys: pd.DataFrame) -> pd.DataFrame:
    """The journeys with each path written as one letter per touch (the committed CSV)."""
    code = {name: k for k, name in PATH_CODES.items()}
    paths = journeys["path"].str.split(PATH_SEPARATOR).map(lambda p: "".join(code[c] for c in p))
    return journeys.assign(path=paths)


def decode_paths(journeys: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`encode_paths`: one letter per touch -> ``"display > search"``."""
    paths = journeys["path"].map(lambda s: PATH_SEPARATOR.join(PATH_CODES[k] for k in s))
    return journeys.assign(path=paths)


def margin_per_conversion() -> float:
    return MEAN_ORDER_VALUE * GROSS_MARGIN


def random_exposure_reach_prob(deg: int = 64) -> dict[str, float]:
    """E_z[expit(a_c + g_c z)] for z ~ Normal(0, 1), by Gauss-Hermite (probabilists') quadrature:
    each channel's population reach share, which random_exposure gives to every user."""
    x, w = hermegauss(deg)
    w = w / w.sum()
    return {c: float(np.sum(w * expit(REACH_A[c] + REACH_G[c] * x))) for c in CHANNELS}


def _holdout_spec(holdout) -> tuple[str, float] | None:
    if not holdout:
        return None
    if len(holdout) != 1:
        raise ValueError(f"holdout takes one channel, e.g. {{'retargeting': 0.2}}; got {holdout}")
    (channel, share), = holdout.items()
    if channel not in CHANNELS:
        raise ValueError(f"holdout channel {channel!r} is not one of {CHANNELS}")
    share = float(share)
    if not 0 < share < 1:
        raise ValueError(f"holdout share must be between 0 and 1, got {share}")
    return channel, share


def _paths(user: np.ndarray, channel: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Users with at least one touch and their paths (touches already sorted by user, time)."""
    names = np.array(CHANNELS)
    starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]]) if user.size else np.array([], int)
    ends = np.r_[starts[1:], user.size]
    labels = names[channel]
    paths = [PATH_SEPARATOR.join(labels[s:e]) for s, e in zip(starts, ends, strict=True)]
    return user[starts], paths


def mta_journeys(
    seed: int = 2036,
    *,
    n_users: int = 100_000,
    random_exposure: bool = False,
    holdout: dict[str, float] | None = None,
) -> SynthResult:
    """Simulated user journeys with targeted exposure and known incremental conversions.

    Tables
    ------
    journeys : ``user_id`` (1..n_users), ``path`` (touches in time order joined by ``" > "``),
        ``converted`` (0/1), ``true_intent`` (z, 4 decimals; truth, hidden in real data). One row
        per user with at least one touch, sorted by ``user_id``. The committed CSV writes the
        same table with one letter per touch (:func:`encode_paths`, ``truth["path_codes"]``).
    holdout : only with ``holdout={"retargeting": 0.2}`` (one channel): ``user_id``,
        ``holdout`` (1 if the channel was withheld), ``converted``, for every user the channel
        would reach. Withheld users lose their touches from it and convert as in its
        counterfactual (same U_i).

    ``random_exposure=True`` gives every user each channel's population reach share
    (``truth["random_exposure_reach_prob"]``), reusing every random draw.
    """
    spec = _holdout_spec(holdout)
    k = len(CHANNELS)
    rng = make_rng(seed)
    z = rng.standard_normal(n_users)
    u = rng.random(n_users)
    reach_u = np.empty((n_users, k))
    n_touch = np.empty((n_users, k), dtype=np.int64)
    for j, c in enumerate(CHANNELS):
        reach_u[:, j] = rng.random(n_users)
        n_touch[:, j] = 1 + rng.poisson(EXTRA_TOUCHES[c], n_users)
    times = [rng.beta(*TIMING[c], int(n_touch[:, j].sum())) for j, c in enumerate(CHANNELS)]

    flat_prob = random_exposure_reach_prob()
    if random_exposure:
        reach_p = np.tile([flat_prob[c] for c in CHANNELS], (n_users, 1))
    else:
        reach_p = expit(np.array([REACH_A[c] for c in CHANNELS])
                        + np.outer(z, [REACH_G[c] for c in CHANNELS]))
    would_reach = reach_u < reach_p
    reached = would_reach.copy()
    held = np.zeros(n_users, dtype=bool)
    if spec is not None:
        h_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(1,))))
        jh = CHANNELS.index(spec[0])
        held = would_reach[:, jh] & (h_rng.random(n_users) < spec[1])
        reached[held, jh] = False

    theta = np.array([THETA[c] for c in CHANNELS])
    eta0 = THETA0 + DELTA * z
    R = reached.astype(float)
    prob = expit(eta0 + R @ theta)
    y = u < prob

    # Touches in long format, then sorted by user and time.
    users, chans, ts = [], [], []
    for j in range(k):
        owner = np.repeat(np.arange(n_users), n_touch[:, j])
        keep = reached[owner, j]
        users.append(owner[keep])
        chans.append(np.full(int(keep.sum()), j))
        ts.append(times[j][keep])
    tu, tc, tt = np.concatenate(users), np.concatenate(chans), np.concatenate(ts)
    order = np.lexsort((tt, tu))
    touched, paths = _paths(tu[order], tc[order])
    journeys = pd.DataFrame({
        "user_id": touched + 1,
        "path": paths,
        "converted": y[touched].astype(int),
        "true_intent": np.round(z[touched], 4),
    })

    inc, inc_exp = {}, {}
    for j, c in enumerate(CHANNELS):
        Rm = R.copy()
        Rm[:, j] = 0.0
        pm = expit(eta0 + Rm @ theta)
        inc[c] = int(y.sum() - (u < pm).sum())
        inc_exp[c] = float((prob - pm).sum())
    p_none = expit(eta0)
    margin = margin_per_conversion()
    truth = {
        "seed": seed,
        "n_users": n_users,
        "channels": list(CHANNELS),
        "path_separator": PATH_SEPARATOR,
        "path_codes": PATH_CODES,
        "csv_encoding": CSV_ENCODING,
        "dgp": {
            "description": DESCRIPTION,
            "intent": "z ~ Normal(0, 1)",
            "reach_a": REACH_A,
            "reach_g": REACH_G,
            "extra_touches_poisson_mean": EXTRA_TOUCHES,
            "timing_beta": {c: list(v) for c, v in TIMING.items()},
            "theta0": THETA0,
            "delta": DELTA,
            "theta": THETA,
            "random_exposure": bool(random_exposure),
            "holdout": None if spec is None else {spec[0]: spec[1]},
        },
        "n_journeys": len(journeys),
        "conversions_total": int(y.sum()),
        "conversions_in_journeys": int(journeys["converted"].sum()),
        "conversions_untracked": int(y.sum() - journeys["converted"].sum()),
        "incremental_conversions": inc,
        "incremental_conversions_expected": sig_dict(inc_exp),
        "incremental_conversions_all_channels": int(y.sum() - (u < p_none).sum()),
        "incremental_conversions_all_channels_expected": sig(float((prob - p_none).sum())),
        "definition": DEFINITION,
        "reach_share": sig_dict({c: reached[:, j].mean() for j, c in enumerate(CHANNELS)}),
        "mean_intent_reached": sig_dict({c: z[reached[:, j]].mean()
                                         for j, c in enumerate(CHANNELS)}),
        "conversion_rate_reached": sig_dict({c: y[reached[:, j]].mean()
                                             for j, c in enumerate(CHANNELS)}),
        "conversion_rate_not_reached": sig_dict({c: y[~reached[:, j]].mean()
                                                 for j, c in enumerate(CHANNELS)}),
        "economics": {
            "spend": SPEND,
            "mean_order_value": MEAN_ORDER_VALUE,
            "gross_margin": GROSS_MARGIN,
            "margin_per_conversion": sig(margin),
            "scalable": SCALABLE,
            "note": "Monthly spend is an assumption of the synthetic world. Margin per "
                    "conversion = one order at the retailer's population mean spend (truth.json "
                    "retailer.population_mean_spend) x the workshop's gross margin (truth.json "
                    "channel_value.margin); repeat purchases are ignored. Email is list-limited "
                    "and cannot scale.",
        },
        "true_cost_per_incremental_conversion": {
            c: (sig(SPEND[c] / inc[c]) if inc[c] > 0 else None) for c in CHANNELS
        },
        "random_exposure_reach_prob": sig_dict(flat_prob),
    }
    data = {"journeys": journeys}
    if spec is not None:
        channel, share = spec
        jh = CHANNELS.index(channel)
        eligible = np.flatnonzero(would_reach[:, jh])
        data["holdout"] = pd.DataFrame({
            "user_id": eligible + 1,
            "holdout": held[eligible].astype(int),
            "converted": y[eligible].astype(int),
        })
        served = would_reach[:, jh] & ~held
        Rm = R.copy()
        Rm[:, jh] = 0.0
        pm = expit(eta0 + Rm @ theta)
        truth["holdout"] = {
            "channel": channel,
            "share": share,
            "users_eligible": int(eligible.size),
            "users_held_out": int(held.sum()),
            "users_served": int(served.sum()),
            "conversion_rate_served": sig(y[served].mean()),
            "conversion_rate_held_out": sig(y[held].mean()) if held.any() else None,
            "incremental_conversions_served": int(y[served].sum() - (u[served] < pm[served]).sum()),
            "incremental_conversions_served_expected": sig(float((prob - pm)[served].sum())),
            "assignment": "SeedSequence(seed, spawn_key=(1,)): its own stream, so the default "
                          "dataset's draws do not depend on it",
        }
    return SynthResult(name="mta", data=data, truth=truth)
