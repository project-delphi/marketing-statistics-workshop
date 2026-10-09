"""Checks used inside ``with workshop.checkpoint(n):`` blocks.

Each check raises ``AssertionError`` with a message that says what is wrong and where to
look, and returns quietly (or returns a useful value) when the input passes. Intervals
are always named: a probability and a kind, ``"hdi"`` (highest density) or ``"eti"``
(equal tailed), computed here with numpy from the draws. Nothing depends on ArviZ's
defaults, which changed between versions (ArviZ 1.x defaults to a 0.89 ETI).

Every check has a test that feeds it a broken input and expects it to fail
(tests/test_checks.py).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

KINDS = ("hdi", "eti")


def _label(name: str | None, default: str) -> str:
    return name if name else default


def _fail(message: str) -> None:
    raise AssertionError(message)


def _frame_columns(df) -> list[str]:
    cols = getattr(df, "columns", None)
    if cols is None:
        _fail(f"Expected a table (a pandas DataFrame), got {type(df).__name__}.")
    return [str(c) for c in cols]


def columns(df, required: Iterable[str], name: str | None = None) -> None:
    """The table has every required column (extra columns are fine)."""
    what = _label(name, "The table")
    have = _frame_columns(df)
    missing = [c for c in required if c not in have]
    if missing:
        _fail(
            f"{what} is missing the column(s) {missing}. It has {have}. Check the names"
            " (spelling, capitals) your function gives the columns."
        )


def _numeric(values, what: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        _fail(f"{what} should hold numbers; it holds {np.asarray(values).dtype}.")
    return arr


def rfm_table(
    rfm,
    frequency: str = "frequency",
    recency: str = "recency",
    T: str = "T",
    customer_id: str | None = None,
    name: str | None = None,
) -> None:
    """An RFM (frequency, recency, age) table in the BTYD convention.

    One row per customer; ``frequency`` = repeat purchases (whole numbers >= 0, purchase
    periods after the first); ``recency`` = time from the first to the last purchase;
    ``T`` = time from the first purchase to the end of the observation window, all in the
    same time unit. Then ``0 <= recency <= T``, a customer with no repeat purchase has
    ``recency == 0``, and ``frequency <= recency`` (each repeat purchase falls in a later
    period than the first).
    """
    what = _label(name, "The RFM table")
    columns(rfm, [frequency, recency, T], what)
    if len(rfm) == 0:
        _fail(f"{what} has no rows.")
    if customer_id is not None:
        columns(rfm, [customer_id], what)
        dup = int(rfm[customer_id].duplicated().sum())
        if dup:
            _fail(f"{what} has {dup} repeated {customer_id} value(s): one row per customer.")
    f = _numeric(rfm[frequency], f"{what}: {frequency}")
    r = _numeric(rfm[recency], f"{what}: {recency}")
    t = _numeric(rfm[T], f"{what}: {T}")
    for arr, col in ((f, frequency), (r, recency), (t, T)):
        if not np.isfinite(arr).all():
            _fail(f"{what}: {col} has missing or infinite values.")
    if (f < 0).any() or not np.allclose(f, np.round(f)):
        _fail(
            f"{what}: {frequency} must be whole numbers >= 0 (repeat purchases). Found values"
            f" like {f[(f < 0) | ~np.isclose(f, np.round(f))][:3].tolist()}: did you swap"
            f" {frequency} and {recency}, or count the first purchase?"
        )
    if (r < 0).any():
        _fail(f"{what}: {recency} is negative for {int((r < 0).sum())} customers.")
    if (r > t + 1e-9).any():
        _fail(
            f"{what}: {recency} exceeds {T} for {int((r > t + 1e-9).sum())} customers. Both are"
            " measured from the first purchase; T runs to the end of the window."
        )
    zero = (f == 0) & (r > 1e-9)
    if zero.any():
        _fail(
            f"{what}: {int(zero.sum())} customers have no repeat purchase but {recency} > 0."
            " Recency is the time of the last REPEAT purchase since the first: 0 without one."
        )
    over = f > r + 1e-9
    if over.any():
        _fail(
            f"{what}: {frequency} exceeds {recency} for {int(over.sum())} customers. Each repeat"
            " purchase falls in a later period than the first, so frequency <= recency in the"
            " same time unit. Did you swap the two columns, or count purchases on the same"
            " day (or week) as separate repeats?"
        )


def interval(draws, prob: float = 0.94, kind: str = "hdi") -> tuple[float, float]:
    """The ``prob`` interval of ``kind`` ("hdi" or "eti") of the draws (all axes pooled)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if not 0 < prob < 1:
        raise ValueError(f"prob must be between 0 and 1, not {prob}")
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least two finite draws")
    if kind == "eti":
        lo, hi = np.quantile(x, [(1 - prob) / 2, 1 - (1 - prob) / 2])
        return float(lo), float(hi)
    n = x.size
    k = max(1, min(n - 1, int(math.ceil(prob * n)) - 1))  # window spans k + 1 draws
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def _interval_name(prob: float, kind: str) -> str:
    return f"{prob * 100:g}% {kind.upper()}"


def in_interval(
    truth: float,
    draws,
    prob: float = 0.94,
    kind: str = "hdi",
    name: str | None = None,
) -> tuple[float, float]:
    """The true value lies inside the named posterior interval of the draws. Returns it."""
    lo, hi = interval(draws, prob, kind)
    what = _label(name, "the parameter")
    label = _interval_name(prob, kind)
    if not lo <= float(truth) <= hi:
        _fail(
            f"The true value of {what}, {float(truth):.4g}, is outside the {label}"
            f" [{lo:.4g}, {hi:.4g}] of the posterior draws. Look at the sampler diagnostics"
            " (divergences, R-hat), the priors, and whether the model matches the process"
            " that made the data."
        )
    return lo, hi


def k_of_K_in_interval(
    truths: Mapping[str, float] | Sequence[float],
    draws: Mapping[str, object] | Sequence[object],
    k: int,
    prob: float = 0.94,
    kind: str = "hdi",
    name: str | None = None,
) -> dict:
    """At least ``k`` of the ``K`` true values lie inside their named intervals.

    ``truths`` and ``draws`` are both mappings with the same keys (e.g. channel names), or
    sequences in the same order. Returns ``{key: (lo, hi, inside)}``.
    """
    if isinstance(truths, Mapping):
        if not isinstance(draws, Mapping) or set(truths) != set(draws):
            _fail("truths and draws must have the same keys (one set of draws per truth).")
        keys = list(truths)
        pairs = [(key, truths[key], draws[key]) for key in keys]
    else:
        if len(truths) != len(draws):
            _fail(f"{len(truths)} truths but {len(draws)} sets of draws.")
        pairs = [(str(i), t, d) for i, (t, d) in enumerate(zip(truths, draws, strict=True))]
    out = {}
    for key, t, d in pairs:
        lo, hi = interval(d, prob, kind)
        out[key] = (lo, hi, bool(lo <= float(t) <= hi))
    inside = sum(v[2] for v in out.values())
    what = _label(name, "the parameters")
    label = _interval_name(prob, kind)
    if inside < k:
        outside = ", ".join(
            f"{key} (truth {float(t):.4g}, interval [{out[key][0]:.4g}, {out[key][1]:.4g}])"
            for key, t, _ in pairs
            if not out[key][2]
        )
        _fail(
            f"Only {inside} of {len(pairs)} true values of {what} are inside their {label};"
            f" at least {k} should be. Outside: {outside}. With {len(pairs)} parameters a"
            " miss or two can happen by chance; more points at the model or the sampler."
        )
    return out


def close(estimate, truth, rel: float | None = None, abs: float | None = None, name=None):
    """``|estimate - truth| <= abs + rel * |truth|`` elementwise. Give rel, abs or both."""
    if rel is None and abs is None:
        raise ValueError("close(): give a tolerance, rel=... and/or abs=...")
    rel = 0.0 if rel is None else rel
    atol = 0.0 if abs is None else abs
    e = _numeric(estimate, "The estimate")
    t = _numeric(truth, "The true value")
    if e.shape != t.shape and e.size != 1 and t.size != 1:
        _fail(f"The estimate has shape {e.shape} but the truth has shape {t.shape}.")
    gap = np.abs(e - t)
    allowed = atol + rel * np.abs(t)
    bad = ~(gap <= allowed)
    if bad.any():
        what = _label(name, "the estimate")
        if e.size == 1 and t.size == 1:
            _fail(
                f"{what} is {float(e.ravel()[0]):.6g}; expected {float(t.ravel()[0]):.6g}"
                f" (allowed difference {float(np.ravel(allowed)[0]):.3g}). Check the formula"
                " and the units."
            )
        _fail(
            f"{what}: {int(bad.sum())} of {bad.size} values differ from the expected ones by"
            f" more than the tolerance (largest gap {float(gap.max()):.3g})."
        )


def probability(x, name: str | None = None) -> None:
    """Every value is a probability: finite and in [0, 1]."""
    what = _label(name, "The values")
    arr = _numeric(x, what)
    if not np.isfinite(arr).all():
        _fail(f"{what} include missing or infinite values; probabilities are numbers in [0, 1].")
    if (arr < 0).any() or (arr > 1).any():
        _fail(
            f"{what} should lie in [0, 1]; found min {arr.min():.4g}, max {arr.max():.4g}."
            " Did you return a count or a rate instead of a probability?"
        )


def monotone(x, increasing: bool = True, strict: bool = False, name: str | None = None) -> None:
    """The values never go down (``increasing``) or never go up, in order."""
    what = _label(name, "The values")
    arr = _numeric(x, what).ravel()
    d = np.diff(arr)
    if increasing:
        bad = d <= 0 if strict else d < 0
        word = "increase" if strict else "never decrease"
    else:
        bad = d >= 0 if strict else d > 0
        word = "decrease" if strict else "never increase"
    if bad.any():
        i = int(np.argmax(bad))
        _fail(
            f"{what} should {word}, but step {i} goes from {arr[i]:.4g} to {arr[i + 1]:.4g}."
            " Check the sort order and the formula."
        )


def frames_agree(
    left,
    right,
    on: str | Sequence[str],
    rel: float = 1e-9,
    abs: float = 1e-9,
    names: tuple[str, str] = ("the first table", "the second table"),
) -> None:
    """Two tables hold the same rows (matched on ``on``) and the same values in their
    shared columns: exact for text and dates, within tolerance for numbers."""
    import pandas as pd

    keys = [on] if isinstance(on, str) else list(on)
    a_name, b_name = names
    columns(left, keys, a_name)
    columns(right, keys, b_name)
    if len(left) != len(right):
        _fail(f"{a_name} has {len(left)} rows but {b_name} has {len(right)}.")
    for df, label in ((left, a_name), (right, b_name)):
        dup = int(df.duplicated(keys).sum())
        if dup:
            _fail(f"{label} has {dup} repeated value(s) of {keys}: expected one row per key.")
    shared = [c for c in left.columns if c in set(right.columns) and c not in keys]
    if not shared:
        _fail(f"{a_name} and {b_name} have no columns in common besides {keys}.")
    a = left.reset_index(drop=True)
    b = right.reset_index(drop=True)
    merged = a[keys + shared].merge(
        b[keys + shared], on=keys, how="outer", suffixes=("__a", "__b"), indicator=True
    )
    only = merged[merged["_merge"] != "both"]
    if len(only):
        _fail(
            f"{len(only)} key(s) appear in only one of {a_name} and {b_name}, e.g."
            f" {only[keys].head(3).to_dict('records')}."
        )
    for c in shared:
        x, y = merged[f"{c}__a"], merged[f"{c}__b"]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            bad = ~np.isclose(xv, yv, rtol=rel, atol=abs, equal_nan=True)
        else:
            xs = pd.to_datetime(x) if pd.api.types.is_datetime64_any_dtype(y) else x
            ys = pd.to_datetime(y) if pd.api.types.is_datetime64_any_dtype(x) else y
            bad = ~((xs == ys) | (xs.isna() & ys.isna())).to_numpy()
        if bad.any():
            row = merged.loc[bad, keys + [f"{c}__a", f"{c}__b"]].head(3)
            _fail(
                f"Column {c!r} differs between {a_name} and {b_name} for {int(bad.sum())}"
                f" row(s), e.g. {row.to_dict('records')}. Check how each computes {c!r}"
                " (types, rounding, which rows are counted)."
            )
