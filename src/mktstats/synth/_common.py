"""Shared pieces of the synthetic generators: the result type, the RNG and rounding rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SynthResult:
    """What every generator returns.

    ``data`` maps a table name to a DataFrame (e.g. ``"transactions"``, ``"customers"``);
    ``truth`` holds the true parameters, as plain JSON-serialisable Python values.
    Tables are also reachable as attributes: ``res.transactions`` is ``res.data["transactions"]``.
    """

    name: str
    data: dict[str, pd.DataFrame]
    truth: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> pd.DataFrame:
        data = self.__dict__.get("data", {})
        if item in data:
            return data[item]
        raise AttributeError(f"{type(self).__name__} has no table or attribute {item!r}; "
                             f"tables: {sorted(data)}")

    def __repr__(self) -> str:
        shapes = ", ".join(f"{k}={v.shape}" for k, v in self.data.items())
        return f"SynthResult({self.name!r}: {shapes}; truth keys={sorted(self.truth)})"


def make_rng(seed: int) -> np.random.Generator:
    """The only RNG the generators use: numpy ``Generator(PCG64(seed))``."""
    return np.random.Generator(np.random.PCG64(seed))


def sig(x: float, digits: int = 6) -> float:
    """Round to ``digits`` significant digits, so truth.json is stable across platforms."""
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return x
    return float(f"{x:.{digits}g}")


def sig_dict(d: dict[str, float], digits: int = 6) -> dict[str, float]:
    return {k: sig(v, digits) for k, v in d.items()}
