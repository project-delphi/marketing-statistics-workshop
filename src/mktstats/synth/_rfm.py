"""RFM summary with exactly the conventions of ``pymc_marketing.clv.utils.rfm_summary`` (1.2.0).

Pure pandas, so ``mktstats`` never has to import pymc-marketing to build the table. Conventions
(read from the installed pymc-marketing 1.2.0 source, ``clv/utils.py``):

* dates are bucketed into periods of ``time_unit`` ("D", "W", "M", "h"); several transactions of a
  customer in the same period count once (their amounts are summed);
* ``frequency`` = number of periods with a purchase, minus one (repeat purchases);
* ``recency`` = (last purchase period - first purchase period) / ``time_scaler``;
* ``T`` = (observation end period - first purchase period) / ``time_scaler``;
* ``monetary_value`` = mean of the per-period amounts *excluding the first purchase*, 0 if none.

``mktstats.data.rfm_summary`` wraps this function and adds the ``MKTSTATS_SABOTAGE`` hook.
"""

from __future__ import annotations

import pandas as pd

_PERIOD_ALIASES = {"H": "h"}


def rfm_core(
    transactions: pd.DataFrame,
    customer_id_col: str = "customer_id",
    datetime_col: str = "date",
    monetary_value_col: str | None = None,
    observation_period_end=None,
    time_unit: str = "D",
    time_scaler: float = 1,
) -> pd.DataFrame:
    unit = _PERIOD_ALIASES.get(time_unit, time_unit)
    cols = [customer_id_col, datetime_col] + ([monetary_value_col] if monetary_value_col else [])
    tx = transactions[cols].copy()
    tx[datetime_col] = pd.to_datetime(tx[datetime_col])
    if observation_period_end is None:
        end = tx[datetime_col].max()
    else:
        end = pd.to_datetime(observation_period_end)
    end_period = end.to_period(unit)
    tx["_period"] = tx[datetime_col].dt.to_period(unit)
    tx = tx.loc[tx["_period"] <= end_period]
    tx["_ord"] = tx["_period"].array.asi8
    if monetary_value_col:
        per = (tx.groupby([customer_id_col, "_ord"], sort=True)[monetary_value_col]
               .sum().reset_index())
    else:
        per = tx[[customer_id_col, "_ord"]].drop_duplicates().sort_values([customer_id_col, "_ord"])
    g = per.groupby(customer_id_col, sort=True)["_ord"]
    first = g.min()
    last = g.max()
    count = g.size()
    out = pd.DataFrame(
        {
            "customer_id": first.index.to_numpy(),
            "frequency": (count - 1).to_numpy(dtype=float),
            "recency": ((last - first) / time_scaler).to_numpy(dtype=float),
            "T": ((end_period.ordinal - first) / time_scaler).to_numpy(dtype=float),
        }
    )
    if monetary_value_col:
        is_first = per["_ord"].to_numpy() == per[customer_id_col].map(first).to_numpy()
        rep = per.loc[~is_first]
        mv = rep.groupby(customer_id_col)[monetary_value_col].mean()
        out["monetary_value"] = (
            pd.Series(out["customer_id"].to_numpy()).map(mv).fillna(0.0).to_numpy(dtype=float)
        )
    return out


def swap_recency_frequency(rfm: pd.DataFrame) -> pd.DataFrame:
    """The ``swap_rf`` sabotage: exchange the recency and frequency columns."""
    out = rfm.copy()
    out["recency"], out["frequency"] = rfm["frequency"].to_numpy(), rfm["recency"].to_numpy()
    return out
