"""Seeded synthetic generators with known truth (pure numpy / pandas / scipy).

Every generator takes a ``seed`` (numpy ``Generator(PCG64(seed))``) and keyword parameters, and
returns a :class:`SynthResult` with ``.data`` (a dict of DataFrames, also reachable as attributes)
and ``.truth`` (a JSON-serialisable dict). The committed copies in ``data/synthetic/`` are the
default-seed outputs written by ``scripts/make_synthetic.py``; R reads only those files.

>>> from mktstats import synth
>>> res = synth.retailer()
>>> res.transactions.head()       # doctest: +SKIP
>>> res.truth["pnbd"]             # doctest: +SKIP
"""

from mktstats.synth._common import SynthResult, make_rng
from mktstats.synth.btyd import btyd_bgnbd, retailer
from mktstats.synth.email import email_experiment
from mktstats.synth.geo import geo_panel
from mktstats.synth.mmm import geometric_adstock, logistic_saturation, mmm

__all__ = [
    "SynthResult",
    "btyd_bgnbd",
    "email_experiment",
    "geo_panel",
    "geometric_adstock",
    "logistic_saturation",
    "make_rng",
    "mmm",
    "retailer",
]
