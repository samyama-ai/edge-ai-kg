"""Benchmark / demo query catalog for the Edge AI deployment KG.

Each entry is a question an edge-AI team actually asks while trying to land a
model on custom silicon. The `why_graph` field records why the question is
awkward in a relational or document store -- most of these are variable-depth
joins across hardware, kernel-library and model structure, or anti-joins
("which operator has NO kernel here").

Engine notes:
  * `NOT (pattern)` does not parse on this build; anti-joins are written as
    `OPTIONAL MATCH ... WITH ... count(x) AS n ... WHERE n = 0`.
  * String literals must be double-quoted.

The entries themselves live in `benchmarks/catalog/`, one module per theme,
because this file had grown to 916 lines and the review harness stops reading
at 500. This module is the import surface and stays that way: every caller in
the repo imports from here, so the split cost no call sites and the package
can be reorganised again without touching them.
"""
from __future__ import annotations

from benchmarks.catalog import (
    BY_ID,
    EA17_SUBJECT,
    EA21_ALERTS,
    MAX_STAGE_HOPS,
    QUERIES,
    retargeted_ea21,
)

__all__ = ["BY_ID", "EA17_SUBJECT", "EA21_ALERTS", "MAX_STAGE_HOPS", "QUERIES",
           "retargeted_ea21"]
