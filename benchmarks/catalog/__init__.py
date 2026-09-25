"""The query catalog, assembled from one module per theme.

Split out of a single `benchmarks/queries.py` that had reached 916 lines --
past the 500 the review harness reads, so it was skipped unread, and a
skipped file is an unreviewed one. `benchmarks/queries.py` remains the
import surface: nothing outside this package imports from here directly.

Order is the catalog's order, `EA01` through `EA21`, because `QUERIES` is what
`run_benchmark` sweeps and what the demo walks. Each module holds a
contiguous range, so the ids stay in order without anything sorting them.

`EA20` was reserved rather than taken while the `Site` spine (#115) held that
id in review; both have since merged and the range has no gap. The reservation
is worth recording, because two queries sharing an id costs a silent collision
here -- `BY_ID` keeps the second and the catalog is simply one query short.
"""
from __future__ import annotations

from benchmarks.catalog.alerting import ALERTING
from benchmarks.catalog.core import CORE
from benchmarks.catalog.real_layer import REAL_LAYER
from benchmarks.catalog.subjects import EA17_SUBJECT, EA21_ALERTS, alert_list
from benchmarks.catalog.triage import TRIAGE

QUERIES: list[dict] = [*CORE, *REAL_LAYER, *ALERTING, *TRIAGE]

BY_ID = {q["id"]: q for q in QUERIES}

def retargeted_ea21(alert_ids) -> str:
    """`EA21` asking about `alert_ids` instead of the catalog's set.

    Here rather than in `tests/`, because the catalog is what production reads
    and a caller with a live alert set is the point of the query -- an MCP
    tool would call this, not a test helper.

    The set appears twice in the query, once per `IN` list, and both have to
    move together: a rewrite that caught one would rank one population against
    another and the ranking would read as a finding. Both are replaced, and
    the count is asserted rather than assumed.
    """
    original = BY_ID["EA21"]["cypher"]
    catalog_set = alert_list(EA21_ALERTS)
    occurrences = original.count(catalog_set)
    if occurrences != 2:
        raise ValueError(
            f"`EA21` names its alert set {occurrences} times, not 2. Either "
            f"the query changed shape or the set moved; this retarget rewrites "
            f"every occurrence and cannot do that blind.")
    return original.replace(catalog_set, alert_list(alert_ids))


__all__ = ["BY_ID", "EA17_SUBJECT", "EA21_ALERTS", "QUERIES", "retargeted_ea21"]
