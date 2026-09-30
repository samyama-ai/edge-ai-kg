"""`EA17` and `EA21` bound their `NEXT_STAGE` walks, and the bound has to stay safe.

From `samyama` 1.8.0 the planner refuses an unbounded variable-length pattern
producing over a million paths (engine note 14). The stage graph is tiny --
16 stages -- but it contains cycles, so *paths* explode while *depth* stays
small. `MAX_STAGE_HOPS` is the upper bound that makes the pattern legal.

A bound is only honest while the data stays shorter than it. `EA17` reports
how deep a blast radius reaches, so a chain longer than the bound would be
truncated and the answer would look complete. That is what #110 objected to
when it argued against a cap, and it is why the bound is checked here against
the fleet rather than asserted in a comment:

- the deepest chain reachable from any sensor must stay clear of the bound;
- every walk in the catalog must carry the bound, so a new leg cannot be
  added unbounded and quietly re-break the 1.8 planner.
"""
from __future__ import annotations

import collections
import re

import pytest

from benchmarks.catalog.subjects import MAX_STAGE_HOPS
from benchmarks.queries import BY_ID
from tests.test_certification_alerts import (
    loaded_fleet,  # noqa: F401 -- shared fixture
)

HEADROOM = 2


@pytest.fixture(scope="module")
def stage_depths(loaded_fleet):  # noqa: F811 -- the imported fixture
    """Deepest `NEXT_STAGE` chain reachable from each sensor, by BFS.

    Depth, not path count: the graph has cycles, so counting paths would
    diverge while the question here -- how far downstream an answer has to
    reach -- is a shortest-path property.
    """
    _client, fleet = loaded_fleet
    successors = collections.defaultdict(list)
    feeds = collections.defaultdict(set)
    for _sl, src, rel, _tl, tgt, _p in fleet.edges:
        if rel == "NEXT_STAGE":
            successors[src].append(tgt)
        elif rel == "FEEDS":
            feeds[src].add(tgt)

    def depth_from(starts):
        seen, frontier, hops = set(starts), list(starts), 0
        while frontier:
            nxt = [n for f in frontier for n in successors[f] if n not in seen]
            seen.update(nxt)
            frontier = nxt
            if nxt:
                hops += 1
        return hops

    return {row["id"]: depth_from(feeds[row["id"]])
            for row in fleet.nodes.get("Sensor", ())}


def test_the_fleet_stays_well_inside_the_bound(stage_depths):
    """The measurement the bound rests on, re-run rather than remembered.

    Measured when the bound was chosen: 5 hops deepest, median 4, across 14
    sensors. If a generator change grows a chain to within `HEADROOM` of the
    bound, this fails while the answers are still correct -- which is the
    point. Waiting until a chain exceeds the bound would mean finding out
    from a silently short blast radius.
    """
    deepest = max(stage_depths.values())
    assert deepest + HEADROOM <= MAX_STAGE_HOPS, (
        f"the deepest chain reachable from a sensor is {deepest} hops and the "
        f"walks are bounded at {MAX_STAGE_HOPS}. Raise MAX_STAGE_HOPS and "
        f"re-check that EA17 still completes on the engine the floor allows, "
        f"or the blast radius will be truncated without saying so.")


def test_every_catalog_walk_carries_the_bound():
    """No `NEXT_STAGE*0..` may go back in unbounded.

    `EA07` walks the same relationship with its own fixed `*0..3`, which is a
    different decision documented on that entry, so the check is that no walk
    is *unbounded* rather than that every walk uses this constant.
    """
    unbounded = {qid: entry["cypher"] for qid, entry in BY_ID.items()
                 if re.search(r"NEXT_STAGE\*\d*\.\.\]", entry["cypher"])}
    assert not unbounded, (
        f"{sorted(unbounded)} walk NEXT_STAGE with no upper hop bound. From "
        f"samyama 1.8.0 the planner refuses that outright (engine note 14).")


def test_the_bound_is_at_least_the_depth_the_answers_need(stage_depths):
    """A bound below the real depth would truncate rather than fail."""
    deepest = max(stage_depths.values())
    assert MAX_STAGE_HOPS >= deepest, (
        f"MAX_STAGE_HOPS={MAX_STAGE_HOPS} is below the fleet's deepest chain "
        f"({deepest} hops), so EA17 would report a blast radius that stops "
        f"short and looks complete")
