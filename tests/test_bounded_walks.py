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
- every walk the repo ships -- catalog and demo -- must carry an upper
  bound, so a new leg cannot be added unbounded and quietly re-break the 1.8
  planner;
- and bounding must change no answer, which is asked of the engine rather
  than argued: each query is run both ways and the rows compared.

An earlier version of this file also asserted `MAX_STAGE_HOPS >= deepest`.
That is implied by the headroom check below, so it was dropped rather than
kept as a second name for the same fact.
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

GRAPH = "default"
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


def test_every_walk_carries_the_bound():
    """No variable-length pattern anywhere may go back in without an upper bound.

    Three widenings on the first version of this guard, each one a hole the
    review of #126 pointed at:

    - it matched `NEXT_STAGE` alone. Note 14 is about *any* variable-length
      pattern producing over a million paths, so a new query walking some
      other relationship would have slipped past a guard named after the one
      relationship that broke first;
    - it matched `*0..]` but not a bare `*]` or `*..]`, which are equally
      unbounded. What makes a walk legal is an upper limit, however it is
      spelled -- `*3]`, `*0..3]`, `*..3]` -- so the check is for its absence;
    - it scanned `BY_ID` only. `demo/demo.py`'s `TASKS_OVER_BUDGET` is not in
      the catalog and carried the note-14 failure this change exists to
      remove; the demo is a shipped entry point, so it is scanned by name.

    `EA07` walks the same relationship with its own fixed `*0..3`, a tighter
    bound chosen for what it answers rather than for the planner, which is why
    the check is "has an upper bound" and not "uses `MAX_STAGE_HOPS`".
    """
    # Deferred: importing the demo pulls in rich, which this test is alone in
    # needing.
    from demo import demo

    cyphers = {qid: entry["cypher"] for qid, entry in BY_ID.items()}
    cyphers["demo.TASKS_OVER_BUDGET"] = demo.TASKS_OVER_BUDGET

    unbounded = {name: m.group(0)
                 for name, cypher in cyphers.items()
                 if (m := re.search(r"\*(\d+\.\.|\.\.)?\]", cypher))}
    assert not unbounded, (
        f"{unbounded} walk a variable-length relationship with no upper hop "
        f"bound. From samyama 1.8.0 the planner refuses that outright "
        f"(engine note 14).")


@pytest.mark.parametrize("qid", ["EA17", "EA21"])
def test_bounding_changes_no_answer(loaded_fleet, qid):  # noqa: F811 -- shared fixture
    """The claim the whole bound rests on, run rather than remembered.

    Both queries are asked twice against the same graph -- once as the catalog
    ships them, once with the bound removed -- and the results compared as
    sets. If a chain in this fleet were longer than `MAX_STAGE_HOPS`, the
    bounded answer would be a strict subset and this fails.

    Compared as sets, not sequences, deliberately: `EA21` has tied rows and no
    tiebreaker (note 3b forbids a second `ORDER BY` key), and running the
    *unbounded* query three times in a row returns those ties in different
    orders. Comparing sequences would fail on that instability and blame the
    bound for it.

    An earlier version of this file asserted the depth and left this
    comparison to a one-off run pasted into a PR description, while two
    comments claimed the test existed. That is the shape of claim this repo
    exists not to make.
    """
    client, _fleet = loaded_fleet
    bounded = BY_ID[qid]["cypher"]
    unbounded = bounded.replace(f"*0..{MAX_STAGE_HOPS}]", "*0..]")
    assert unbounded != bounded, f"{qid} no longer carries the bound this test measures"

    got_bounded = sorted(map(str, client.query(bounded, GRAPH).records))
    got_unbounded = sorted(map(str, client.query(unbounded, GRAPH).records))

    assert got_bounded == got_unbounded, (
        f"{qid} answers differently once bounded at {MAX_STAGE_HOPS} hops:\n"
        f"  bounded  : {got_bounded}\n"
        f"  unbounded: {got_unbounded}\n"
        f"A chain in this fleet is longer than the bound, so the bounded "
        f"answer is truncated while looking complete.")


def test_bounding_changes_no_answer_for_the_demo_query(loaded_fleet):  # noqa: F811
    """The same check for the one shipped walk that is not in the catalog.

    `demo/demo.py`'s `TASKS_OVER_BUDGET` was bounded in the same change that
    bounded `EA17` and `EA21`, and a bound that quietly changed the demo's
    numbers would be worse than one that changed a benchmark's: the demo is
    the narrated version, and beat 7 reads its result out as a sentence a
    human is sent.

    Kept separate from the catalog parametrisation rather than folded into it,
    because the demo query is not in `BY_ID` and reaching it needs an import
    of a module that pulls in `rich`.
    """
    from demo import demo

    client, _fleet = loaded_fleet
    bounded = demo.TASKS_OVER_BUDGET
    unbounded = bounded.replace(f"*0..{MAX_STAGE_HOPS}]", "*0..]")
    assert unbounded != bounded, (
        "TASKS_OVER_BUDGET no longer carries the bound this test measures")

    got_bounded = sorted(map(str, client.query(bounded, GRAPH).records))
    got_unbounded = sorted(map(str, client.query(unbounded, GRAPH).records))

    assert got_bounded == got_unbounded, (
        f"the demo's over-budget counts change once bounded at "
        f"{MAX_STAGE_HOPS} hops:\n"
        f"  bounded  : {got_bounded}\n"
        f"  unbounded: {got_unbounded}\n"
        f"Beat 7 reads these numbers out in a sentence, so a truncated "
        f"answer would be narrated as a fact.")
