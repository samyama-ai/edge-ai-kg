"""`EA21`: twenty alerts, one fault -- which of them is upstream (#36)?

Two fixtures, because one of them alone proves nothing.

**A known root.** A chain where `root` feeds the head of a pipeline that
`mid` and `leaf` sit further down, so the right answer is known before the
query runs: 2, 1, 0. That pins the ranking.

**Genuinely independent alerts.** Four sensors on four separate chains, all
alerting at once, with no dependency between any of them. Every count must be
0 and every row must still be *present*. This is the fixture that matters: a
query that sorted by degree, or by how busy a sensor's pipeline is, would
rank these confidently and be wrong, and a query that used an inner `MATCH`
would return nothing at all and read as "no answer" rather than "no root".

The third check is the same ranking computed in Python from the generated
fleet and compared row for row, so the query is held to a truth that was not
derived from it.
"""
from __future__ import annotations

import collections

import pytest

from benchmarks.queries import BY_ID, EA21_ALERTS
from etl.helpers import create_edges, create_nodes
from tests.test_blast_radius import (
    SCALE,
    SEED,
    build_and_load,
    # The fixture itself, not a copy of it. A copy is what this module had,
    # and its `except` had widened to `Exception` -- so a broken
    # `onnx_catalog` parser or a corrupt cache would have reported "data/ is
    # not built" and skipped, where the original narrows to
    # `FileNotFoundError` precisely so `--no-skips` can tell a missing-data
    # skip from a real failure (`tests/test_environment_skips.py`). Imported,
    # there is one definition and it cannot drift again.
    operators,  # noqa: F401 -- used as a fixture by `loaded_fleet`
)

GRAPH = "default"


def retargeted_ea21(alert_ids) -> str:
    """`EA21` asking about `alert_ids` instead of the catalog's set.

    The catalog's three ids appear twice in the query -- once per `IN` list --
    and both have to move together: a rewrite that caught one would rank one
    population against another, and the ranking would look like a finding.
    Both occurrences are replaced, and the count is asserted rather than
    assumed.
    """
    original = BY_ID["EA21"]["cypher"]
    catalog_set = "[" + ", ".join(f'"{a}"' for a in EA21_ALERTS) + "]"
    occurrences = original.count(catalog_set)
    assert occurrences == 2, (
        f"`EA21` names its alert set {occurrences} times, not 2. Either the "
        f"query changed shape or the set moved; this retarget rewrites every "
        f"occurrence and cannot do that blind.")
    wanted = "[" + ", ".join(f'"{a}"' for a in alert_ids) + "]"
    return original.replace(catalog_set, wanted)


def rank(client, alert_ids) -> list[tuple[str, int]]:
    """`(alert, downstream_alerts)` per row, in the order the query returned."""
    rows = client.query(retargeted_ea21(alert_ids), GRAPH).records
    return [(row[0], row[1]) for row in rows]


def reached(client, alert_ids) -> dict[str, list[str]]:
    """The `reaches` column, which is what makes a cycle visible to a caller.

    Sorted here, not by the query: `collect` has no order to rely on, and the
    set is the claim.
    """
    rows = client.query(retargeted_ea21(alert_ids), GRAPH).records
    return {row[0]: sorted(row[2]) for row in rows}


def build_chain(client, sensors_to_stage, chain_edges):
    """One sensor per entry stage, plus the `NEXT_STAGE` chain between stages."""
    stages = sorted({stage for stage in sensors_to_stage.values()}
                    | {end for pair in chain_edges for end in pair})
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": sid, "modality": "ecg"} for sid in sensors_to_stage])
    create_nodes(client, GRAPH, "SignalStage",
                 [{"id": stage, "kind": "filter"} for stage in stages])
    create_edges(client, GRAPH, [
        ("Sensor", sid, "FEEDS", "SignalStage", stage, None)
        for sid, stage in sensors_to_stage.items()
    ] + [
        ("SignalStage", a, "NEXT_STAGE", "SignalStage", b, None)
        for a, b in chain_edges
    ])


@pytest.fixture
def known_root(engine_factory):
    """`root` -> `mid` -> `leaf` down one pipeline. The answer is 2, 1, 0."""
    client = engine_factory()
    build_chain(client,
                {"sensor:root": "stage:a", "sensor:mid": "stage:b",
                 "sensor:leaf": "stage:c"},
                [("stage:a", "stage:b"), ("stage:b", "stage:c")])
    return client


@pytest.fixture
def independent_alerts(engine_factory):
    """Four sensors, four separate chains, no dependency between any of them."""
    client = engine_factory()
    build_chain(client,
                {f"sensor:i{i}": f"stage:i{i}" for i in range(4)},
                [(f"stage:i{i}", f"stage:i{i}x") for i in range(4)])
    return client


def test_the_root_ranks_first_and_the_leaf_last(known_root):
    """The ordering a human acts on, against a fixture whose answer is known.

    This is also where an unenforced join would show. `OPTIONAL MATCH`
    re-binds `s`, and note 1 is about a *trailing* bound variable, so the join
    should hold -- "should" not being a measurement. The distinct 2/1/0 is
    what settles it: a cartesian product would give every alert every other
    one, so all three would score 2. The independent-alerts fixture cannot
    tell those apart on its own, because 0 and "joined to nothing" look the
    same there.
    """
    ranked = rank(known_root, ["sensor:root", "sensor:mid", "sensor:leaf"])
    assert ranked == [("sensor:root", 2), ("sensor:mid", 1), ("sensor:leaf", 0)], (
        f"expected the root to reach both others, the middle one to reach the "
        f"leaf, and the leaf to reach nothing; got {ranked}. All three scoring "
        f"2 would mean the `OPTIONAL MATCH` join is not enforced -- engine "
        f"note 1's shape in a query written not to have it.")


def test_independent_alerts_all_score_zero_and_none_is_dropped(independent_alerts):
    """The case that separates ranking by dependency from ranking by anything else.

    Four alerts, no dependencies. Every count must be 0 -- a query sorting by
    degree, by pipeline length or by any property of the sensor itself would
    still produce a confident order here -- and all four rows must come back,
    because "no root" is an answer and an empty result is not.
    """
    ids = [f"sensor:i{i}" for i in range(4)]
    ranked = rank(independent_alerts, ids)
    assert sorted(ranked) == sorted((sid, 0) for sid in ids), (
        f"independent alerts must all score 0 and all be reported; got {ranked}")


@pytest.fixture
def cyclic_alerts(engine_factory):
    """Two sensors whose entry stages point at each other, and one outside it.

    The shipped fleet has this shape -- `etl/generate.py` samples each
    sensor's chain from one shared pool of stages, so the union has cycles,
    and `tests/test_blast_radius_semantics.py` names one. Here it is built
    deliberately and small enough to reason about.
    """
    client = engine_factory()
    build_chain(client,
                {"sensor:x": "stage:x", "sensor:y": "stage:y",
                 "sensor:out": "stage:out"},
                [("stage:x", "stage:y"), ("stage:y", "stage:x")])
    return client


def test_two_alerts_in_a_cycle_each_reach_the_other(cyclic_alerts):
    """A cycle has no upstream-of, and `reaches` is what says so.

    This is the case the query's ranking cannot resolve and must not hide.
    `x` and `y` reach each other, so neither is the root, and they tie at 1 --
    which on the counts alone is indistinguishable from "two peers one hop
    below a root". The `reaches` column is the difference: each appears in the
    other's list, and a caller reading two ids that name each other knows the
    pair is circular.

    `out` is here so the fixture is not all-cycle: it reaches nothing and
    nothing reaches it, which is what a genuinely independent alert looks like
    beside a circular pair.
    """
    ids = ["sensor:x", "sensor:y", "sensor:out"]
    ranked = dict(rank(cyclic_alerts, ids))
    assert ranked == {"sensor:x": 1, "sensor:y": 1, "sensor:out": 0}, ranked

    seen = reached(cyclic_alerts, ids)
    assert seen["sensor:x"] == ["sensor:y"], seen
    assert seen["sensor:y"] == ["sensor:x"], seen
    assert seen["sensor:out"] == [], seen
    assert "sensor:y" in seen["sensor:x"] and "sensor:x" in seen["sensor:y"], (
        f"the mutual pair is invisible in the result: {seen}. Without it the "
        f"output claims an order over two alerts that have none.")


def test_an_unknown_alert_id_is_reported_as_absent_not_as_a_zero(known_root):
    """An id that is not a sensor gets no row, rather than a row scoring 0.

    Worth pinning because the two read very differently on a pager: a 0 says
    "this alert has nothing beneath it", while no row says "this id is not in
    the graph". `MATCH (s:Sensor)` is what makes the difference, and a future
    edit to `OPTIONAL MATCH` there would silently swap one for the other.
    """
    ranked = rank(known_root, ["sensor:root", "sensor:not-in-this-graph"])
    assert [alert for alert, _ in ranked] == ["sensor:root"], (
        f"an unknown id should not appear at all; got {ranked}")


def truth_for(fleet, alert_ids) -> list[tuple[str, int]]:
    """The same ranking, computed in Python from the fleet's own edges.

    Deliberately a different algorithm from the query's: a forward BFS over
    `NEXT_STAGE` from each alerting sensor's entry stages, then a set
    intersection with the other alerts' entry stages. If both agree, they are
    unlikely to be wrong in the same way.
    """
    feeds = collections.defaultdict(set)
    successors = collections.defaultdict(list)
    for _sl, src, rel, _tl, tgt, _p in fleet.edges:
        if rel == "FEEDS":
            feeds[src].add(tgt)
        elif rel == "NEXT_STAGE":
            successors[src].append(tgt)

    def reachable(stages):
        seen, queue = set(stages), list(stages)
        while queue:
            for nxt in successors[queue.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    known = {row["id"] for row in fleet.nodes.get("Sensor", ())}
    counts = []
    for alert in alert_ids:
        if alert not in known:
            continue
        downstream = reachable(feeds[alert])
        others = sorted(other for other in alert_ids
                        if other != alert and other in known
                        and feeds[other] & downstream)
        counts.append((alert, len(others), others))
    return sorted(counts, key=lambda row: (-row[1], row[0]))


def mutual_pairs(truth) -> list[tuple[str, str]]:
    """Pairs that reach each other: a cycle, where neither is upstream.

    Read off the truth rather than the query, so the test that reports them is
    not the thing being tested.
    """
    reaches = {alert: set(others) for alert, _n, others in truth}
    return sorted({tuple(sorted((a, b)))
                   for a, seen in reaches.items()
                   for b in seen
                   if a in reaches.get(b, ())})


@pytest.fixture(scope="module")
def loaded_fleet(engine_factory, operators):  # noqa: F811 -- the imported fixture
    """The generated fleet, loaded once -- the same builder the blast-radius
    tests use, so both read the identical graph rather than two graphs that
    happen to share a seed."""
    return build_and_load(engine_factory, operators, seed=SEED, scale=SCALE)


def test_ea21_matches_ground_truth_on_the_generated_fleet(loaded_fleet):
    """The catalog's own alert set, against truth computed from the `Fleet`.

    **Counts, not order.** Both sides are sorted before comparing, so what
    this checks is the number each alert reaches. The `ORDER BY` is covered by
    `known_root`, where the expected sequence is known in advance; on a
    generated fleet the counts tie, and asserting their order would pin an
    arbitrary one.

    The two guards below are what stop this passing on an empty comparison.
    `truth_for` drops ids the fleet does not hold, and this fixture's seed and
    scale (`4242`, `0.3`) are the blast-radius module's -- not the ones the
    catalog's alert set was chosen against. A fleet without those sensors, or
    one where none of them reaches another, would leave `[] == []` or all
    zeros on both sides, and the test would pass while measuring nothing.
    """
    client, fleet = loaded_fleet
    rows = client.query(retargeted_ea21(EA21_ALERTS), GRAPH).records
    got = sorted(((row[0], row[1], sorted(row[2])) for row in rows),
                 key=lambda row: (-row[1], row[0]))
    want = truth_for(fleet, list(EA21_ALERTS))

    assert len(want) == len(EA21_ALERTS), (
        f"the fixture fleet (seed {SEED}, scale {SCALE}) holds only "
        f"{[a for a, _n, _seen in want]} of {list(EA21_ALERTS)}; `truth_for` "
        f"drops "
        f"what it does not know, so this comparison would be narrower than it "
        f"reads. Pick alert ids that exist at this scale.")
    assert any(count for _alert, count, _seen in want), (
        f"no alert in {list(EA21_ALERTS)} reaches another at seed {SEED} "
        f"scale {SCALE}: the truth is {want}, so an all-zero answer would "
        f"match an engine that had computed nothing. Pick a set with a "
        f"dependency in it.")

    assert got == want, (
        f"EA21 ranked {got}; the fleet's own edges give {want}. One of the two "
        f"is reading the graph differently from the other.")

    # The cycles are in the shipped data, not only in the hand-built fixture,
    # so the mutual pairs are asserted here rather than described. At this
    # fixture's seed and scale every pair is mutual -- all three alerts reach
    # each other, so the counts tie at 2 and no row is upstream of another. At
    # `--scale 1.0` it is one pair (`sensor:00003` with `sensor:00007`) under
    # a genuine root (`sensor:00000`). Either way the claim being tested is
    # the same: whatever the truth says is mutual, the `reaches` column names
    # in both directions, because that is the only place a caller can see it.
    both_ways = mutual_pairs(want)
    assert both_ways, (
        f"no mutual pair at seed {SEED} scale {SCALE}: {want}. That is not a "
        f"failure of the query, but this assertion exists to keep the cyclic "
        f"case measured on real data -- if the generator stops producing "
        f"cycles, move this check to the cyclic fixture and say so here.")
    seen = {alert: set(others) for alert, _n, others in got}
    for a, b in both_ways:
        assert b in seen[a] and a in seen[b], (
            f"{a} and {b} reach each other in the fleet, and the query's "
            f"`reaches` column does not say so: {seen}. The counts alone "
            f"would present one of them as upstream of the other.")
