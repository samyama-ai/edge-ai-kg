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
from tests.test_blast_radius import SCALE, SEED, build_and_load

GRAPH = "default"


def retargeted_ea20(alert_ids) -> str:
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
    rows = client.query(retargeted_ea20(alert_ids), GRAPH).records
    return [(row[0], row[1]) for row in rows]


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


@pytest.fixture(scope="module")
def operators():
    """The cached ONNX catalogue, skipping in **setup** when it is absent."""
    from etl import onnx_catalog as oc
    try:
        return oc.load_cached()
    except Exception as exc:                      # no data means skip, not fail
        pytest.skip(f"data/ is not built: {exc}")


def test_the_root_ranks_first_and_the_leaf_last(known_root):
    """The ordering a human acts on, against a fixture whose answer is known."""
    ranked = rank(known_root, ["sensor:root", "sensor:mid", "sensor:leaf"])
    assert ranked == [("sensor:root", 2), ("sensor:mid", 1), ("sensor:leaf", 0)], (
        f"expected the root to reach both others, the middle one to reach the "
        f"leaf, and the leaf to reach nothing; got {ranked}")


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


def test_an_unenforced_join_would_be_visible_here(independent_alerts):
    """`OPTIONAL MATCH` re-binds `s`, and this is the check that the join holds.

    Engine note 1 is about a *trailing* bound variable, and this pattern binds
    `s` in leading position -- so the join should be enforced. "Should" is not
    a measurement: if it were not, every sensor would see all three others
    through a cartesian product and score 3 rather than 0. The assertion above
    would catch it; this one says so in a message that names the cause.
    """
    ids = [f"sensor:i{i}" for i in range(4)]
    scores = {alert: count for alert, count in rank(independent_alerts, ids)}
    assert set(scores.values()) == {0}, (
        f"a sensor on its own chain scored {sorted(set(scores.values()))}; an "
        f"unenforced join would give 3 here, which is engine note 1's shape "
        f"showing up in a query that is supposed not to have it")


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
        others = sum(1 for other in alert_ids
                     if other != alert and other in known
                     and feeds[other] & downstream)
        counts.append((alert, others))
    return sorted(counts, key=lambda row: (-row[1], row[0]))


@pytest.fixture(scope="module")
def loaded_fleet(engine_factory, operators):
    """The generated fleet, loaded once -- the same builder the blast-radius
    tests use, so both read the identical graph rather than two graphs that
    happen to share a seed."""
    return build_and_load(engine_factory, operators, seed=SEED, scale=SCALE)


def test_ea20_matches_ground_truth_on_the_generated_fleet(loaded_fleet):
    """The catalog's own alert set, against truth computed from the `Fleet`."""
    client, fleet = loaded_fleet
    got = sorted(rank(client, EA21_ALERTS), key=lambda row: (-row[1], row[0]))
    want = truth_for(fleet, list(EA21_ALERTS))
    assert got == want, (
        f"EA21 ranked {got}; the fleet's own edges give {want}. One of the two "
        f"is reading the graph differently from the other.")
