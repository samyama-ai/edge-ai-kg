"""`EA19`: which certifications a sensor failure touches (#40).

The question is the difference between an ops ticket and a reportable event.
`Certification` and `Sensor` share no edge — they meet only through
`ClinicalTask`, which `REQUIRES_SENSOR` on one side and is `GOVERNED_BY` on the
other. That is the join a table cannot make without being told to.

Two things are checked, because each can be wrong on its own:

**The answer, against the fleet.** Ground truth recomputed in Python from the
same `Fleet`, for every sensor rather than one — an id that happens to work is
not the claim.

**The distinction, on graphs built to show it.** The shipped fleet governs
*every* clinical task, so "a failure that touches a governed task" is the only
case it contains and the query would look correct while ignoring the
`GOVERNED_BY` edge entirely. The fixture tests supply the case the fleet cannot:
a task with no certification, whose sensor must therefore implicate none.

That second half is the one #40 asks for in as many words — "tested against a
fixture where the failure touches a governed task and one where it does not" —
and it is the half the real data cannot provide.
"""
from __future__ import annotations

import collections

import pytest

from benchmarks.queries import BY_ID
from etl.helpers import create_edges, create_nodes

GRAPH = "default"


def run_ea19(client, sensor_id: str):
    """`[(certification, body, class, tasks_affected)]`, sorted for comparison.

    Retargets on the quoted literal, as `retargeted_ea17` does: a bare
    `.replace` is a substring match, and an id that merely starts with the
    catalog's would be rewritten too. And checked by count, as that helper is:
    a membership check is vacuously true when `sensor_id` *is* `sensor:00000`,
    which is what every fixture test below passes.
    """
    original = BY_ID["EA19"]["cypher"]
    assert original.count('"sensor:00000"') == 1, (
        "EA19 should name its subject `sensor:00000` exactly once, in the one "
        "`WHERE`; it does not, so this retarget no longer does what it says")
    cypher = original.replace('"sensor:00000"', f'"{sensor_id}"')
    assert cypher.count(f'"{sensor_id}"') == 1, (
        f"retargeting to {sensor_id} left {cypher.count(chr(34) + sensor_id + chr(34))} "
        f"occurrences, expected 1")
    return sorted(tuple(row) for row in client.query(cypher, GRAPH).records)


def truth_for(fleet, sensor_id: str):
    """The same answer from the `Fleet`, walking the two edges independently.

    Grouped by `(name, body, class)`, because that is what `EA19` groups by (note
    9 rules out grouping on the node). Keying on the certification id instead
    would disagree with the query the day two ids share those properties --
    and the query, not this, would look wrong.
    """
    requires = collections.defaultdict(set)
    governs = collections.defaultdict(set)
    for _sl, src, rel, _tl, tgt, _p in fleet.edges:
        if rel == "REQUIRES_SENSOR":
            requires[tgt].add(src)
        elif rel == "GOVERNED_BY":
            governs[src].add(tgt)
    certs = {r["id"]: (r["name"], r["body"], r["class"])
             for r in fleet.nodes["Certification"]}

    per_cert = collections.defaultdict(set)
    for task in requires[sensor_id]:
        for cert in governs[task]:
            per_cert[certs[cert]].add(task)
    return sorted((*key, len(tasks)) for key, tasks in per_cert.items())


def test_ea19_matches_ground_truth_for_every_sensor(loaded_fleet):
    """Every sensor, not the one the catalog hardcodes."""
    client, fleet = loaded_fleet
    sensors = [r["id"] for r in fleet.nodes["Sensor"]]
    assert sensors, "no sensors in the fixture, so this proves nothing"

    disagreed = []
    for sensor_id in sensors:
        got, expected = run_ea19(client, sensor_id), truth_for(fleet, sensor_id)
        if got != expected:
            disagreed.append(f"{sensor_id}: EA19 {got} != Python {expected}")
    assert not disagreed, (
        "EA19 disagrees with the Fleet:\n  " + "\n  ".join(disagreed))


def test_every_sensor_in_this_fleet_implicates_something(loaded_fleet):
    """Why the fixtures below exist: the fleet contains only the governed case.

    All 18 clinical tasks are `GOVERNED_BY` something, so every sensor that any
    task requires implicates at least one certification. A query that ignored
    the `GOVERNED_BY` edge and returned every task's certifications — or one
    that returned a constant — would agree with the fleet on every row.
    """
    _client, fleet = loaded_fleet
    governed = {src for _sl, src, rel, _tl, _t, _p in fleet.edges
                if rel == "GOVERNED_BY"}
    tasks = {r["id"] for r in fleet.nodes["ClinicalTask"]}
    assert governed == tasks, (
        f"{len(tasks - governed)} tasks are ungoverned in the fleet. If that "
        f"is now true, this module's fixtures are no longer the only source of "
        f"the un-implicated case and the docstring above needs revisiting."
    )


def test_limit_20_cannot_truncate_ea19(loaded_fleet):
    """Why `EA19`'s untied `ORDER BY ... LIMIT 20` is safe (note 3b: one key only).

    A sensor can implicate at most every certification. While there are no
    more than 20, the limit never cuts, ties only reorder, and the sorted
    comparison above is exact. Past 20, which rows survive among equal counts
    is arbitrary and that comparison becomes flaky -- so this fails first.
    """
    _client, fleet = loaded_fleet
    certs = len(fleet.nodes["Certification"])
    assert certs <= 20, (
        f"{certs} certifications: EA19's LIMIT 20 can now truncate among ties, "
        f"which a single ORDER BY key cannot break. Raise the limit or rethink "
        f"the ordering.")


def certification_fixture(client, *, task_is_governed: bool, tasks: int = 1):
    """`tasks` tasks requiring one sensor, governed by one certification or not.

    `tasks=2` is the case the fleet does not contain: every certification there
    governs exactly one task that any given sensor requires, so `tasks_affected`
    is 1 on every real row and a query returning a constant 1 would agree with
    the whole fleet.
    """
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": "sensor:00000", "name": "ecg", "modality": "ecg"}])
    create_nodes(client, GRAPH, "ClinicalTask",
                 [{"id": f"task:{n}", "name": f"monitoring-{n}",
                   "category": "cardiac", "latency_budget_ms": 500,
                   "min_sensitivity": 0.9} for n in range(tasks)])
    create_nodes(client, GRAPH, "Certification",
                 [{"id": "cert:iec", "name": "IEC 62304 Class C",
                   "body": "IEC", "class": "C"}])
    edges = [("ClinicalTask", f"task:{n}", "REQUIRES_SENSOR",
              "Sensor", "sensor:00000", None) for n in range(tasks)]
    if task_is_governed:
        edges += [("ClinicalTask", f"task:{n}", "GOVERNED_BY",
                   "Certification", "cert:iec", None) for n in range(tasks)]
    create_edges(client, GRAPH, edges)
    return client


def test_tasks_affected_counts_the_tasks_not_a_constant(engine_factory):
    """The count is exercised, which the fleet cannot do.

    Every certification in the shipped fleet governs exactly one task that any
    given sensor requires, so `tasks_affected` is 1 on every real row -- and a
    query returning a literal 1 would match the ground truth everywhere. Two
    tasks under one certification is the smallest graph that tells them apart.
    """
    got = run_ea19(certification_fixture(engine_factory(),
                                         task_is_governed=True, tasks=2),
                   "sensor:00000")
    assert got == [("IEC 62304 Class C", "IEC", "C", 2)], (
        f"two governed tasks require this sensor, so `tasks_affected` must be "
        f"2; EA19 returned {got}"
    )


def test_a_failure_touching_a_governed_task_names_the_certification(engine_factory):
    got = run_ea19(certification_fixture(engine_factory(), task_is_governed=True),
                   "sensor:00000")
    assert got == [("IEC 62304 Class C", "IEC", "C", 1)], got


def test_a_failure_touching_no_governed_task_names_none(engine_factory):
    """The case the shipped fleet cannot produce, and the point of the query.

    The `Certification` node exists and the task requiring the sensor exists;
    only the `GOVERNED_BY` edge is missing. So a query that reached
    certifications any other way — or ignored the edge — would still return the
    row, and this is what distinguishes "implicated" from "present in the
    graph".
    """
    got = run_ea19(certification_fixture(engine_factory(), task_is_governed=False),
                   "sensor:00000")
    assert got == [], (
        f"nothing governs the affected task, so no certification is "
        f"implicated; EA19 returned {got}"
    )


@pytest.fixture(scope="module")
def loaded_fleet(engine_factory):
    from etl import generate as gen
    from etl import onnx_catalog as oc
    from etl import real_layer
    from etl.loader import NODE_LABELS

    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    client = engine_factory()
    fleet = gen.generate(seed=20260814, scale=0.3, operators=ops)
    real_layer.build_real(fleet, ops)
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client, fleet
