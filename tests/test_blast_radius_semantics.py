"""`EA17`'s semantics, on graphs built to make one distinction at a time (#35).

Split from `tests/test_blast_radius.py`, which compares `EA17` against a
breadth-first search over the generated fleet. That comparison answers "does the
query agree with Python on the real graph"; these answer "does it mean what the
question means", and they need graphs the generator does not produce.

Each fixture here isolates exactly one property, because the shipped graph
cannot show any of them:

- **a shared stage** -- `only_via_me` must exclude what keeps running.
- **a board shared through a different deployment** -- the case a shared *stage*
  cannot reach, and a real bug it hid.
- **a twelve-stage chain** -- the case that makes `*0..` a requirement rather
  than a preference.

On the shipped graph `only_via_me` is 0 for every kind, because every stage has
another feeder, so an over-report has nothing to be wrong about. That is why
these fixtures exist and why the ground-truth sweep alone is not enough.
"""
from __future__ import annotations

from benchmarks.queries import BY_ID, EA17_SUBJECT
from etl.helpers import create_edges, create_nodes
from tests.test_blast_radius import (
    EA17_SUBJECT_OCCURRENCES,
    retargeted_ea17,
    run_ea17,
)

GRAPH = "default"


def test_retargeting_replaces_every_occurrence_of_the_sensor_id():
    """The subject id appears once per leg per clause -- ten times in one query.

    Editing nine of them leaves the tenth pointing at `sensor:00000`, and the
    result is one leg answering about a different sensor than the other four.
    Nothing about the output would look wrong. `run_ea17` retargets with a
    `.replace`, which is all-or-nothing; this pins that the query stays
    replaceable and that no occurrence survives.
    """
    cypher = BY_ID["EA17"]["cypher"]
    occurrences = cypher.count(EA17_SUBJECT)
    assert occurrences == EA17_SUBJECT_OCCURRENCES, (
        f"expected the subject id once per leg at least, found {occurrences}. "
        f"If the query stopped hardcoding it, this test and the retargeting in "
        f"`run_ea17` both need rewriting."
    )
    # Through `retargeted_ea17`, which is what `run_ea17` uses. This pinned a
    # bare `cypher.replace("sensor:00000", ...)` while the code performed a
    # quoted-literal replace -- so the test could pass on a query the helper
    # could not retarget, and vice versa.
    retargeted = retargeted_ea17("sensor:00007")
    assert "sensor:00000" not in retargeted, (
        "the retarget left an occurrence behind, so one leg would answer about "
        "a different sensor than the rest"
    )
    assert retargeted.count("sensor:00007") == occurrences


def test_a_shared_stage_is_a_firebreak(engine_factory):
    """The case the shipped graph cannot show, and the first EA17 got wrong.

    Two sensors. `sensor:00000` feeds `shared`; `sensor:00001` also feeds
    `shared` and then continues to `tail`, which precedes a model. Nothing past
    `shared` stops when `sensor:00000` fails -- `shared` still has a live
    feeder, so `tail` and the model keep running.

    A plain reachability query reports all of it as blast radius. `only_via_me`
    is what makes the difference visible, and this asserts it is zero for
    everything beyond the firebreak while `affected` still counts it.
    """
    client = engine_factory()
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": "sensor:00000", "name": "failing", "modality": "ECG"},
                  {"id": "sensor:00001", "name": "healthy", "modality": "PPG"}])
    create_nodes(client, GRAPH, "SignalStage",
                 [{"id": "mine", "name": "mine"},
                  {"id": "shared", "name": "shared"},
                  {"id": "tail", "name": "tail"}])
    create_nodes(client, GRAPH, "Model",
                 [{"id": "downstream", "name": "downstream", "family": "cnn"}])
    create_edges(client, GRAPH, [
        ("Sensor", "sensor:00000", "FEEDS", "SignalStage", "mine", None),
        ("SignalStage", "mine", "NEXT_STAGE", "SignalStage", "shared", None),
        ("Sensor", "sensor:00001", "FEEDS", "SignalStage", "shared", None),
        ("SignalStage", "shared", "NEXT_STAGE", "SignalStage", "tail", None),
        ("SignalStage", "tail", "PRECEDES", "Model", "downstream", None),
    ])
    got = run_ea17(client, "sensor:00000")

    affected, only_via_me, _ = got["SignalStage"]
    assert affected == 3, f"all three stages are reachable: {got}"
    assert only_via_me == 1, (
        f"only `mine` has no other feeder; `shared` and `tail` keep running "
        f"because sensor:00001 still feeds them. Got {got['SignalStage']}. "
        f"A reachability-only query reports 3 here, which is the bug this "
        f"column exists to prevent."
    )
    assert got["Model"] == (1, 0, 4), (
        f"the model is reachable but does not stop -- it is fed through a stage "
        f"with a live feeder. Got {got['Model']}"
    )


def test_a_board_shared_through_a_different_deployment_is_not_exclusive(engine_factory):
    """The case the firebreak fixture cannot reach, and a real bug it hid.

    `test_a_shared_stage_is_a_firebreak` shares a *stage*, so both sensors
    converge before any deployment exists. This shares only the **board**: two
    sensors, two independent model/variant/deployment chains, landing on one
    board. Nothing upstream is shared.

    The Board leg's `OPTIONAL MATCH` used to re-bind `m`, `v` **and** `d` from
    the main pattern, which forced the alternative path through the *same*
    deployment. A board hosts many deployments, so another sensor reaching it
    by a different one was invisible and `only_via_me` over-reported -- it said
    the board stops with this sensor when it does not.

    It survived the ground-truth sweep because `only_via_me` is 0 for every kind
    on the shipped graph (every stage has another feeder), so an over-report had
    nothing to be wrong about. A silent bug needs a fixture built for it.
    """
    client = engine_factory()
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": "sensor:00000", "name": "failing", "modality": "ECG"},
                  {"id": "sensor:00001", "name": "healthy", "modality": "PPG"}])
    create_nodes(client, GRAPH, "SignalStage",
                 [{"id": "stage-a", "name": "stage-a"},
                  {"id": "stage-b", "name": "stage-b"}])
    create_nodes(client, GRAPH, "Model",
                 [{"id": "model-a", "name": "model-a", "family": "cnn"},
                  {"id": "model-b", "name": "model-b", "family": "cnn"}])
    create_nodes(client, GRAPH, "ModelVariant",
                 [{"id": "variant-a", "precision": "int8"},
                  {"id": "variant-b", "precision": "int8"}])
    create_nodes(client, GRAPH, "Deployment",
                 [{"id": "deploy-a", "fits": 1}, {"id": "deploy-b", "fits": 1}])
    create_nodes(client, GRAPH, "Board",
                 [{"id": "shared-board", "name": "shared-board"}])
    edges = []
    for sensor, stage, model, variant, deploy in (
            ("sensor:00000", "stage-a", "model-a", "variant-a", "deploy-a"),
            ("sensor:00001", "stage-b", "model-b", "variant-b", "deploy-b")):
        edges += [
            ("Sensor", sensor, "FEEDS", "SignalStage", stage, None),
            ("SignalStage", stage, "PRECEDES", "Model", model, None),
            ("ModelVariant", variant, "VARIANT_OF", "Model", model, None),
            ("Deployment", deploy, "OF_VARIANT", "ModelVariant", variant, None),
            ("Deployment", deploy, "ON_BOARD", "Board", "shared-board", None),
        ]
    create_edges(client, GRAPH, edges)

    got = run_ea17(client, "sensor:00000")
    assert got["Board"] == (1, 0, 5), (
        f"the board is reachable from sensor:00001 through deploy-b, so it does "
        f"not stop when sensor:00000 does: expected (1 affected, 0 only_via_me, "
        f"depth 5), got {got['Board']}"
    )
    assert got["Deployment"] == (1, 1, 4), (
        f"`deploy-a` really is reachable only through sensor:00000, so this one "
        f"must stay exclusive -- otherwise the fix over-corrected: {got['Deployment']}"
    )


def test_fixed_depth_misses_the_far_end_of_a_long_chain(engine_factory):
    """The reason `EA17` is `*0..` and not `*0..N`.

    A chain of 12 stages, longer than any bound someone would write by hand. The
    bounded query finds the near end and stops; the unbounded one reaches the
    model at the far end. This is the assertion that makes "unbounded" a
    requirement rather than a preference.

    Purpose-built rather than taken from the shipped graph, and the reason is
    not the one an earlier version of this comment gave. That version cited
    "`*0..3` loses stages for 9 of 14 sensors" on the shipped graph, which is
    true and proves nothing: pipelines there are 3-5 stages by construction, so
    a bound of 3 truncates at most two of a sensor's *own* stages and the rest
    of the loss is other sensors' stages reached through the shared-pool
    leakage. The measurement supported the bound as much as it refuted it.

    So the argument for `*0..` is made here instead, on a graph where the chain
    length is the only variable: 12 stages, one sensor, nothing shared.
    """
    client = engine_factory()
    depth = 12
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": "sensor:00000", "name": "long-chain", "modality": "ECG"}])
    create_nodes(client, GRAPH, "SignalStage",
                 [{"id": f"stage{i}", "name": f"stage-{i}"} for i in range(depth)])
    create_nodes(client, GRAPH, "Model",
                 [{"id": "far-model", "name": "far-model", "family": "cnn"}])
    edges = [("Sensor", "sensor:00000", "FEEDS", "SignalStage", "stage0", None)]
    edges += [("SignalStage", f"stage{i}", "NEXT_STAGE", "SignalStage", f"stage{i + 1}",
               None) for i in range(depth - 1)]
    edges.append(("SignalStage", f"stage{depth - 1}", "PRECEDES", "Model",
                  "far-model", None))
    create_edges(client, GRAPH, edges)

    unbounded = BY_ID["EA17"]["cypher"]
    bounded = unbounded.replace("*0..]", "*0..3]")

    got_unbounded = {r[0]: (r[1], r[2]) for r in client.query(unbounded, GRAPH).records}
    got_bounded = {r[0]: (r[1], r[2]) for r in client.query(bounded, GRAPH).records}

    assert got_unbounded["SignalStage"][0] == depth, (
        f"unbounded should reach all {depth} stages, got {got_unbounded}")
    assert got_bounded["SignalStage"][0] == 4, (
        f"*0..3 reaches the first four stages only, got {got_bounded}")
    assert got_unbounded["Model"][0] == 1, (
        f"the model at the far end is downstream and must be reported: "
        f"{got_unbounded}")
    # One sensor, so nothing has another feeder: everything reachable is also
    # exclusive. That is the contrast with the firebreak fixture above.
    assert got_unbounded["SignalStage"][1] == depth, (
        f"with a single sensor every stage stops with it: {got_unbounded}")
    assert got_bounded.get("Model", (0, 0, 0))[0] == 0, (
        f"a bound of 3 cannot reach a model {depth} stages away, yet the bounded "
        f"query reported {got_bounded.get('Model')}. If this fails, the bound is "
        f"no longer doing what the test assumes and the comparison is void.")


def test_a_task_stops_when_it_loses_the_last_sensor_of_a_modality(engine_factory):
    """A task's other sensors are alternatives only if they share its modality.

    `etl/generate.py:425` links a `ClinicalTask` to **every** sensor whose
    modality it requires, so "this task has other sensors" and "this task has a
    replacement for this sensor" are different statements. `EA17` used to treat
    them as one, and reported a task as surviving whenever it had any second
    sensor at all.

    The fixture is the smallest graph that separates them: one task requiring
    ECG and PPG, with two ECG sensors and one PPG sensor.

    - lose an ECG sensor -> the other ECG sensor covers it, task keeps running.
    - lose the PPG sensor -> nothing else supplies PPG, task stops.

    Both are `affected`; only the second is `only_via_me`. Under the old rule
    neither was, because the task had three sensors. At `--scale 1.0` that rule
    found 1 exclusive task where there are 7.
    """
    client = engine_factory()
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": "sensor:00000", "name": "ecg-a", "modality": "ecg"},
                  {"id": "sensor:00001", "name": "ecg-b", "modality": "ecg"},
                  {"id": "sensor:00002", "name": "ppg-only", "modality": "ppg"}])
    create_nodes(client, GRAPH, "ClinicalTask",
                 [{"id": "task:00000", "name": "dual", "category": "monitoring",
                   "latency_budget_ms": 100, "min_sensitivity": 0.9}])
    create_edges(client, GRAPH, [
        ("ClinicalTask", "task:00000", "REQUIRES_SENSOR", "Sensor", s, None)
        for s in ("sensor:00000", "sensor:00001", "sensor:00002")])

    def task_row(sensor_id):
        return run_ea17(client, sensor_id).get("ClinicalTask")

    redundant = task_row("sensor:00000")
    assert redundant == (1, 0, 1), (
        f"losing one of two ECG sensors must leave the task affected but not "
        f"exclusive -- the other ECG sensor still supplies the modality. "
        f"Got {redundant}"
    )
    last_of_its_kind = task_row("sensor:00002")
    assert last_of_its_kind == (1, 1, 1), (
        f"the PPG sensor is the only one the task has, so losing it stops the "
        f"task and `only_via_me` must be 1. Got {last_of_its_kind}. A rule "
        f"keyed on 'is this the task's only sensor' returns 0 here, because "
        f"the task has three."
    )


def test_ea07s_fixed_bound_is_lossy_and_that_is_a_known_trade(engine_factory):
    """The argument of this PR, applied to the query it did not change.

    `EA17` uses `*0..` because a fixed bound on a chain of unknown length is
    wrong -- that is what `test_fixed_depth_misses_the_far_end_of_a_long_chain`
    above demonstrates. `EA07` walks the same chain and still says `*0..3`.

    That is a deliberate trade, not an oversight, and the reason is note 12:
    `*0..` is the construct the 1.7.0 HTTP server refuses, so lifting `EA07`'s
    bound would make it a second embedded-only query. `EA17` is already the one
    query that cannot be asked over HTTP.

    The cost is real and is asserted here rather than described. On the shipped
    fleet at `--scale 1.0` the longest `NEXT_STAGE` chain is 13 hops and the two
    spellings return different top-tens; this fixture reproduces that
    deterministically with one chain of five stages. If a future engine answers
    `*0..` over HTTP, this test still passes -- it is about the bound, not the
    server -- but the trade it records is the thing to revisit.
    """
    client = engine_factory()
    stages = [{"id": f"st:{i}", "name": f"stage-{i}"} for i in range(5)]
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": "sensor:00000", "name": "far", "modality": "ecg",
                   "sample_rate_hz": 500}])
    create_nodes(client, GRAPH, "SignalStage", stages)
    create_nodes(client, GRAPH, "Model",
                 [{"id": "model:far", "name": "far-model", "family": "cnn"}])
    create_nodes(client, GRAPH, "ModelVariant",
                 [{"id": "var:far", "precision": "int8"}])
    create_nodes(client, GRAPH, "Deployment",
                 [{"id": "dep:far", "fits": 1, "latency_ms": 1.0}])
    create_nodes(client, GRAPH, "Board", [{"id": "board:far", "name": "far-board"}])

    edges = [("Sensor", "sensor:00000", "FEEDS", "SignalStage", "st:0", None)]
    edges += [("SignalStage", f"st:{i}", "NEXT_STAGE", "SignalStage", f"st:{i + 1}", None)
              for i in range(4)]
    edges += [
        ("SignalStage", "st:4", "PRECEDES", "Model", "model:far", None),
        ("ModelVariant", "var:far", "VARIANT_OF", "Model", "model:far", None),
        ("Deployment", "dep:far", "OF_VARIANT", "ModelVariant", "var:far", None),
        ("Deployment", "dep:far", "ON_BOARD", "Board", "board:far", None),
    ]
    create_edges(client, GRAPH, edges)

    shipped = BY_ID["EA07"]["cypher"]
    assert "*0..3" in shipped, (
        "EA07 no longer carries a fixed bound. If it was lifted to `*0..`, "
        "check note 12 -- the 1.7.0 server refuses that -- and replace this "
        "test with one that pins the new behaviour."
    )
    bounded = client.query(shipped, GRAPH).records
    unbounded = client.query(shipped.replace("*0..3", "*0.."), GRAPH).records

    assert not bounded, (
        f"the model is five hops from the sensor, so `*0..3` must miss it. "
        f"Got {bounded} -- if this fixture's chain got shorter the test is "
        f"no longer demonstrating anything."
    )
    assert len(unbounded) == 1 and unbounded[0][3] == "far-model", (
        f"`*0..` must reach it, or the comparison says nothing about the "
        f"bound. Got {unbounded}"
    )
