"""`EA17` against ground truth, and against the fixed-depth mistake it avoids (#35).

The blast-radius question -- *this sensor stops, what stops with it?* -- walks
`Sensor -[:FEEDS]-> SignalStage -[:NEXT_STAGE*]-> ... -[:PRECEDES]-> Model` and
on to variants, deployments and boards. The DSP chain is of **unknown length**,
which is what makes it a graph question: a relational schema needs a fixed
number of self-joins, and so does `*0..3`.

Two things are checked here, and the second is the one that matters.

## What `EA17` carries, in full

The catalog entry points here for these, because each is a paragraph:

1. **Note 1's trailing-rebind shape.** Each `OPTIONAL MATCH` leg re-binds, in
   trailing position, the variable its own `MATCH` bound one line above -- `x`
   in the stage leg, `m`, `d` and `b` in the others -- which `CLAUDE.md` says
   produces an unenforced join. It does not here: compared against a Python
   breadth-first search at `--scale 1.0`, all 14 sensors, both counts and the
   depth, **0 disagreements** (`test_ea17_matches_ground_truth_at_full_scale`,
   `--full-scale`). The fixture-scale comparison in this module runs every
   time and would catch an inflated count; the full-scale one is what settles
   *cardinality*, which note 1 says a small graph cannot. Do not edit those
   patterns without re-running it.
2. **It needs the `samyama>=1.7.1` floor #105 wrote and #104 landed.** Each
   leg's second `WITH` introduces all-new aliases -- note 10's shape, which
   0.6.x rejects, so below the floor `EA17` does not run at all;
   `tests/test_engine_version.py` fails if the floor is lowered.
3. **`+1/+2/+4/+5` are single schema-fixed hops**, not an assumed chain
   length: `FEEDS`, `PRECEDES`, `VARIANT_OF`+`OF_VARIANT`, `ON_BOARD`. The
   variable part is `size(r)` over `NEXT_STAGE`; a schema change that inserts
   a hop must move them.
4. **The sensor id appears ten times.** Retarget with `.replace`, never by hand;
   `test_retargeting_replaces_every_occurrence_of_the_sensor_id`
   (in `tests/test_blast_radius_semantics.py`) pins it. If the
   id names no node every leg returns nothing, so the answer is an empty blast
   radius rather than an error -- the same shape `EA01` and `EA06` have.

**The answer is right.** `EA17` reports two counts and a nearest depth per kind.
Those are compared against a breadth-first search computed in Python from the
`Fleet`, for every sensor -- not against a remembered number, and not against a
second Cypher query, which would only prove the engine agrees with itself.

**What a `Fleet` ground truth cannot catch, and how that bit here.** The BFS
below follows the same relationship types in the same directions as the query.
When the query and the truth share a *modelling* assumption, agreement proves
the traversal walks what it was told to walk -- not that walking that is the
answer to the question. The first version of `EA17` reported plain reachability
and matched this BFS exactly for all 14 sensors, and both were wrong together:
`etl/generate.py` samples each sensor's chain (`rng.sample(stages, ...)`)
from one shared 16-stage pool,
so an unbounded walk leaves the sensor's own 3-5 stages and reaches 15 of 16.
"What stops when this sensor fails" came back as the whole fleet.

The fix is a second number, `only_via_me`, and the check for it is deliberately
*not* a BFS: `Truth.exclusive` computes reachability from every **other** sensor
and subtracts, which is a different computation rather than the same one in
Python. `test_a_shared_stage_is_a_firebreak`
(`tests/test_blast_radius_semantics.py`) then pins the case the shipped
graph cannot show.

**A fixed bound is wrong, on a graph built to show it.**
`test_fixed_depth_misses_the_far_end_of_a_long_chain`
(`tests/test_blast_radius_semantics.py`)
builds a chain longer than any bound a person would guess and asserts `*0..3`
misses the far end while `*0..` finds it. Without that, "unbounded" is a style
choice; with it, it is a requirement. It is a purpose-built fixture because the
shipped graph is a weak witness: `*0..3` loses stages for 9 of its 14 sensors
but happens to lose no *model*, so a test that only walked the shipped graph
could pass with a bound in place.
"""
from __future__ import annotations

import collections
import re

import pytest

from benchmarks.queries import BY_ID, EA17_SUBJECT
from etl.helpers import create_edges, create_nodes, cypher_literal

GRAPH = "default"
SEED = 4242
SCALE = 0.3

# Edges EA17 walks, and the direction it walks them in. `VARIANT_OF` and
# `OF_VARIANT` point *up* (variant -> model, deployment -> variant), so
# downstream traversal follows them backwards -- getting this wrong is the
# easiest way to write a ground truth that agrees with a broken query.
DOWNSTREAM = {"FEEDS": "forward", "NEXT_STAGE": "forward", "PRECEDES": "forward",
              "VARIANT_OF": "backward", "OF_VARIANT": "backward",
              "ON_BOARD": "forward"}


@pytest.fixture(scope="module")
def operators():
    """The cached ONNX catalogue, skipping in **setup** when it is absent.

    A fixture, not a `try` inside `build_and_load`: that helper is also called
    from the `--full-scale` test body, where a missing-data skip would slip past
    `--no-skips` (`tests/test_environment_skips.py`).
    """
    from etl import onnx_catalog as oc

    try:
        return oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


def build_and_load(engine_factory, ops, seed: int, scale: float):
    """One loaded graph and the fleet behind it.

    Shared by the module fixture and the `--full-scale` test, which had this
    body twice with different hardcoded seeds -- so the two could drift apart
    and the slow check would stop being the same comparison as the fast one.

    Each `engine_factory()` is a fresh `SamyamaClient.embedded()`, which is its
    own in-memory graph. That is why a module-scoped graph and a session-scoped
    factory do not interfere: a later client cannot reach into this one's store.
    """
    from etl import generate as gen
    from etl import real_layer
    from etl.loader import NODE_LABELS

    client = engine_factory()
    fleet = gen.generate(seed=seed, scale=scale, operators=ops)
    real_layer.build_real(fleet, ops)
    # Loudly, rather than by omission. Iterating `NODE_LABELS` silently drops
    # any label the generator grew since, and the ground truth below is
    # computed from `fleet` -- so the comparison would quietly stop covering
    # the new label instead of failing.
    unknown = sorted(set(fleet.nodes) - set(NODE_LABELS))
    assert not unknown, (
        f"the fleet holds {unknown}, which `etl.loader.NODE_LABELS` does not "
        f"list, so those nodes would never be loaded while the ground truth "
        f"below still counts them. Add them to NODE_LABELS.")
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client, fleet


@pytest.fixture(scope="module")
def loaded(engine_factory, operators):
    return build_and_load(engine_factory, operators, seed=SEED, scale=SCALE)


class Truth:
    """The ground truth for one fleet, computed in Python, built once.

    A class rather than free functions that stamp `fleet._downstream_adjacency`
    and `fleet._reachable_by_sensor` onto the caller's object. Three things
    needed caching -- the adjacency map, the label index, and each sensor's
    reachable set -- and the private-attribute form cached two of them, missed
    the third, and mutated a `Fleet` the test did not own. Here the lifetime is
    the object's: build one per fleet, throw it away with the fleet.

    The costs are real, not hypothetical. `exclusive` walks every *other*
    sensor, so 14 sensors meant 196 passes over 76,303 edges at `--scale 1.0`
    without the per-sensor cache, and `labels` was rebuilt over all 25,150 rows
    once per sensor inside that loop.
    """

    def __init__(self, fleet):
        self.fleet = fleet
        self.adjacency = collections.defaultdict(list)
        for _sl, src, rel, _tl, tgt, _p in fleet.edges:
            direction = DOWNSTREAM.get(rel)
            if direction == "forward":
                self.adjacency[src].append(tgt)
            elif direction == "backward":
                self.adjacency[tgt].append(src)
        self.labels = {row["id"]: label
                       for label, rows in fleet.nodes.items() for row in rows}
        self.requires = collections.defaultdict(set)
        for _sl, src, rel, _tl, tgt, _p in fleet.edges:
            if rel == "REQUIRES_SENSOR":
                self.requires[src].add(tgt)
        # The one-hop invariant `by_kind` reads below, checked where the data
        # is rather than assumed where it is used: every `REQUIRES_SENSOR`
        # endpoint is a `Sensor`, so a task that requires this sensor is
        # adjacent to it and its blast-radius depth is 1. A schema change that
        # routed this through a stage would make that depth wrong -- and wrong
        # in the same direction as the query, which is the pair of errors that
        # cancel and leave a green test.
        # Both ends. Checking only the target left the other half of "one hop
        # from a ClinicalTask to a Sensor" unexamined: an edge from something
        # else *to* a Sensor would satisfy the target check and still mean the
        # depth of 1 was measuring a different path.
        off_sensor = {tgt for sensors in self.requires.values() for tgt in sensors
                      if self.labels.get(tgt) != "Sensor"}
        off_task = {src for src in self.requires
                    if self.labels.get(src) != "ClinicalTask"}
        assert not off_sensor, (
            f"REQUIRES_SENSOR points at {sorted(off_sensor)[:3]}, which is not "
            f"a Sensor, so ClinicalTask is no longer one hop from the sensor "
            f"and `by_kind`'s depth of 1 is wrong. Derive it instead.")
        assert not off_task, (
            f"REQUIRES_SENSOR starts at {sorted(off_task)[:3]}, which is not a "
            f"ClinicalTask. `by_kind` counts these as tasks at depth 1, so the "
            f"kind is as load-bearing as the hop count.")
        self.modality = {row["id"]: row.get("modality")
                         for row in fleet.nodes.get("Sensor", ())}
        self.reachable = {row["id"]: self._reachable_from(row["id"])
                          for row in fleet.nodes["Sensor"]}

    def distances(self, sensor_id: str) -> dict[str, int]:
        """Shortest hop count from a sensor to everything downstream, in Python."""
        distance = {sensor_id: 0}
        queue = collections.deque([sensor_id])
        while queue:
            node = queue.popleft()
            for neighbour in self.adjacency.get(node, ()):
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
        return distance

    def _reachable_from(self, sensor_id: str) -> set:
        return {node for node in self.distances(sensor_id) if node != sensor_id}

    def exclusive(self, sensor_id: str) -> set:
        """Everything reachable from this sensor and from **no other** sensor.

        A set difference, not a second BFS with the answer baked in: it walks
        the other sensors independently and subtracts. That is what makes it a
        check on `only_via_me` rather than a restatement of it.

        ClinicalTask is handled by its own rule -- a task stops only if this
        sensor is the *only* sensor it requires -- because it is a consumer of
        the sensor rather than something downstream of it.
        """
        others = set()
        for other_id, reached in self.reachable.items():
            if other_id != sensor_id:
                others |= reached
        exclusive = self.reachable[sensor_id] - others
        exclusive |= {task for task in self.requires
                      if self.task_stops_without(task, sensor_id)}
        return exclusive

    def task_stops_without(self, task: str, sensor_id: str) -> bool:
        """Whether losing this sensor leaves the task short of a modality.

        Not "is this the task's only sensor". `etl/generate.py` links a task
        to every sensor whose modality it requires, so other sensors are
        alternatives only when they supply the same modality: a task needing
        ECG and PPG, with two ECG sensors and one PPG, survives losing an ECG
        sensor and stops on losing the PPG one.

        **This half restates the query's rule rather than checking it
        independently.** `EA17`'s ClinicalTask leg matches
        `o.modality = s.modality` and so does this; there is no second way to
        say "the task loses a modality". What it proves is that the engine
        agrees with Python at full cardinality -- a lost join or a mis-grouped
        aggregate still shows up. The *rule* is checked in
        `tests/test_blast_radius_semantics.py`, on a fixture where the answer
        is known by construction. The other four kinds are genuinely
        independent: `exclusive` walks every other sensor and subtracts.
        """
        sensors = self.requires[task]
        if sensor_id not in sensors:
            return False
        mine = self.modality.get(sensor_id)
        return not any(other != sensor_id and self.modality.get(other) == mine
                       for other in sensors)

    def by_kind(self, sensor_id: str) -> dict[str, tuple[int, int, int]]:
        """`{kind: (affected, only_via_me, nearest_depth)}`, EA17's shape."""
        distance = self.distances(sensor_id)
        exclusive = self.exclusive(sensor_id)
        per_kind: dict[str, list[int]] = collections.defaultdict(list)
        only: dict[str, int] = collections.defaultdict(int)
        for node_id, depth in distance.items():
            if node_id == sensor_id:
                continue
            kind = self.labels.get(node_id)
            if kind in ("SignalStage", "Model", "Deployment", "Board"):
                per_kind[kind].append(depth)
                if node_id in exclusive:
                    only[kind] += 1

        # ClinicalTask points *at* the sensor (`REQUIRES_SENSOR`), so it is not
        # reached by the downstream walk above -- it is a consumer that stops
        # for the same reason. Depth 1: it is adjacent to the sensor.
        tasks = {task for task, sensors in self.requires.items()
                 if sensor_id in sensors}
        if tasks:
            # Depth 1 is the schema's one hop, and the invariant that makes
            # it one is asserted in `__init__` rather than assumed here.
            per_kind["ClinicalTask"] = [1] * len(tasks)
            only["ClinicalTask"] = len(tasks & exclusive)

        return {kind: (len(depths), only[kind], min(depths))
                for kind, depths in per_kind.items()}


# Ten: five legs, each naming the subject twice (`s.id =` and `o.id <>`).
# Exact rather than a floor, so adding or removing a leg has to be an edit here
# too -- a retarget that silently covered four legs of five is the failure this
# number exists to catch.
EA17_SUBJECT_OCCURRENCES = 10


def retargeted_ea17(sensor_id: str) -> str:
    """`EA17` asking about `sensor_id` instead of the hardcoded subject.

    The quoted literal, not the bare id. `replace(EA17_SUBJECT, ...)` is a
    substring match: it would also rewrite a longer id that merely starts with
    it, and would rewrite the string wherever else it appeared. Ten
    occurrences, all of them quoted, so anchoring on the quotes is exact.

    A function rather than a line inside `run_ea17`, because
    `tests/test_blast_radius_semantics.py` retargets too -- and pinned the
    *bare* replace while this performed the quoted one, so the test and the
    code were checking different things.
    """
    # Escaped, not interpolated. Building the literal with an f-string meant
    # an id carrying a quote ended it early: `'x") OR true //'` produced
    # `WHERE s.id = "x") OR true //"`, which closes the predicate and comments
    # out the rest of the line. Harmless while every caller is a test, and
    # not harmless the moment one of these helpers is promoted beside the
    # catalog -- which is exactly what happened to `EA21`'s (#118).
    #
    # A non-`str` is refused rather than stringified: a caller reaching for
    # `EA21`'s signature would pass a list, `str(["a"])` would become
    # `"['a']"`, and the retarget would ask about a sensor that cannot exist
    # while looking like it worked.
    if not isinstance(sensor_id, str):
        raise TypeError(
            f"a sensor id must be a string, not {type(sensor_id).__name__}: "
            f"`EA17` retargets one sensor, unlike `EA21` which takes a set.")

    original = BY_ID["EA17"]["cypher"]
    subject, wanted = cypher_literal(EA17_SUBJECT), cypher_literal(sensor_id)
    # Count, not membership. `wanted in cypher` is trivially true when
    # `sensor_id` *is* `EA17_SUBJECT` -- which is the first sensor in the
    # fleet and the one most callers pass -- so the "catalog subject moved"
    # check never fired for the commonest case.
    occurrences = original.count(subject)
    assert occurrences == EA17_SUBJECT_OCCURRENCES, (
        f"`EA17` names `{EA17_SUBJECT}` {occurrences} times, not "
        f"{EA17_SUBJECT_OCCURRENCES}. Either a leg was added or removed -- in "
        f"which case update EA17_SUBJECT_OCCURRENCES here -- or the catalog's "
        f"subject moved and this retarget no longer rewrites every leg.")
    cypher = original.replace(subject, wanted)
    assert cypher.count(wanted) == occurrences, (
        f"retargeting to {sensor_id} rewrote {cypher.count(wanted)} of "
        f"{occurrences} occurrences; a partial rewrite makes one leg answer "
        f"about a different sensor than the rest")
    return cypher


def assert_one_row_per_kind(rows) -> None:
    """One row per kind, or a dict built from `rows` hides half the answer.

    `EA17` is five `UNION ALL` legs each returning one row, so two rows for a
    kind means a leg stopped aggregating -- which a dict comprehension would
    swallow. Shared with `tests/test_blast_radius_semantics.py`, which checks
    rows it fetched itself: the invariant is `EA17`'s, not one test's.
    """
    kinds = [row[0] for row in rows]
    assert len(kinds) == len(set(kinds)), (
        f"EA17 returned more than one row for some kind: {kinds}. Each leg "
        f"aggregates to a single row; this mapping would keep only the last."
    )


def run_ea17(client, sensor_id: str) -> dict[str, tuple[int, int, int]]:
    rows = client.query(retargeted_ea17(sensor_id), GRAPH).records
    assert_one_row_per_kind(rows)
    return {row[0]: (row[1], row[2], row[3]) for row in rows}


def test_ea17_matches_ground_truth_for_every_sensor(loaded):
    """Every sensor, both numbers, against a Python BFS."""
    client, fleet = loaded
    sensors = [row["id"] for row in fleet.nodes["Sensor"]]
    assert sensors, "the fixture has no sensors, so this proves nothing"

    truth = Truth(fleet)
    disagreed = []
    for sensor_id in sensors:
        got, expected = run_ea17(client, sensor_id), truth.by_kind(sensor_id)
        if got != expected:
            disagreed.append(f"{sensor_id}: EA17 {got} != Python {expected}")
    assert not disagreed, "EA17 disagrees with the Fleet:\n  " + "\n  ".join(disagreed)


def test_ea17_reports_every_kind_the_question_names(loaded):
    """The four kinds every sensor has downstream, asserted by name.

    #35 names five, and `ClinicalTask` is deliberately not among the four.
    `etl/generate.py` emits `REQUIRES_SENSOR` only when the task's modality
    matches, so a given sensor may legitimately have no task and EA17 correctly
    returns no `ClinicalTask` row. Asserting it here would be asserting a
    property of the fixture rather than of the query; `test_ea17_matches_
    ground_truth_for_every_sensor` covers it wherever it does occur.

    The dict comparison there is full equality, so a dropped kind already
    fails. This one is kept as the targeted signal: it names *which* kind went
    missing, where the other prints two dicts to diff.
    """
    client, fleet = loaded
    sensor_id = fleet.nodes["Sensor"][0]["id"]
    got = run_ea17(client, sensor_id)
    for kind in ("SignalStage", "Model", "Deployment", "Board"):
        assert kind in got, f"EA17 does not report {kind}: {sorted(got)}"
    assert got["SignalStage"][2] == 1, "the first stage is one hop from the sensor"


def test_ea17_matches_ground_truth_at_full_scale(request, engine_factory,
                                                 operators):
    """The same comparison at `--scale 1.0`, because 0.3 cannot settle it.

    Every `OPTIONAL MATCH` leg re-binds a variable the first `MATCH` already
    bound, in trailing position -- `x`, `m`, `d`, `b`. `CLAUDE.md` is explicit
    that this is the shape whose join "isn't enforced" and that "at least one
    bug only appears once cardinalities are real", so a passing run at
    `SCALE = 0.3` plus three-node fixtures proves nothing about it.

    It matters here more than usual because the failure would be *quiet and
    self-confirming*. An unenforced join inflates `count(o.id)`, pushes `others`
    above zero, and drives `only_via_me` **down** -- reporting that nothing
    stops exclusively. That is also the correct answer for a shared 16-stage
    pool, so the bug and the finding are indistinguishable from the outside.

    **Measured 2026-09-10 at scale 1.0** -- 25,150 nodes, 76,303 edges, all 14
    sensors, both counts and the nearest depth: **0 disagreements**. The join
    holds and `only_via_me = 0` is the real answer, not an artifact.

    Opt-in via `--full-scale`: the load costs about three minutes. Skipped from
    the body rather than a fixture on purpose -- `conftest.py` converts only
    setup-phase skips to failures under `--no-skips`, and "you did not ask for
    the slow check" is not a broken environment.
    """
    if not request.config.getoption("--full-scale"):
        pytest.skip("needs --full-scale (about three minutes to build the graph)")

    client, fleet = build_and_load(engine_factory, operators, seed=SEED,
                                   scale=1.0)

    truth = Truth(fleet)
    disagreed = []
    for row in fleet.nodes["Sensor"]:
        sensor_id = row["id"]
        got, expected = run_ea17(client, sensor_id), truth.by_kind(sensor_id)
        if got != expected:
            disagreed.append(f"{sensor_id}: EA17 {got} != Python {expected}")
    assert not disagreed, (
        "EA17 disagrees with the Fleet at full cardinality, which is where the "
        "trailing-rebind join fails if it fails:\n  " + "\n  ".join(disagreed))


def test_ea17_binds_nothing_it_does_not_use():
    """A named node in a pattern that nothing references reads as a join.

    `EA17` is five legs of one long pattern, and the question about it is
    which variables are joined to which. A name that is bound and never used
    makes that harder to answer; an anonymous node says "this hop exists and I
    do not refer to it", which is the truth.

    Per leg, because `UNION ALL` legs share no scope: a name used in leg 2 is
    still unused in leg 4.
    """
    # String literals stripped first. Every binding here is one or two letters,
    # so `\bs\b` inside a quoted `"SignalStage"` or an id would count as a use
    # and hide the thing this looks for.
    cypher = re.sub(r'"[^"]*"', '""', BY_ID["EA17"]["cypher"])
    unused = {}
    for n, leg in enumerate(cypher.split("UNION ALL"), 1):
        for name in set(re.findall(r"[(\[](\w+):", leg)):
            if len(re.findall(rf"\b{name}\b", leg)) == 1:
                unused.setdefault(n, []).append(name)
    assert not unused, (
        f"EA17 binds names it never uses: {unused}. Make them anonymous -- "
        f"`(:SignalStage)` rather than `(st:SignalStage)` -- so the pattern "
        f"says which variables are actually joined."
    )


def test_ea17_never_compares_an_id_with_a_bare_inequality():
    """Engine note 8b: `<>` matches null properties.

    Every `<>` in `EA17` excludes the subject sensor from its own alternatives.
    Unguarded, a null `id` would satisfy it, count as another feeder, and turn
    a real `only_via_me` into a zero -- the query would under-report what
    stops, silently, in the direction that looks like good news.
    """
    cypher = BY_ID["EA17"]["cypher"]
    # Per `WHERE` clause. The guard has to sit in the same clause as the
    # comparison it protects -- a guard on a different property, or in the
    # leg's other `WHERE`, constrains different rows and protects nothing.
    unguarded = []
    for n, leg in enumerate(cypher.split("UNION ALL"), 1):
        for clause in re.split(r"\b(?:WITH|RETURN|MATCH|OPTIONAL MATCH)\b", leg):
            if "WHERE" not in clause:
                continue
            where = clause[clause.index("WHERE"):]
            unguarded.extend(
                f"leg {n}: {prop}"
                for prop in re.findall(r"(\w+\.\w+)\s*<>", where)
                if not re.search(rf"{re.escape(prop)}\s+IS NOT NULL", where))
    assert not unguarded, (
        f"`<>` on a property with no `IS NOT NULL` for that same property in "
        f"the same leg: {unguarded}. Engine note 8b says `<>` matches nulls."
    )
    assert re.findall(r"\w+\.\w+\s*<>", cypher), (
        "no `<>` in EA17 at all -- this test would pass vacuously, so it "
        "checks the comparison it guards is still there"
    )


def test_a_quoted_sensor_id_cannot_break_out_of_the_retarget(engine_factory):
    """An id carrying a quote is data, not Cypher (#118).

    `retargeted_ea17` built its literal with an f-string, so
    `'x") OR true //'` produced `WHERE s.id = "x") OR true //"`: the predicate
    closes early and the rest of the line is commented out. Every caller here
    is a test, so nothing was exposed -- but `EA21`'s equivalent helper was
    promoted into `benchmarks/queries.py`, where an MCP tool calls it with a
    live alert set, and the same shape became a real injection. This is the
    same one-line fix applied before that happens again.

    The fixture holds one sensor the payload does not name, so an escape is
    visible as rows coming back at all.
    """
    client = engine_factory()
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": EA17_SUBJECT, "modality": "ecg"}])
    create_nodes(client, GRAPH, "SignalStage",
                 [{"id": "stage:only", "kind": "filter"}])
    create_edges(client, GRAPH, [
        ("Sensor", EA17_SUBJECT, "FEEDS", "SignalStage", "stage:only", None)])

    cypher = retargeted_ea17('x") OR true //')
    assert '"x) OR true //"' in cypher, (
        f"expected `cypher_literal` to strip the quote; the first leg reads "
        f"{[ln for ln in cypher.splitlines() if 's.id =' in ln][:1]}")
    assert [list(row) for row in client.query(cypher, GRAPH).records] == [], (
        "a quoted id escaped the literal and matched a sensor it does not name")

    with pytest.raises(TypeError, match="must be a string"):
        retargeted_ea17([EA17_SUBJECT])
