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
in `tests/blast_radius_truth.py` follows the same relationship types in the
same directions as the query.
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

import re

import pytest

from benchmarks.queries import BY_ID, EA17_SUBJECT
from etl.helpers import create_edges, create_nodes, cypher_literal

# `tests/blast_radius_truth.py` holds `Truth` because nothing in it touches an
# engine, which is the seam this file was split on. Imported because the tests
# below use it -- not re-exported: the three sibling modules that import from
# this file (`test_blast_radius_semantics.py`, `test_root_cause.py`,
# `test_alerting_demo.py`) take fixtures and helpers, never `Truth`.
from tests.blast_radius_truth import Truth

GRAPH = "default"
SEED = 4242
SCALE = 0.3


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
    # any label the generator grew since, and `Truth` is computed from
    # `fleet` -- so the comparison would quietly stop covering the new label
    # instead of failing.
    unknown = sorted(set(fleet.nodes) - set(NODE_LABELS))
    assert not unknown, (
        f"the fleet holds {unknown}, which `etl.loader.NODE_LABELS` does not "
        f"list, so those nodes would never be loaded while the ground truth "
        f"still counts them. Add them to NODE_LABELS.")
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client, fleet


@pytest.fixture(scope="module")
def loaded(engine_factory, operators):
    return build_and_load(engine_factory, operators, seed=SEED, scale=SCALE)


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
    # Refused, not normalised. `cypher_literal` strips the characters that
    # would end the literal early, so `sensor:x"` renders as `"sensor:x"` --
    # a different sensor, answered silently. `benchmarks/queries.py` takes the
    # same line for `EA21`'s alert set.
    if wanted != f'"{sensor_id}"':
        raise ValueError(
            f"sensor id {sensor_id!r} cannot be asked about safely: escaping "
            f"it gives {wanted}, which names a different sensor.")
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


def test_a_quoted_sensor_id_is_refused_rather_than_normalised(engine_factory):
    """An id carrying a quote is refused, not silently turned into another (#118).

    `retargeted_ea17` built its literal with an f-string, so
    `'x") OR true //'` produced `WHERE s.id = "x") OR true //"`: the predicate
    closes early and the rest of the line is commented out. Every caller is a
    test today, so nothing was exposed -- but `EA21`'s equivalent helper was
    promoted into `benchmarks/queries.py`, where an MCP tool calls it with a
    live alert set, and there the same shape is a real injection.

    Escaping alone would fix the injection and introduce a quieter bug:
    `cypher_literal` strips the quote, so `sensor:x"` and `sensor:x` become
    the same literal and a caller asking about one is answered about the
    other. For a query a human pages on, answering about a different sensor is
    worse than an error, so ids that change under escaping are refused.

    The positive control is the point of the fixture: the same graph, asked
    about the id it really holds, must return rows. Without it an empty result
    from a broken fixture would look exactly like an injection that failed.
    """
    client = engine_factory()
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": EA17_SUBJECT, "modality": "ecg"}])
    create_nodes(client, GRAPH, "SignalStage",
                 [{"id": "stage:only", "kind": "filter"}])
    create_edges(client, GRAPH, [
        ("Sensor", EA17_SUBJECT, "FEEDS", "SignalStage", "stage:only", None)])

    # Positive control first: this fixture can answer at all.
    control = client.query(retargeted_ea17(EA17_SUBJECT), GRAPH).records
    assert control, (
        "the fixture returned nothing for its own sensor, so an empty result "
        "below would prove nothing about the payload")

    with pytest.raises(ValueError, match="names a different sensor"):
        retargeted_ea17('x") OR true //')


def test_a_sensor_id_that_is_not_a_string_is_refused(engine_factory):
    """A list, as a caller reaching for `EA21`'s signature would pass.

    Its own test rather than a tail on the injection one: a different mistake
    with a different failure. `str(["sensor:00000"])` would render as
    `"['sensor:00000']"` and ask about a sensor that cannot exist, looking
    like it worked.
    """
    with pytest.raises(TypeError, match="must be a string"):
        retargeted_ea17([EA17_SUBJECT])
