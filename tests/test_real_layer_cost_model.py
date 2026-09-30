"""Why `EA10` and `EA12` are empty on a freshly loaded real layer.

#114 recorded three queries answering differently on the real layer between
the embedded build and the 1.7.0 server: `EA08` returning fewer rows embedded,
`EA10` and `EA12` returning none. Two of those three are not a build
disagreement at all, and this module is the measurement that says so.

Six catalog queries mention `fits`, and `EA10` and `EA12` are the only two
that `fits` **alone** empties -- derived below by stripping the predicate,
not claimed here. (`EA03`, `EA06` and `EA07` filter on it too and `EA04`
reads it inside a `CASE`; all four are empty for a different reason, their
opening patterns binding nothing on the real layer.) `fits` is written by the
cost model in `etl/generate.py` and by nothing else -- `git log -S"fits" --
etl/real_layer.py` is empty across every revision -- so the real layer's
`Deployment` nodes, which are MLPerf Tiny submission rows, do not carry it.
The filter cannot match, on any engine, and `EA10` also selects four more
properties the real layer has never had.

So they are **generated-layer queries**: the same kind of absence the README
already documents for the six clinical-spine labels, not a divergence.

**The server half is measured too, and the divergence is real -- it is
engine note 8.** On 2026-09-30 the real layer was loaded into
`ghcr.io/samyama-ai/samyama-graph:1` (the 1.7.0 image the notes describe) two
ways. Into a *fresh* server it returns exactly what embedded 1.7.1 returns,
`EA08` row-for-row included. Onto a server that had previously held the full
fleet, the 73 MLPerf `Deployment` nodes come back carrying the generated cost
model's `fits`, `latency_ms` and `accelerator_kind` -- note 8, which embedded
1.7.1 fixed and the server did not -- and `EA10` and `EA12` return 5 rows
each. That is what #114 recorded, so its figure was right and its diagnosis
was the thing that was missing.

Everything below is about a graph loaded from nothing, which is what this
suite builds. The resurrection case cannot be reproduced embedded at all.

Those measurements cannot live in this suite, because they need a container.
They live in `docs/engine-notes.md` with their reproductions, the way every
other server measurement here does. What this module pins is the half that
runs everywhere:

- no real `Deployment` carries a cost-model field, and a generated one
  carries all four (the anti-vacuity half);
- `fits` alone is what empties `EA10` and `EA12`, derived by stripping the
  predicate from every query that mentions it rather than claimed in prose;
- `EA08`'s answer matches a ground truth computed in Python from the same
  source files;
- embedded 1.7.1's semantics for a comparison against a missing property.

That last one is not decoration. Run against both builds (with `n.v` rather
than `n.fits`, on a probe node of its own -- note 8b's section has that
table), the server **agrees** on `= 1` and **disagrees** on `<> 1`: engine
note 8b still live there after the embedded build fixed it. The `= 1` half is
why a *fresh* server answers `EA10` and `EA12` the same way embedded does,
and note 8 is why a reused one does not.
"""
from __future__ import annotations

import collections

import pytest

from benchmarks.queries import BY_ID
from tests.test_real_layer_shape import (
    GRAPH,
    real_only,  # noqa: F401 -- shared fixture
)

# The four properties `EA10` reads off a `Deployment`, all of them produced by
# the cost model in `etl/generate.py`. Named from the query rather than from
# the generator: if the generator renames one, the query breaks and this list
# is where the rename is noticed. (`EA10` returns five columns; the fifth,
# `count(d)`, reads no property.)
COST_MODEL_FIELDS = {"fits", "accelerator_kind", "fallback_fraction",
                     "latency_ms"}

# Every textual form of a `fits` predicate in the catalog. Stripping all of
# them is deliberately cruder than stripping only the `WHERE`-level ones: the
# question the test below asks is "would this query return rows if `fits` were
# not mentioned at all", and a `CASE` arm mentioning it counts.
FITS_PREDICATES = ("WHERE d.fits = 1\n", " AND d.fits = 1", " AND d.fits = 0")


@pytest.fixture(scope="module")
def generated_deployment_keys():
    """Property names on a *generated* `Deployment`, for the anti-vacuity half.

    Without this, `test_no_real_deployment_carries_a_cost_model_field` would
    pass just as happily if `COST_MODEL_FIELDS` named properties nothing in
    the repo ever writes.

    A small scale deliberately: the question is which keys exist, not how
    many rows, and a 0.05 fleet answers it in a fraction of the time.
    """
    from etl import generate as gen
    fleet = gen.generate(seed=0, scale=0.05)
    deployments = fleet.nodes.get("Deployment", ())
    assert deployments, "the generated fleet has no Deployment nodes at all"
    return {key for node in deployments for key in node}


def test_no_real_deployment_carries_a_cost_model_field(
        real_only, generated_deployment_keys):  # noqa: F811 -- the imported fixture
    """The fact the rest of this module rests on, measured rather than asserted.

    MLPerf Tiny publishes what it measured -- throughput, accuracy, energy --
    and a submission round and division. It does not publish whether a model
    fits a board, which is our cost model's judgement about our generated
    fleet.
    """
    _client, fleet = real_only
    deployments = fleet.nodes.get("Deployment", ())
    assert deployments, "the real layer has no Deployment nodes; the fixture is wrong"

    real_keys = {key for node in deployments for key in node}
    assert not (real_keys & COST_MODEL_FIELDS), (
        f"real-layer Deployment nodes now carry "
        f"{sorted(real_keys & COST_MODEL_FIELDS)}. If the cost model has been "
        f"extended to MLPerf rows, EA10 and EA12 can answer from the real "
        f"layer and the README section on this must be re-measured.")

    assert COST_MODEL_FIELDS <= generated_deployment_keys, (
        f"{sorted(COST_MODEL_FIELDS - generated_deployment_keys)} is not "
        f"written by etl/generate.py either, so the check above proves "
        f"nothing -- it is comparing against property names no layer uses.")


def test_the_fits_filter_is_the_sole_reason_only_ea10_and_ea12_are_empty(
        real_only):  # noqa: F811 -- the imported fixture
    """Derived from the catalog, not claimed: which queries `fits` alone empties.

    An earlier version of this module said `EA10` and `EA12` were "the
    catalog's only entries that filter `WHERE d.fits = 1`". That is false --
    `EA03`, `EA06` and `EA07` filter on it too, and `EA04` reads it inside a
    `CASE`. Six queries mention it. In a change whose whole case is "measured,
    not assumed", an unchecked "only" is the wrong kind of sentence, so the
    set is measured here instead of written down.

    The measurement: take every query that mentions `fits`, remove the
    predicate, and see which then return rows. Those are the queries `fits`
    alone empties. The other four stay empty because their opening patterns
    bind nothing on the real layer -- no `ClinicalTask`, no `ModelVariant`,
    no `Sensor` -- which is the absence the README already documents and has
    nothing to do with the cost model's properties.
    """
    client, _fleet = real_only

    mentions_fits = {qid for qid, spec in BY_ID.items()
                     if "fits" in spec["cypher"]}
    assert len(mentions_fits) > 2, (
        f"only {sorted(mentions_fits)} mention `fits`, so this test is no "
        f"longer distinguishing between queries and the 'only' it replaced "
        f"would have been true after all")

    emptied_by_fits = set()
    for qid in mentions_fits:
        cypher = BY_ID[qid]["cypher"]
        stripped = cypher
        for predicate in FITS_PREDICATES:
            stripped = stripped.replace(predicate, "")
        assert stripped != cypher, (
            f"{qid} mentions `fits` in a form FITS_PREDICATES does not know "
            f"how to remove, so it would be silently classified as 'not "
            f"emptied by fits'. Add the form.")
        assert not client.query(cypher, GRAPH).records, (
            f"{qid} returns rows from the real layer while filtering on "
            f"`fits`, which no real Deployment carries")
        if client.query(stripped, GRAPH).records:
            emptied_by_fits.add(qid)

    assert emptied_by_fits == {"EA10", "EA12"}, (
        f"`fits` alone empties {sorted(emptied_by_fits)} on the real layer, "
        f"not EA10 and EA12. The README, CLAUDE.md and #114 all name that "
        f"pair; whichever changed, they need the same edit.")


def test_ea10_and_ea12_are_empty_because_the_cost_model_is_generated(
        real_only):  # noqa: F811 -- the imported fixture
    """Not just *that* they are empty -- which half of each query is missing.

    `tests/test_real_layer_shape.py` already pins both as empty. What it
    cannot say is why, and "why" is the whole of #114: an empty result from a
    filter that cannot match is a fact about our ETL, while an empty result
    from a join that should bind would be a fact about the engine.

    So each query is run again with the `fits` filter removed:

    - `EA12` then returns rows, so its `ON_BOARD -> HAS_SOC -> MADE_BY` join
      traverses the real layer perfectly well and only the filter empties it;
    - `EA10` returns a single row whose grouping key and every *aggregate over
      a property* are `NULL`. Only `count(d)` is not, because counting rows
      needs no property -- which is the shape that says the deployments are
      there and their cost-model columns are not.

    Neither depends on which engine runs it **on a graph loaded from
    nothing**, and that is measured rather than argued: the same real layer
    loaded into a freshly started 1.7.0 HTTP server returns the same nothing.
    It does depend on the graph's history -- loaded over a previous full
    fleet, the server returns 5 rows for each, which is engine note 8 and is
    what #114 recorded. This module's docstring has both.
    """
    client, _fleet = real_only

    for qid in ("EA10", "EA12"):
        assert not client.query(BY_ID[qid]["cypher"], GRAPH).records, (
            f"{qid} returns rows from the real layer. Its filter is "
            f"`d.fits = 1` and no real Deployment carries `fits`, so either "
            f"the ETL changed or the engine now matches a comparison against "
            f"a missing property.")

    unfiltered = BY_ID["EA12"]["cypher"].replace("WHERE d.fits = 1\n", "")
    assert unfiltered != BY_ID["EA12"]["cypher"], "EA12 no longer carries the filter"
    assert client.query(unfiltered, GRAPH).records, (
        "EA12 without its `fits` filter still returns nothing, so the filter "
        "is not what empties it and the conclusion above does not follow. "
        "Re-measure before trusting the README section this pins.")

    unfiltered = BY_ID["EA10"]["cypher"].replace("WHERE d.fits = 1\n", "")
    assert unfiltered != BY_ID["EA10"]["cypher"], "EA10 no longer carries the filter"
    rows = client.query(unfiltered, GRAPH).records
    assert len(rows) == 1, (
        f"EA10 without its filter groups the real layer's deployments by "
        f"`accelerator_kind`, which they do not have, so one row is the "
        f"expected shape. Got {len(rows)}: {rows[:3]}")
    kind, deployments, *aggregates = rows[0]
    assert kind is None, (
        f"EA10 now groups the real layer into accelerator kinds ({kind!r}), "
        f"so it has gained the cost model's fields")
    assert deployments, "the row counts no deployments, so the fixture is empty"
    assert all(value is None for value in aggregates), (
        f"every aggregate EA10 takes over a cost-model property must be NULL "
        f"on the real layer; got {aggregates} beside a count of {deployments}")


def test_ea08_on_the_real_layer_matches_a_ground_truth_computed_in_python(
        real_only):  # noqa: F811 -- the imported fixture
    """The third query #114 names, and the embedded build gets it right.

    #114 says `EA08` "returns fewer rows" embedded than over HTTP. Fewer than
    the server is only a defect if the server is right, so the question is
    settled against the source data instead: the same counts recomputed from
    `fleet.edges` in Python, which is the convention `tests/test_correctness.py`
    uses for exactly this reason.

    Counts are not pinned here -- they move when ONNX Runtime publishes. Both
    sides are recomputed from the same build, so this stays true across an
    upstream refresh and fails only if the engine and plain Python disagree.

    Compared as a multiset, plus a separate check that the engine's counts do
    not increase. The three counts are distinct today (296 / 237 / 205), but
    `EA08` carries one `ORDER BY` key and note 2 says a tie can come back in
    any order -- so an upstream refresh that made two of them equal would
    fail an order-sensitive comparison for a reason that has nothing to do
    with correctness. This test is supposed to survive that.
    """
    client, fleet = real_only

    accelerators = {n["id"]: n for n in fleet.nodes.get("Accelerator", ())}
    runtimes = {n["id"]: n for n in fleet.nodes.get("Runtime", ())}
    runs_on = collections.defaultdict(list)
    provided_by = {}
    implements = collections.defaultdict(set)
    for _sl, src, rel, _tl, tgt, _p in fleet.edges:
        if rel == "RUNS_ON":
            runs_on[src].append(tgt)
        elif rel == "PROVIDED_BY":
            # One runtime per kernel, asserted rather than assumed: a dict
            # would keep the last edge silently, and this side of the
            # comparison would then undercount while the engine's side --
            # which joins, and so sees both -- did not.
            assert src not in provided_by, (
                f"kernel {src} has more than one PROVIDED_BY edge; this "
                f"recount keeps one runtime per kernel and would disagree "
                f"with the engine for that reason rather than a real one")
            provided_by[src] = tgt
        elif rel == "IMPLEMENTS":
            implements[src].add(tgt)

    operators = collections.defaultdict(set)
    for kernel in fleet.nodes.get("Kernel", ()):
        runtime = provided_by.get(kernel["id"])
        if runtime is None:
            continue
        for accelerator in runs_on.get(kernel["id"], ()):
            key = (accelerators[accelerator]["kind"], runtimes[runtime]["name"])
            operators[key] |= implements.get(kernel["id"], set())

    expected = sorted(((kind, runtime, len(ops))
                       for (kind, runtime), ops in operators.items()),
                      key=lambda row: -row[2])
    assert expected, "the real layer has no accelerator/runtime pair with kernels"
    # `EA08` ends `ORDER BY operators DESC LIMIT 20`. While there are fewer
    # than 20 pairs the limit never bites and both sides are the whole answer.
    # Once there are 20 or more, a tie at the cutoff lets the engine and this
    # recount keep *different* rows -- both correct, and the comparison below
    # would fail for that rather than for a disagreement. So the limit is
    # asserted away rather than reproduced: upstream growing the real layer
    # past 20 accelerator/runtime pairs fails here with this message instead
    # of an unreadable row diff.
    assert len(expected) < 20, (
        f"the real layer now has {len(expected)} accelerator/runtime pairs, so "
        f"EA08's LIMIT 20 bites and a tie at the cutoff would make this "
        f"comparison unreliable. Compare the top rows by count instead of "
        f"comparing the whole answer.")

    got = [tuple(row) for row in client.query(BY_ID["EA08"]["cypher"], GRAPH).records]
    assert sorted(got) == sorted(tuple(row) for row in expected), (
        f"EA08 disagrees with the same question computed in Python:\n"
        f"  engine: {sorted(got)}\n"
        f"  python: {sorted(expected)}\n"
        f"#114 recorded the embedded build returning fewer rows than the "
        f"server here. The 1.7.0 server has since been measured returning "
        f"these same rows, so while this passes both builds agree with a "
        f"recount and there is nothing left to reconcile.")

    counts = [row[2] for row in got]
    assert counts == sorted(counts, reverse=True), (
        f"EA08 asks for `ORDER BY operators DESC` and returned {counts}. "
        f"The rows are compared above without order because note 2 lets ties "
        f"come back either way round; this is the ordering claim itself, "
        f"which ties do not excuse.")


def test_a_comparison_against_a_missing_property_matches_nothing(
        real_only):  # noqa: F811 -- the imported fixture
    """Embedded 1.7.1's semantics, and the probe that closed #114.

    Three-valued logic: a property that is not there is `NULL`, `NULL = 1` is
    `NULL`, and a `WHERE` keeps only rows that are `true`. So both `= 1` and
    `<> 1` must match nothing, which is the pair worth checking -- an engine
    treating the comparison as `false` rather than `NULL` would still pass a
    test that only looked at `= 1`.

    Run against a probe node rather than a `Deployment`, so it says what the
    engine does rather than what this graph happens to hold, and so the same
    lines can be pasted at a server. They were, using `n.v` on a probe node of
    its own: the 1.7.0 server agrees on `= 1` -- which is why a freshly
    started server answers `EA10` and `EA12` exactly as embedded does -- and
    **disagrees on `<> 1`**, matching the row whose property is absent. That
    is engine note 8b, which `docs/engine-notes.md`'s banner had listed as
    gone; it is gone on embedded and live on the server.

    So the `<> 1` assertion below is not symmetry for its own sake. It is the
    one that would notice the embedded build regressing to what the server
    still does.
    """
    client, _fleet = real_only
    # This writes into the module-scoped graph the other tests read. The
    # `finally` below removes the node, but by engine note 8 its property
    # *columns* outlive the delete on some builds -- which is this module's
    # own subject matter. `CostModelProbe` and `fits` are chosen so that even
    # a resurrected column lands on a label nothing else here matches; if a
    # later test in this file starts creating nodes, give it a label of its
    # own rather than assuming this ran last.
    client.query('CREATE (:CostModelProbe {id: "probe:1"})', GRAPH)
    try:
        def ids(clause):
            return [row[0] for row in client.query(
                f"MATCH (n:CostModelProbe) WHERE {clause} RETURN n.id",
                GRAPH).records]

        assert ids("n.fits IS NULL") == ["probe:1"], (
            "the probe node did not load, so the two checks below would pass "
            "against an empty graph")
        assert ids("n.fits = 1") == [], (
            "this build matches `= 1` against a property that is not there. "
            "EA10 and EA12 would then return rows from the real layer, and "
            "#114's server-side reading is this build's behaviour too.")
        assert ids("n.fits <> 1") == [], (
            "this build matches `<> 1` against a missing property, treating "
            "the comparison as false rather than NULL. That is engine note "
            "8b, which the 1.7.0 server still shows and embedded 1.7.1 does "
            "not -- so if this fires, the embedded build has regressed to the "
            "server's behaviour and every `<>` in the catalog needs its "
            "`IS NOT NULL` guard re-checked.")
    finally:
        client.query("MATCH (n:CostModelProbe) DETACH DELETE n", GRAPH)
