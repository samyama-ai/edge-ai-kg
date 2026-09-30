"""Why `EA10` and `EA12` are empty on the real layer, and why no engine changes it.

#114 recorded three queries answering differently on the real layer between
the embedded build and the 1.7.0 server: `EA08` returning fewer rows embedded,
`EA10` and `EA12` returning none. Two of those three are not a build
disagreement at all, and this module is the measurement that says so.

`EA10` and `EA12` are the catalog's only entries that filter
`WHERE d.fits = 1`. `fits` is written by the cost model in `etl/generate.py`
and by nothing else -- `git log -S"fits" -- etl/real_layer.py` is empty across
every revision -- so the real layer's `Deployment` nodes, which are MLPerf Tiny
submission rows, do not carry it. The filter cannot match, on any engine, and
`EA10` also selects four more properties the real layer has never had.

So they are **generated-layer queries**: the same kind of absence the README
already documents for the six clinical-spine labels, not a divergence.

**The server half is measured too, and there is no divergence.** On
2026-09-30 the real layer was loaded into `ghcr.io/samyama-ai/samyama-graph:1`
(the 1.7.0 image the notes describe) and every catalog query run against it.
It returns exactly what embedded 1.7.1 returns -- the same six queries with
rows, the same row counts, and `EA08` row-for-row identical. So #114's
premise was a measurement error: nothing on the real layer disagrees between
the builds.

That measurement cannot live in this suite, because it needs a container. It
lives in `docs/engine-notes.md` with its reproduction, the way every other
server measurement here does. What this module pins is the half that runs
everywhere:

- no real `Deployment` carries a cost-model field, and a generated one
  carries all four (the anti-vacuity half);
- `fits` alone is what empties `EA10` and `EA12`, derived by stripping the
  predicate from every query that mentions it rather than claimed in prose;
- `EA08`'s answer matches a ground truth computed in Python from the same
  source files;
- embedded 1.7.1's semantics for a comparison against a missing property.

That last one is not decoration. The server **agrees** on `= 1` and
**disagrees** on `<> 1`, which is engine note 8b still being live there after
the embedded build fixed it -- found by running this probe against both. The
`= 1` half is why #114 closed as a measurement error rather than as an engine
defect.
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

    Neither depends on which engine runs it, and that is now measured rather
    than argued: the same real layer loaded into the 1.7.0 HTTP server returns
    the same nothing (see this module's docstring).
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
                      key=lambda row: -row[2])[:20]
    assert expected, "the real layer has no accelerator/runtime pair with kernels"

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
    lines can be pasted at a server. They were: the 1.7.0 server agrees on
    `= 1` -- which is what settled #114, since `EA10` and `EA12` could not
    have returned rows there either -- and **disagrees on `<> 1`**, matching
    the missing row. That is engine note 8b, which this file's banner had
    listed as gone; it is gone on embedded and live on the server.

    So the `<> 1` assertion below is not symmetry for its own sake. It is the
    one that would notice the embedded build regressing to what the server
    still does.
    """
    client, _fleet = real_only
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
            "this build matches `<> 1` against a missing property, so it "
            "treats the comparison as false rather than NULL. That is a "
            "finding worth an engine note in its own right.")
    finally:
        client.query("MATCH (n:CostModelProbe) DETACH DELETE n", GRAPH)
