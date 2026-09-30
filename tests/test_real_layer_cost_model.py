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

That leaves two things this module can still settle, and one it cannot:

- `EA08`'s embedded answer is checked against a ground truth computed in
  Python from the same source files. It matches, so the embedded build is not
  the one that is wrong there;
- embedded 1.7.1's own semantics for a comparison against a missing property
  are pinned, because that is the half of the open question that does not
  need a server;
- what the 1.7.0 **server** returned is still unmeasured. If it matched
  `d.fits = 1` against absent properties, that is an engine note with a
  three-line reproduction; if it did not, the README paragraph was a
  measurement error. Nothing here can decide it, and nothing here pretends to.
"""
from __future__ import annotations

import collections

import pytest

from benchmarks.queries import BY_ID
from tests.test_real_layer_shape import (
    GRAPH,
    real_only,  # noqa: F401 -- shared fixture
)

# The five properties `EA10` reads, all of them produced by the cost model in
# `etl/generate.py`. Named from the query rather than from the generator: if
# the generator renames one, the query breaks and this list is where the
# rename is noticed.
COST_MODEL_FIELDS = {"fits", "accelerator_kind", "fallback_fraction",
                     "latency_ms"}


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
    - `EA10` returns a single row whose grouping key and every aggregate are
      `NULL`, because it reads four more properties the real layer lacks.

    Neither depends on which engine runs it.
    """
    client, _fleet = real_only

    for qid in ("EA10", "EA12"):
        assert not client.query(BY_ID[qid]["cypher"], GRAPH).records, (
            f"{qid} returns rows from the real layer. Its filter is "
            f"`d.fits = 1` and no real Deployment carries `fits`, so either "
            f"the ETL changed or the engine now matches a comparison against "
            f"a missing property -- which is exactly the question #114 asks "
            f"of the 1.7.0 server.")

    unfiltered = BY_ID["EA12"]["cypher"].replace("WHERE d.fits = 1\n", "")
    assert unfiltered != BY_ID["EA12"]["cypher"], "EA12 no longer carries the filter"
    rows = client.query(unfiltered, GRAPH).records
    assert rows, (
        "EA12 without its `fits` filter still returns nothing, so the filter "
        "is not what empties it and the conclusion above does not follow. "
        "Re-measure before trusting the README section this pins.")

    unfiltered = BY_ID["EA10"]["cypher"].replace("WHERE d.fits = 1\n", "")
    assert unfiltered != BY_ID["EA10"]["cypher"], "EA10 no longer carries the filter"
    rows = client.query(unfiltered, GRAPH).records
    assert len(rows) == 1 and rows[0][0] is None, (
        f"EA10 without its filter groups the real layer's deployments by "
        f"`accelerator_kind`, which they do not have, so one all-NULL row is "
        f"the expected shape. Got {rows[:3]}. If it now groups into real "
        f"kinds, the real layer has gained the cost model's fields.")


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
    assert got == [tuple(row) for row in expected], (
        f"EA08 disagrees with the same question computed in Python:\n"
        f"  engine: {got}\n"
        f"  python: {expected}\n"
        f"#114 records the embedded build returning fewer rows than the "
        f"server here. While this passes, the embedded rows are the right "
        f"ones and 'fewer' is not the same as 'wrong'.")


def test_a_comparison_against_a_missing_property_matches_nothing(
        real_only):  # noqa: F811 -- the imported fixture
    """The half of #114's open question that needs no server.

    Three-valued logic: a property that is not there is `NULL`, `NULL = 1` is
    `NULL`, and a `WHERE` keeps only rows that are `true`. So both `= 1` and
    `<> 1` must match nothing, which is the pair worth checking -- an engine
    treating the comparison as `false` rather than `NULL` would still pass a
    test that only looked at `= 1`.

    Run against a probe node rather than a `Deployment`, so it says what the
    engine does rather than what this graph happens to hold, and so the same
    three lines can be pasted at a 1.7.0 server. If they return a row there,
    #114 is an engine note with a minimal reproduction; if they do not, the
    README's paragraph was a measurement error. That is the whole of what is
    left to measure.
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
