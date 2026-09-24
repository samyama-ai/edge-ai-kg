"""Zero rows is an answer, so the catalog is driven to it deliberately.

Every catalog query returns rows against the shipped graph. That is what a demo
needs and not what a test needs: a query only ever run against data that
satisfies it cannot tell "the answer is none" from "the query is broken" (#27).

That claim is checked rather than assumed --
`tests/test_correctness.py::test_every_catalog_query_runs_and_returns_rows`
sweeps the catalog -- though it is checked for 17 of the 19. Two ids are
excused there, both named in that module's `EMPTY_IS_A_VALID_ANSWER` with the
test that pins each zero: `EA04`, whose combination may legitimately not occur
at the fixture's scale, and `EA18`, which is empty because no deployment in
this fleet misses a latency budget. `EA01` and `EA02` used to be excused too,
under a non-strict `xfail` for engine note 10; #105 raised the engine floor to
1.7.1 and removed them (reaching `main` with #104), and the note stopped
reproducing.

The alerting queries keep their zero-and-one-row pairs beside their other
fixtures rather than here: `EA18` in `tests/test_latency_budget.py` (over budget
and within it), `EA19` in `tests/test_certification_alerts.py` (a governed task
and an ungoverned one). This module covers the catalog queries that have no
module of their own.

It matters here more than most repos, because the catalog's central question is
a **negative** -- operators with *no* kernel -- so an empty result is the
meaningful, correct answer for a fully covered accelerator.

Each query below is run twice on purpose-built fixtures: once where the correct
answer is zero rows, and once where it is exactly one known row. The second half
is what makes the first meaningful. Asserting only the empty case would pass
just as well against a query that can never return anything.

The fixtures are the whole graph. `EA11` and `EA13` are fleet-wide and carry no
id filter, so an embedded engine holding nothing else gives an exactly known
answer. Each fixture resets before loading, as every other module here does --
measured on `samyama` 0.6.1 two `SamyamaClient.embedded()` instances in one
process are independent, so nothing another module loads reaches here, but that
is a property of this build rather than a guarantee and these assertions are
exact counts that would fail confusingly if it changed.

## What a 6-node fixture cannot show

`EA11`'s anti-join is `OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op),
(k)-[:RUNS_ON]->(a:Accelerator)` -- the comma-separated single-`MATCH` shape that
`docs/engine-notes.md` records as correct on a toy graph and **wrong** at scale,
with the explicit warning that "a passing 6-node reproduction proves nothing".

So these two tests establish that `EA11` distinguishes zero from one, and
nothing about whether it is right on the real graph. That is asserted
separately, against ground truth computed from the `Fleet`, in
`tests/test_correctness.py::test_ea11_cpu_only_models_match_ground_truth`.
Checked once by hand at full `--scale 1.0` (24,115 nodes) as the engine note
asks: 60 of 60 models are CPU-only somewhere, and `EA11`'s top ten counts
`[12, 11, 10, 10, 10, 9, 9, 9, 9, 9]` match Python exactly.

## Why not EA01

The issue asks for the fallback audit itself, and that is the right instinct --
`EA01` *is* the hero question. It could not be run here at first: on the
embedded engine it raised `Query error: Variable not found: operator` before
returning anything, which was engine note 10 / #56, not a property of any
fixture.

`EA11` was added as the closest available stand-in -- the same anti-join, asked
fleet-wide instead of against one accelerator -- and `test_ea01_zero_row_case`
below was marked `xfail(strict=True)` so that the day #56 resolved, the suite
would fail and say this coverage could now be added. #56 resolved by pinning the
embedded engine to `samyama>=1.7.1`, the mark fired exactly that way, and
`EA01` is asserted directly now. `EA11` stays: it covers the fleet-wide shape,
which is a different question, not a substitute for this one.

**`EA01` is paired like everything else here, and was not at first.** While the
mark was on, the zero-row test used ids `EA01` does not filter on -- the query
filters `m.id = "model:00000"` and `a.id = "accel:00001"` -- so its opening
`MATCH` bound nothing and `[]` came back regardless of the graph. An xfailed
test hides that; an unmarked one does not, but only if something can tell empty
from vacuous. That something is the control, and `EA01` was the one query in
this module without one (Tarun's review, made on #94 before that PR was
closed and rebuilt as #109).
"""
import pytest

from benchmarks.queries import BY_ID
from etl.helpers import create_edges, create_nodes

GRAPH = "default"


@pytest.fixture
def engine():
    """A fresh embedded client, skipping in **setup** if there is not one.

    A fixture rather than a helper called from the test body. The skip is the
    reason: `conftest.py` turns setup-phase skips into failures under
    `--no-skips`, and only setup-phase ones -- so while this was a plain
    function, a CI job with no engine skipped all three of these regressions
    and the suite reported green. That is the failure `--no-skips` exists to
    catch, in the module whose subject is answers that are empty for the wrong
    reason.

    Function-scoped, so each test gets its own client and its own graph.
    """
    try:
        from samyama import SamyamaClient
        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")
    # Every fixture here asserts an exact count, so a shared graph would not
    # merely add rows -- ids repeat across fixtures, and `create_edges` resolves
    # endpoints with `WHERE v.id = ...`, so each variable would bind twice and
    # `CREATE` would emit the cartesian product. Reset regardless.
    client.query("MATCH (n) DETACH DELETE n", GRAPH)
    # Verified, not assumed. The reset used to swallow its own failure, so a
    # build that would not empty a graph produced a count assertion failing for
    # a reason the message did not mention -- and these tests are all exact
    # counts.
    held = client.query("MATCH (n) RETURN count(n)", GRAPH).records
    assert (held[0][0] if held else 0) == 0, (
        f"reset left {held} in `{GRAPH}`; every assertion in this module is an "
        f"exact count, so they would fail for a reason none of them names"
    )
    return client


def run(client, query_id: str):
    return client.query(BY_ID[query_id]["cypher"].strip(), GRAPH).records


def fleet_with_coverage(client, *, hardswish_is_cpu_only: bool):
    """One model using two operators, both implemented somewhere.

    With `hardswish_is_cpu_only`, the only kernel for `HardSwish` runs on the
    `MCU-CPU`, which is the case EA11 exists to find.
    """
    create_nodes(client, GRAPH, "Model",
                 [{"id": "m1", "name": "covered-model", "family": "cnn"}])
    # Note 8: `props_map` drops absent keys, and the columnar store survives
    # `DETACH DELETE` -- a key omitted here can report a *previous* load's value
    # rather than null. Every property EA11 and EA13 read is set explicitly.
    create_nodes(client, GRAPH, "Operator",
                 [{"id": "op1", "name": "Conv", "category": "convolution",
                   "domain": "ai.onnx"},
                  {"id": "op2", "name": "HardSwish", "category": "activation",
                   "domain": "ai.onnx"}])
    # `<>` matches a null property on this engine, so `kind` is always set.
    # `is_cpu_fallback` is what EA11 filters on since #69 -- it used to infer
    # "is a CPU" from `kind <> "MCU-CPU"`, which read ONNX Runtime's CPU
    # execution provider as an accelerator. A fixture omitting it leaves the
    # property null, and note 8 means a null here is not merely "not 0".
    create_nodes(client, GRAPH, "Accelerator",
                 [{"id": "npu", "kind": "NPU-Lite", "is_cpu_fallback": 0},
                  {"id": "cpu", "kind": "MCU-CPU", "is_cpu_fallback": 1}])
    create_nodes(client, GRAPH, "Kernel",
                 [{"id": "k1", "execution_provider": "CPUExecutionProvider"},
                  {"id": "k2", "execution_provider": "CPUExecutionProvider"}])
    create_edges(client, GRAPH, [
        ("Model", "m1", "USES_OPERATOR", "Operator", "op1", None),
        ("Model", "m1", "USES_OPERATOR", "Operator", "op2", None),
        ("Kernel", "k1", "IMPLEMENTS", "Operator", "op1", None),
        ("Kernel", "k1", "RUNS_ON", "Accelerator", "npu", None),
        ("Kernel", "k2", "IMPLEMENTS", "Operator", "op2", None),
        ("Kernel", "k2", "RUNS_ON", "Accelerator",
         "cpu" if hardswish_is_cpu_only else "npu", None),
    ])
    return client


def onnx_runtime_with(client, *, cuda_implements_it: bool):
    """One `ai.onnx` operator with a CPU kernel, and optionally a CUDA one."""
    create_nodes(client, GRAPH, "Operator",
                 [{"id": "op1", "name": "Conv", "category": "convolution",
                   "domain": "ai.onnx"}])
    kernels = [{"id": "kcpu", "execution_provider": "CPUExecutionProvider"}]
    edges = [("Kernel", "kcpu", "IMPLEMENTS", "Operator", "op1", None)]
    if cuda_implements_it:
        kernels.append({"id": "kcuda", "execution_provider": "CUDAExecutionProvider"})
        edges.append(("Kernel", "kcuda", "IMPLEMENTS", "Operator", "op1", None))
    create_nodes(client, GRAPH, "Kernel", kernels)
    create_edges(client, GRAPH, edges)
    return client


def test_ea11_is_empty_when_every_operator_is_accelerated(engine):
    """The answer is none, and none is correct."""
    rows = run(fleet_with_coverage(engine, hardswish_is_cpu_only=False), "EA11")
    assert not rows, (
        f"every operator has a non-MCU-CPU kernel, so no model is CPU-only; "
        f"EA11 returned {rows}"
    )


def test_ea11_finds_the_model_when_an_operator_is_cpu_only(engine):
    """The control. Without this, the assertion above proves nothing."""
    rows = run(fleet_with_coverage(engine, hardswish_is_cpu_only=True), "EA11")
    assert len(rows) == 1, f"expected exactly the one CPU-only model, got {rows}"
    model, _family, count, operators = rows[0]
    assert (model, count, list(operators)) == ("covered-model", 1, ["HardSwish"]), (
        f"expected covered-model to be CPU-only on HardSwish alone, got {rows[0]}"
    )


def test_ea13_is_empty_when_cuda_implements_everything(engine):
    rows = run(onnx_runtime_with(engine, cuda_implements_it=True), "EA13")
    assert not rows, (
        f"CUDA implements the only ai.onnx operator, so nothing is CPU-only; "
        f"EA13 returned {rows}"
    )


def test_ea13_finds_the_operator_when_cuda_does_not(engine):
    rows = run(onnx_runtime_with(engine, cuda_implements_it=False), "EA13")
    assert len(rows) == 1 and rows[0][0] == "Conv", (
        f"expected Conv to be reported as CPU-only, got {rows}"
    )


def ea01_fixture(client, *, covered_by_the_audited_accelerator: bool):
    """The audited model and accelerator, with one operator either covered or not.

    **The ids must be `model:00000` and `accel:00001`.** `EA01` hardcodes them
    (`benchmarks/queries.py:28,30`) because the hero question is asked about one
    named model on one named board. A fixture using any other id leaves the
    opening `MATCH` binding nothing, and the query returns `[]` for every graph
    -- which is what an earlier version of this test did, passing vacuously.

    The `engine` fixture resets the graph first, so there is no generated
    fleet to collide with; reusing the generator's id shape is safe and is the
    only way the query sees the fixture at all.

    Takes the client rather than making one: the skip for a missing engine has
    to happen in setup, which is what `--no-skips` converts (#106).
    """
    create_nodes(client, GRAPH, "Model",
                 [{"id": "model:00000", "name": "audited-model", "family": "cnn"}])
    # Note 8: a key omitted here can report a *previous* load's value rather
    # than null, so every property EA01 projects is set explicitly.
    create_nodes(client, GRAPH, "Operator",
                 [{"id": "op1", "name": "Conv", "category": "convolution",
                   "since_version": 11}])
    create_nodes(client, GRAPH, "Accelerator",
                 [{"id": "accel:00001", "kind": "NPU-Lite", "is_cpu_fallback": 0},
                  {"id": "accel:00002", "kind": "NPU-Pro", "is_cpu_fallback": 0}])
    create_nodes(client, GRAPH, "Kernel",
                 [{"id": "k1", "execution_provider": "CPUExecutionProvider"}])
    runs_on = "accel:00001" if covered_by_the_audited_accelerator else "accel:00002"
    create_edges(client, GRAPH, [
        ("Model", "model:00000", "USES_OPERATOR", "Operator", "op1", None),
        ("Kernel", "k1", "IMPLEMENTS", "Operator", "op1", None),
        ("Kernel", "k1", "RUNS_ON", "Accelerator", runs_on, None),
    ])
    return client


def test_ea01_zero_row_case(engine):
    """An accelerator implementing everything must produce an empty audit.

    This is the hero question's own zero-row case, which #27 asks for and which
    could not be asserted while engine note 10 made `EA01` raise on the embedded
    build. #56 pinned the engine to `samyama>=1.7.1` and the query runs.
    """
    rows = run(ea01_fixture(engine, covered_by_the_audited_accelerator=True), "EA01")
    assert rows == [], (
        f"the audited accelerator has a kernel for the model's only operator, "
        f"so the fallback audit must be empty; EA01 returned {rows}"
    )


def test_ea01_finds_the_operator_when_the_audited_accelerator_lacks_a_kernel(engine):
    """The control. Without it the assertion above proves nothing.

    An earlier version of the zero-row test used ids the query does not filter
    on, so its `MATCH` bound nothing and `[]` came back whatever the graph held
    -- it would have passed with the `RUNS_ON` edge deleted. Every empty-answer
    assertion in this module is paired for exactly that reason; `EA01` was the
    one that had no pair, which is how the vacuity survived review.
    """
    rows = run(ea01_fixture(engine, covered_by_the_audited_accelerator=False), "EA01")
    assert len(rows) == 1, (
        f"the only kernel runs on accel:00002, not the audited accel:00001, so "
        f"Conv must be reported as falling back; EA01 returned {rows}"
    )
    assert tuple(rows[0]) == ("Conv", "convolution", 11), rows[0]
