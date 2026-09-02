"""Zero rows is an answer, so the catalog is driven to it deliberately.

All sixteen catalog queries return rows against the shipped graph. That is what
a demo needs and not what a test needs: a query only ever run against data that
satisfies it cannot tell "the answer is none" from "the query is broken" (#27).

It matters here more than most repos, because the catalog's central question is
a **negative** -- operators with *no* kernel -- so an empty result is the
meaningful, correct answer for a fully covered accelerator.

Each query below is run twice on purpose-built fixtures: once where the correct
answer is zero rows, and once where it is exactly one known row. The second half
is what makes the first meaningful. Asserting only the empty case would pass
just as well against a query that can never return anything.

The fixtures are the whole graph. `EA11` and `EA13` are fleet-wide and carry no
id filter, so a fresh embedded engine holding nothing else gives an exactly
known answer.

## Why not EA01

The issue asks for the fallback audit itself, and that is the right instinct --
`EA01` *is* the hero question. It cannot be run here: on the embedded engine it
raises `Query error: Variable not found: operator` before returning anything,
which is engine note 10 / #56, not a property of any fixture.

`EA11` is the closest available stand-in -- the same anti-join, asked
fleet-wide instead of against one accelerator -- and `test_ea01_zero_row_case`
below is marked `xfail(strict=True)` so that the day #56 is resolved, the suite
fails and says this coverage can now be added.
"""
import pytest

from benchmarks.queries import BY_ID
from etl.helpers import create_edges, create_nodes

GRAPH = "default"


def engine():
    try:
        from samyama import SamyamaClient
        return SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")


def run(client, query_id: str):
    return client.query(BY_ID[query_id]["cypher"].strip(), GRAPH).records


def fleet_with_coverage(*, hardswish_is_cpu_only: bool):
    """One model using two operators, both implemented somewhere.

    With `hardswish_is_cpu_only`, the only kernel for `HardSwish` runs on the
    `MCU-CPU`, which is the case EA11 exists to find.
    """
    client = engine()
    create_nodes(client, GRAPH, "Model",
                 [{"id": "m1", "name": "covered-model", "family": "cnn"}])
    create_nodes(client, GRAPH, "Operator",
                 [{"id": "op1", "name": "Conv", "category": "convolution"},
                  {"id": "op2", "name": "HardSwish", "category": "activation"}])
    # `<>` matches a null property on this engine, so `kind` is always set.
    create_nodes(client, GRAPH, "Accelerator",
                 [{"id": "npu", "kind": "NPU-Lite"}, {"id": "cpu", "kind": "MCU-CPU"}])
    create_nodes(client, GRAPH, "Kernel", [{"id": "k1"}, {"id": "k2"}])
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


def onnx_runtime_with(*, cuda_implements_it: bool):
    """One `ai.onnx` operator with a CPU kernel, and optionally a CUDA one."""
    client = engine()
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


def test_ea11_is_empty_when_every_operator_is_accelerated():
    """The answer is none, and none is correct."""
    rows = run(fleet_with_coverage(hardswish_is_cpu_only=False), "EA11")
    assert rows == [], (
        f"every operator has a non-MCU-CPU kernel, so no model is CPU-only; "
        f"EA11 returned {rows}"
    )


def test_ea11_finds_the_model_when_an_operator_is_cpu_only():
    """The control. Without this, the assertion above proves nothing."""
    rows = run(fleet_with_coverage(hardswish_is_cpu_only=True), "EA11")
    assert len(rows) == 1, f"expected exactly the one CPU-only model, got {rows}"
    model, _family, count, operators = rows[0]
    assert (model, count, operators) == ("covered-model", 1, ["HardSwish"]), (
        f"expected covered-model to be CPU-only on HardSwish alone, got {rows[0]}"
    )


def test_ea13_is_empty_when_cuda_implements_everything():
    rows = run(onnx_runtime_with(cuda_implements_it=True), "EA13")
    assert rows == [], (
        f"CUDA implements the only ai.onnx operator, so nothing is CPU-only; "
        f"EA13 returned {rows}"
    )


def test_ea13_finds_the_operator_when_cuda_does_not():
    rows = run(onnx_runtime_with(cuda_implements_it=False), "EA13")
    assert len(rows) == 1 and rows[0][0] == "Conv", (
        f"expected Conv to be reported as CPU-only, got {rows}"
    )


@pytest.mark.xfail(
    reason="#56 / engine note 10: EA01 projects through a second WITH, which the "
           "embedded build does not register, so it raises `Variable not found: "
           "operator` before any fixture matters. Strict: when #56 is resolved "
           "this passes, the suite fails, and the fallback audit's zero-row case "
           "-- the one the issue actually asks for -- can be asserted here.",
    strict=True,
)
def test_ea01_zero_row_case():
    """An accelerator implementing everything must produce an empty audit."""
    client = engine()
    create_nodes(client, GRAPH, "Model", [{"id": "model:00000", "name": "m"}])
    create_nodes(client, GRAPH, "Operator",
                 [{"id": "op1", "name": "Conv", "category": "convolution",
                   "since_version": 1}])
    create_nodes(client, GRAPH, "Accelerator", [{"id": "accel:00001", "kind": "NPU-Lite"}])
    create_nodes(client, GRAPH, "Kernel", [{"id": "k1"}])
    create_edges(client, GRAPH, [
        ("Model", "model:00000", "USES_OPERATOR", "Operator", "op1", None),
        ("Kernel", "k1", "IMPLEMENTS", "Operator", "op1", None),
        ("Kernel", "k1", "RUNS_ON", "Accelerator", "accel:00001", None),
    ])
    assert run(client, "EA01") == []
