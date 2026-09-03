"""`opset_ceiling` is a hard cut on kernel existence, so the graph is held to it.

`Accelerator.opset_ceiling` sat on the node the hero question turns on, with
nothing saying what it meant operationally (#25). `etl/generate.py` answers it:
a `Kernel` is created only when, among other conditions,

    if o.since_version > a["opset_ceiling"]:
        continue

so the rule is **no kernel exists on an accelerator for an operator whose
`since_version` is above that accelerator's ceiling** -- and it is a rule, not a
hint. `docs/schema.md` now states it; this file holds the graph to it.

Two things worth knowing before reading the assertions.

**`99` is a sentinel, not a ceiling.** `MCU-CPU` carries 99 and ONNX's highest
`since_version` in this catalog is 28, so it excludes nothing. That is what
makes the MCU-CPU the universal fallback, and why the hero question always has
an answer: something can always run, just slowly. The boundary assertion below
skips it, because there is no boundary to reach.

**The real layer has no ceiling at all.** All six real accelerators -- three
MLPerf Tiny NPUs and ONNX Runtime's CPU, CUDA and DirectML providers -- carry
none, because their kernel registrations are read from upstream rather than
derived from a rule. Their placements are outside the invariant: not violating
it, not covered by it. Asserted, so a query that filters on `opset_ceiling`
silently dropping them stays a documented fact rather than a surprise.

The comparison is done in Python rather than in the `WHERE` clause. That follows
`tests/test_correctness.py`'s approach of recomputing the answer here and
asserting the engine agrees, and it sidesteps a `Type error: AND requires
boolean operands` this shape raised on the embedded build when a null ceiling
was in play -- which I could not reduce to a minimal case, so it is avoided
rather than relied upon.
"""
from collections import defaultdict

import pytest

from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
SCALE = 0.3
SEED = 4242

# etl/generate.py gives MCU-CPU this so nothing is ever excluded.
NO_CEILING_SENTINEL = 99


@pytest.fixture(scope="module")
def loaded():
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    try:
        from samyama import SamyamaClient
        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

    fleet = gen.generate(seed=SEED, scale=SCALE, operators=ops)
    from etl import real_layer
    real_layer.build_real(fleet, ops)
    try:
        client.query("MATCH (n) DETACH DELETE n", GRAPH)
    except Exception:
        pass
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client


def placements(client):
    """(kernel id, operator since_version, accelerator ceiling) per placement.

    One linear pattern through `Kernel`: engine note 1 is that a variable
    re-bound in a trailing position of a second `MATCH` is silently not joined.
    """
    return client.query(
        "MATCH (op:Operator)<-[:IMPLEMENTS]-(k:Kernel)-[:RUNS_ON]->(a:Accelerator) "
        "RETURN k.id, op.since_version, a.opset_ceiling", GRAPH).records


def accelerators(client):
    return client.query(
        "MATCH (a:Accelerator) RETURN a.id, a.opset_ceiling, a.provenance, a.kind",
        GRAPH).records


def test_no_kernel_exceeds_its_accelerators_ceiling(loaded):
    """The rule `docs/schema.md` now states."""
    over = [(kid, since, ceiling) for kid, since, ceiling in placements(loaded)
            if ceiling is not None and since > ceiling]
    assert not over, (
        f"{len(over)} kernels implement an operator above their accelerator's "
        f"opset_ceiling, e.g. {over[:5]}. The generator excludes these, so the "
        f"graph disagreeing with it means one of the two has changed."
    )


def test_the_rule_actually_excludes_something(loaded):
    """Guard: an invariant nothing could break is not evidence of anything.

    For every real ceiling there must be operators in the graph above it --
    otherwise the assertion above would hold on a catalog that simply never
    reaches the bound, and would keep holding if the rule were deleted.
    """
    ceilings = {c for _, c, _, _ in accelerators(loaded)
                if c is not None and c != NO_CEILING_SENTINEL}
    assert ceilings, "no accelerator carries a real ceiling"

    versions = [r[0] for r in loaded.query(
        "MATCH (op:Operator) RETURN op.since_version", GRAPH).records
        if r[0] is not None]
    for ceiling in sorted(ceilings):
        above = sum(1 for v in versions if v > ceiling)
        assert above > 0, (
            f"no operator in the graph has since_version above {ceiling}, so a "
            f"ceiling of {ceiling} excludes nothing and the invariant is vacuous"
        )


def test_the_ceiling_is_reached_not_merely_respected(loaded):
    """It is a boundary, not decoration: every band uses operators right up to it.

    If a band's highest `since_version` sat well below its ceiling, the ceiling
    would not be the thing shaping coverage and the README's fallback story
    would be resting on something else.
    """
    highest = defaultdict(int)
    for _, since, ceiling in placements(loaded):
        if ceiling is not None and ceiling != NO_CEILING_SENTINEL:
            highest[ceiling] = max(highest[ceiling], since)
    assert highest, "no placements on an accelerator with a real ceiling"

    short = {c: hi for c, hi in highest.items() if hi != c}
    assert not short, (
        f"bands whose highest operator opset does not reach the ceiling: {short}. "
        f"The ceiling is then not what limits coverage in those bands."
    )


def test_real_accelerators_carry_no_ceiling(loaded):
    """The documented gap, pinned.

    Real kernel registrations come from upstream rather than a rule, so the six
    real accelerators have no ceiling and their placements sit outside the
    invariant. A query filtering on `opset_ceiling` drops them silently, which
    `docs/schema.md` says and this keeps true.
    """
    by_provenance = defaultdict(lambda: [0, 0])
    for _, ceiling, provenance, _ in accelerators(loaded):
        by_provenance[provenance][0] += 1
        if ceiling is None:
            by_provenance[provenance][1] += 1

    synthetic_total, synthetic_without = by_provenance["synthetic"]
    real_total, real_without = by_provenance["real"]
    assert synthetic_without == 0, (
        f"{synthetic_without} of {synthetic_total} generated accelerators have "
        f"no opset_ceiling; the generator gives every archetype one"
    )
    assert real_without == real_total and real_total > 0, (
        f"{real_without} of {real_total} real accelerators lack a ceiling; "
        f"docs/schema.md says all of them do"
    )
