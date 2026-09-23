"""What `--layers real` actually contains, pinned against what the README claims.

The README offers `python -m etl.loader --layers real` as a first-class way to
use the repo, and #21 asked the obvious unmeasured question: is the public-source
subgraph still a graph, or three islands sharing a database?

Measured (see the README section this pins): **it is connected** -- 1,240 nodes,
2,478 edges, and only 18 orphans, all of them `Operator` nodes no ONNX Runtime
kernel registers. It is not 1,030 orphans.

What it is missing is a *half*, not the joins. The real layer carries the
hardware and kernel spine and none of the clinical one, so seven labels and eleven
edge types are empty -- including `USES_OPERATOR`, which is the edge the hero
question traverses. That is why 8 of the 16 catalog queries measured for this
were empty against it -- `EA20` joined the catalog later and is empty against it
too, since `Site` is generated-only.

These assertions are about **shape, not counts**. The counts move whenever ONNX
Runtime publishes new kernel registrations -- 734 became 738 during one week --
so pinning them here would produce a test that fails for upstream's reasons
rather than ours. Which labels and edge types `etl/real_layer.py` builds is our
decision, and that is what is asserted.
"""
import pytest

from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

# Shared rather than a second regex over the same comment block: that helper
# already asserts on its match, so a reformatted schema header fails there once
# with a readable message instead of raising AttributeError in two places.
from tests.test_schema_docs import declared_in_schema

GRAPH = "default"

# The hardware and kernel spine, plus the MLPerf submissions.
LABELS_PRESENT = {"Vendor", "SoC", "Accelerator", "Board", "Runtime",
                  "Operator", "Kernel", "Model", "BenchmarkTask", "Deployment"}
# The clinical spine, entirely generated.
LABELS_ABSENT = {"ModelVariant", "Sensor", "SignalStage", "ClinicalTask",
                 "Dataset", "Certification",
                 # `Site` is generated-only by decision (#34,
                 # docs/location-scope.md), so the real layer never carries
                 # one. Listing it here is what fails if that ever changes.
                 "Site"}

EDGES_PRESENT = {"HAS_SOC", "IMPLEMENTS", "MADE_BY", "MEASURES", "ON_BOARD",
                 "PROVIDED_BY", "RUNS_ON", "SOLVES", "TARGETS",
                 "USES_ACCELERATOR", "VIA_RUNTIME"}


@pytest.fixture(scope="module")
def real_only():
    """The graph `--layers real` produces: no generated fleet underneath it."""
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    try:
        from samyama import SamyamaClient
        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

    # Reset first, as every other embedded fixture in this suite does. Measured
    # on `samyama` 0.6.1, two `SamyamaClient.embedded()` instances in one
    # process are independent, so nothing another module loads reaches here --
    # but that is a property of this build, not a guarantee, and
    # `assert synthetic == 0` below would be the confusing way to find out it
    # had changed.
    try:
        client.query("MATCH (n) DETACH DELETE n", GRAPH)
    except Exception:
        pass

    # Exactly what etl/loader.py does for --layers real: start from an empty
    # Fleet rather than filtering a generated one.
    from etl import generate as gen
    from etl import real_layer
    fleet = gen.Fleet(seed=0, scale=1.0)
    real_layer.build_real(fleet, ops)
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client, fleet


def scalar(client, cypher: str) -> int:
    result = client.query(cypher, GRAPH)
    return result.records[0][0] if result.records else 0


def declared_edge_types() -> set[str]:
    return set(declared_in_schema("Edge types"))


def test_the_fixture_built_a_real_only_graph(real_only):
    client, _ = real_only
    total = scalar(client, "MATCH (x) RETURN count(x) AS n")
    assert total > 1000, f"only {total} nodes; the real layer did not build"
    synthetic = scalar(
        client, 'MATCH (x) WHERE x.provenance = "synthetic" RETURN count(x) AS n')
    assert synthetic == 0, f"{synthetic} generated nodes leaked into a real-only load"


def test_which_labels_the_real_layer_carries(real_only):
    client, _ = real_only
    present = {label for label in NODE_LABELS
               if scalar(client, f"MATCH (x:{label}) RETURN count(x) AS n")}
    assert present == LABELS_PRESENT, (
        f"the real layer's label set changed: "
        f"gained {sorted(present - LABELS_PRESENT)}, "
        f"lost {sorted(LABELS_PRESENT - present)}. The README documents which "
        f"catalog queries work against `--layers real`; update it too."
    )
    assert not (present & LABELS_ABSENT), "a clinical-spine label gained real nodes"


def test_which_edge_types_the_real_layer_carries(real_only):
    client, _ = real_only
    present = {t for t in declared_edge_types()
               if scalar(client, f"MATCH (a)-[:{t}]->(b) RETURN count(a.id) AS n")}
    assert present == EDGES_PRESENT, (
        f"the real layer's edge set changed: "
        f"gained {sorted(present - EDGES_PRESENT)}, "
        f"lost {sorted(EDGES_PRESENT - present)}"
    )


def test_the_hero_question_cannot_be_asked_of_the_real_layer(real_only):
    """The documented limitation, asserted so it cannot drift out of the README.

    The hero question walks `Model -[:USES_OPERATOR]-> Operator`. The real layer
    knows which kernels implement which operators, but nothing records which
    operators a model uses -- so `EA01`, `EA02` and `EA11` return nothing.

    If someone gives the real layer a model's operator surface, this fails and
    says the README claim is now wrong, which is the point.
    """
    client, _ = real_only
    uses = scalar(client, "MATCH (a)-[:USES_OPERATOR]->(b) RETURN count(a.id) AS n")
    assert uses == 0, (
        f"{uses} USES_OPERATOR edges in the real layer; the hero question may now "
        f"be answerable there, and the README says it is not"
    )


def test_the_real_layer_is_connected_apart_from_unregistered_operators(real_only):
    """#21's actual question. It is not 1,030 orphans.

    The only nodes with no edge are `Operator`s that no ONNX Runtime kernel
    registers -- a real fact about the upstream data, not a broken join. The
    bound is deliberately loose: the exact number moves when ONNX or ONNX
    Runtime publish, and the claim being defended is "a handful", not a figure.
    """
    client, _ = real_only
    # Built from EDGES_PRESENT rather than every declared type: a newly added
    # edge type would leave its endpoints looking orphaned here. That is safe
    # only because test_which_edge_types_the_real_layer_carries fails first and
    # names the new type -- if that test is ever relaxed, widen this too.
    connected = set()
    for rel in EDGES_PRESENT:
        for src, dst in client.query(
                f"MATCH (a)-[:{rel}]->(b) RETURN a.id, b.id", GRAPH).records:
            connected.add(src)
            connected.add(dst)

    orphans = {}
    for label in LABELS_PRESENT:
        ids = [r[0] for r in
               client.query(f"MATCH (x:{label}) RETURN x.id", GRAPH).records]
        missing = [i for i in ids if i not in connected]
        if missing:
            orphans[label] = len(missing)

    assert set(orphans) <= {"Operator"}, (
        f"labels other than Operator have orphaned nodes: {orphans}"
    )
    total_nodes = scalar(client, "MATCH (x) RETURN count(x) AS n")
    assert sum(orphans.values()) < total_nodes * 0.05, (
        f"{sum(orphans.values())} of {total_nodes} nodes are orphaned; the "
        f"README describes the real layer as connected"
    )
