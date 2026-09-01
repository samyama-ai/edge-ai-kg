"""`provenance` and `source` are a headline claim, so they are asserted here.

The README says the real/synthetic split is "queryable, not just documented --
every node carries `provenance` and `source`", and `--layers real`, the dataset
card's honesty argument and any provenance breakdown all depend on it.

A node created without those properties fails nothing. The node count stays
right; it is simply invisible to `--layers real` and absent from the split,
which quietly stops adding up. That failure has no symptom, so it is pinned
here (#20).

Assertions are made against a loaded graph rather than the `Fleet`, because the
claim the README makes is about what the graph holds -- `Fleet.add_nodes` doing
the stamping is the mechanism, not the guarantee.
"""
import pytest

from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
SCALE = 0.3
SEED = 4242
VALID_PROVENANCE = {"real", "synthetic"}


@pytest.fixture(scope="module")
def loaded():
    """Both layers in an embedded engine -- the real layer is half the point."""
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
    return client, fleet


def scalar(client, cypher: str) -> int:
    result = client.query(cypher, GRAPH)
    return result.records[0][0] if result.records else 0


def provenance_counts(client) -> dict[str, int]:
    result = client.query(
        "MATCH (n) WITH n.provenance AS provenance, count(n) AS c "
        "RETURN provenance, c", GRAPH)
    return {rec[0]: rec[1] for rec in result.records}


def test_no_node_is_missing_provenance(loaded):
    client, _ = loaded
    missing = scalar(client, "MATCH (n) WHERE n.provenance IS NULL RETURN count(n) AS n")
    assert missing == 0, (
        f"{missing} nodes carry no `provenance`; they are invisible to "
        f"`--layers real` and absent from every provenance breakdown"
    )


def test_provenance_is_only_real_or_synthetic(loaded):
    client, _ = loaded
    seen = set(provenance_counts(client))
    unexpected = seen - VALID_PROVENANCE
    # sorted by repr: a node with no `provenance` shows up here as None, and
    # sorting a mixed str/None set raises TypeError -- which would blame this
    # test rather than the data it caught.
    assert not unexpected, (
        f"unexpected provenance values: {sorted(unexpected, key=repr)}"
    )


def test_no_node_is_missing_a_source(loaded):
    client, _ = loaded
    missing = scalar(client, "MATCH (n) WHERE n.source IS NULL RETURN count(n) AS n")
    empty = scalar(client, 'MATCH (n) WHERE n.source = "" RETURN count(n) AS n')
    assert (missing, empty) == (0, 0), (
        f"{missing} nodes have no `source` and {empty} have an empty one"
    )


def test_the_split_adds_up_to_the_node_count(loaded):
    """The README quotes the two halves; they have to be the whole."""
    client, _ = loaded
    counts = provenance_counts(client)
    total = scalar(client, "MATCH (n) RETURN count(n) AS n")
    # Only the two valid buckets are summed. Summing every bucket would include
    # a None one and still reach the total, so a node with no `provenance`
    # would slip past exactly the assertion meant to catch it.
    stamped = sum(counts.get(value, 0) for value in VALID_PROVENANCE)
    assert stamped == total, (
        f"real + synthetic = {stamped}, but the graph holds {total} nodes; "
        f"full breakdown {counts}"
    )
    assert counts.get("real", 0) > 0 and counts.get("synthetic", 0) > 0, (
        f"both layers should be present in this fixture, got {counts}"
    )
