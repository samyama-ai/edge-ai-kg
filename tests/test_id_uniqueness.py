"""`id` uniqueness is a loader invariant, so the loader is what gets checked.

`schema/edge_ai_kg.cypher` says this engine does not parse
`CREATE CONSTRAINT ... REQUIRE ... IS UNIQUE`, so uniqueness of `id` is
"guaranteed by the loader, which mints ids deterministically". Nothing enforced
that. A generator change that reused an id, or a second load against a live
graph, produces duplicate nodes and every published count goes quietly wrong
(#19).

Measured, and it is worth knowing which way round it is:

    1st load                8,463 nodes   8,463 distinct ids
    2nd load, --no-reset   16,926 nodes   8,463 distinct ids   <- every id twice
    after reset + reload     8,463 nodes   8,463 distinct ids

So the loader is idempotent on its default path (`--reset` is on) and is not
with `--no-reset`. The duplication is invisible to a node count -- the count
simply doubles and looks like more data -- but `count(DISTINCT x.id)` catches
it, which is why that is the invariant asserted here rather than any count.
"""
import pytest

from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_nodes
from etl.loader import NODE_LABELS, reset_graph

GRAPH = "default"
SCALE = 0.3
SEED = 4242


def load_nodes(client, fleet):
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])


def scalar(client, cypher: str) -> int:
    result = client.query(cypher, GRAPH)
    return result.records[0][0] if result.records else 0


def counts(client, label: str | None = None) -> tuple[int, int]:
    """(node count, distinct id count) for one label, or for the whole graph."""
    match = f"(x:{label})" if label else "(x)"
    return (scalar(client, f"MATCH {match} RETURN count(x) AS n"),
            scalar(client, f"MATCH {match} RETURN count(DISTINCT x.id) AS n"))


@pytest.fixture(scope="module")
def fleet():
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    built = gen.generate(seed=SEED, scale=SCALE, operators=ops)
    from etl import real_layer
    real_layer.build_real(built, ops)
    return built


@pytest.fixture
def client(fleet):
    try:
        from samyama import SamyamaClient
        c = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")
    reset_graph(c, GRAPH)
    load_nodes(c, fleet)
    return c


def test_every_label_has_distinct_ids(client):
    duplicated = {}
    for label in NODE_LABELS:
        total, distinct = counts(client, label)
        if total != distinct:
            duplicated[label] = (total, distinct)
    assert not duplicated, (
        f"labels whose (count, distinct ids) disagree: {duplicated}; "
        f"ids are minted deterministically, so a collision means the generator "
        f"reused one or the graph was loaded twice"
    )


def test_ids_are_distinct_graph_wide(client):
    """Not just per label -- every id carries its label as a prefix."""
    total, distinct = counts(client)
    assert total == distinct, (
        f"{total} nodes but {distinct} distinct ids: {total - distinct} "
        f"collisions across labels"
    )


def test_loading_twice_without_a_reset_duplicates_every_id(client, fleet):
    """The documented behaviour, pinned. `--no-reset` is not idempotent.

    Asserted rather than merely written down because the failure is invisible to
    a node count -- the count doubles and reads as more data.
    """
    before, before_ids = counts(client)
    load_nodes(client, fleet)                      # second load, no reset
    after, after_ids = counts(client)

    assert after == before * 2, f"expected the node count to double, {before} -> {after}"
    assert after_ids == before_ids, (
        f"distinct ids should not change, {before_ids} -> {after_ids}"
    )
    assert after != after_ids, "the uniqueness assertions above must catch this"


def test_reloading_after_a_reset_is_idempotent(client, fleet):
    """The default path -- `--reset` is on unless asked otherwise."""
    before = counts(client)
    reset_graph(client, GRAPH)
    assert counts(client) == (0, 0), "reset_graph left nodes behind"
    load_nodes(client, fleet)
    assert counts(client) == before, "reload after reset did not reproduce the graph"
