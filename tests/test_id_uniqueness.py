"""`id` uniqueness is a loader invariant, so the loader is what gets checked.

`schema/edge_ai_kg.cypher` says this engine does not parse
`CREATE CONSTRAINT ... REQUIRE ... IS UNIQUE`, so uniqueness of `id` is
"guaranteed by the loader, which mints ids deterministically". Nothing enforced
that. A generator change that reused an id, or a second load against a live
graph, produces duplicate nodes and every published count goes quietly wrong
(#19).

Checked at two levels. `test_generator_mints_distinct_ids` needs no engine and
pins the claim the schema comment actually makes -- deterministic minting. The
rest load a graph, because what the README publishes is what the graph holds.

**Scope: `SCALE = 0.3`, `SEED = 4242`.** The automated check is small-scale.
A full load (25,150 nodes, all 16 labels) was verified by hand and holds, but a
collision that only appears once the generator's ranges widen would not be
caught here.

**On `--no-reset`.** Loading twice into the same graph mints every id a second
time, and nothing rejects it at write time. The run does not survive, though,
and not via the loader's own node-count check: duplicated ids make each edge
batch's `MATCH` bind more than one node per endpoint -- one submitted edge
becomes four when both endpoints are duplicated -- and on a real batch the
engine is OOM-killed during edge creation, before that check runs. Verified
twice against v1.7.0 (`OOMKilled=true`, exit 137).

That is why the CLI path is not exercised here: a test of
`python -m etl.loader --no-reset` run twice reliably kills the server. What the
mutating tests below assert is the node-level duplication itself.
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


def id_set(client) -> set[str]:
    result = client.query("MATCH (x) RETURN x.id AS id", GRAPH)
    return {rec[0] for rec in result.records}


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


def embedded_with(fleet):
    try:
        from samyama import SamyamaClient
        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")
    reset_graph(client, GRAPH)
    load_nodes(client, fleet)

    # The fixture's own post-condition. `(0, 0)` compares equal, so a label that
    # loaded nothing would satisfy every distinctness assertion below without
    # ever being looked at.
    for label, rows in fleet.nodes.items():
        if rows and label in NODE_LABELS:
            loaded = scalar(client, f"MATCH (x:{label}) RETURN count(x) AS n")
            assert loaded == len(rows), (
                f"fixture loaded {loaded} {label} nodes, fleet holds {len(rows)}"
            )
    return client


@pytest.fixture(scope="module")
def loaded(fleet):
    """Shared by the read-only assertions -- loading this fleet is not cheap."""
    return embedded_with(fleet)


@pytest.fixture
def fresh(fleet):
    """A private graph for the tests that mutate it."""
    return embedded_with(fleet)


def test_generator_mints_distinct_ids(fleet):
    """The schema comment's actual claim, checked without an engine."""
    collisions = {}
    for label, rows in fleet.nodes.items():
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            collisions[label] = len(ids) - len(set(ids))
    assert not collisions, f"the generator reused ids: {collisions}"

    everything = [r["id"] for rows in fleet.nodes.values() for r in rows]
    assert len(everything) == len(set(everything)), (
        f"{len(everything) - len(set(everything))} ids collide across labels"
    )


def test_every_label_has_distinct_ids(loaded):
    duplicated = {}
    for label in NODE_LABELS:
        total, distinct = counts(loaded, label)
        if total != distinct:
            duplicated[label] = (total, distinct)
    assert not duplicated, (
        f"labels whose (count, distinct ids) disagree: {duplicated}; "
        f"ids are minted deterministically, so a collision means the generator "
        f"reused one or the graph was loaded twice"
    )


def test_ids_are_distinct_graph_wide(loaded):
    """Not just per label -- every id carries its label as a prefix."""
    total, distinct = counts(loaded)
    assert total == distinct, (
        f"{total} nodes but {distinct} distinct ids: {total - distinct} "
        f"collisions across labels"
    )


def test_loading_twice_without_a_reset_duplicates_every_id(fresh, fleet):
    """Nothing rejects the second write. See the module docstring for what then
    happens to a real load."""
    before, before_ids = counts(fresh)
    load_nodes(fresh, fleet)                       # second load, no reset
    after, after_ids = counts(fresh)

    assert after == before * 2, f"expected the node count to double, {before} -> {after}"
    assert after_ids == before_ids, (
        f"distinct ids should not change, {before_ids} -> {after_ids}"
    )
    assert after != after_ids, "the uniqueness assertions above must catch this"


def test_reloading_after_a_reset_reproduces_the_same_ids(fresh, fleet):
    """The default path -- `--reset` is on unless asked otherwise.

    Compares the id *set*, not counts. `docs/engine-notes.md` item 8 records
    that `DETACH DELETE` is not a true reset -- the property store survives it --
    so this asserts identity of ids and makes no claim about property values.
    """
    before = id_set(fresh)
    reset_graph(fresh, GRAPH)
    assert counts(fresh) == (0, 0), "reset_graph left nodes behind"
    load_nodes(fresh, fleet)
    assert id_set(fresh) == before, "reload after reset did not reproduce the same ids"
