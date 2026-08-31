"""A short edge load is silent, so the loader has to look for it.

`create_edges` returns the number of edges it *submitted*, not the number the
graph received. Endpoints are matched by id and a whole batch shares one
`MATCH`, so one unresolvable id drops every edge in that batch -- with no error
raised, and with the node counts still correct. Measured: submitting five edges
where one endpoint id does not exist creates **zero**, and the loader used to
report all five as loaded.

These pin `verify_edges` on a fixture where the correct answer is known, in both
directions -- it must report the shortfall, and it must not invent one on a
healthy load.
"""
import pytest

from etl.helpers import create_edges, create_nodes
from etl.loader import verify_edges

GRAPH = "default"


EDGE_TYPES = ("VLINK", "VDROP", "VOK", "VBAD")


def count(c, cypher: str) -> int:
    """A count over a relationship type with no instances returns no rows here,
    not a row holding 0 -- the same guard `count_edges_by_type` carries."""
    result = c.query(cypher, GRAPH)
    return result.records[0][0] if result.records else 0


@pytest.fixture
def client():
    try:
        from samyama import SamyamaClient
        c = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

    for label in ("VA", "VB"):
        try:
            c.query(f"MATCH (n:{label}) DETACH DELETE n", GRAPH)
        except Exception:
            pass
    create_nodes(c, GRAPH, "VA", [{"id": f"va{i}"} for i in range(5)])
    create_nodes(c, GRAPH, "VB", [{"id": f"vb{i}"} for i in range(5)])

    # Assert the fixture's own post-condition. The cleanup above swallows its
    # error, and this engine parses no uniqueness constraint, so a failed
    # DETACH DELETE would leave a second node sharing each id -- every `WHERE
    # v.id = ...` would then bind twice and the assertions below would report
    # ("VLINK", 5, 10), indicting verify_edges instead of the leftover state.
    for label in ("VA", "VB"):
        got = count(c, f"MATCH (n:{label}) RETURN count(n.id) AS n")
        assert got == 5, f"dirty fixture: {got} {label} nodes, expected 5"
    for rel in EDGE_TYPES:
        got = count(c, f"MATCH (a)-[:{rel}]->(b) RETURN count(a.id) AS n")
        assert got == 0, f"dirty fixture: {got} leftover {rel} edges from a prior test"
    return c


def test_a_healthy_load_reports_no_shortfall(client):
    edges = [("VA", f"va{i}", "VLINK", "VB", f"vb{i}", None) for i in range(5)]
    create_edges(client, GRAPH, edges)

    rows = verify_edges(client, GRAPH, edges)
    assert rows == [("VLINK", 5, 5)]


def test_one_unresolvable_id_is_reported(client):
    """The batch shares a MATCH, so the whole batch is lost, not just one edge."""
    edges = [("VA", f"va{i}", "VDROP", "VB", f"vb{i}", None) for i in range(4)]
    edges.append(("VA", "va4", "VDROP", "VB", "no-such-id", None))

    submitted = create_edges(client, GRAPH, edges)
    assert submitted == 5, "create_edges still reports what it submitted"

    rows = verify_edges(client, GRAPH, edges)
    assert rows == [("VDROP", 5, 0)], (
        "one bad id should cost the whole batch, and be visible as intended 5 / actual 0"
    )


def test_shortfalls_sort_first(client):
    """The load output shows the worst shortfall at the top."""
    good = [("VA", f"va{i}", "VOK", "VB", f"vb{i}", None) for i in range(3)]
    bad = [("VA", "va0", "VBAD", "VB", "no-such-id", None)]
    create_edges(client, GRAPH, good)
    create_edges(client, GRAPH, bad)

    rows = verify_edges(client, GRAPH, good + bad)
    assert rows[0][0] == "VBAD", f"shortfall should sort first, got {rows}"
    assert rows[0][1:] == (1, 0)
