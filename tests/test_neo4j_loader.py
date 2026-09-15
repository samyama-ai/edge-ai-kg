"""The Neo4j loader's guarantees: same graph in, or no timing out (#47).

`benchmarks/neo4j_client.py` fills a Neo4j with the same `Fleet` the Samyama
side gets, and translates `schema/edge_ai_kg.cypher` so both ends have the same
22 indexes. Everything here is about whether that is true, because a comparison
given two different graphs measures nothing -- and the ways it can go wrong are
all quiet:

- **edge properties dropped**, so Neo4j got a smaller graph to load and a
  different one to query;
- **a partial load**, which `UNWIND ... MATCH ... CREATE` performs without
  complaint and which reports a *faster* time than a complete one;
- **nulls kept on nodes but stripped on edges**, one loader disagreeing with
  itself about what a property is;
- **a destructive wipe running before validation**, which emptied the target
  and then refused to load.

That last one was wrong twice in review -- the identifier check was moved above
the node `CREATE` loop and was still below the `DETACH DELETE`. Ordering is the
guarantee, so it is asserted directly rather than inferred from behaviour.

The comparison harness that uses this loader, and the timings themselves, are
separate: see `benchmarks/compare_neo4j.py` and its tests.
"""
from __future__ import annotations
from __future__ import annotations

import types

import pytest


def test_the_neo4j_loader_writes_edge_properties():
    """Both engines must be given the same graph, not just the same queries.

    `etl/helpers.py` writes edge properties on the Samyama side; this loader
    dropped them, so Neo4j got 310 `USES_OPERATOR` edges without their `count`
    at scale 1.0. No catalog query binds an edge variable today, so no
    published number moved -- the first one that did would have had two
    answers and no way to tell which was right.
    """
    from benchmarks import neo4j_client

    sent = []

    class Fake:
        """Answers the two counts the loader checks: nodes held, edges created.

        `count(r)` must report the number of edges in the fleet or the
        partial-load guard fires -- which is the guard working, but it is not
        what this test is about.
        """

        def run(self, statement, parameters=None):
            sent.append((statement, parameters))
            if "count(r)" in statement:
                return [[len(fleet.edges)]]
            if "count(n)" in statement or "RETURN count" in statement:
                return [[0]]
            return []

    fleet = types.SimpleNamespace(
        nodes={"Model": [{"id": "model:1"}], "Operator": [{"id": "op:1"}]},
        edges=[("Model", "model:1", "USES_OPERATOR", "Operator", "op:1",
                {"count": 3, "_internal": "x", "empty": None})])

    neo4j_client.load_neo4j(Fake(), fleet)
    edge = [(s, p) for s, p in sent if "CREATE (a)-[" in s]
    assert edge, f"no edge CREATE was issued: {[s for s, _ in sent]}"
    statement, params = edge[0]
    assert "SET r = row.props" in statement, (
        f"edge properties are not written: {statement!r}"
    )
    assert params["rows"][0]["props"] == {"count": 3}, (
        f"`_`-prefixed internals and nulls must be stripped, as "
        f"`etl/helpers.py` does: got {params['rows'][0]['props']!r}"
    )


def test_nothing_is_sent_before_the_identifiers_are_checked():
    """The refusal has to come before the first interpolation, not after it.

    The check sat below the node-`CREATE` loop, which had already interpolated
    and sent every label -- so the one statement it existed to gate ran first,
    and the refusal arrived too late to prevent anything. Ordering is the whole
    guarantee here, and nothing else in the suite can see it.
    """
    from benchmarks import neo4j_client

    sent = []

    class Fake:
        def run(self, statement, parameters=None):
            sent.append(statement)
            return [[0]]

    fleet = types.SimpleNamespace(
        nodes={"Model": [{"id": "m:1"}], "Bad Label": [{"id": "b:1"}]},
        edges=[("Model", "m:1", "USES", "Model", "m:1", {})])

    with pytest.raises(neo4j_client.Neo4jLoadError, match="not plain identifiers"):
        neo4j_client.load_neo4j(Fake(), fleet, force_wipe=True)
    # Every statement, not just the ones containing "CREATE". Filtering on
    # CREATE is how this test missed that `DETACH DELETE` ran first: the wipe
    # emptied the target database and the refusal arrived afterwards, so a
    # fleet with an unusable label destroyed the data and loaded nothing. The
    # guarantee is that *nothing* is sent, so that is what is asserted.
    assert sent == [], (
        f"the identifier check must run before anything reaches the server -- "
        f"the count query and the wipe included. These were sent first: {sent}"
    )


def test_node_rows_and_edge_props_agree_about_nulls():
    """One loader, two halves, one rule.

    `SET n = row` with a null is a no-op in Neo4j, so this changes nothing the
    server stores -- it stops the two halves disagreeing about what a property
    is, which is the disagreement under which the edge half silently dropped
    every property it had.
    """
    from benchmarks import neo4j_client

    sent = []

    class Fake:
        def run(self, statement, parameters=None):
            sent.append((statement, parameters))
            return [[len(fleet.edges)]] if "count(r)" in statement else [[0]]

    fleet = types.SimpleNamespace(
        nodes={"Model": [{"id": "m:1", "nothing": None, "_hidden": 1}]},
        edges=[("Model", "m:1", "USES", "Model", "m:1",
                {"count": 2, "nothing": None, "_hidden": 1})])
    neo4j_client.load_neo4j(Fake(), fleet)

    node = next(p for s, p in sent if "CREATE (n:" in s)
    edge = next(p for s, p in sent if "CREATE (a)-[" in s)
    assert node["rows"][0] == {"id": "m:1"}, node["rows"][0]
    assert edge["rows"][0]["props"] == {"count": 2}, edge["rows"][0]["props"]
