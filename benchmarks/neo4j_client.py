"""Talking to Neo4j: the HTTP client, its schema translation, and its loader.

Split from `benchmarks/compare_neo4j.py`, which grew past the point a reviewer
would read in one pass. The seam is I/O against Neo4j on this side, and timing
and comparison on the other -- everything here is about getting a graph into
Neo4j correctly, and none of it is about measuring anything.

Correctness matters more here than anywhere else in the comparison, because a
mistake produces a *plausible number* rather than an error:

- endpoints are matched **with their labels**, because every index in
  `schema/edge_ai_kg.cypher` is per-label and a label-free `MATCH (a) WHERE
  a.id = ...` plans an `AllNodesScan`;
- `neo4j_indexes` refuses to return an empty list, because a silent zero would
  create no indexes and verify none;
- the wipe is gated and batched, and sits outside the returned time.
"""
from __future__ import annotations

import base64
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

NEO4J_DB = "neo4j"
SCHEMA_PATH = str(pathlib.Path(__file__).resolve().parent.parent
                  / "schema" / "edge_ai_kg.cypher")


class Neo4jLoadError(RuntimeError):
    """Anything that makes a load impossible or its timing meaningless.

    One type for the module, raised rather than exited. `SystemExit` from a
    library helper takes the choice away from the caller -- and this module had
    both disciplines at once, `SystemExit` in four places and `ValueError` in
    one, which meant a caller wanting to handle a bad fleet had to catch two
    unrelated things and hope it had found them all. `compare_neo4j.main` turns
    this into an exit, because it is the caller that genuinely cannot continue.
    """


class Neo4j:
    """The HTTP transactional endpoint, so no driver dependency is added."""

    def __init__(self, url: str, user: str, password: str,
                 database: str = NEO4J_DB):
        """`database` is configurable because the other three were.

        It was hardcoded to `neo4j` while host, user and password all came from
        flags -- so pointing this at a server whose benchmark data lives in
        another database silently talked to the wrong one, or to an empty one,
        and reported the load time for it.
        """
        self.url = f"{url.rstrip('/')}/db/{database}/tx/commit"
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.headers = {"Content-Type": "application/json",
                        "Authorization": f"Basic {token}"}

    def run(self, statement: str, parameters: dict | None = None) -> list:
        payload = {"statements": [{"statement": statement,
                                   "parameters": parameters or {}}]}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                         headers=self.headers)
        try:
            with urllib.request.urlopen(request) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            # A wrong --password is a 401, and a raw traceback is a poor way to
            # say so in a module that otherwise explains its failures.
            hint = ("  check --user/--password" if exc.code in (401, 403)
                    else "  check --neo4j points at the HTTP port (7474, not 7687)")
            raise Neo4jLoadError(f"Neo4j returned HTTP {exc.code} {exc.reason}\n{hint}"
                             ) from exc
        except urllib.error.URLError as exc:
            raise Neo4jLoadError(
                f"cannot reach Neo4j at {self.url}: {exc.reason}\n"
                f"  start one with: docker run -d -p 7474:7474 "
                f"-e NEO4J_AUTH=neo4j/benchmarkpw neo4j:5-community") from exc
        if body["errors"]:
            error = body["errors"][0]
            raise RuntimeError(f"{error['code']}: {error['message'].splitlines()[0]}")
        return [row["row"] for row in body["results"][0]["data"]]


# `schema/edge_ai_kg.cypher` in Neo4j's syntax. Parsed from that file rather
# than restated, so the two cannot drift -- an index added there is an index
# Neo4j gets, and a comparison that silently gave one engine fewer indexes than
# the other would be measuring index availability.


def neo4j_indexes(source: str = SCHEMA_PATH) -> list[str]:
    text = pathlib.Path(source).read_text(encoding="utf-8")
    out = [f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
           for label, prop in
           re.findall(r"^CREATE INDEX ON :(\w+)\((\w+)\);", text, re.MULTILINE)]
    # Empty is never right, and silence here is the worst outcome available: a
    # changed spelling in the schema file would make `load_neo4j` create no
    # indexes and `--reuse-neo4j` verify none, so the comparison would run
    # unindexed against an indexed Samyama and say nothing. That is the bug this
    # module was corrected for; a parser that can return `[]` reintroduces it.
    if not out:
        raise Neo4jLoadError(
            f"no `CREATE INDEX ON :Label(prop);` statements parsed from "
            f"{source}. Either the file moved or its spelling changed -- fix "
            f"this parser. Continuing would benchmark an unindexed Neo4j.")
    return out


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_identifiers(fleet) -> None:
    """Refuse any label or relationship type that is interpolated but unquotable.

    They come from `Fleet`, which this repo builds from its own generator and
    parsers -- not from anything a caller supplies -- but interpolation is
    interpolation.

    `src_label` and `tgt_label` are included, not only `rel` and the
    `fleet.nodes` keys: an edge may name an endpoint label with no node rows,
    which is then absent from `fleet.nodes` and was never checked. Today every
    endpoint label does appear there, so the narrower set was right by luck.

    `Neo4jLoadError`, not `SystemExit`: see that class. A caller may want to
    handle a bad fleet, and killing the interpreter from a library helper takes
    the choice away.
    """
    # The tuple shape first. This check used to live beside the edge loop, far
    # below -- unreachable, because the comprehension underneath unpacks the
    # same tuples and would raise a bare `ValueError: not enough values to
    # unpack` before the friendly message could run.
    wrong = {len(edge) for edge in fleet.edges} - {6}
    if wrong:
        raise Neo4jLoadError(
            f"`Fleet.edges` contains {sorted(wrong)}-tuples, not the 6-tuple "
            f"(src_label, src_id, rel, tgt_label, tgt_id, props) this unpacks. "
            f"Checking only the first entry would have missed a mixed list. "
            f"`etl/helpers.py` changed shape; fix this loader rather than "
            f"letting it mislabel endpoints.")

    suspicious = sorted({part for sl, _s, rel, tl, _t, _p in fleet.edges
                         for part in (sl, rel, tl) if not IDENTIFIER.match(part)}
                        | {label for label in fleet.nodes
                           if not IDENTIFIER.match(label)})
    if suspicious:
        raise Neo4jLoadError(
            f"labels or relationship types that are not plain identifiers: "
            f"{suspicious}. These are interpolated into Cypher and would have "
            f"to be quoted; refusing rather than guessing.")


def load_neo4j(client: Neo4j, fleet, batch: int = 1000, force_wipe: bool = False) -> float:
    """Fill Neo4j with the same fleet, via parameterised UNWIND.

    **Wipes the target database first**, and refuses to do that silently. The
    URL comes from `--neo4j` and nothing stopped this pointing at something that
    matters; a benchmark is not a reason to delete someone's graph without
    saying so.

    The wipe is outside the returned time. It is not part of loading, Samyama's
    path does not do it -- an embedded client starts empty -- and including it
    would put a delete of 25,150 nodes on one side of a figure printed next to
    the other.
    """
    # First, before the count and long before the wipe. This used to sit below
    # the node `CREATE` loop; hoisting it above that loop was not far enough,
    # because `DETACH DELETE` runs earlier still -- so a fleet with an
    # unusable label emptied the target database and *then* refused to load,
    # leaving it worse than it was found. Nothing is sent until the fleet is
    # known to be loadable.
    check_identifiers(fleet)

    held = client.run("MATCH (n) RETURN count(n)")[0][0]
    if held and not force_wipe:
        raise Neo4jLoadError(
            f"the Neo4j at this URL holds {held:,} nodes and loading would "
            f"delete them. Pass --force-wipe if that is what you want, or "
            f"--reuse-neo4j if it already holds this exact fleet.")
    if held:
        # Batched. A single `MATCH (n) DETACH DELETE n` builds one transaction
        # over every node, which is fine at 25,150 and not fine on a database
        # someone points this at by mistake.
        while True:
            deleted = client.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n "
                "RETURN count(*)")[0][0]
            if not deleted:
                break

    # Before the first interpolation, not after it. This check used to sit
    # below the node loop, which had already sent every label to the server --
    # so the one statement it was meant to gate ran first and the refusal
    # arrived too late to prevent anything.
    start = time.perf_counter()
    for label, rows in fleet.nodes.items():
        if not rows:
            continue
        for i in range(0, len(rows), batch):
            # Nulls dropped, as the edge props below do and as
            # `etl/helpers.py`'s `props_map` does on the Samyama side. `SET n =
            # row` with a null is a no-op in Neo4j, so this changes nothing
            # Neo4j stored -- it stops the two halves of this loader disagreeing
            # about what a property is, which is how the edge half came to drop
            # them silently in the first place.
            chunk = [{k: v for k, v in row.items()
                      if not k.startswith("_") and v is not None}
                     for row in rows[i:i + batch]]
            client.run(f"UNWIND $rows AS row CREATE (n:{label}) SET n = row",
                       {"rows": chunk})
    # Created here, after the nodes and before the edges: an index built over
    # existing rows is cheaper than one maintained through every insert, and
    # the endpoint lookups below need them -- the same reason `etl/helpers.py`
    # matches endpoints with `WHERE id = ...`.
    #
    # Timed separately and subtracted, rather than moved. Moving it before the
    # nodes would change what is being measured; leaving it inside would put a
    # `db.awaitIndexes` wait into Neo4j's figure while Samyama's side applies
    # its schema outside its own timed region.
    index_start = time.perf_counter()
    for statement in neo4j_indexes():
        client.run(statement)
    client.run("CALL db.awaitIndexes(300)")
    index_seconds = time.perf_counter() - index_start

    # Grouped by (source label, type, target label), and the labels are in the
    # MATCH. Every index in `schema/edge_ai_kg.cypher` is per-label, so a
    # label-free `MATCH (a) WHERE a.id = ...` can use none of them: Neo4j plans
    # an AllNodesScan, twice per edge, over every node in the graph. Confirmed
    # from the plan --
    #
    #   MATCH (a)        WHERE a.id = "x"  ->  Filter, AllNodesScan
    #   MATCH (a:Board)  WHERE a.id = "x"  ->  NodeIndexSeek
    #
    # An earlier version shipped the label-free form while calling
    # `db.awaitIndexes` immediately above it, so it waited for indexes it then
    # could not use, and printed the resulting load time next to Samyama's.
    by_shape: dict[tuple[str, str, str], list[dict]] = {}
    # Edge properties, which this loader used to drop. `etl/helpers.py` writes
    # them on the Samyama side (`props_map`), and 310 `USES_OPERATOR` edges
    # carry a `count` at scale 1.0 -- so Neo4j was being given a smaller graph
    # to load and a different graph to query. No catalog query binds an edge
    # variable today, so no published number moved; the first one that does
    # would have got two different answers and one of them silently right.
    #
    # `SET r = row.props` rather than an interpolated map: the values come from
    # the generator, and a parameter needs no quoting rules.
    for src_label, src, rel, tgt_label, tgt, props in fleet.edges:
        by_shape.setdefault((src_label, rel, tgt_label), []).append(
            {"src": src, "tgt": tgt,
             "props": {k: v for k, v in (props or {}).items()
                       if not k.startswith("_") and v is not None}})
    for (src_label, rel, tgt_label), pairs in by_shape.items():
        for i in range(0, len(pairs), batch):
            client.run(
                f"UNWIND $rows AS row "
                f"MATCH (a:{src_label}) WHERE a.id = row.src "
                f"MATCH (b:{tgt_label}) WHERE b.id = row.tgt "
                f"CREATE (a)-[r:{rel}]->(b) SET r = row.props",
                {"rows": pairs[i:i + batch]})
    elapsed = time.perf_counter() - start - index_seconds

    # `UNWIND ... MATCH ... CREATE` creates nothing for a row whose endpoints do
    # not match, and says nothing about it. A fleet that half-loaded would
    # report a *faster* time than a complete one and be compared against
    # Samyama's full graph -- the failure flattering us, and silently.
    created = client.run("MATCH ()-[r]->() RETURN count(r)")[0][0]
    if created != len(fleet.edges):
        raise Neo4jLoadError(
            f"Neo4j holds {created:,} relationships after loading "
            f"{len(fleet.edges):,} edges. `UNWIND ... MATCH ... CREATE` skips a "
            f"row whose endpoints it cannot find and reports nothing, so a "
            f"partial load would have been timed as a complete one -- and a "
            f"smaller graph answers faster.")
    return elapsed
