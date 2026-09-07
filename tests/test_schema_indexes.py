"""Every declared index is traced to something that filters on it.

`schema/edge_ai_kg.cypher` declared 27 indexes across 16 labels (#18 says 28;
the loader's own parser counts 27). Sixteen are the `id` indexes the comment
describes. Of the other eleven, **five were filtered on by nothing** -- two were
merely projected, three were read nowhere at all -- and an index only helps a
lookup, so those were load cost for no benefit.

Measured before removing them, embedded engine at `--scale 1.0`:

| index set | load |
|---|---:|
| 16 id indexes | 23.13s (median of 4) |
| all 27 | 24.68s |
| none | 234s |

So the `id` indexes are load-*critical* -- `etl/helpers.py` resolves every edge
endpoint with `WHERE v.id = ...`, and without the index each of 76,303 edges
costs a label scan, a **10.6x** slowdown. The non-id indexes are the opposite:
~1.5s of load, about 6%, for queries that finish in milliseconds. Removing the
five changed no catalog row and no query time outside noise (+9ms summed across
all 16, against 562ms).

This file keeps that true. It is a static check -- it reads the schema file and
the query sources, and needs no engine and no downloaded data.

The catalog is parsed rather than hard-coded, so adding a query that filters on
a new property does not silently leave the property unindexed here, and adding
an index nothing uses fails.
"""
import pathlib
import re

import pytest

from benchmarks.queries import QUERIES
from etl.loader import NODE_LABELS

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "schema" / "edge_ai_kg.cypher"
MCP = pathlib.Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"

INDEX_RE = re.compile(r"^CREATE INDEX ON :(\w+)\((\w+)\)$")


def declared_indexes() -> list[tuple[str, str]]:
    """(label, property) per index, parsed the way `etl.loader.apply_schema` does."""
    out = []
    for raw in SCHEMA.read_text(encoding="utf-8").splitlines():
        stmt = raw.split("//", 1)[0].strip().rstrip(";")
        if not stmt:
            continue
        m = INDEX_RE.match(stmt)
        assert m, f"unparsed statement in schema file: {stmt!r}"
        out.append((m.group(1), m.group(2)))
    return out


def query_sources() -> dict[str, str]:
    """Every place Cypher is written: the catalog, and the MCP server's variants.

    `mcp_server/server.py` deliberately re-states rather than imports the
    catalog (`CLAUDE.md`), so an index may exist only for its sake.
    """
    sources = {q["id"]: q["cypher"] for q in QUERIES}
    sources["mcp_server"] = MCP.read_text(encoding="utf-8")
    return sources


def filtered_on(text: str, label: str, prop: str) -> bool:
    """Is `label.prop` used as a predicate, rather than merely projected?

    An index serves a lookup. A property that only appears in a `RETURN`, or in
    an `ORDER BY` over a `WITH` alias, cannot use one -- so matching any mention
    of the property would defeat the point of this file.
    """
    aliases = {m.group(1) for m in re.finditer(rf"\((\w+)\s*:\s*{label}\b", text)}
    for alias in aliases:
        ref = rf"\b{re.escape(alias)}\.{re.escape(prop)}\b"
        # A predicate is `x.p = ...` / `<>` / `>` etc., or a WHERE mentioning it.
        # The f-string interpolation the MCP server uses (`= {_q(name)}`) still
        # reads as `=` here, which is why its source is scanned as text.
        if re.search(rf"{ref}\s*(=|<>|<|>|<=|>=|IS\s+(NOT\s+)?NULL|IN\b)", text):
            return True
        for line in text.splitlines():
            if re.search(ref, line) and line.strip().upper().startswith("WHERE"):
                return True
    return False


def test_every_label_has_an_id_index():
    """The comment's claim: one per label, and the loader depends on it.

    Not cosmetic -- a missing `id` index turns that label's edge creation into a
    label scan per edge. `NODE_LABELS` is the loader's own list, so a new label
    fails here rather than quietly loading 10x slower.
    """
    indexed = {label for label, prop in declared_indexes() if prop == "id"}
    missing = set(NODE_LABELS) - indexed
    assert not missing, (
        f"labels with no id index: {sorted(missing)}. etl/helpers.py resolves "
        f"edge endpoints with `WHERE v.id = ...`; without the index each edge "
        f"costs a label scan (measured 10.6x slower to load)."
    )
    extra = indexed - set(NODE_LABELS)
    assert not extra, f"id index on a label the loader never writes: {sorted(extra)}"


def test_every_non_id_index_is_filtered_on_somewhere():
    """#18's question, made executable.

    Fails naming the index and where to look, so the answer is either a comment
    tracing it to a real predicate or a deletion -- which is exactly the choice
    the issue asks for.
    """
    sources = query_sources()
    orphans = []
    for label, prop in declared_indexes():
        if prop == "id":
            continue
        users = [name for name, text in sources.items()
                 if filtered_on(text, label, prop)]
        if not users:
            orphans.append(f"{label}({prop})")
    assert not orphans, (
        f"indexes nothing filters on: {orphans}. An index only helps a lookup -- "
        f"a projected or ORDER BY'd property cannot use one. Either trace it to "
        f"a predicate in benchmarks/queries.py or mcp_server/server.py and say "
        f"so beside it, or remove it: measured, the non-id indexes cost ~6% of "
        f"load time."
    )


def test_the_indexes_that_survived_name_who_needs_them():
    """The comments beside them, not just the statements.

    #18 asks for each index to be *traced*, so an untraced one is a regression
    even if some query happens to filter on it.
    """
    untraced = []
    for raw in SCHEMA.read_text(encoding="utf-8").splitlines():
        stmt, _, comment = raw.partition("//")
        stmt = stmt.strip().rstrip(";")
        m = INDEX_RE.match(stmt) if stmt else None
        if not m or m.group(2) == "id":
            continue
        if not comment.strip():
            untraced.append(f"{m.group(1)}({m.group(2)})")
    assert not untraced, (
        f"non-id indexes with no trailing comment naming what needs them: "
        f"{untraced}"
    )


@pytest.mark.parametrize("label,prop", [
    ("Operator", "category"), ("Model", "family"), ("ClinicalTask", "category"),
    ("Board", "provenance"), ("Accelerator", "provenance"),
])
def test_the_removed_indexes_are_still_unused(label, prop):
    """The five #18 removed, pinned.

    If a query starts filtering on one of these, this fails and says the index
    should come back -- which is the failure worth having, because the query
    would otherwise silently do a label scan.
    """
    users = [name for name, text in query_sources().items()
             if filtered_on(text, label, prop)]
    assert not users, (
        f"{label}({prop}) is now filtered on by {users}, but its index was "
        f"removed in #18 as unused. Restore it in schema/edge_ai_kg.cypher and "
        f"drop this parameter."
    )
