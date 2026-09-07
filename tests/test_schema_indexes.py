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
    """(label, property) per index, parsed the way `etl.loader.apply_schema` does.

    Stricter than the loader on purpose: `apply_schema` sends any non-comment
    line to the engine and merely warns if it is rejected, so a stray statement
    would load silently. Here anything that is not `CREATE INDEX ON :L(p)` fails
    with the line quoted -- the schema file is index declarations and nothing
    else, and a new kind of statement should be a decision rather than a
    surprise.
    """
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


CASE_SPAN = re.compile(r"\bCASE\b.*?\bEND\b", re.DOTALL | re.IGNORECASE)


def without_case_expressions(text: str) -> str:
    """Blank out `CASE ... END` spans before looking for predicates.

    `sum(CASE WHEN v.precision = "fp32" ...)` reads like a predicate and is not:
    conditional aggregation runs over rows the `MATCH` has already produced, so
    no index is consulted. Counting it would bless an index whose only user is
    an aggregate -- the same overclaim this file exists to prevent, and the one
    the surviving trace comments were guilty of before review.

    Blanked rather than deleted so line structure survives for the `WHERE`
    check below.
    """
    return CASE_SPAN.sub(lambda m: " " * len(m.group(0)), text)


def filtered_on(text: str, label: str, prop: str) -> bool:
    """Is `label.prop` used as a predicate, rather than merely projected?

    An index serves a lookup. A property that only appears in a `RETURN`, in an
    `ORDER BY` over a `WITH` alias, or inside a `CASE` expression cannot use one
    -- so matching any mention of the property would defeat the point of this
    file.
    """
    text = without_case_expressions(text)
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


def traced_indexes() -> list[tuple[str, str, str]]:
    """(label, property, trailing comment) for each non-id index."""
    out = []
    for raw in SCHEMA.read_text(encoding="utf-8").splitlines():
        stmt, _, comment = raw.partition("//")
        stmt = stmt.strip().rstrip(";")
        m = INDEX_RE.match(stmt) if stmt else None
        if m and m.group(2) != "id":
            out.append((m.group(1), m.group(2), comment.strip()))
    return out


def test_the_indexes_that_survived_name_who_needs_them():
    """The comments beside them, not just the statements.

    #18 asks for each index to be *traced*, so an untraced one is a regression
    even if some query happens to filter on it.
    """
    untraced = [f"{label}({prop})" for label, prop, comment in traced_indexes()
                if not comment]
    assert not untraced, (
        f"non-id indexes with no trailing comment naming what needs them: "
        f"{untraced}"
    )


def test_each_query_named_in_a_trace_comment_really_filters_on_it():
    """The trace comment is the deliverable, so it is checked like one.

    Added because the first version of these comments overclaimed: they named
    every query that *read* the property, including `WITH op.name AS operator`
    projections, which is precisely the justification this file rejects when it
    removes an index. Asserting a comment merely exists let that through.

    Only the `EAnn` ids are checked. The `mcp <tool>` half names a Python
    function rather than a query id, and `mcp_server/server.py` is already
    scanned whole by the test above.
    """
    sources = query_sources()
    wrong = []
    for label, prop, comment in traced_indexes():
        for qid in re.findall(r"\bEA\d{2}\b", comment):
            if qid not in sources:
                wrong.append(f"{label}({prop}) names {qid}, which is not in the catalog")
            elif not filtered_on(sources[qid], label, prop):
                wrong.append(f"{label}({prop}) names {qid}, which only projects it")
    assert not wrong, (
        "trace comments naming queries that do not filter on the property:\n  "
        + "\n  ".join(wrong)
        + "\nAn index serves a lookup; a projection or a CASE expression is not "
          "one. List only the queries with the property in a predicate."
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
