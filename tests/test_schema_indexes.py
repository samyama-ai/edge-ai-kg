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
costs a label scan -- **10.6x** slower on `samyama` 0.6.1 and **6.4x** on
1.7.1, reproducible with `python -m benchmarks.ingest --no-indexes`. The
non-id indexes are the opposite:
~1.5s of load, about 6%, for queries that finish in milliseconds. Removing the
five changed no catalog row and no query time outside noise (+9ms summed across
all 16, against 562ms).

This file keeps that true. It is a static check -- it reads the schema file and
the Cypher sources, and needs no engine and no downloaded data.

## Under-detection is the dangerous direction

The remedy this file prints is *delete this index*, so a missed predicate does
not leave the schema as it was -- it degrades it, and the test that should have
caught the mistake is the one that caused it. Two rounds of review found exactly
that failure twice:

- `query_sources()` claimed to read "every place Cypher is written" and read two
  of five. `demo/demo.py` alone carries nine queries with real predicates, so a
  demo beat filtering `m.family` would have had `Model(family)` reported as an
  orphan and its index deleted.
- `filtered_on()` matched only `prop <op>` or a line beginning `WHERE`, so
  `AND op.name STARTS WITH ...` on a continuation line read as *unfiltered*.

Both are fixed by construction rather than by adding cases:

- `CYPHER_SOURCES` is checked for completeness by
  `test_every_file_holding_cypher_is_scanned`, so a new file carrying queries
  fails here instead of being silently unscanned.
- Predicates are found by extracting the `WHERE` clause and asking whether the
  property appears anywhere inside it. Operator spelling stops mattering, so
  `STARTS WITH`, `CONTAINS`, `=~` and bare `NOT b.flag` all work without being
  enumerated.

Extracting `WHERE` spans also removed the need to strip `CASE ... END`: measured
across the catalog, the demo and the MCP server, no `WHERE` clause contains a
`CASE`, and every `CASE` lives in a `WITH` or `RETURN`. The previous
case-insensitive strip ran over Python docstrings too, so the word "case" in a
docstring blanked every predicate up to the next `END)`.

Queries are compared one at a time. Scanning a whole file at once lets an alias
bound in one query pair with a predicate in another, inventing a user that no
single query is.
"""
import pathlib
import re

import pytest

from benchmarks.queries import QUERIES
from etl.loader import NODE_LABELS

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema" / "edge_ai_kg.cypher"
MCP = ROOT / "mcp_server" / "server.py"

INDEX_RE = re.compile(r"^CREATE INDEX ON :(\w+)\((\w+)\)$")

# Every non-test file that writes Cypher. Completeness is asserted below rather
# than trusted -- this list being short is what made an index look unused.
CYPHER_SOURCES = (
    # Two `count(n)` sanity queries, used to decide whether the graph is empty.
    # The loader lives in `benchmarks/neo4j_client.py` and the natural hero
    # query in `benchmarks/natural_ea01.py`. Scanned because the guard's
    # contract is "every file with Cypher is read".
    "benchmarks/compare_neo4j.py",
    # `NATURAL_EA01`, the hand-written hero query. Filters `m.id` and `a.id`,
    # which every label already indexes, so it justifies nothing new.
    "benchmarks/natural_ea01.py",
    # The engine-notes probe. Listed because it holds Cypher, and inert: every
    # probe builds its own throwaway labels (`:B2`, `:D3`, `:V4`, ...), which
    # name nothing the schema indexes. Notes 2 and 3 used to build real
    # `:Board` and `:Deployment` rows; they no longer do.
    #
    # Scanned regardless: the guard's contract is "every file with Cypher is
    # read", and exempting one because it *looks* irrelevant is how the
    # `demo/demo.py` gap happened. The CLI module holds no Cypher of its own;
    # the harness only resets and counts.
    "benchmarks/engine_notes_cases.py",
    "benchmarks/engine_notes_harness.py",
    # The Neo4j loader half of the comparison: `UNWIND ... CREATE` plus endpoint
    # lookups that filter on `id`, which every label already indexes.
    "benchmarks/neo4j_client.py",
    "benchmarks/queries.py",
    # Its Cypher is two count queries used to decide whether the graph is empty.
    # They do name a property -- `count(n.id)` -- but only in a `RETURN`, never
    # in a `WHERE`, so they justify no index under this file's rule that only a
    # filter counts as a user. Scanned because the guard's contract is "every
    # file with Cypher is read", not "every file that might justify an index".
    "benchmarks/snapshot.py",
    "benchmarks/vector_probe.py",
    "demo/demo.py",
    "demo/questions.py",
    "etl/loader.py",
    "mcp_server/server.py",
)

# A Cypher clause keyword, used to find where a WHERE clause stops.
CLAUSE = (r"(?:OPTIONAL\s+MATCH|MATCH|WITH|RETURN|ORDER\s+BY|LIMIT|SKIP"
          r"|CREATE|MERGE|DELETE|SET|UNION|DETACH)")
WHERE_SPAN = re.compile(rf"\bWHERE\b(.*?)(?=\b{CLAUSE}\b|$)", re.DOTALL | re.IGNORECASE)

# A triple-quoted string holding a query. Used to split a .py file into units so
# aliases and predicates cannot be paired across unrelated queries.
TRIPLE_QUOTED = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)

# `@mcp.tool()`-decorated functions only, sync or async. Matching a bare `def`
# also collected `client`, `_rows` and `_q`, so a trace comment naming `mcp _q`
# would have passed the existence check; matching only `def` missed `async def`
# entirely, which would report a real tool as non-existent.
TOOL_DEF = re.compile(r"^@mcp\.tool\(\)[^\n]*\n(?:async\s+)?def\s+(\w+)", re.MULTILINE)


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


def mcp_tools() -> dict[str, str]:
    """Each `@mcp.tool()` function's source, separately.

    Per function rather than one blob: `filtered_on` takes aliases and
    predicates from whatever text it is given, so the whole file would let an
    alias bound in `device_path` combine with a predicate in `fallback_audit`
    into a "user" no single tool is.
    """
    src = MCP.read_text(encoding="utf-8")
    starts = [(m.group(1), m.start()) for m in TOOL_DEF.finditer(src)]
    out = {}
    for i, (name, pos) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(src)
        out[name] = src[pos:end]
    return out


def query_units() -> dict[str, str]:
    """One entry per query, from every file in `CYPHER_SOURCES`.

    The catalog is taken structurally from `QUERIES`; the MCP server per
    decorated tool; everything else by triple-quoted string, which is how the
    demo and the loader write Cypher. Units are kept apart so a predicate in one
    query cannot justify an index for another.
    """
    units = {q["id"]: q["cypher"] for q in QUERIES}
    units.update({f"mcp {name}": body for name, body in mcp_tools().items()})
    for rel in CYPHER_SOURCES:
        if rel in ("benchmarks/queries.py", "mcp_server/server.py"):
            continue                      # already covered, structurally
        text = (ROOT / rel).read_text(encoding="utf-8")
        for i, m in enumerate(TRIPLE_QUOTED.finditer(text)):
            body = m.group(1) or m.group(2) or ""
            if re.search(r"\bMATCH\b", body, re.IGNORECASE):
                units[f"{rel}#{i}"] = body
        # Single-line queries passed straight to .query("...") are not
        # triple-quoted; catch those too.
        for i, m in enumerate(re.finditer(r"""["'](\s*MATCH\b[^"']*)["']""", text)):
            units[f"{rel}~{i}"] = m.group(1)
    return units


def filtered_on(text: str, label: str, prop: str) -> bool:
    """Is `label.prop` used as a predicate, rather than merely projected?

    An index serves a lookup. A property that only appears in a `RETURN`, in an
    `ORDER BY` over a `WITH` alias, or inside a `CASE` aggregate cannot use one.

    Asks whether the property appears inside a `WHERE` clause, rather than
    whether it is followed by an operator this file happens to know. That is
    what makes `AND op.name STARTS WITH ...` and `AND NOT b.battery_powered`
    count -- both are predicates, and neither matched the operator list this
    replaced.
    """
    aliases = {m.group(1) for m in re.finditer(rf"\((\w+)\s*:\s*{label}\b", text)}
    if not aliases:
        return False
    ref = re.compile(rf"\b(?:{'|'.join(map(re.escape, aliases))})\.{re.escape(prop)}\b")
    return any(ref.search(clause) for clause in WHERE_SPAN.findall(text))


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


def removed_indexes() -> list[tuple[str, str]]:
    """The `// Removed as unused (#18)` block, read from the schema file.

    Parsed rather than restated here. The list lived in two places -- that
    comment and a `parametrize` -- with nothing tying them together, so
    restoring an index and updating one would leave the other silently wrong.
    """
    text = SCHEMA.read_text(encoding="utf-8")
    # Stops at the first bare `//`, which separates this block from the
    # "Deliberately NOT added" one below it. Both are runs of `//` lines listing
    # `Label(property)`, so a greedy match silently merges the two and every
    # not-added property gets asserted as a removed index.
    # `[^\S\n]` not `\s`: `\s` matches the newline, so a bare `//` separator
    # line lets the run bridge straight into the next block.
    block = re.search(r"^// Removed as unused \(#18\)[^\n]*\n((?://[^\S\n]+\S[^\n]*\n)+)",
                      text, re.MULTILINE)
    assert block, (
        "no `// Removed as unused (#18)` block in schema/edge_ai_kg.cypher; "
        "this test reads that comment rather than keeping its own copy"
    )
    found = re.findall(r"^//\s+(\w+)\((\w+)\)", block.group(1), re.MULTILINE)
    assert found, "the `Removed as unused` block lists no `Label(property)` entries"
    return found


def test_every_file_holding_cypher_is_scanned():
    """`CYPHER_SOURCES` is complete, so an index cannot look unused.

    The list being short is the defect this closes: `demo/demo.py` carries nine
    queries with real predicates and was not read, so a demo beat filtering
    `m.family` would have had `Model(family)` reported as an orphan and deleted.

    `tests/` is excluded deliberately. Tests are not what an index is there to
    serve, and letting one justify an index would make the check circular.
    """
    holding = set()
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith((".venv/", "tests/")) or "egg-info" in rel:
            continue
        if re.search(r"\bMATCH\s*\(", path.read_text(encoding="utf-8")):
            holding.add(rel)
    missing = holding - set(CYPHER_SOURCES)
    assert not missing, (
        f"files writing Cypher that CYPHER_SOURCES does not scan: "
        f"{sorted(missing)}. Add them -- an unscanned query makes its index look "
        f"unused, and the failure message here tells the author to delete it."
    )
    gone = set(CYPHER_SOURCES) - holding
    assert not gone, f"CYPHER_SOURCES lists files that no longer hold Cypher: {sorted(gone)}"


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
        f"costs a label scan (measured 10.6x slower to load on 0.6.1, 6.4x "
        f"on 1.7.1 -- `python -m benchmarks.ingest --no-indexes`)."
    )
    extra = indexed - set(NODE_LABELS)
    assert not extra, f"id index on a label the loader never writes: {sorted(extra)}"


def test_every_non_id_index_is_filtered_on_somewhere():
    """#18's question, made executable.

    The message deliberately does **not** say "delete it". This detector can
    only under-detect -- it reads static text -- and its remedy is destructive,
    so a failure here is a prompt to check by hand first.
    """
    units = query_units()
    orphans = []
    for label, prop in declared_indexes():
        if prop == "id":
            continue
        if not any(filtered_on(text, label, prop) for text in units.values()):
            orphans.append(f"{label}({prop})")
    assert not orphans, (
        f"indexes no scanned query filters on: {orphans}.\n"
        f"An index only helps a lookup -- a projection, an ORDER BY over a WITH "
        f"alias, or a CASE aggregate cannot use one.\n"
        f"Before removing anything, confirm by hand that nothing filters on it: "
        f"this check reads {len(CYPHER_SOURCES)} files statically and can miss a "
        f"query built at runtime. If it is genuinely unused, removing it saves "
        f"about 6% of load time; if it is not, removing it turns an index lookup "
        f"into a label scan."
    )


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

    Added because the first version of these comments overclaimed -- they named
    every query that *read* the property, projections included, which is
    precisely the justification this file rejects when it removes an index.
    Asserting a comment merely exists let that through; so did checking only the
    `EAnn` half, which left two invented MCP tool names unnoticed.

    Comment grammar: `EAnn` ids and `mcp <tool>[, <tool>]`, in any order,
    separated by `;` or `,`. Trailing prose in parentheses is ignored, so
    `// EA11; mcp operator_coverage (weakest)` does not read `weakest` as a tool.
    """
    units = query_units()
    tools = mcp_tools()
    wrong = []
    for label, prop, comment in traced_indexes():
        for qid in re.findall(r"\bEA\d{2}\b", comment):
            if qid not in units:
                wrong.append(f"{label}({prop}) names {qid}, which is not in the catalog")
            elif not filtered_on(units[qid], label, prop):
                wrong.append(f"{label}({prop}) names {qid}, which does not filter on it")
        named = re.search(r"\bmcp\s+(\w+(?:\s*,\s*\w+)*)", comment)
        for tool in (re.split(r"\s*,\s*", named.group(1)) if named else []):
            if tool not in tools:
                wrong.append(
                    f"{label}({prop}) names mcp tool {tool!r}, which is not an "
                    f"@mcp.tool() function in mcp_server/server.py"
                )
            elif not filtered_on(tools[tool], label, prop):
                wrong.append(
                    f"{label}({prop}) names mcp tool {tool!r}, which does not "
                    f"filter on it"
                )
    assert not wrong, (
        "trace comments naming queries that do not filter on the property:\n  "
        + "\n  ".join(wrong)
        + "\nList only the queries with the property in a WHERE clause."
    )


@pytest.mark.parametrize("label,prop", removed_indexes())
def test_the_removed_indexes_are_still_unused(label, prop):
    """The five #18 removed, read from the schema file's own list.

    If a query starts filtering on one of these, this fails and says the index
    should come back -- the failure worth having, because the query would
    otherwise silently do a label scan.
    """
    users = [name for name, text in query_units().items()
             if filtered_on(text, label, prop)]
    assert not users, (
        f"{label}({prop}) is now filtered on by {users}, but its index was "
        f"removed in #18 as unused. Restore it in schema/edge_ai_kg.cypher and "
        f"delete its line from the `Removed as unused` comment there."
    )


# Properties filtered somewhere but carrying no index. Not a defect list -- each
# was measured and rejected. Adding all five below plus the two `name` lookups
# moved the whole 16-query catalog by **+0.9ms on 575ms**, with individual
# queries swinging -8.6ms to +3.0ms: noise. Every one sits on a small label
# (Deployment 1,513 rows, Operator 376, Board 134, ClinicalTask 18), where a
# scan is already cheap and an index is load cost for nothing.
#
# Recorded so a *new* one is a decision rather than an oversight -- which is how
# `Operator(domain)` came to be a real equality predicate in EA13 with no index
# and no mention, while the schema file's "Deliberately NOT added" note read as
# a complete list.
KNOWN_UNINDEXED = {
    ("ClinicalTask", "name"),             # mcp boards_for_task; 5.99ms -> 5.55ms
    ("ClinicalTask", "latency_budget_ms"),
    ("Model", "name"),                    # mcp fallback_audit
    ("Operator", "domain"),               # EA13, `AND op.domain = "ai.onnx"`
    ("Deployment", "fits"),               # EA06, EA07 and the demo
    ("Deployment", "latency_ms"),
    ("Board", "battery_powered"),         # demo/demo.py
    # EA11 (#69). Measured on its own: adding this index moved EA11 from
    # 105.6ms to 115.5ms and the catalog from 541ms to 615ms -- slower, because
    # Accelerator has 91 rows and the index is overhead a scan does not need.
    ("Accelerator", "is_cpu_fallback"),
}


def test_no_new_property_is_filtered_without_an_index_or_a_decision():
    """The other direction: predicate -> index, which nothing checked.

    Every other test here asks "is this index used?". None asked "is this
    filter indexed?", so `Operator(domain)` -- a real equality predicate in
    EA13, on a 205-row label -- sat unindexed and unmentioned while the schema
    file's "Deliberately NOT added" note read as a complete list.
    """
    indexed = set(declared_indexes())
    labels = set(NODE_LABELS)
    filtered = set()
    for text in query_units().values():
        for m in re.finditer(rf"\((\w+)\s*:\s*({'|'.join(labels)})\b", text):
            alias, label = m.group(1), m.group(2)
            for clause in WHERE_SPAN.findall(text):
                for p in re.findall(rf"\b{re.escape(alias)}\.(\w+)\b", clause):
                    filtered.add((label, p))
    undecided = filtered - indexed - KNOWN_UNINDEXED
    assert not undecided, (
        f"properties filtered on with no index and no recorded decision: "
        f"{sorted(undecided)}. Either add the index -- measure first, the "
        f"non-id indexes cost ~6% of load -- or add it to KNOWN_UNINDEXED with "
        f"the measurement that says it is not worth one."
    )
