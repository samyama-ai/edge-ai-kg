"""The node labels are written down in three places; this pins them together.

`schema/edge_ai_kg.cypher` declares them in its header comment, `etl/loader.py`
lists them in `NODE_LABELS`, and `docs/schema.md` gives each one a row. Nothing
connected the three, which is how `BenchmarkTask` came to be declared and loaded
but absent from the document that describes the schema (#15).

These parse the prose rather than the graph, so they need no engine and no
`etl.download_data` -- a label that is added to one list and forgotten in another
fails here, naming the lists that disagree.
"""
import re
from pathlib import Path

from etl.loader import NODE_LABELS

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema" / "edge_ai_kg.cypher"
SCHEMA_DOC = ROOT / "docs" / "schema.md"


def declared_in_schema(heading: str) -> list[str]:
    """A `// <heading>: A, B, C` block, which wraps over several lines.

    Returns a list, not a set, so callers can also see duplicates.
    """
    text = SCHEMA.read_text(encoding="utf-8")
    match = re.search(rf"^// {heading}:(.*?)^//\s*$", text, re.DOTALL | re.MULTILINE)
    assert match, f"no `// {heading}:` block in schema/edge_ai_kg.cypher"
    body = re.sub(r"^//", "", match.group(1), flags=re.MULTILINE)
    return [name.strip() for name in body.split(",") if name.strip()]


def duplicates(names) -> list[str]:
    return sorted({name for name in names if list(names).count(name) > 1})


def first_column_of(section: str) -> set[str]:
    """The first column of one table, scoped to its own `## ` section.

    Scoping matters: the node table, the edge table and the accelerator
    archetype table all share a row shape, and matching the whole file finds
    `CERTIFIED_FOR` and `DSP` among the node labels.
    """
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    match = re.search(rf"^## {section}$(.*?)^## ", text, re.DOTALL | re.MULTILINE)
    assert match, f"no `## {section}` section in docs/schema.md"
    return set(re.findall(r"^\|\s*`(\w+)`\s*\|", match.group(1), re.MULTILINE))


def labels_declared_in_schema() -> set[str]:
    return set(declared_in_schema("Node labels"))


def labels_in_the_node_table() -> set[str]:
    return first_column_of("Node labels")


def edges_declared_in_schema() -> set[str]:
    return set(declared_in_schema("Edge types"))


def edges_in_the_edge_table() -> set[str]:
    return first_column_of("Edge types")


def test_schema_header_and_loader_agree():
    assert labels_declared_in_schema() == set(NODE_LABELS)


def test_node_labels_has_no_duplicate_entries():
    """A set comparison alone would pass a duplicated entry.

    `etl/loader.py` iterates this list, and the engine parses no uniqueness
    constraint, so a duplicate would load a label's rows twice without anything
    rejecting the repeated ids.
    """
    assert not duplicates(NODE_LABELS), \
        f"duplicated in NODE_LABELS: {duplicates(NODE_LABELS)}"


def test_the_schema_header_lists_nothing_twice():
    """The set comparisons above would hide a name repeated in either block."""
    for heading in ("Node labels", "Edge types"):
        listed = declared_in_schema(heading)
        assert not duplicates(listed), (
            f"`// {heading}:` in schema/edge_ai_kg.cypher lists "
            f"{duplicates(listed)} more than once"
        )


def test_every_declared_label_has_a_row_in_the_schema_doc():
    declared = labels_declared_in_schema()
    documented = labels_in_the_node_table()
    assert declared <= documented, (
        "declared in schema/edge_ai_kg.cypher but missing a row in "
        f"docs/schema.md: {sorted(declared - documented)}"
    )


def test_the_schema_doc_documents_no_label_that_does_not_exist():
    declared = labels_declared_in_schema()
    documented = labels_in_the_node_table()
    assert documented <= declared, (
        "has a row in docs/schema.md but is not declared in "
        f"schema/edge_ai_kg.cypher: {sorted(documented - declared)}"
    )


def test_every_declared_edge_type_has_a_row_in_the_schema_doc():
    declared = edges_declared_in_schema()
    documented = edges_in_the_edge_table()
    assert declared <= documented, (
        "declared in schema/edge_ai_kg.cypher but missing a row in "
        f"docs/schema.md: {sorted(declared - documented)}"
    )


def test_the_schema_doc_documents_no_edge_type_that_does_not_exist():
    declared = edges_declared_in_schema()
    documented = edges_in_the_edge_table()
    assert documented <= declared, (
        "has a row in docs/schema.md but is not declared in "
        f"schema/edge_ai_kg.cypher: {sorted(documented - declared)}"
    )


def stated_counts() -> tuple[int, int]:
    """The `N node labels, M edge types` sentence docs/schema.md opens with.

    Matched as one whole sentence at the start of a line rather than as two
    loose `(\\d+) node labels` searches: a later paragraph mentioning either
    phrase would otherwise capture the count and let the opening line drift
    without failing anything.
    """
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    match = re.search(r"^(\d+) node labels, (\d+) edge types\.", text, re.MULTILINE)
    assert match, (
        "docs/schema.md no longer opens with an `N node labels, M edge types.` "
        "sentence; update this test alongside the rewording"
    )
    return int(match.group(1)), int(match.group(2))


def test_the_stated_label_count_matches_the_table():
    assert stated_counts()[0] == len(labels_in_the_node_table())


def test_the_stated_edge_type_count_matches_the_table():
    assert stated_counts()[1] == len(edges_in_the_edge_table())


# --- the edge table's Count column, which #16 read as 160 short -------------


EDGE_ROW = re.compile(r"^\|\s*`(\w+)`\s*\|", re.MULTILINE)
COUNT_CELL = re.compile(r"^([\d,]+)(?:\s*\(\+([\d,]+) real\))?$")


def edge_table_lines() -> list[str]:
    """The data rows of the `## Edge types` table, header and rule excluded."""
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    match = re.search(r"^## Edge types$(.*?)^## ", text, re.DOTALL | re.MULTILINE)
    assert match, "no `## Edge types` section in docs/schema.md"
    return [line for line in match.group(1).splitlines() if EDGE_ROW.match(line)]


def edge_counts() -> dict[str, tuple[int, int]]:
    """`{edge type: (generated, real)}` from the table's Count column.

    Cells are split on unescaped pipes only. The distinction is the whole of
    #16: `MADE_BY` used to write its two source labels `Board\\|SoC`, and a
    split on every `|` yields an extra column -- `160` read as the endpoints,
    `supply chain` read as the count -- so `MADE_BY` scored zero and the column
    came out 73,665 against a stated 73,825. Exactly its 160.
    """
    out = {}
    for line in edge_table_lines():
        cells = [c.replace(r"\|", "|").strip()
                 for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        assert len(cells) == 4, (
            f"edge table row has {len(cells)} cells, expected 4: {line!r}"
        )
        match = COUNT_CELL.match(cells[2])
        assert match, (
            f"unparsed Count cell for {cells[0]}: {cells[2]!r}. Expected `N` or "
            f"`N (+M real)`, the convention docs/schema.md states."
        )
        out[cells[0].strip("`")] = (
            int(match.group(1).replace(",", "")),
            int((match.group(2) or "0").replace(",", "")),
        )
    return out


def test_no_edge_table_cell_contains_an_escaped_pipe():
    """The cause of #16, banned rather than merely fixed.

    An escaped pipe is valid Markdown and renders correctly, which is why this
    survived review: the table *looked* right and only broke for anything
    splitting the row mechanically. Prose like `Board or SoC` costs nothing and
    keeps the column addable.
    """
    offenders = [line.strip() for line in edge_table_lines() if r"\|" in line]
    assert not offenders, (
        "edge table rows containing an escaped pipe, which makes the Count "
        f"column unparseable by anything splitting on `|`: {offenders}. "
        "Write the alternation in prose instead -- `Board or SoC -> Vendor`."
    )


def test_the_edge_counts_sum_to_the_stated_total():
    """#16's actual question. The table has always added up; it was unreadable.

    Only the generated column is pinned. The `(+M real)` figures move whenever
    ONNX Runtime or MLPerf publish -- 734 kernel registrations became 738 in one
    week of this backlog -- and a test that fails for upstream's reasons is one
    people learn to ignore. `tests/test_real_layer_shape.py` makes the same
    choice for the same reason.
    """
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    stated = re.search(r"\*\*([\d,]+) nodes, ([\d,]+) edges\*\*", text)
    assert stated, (
        "docs/schema.md no longer opens with a `**N nodes, M edges**` total; "
        "update this test alongside the rewording"
    )
    expected = int(stated.group(2).replace(",", ""))
    counts = edge_counts()
    total = sum(generated for generated, _ in counts.values())
    assert total == expected, (
        f"the edge table's generated counts sum to {total:,}, but the page "
        f"states {expected:,} -- a difference of {expected - total:,}. Rows: "
        f"{ {k: v[0] for k, v in sorted(counts.items())} }"
    )


def test_every_edge_type_in_the_table_has_a_count():
    """A row with no number is a row that cannot be checked."""
    missing = [name for name, (generated, real) in edge_counts().items()
               if generated == 0 and real == 0]
    assert not missing, (
        f"edge types whose Count cell is zero in both layers: {missing}. Either "
        f"the type is not built and should say so, or the count is wrong."
    )


# --- the node table's Count column, the analogue of the edge check ----------


def node_counts() -> dict[str, tuple[int, int]]:
    """`{label: (generated, real)}` from the node table's Count column.

    Same `N (+M real)` grammar as the edge table, parsed the same way, so the
    two halves of the page cannot drift into different conventions.
    """
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    match = re.search(r"^## Node labels$(.*?)^## ", text, re.DOTALL | re.MULTILINE)
    assert match, "no `## Node labels` section in docs/schema.md"
    out = {}
    for line in match.group(1).splitlines():
        if not EDGE_ROW.match(line):
            continue
        cells = [c.replace(r"\|", "|").strip()
                 for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        assert len(cells) == 3, (
            f"node table row has {len(cells)} cells, expected 3: {line!r}"
        )
        cell = COUNT_CELL.match(cells[1])
        assert cell, (
            f"unparsed Count cell for {cells[0]}: {cells[1]!r}. Expected `N` or "
            f"`N (+M real)`, the convention docs/schema.md states."
        )
        out[cells[0].strip("`")] = (
            int(cell.group(1).replace(",", "")),
            int((cell.group(2) or "0").replace(",", "")),
        )
    return out


def test_the_node_counts_sum_to_the_stated_generated_total():
    """#14: the page's bare counts are the generated layer, and say so.

    They summed to 24,115 while the README quoted 25,145, and nothing on either
    page said the two were counting to different edges. Only the generated
    column is pinned -- the `(+M real)` figures move when ONNX Runtime or MLPerf
    publish, exactly as for the edge table.
    """
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    stated = re.search(r"generated\s+layer is \*\*([\d,]+) nodes, ([\d,]+) edges\*\*",
                       text)
    assert stated, (
        "docs/schema.md no longer states its generated-layer total in the form "
        "`generated layer is **N nodes, M edges**`; update this test alongside "
        "the rewording"
    )
    expected = int(stated.group(1).replace(",", ""))
    counts = node_counts()
    total = sum(generated for generated, _ in counts.values())
    assert total == expected, (
        f"the node table's generated counts sum to {total:,}, but the page "
        f"states {expected:,} -- a difference of {expected - total:,}. Rows: "
        f"{ {k: v[0] for k, v in sorted(counts.items())} }"
    )


def test_every_node_label_row_declares_its_layer():
    """A bare count means "generated only", so it must really be generated only.

    Catches the row that gains real nodes upstream and keeps a bare count --
    which is how `Accelerator` read 85 while the loader created 91.
    """
    bare = {label for label, (_gen, real) in node_counts().items() if real == 0}
    generated_only = {
        "ModelVariant", "Sensor", "SignalStage", "ClinicalTask",
        "Dataset", "Certification", "Site",
    }
    # Equality, not a one-way subset. The subset form caught a label that gained
    # real nodes and kept a bare count, but not the reverse -- a label listed
    # here that quietly gained a `(+M real)` cell -- and it let the prose claim
    # "eight labels" while the table had six, with nothing to contradict it on a
    # page whose whole subject is counts that add up.
    assert bare == generated_only, (
        f"the set of generated-only labels changed.\n"
        f"  bare count but absent from `generated_only`: "
        f"{sorted(bare - generated_only)}\n"
        f"  listed in `generated_only` but now carrying `(+M real)`: "
        f"{sorted(generated_only - bare)}\n"
        f"A bare count asserts 'the real layer never adds to this'. Either write "
        f"`N (+M real)` in docs/schema.md, or amend `generated_only` here -- and "
        f"check the prose above the table, which states how many there are."
    )
