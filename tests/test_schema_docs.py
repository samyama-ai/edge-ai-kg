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
