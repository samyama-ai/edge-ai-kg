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


def declared_in_schema(heading: str) -> set[str]:
    """A `// <heading>: A, B, C` block, which wraps over several lines."""
    text = SCHEMA.read_text(encoding="utf-8")
    match = re.search(rf"^// {heading}:(.*?)^//\s*$", text, re.DOTALL | re.MULTILINE)
    assert match, f"no `// {heading}:` block in schema/edge_ai_kg.cypher"
    body = re.sub(r"^//", "", match.group(1), flags=re.MULTILINE)
    return {name.strip() for name in body.split(",") if name.strip()}


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
    return declared_in_schema("Node labels")


def labels_in_the_node_table() -> set[str]:
    return first_column_of("Node labels")


def edges_declared_in_schema() -> set[str]:
    return declared_in_schema("Edge types")


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
    assert len(NODE_LABELS) == len(set(NODE_LABELS)), (
        f"duplicated in NODE_LABELS: "
        f"{sorted({x for x in NODE_LABELS if NODE_LABELS.count(x) > 1})}"
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


def stated_count(noun: str) -> int:
    """The `N node labels, M edge types` sentence docs/schema.md opens with."""
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    match = re.search(rf"(\d+) {noun}", text)
    assert match, f"docs/schema.md no longer states a count of {noun}"
    return int(match.group(1))


def test_the_stated_label_count_matches_the_table():
    assert stated_count("node labels") == len(labels_in_the_node_table())


def test_the_stated_edge_type_count_matches_the_table():
    assert stated_count("edge types") == len(edges_in_the_edge_table())
