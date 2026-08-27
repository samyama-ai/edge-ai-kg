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


def labels_declared_in_schema() -> set[str]:
    """The `// Node labels: ...` block, which wraps over several lines."""
    text = SCHEMA.read_text(encoding="utf-8")
    match = re.search(r"^// Node labels:(.*?)^//\s*$", text, re.DOTALL | re.MULTILINE)
    assert match, "no `// Node labels:` block in schema/edge_ai_kg.cypher"
    body = re.sub(r"^//", "", match.group(1), flags=re.MULTILINE)
    return {name.strip() for name in body.split(",") if name.strip()}


def labels_in_the_node_table() -> set[str]:
    """The first column of the node-label table only.

    Scoped to the `## Node labels` section: the edge table and the accelerator
    archetype table further down the page have the same row shape, and matching
    the whole file picks up `CERTIFIED_FOR` and `DSP` as node labels.
    """
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    match = re.search(r"^## Node labels$(.*?)^## ", text, re.DOTALL | re.MULTILINE)
    assert match, "no `## Node labels` section in docs/schema.md"
    return set(re.findall(r"^\|\s*`(\w+)`\s*\|", match.group(1), re.MULTILINE))


def test_schema_header_and_loader_agree():
    assert labels_declared_in_schema() == set(NODE_LABELS)


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


def test_the_stated_label_count_matches_the_table():
    """docs/schema.md opens with "N node labels"; N is the third thing to drift."""
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    stated = int(re.search(r"^(\d+) node labels", text, re.MULTILINE).group(1))
    assert stated == len(labels_in_the_node_table())
