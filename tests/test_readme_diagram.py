"""The README's spine diagram is checked against the edges that exist.

The diagram is the first thing a reader uses to understand the graph, and it is
hand-drawn prose about a machine-readable structure with no connection to it
(#24). A picture that quietly stops matching is worse than no picture.

Measured on the diagram as it stands: it names **3 of 22** edge types and shows
**12 of 16** node labels. That is a legitimate choice -- it is an orientation
sketch, not the schema -- so this file does not demand completeness. It demands
that everything the diagram *does* say is true, and that what it omits is
recorded rather than discovered.

Parses `README.md` and `schema/edge_ai_kg.cypher`; no engine, no `data/`.
"""
import pathlib
import re

from etl.loader import NODE_LABELS
from tests.test_schema_docs import edges_declared_in_schema

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"

# What the diagram leaves out today. Not a defect list -- an orientation sketch
# is allowed to omit things -- but a recorded decision, so that adding a label
# or edge type to the schema and forgetting the diagram shows up here.
OMITTED_LABELS = {"BenchmarkTask", "Certification", "ClinicalTask", "Dataset"}


def diagram() -> str:
    """The fenced block that opens `## What is in it`."""
    text = README.read_text(encoding="utf-8")
    match = re.search(r"^```\n(Vendor.*?)^```", text, re.DOTALL | re.MULTILINE)
    assert match, (
        "no spine diagram found in README.md -- expected a fenced block "
        "starting with `Vendor`. If it was reworded, update this test with it."
    )
    return match.group(1)


# An ALL-CAPS token in the diagram is an edge-type label. Read out of the
# diagram rather than intersected with the schema: `{e for e in declared if e in
# diagram}` can never contain a name the schema lacks, so the ghost check built
# on it was vacuous -- renaming PROVIDED_BY to SUPPLIED_BY in the diagram passed.
EDGE_TOKEN = re.compile(r"\b[A-Z][A-Z_]{3,}\b")

# Caps words in the diagram that are not edge types.
NOT_EDGE_TYPES = {"MCU", "CPU", "DSP", "NPU", "GPU"}


def named_edge_types() -> set[str]:
    return set(EDGE_TOKEN.findall(diagram())) - NOT_EDGE_TYPES


def shown_labels() -> set[str]:
    return {label for label in NODE_LABELS
            if re.search(rf"\b{label}\b", diagram())}


def test_every_edge_type_the_diagram_names_exists():
    """#24's actual requirement.

    A diagram naming an edge type the schema dropped is worse than one that
    names none, because a reader will write a query against it.
    """
    declared = edges_declared_in_schema()
    named = named_edge_types()
    assert named, (
        "the diagram names no edge type at all; it named PROVIDED_BY, TARGETS "
        "and USES_OPERATOR when this test was written"
    )
    ghosts = named - declared
    assert not ghosts, (
        f"the README diagram names edge types that no longer exist: "
        f"{sorted(ghosts)}. Update the diagram, or restore them in "
        f"schema/edge_ai_kg.cypher."
    )


def test_every_node_label_the_diagram_shows_exists():
    """Same, for the boxes rather than the arrows."""
    ghosts = shown_labels() - set(NODE_LABELS)
    assert not ghosts, (
        f"the README diagram shows node labels the loader never writes: "
        f"{sorted(ghosts)}"
    )


def test_what_the_diagram_omits_is_the_recorded_set():
    """The reporting half of #24: *which* real edges the diagram leaves out.

    Fails in both directions on purpose. A new label absent from the diagram
    fails so the omission is a decision; a label quietly *added* to the diagram
    fails so this list stays true. Neither is a bug in the diagram -- both are
    the diagram and the schema drifting apart unobserved, which is the thing
    #24 is about.
    """
    omitted = set(NODE_LABELS) - shown_labels()
    assert omitted == OMITTED_LABELS, (
        f"the diagram's omissions changed: now missing "
        f"{sorted(omitted - OMITTED_LABELS)}, now shown "
        f"{sorted(OMITTED_LABELS - omitted)}.\n"
        f"The diagram is an orientation sketch and may omit things, but the "
        f"README section below it says which -- update both together."
    )


def test_the_readme_says_what_the_diagram_leaves_out():
    """The sketch is allowed to be partial only if it admits to being partial.

    Without this, the three tests above pass on a diagram that a reader
    reasonably mistakes for the whole schema.
    """
    text = README.read_text(encoding="utf-8")
    assert "docs/schema.md" in text, (
        "README.md no longer points at docs/schema.md for the full schema"
    )
    # Scoped to the 1,200 characters after the diagram, not the whole file: the
    # caveat has to be where the reader meets the picture, and "omits" appearing
    # in some unrelated paragraph is not a caveat.
    end = text.index(diagram()) + len(diagram())
    nearby = text[end:end + 1200]
    assert re.search(r"simplified|orientation|not the full|omits|leaves out",
                     nearby, re.IGNORECASE), (
        "the README does not say the diagram is partial. It shows 12 of 16 "
        "labels and names 3 of 22 edge types, so a reader has no way to know "
        "that ClinicalTask, Certification, Dataset and BenchmarkTask exist."
    )
