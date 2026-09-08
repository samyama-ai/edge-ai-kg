"""Every node/edge count quoted in the docs, checked against a loaded graph.

`README.md`, `DATASET_CARD.md`, `docs/schema.md`, `benchmarks/README.md` and
`docs/data-provenance.md` all quote figures. Nothing checked any of them against
a graph (#17), and the other count tests in this suite check the documents
against *themselves* -- `tests/test_schema_docs.py` sums a table and compares it
to the total stated on the same page. Internally consistent and externally
wrong is exactly the state that produced #14 and #16.

When this was written, the drift was already there:

| | quoted | actual | drift |
|---|---:|---:|---:|
| generated nodes | 24,115 | 24,115 | 0 |
| generated edges | 73,825 | 73,825 | 0 |
| both-layer nodes | 25,145 | 25,150 | **+5** |
| both-layer edges | 76,291 | 76,303 | **+12** |
| real-layer nodes | 1,030 | 1,035 | **+5** |
| real-layer edges | 2,466 | 2,478 | **+12** |

**The generated layer matched exactly and every upstream-derived figure had
moved.** That split is the whole design of this file: the generated layer is
ours and deterministic from the seed, so a mismatch there is a bug. The real
layer is ONNX Runtime's and MLPerf's, so a mismatch there means upstream
published and the documents need re-measuring -- a different diagnosis, and the
failure message says which.

This is deliberately *not* the "don't pin upstream counts" rule that
`tests/test_real_layer_shape.py` follows. That rule is about not asserting a
figure a *test* invented. Here the figures are published claims in documents a
reader will quote, and a published number that is silently wrong is worse than
one that fails loudly.

Needs `data/` and an engine, so it skips without them.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Where each figure is published, and which layer it describes.
GENERATED_DOCS = ("docs/schema.md", "benchmarks/README.md")
BOTH_LAYER_DOCS = ("README.md", "DATASET_CARD.md", "benchmarks/README.md",
                   "docs/data-provenance.md")


@pytest.fixture(scope="module")
def counts():
    """Node and edge counts for each layer, from the Fleet.

    The `Fleet` rather than a loaded graph: `tests/test_correctness.py`
    already asserts the graph holds exactly what the Fleet says
    (`test_graph_loaded_completely`), so going through the engine here would
    add a 25-second load to re-check something already pinned.
    """
    from etl import generate as gen
    from etl import onnx_catalog as oc
    from etl import real_layer
    # Only the cache read is guarded. Wrapping the imports too would turn a
    # genuinely missing or broken module into a skip, which is the same
    # "passes while testing nothing" failure this file is about.
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")

    fleet = gen.generate(seed=20260814, scale=1.0, operators=ops)
    generated = (fleet.node_count, fleet.edge_count)
    real_layer.build_real(fleet, ops)
    both = (fleet.node_count, fleet.edge_count)

    real_only = gen.Fleet(seed=0, scale=1.0)
    real_layer.build_real(real_only, ops)
    return {
        "generated": generated,
        "both": both,
        "real": (both[0] - generated[0], both[1] - generated[1]),
        "real_only": (real_only.node_count, real_only.edge_count),
    }


def figures_in(relative_path: str) -> set[tuple[int, str]]:
    """`(number, unit)` pairs a document publishes, e.g. `(25150, "nodes")`.

    Only counts written as `N nodes` / `N edges` with a thousands separator, so
    a stray four-digit number in prose is not read as a claim about the graph.

    The consequence, which the reason above does not cover: **a published count
    below 1,000 is invisible to every test in this file.** No node or edge total
    is that small today, and widening the pattern would start matching prose
    like "16 node labels", so this is a deliberate blind spot rather than an
    oversight.
    """
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    found = set()
    for match in re.finditer(r"([\d]{1,3}(?:,[\d]{3})+)\s*(nodes|edges)", text):
        found.add((int(match.group(1).replace(",", "")), match.group(2)))
    return found


def unexplained(relative_path: str,
                known: set[tuple[int, str]]) -> set[tuple[int, str]]:
    return {pair for pair in figures_in(relative_path) if pair not in known}


# Documents that legitimately publish no generated-layer figure. An explicit
# list, not an inferred one: the previous version skipped any document with no
# node count, which silently disabled the whole check for a page that stopped
# publishing one -- the failure this file exists to prevent.
NO_GENERATED_FIGURES: set[str] = set()


def test_the_generated_layer_figures_are_exact(counts):
    """Ours, deterministic from the seed. A mismatch here is a bug, not drift.

    Both units are asserted separately. An earlier version wrote
    `(nodes, "nodes") in published or (nodes, "edges") in published`, which is
    unit-blind -- a document publishing `24,115 edges` satisfied a *node*-count
    assertion -- and never checked the generated edge count at all, though the
    PR listing these six figures claimed it did.
    """
    nodes, edges = counts["generated"]
    for doc in GENERATED_DOCS:
        if doc in NO_GENERATED_FIGURES:
            continue
        published = figures_in(doc)
        for value, unit in ((nodes, "nodes"), (edges, "edges")):
            assert (value, unit) in published, (
                f"{doc} does not publish the generated {unit} count "
                f"{value:,}. Figures found: {sorted(published)}. The generated "
                f"layer is deterministic from seed 20260814, so this is a "
                f"documentation error rather than upstream drift -- if the "
                f"document deliberately omits it, add it to "
                f"NO_GENERATED_FIGURES."
            )


def test_every_published_figure_matches_some_layer_of_the_graph(counts):
    """The check #17 asks for: no published number the graph does not hold.

    Fails naming the document and both numbers. A figure that matches *no*
    layer is either stale or invented, and the message says which layers were
    considered so the fix is obvious.
    """
    # Pairs, not a flat bag of numbers. Flattening ignored the unit, so
    # `25,150 edges` in a document passed because 25,150 is a valid *node*
    # count somewhere -- and any real-layer figure passed where a both-layer
    # figure belonged.
    known = {(nodes, "nodes") for nodes, _ in counts.values()}
    known |= {(edges, "edges") for _, edges in counts.values()}
    problems = []
    for doc in sorted(set(GENERATED_DOCS) | set(BOTH_LAYER_DOCS)):
        for number, unit in sorted(unexplained(doc, known)):
            problems.append(f"{doc}: publishes {number:,} {unit}")
    assert not problems, (
        "figures published in the docs that no layer of the graph holds:\n  "
        + "\n  ".join(problems)
        + "\n\nThe counts the graph actually holds:\n"
        + f"  generated : {counts['generated'][0]:,} nodes / {counts['generated'][1]:,} edges\n"
        + f"  both      : {counts['both'][0]:,} nodes / {counts['both'][1]:,} edges\n"
        + f"  real adds : {counts['real'][0]:,} nodes / {counts['real'][1]:,} edges\n"
        + f"  real only : {counts['real_only'][0]:,} nodes / {counts['real_only'][1]:,} edges\n"
        + "\nIf the generated figures match and only the real ones moved, ONNX "
        + "Runtime or MLPerf published and the documents need re-measuring -- "
        + "that is expected periodically, not a bug."
    )


def test_the_docs_agree_with_each_other_on_the_headline(counts):
    """Four documents quote the same headline; they must not disagree.

    Cheap, and it catches the half-update -- fixing README.md and forgetting
    DATASET_CARD.md leaves two published truths, which is how the 24,115 vs
    25,145 confusion in #14 survived.
    """
    nodes, edges = counts["both"]
    disagreeing = [doc for doc in BOTH_LAYER_DOCS
                   if (nodes, "nodes") not in figures_in(doc)]
    assert not disagreeing, (
        f"documents not quoting the current both-layer node count {nodes:,}: "
        f"{disagreeing}. All four publish the headline, so they must be updated "
        f"together or they become two published truths."
    )
    disagreeing_edges = [doc for doc in BOTH_LAYER_DOCS
                         if (edges, "edges") not in figures_in(doc)]
    assert not disagreeing_edges, (
        f"documents not quoting the current both-layer edge count {edges:,}: "
        f"{disagreeing_edges}"
    )
