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
    # The imports stay outside the guard: wrapping them would turn a genuinely
    # missing or broken module into a skip, which is the same "passes while
    # testing nothing" failure this file is about.
    #
    # The guard covers `build_real` as well as `oc.load_cached()`, because
    # `build_real` reads two more caches of its own (`ort.load_cached()` and
    # `tiny.load_cached()`, etl/real_layer.py). Guarding only the first meant a
    # partially-built `data/` errored instead of skipping, contradicting the
    # docstring above.
    try:
        ops = oc.load_cached()
        fleet = gen.generate(seed=20260814, scale=1.0, operators=ops)
        generated = (fleet.node_count, fleet.edge_count)
        real_layer.build_real(fleet, ops)
        real_only = gen.Fleet(seed=0, scale=1.0)
        real_layer.build_real(real_only, ops)
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    both = (fleet.node_count, fleet.edge_count)
    return {
        "generated": generated,
        "both": both,
        "real": (both[0] - generated[0], both[1] - generated[1]),
        "real_only": (real_only.node_count, real_only.edge_count),
    }


# `25,150 nodes` -- the number then the unit, as prose.
PROSE_FIGURE = re.compile(r"([\d]{1,3}(?:,[\d]{3})+)\s*(nodes|edges)")

# `| nodes | 1,240 |` -- a two-cell table row, unit first. Added because the
# real-layer counts in README.md and DATASET_CARD.md are published this way, and
# those are the *upstream-derived* figures this file's docstring says will drift.
# Missing them meant the test passed while leaving the highest-risk published
# numbers unguarded.
TABLE_FIGURE = re.compile(r"^\|\s*(nodes|edges)\s*\|\s*([\d]{1,3}(?:,[\d]{3})+)\s*\|",
                          re.MULTILINE)


def figures_in(relative_path: str) -> set[tuple[int, str]]:
    """`(number, unit)` pairs a document publishes, e.g. `(25150, "nodes")`.

    Two forms are recognised: prose (`25,150 nodes`) and the two-cell table row
    (`| nodes | 1,240 |`).

    Two blind spots, both deliberate and both stated so they are choices rather
    than oversights:

    1. **A published count below 1,000 is invisible.** The thousands separator
       is what distinguishes a graph total from prose like "16 node labels".
       No node or edge total is that small today.
    2. **A number with no adjacent unit is invisible** -- for example
       `docs/schema.md`'s "those add 2,478 more", where the unit is three lines
       earlier. Matching bare numbers would mean matching every figure in every
       document, so these are left to
       `tests/test_schema_docs.py`, which parses that page's tables structurally
       and would fail if 2,478 stopped being the sum of its `(+M real)` column.
    """
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    found = {(int(m.group(1).replace(",", "")), m.group(2))
             for m in PROSE_FIGURE.finditer(text)}
    found |= {(int(m.group(2).replace(",", "")), m.group(1))
              for m in TABLE_FIGURE.finditer(text)}
    return found


def unexplained(relative_path: str,
                known: set[tuple[int, str]]) -> set[tuple[int, str]]:
    return {pair for pair in figures_in(relative_path) if pair not in known}


# Documents that legitimately publish no generated-layer figure. An explicit
# list, not an inferred one: the previous version skipped any document with no
# node count, which silently disabled the whole check for a page that stopped
# publishing one -- the failure this file exists to prevent.
NO_GENERATED_FIGURES: set[str] = set()


# Figures that describe a *different artifact* than the current build, so no
# layer of the graph holds them and that is correct.
#
# Deliberately narrow, and each entry carries its reason. The temptation on a
# failure here is to add the number and move on, which would turn this file into
# a list of numbers someone once saw. An entry is only defensible when the
# document says, in the same breath, what the figure is a count *of*.
# Each entry is `(doc, number, unit) -> marker`, and the marker must appear on
# the *same line* as the figure.
#
# The line requirement is not decoration. 25,145 / 76,291 are the exact numbers
# this file was written to catch: `test_published_counts`'s own docstring
# records them as the #17 drift, quoted as the build's counts when the build
# held 25,150 / 76,303. A file-wide exemption would re-legitimise that bug
# verbatim -- a document could revert to publishing 25,145 as its headline and
# this guard would wave it through, because the number also appears once in a
# snapshot sentence. Tying the exemption to the sentence keeps it an exemption
# for *that claim* rather than for the number.
PUBLISHED_ELSEWHERE: dict[tuple[str, int, str], str] = {
    ("README.md", 25_145, "nodes"): "snapshot",
    ("README.md", 76_291, "edges"): "snapshot",
    ("DATASET_CARD.md", 25_145, "nodes"): "snapshot",
    ("DATASET_CARD.md", 76_291, "edges"): "snapshot",
    # Counts of a graph that was *measured*, before `Site` and its 1,440
    # `DEPLOYED_AT` edges landed with #34. Re-stating them at today's totals
    # would claim a run nobody made -- the duplicate-import incident really did
    # double 76,303 edges, and the 12-query and 16-query benchmark runs really
    # were taken on the graphs named beside them. Each line says "recorded",
    # which is what the marker binds the exemption to.
    ("README.md", 76_303, "edges"): "recorded",
    ("benchmarks/README.md", 24_115, "nodes"): "recorded",
    ("benchmarks/README.md", 73_825, "edges"): "recorded",
    ("benchmarks/README.md", 25_150, "nodes"): "recorded",
    ("benchmarks/README.md", 76_303, "edges"): "recorded",
}


def exempt_lines(relative_path: str, number: int, unit: str, marker: str) -> list[str]:
    """Lines publishing `number unit` that also carry `marker`."""
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    formatted = f"{number:,}"
    return [line for line in text.splitlines()
            if formatted in line and unit in line and marker.lower() in line.lower()]


def is_exempt(relative_path: str, number: int, unit: str) -> bool:
    marker = PUBLISHED_ELSEWHERE.get((relative_path, number, unit))
    return bool(marker) and bool(exempt_lines(relative_path, number, unit, marker))


def test_every_exempted_figure_is_still_published():
    """`PUBLISHED_ELSEWHERE` cannot outlive the text it excuses.

    Without this the list only ever grows: a figure gets corrected in the
    document, its exemption stays, and the next figure that happens to collide
    with that number is waved through. An exemption is a claim about a specific
    sentence, so it fails when that sentence goes.
    """
    stale = [f"{doc}: {number:,} {unit} (expected a line mentioning {marker!r})"
             for (doc, number, unit), marker in PUBLISHED_ELSEWHERE.items()
             if not exempt_lines(doc, number, unit, marker)]
    assert not stale, (
        "PUBLISHED_ELSEWHERE excuses figures no longer published on a line that "
        "explains them:\n  " + "\n  ".join(stale)
        + "\n\nRemove the entries. An exemption that outlives its sentence "
          "silently excuses the next figure that happens to match."
    )


def test_an_exempted_figure_is_not_excused_elsewhere_in_the_same_document(counts):
    """A snapshot figure must not double as the document's headline count.

    25,145 / 76,291 are the #17 drift exactly -- quoted as the build's counts
    when the build held 25,150 / 76,303. The exemption exists for one sentence
    about the published `.sgsnap`; this asserts it has not become licence to
    publish those numbers as the graph's own.
    """
    both_nodes, both_edges = counts["both"]
    problems = []
    for (doc, number, unit), marker in PUBLISHED_ELSEWHERE.items():
        text = (ROOT / doc).read_text(encoding="utf-8")
        formatted = f"{number:,}"
        carrying = [line for line in text.splitlines()
                    if formatted in line and unit in line]
        unexplained_lines = [line.strip()[:90] for line in carrying
                             if marker.lower() not in line.lower()]
        if unexplained_lines:
            problems.append(f"{doc}: {formatted} {unit} also appears without "
                            f"{marker!r} on: {unexplained_lines}")
        # And the document must still publish the real figure, so a revert
        # cannot satisfy the exemption by deleting the true one.
        current = both_nodes if unit == "nodes" else both_edges
        if (current, unit) not in figures_in(doc):
            problems.append(f"{doc}: no longer publishes the build's own "
                            f"{current:,} {unit}, so the exempted {formatted} "
                            f"is the only figure of its kind on the page")
    assert not problems, "\n  ".join(["", *problems])


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

    `known` pools all four layers, so a document publishing the real-only count
    where the both-layer count belongs still passes here. That is a deliberate
    tradeoff rather than an omission --
    `test_the_docs_agree_with_each_other_on_the_headline` covers the case that
    matters, and pinning every figure to its own layer would make this test
    fail whenever a page legitimately quotes more than one.
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
            if is_exempt(doc, number, unit):
                continue
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


# Every shape the docs use to state how big the catalog is. Found by grepping
# for the phrasings actually in the tree, not invented: a new spelling is a new
# entry here, and `test_at_least_one_document_states_the_catalog_size` fails if
# every one of them disappears.
CATALOG_SIZE_PATTERNS = (
    r"(\d+)-query catalog",
    r"catalog is (\d+) Cypher queries",
    r"`benchmarks/queries\.py` holds (\d+) entries",
    r"Walks all (\d+) queries",
)
CATALOG_SIZE_DOCS = ("README.md", "CLAUDE.md", "DATASET_CARD.md",
                     "demo/README.md", "docs/why-this-engine.md")


def catalog_size_claims() -> list[tuple[str, int, str]]:
    """Every published claim about the catalog's size, as (doc, number, phrase)."""
    found = []
    for doc in CATALOG_SIZE_DOCS:
        text = (ROOT / doc).read_text(encoding="utf-8")
        for pattern in CATALOG_SIZE_PATTERNS:
            found.extend((doc, int(m.group(1)), m.group(0))
                         for m in re.finditer(pattern, text))
    return found


def test_every_document_states_the_catalog_size_correctly():
    """One catalog, five documents, one number.

    `EA20` landed in three places that quote this figure and was updated in
    one of them, leaving `docs/why-this-engine.md` saying 16 in two spots and
    `DATASET_CARD.md` in a third. Nothing failed, because nothing compared a
    published count to `BY_ID` -- the sibling checks above do exactly that for
    node and edge counts, and this closes the same gap for the catalog.

    The number is derived, never restated here: adding a query and forgetting a
    document fails with the document named.
    """
    from benchmarks.queries import BY_ID

    actual = len(BY_ID)
    wrong = [(doc, phrase, stated) for doc, stated, phrase in catalog_size_claims()
             if stated != actual]
    assert not wrong, (
        f"the catalog holds {actual} queries; these documents say otherwise: "
        + "; ".join(f"{doc} — {phrase!r}" for doc, phrase, _ in wrong)
        + ". A published count that is silently wrong is worse than one that "
          "fails loudly."
    )


def test_at_least_one_document_states_the_catalog_size():
    """The check above passes vacuously if every phrasing is reworded away."""
    claims = catalog_size_claims()
    assert len(claims) >= 4, (
        f"only {len(claims)} catalog-size claims found in {list(CATALOG_SIZE_DOCS)}. "
        f"Either the docs stopped publishing the figure -- unlikely -- or the "
        f"phrasing changed and CATALOG_SIZE_PATTERNS needs the new spelling, "
        f"or this test is checking nothing."
    )


# The schema's two totals, and every published phrasing that states one. Same
# treatment as the catalog size above and for the same reason: `Site` and
# `DEPLOYED_AT` moved both totals, and the count went stale in four places in
# `README.md` alone -- the loader summary, two rows of the real-layer table,
# and the inventory paragraph -- because nothing compared any of them to the
# schema. Every pattern here was found by grepping the tree, not invented.
#
# Deliberately narrow. `DATASET_CARD.md` also says "3 edge types carry 87% of
# edges", which is a share and not a total; a looser `(\d+) edge types` would
# match it and demand it equal 23.
# Each pattern captures exactly one group: the **total**. An earlier version
# wrote `(\d+) of the (?:\d+) edge types`, which captured the subset -- on
# "names 4 of the 23 edge types" it took 4 and would have demanded the schema
# hold four edge types. One group per pattern, and the group is the total.
LABEL_TOTAL_PATTERNS = (
    r"(\d+) node labels",
    r"\| labels with nodes \| \d+ of (\d+) \|",
    r"shows \d+ of the (\d+)\s+node labels",
)
EDGE_TYPE_TOTAL_PATTERNS = (
    r"· (\d+) edge types",
    r"labels, (\d+) edge types",
    r"names \d+ of the (\d+) edge types",
    r"\| edge types present \| \d+ of (\d+) \|",
    # Anchored to the loader's own sentence. Bare `across (\d+) types` matched
    # any prose that happened to say "across N types" and demanded it equal the
    # schema's total.
    r"intended edges across (\d+) types",
)
SCHEMA_TOTAL_DOCS = ("README.md", "DATASET_CARD.md", "docs/schema.md")


def schema_total_claims(patterns) -> list[tuple[str, int, str]]:
    """Every published total, with the doc and the phrase that stated it.

    Whitespace is flattened first: these phrases wrap across lines in prose
    -- `README.md`'s diagram caption says "12 of the 17\nnode labels" -- and a
    pattern matched against the raw text simply misses them, which is the
    silent half of this whole class of drift.
    """
    found = []
    for doc in SCHEMA_TOTAL_DOCS:
        text = re.sub(r"\s+", " ", (ROOT / doc).read_text(encoding="utf-8"))
        for pattern in patterns:
            found.extend((doc, int(m.group(1)), m.group(0))
                         for m in re.finditer(pattern, text))
    return found


def test_every_document_states_the_schema_totals_correctly():
    """One schema, three documents, two numbers -- derived, never restated.

    `etl.loader.NODE_LABELS` and `schema/edge_ai_kg.cypher` are the sources;
    a spine that adds a label and forgets a document fails here with the
    document and the phrase named, rather than in review three rounds later.
    """
    from etl.loader import NODE_LABELS
    from tests.test_schema_docs import declared_in_schema

    labels, edges = len(NODE_LABELS), len(set(declared_in_schema("Edge types")))
    wrong = [(doc, phrase, stated, labels)
             for doc, stated, phrase in schema_total_claims(LABEL_TOTAL_PATTERNS)
             if stated != labels]
    wrong += [(doc, phrase, stated, edges)
              for doc, stated, phrase in schema_total_claims(EDGE_TYPE_TOTAL_PATTERNS)
              if stated != edges]
    assert not wrong, (
        "published schema totals that disagree with the schema "
        f"({labels} labels, {edges} edge types): "
        + "; ".join(f"{doc} — {phrase!r} (should be {want})"
                    for doc, phrase, _got, want in wrong))


def test_every_schema_total_pattern_still_matches_something():
    """Per pattern, not per family -- a floor on the total hides a dead one.

    `len(edges) >= 3` stayed satisfied while any one phrasing was reworded
    away, so a pattern could stop matching and the document it guarded would
    drift unchecked behind the other four. Each is asserted on its own.
    """
    for label, patterns in (("label", LABEL_TOTAL_PATTERNS),
                            ("edge-type", EDGE_TYPE_TOTAL_PATTERNS)):
        for pattern in patterns:
            assert schema_total_claims((pattern,)), (
                f"the {label} pattern {pattern!r} matches nothing in "
                f"{list(SCHEMA_TOTAL_DOCS)}. Either that phrasing was reworded "
                f"-- update the pattern -- or the document stopped publishing "
                f"the total, in which case drop the pattern deliberately "
                f"rather than leaving a dead one that guards nothing.")
