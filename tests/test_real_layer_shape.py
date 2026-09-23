"""What `--layers real` actually contains, pinned against what the README claims.

The README offers `python -m etl.loader --layers real` as a first-class way to
use the repo, and #21 asked the obvious unmeasured question: is the public-source
subgraph still a graph, or three islands sharing a database?

Measured (see the README section this pins): **it is connected** -- 1,240 nodes,
2,478 edges, and only 18 orphans, all of them `Operator` nodes no ONNX Runtime
kernel registers. It is not 1,030 orphans.

What it is missing is a *half*, not the joins. The real layer carries the
hardware and kernel spine and none of the clinical one, so six labels and eleven
edge types are empty -- including `USES_OPERATOR`, which is the edge the hero
question traverses. That is why most of the catalog queries come back empty
against it.

These assertions are about **shape, not counts**. The counts move whenever ONNX
Runtime publishes new kernel registrations -- 734 became 738 during one week --
so pinning them here would produce a test that fails for upstream's reasons
rather than ours. Which labels and edge types `etl/real_layer.py` builds is our
decision, and that is what is asserted.

**The catalog partition is pinned, and read from the README rather than
restated.** Which queries return rows against the real layer is a published
claim -- a table a reader will quote -- and it went stale once already: the
README said 8 of 16 including `EA10` and `EA12` long after an upstream refresh
emptied them, and nothing failed. `test_the_readme_catalog_partition_is_true`
parses that table and runs the queries. It is *not* exempt from the
counts-move-upstream rule above; it is the deliberate exception, for the same
reason `tests/test_published_counts.py` pins published figures: a documented
number that is silently wrong is worse than one that fails loudly.
"""
import pathlib
import re

import pytest

from benchmarks.queries import BY_ID
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

# Shared rather than a second regex over the same comment block: that helper
# already asserts on its match, so a reformatted schema header fails there once
# with a readable message instead of raising AttributeError in two places.
from tests.test_schema_docs import declared_in_schema

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = "default"

# The hardware and kernel spine, plus the MLPerf submissions.
LABELS_PRESENT = {"Vendor", "SoC", "Accelerator", "Board", "Runtime",
                  "Operator", "Kernel", "Model", "BenchmarkTask", "Deployment"}
# The clinical spine, entirely generated.
LABELS_ABSENT = {"ModelVariant", "Sensor", "SignalStage", "ClinicalTask",
                 "Dataset", "Certification"}

EDGES_PRESENT = {"HAS_SOC", "IMPLEMENTS", "MADE_BY", "MEASURES", "ON_BOARD",
                 "PROVIDED_BY", "RUNS_ON", "SOLVES", "TARGETS",
                 "USES_ACCELERATOR", "VIA_RUNTIME"}


@pytest.fixture(scope="module")
def real_only():
    """The graph `--layers real` produces: no generated fleet underneath it."""
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    try:
        from samyama import SamyamaClient
        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

    # Reset first, as every other embedded fixture in this suite does. Measured
    # on `samyama` 0.6.1, two `SamyamaClient.embedded()` instances in one
    # process are independent, so nothing another module loads reaches here --
    # but that is a property of this build, not a guarantee, and
    # `assert synthetic == 0` below would be the confusing way to find out it
    # had changed.
    try:
        client.query("MATCH (n) DETACH DELETE n", GRAPH)
    except Exception:
        pass

    # Exactly what etl/loader.py does for --layers real: start from an empty
    # Fleet rather than filtering a generated one.
    from etl import generate as gen
    from etl import real_layer
    fleet = gen.Fleet(seed=0, scale=1.0)
    real_layer.build_real(fleet, ops)
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client, fleet


def scalar(client, cypher: str) -> int:
    result = client.query(cypher, GRAPH)
    return result.records[0][0] if result.records else 0


def declared_edge_types() -> set[str]:
    return set(declared_in_schema("Edge types"))


def test_the_fixture_built_a_real_only_graph(real_only):
    client, _ = real_only
    total = scalar(client, "MATCH (x) RETURN count(x) AS n")
    assert total > 1000, f"only {total} nodes; the real layer did not build"
    synthetic = scalar(
        client, 'MATCH (x) WHERE x.provenance = "synthetic" RETURN count(x) AS n')
    assert synthetic == 0, f"{synthetic} generated nodes leaked into a real-only load"


def test_which_labels_the_real_layer_carries(real_only):
    client, _ = real_only
    present = {label for label in NODE_LABELS
               if scalar(client, f"MATCH (x:{label}) RETURN count(x) AS n")}
    assert present == LABELS_PRESENT, (
        f"the real layer's label set changed: "
        f"gained {sorted(present - LABELS_PRESENT)}, "
        f"lost {sorted(LABELS_PRESENT - present)}. The README documents which "
        f"catalog queries work against `--layers real`; update it too."
    )
    assert not (present & LABELS_ABSENT), "a clinical-spine label gained real nodes"


def test_which_edge_types_the_real_layer_carries(real_only):
    client, _ = real_only
    present = {t for t in declared_edge_types()
               if scalar(client, f"MATCH (a)-[:{t}]->(b) RETURN count(a.id) AS n")}
    assert present == EDGES_PRESENT, (
        f"the real layer's edge set changed: "
        f"gained {sorted(present - EDGES_PRESENT)}, "
        f"lost {sorted(EDGES_PRESENT - present)}"
    )


def test_the_hero_question_cannot_be_asked_of_the_real_layer(real_only):
    """The documented limitation, asserted so it cannot drift out of the README.

    The hero question walks `Model -[:USES_OPERATOR]-> Operator`. The real layer
    knows which kernels implement which operators, but nothing records which
    operators a model uses -- so `EA01`, `EA02` and `EA11` return nothing.

    If someone gives the real layer a model's operator surface, this fails and
    says the README claim is now wrong, which is the point.
    """
    client, _ = real_only
    uses = scalar(client, "MATCH (a)-[:USES_OPERATOR]->(b) RETURN count(a.id) AS n")
    assert uses == 0, (
        f"{uses} USES_OPERATOR edges in the real layer; the hero question may now "
        f"be answerable there, and the README says it is not"
    )


# `([^|]+?)` rather than `(.+?)`: the second absorbs a third column whole if the
# table ever gains one, so a row silently becomes "these ids plus whatever else
# was on the line". Bounded to the cell.
README_PARTITION = re.compile(
    r"^\|\s*(Return rows|Empty)\s*\|\s*([^|]+?)\s*\|\s*$", re.MULTILINE)

# The sentence introducing that table. Pinned too, because prose and table drift
# independently: "8 of the 16" survived above a table nobody had re-measured,
# which is the staleness this module exists to catch. Matching the numbers, not
# the wording, so the sentence can be rewritten without breaking the check.
# `\s+` between every word, not `\s*\n?\s*` at the two places the line
# happened to wrap. The old form was pinned to the current wrap, so reflowing
# the paragraph -- which a formatter or an unrelated edit does -- made the
# sentence unfindable and the test failed with "could not find the sentence"
# rather than with anything about the numbers.
README_PROSE = re.compile(
    r"\*\*Against\s+the\s+HTTP\s+server,\s+(\d+)\s+of\s+the\s+(\d+)\s+catalog"
    r"\s+queries\s+return\s+rows\*\*,\s+(\d+)\s+come\s+back\s+empty")


def readme_partition() -> dict[str, set[str]]:
    """The two rows of the README's real-layer table, as sets of query ids.

    Two failure modes a dict comprehension would swallow, so both are asserted:
    a second table elsewhere in the README with the same first cells would
    silently overwrite this one, and an id listed under *both* headings would
    satisfy the coverage check below while being read as "returns rows".
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    matches = README_PARTITION.findall(text)
    labels = [label for label, _ in matches]
    assert sorted(labels) == ["Empty", "Return rows"], (
        f"expected exactly one `Return rows` row and one `Empty` row in the "
        f"README; found {labels}. A second table with the same first cells "
        f"would overwrite this one silently. If the table moved or was "
        f"reformatted, fix this parser -- do not delete the check, which is "
        f"how the 8-of-16 claim went stale."
    )
    # An empty cell yields `{""}` -- one query named "" -- and a literal `none`
    # yields `{"none"}`, which fails the coverage assert with "not catalog
    # queries: ['none']". Neither message points at the actual problem, which
    # is a row with no ids in it. Blank cells are filtered and reported here,
    # where the cause is visible; `none` is left to the coverage check, since
    # it is at least a legible complaint about a specific token.
    found = {label: {q.strip(" `") for q in cells.split(",") if q.strip(" `")}
             for label, cells in matches}
    blank = sorted(label for label, ids in found.items() if not ids)
    assert not blank, (
        f"README rows with no query ids: {blank}. An empty row is a real "
        f"claim -- 'nothing lands here' -- but it is more often a half-finished "
        f"edit, and the coverage check below cannot tell the difference. Write "
        f"the row out or remove it."
    )
    both = found["Return rows"] & found["Empty"]
    assert not both, (
        f"{sorted(both)} appear in both README rows, so the table says a query "
        f"both returns rows and is empty. The coverage check below would pass "
        f"and the id would be read as returning rows."
    )
    return found


def test_the_readme_prose_agrees_with_the_table_below_it():
    """The sentence and the table must say the same thing.

    They are two claims about one measurement, edited separately. The parser
    below reads only the table, so a sentence saying "8 of the 16" above a table
    listing six could sit there indefinitely -- which is how the figure this
    module corrected went stale in the first place.

    No `real_only` fixture. This compares text against text and never touches
    the engine, and taking the fixture made it skip whenever `data/` was absent
    or the embedded build was unavailable -- so on any machine without the
    downloads, the check on the sentence quietly did not run. The tests below
    that *do* query the graph keep the fixture.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = README_PROSE.search(text)
    assert match, (
        "could not find the sentence introducing the real-layer table. If it "
        "was reworded, update README_PROSE -- do not drop the check; an "
        "unpinned sentence above a pinned table is where the stale number lived."
    )
    returns, total, empty = (int(g) for g in match.groups())
    published = readme_partition()
    assert total == len(BY_ID), (
        f"the sentence says {total} catalog queries; there are {len(BY_ID)}")
    assert returns == len(published["Return rows"]), (
        f"the sentence says {returns} return rows; the table lists "
        f"{len(published['Return rows'])}")
    assert empty == len(published["Empty"]), (
        f"the sentence says {empty} empty; the table lists "
        f"{len(published['Empty'])}")


def test_the_readme_catalog_partition_is_true(real_only):
    """Every query id the README places in a row must actually land there.

    Parsed from the README rather than restated here, so the two cannot drift:
    a claim edited in the document is a claim this test starts checking.

    **One coupling to know about.** The table is headed "Against the HTTP
    server"; this fixture is **embedded**. That is sound only while the two
    builds partition the catalog the same way, which was measured on 2026-09-09
    against server 1.7.0: identical rows-vs-empty on the real layer.

    It is a narrower guarantee than it sounds. On the *full* graph the same two
    builds return the same number of *different* rows for seven queries -- ties
    under `ORDER BY ... LIMIT` -- and `EA17` raises on the server entirely
    (engine note 12). None of that changes rows-vs-empty on the real layer,
    which is all this test reads. If they ever diverge on that, the failure will
    look like a stale README rather than a build difference. Re-measure both
    before believing either.

    **`EA17` and note 12 do not contradict each other**, though they read as if
    they do: the README files `EA17` under Empty and says none of the table's
    queries error, while note 12 says the server refuses it. Both are true of
    the real layer, because it has no `Sensor` -- the opening `MATCH` binds
    nothing, `size(r)` is never evaluated, and the variable-length traversal the
    server refuses is never reached. The README says this in full under its
    table; it is repeated here because this test is where the two claims meet.
    """
    client, _fleet = real_only
    published = readme_partition()
    listed = published["Return rows"] | published["Empty"]
    assert listed == set(BY_ID), (
        f"the README's table does not cover the catalog. Missing from the "
        f"table: {sorted(set(BY_ID) - listed)}; not catalog queries: "
        f"{sorted(listed - set(BY_ID))}"
    )

    wrong = []
    for qid, spec in BY_ID.items():
        expected_rows = qid in published["Return rows"]
        # A raise is a finding about the table, not a crash. Uncaught, the first
        # query the build refuses aborted the loop with a traceback -- so the
        # remaining queries went unchecked and the reader got a stack trace
        # where the point is a two-column diff. `EA17` is the live example:
        # engine note 12 says the 1.7.0 server refuses it.
        try:
            got = client.query(spec["cypher"], GRAPH).records
        except Exception as exc:  # a refusal is a result here, not a crash
            reason = (str(exc).splitlines() or [""])[0][:60]
            # "Refuses this query" and "has stopped answering" are different
            # findings, and only the first says anything about the README. The
            # same trap as engine_notes_probe's notes 5 and 6: without this,
            # a dropped connection is reported as a build rejecting Cypher.
            try:
                client.query("RETURN 1", GRAPH)
            except Exception:
                pytest.fail(
                    f"the engine stopped answering during {qid} "
                    f"({type(exc).__name__}: {reason}); `RETURN 1` fails too, "
                    f"so this run says nothing about the README table")
            wrong.append(f"{qid}: README says "
                         f"{'rows' if expected_rows else 'empty'}, but this "
                         f"build refuses the query -- {type(exc).__name__}: "
                         f"{reason}")
            continue
        if bool(got) != expected_rows:
            wrong.append(f"{qid}: README says {'rows' if expected_rows else 'empty'}, "
                         f"got {len(got)} rows")
    assert not wrong, (
        "the README's real-layer table no longer matches the graph:\n  "
        + "\n  ".join(wrong)
        + "\n\nUpstream may have published, or the build may have changed what "
          "it accepts. Re-measure and update the table -- this is a claim a "
          "reader will quote."
    )


def test_the_real_layer_is_connected_apart_from_unregistered_operators(real_only):
    """#21's actual question. It is not 1,030 orphans.

    The only nodes with no edge are `Operator`s that no ONNX Runtime kernel
    registers -- a real fact about the upstream data, not a broken join. The
    bound is deliberately loose: the exact number moves when ONNX or ONNX
    Runtime publish, and the claim being defended is "a handful", not a figure.
    """
    client, _ = real_only
    # Built from EDGES_PRESENT rather than every declared type: a newly added
    # edge type would leave its endpoints looking orphaned here. That is safe
    # only because test_which_edge_types_the_real_layer_carries fails first and
    # names the new type -- if that test is ever relaxed, widen this too.
    connected = set()
    for rel in EDGES_PRESENT:
        for src, dst in client.query(
                f"MATCH (a)-[:{rel}]->(b) RETURN a.id, b.id", GRAPH).records:
            connected.add(src)
            connected.add(dst)

    orphans = {}
    for label in LABELS_PRESENT:
        ids = [r[0] for r in
               client.query(f"MATCH (x:{label}) RETURN x.id", GRAPH).records]
        missing = [i for i in ids if i not in connected]
        if missing:
            orphans[label] = len(missing)

    assert set(orphans) <= {"Operator"}, (
        f"labels other than Operator have orphaned nodes: {orphans}"
    )
    total_nodes = scalar(client, "MATCH (x) RETURN count(x) AS n")
    assert sum(orphans.values()) < total_nodes * 0.05, (
        f"{sum(orphans.values())} of {total_nodes} nodes are orphaned; the "
        f"README describes the real layer as connected"
    )
