"""The ingest benchmark reports what it claims to, and the batch default is real.

#10 asked for velocity to be a tracked artifact rather than a sentence in
`docs/engine-notes.md`. `benchmarks/ingest.py` is that artifact; this holds it
to its shape.

**No timing is asserted.** Rates are machine-, build- and load-specific -- the
same box gave 2,594 and 3,146 edges/s for two batch sizes, and a CI runner will
give neither. A performance figure pinned in a test fails for the hardware's
reasons, which is the class of test people learn to ignore
(`tests/test_real_layer_shape.py` makes the same call about upstream counts).

What *is* asserted: the report carries every field the issue asked for, the
arithmetic between them is consistent, `patterns_per_statement` tracks the thing
the cost actually depends on, and the batch default in `etl/helpers.py` is the
one the measurement chose rather than a number that drifted back.
"""
import inspect
import math

import pytest

from benchmarks import ingest
from etl.helpers import create_edges


def test_the_edge_batch_default_is_the_measured_optimum():
    """50, not the 100 it was.

    Measured at `--scale 1.0` (embedded): 100 gives 2,594 edges/s against 3,140
    at 50, because cost is superlinear in `MATCH` patterns per statement rather
    than in the number of statements. The sweep is in the module docstring and
    reproducible with `--sweep-edge-batch`.

    Pinned because the change is a one-character edit that looks like a
    rounding-down and reads as arbitrary without the measurement beside it.
    """
    default = inspect.signature(create_edges).parameters["batch"].default
    assert default == 50, (
        f"create_edges batches {default}, but 50-60 was measured fastest and 100 "
        f"costs 21% of edge throughput. If this is a deliberate change, re-run "
        f"`python -m benchmarks.ingest --sweep-edge-batch` and update the table "
        f"in etl/helpers.py and docs/engine-notes.md with it."
    )


def test_patterns_per_statement_counts_distinct_endpoints():
    """The number the cost tracks, on a fixture small enough to count by hand.

    Six edges over four nodes. Batched two at a time the first batch touches
    a, b, c (3 patterns), the second c, d and a, d -- so the mean is what the
    engine actually sees, not `2 * batch`.
    """
    edges = [
        ("A", "a", "R", "B", "b", None),
        ("A", "a", "R", "B", "c", None),
        ("A", "d", "R", "B", "b", None),
        ("A", "d", "R", "B", "c", None),
    ]
    # Batch of 2: {a,b} then {a,c} then... -- each pair shares its source.
    assert ingest.patterns_per_statement(edges, 2) == 3.0
    # One per statement: always both endpoints.
    assert ingest.patterns_per_statement(edges, 1) == 2.0
    # All four at once: only the four distinct nodes, not eight.
    assert ingest.patterns_per_statement(edges, 4) == 4.0
    assert ingest.patterns_per_statement([], 10) == 0.0


def test_shared_endpoints_reduce_the_pattern_count():
    """The dedup that makes batching worth anything at all.

    If every edge in a batch shares one endpoint, the statement carries
    `n + 1` patterns rather than `2n`. This is why the `Fleet`'s natural
    ordering beats sorting by relationship type -- runs of edges sharing a
    source stay together.
    """
    fan_out = [("A", "hub", "R", "B", f"leaf{i}", None) for i in range(10)]
    assert ingest.patterns_per_statement(fan_out, 10) == 11.0

    all_distinct = [("A", f"s{i}", "R", "B", f"t{i}", None) for i in range(10)]
    assert ingest.patterns_per_statement(all_distinct, 10) == 20.0


@pytest.fixture(scope="module")
def report():
    """One real load, small enough to be quick. Skips without `data/`."""
    try:
        fleet = ingest.build_fleet(seed=4242, scale=0.05, layers="generated")
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    try:
        from samyama import SamyamaClient
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")
    return ingest.time_one_load(
        SamyamaClient.embedded, fleet, node_batch=250, edge_batch=50,
        graph="default")


def test_the_report_carries_every_field_the_issue_asked_for(report):
    """#10 wants nodes/s and edges/s recorded, with the configuration."""
    required = {
        "nodes", "edges", "node_seconds", "edge_seconds",
        "nodes_per_s", "edges_per_s", "node_batch", "edge_batch",
        "patterns_per_statement",
    }
    missing = required - set(report)
    assert not missing, f"ingest report is missing {sorted(missing)}"
    assert all(report[k] is not None for k in required), (
        f"null fields in the report: "
        f"{sorted(k for k in required if report[k] is None)}"
    )


def test_the_reported_rates_match_the_reported_times(report):
    """Rate and duration must describe the same load.

    Cheap, and it is the failure that would make every published figure wrong
    while the report still looked well-formed -- e.g. dividing by the total
    elapsed rather than the phase.
    """
    for count, seconds, rate in (
        ("nodes", "node_seconds", "nodes_per_s"),
        ("edges", "edge_seconds", "edges_per_s"),
    ):
        expected = report[count] / report[seconds]
        assert math.isclose(report[rate], expected, rel_tol=0.02), (
            f"{rate} is {report[rate]:,} but {report[count]:,} / "
            f"{report[seconds]}s is {expected:,.0f}"
        )


def test_the_load_actually_did_the_work(report):
    """A benchmark that measures an empty load would report a spectacular rate."""
    assert report["nodes"] > 0 and report["edges"] > 0, (
        "the fleet is empty, so the timings measure nothing"
    )
    assert report["node_seconds"] > 0 and report["edge_seconds"] > 0, (
        "a phase took no measurable time, which means it did not run"
    )
    assert 2 <= report["patterns_per_statement"] <= 2 * report["edge_batch"], (
        f"patterns/statement is {report['patterns_per_statement']}, outside the "
        f"2..{2 * report['edge_batch']} a batch of {report['edge_batch']} can "
        f"produce -- the endpoint dedup is wrong"
    )
