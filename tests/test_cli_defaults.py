"""The target graph is one setting spread across seven files.

Issue #5: the loader defaulted to `default` and the benchmark runner to
`edge_ai`. Nothing caught it because the engine ignores the graph argument on
the OSS HTTP path (engine note 7), so both landed in `default` anyway. When that
is fixed upstream the documented two-step -- load, then benchmark -- queries a
graph nothing was written to, and every catalog query returns 0 rows without
raising.

That failure has no error to grep for, so it is pinned here instead: the two
CLIs, the MCP server's config, and the module-level `GRAPH` constants the demos
and the test fixtures use must all name the same graph.
"""
import re
from pathlib import Path

import yaml

from benchmarks.run_benchmark import main as benchmark_cli
from etl.loader import main as loader_cli

ROOT = Path(__file__).resolve().parent.parent
MCP_CONFIG = ROOT / "mcp_server" / "config.yaml"
GRAPH_CONSTANT_FILES = [
    "demo/demo.py",
    "demo/questions.py",
    "tests/test_correctness.py",
    "tests/test_edge_verification.py",
    # The shared fixture and its own tests. This list already pins two test
    # modules, so a third and the root conftest belong in it: a `GRAPH` that
    # disagreed here would send every fixture-built graph to a different
    # tenant name than the CLIs use, and `--graph` being ignored on OSS
    # (engine note 7) is exactly what made the last such disagreement
    # invisible.
    "conftest.py",
    "tests/test_embedded_engine_fixture.py",
]


def option_default(command, name: str):
    for param in command.params:
        if param.name == name:
            return param.default
    raise AssertionError(f"{command.name} has no --{name} option")


def graph_constant(relative: str) -> str:
    """The `GRAPH = "..."` module constant, read rather than imported.

    This module already imports `etl.loader` and `benchmarks.run_benchmark`, so
    the ETL package is in the import graph regardless. Reading the source keeps
    `demo/*` out of it -- importing those pulls in `rich` and runs their
    module-level setup for a one-line assertion.
    """
    text = (ROOT / relative).read_text(encoding="utf-8")
    match = re.search(r'^GRAPH = "([^"]+)"', text, re.MULTILINE)
    assert match, f"{relative} no longer defines a module-level GRAPH constant"
    return match.group(1)


def test_loader_and_benchmark_target_the_same_graph():
    loader = option_default(loader_cli, "graph")
    benchmark = option_default(benchmark_cli, "graph")
    assert loader == benchmark, (
        f"loader defaults to {loader!r} but the benchmark runner defaults to "
        f"{benchmark!r}; benchmarking a fresh load would query an empty graph"
    )


def test_mcp_server_targets_the_same_graph():
    configured = yaml.safe_load(MCP_CONFIG.read_text(encoding="utf-8"))["graph"]["graph"]
    assert configured == option_default(loader_cli, "graph"), (
        f"mcp_server/config.yaml pins graph {configured!r}, which is not the "
        f"graph the loader writes to"
    )


def test_every_graph_constant_matches_the_loader():
    loader = option_default(loader_cli, "graph")
    found = {f: graph_constant(f) for f in GRAPH_CONSTANT_FILES}
    wrong = {f: g for f, g in found.items() if g != loader}
    assert not wrong, (
        f"these name a different graph than the loader writes to ({loader!r}): {wrong}"
    )
