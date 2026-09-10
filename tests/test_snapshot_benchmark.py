"""The snapshot benchmark's guards, which is where its bugs were (#45).

The timing itself needs a running server and is not tested here -- CI has no
engine over HTTP, and a test that skips at setup would be converted to a failure
by `--no-skips`. What *is* tested is every rule that decides whether a number is
allowed to be produced at all, because the first version of this command
happily printed five clean-looking timings over a graph that had quadrupled
underneath it.

The fake client below lies in `status()` the way the real server does after a
delete -- reporting edges the graph no longer has -- so a benchmark that read
status instead of querying would fail these.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from benchmarks import snapshot


class FakeResult:
    def __init__(self, value):
        self.records = [[value]]


class FakeStatus:
    version = "1.7.0"

    def __init__(self, nodes, edges):
        self.nodes, self.edges = nodes, edges


class FakeClient:
    """Answers queries from `counts`, and reports something else in `status()`."""

    def __init__(self, counts, status=(0, 999_999)):
        self.counts = list(counts)          # (nodes, edges) seen by each query pair
        self._status = status
        self.deleted = 0

    def query(self, cypher, graph):
        """`graph_holds` asks for nodes then edges; the pair advances on edges.

        Advancing on the second of the pair, not the first, so a single entry
        can be read repeatedly -- the last one stays current once the script
        runs out, which is what a one-run invocation needs.
        """
        nodes, edges = self.counts[0]
        if "count(n.id)" in cypher:
            return FakeResult(nodes)
        if "count(r)" in cypher:
            if len(self.counts) > 1:
                self.counts.pop(0)
            return FakeResult(edges)
        raise AssertionError(f"unexpected query: {cypher}")

    def status(self):
        return FakeStatus(*self._status)

    def delete_graph(self, graph):
        self.deleted += 1


@pytest.fixture
def snapfile(tmp_path) -> Path:
    path = tmp_path / "kg.sgsnap"
    path.write_bytes(b"x" * 1024)
    return path


def run(monkeypatch, client, snapfile, *args, times=None):
    monkeypatch.setattr(snapshot, "_client", lambda url: client)
    supplied = list(times or [0.25] * 10)
    monkeypatch.setattr(snapshot, "time_import", lambda url, path: supplied.pop(0))
    monkeypatch.setattr(snapshot, "_restart", lambda cmd, url: None)
    return CliRunner().invoke(
        snapshot.main,
        ["--url", "http://x", "--file", str(snapfile), *args])


def test_graph_holds_asks_the_graph_rather_than_reading_status():
    """The behaviour the module exists to be careful about.

    After `delete_graph` the real server reported `edges=152594` on a graph whose
    own `count(r)` was `0`. A benchmark trusting status would refuse to run
    against a server that was in fact empty.
    """
    client = FakeClient([(0, 0)], status=(0, 152_594))
    assert snapshot.graph_holds(client) == (0, 0)
    assert client.status().edges == 152_594, "the fake must actually disagree"


def test_repeats_above_one_require_a_restart_command(monkeypatch, snapfile):
    """Because `delete_graph` does not clear edges and re-import resurrects them."""
    result = run(monkeypatch, FakeClient([(0, 0)]), snapfile, "--repeats", "5")
    assert result.exit_code != 0
    assert "--restart-cmd" in result.output
    assert "resurrect" in result.output


def test_a_single_run_needs_no_restart_command(monkeypatch, snapfile):
    client = FakeClient([(0, 0), (25_150, 76_303)])
    result = run(monkeypatch, client, snapfile)
    assert result.exit_code == 0, result.output
    assert "25,150 nodes / 76,303 edges" in result.output


def test_it_refuses_to_time_an_import_into_a_non_empty_graph(monkeypatch, snapfile):
    """Import appends, so this would measure a merge into an oversized graph."""
    result = run(monkeypatch, FakeClient([(25_150, 76_303)]), snapfile)
    assert result.exit_code != 0
    assert "already holds" in result.output
    assert "APPENDS" in result.output


def test_a_run_that_does_not_match_the_first_is_rejected(monkeypatch, snapfile):
    """The check that would have caught the delete_graph version of this command.

    Run 2 starts from an apparently empty graph -- `MATCH ()-[r]->()` finds
    nothing while no nodes exist -- and then lands on 152,606 edges once the
    import puts the nodes back and the old edges become reachable again. Only a
    comparison against run 1 sees it; a comparison against zero does not.
    """
    client = FakeClient([(0, 0), (25_150, 76_303),        # run 1: clean
                         (0, 0), (25_150, 152_606)])      # run 2: edges resurrected
    result = run(monkeypatch, client, snapfile, "--repeats", "2",
                 "--restart-cmd", "true")
    assert result.exit_code != 0
    assert "152,606" in result.output and "76,303" in result.output
    assert "not comparable" in result.output


def test_export_refuses_when_there_is_nothing_to_export(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "_client", lambda url: FakeClient([(0, 0)]))
    result = CliRunner().invoke(
        snapshot.main, ["--url", "http://x", "--export", str(tmp_path / "o.sgsnap")])
    assert result.exit_code != 0
    assert "nothing to export" in result.output


def test_export_and_import_cannot_run_together(monkeypatch, snapfile, tmp_path):
    """They are mutually exclusive by construction, not by preference.

    `--export` needs a graph with something in it; timing an import needs an
    empty one. Passing both always died at the second check, *after* the export
    had already run and written a file. It now refuses up front.
    """
    monkeypatch.setattr(snapshot, "_client", lambda url: FakeClient([(25_150, 76_303)]))
    result = CliRunner().invoke(snapshot.main, [
        "--url", "http://x", "--file", str(snapfile),
        "--export", str(tmp_path / "o.sgsnap")])
    assert result.exit_code != 0
    assert "cannot run in one invocation" in result.output
    assert not (tmp_path / "o.sgsnap").exists(), "it refused after exporting"


def test_it_asks_for_something_to_do():
    result = CliRunner().invoke(snapshot.main, ["--url", "http://x"])
    assert result.exit_code != 0
    assert "--file" in result.output and "--export" in result.output
