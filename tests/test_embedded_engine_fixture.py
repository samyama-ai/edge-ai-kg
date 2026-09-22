"""The shared embedded-engine fixture, and the skip discipline around it.

Infrastructure, landing ahead of the tests that use it. It is here rather than
in a test module because a fixture cannot be imported from a sibling test file,
and the alternative -- `from conftest import ...` -- depends on rootdir being on
`sys.path`, which is the sort of thing that works until it does not.

Two properties are worth asserting rather than assuming:

**Each client is its own graph.** `SamyamaClient.embedded()` is in-memory and
per-process, so two clients cannot see each other's nodes. Every fixture built
on this relies on that; if it ever stopped being true, tests would start
inheriting each other's data and the failures would look like query bugs.

**The skip happens in setup.** `conftest.py` converts setup-phase skips to
failures under `--no-skips`, so a CI job with no engine fails loudly instead of
reporting green over a suite that never ran. A helper called from inside a test
body would skip in the body and slip past that.
"""
import os
import pathlib
import types

import pytest

GRAPH = "default"


ROOT_CONFTEST = pathlib.Path(__file__).resolve().parent.parent / "conftest.py"


def held_nodes(client) -> int:
    """`count(n)` as a number, without assuming the engine returns a row.

    Measured on the embedded build: an aggregate over an empty match does
    return one row, `[[0]]` -- unlike a plain match, which returns `[]`. So
    `.records[0][0]` works today. It is wrapped anyway, because if that ever
    changed these tests would die on a bare `IndexError` and none of the
    failure messages below would print, which is the opposite of what they are
    written for.

    `count(n)`, not `count(n.id)`: the id form counts only nodes that have an
    `id`, and a leftover without one is exactly what these assertions are
    looking for. CLAUDE.md's rule against aggregating a bare node variable is
    about `count(DISTINCT x)` over a multi-variable MATCH (engine note 9), not
    about this.
    """
    records = client.query("MATCH (n) RETURN count(n)", GRAPH).records
    return records[0][0] if records else 0


def test_the_factory_gives_a_working_client(engine_factory, reset_embedded_graph):
    client = engine_factory()
    try:
        client.query('CREATE (:T {id:"a"})', GRAPH)
        assert client.query("MATCH (n:T) RETURN count(n.id)",
                            GRAPH).records[0][0] == 1
    finally:
        # The client is this test's own, so nothing else can see the node --
        # but leaving it makes that an assumption rather than a fact, and the
        # next test to reuse a client would inherit it.
        reset_embedded_graph(client)


def test_each_client_is_its_own_graph(engine_factory, reset_embedded_graph):
    """The property every fixture built on this depends on.

    The second client is taken **un-reset**. With the reset, this test could not
    fail: `make()` empties the graph on the way out, so if the two clients did
    share one in-memory graph the reset would delete the first client's node and
    `held == 0` would still pass -- while `first` silently lost its data. That
    version asserted the reset works, which is the test below.

    So both halves are checked: the new client sees nothing, and the old one
    still sees its own node afterwards. The second assertion is the one that
    catches sharing.
    """
    first = engine_factory()
    try:
        first.query('CREATE (:T {id:"a"})', GRAPH)
        second = engine_factory(reset=False)

        held = held_nodes(second)
        assert held == 0, (
            f"a second embedded client sees {held} nodes from the first, with "
            f"no reset in between. Fixtures here assume a fresh graph per "
            f"client; if that stops holding they inherit each other's data and "
            f"the failures look like query bugs."
        )
        survived = first.query("MATCH (n:T) RETURN count(n.id)",
                               GRAPH).records[0][0]
        assert survived == 1, (
            f"the first client lost its node when a second was constructed, so "
            f"the two are sharing a graph. Got {survived}."
        )
    finally:
        # After both assertions, never before: resetting `first` earlier would
        # empty a shared graph and hide exactly the sharing this checks for.
        reset_embedded_graph(first)


def test_reset_empties_a_reused_client(engine_factory, reset_embedded_graph):
    """Through the `reset_embedded_graph` fixture, not `from conftest import ...`.

    An earlier version of this test opened with that import -- the exact thing
    the module docstring above argues the fixture exists to avoid. If the
    import is safe enough to use here the rationale is wrong; if it is not,
    this test is the one place that breaks.
    """
    client = engine_factory()
    client.query('CREATE (:T {id:"a"})', GRAPH)
    reset_embedded_graph(client)
    assert held_nodes(client) == 0


def test_full_scale_is_registered(request):
    """That the option exists, which is what can actually be asserted here.

    Named for what it checks. No test reads the option yet -- it is plumbing
    landing ahead of the scale-sensitive tests in #96 and #100, which will call
    `request.config.getoption("--full-scale")` and skip on it. If the option
    were not registered, that call raises `ValueError` inside each of them,
    turning an opt-out into a crash. `getoption` not raising is the check.

    Whether the *default* is off cannot be read from a run that may have been
    given `--full-scale` on the command line, so this does not pretend to.
    """
    assert request.config.getoption("--full-scale") in (True, False)


def test_the_probe_skips_a_missing_engine_and_re_raises_a_broken_reset(
        conftest_internals):
    """The classification `engine_factory` makes, driven through `probe_engine`.

    An earlier version of this test asserted only that `reset_embedded` raises,
    and its docstring claimed it covered the probe. It did not: the swallow
    lived in `engine_factory`, which is a fixture body and cannot be called. So
    the property most worth pinning -- that the probe re-raises rather than
    skips -- went unchecked while the docstring said otherwise.

    `probe_engine` is that logic as a function, and this drives both branches:

    - the engine is not importable, or will not construct -> **skip**, because
      a machine without the extension is not a failing suite;
    - the engine works and will not empty a graph -> **raise**, because every
      fixture built on this depends on a clean one. Reporting that as
      "unavailable" would give a green run to a suite whose fixtures were all
      sharing one dirty graph, with a skip line explaining it away.
    """
    internals = conftest_internals

    def missing():
        raise ModuleNotFoundError("No module named 'samyama'")

    with pytest.raises(pytest.skip.Exception):
        internals.probe_engine(missing)

    def broken_reset():
        raise internals.ResetIncomplete("reset left 5 nodes in `default`")

    # Not `pytest.raises`: it does not catch `Skipped`, which is a
    # `BaseException`, so a `probe_engine` that had lost its `raise` would let
    # the skip propagate and this test would be *reported as skipped* -- and
    # `--no-skips` converts setup-phase skips, not call-phase ones. The guard
    # against a dirty shared graph would then be green and silent, which is
    # the failure this whole module exists to prevent.
    try:
        internals.probe_engine(broken_reset)
    except internals.ResetIncomplete as exc:
        assert "reset left 5 nodes" in str(exc)
    except BaseException as exc:                    # including Skipped
        pytest.fail(f"probe_engine turned a broken reset into {exc!r}, so a "
                    f"suite sharing one dirty graph would report green")
    else:
        pytest.fail("probe_engine returned on a broken reset")


def test_a_reset_that_leaves_rows_raises(conftest_internals):
    """`reset_embedded` itself, on a client that answers but deletes nothing."""
    internals = conftest_internals

    class NeverEmpties:
        """A live engine with a broken delete."""

        def query(self, statement, graph):
            if "count(" in statement:
                return types.SimpleNamespace(records=[[5]])
            return types.SimpleNamespace(records=[])

    with pytest.raises(internals.ResetIncomplete, match="reset left 5 nodes"):
        internals.reset_embedded(NeverEmpties())

    assert issubclass(internals.ResetIncomplete, Exception), (
        "it is an Exception subclass, which is exactly why `probe_engine` has "
        "to name it rather than relying on `except Exception` missing it"
    )


def _run_isolated(tmp_path, *args):
    """Run pytest over a copy of the real `conftest.py`, with no engine installed.

    A subprocess and a copy, rather than `pytester`: the property under test is
    *where* the skip happens, and that is decided by this repo's own
    `conftest.py` -- so the file itself is copied in rather than a stub of it
    being written. `samyama.py` on the path raises on import, which is the
    condition a machine without the extension is in.
    """
    import shutil
    import subprocess
    import sys

    shutil.copy(ROOT_CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "samyama.py").write_text(
        "raise ImportError('no extension here')\n", encoding="utf-8")
    (tmp_path / "test_needs_the_engine.py").write_text(
        "def test_wants_a_client(engine_factory):\n"
        "    assert engine_factory() is not None\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path)})


def test_the_engine_skip_happens_in_setup_so_no_skips_converts_it(tmp_path):
    """The headline property of this fixture, pinned by running it.

    `--no-skips` converts a *setup-phase* skip into an error. Move the skip
    into the returned `make` -- so it fires at call time instead -- and the run
    goes green with a skip line, under `--no-skips`, with nothing to notice.
    Nothing else in this file catches that, because the difference is invisible
    to any test that imports the fixture rather than running it.
    """
    plain = _run_isolated(tmp_path, ".")
    assert plain.returncode == 0, (
        f"without --no-skips a missing engine should skip, not fail:\n"
        f"{plain.stdout[-800:]}")
    assert "1 skipped" in plain.stdout, plain.stdout[-800:]

    strict = _run_isolated(tmp_path, ".", "--no-skips")
    assert strict.returncode != 0, (
        "--no-skips did not fail a run whose engine fixture skipped. The skip "
        "is firing at call time rather than in setup, which is the one thing "
        "this fixture is shaped around:\n" + strict.stdout[-800:])
    assert "error" in strict.stdout.lower(), strict.stdout[-800:]
