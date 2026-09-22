"""Repo-wide pytest configuration.

Exists for one reason: **a skipped test must be able to fail CI** (#2).

13 of the test modules call `pytest.skip` when `data/` is absent, because
`etl.download_data` has to run before anything works. That is right for a
developer who has not downloaded the sources yet, and it is a trap in CI: a
workflow that forgets the download step goes *green* while running almost
nothing.

`pytest --no-skips` turns every *fixture* skip into a failure, so the workflow asserts
that the suite really ran rather than that it merely exited zero.
"""
import types

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-skips",
        action="store_true",
        default=False,
        help=("Treat a skipped test as a failure. Used by CI, where a skip "
              "means the environment is wrong rather than that the test is "
              "inapplicable."),
    )
    parser.addoption(
        "--full-scale",
        action="store_true",
        default=False,
        help=("Also run the checks that need a `--scale 1.0` graph. Off by "
              "default because building and loading one costs about three "
              "minutes; on for anything that changes a query's join shape, "
              "which `docs/engine-notes.md` note 1 says a small graph cannot "
              "validate."),
    )


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Convert skips to failures under `--no-skips`.

    `xfail` is deliberately exempt. pytest reports an xfailed test as skipped
    with a `wasxfail` attribute, and this suite has some -- currently the
    `test_demo_recording.py` marks for #30, where the recording predates the
    build. (It also carried six for #56 -- in `test_correctness.py`, four
    parameters across two parametrised sweeps plus the whole-test mark on
    `test_ea04_quantization_unlock_is_not_a_cartesian_product`; and in
    `tests/test_empty_answers.py`, `test_ea01_zero_row_case` -- until the
    embedded engine was pinned to `samyama>=1.7.1` and they all came off.) Those
    are *expected* outcomes that CI should tolerate; a `pytest.skip` for a
    missing `data/` is not. Failing on both would make the option unusable here
    and it would simply be turned off.
    """
    outcome = yield
    report = outcome.get_result()
    if not item.config.getoption("--no-skips"):
        return
    # Only setup-phase skips. A skip raised from a *fixture* means the
    # environment could not be built -- no `data/`, no engine -- which is
    # precisely what CI must not tolerate. A skip raised from the test *body*
    # means the case does not apply, like
    # `tests/test_correctness.py`'s "EA06 has no ORDER BY" for a query with
    # nothing to sort. Failing on the second would make this option permanently
    # red and therefore switched off.
    if report.when != "setup":
        return
    if report.skipped and not hasattr(report, "wasxfail"):
        report.outcome = "failed"
        reason = ""
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            reason = report.longrepr[2]
        report.longrepr = (
            f"Skipped, and --no-skips is set: {reason}\n"
            f"In CI a skip means the environment is wrong -- most often that "
            f"`python -m etl.download_data` has not run, which makes 13 test "
            f"modules skip and the suite pass while testing almost nothing."
        )


# --------------------------------------------------------------------------
# A fresh embedded engine, for the modules that build purpose-built fixtures.
# Here rather than in a test module because a fixture cannot be imported from a
# sibling test file the way a helper can -- the alternative is
# `from conftest import ...`, which depends on rootdir being on `sys.path`.
#
# Session-scoped deliberately: it holds no per-test state, it returns a
# factory, and every client that factory makes is new. That is what lets a
# module-scoped fixture request it, which a function-scoped one could not.
#
# It skips in **setup**, which matters: this file converts only setup-phase
# skips to failures under `--no-skips`, so a helper called from inside a test
# body would let a CI job with no engine skip the regressions and stay green.
# --------------------------------------------------------------------------
GRAPH = "default"


class ResetIncomplete(RuntimeError):
    """The reset ran and the graph is not empty.

    Its own type so the availability probe in `engine_factory` cannot swallow
    it. "No engine here, skip" and "the engine will not empty a graph" are
    different findings, and turning the second into the first is how a suite
    reports green while testing nothing.
    """


def reset_embedded(client):
    """`etl.loader.reset_graph`, with this file's fixed graph name.

    Reused rather than reimplemented. A copy here dropped the loader's
    `MATCH (n) DELETE n` fallback and swallowed unconditionally -- and that
    fallback is not decoration: `DETACH DELETE` does fail on some builds, which
    is why the loader nests the two. Without it, `engine_factory()` would hand
    back a *non-empty* client while claiming `reset=True`, and the fixtures
    built on it would see rows nothing created. That is the failure mode this
    file's docstring says the design exists to prevent.

    One implementation also means one place to fix when the engine changes.

    `DETACH DELETE` is not a full reset -- engine note 8 -- and does not need to
    be here. Each `SamyamaClient.embedded()` is its own in-memory graph, so a
    fresh client starts empty and this is belt-and-braces for a reused one.
    Note 8 is about a *property* surviving the delete and resurrecting onto a
    new node with the same id, and every fixture using this sets every property
    it asserts on.
    """
    from etl.loader import reset_graph

    reset_graph(client, GRAPH)
    # And verify, because `reset_graph` cannot fail loudly: it swallows both
    # delete attempts and echoes to stderr, which suits a loader run a person
    # is watching and does not suit a fixture. Without this, `make(reset=True)`
    # could hand back a non-empty client while this docstring promises an empty
    # one -- the same failure the copy-with-no-fallback had, arriving by the
    # other route.
    # `count(n)`, not `count(n.id)`. CLAUDE.md's "aggregate a property, never a
    # bare node variable" is about `count(DISTINCT x)` over a multi-variable
    # MATCH (engine note 9); this is a single-variable `MATCH (n)`, where
    # `count(n)` measures correctly and still returns `[[0]]` on an empty
    # graph. The distinction matters: measured on 1.7.1, a graph holding
    # `{id:"a"}` and `{name:"noid"}` gives `count(n.id)` -> 1 and
    # `count(n)` -> 2, so the id form is blind to exactly the leftover a
    # fixture is least likely to have set -- and this check exists to catch
    # leftovers.
    records = client.query("MATCH (n) RETURN count(n)", GRAPH).records
    held = records[0][0] if records else 0
    if held:
        raise ResetIncomplete(
            f"reset left {held:,} nodes in `{GRAPH}`. `etl.loader.reset_graph` "
            f"tries `DETACH DELETE` then `DELETE` and only warns if both fail, "
            f"so a fixture asking for a clean client has to check rather than "
            f"assume.")


@pytest.fixture(scope="session")
def conftest_internals():
    """The callables `tests/test_embedded_engine_fixture.py` checks.

    Through the fixture system rather than `from conftest import ...`, which is
    the import this file exists to make unnecessary -- and which that module's
    own docstring argues against two paragraphs above where it was doing it.
    """
    return types.SimpleNamespace(probe_engine=probe_engine,
                                 reset_embedded=reset_embedded,
                                 ResetIncomplete=ResetIncomplete)


@pytest.fixture(scope="session")
def reset_embedded_graph():
    """`reset_embedded`, reached through the fixture system rather than imported.

    **Embedded clients only.** It wipes `GRAPH`, and `--graph` is ignored on
    the OSS HTTP path (engine note 7), so handing this a server client would
    empty whatever that server holds in `default`. Nothing does today.

    A test wanting to empty a client's graph would otherwise write
    `from conftest import reset_embedded` -- the exact import this file exists
    to make unnecessary, and one that depends on rootdir being on `sys.path`.

    Not named `reset_graph`: `etl.loader.reset_graph` is a different callable
    with a different signature (`(client, graph)` against `(client)`), and two
    things one import apart sharing a name is a collision waiting to be made.
    """
    return reset_embedded


def probe_engine(make):
    """Build one client to prove the engine works, and classify what goes wrong.

    A function rather than four lines inside the fixture, because the thing
    worth testing is the classification and a fixture body cannot be called.

    `ResetIncomplete` is re-raised by name. The engine is present and answering
    -- it just will not empty a graph, which every fixture built on this
    depends on. Folding that into "engine unavailable" would hand a green run
    to a suite whose fixtures were all sharing one dirty graph, with a skip
    line explaining it away.
    """
    try:
        make()
    except ResetIncomplete:
        raise
    except Exception as exc:  # noqa: BLE001 -- untyped engine errors mean skip
        # Named widely on purpose: `reset_embedded` imports `etl.loader`
        # lazily, so a broken loader import arrives here too and "engine
        # unavailable" alone would send the next reader to the wrong place.
        pytest.skip(f"embedded Samyama engine or its loader unavailable: "
                    f"{type(exc).__name__}: {exc}")


@pytest.fixture(scope="session")
def engine_factory():
    """Returns `make(reset=True)`, giving a fresh embedded client.

    Session-scoped because it holds no per-test state -- it returns a factory,
    and every client it makes is new. The scope is the point: a module-scoped
    fixture cannot request a function-scoped one, and a module-scoped graph is
    what the tests using this need. Without that, a module fixture has to reach
    past the fixture system with `from conftest import ...`, which depends on
    rootdir being on `sys.path`. The tests that do need it land in #96 and #100;
    this is the plumbing, ahead of them.

    `reset=False` returns the client exactly as the engine gave it, which is
    the only way to check that two embedded clients do not share a graph: the
    reset would empty a shared one on the way out and the check would pass
    either way.
    """
    try:
        from samyama import SamyamaClient
    except Exception as exc:  # noqa: BLE001 -- no extension means skip
        # No `pragma: no cover`: a CI job without the extension reaches this on
        # every run, so marking it unreachable was wrong about the commonest
        # environment this file has.
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

    def make(reset: bool = True):
        client = SamyamaClient.embedded()
        if reset:
            reset_embedded(client)
        return client

    probe_engine(make)
    return make
