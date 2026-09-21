"""The measuring half of `engine_notes_probe`: reset, read, and judge.

Split out of `benchmarks/engine_notes_probe.py` so each file stays reviewable:
this module is what decides whether a probe's answer can be trusted at all --
an emptied graph, a fixture that landed, an engine that is refusing a statement
rather than not answering -- and what a trusted answer is called. The probes
themselves are in `engine_notes_cases.py`; the CLI that runs them is
`engine_notes_probe.py`.

None of this needs an engine to be tested, which is the other reason it is
separate: `tests/test_engine_notes_probe.py` drives every function here with
fakes.
"""
from __future__ import annotations

GRAPH = "default"


class ResetFailed(RuntimeError):
    """The reset between probes did not run, so the next verdict is not trustworthy.

    Swallowing this is tempting and wrong. Note 8 measures whether a property
    survives `DETACH DELETE`, so its entire verdict *is* the reset: a silently
    failed delete leaves the old node in place and the probe reports on a graph
    it did not build. Every later probe inherits the contamination.
    """


class FixtureNotBuilt(RuntimeError):
    """A probe's own `CREATE`s did not land, so its verdict would read an empty graph.

    Separate from `ResetFailed` because it is a different fault with a
    different fix: the reset worked, and this probe's fixture is what is
    missing. The CLI reports both as UNSOUND, and the message says which.
    """


def reset(client):
    """Empty the graph, and verify it is empty.

    Raising only when `DETACH DELETE` *errors* was not enough: a delete that
    succeeds and leaves rows behind is the failure mode that matters here,
    because every probe opens with this call and would then measure a graph
    holding the previous probe's fixture. An inherited node shows up as a
    wrong row count, which is exactly what these probes read as a verdict.
    Note 8 relies on this most directly: without it, a `p1` surviving its
    delete would be reported as note 8.
    """
    try:
        client.query("MATCH (n) DETACH DELETE n", GRAPH)
        # The ids themselves, not `count(n)`. A bare-node aggregate is the
        # shape engine note 9 says does not aggregate, and `count(n.id)` would
        # miss a node with no `id` -- so no aggregate at all: every surviving
        # node is one row, whatever properties it has. `id()` is the same
        # function `docs/vector-search.md` bridges internal ids with, and every
        # probe run opens with this line -- if an engine dropped `id()`, the
        # message says the reset failed and names the error, not a verdict.
        left = len(client.query("MATCH (n) RETURN id(n)", GRAPH).records)
    except Exception as exc:
        raise ResetFailed(f"reset failed; later verdicts are unsound: {exc}") from exc
    if left:
        raise ResetFailed(
            f"reset ran without error and left {left:,} nodes; later verdicts "
            f"are unsound, because every probe below would measure a graph "
            f"holding whatever the last one built")


def rows(client, cypher):
    return client.query(cypher, GRAPH).records


def verdict_for(ok):
    """`True` -> FIXED, `False` -> STILL REPRODUCES, `None` -> UNSOUND.

    Its own function so it can be tested. Inline in `main` it was unreachable
    from anything: a swapped `True`/`False` branch changed every line the probe
    prints and not one test result, because nothing exercised `main` at all.

    `None` is the third state and the reason this is not a boolean: a probe
    that cannot tell -- a dead engine, a fixture that did not land, note 8
    showing through note 8b -- must not report either answer.
    """
    if ok is None:
        return "UNSOUND"
    return "FIXED" if ok else "STILL REPRODUCES"


def built(client, label: str, expected: int):
    """Raise unless the fixture for this probe actually landed.

    Every verdict below reads a row count or a row order. An engine that
    accepted the `CREATE`s and stored nothing would hand back the empty answer
    each of those comparisons treats as success -- `[] == sorted([])`,
    `set() == set()` -- so FIXED would mean "did not error" rather than "gave
    the right answer". This is the difference, and it is checked once here
    rather than trusted five times.
    """
    got = rows(client, f"MATCH (n:{label}) RETURN count(n.id)")
    held = got[0][0] if got else 0
    if held != expected:
        raise FixtureNotBuilt(
            f"built {expected} :{label} rows and the graph holds {held}. The "
            f"verdict below reads this fixture, so it would be measuring an "
            f"empty graph rather than the behaviour.")


def parses(client, cypher: str):
    """Whether the engine accepts `cypher`, as `(state, reason)`.

    `state` is `True` (it parsed), `False` (refused) or `None` (cannot tell --
    the engine is not answering at all); `reason` is a line for the output.

    Named for what `True` means. It was `does_not_parse` and returned `True`
    when the statement *did* parse, so every call site read backwards.

    Notes 5 and 6 are "this statement does not parse", so the probe reads a
    raised exception as the note reproducing. Any exception would do that --
    including a dropped connection or an engine that has stopped answering at
    all, which would report "STILL REPRODUCES" for a run that measured nothing.

    So after a failure the client is asked a statement that certainly parses.
    If that also fails, the engine is not refusing this Cypher, it is simply not
    working, and the verdict is `None` -- unsound -- rather than a reproduction.
    """
    try:
        rows(client, cypher)
        return True, "parsed"
    except Exception as exc:  # the engine raises untyped errors
        refusal = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}"
    try:
        rows(client, "RETURN 1")
    except Exception as alive:
        return None, (f"the statement failed ({refusal}) but so did `RETURN 1` "
                      f"({type(alive).__name__}), so the engine is not refusing "
                      f"this Cypher -- it is not answering at all")
    return False, refusal
