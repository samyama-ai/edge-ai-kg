"""The probe's verdict mapping and its buckets, driven with fakes.

`benchmarks/engine_notes_probe.py` had no tests. That matters more than it
sounds: a swapped `True`/`False` branch in the verdict mapping changes every
line the probe prints and **no test result**, because nothing exercised it.
Kalyan's review demonstrated exactly that — inverting note 2's verdict and
forcing note 9 to FIXED left the suite green.

What is pinned here is the part a reader acts on:

- `True` is FIXED, `False` is STILL REPRODUCES, `None` is UNSOUND. Getting that
  backwards tells someone a workaround can be dropped when it cannot.
- `built` refuses a fixture that did not land, which is what stops "the engine
  accepted my CREATEs and stored nothing" reading as FIXED. Five of the eleven
  probes compare against an empty answer for success (`[] == sorted([])`,
  `set() == set()`), so this is the difference between measuring the note and
  measuring nothing.

The probe is three modules -- the CLI, `engine_notes_harness.py` (reset,
read, judge) and `engine_notes_cases.py` (one function per note). The
probes themselves need a real engine and are exercised by running the
module; these are the parts that can be wrong without any engine at all.
"""
from __future__ import annotations

import types

import pytest

from benchmarks import engine_notes_cases as cases
from benchmarks import engine_notes_harness as harness


def test_the_verdict_mapping_is_not_inverted():
    """The three states, named. A swap here is silent everywhere else."""
    assert harness.verdict_for(True) == "FIXED"
    assert harness.verdict_for(False) == "STILL REPRODUCES"
    assert harness.verdict_for(None) == "UNSOUND", (
        "None is the third state, not a falsy second one -- a probe that "
        "cannot tell must not report either answer"
    )


def test_a_fixture_that_did_not_land_is_unsound_not_fixed():
    """The gap that made FIXED mean "did not error" for five probes.

    An engine accepting every `CREATE` and storing nothing returns the empty
    answer that `got == sorted(got)` and `got == set()` both treat as success.
    `built` is what turns that into UNSOUND.
    """
    class StoresNothing:
        def query(self, statement, graph):
            return types.SimpleNamespace(records=[[0]])

    with pytest.raises(harness.FixtureNotBuilt, match="built 6 :Deployment rows"):
        harness.built(StoresNothing(), "Deployment", 6)


def test_a_fixture_that_landed_is_accepted():
    """The other direction, or the check above could be a blanket refusal."""
    class StoresSix:
        def query(self, statement, graph):
            return types.SimpleNamespace(records=[[6]])

    harness.built(StoresSix(), "Deployment", 6)      # must not raise


def test_an_engine_answering_nothing_is_unsound_not_a_reproduction():
    """`parses` distinguishes "refuses this" from "answers nothing".

    Notes 5 and 6 read a raised exception as the note reproducing. A dropped
    connection would otherwise print STILL REPRODUCES for a run that measured
    nothing, so the client is asked a statement that certainly parses first.
    """
    class Dead:
        def query(self, statement, graph):
            raise RuntimeError("connection refused")

    assert harness.parses(Dead(), "MATCH (n) RETURN n")[0] is None

    class RefusesOnlyThis:
        def query(self, statement, graph):
            if "RETURN 1" in statement:
                return types.SimpleNamespace(records=[[1]])
            raise RuntimeError("Query error: unsupported syntax")

    assert harness.parses(RefusesOnlyThis(), "BAD CYPHER")[0] is False


def test_a_reset_that_leaves_rows_is_unsound():
    """`DETACH DELETE` succeeding and leaving rows is the case that matters."""
    class NeverEmpties:
        def query(self, statement, graph):
            if "RETURN id(n)" in statement:
                return types.SimpleNamespace(records=[[i] for i in range(7)])
            return types.SimpleNamespace(records=[])

    with pytest.raises(harness.ResetFailed, match="left 7 nodes"):
        harness.reset(NeverEmpties())


def test_every_probe_is_registered_with_a_title_and_a_callable():
    """`PROBES` is what the run iterates; a typo there drops a note silently."""
    numbers = [number for number, _title, _fn in cases.PROBES]
    assert len(numbers) == len(set(numbers)), f"duplicate probe ids: {numbers}"
    assert 7 not in numbers, (
        "note 7 has no probe -- it is a property of the OSS HTTP path with "
        "nothing to isolate embedded, and the output says so"
    )
    for number, title, fn in cases.PROBES:
        assert title and callable(fn), f"probe {number} is not runnable"


class _MinEngine:
    """Answers note 4's queries: `int_min` for the int sentinel, `float_min` for the float one.

    Everything else -- the reset, the `CREATE`s, `built`'s count -- gets the
    answer a healthy engine would, so only the two `min()` values vary.
    """

    def __init__(self, int_min, float_min):
        self.int_min, self.float_min = int_min, float_min

    def query(self, statement, graph):
        if "min(" in statement:
            value = self.float_min if "999999.0" in statement else self.int_min
            return types.SimpleNamespace(records=[[value]])
        if "count(n.id)" in statement:
            return types.SimpleNamespace(records=[[2]])
        return types.SimpleNamespace(records=[])


def test_note_4_reads_only_the_int_sentinel_for_its_verdict():
    """The float query is the control, not half of the verdict."""
    assert cases.note_4(_MinEngine(6.9, 6.9), 0)[0] is True
    assert cases.note_4(_MinEngine(999999, 6.9), 0)[0] is False


def test_note_4_with_a_broken_control_cannot_tell():
    """`min()` wrong without the int/float mix is not note 4 either way."""
    assert cases.note_4(_MinEngine(6.9, 999999.0), 0)[0] is None


def test_note_4_survives_a_non_numeric_minimum():
    """A None from `min()` is a verdict, not a TypeError landing in ERROR."""
    assert cases.note_4(_MinEngine(None, 6.9), 0)[0] is False
    assert cases.note_4(_MinEngine(6.9, None), 0)[0] is None


def test_note_1_refuses_a_scale_that_cannot_tell():
    """At 0 an empty graph satisfies `len(got) == n`; at 1 the product is the answer."""
    for scale in (0, 1):
        assert cases.note_1(object(), scale)[0] is None


class _NullMatchingEngine:
    """A build where note 8b reproduces: `<>` matches a node with no `kind`."""

    def query(self, statement, graph):
        if 'q.id = "null" RETURN q.kind' in statement:
            return types.SimpleNamespace(records=[[None]])
        if "<>" in statement:
            restricted = 'q.id = "has"' in statement
            hits = [["has"]] if restricted else [["has"], ["null"]]
            if '"NPU"' in statement:
                hits = [] if restricted else [["null"]]
            return types.SimpleNamespace(records=hits)
        return types.SimpleNamespace(records=[])


def test_note_8b_reports_a_reproduction_as_one_not_as_unsound():
    """The control must not be contaminated by the bug the probe measures.

    Over the whole label, `kind <> "GPU"` matches `null` on exactly the build
    where 8b reproduces, and the control used to read that as "`<>` is broken"
    and return UNSOUND -- so the probe could never say STILL REPRODUCES.
    """
    ok, detail = cases.note_8b(_NullMatchingEngine(), 0)
    assert ok is False, detail
