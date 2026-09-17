"""The probe's verdict mapping and its buckets, driven with fakes.

`benchmarks/engine_notes_probe.py` had no tests. That matters more than it
sounds: a swapped `True`/`False` branch in the verdict mapping changes every
line the probe prints and **no test result**, because nothing exercised it.
Kalyan's review demonstrated exactly that — inverting note 2's verdict and
forcing note 9 to FIXED left the suite green.

What is pinned here is the part a reader acts on:

- `True` is FIXED, `False` is STILL REPRODUCES, `None` is UNSOUND. Getting that
  backwards tells someone a workaround can be dropped when it cannot.
- `_built` refuses a fixture that did not land, which is what stops "the engine
  accepted my CREATEs and stored nothing" reading as FIXED. Five of the eleven
  probes compare against an empty answer for success (`[] == sorted([])`,
  `set() == set()`), so this is the difference between measuring the note and
  measuring nothing.

The probes themselves need a real engine and are exercised by running the
module; these are the parts that can be wrong without any engine at all.
"""
from __future__ import annotations

import types

import pytest

from benchmarks import engine_notes_probe as probe


def test_the_verdict_mapping_is_not_inverted():
    """The three states, named. A swap here is silent everywhere else."""
    assert probe.verdict_for(True) == "FIXED"
    assert probe.verdict_for(False) == "STILL REPRODUCES"
    assert probe.verdict_for(None) == "UNSOUND", (
        "None is the third state, not a falsy second one -- a probe that "
        "cannot tell must not report either answer"
    )


def test_a_fixture_that_did_not_land_is_unsound_not_fixed():
    """The gap that made FIXED mean "did not error" for five probes.

    An engine accepting every `CREATE` and storing nothing returns the empty
    answer that `got == sorted(got)` and `got == set()` both treat as success.
    `_built` is what turns that into UNSOUND.
    """
    class StoresNothing:
        def query(self, statement, graph):
            return types.SimpleNamespace(records=[[0]])

    with pytest.raises(probe.ResetFailed, match="built 6 :Deployment rows"):
        probe._built(StoresNothing(), "Deployment", 6)


def test_a_fixture_that_landed_is_accepted():
    """The other direction, or the check above could be a blanket refusal."""
    class StoresSix:
        def query(self, statement, graph):
            return types.SimpleNamespace(records=[[6]])

    probe._built(StoresSix(), "Deployment", 6)      # must not raise


def test_an_engine_answering_nothing_is_unsound_not_a_reproduction():
    """`_does_not_parse` distinguishes "refuses this" from "answers nothing".

    Notes 5 and 6 read a raised exception as the note reproducing. A dropped
    connection would otherwise print STILL REPRODUCES for a run that measured
    nothing, so the client is asked a statement that certainly parses first.
    """
    class Dead:
        def query(self, statement, graph):
            raise RuntimeError("connection refused")

    assert probe._does_not_parse(Dead(), "MATCH (n) RETURN n")[0] is None

    class RefusesOnlyThis:
        def query(self, statement, graph):
            if "RETURN 1" in statement:
                return types.SimpleNamespace(records=[[1]])
            raise RuntimeError("Query error: unsupported syntax")

    assert probe._does_not_parse(RefusesOnlyThis(), "BAD CYPHER")[0] is False


def test_a_reset_that_leaves_rows_is_unsound():
    """`DETACH DELETE` succeeding and leaving rows is the case that matters."""
    class NeverEmpties:
        def query(self, statement, graph):
            if "count(" in statement:
                return types.SimpleNamespace(records=[[7]])
            return types.SimpleNamespace(records=[])

    with pytest.raises(probe.ResetFailed, match="left 7 nodes"):
        probe._reset(NeverEmpties())


def test_every_probe_is_registered_with_a_title_and_a_callable():
    """`PROBES` is what the run iterates; a typo there drops a note silently."""
    numbers = [number for number, _title, _fn in probe.PROBES]
    assert len(numbers) == len(set(numbers)), f"duplicate probe ids: {numbers}"
    assert 7 not in numbers, (
        "note 7 has no probe -- it is a property of the OSS HTTP path with "
        "nothing to isolate embedded, and the output says so"
    )
    for number, title, fn in probe.PROBES:
        assert title and callable(fn), f"probe {number} is not runnable"
