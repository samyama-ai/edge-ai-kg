"""`vector_probe`'s verdicts, driven with fakes (#56).

The probe's whole value is that it says which fault it saw. Three of its
branches were wrong in review and none of them had a test: an index-creation
panic reported as the duplicate-add reproduction, a `vector_search` failure
reported as a failed add, and a "nothing failed" tail printed over a run that
added nothing. These pin the mapping so the next edit has to keep it.

No engine needed -- `run_repro` takes its client from `samyama`, so the fakes
go in through `monkeypatch`, and `holdout_tail` is a pure function.
"""
from __future__ import annotations

import types

from benchmarks import vector_probe


class _Boom(BaseException):
    """Stands in for `PanicException`, which is not an `Exception` either."""


PanicException = type("PanicException", (BaseException,), {})


def _client(fail_on: str | None, exc: BaseException | None = None):
    """A client that works until `fail_on`, then raises `exc`."""
    def raiser():
        raise exc or _Boom("boom")

    return types.SimpleNamespace(
        query=lambda statement, graph: types.SimpleNamespace(records=[[1], [2]]),
        create_vector_index=(lambda *a, **k: raiser()) if fail_on == "index"
        else (lambda *a, **k: None),
        add_vector=(lambda *a, **k: raiser()) if fail_on == "add"
        else (lambda *a, **k: None),
        vector_search=(lambda *a, **k: raiser()) if fail_on == "search"
        else (lambda *a, **k: None),
    )


def _run(monkeypatch, fail_on, exc=None):
    client = _client(fail_on, exc)
    monkeypatch.setattr(vector_probe, "two_nodes", lambda c: ["a", "b"])
    monkeypatch.setitem(__import__("sys").modules, "samyama",
                        types.SimpleNamespace(
                            SamyamaClient=types.SimpleNamespace(
                                embedded=lambda: client)))
    return vector_probe.run_repro()


def test_an_index_panic_is_not_the_duplicate_add_reproduction(monkeypatch):
    """`index_accepted` is False, so no caller may read this as the panic.

    Both raise `PanicException`; only one of them is the finding the page
    records, and the CLI decides between them on this field.
    """
    out = _run(monkeypatch, "index", PanicException("assertion failed"))
    assert out["index_accepted"] is False
    assert out["phase"] == "index"
    assert out["outcome"] == "PanicException"


def test_a_search_failure_is_not_reported_as_a_failed_add(monkeypatch):
    """The adds and the search are separate phases, so the verdict can say which."""
    out = _run(monkeypatch, "search", RuntimeError("connection reset"))
    assert out["index_accepted"] is True
    assert out["phase"] == "search"
    assert out["outcome"] == "RuntimeError"


def test_a_clean_run_reports_ok(monkeypatch):
    """The other direction, or the two checks above could be a blanket failure."""
    out = _run(monkeypatch, None)
    assert (out["index_accepted"], out["phase"], out["outcome"]) == (True, "ok", "ok")


def test_the_tail_reports_a_failure_before_it_reports_an_empty_run():
    """A panic on the first add leaves `added == 0`; the panic is the finding.

    Ordering these the other way -- the bug this replaced -- printed "nothing
    to conclude, check the fleet was generated" over the exact reproduction
    the command exists to produce.
    """
    tail = vector_probe.holdout_tail("PanicException: assertion failed", None, 0, 0)
    assert "add_vector` failed" in tail and "Nothing to conclude" not in tail


def test_the_tail_refuses_to_conclude_when_nothing_ran():
    """"Nothing failed" is vacuously true of a run that did nothing."""
    assert "Nothing to conclude" in vector_probe.holdout_tail(None, None, 0, 0)


def test_the_clean_tail_counts_this_runs_searches():
    """The hold-out is `min(40, len(rows))`, so 40 is not a constant."""
    tail = vector_probe.holdout_tail(None, None, 12, 7)
    assert "All 7 hold-out searches" in tail
    assert "9th of its 40" in tail, "the 0.6.1 measurement is a fixed historical number"
