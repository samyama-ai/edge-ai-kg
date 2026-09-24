"""`vector_probe`'s verdicts, driven with fakes (#56).

The probe's whole value is that it says which fault it saw. Three of its
branches were wrong in review and none of them had a test: an index-creation
panic reported as the duplicate-add reproduction, a `vector_search` failure
reported as a failed add, and a "nothing failed" tail printed over a run that
added nothing. These pin the mapping so the next edit has to keep it.

Both halves are covered, because they can drift apart: `run_repro` decides
*what happened* and the CLI decides *what to call it*. A correct `phase` field
printed under the wrong sentence is the same defect to a reader, so the CLI
branches are driven through `CliRunner` with `run_repro` faked.

No engine needed -- `run_repro` takes its client from `samyama`, so the fakes
go in through `monkeypatch`, and `holdout_tail` is a pure function.
"""
from __future__ import annotations

import sys
import types

from click.testing import CliRunner

from benchmarks import vector_probe

# One stand-in, named for the thing it stands in for. `PanicException` comes
# from the Rust extension and is a `BaseException` rather than an `Exception`,
# which is why every `except` in the probe is written the way it is; a fake
# that inherited from `Exception` would not exercise that.
PanicException = type("PanicException", (BaseException,), {})


def _client(fail_on: str | None, exc: BaseException | None = None):
    """A client that works until `fail_on`, then raises `exc`."""
    def raiser():
        raise exc or PanicException("boom")

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
    monkeypatch.setitem(sys.modules, "samyama",
                        types.SimpleNamespace(
                            SamyamaClient=types.SimpleNamespace(
                                embedded=lambda: client)))
    return vector_probe.run_repro()


def _repro_output(monkeypatch, **fields):
    """`--repro`'s printed verdict for a given `run_repro` result."""
    monkeypatch.setattr(vector_probe, "run_repro",
                        lambda *a, **k: {"detail": "", **fields})
    result = CliRunner().invoke(vector_probe.main, ["--repro"])
    assert result.exit_code == 0, result.output
    return result.output


def test_an_index_panic_is_not_the_duplicate_add_reproduction(monkeypatch):
    """`index_accepted` is False, so no caller may read this as the panic.

    Both raise `PanicException`; only one of them is the finding the page
    records, and the CLI decides between them on this field.
    """
    out = _run(monkeypatch, "index", PanicException("assertion failed"))
    assert out["index_accepted"] is False
    assert out["phase"] == "index"
    assert out["outcome"] == "PanicException"


def test_a_duplicate_add_panic_is_the_reproduction_the_page_records(monkeypatch):
    """The add phase, which is the one `docs/vector-search.md` measured."""
    out = _run(monkeypatch, "add", PanicException("assertion failed"))
    assert (out["index_accepted"], out["phase"], out["outcome"]) == (
        True, "add", "PanicException")


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


def test_the_cli_calls_only_an_add_panic_the_recorded_reproduction(monkeypatch):
    """A panic is the 0.6.1 finding in the add phase and something else elsewhere.

    The field can be right while the sentence is wrong, which is what this
    catches: every panic used to print "this is the 0.6.1 failure mode",
    including one from `vector_search` -- the hold-out's finding -- and one
    from `create_vector_index`, which is neither.
    """
    added = _repro_output(monkeypatch, index_accepted=True, phase="add",
                          outcome="PanicException")
    assert "is the 0.6.1 failure mode" in added
    assert "duplicate add" in added

    searched = _repro_output(monkeypatch, index_accepted=True, phase="search",
                             outcome="PanicException")
    assert "is the 0.6.1 failure mode" not in searched
    assert "not the duplicate-add reproduction" in searched

    indexed = _repro_output(monkeypatch, index_accepted=False, phase="index",
                            outcome="PanicException")
    assert "index itself was rejected" in indexed
    assert "0.6.1 failure mode" not in indexed


def test_the_cli_header_names_the_phase_that_ran(monkeypatch):
    """It said "added twice" whatever happened, including when nothing was added."""
    assert "creating the index" in _repro_output(
        monkeypatch, index_accepted=False, phase="index", outcome="PanicException")
    assert "searching after the duplicate add" in _repro_output(
        monkeypatch, index_accepted=True, phase="search", outcome="RuntimeError")


def test_the_cli_reports_a_clean_run_without_claiming_a_version_fixed_it(monkeypatch):
    """The module refuses version claims; "fixed as of 1.7.1" is one."""
    out = _repro_output(monkeypatch, index_accepted=True, phase="ok", outcome="ok")
    assert "no panic on this build" in out
    assert "1.7.1" not in out, (
        "the clean branch must report this run, not assert what a version does")


def test_the_tail_reports_a_failure_before_it_reports_an_empty_run():
    """A panic on the first add leaves `added == 0`; the panic is the finding.

    Ordering these the other way -- the bug this replaced -- printed "nothing
    to conclude, check the fleet was generated" over the exact reproduction
    the command exists to produce.
    """
    tail = vector_probe.holdout_tail("PanicException: assertion failed", None, 0, 0)
    assert "add_vector` failed" in tail and "Nothing to conclude" not in tail


def test_the_add_failure_tail_does_not_claim_the_page_s_reproduction():
    """`docs/vector-search.md` records every add accepted and the *search* panicking.

    So an add that fails -- panic or not -- is a different finding, and saying
    "a PanicException there is the 0.6.1 failure mode" contradicted the page
    this tail sends the reader to.
    """
    tail = vector_probe.holdout_tail("PanicException: assertion failed", None, 0, 0)
    assert "every add was accepted and the search panicked" in tail
    assert "is the 0.6.1 failure mode" not in tail, (
        "the page records the search panicking after every add succeeded, so "
        "an add failure cannot be that reproduction whatever it raised")


def test_the_tail_separates_a_panicking_search_from_any_other_failure():
    """The hold-out's whole claim is that deduping does not stop the panic."""
    panicked = vector_probe.holdout_tail(
        None, "PanicException: assertion failed", 307, 8)
    assert "the search still panics" in panicked

    other = vector_probe.holdout_tail(None, "RuntimeError: connection reset", 307, 8)
    assert "not with a panic" in other
    assert "the search still panics" not in other, (
        "a dropped connection is not the HNSW assertion, and reporting it as "
        "one is the mislabelling this file exists to catch")


def test_the_tail_refuses_to_conclude_when_nothing_ran():
    """"Nothing failed" is vacuously true of a run that did nothing."""
    assert "Nothing to conclude" in vector_probe.holdout_tail(None, None, 0, 0)


def test_the_clean_tail_counts_this_runs_searches():
    """The hold-out is `min(40, len(rows))`, so 40 is not a constant."""
    tail = vector_probe.holdout_tail(None, None, 12, 7)
    assert "All 7 hold-out searches" in tail
    assert "9th" not in tail, (
        "the 0.6.1 figure belongs in docs/vector-search.md, which this tail "
        "cites; restating it here is a second copy to keep in step")
