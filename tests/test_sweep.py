"""`benchmarks/sweep.py`'s row comparison and buckets, driven with fakes (#47).

Separate from `tests/test_neo4j_comparison.py`, which is near the 500-line limit
the review harness skips a file at. Everything here runs without an engine.
"""
from __future__ import annotations

from benchmarks import sweep


def test_floats_agree_relatively_at_any_magnitude():
    """The tolerance is relative, so it holds at 1e12 as well as at 1.

    The first version rounded to multiples of an absolute 1e-9. At 1e12 that is
    finer than a double resolves, so two sums differing in the last bit landed
    in different buckets and read as a disagreement.
    """
    big = 1e12
    nudged = big + big * 1e-15          # a few ulps: summation order, not a disagreement
    assert big != nudged
    assert sweep.same_rows(sweep.normalised([(big,)]), sweep.normalised([(nudged,)]))
    assert not sweep.same_rows(sweep.normalised([(big,)]),
                               sweep.normalised([(big * 1.001,)])), (
        "a real difference at large magnitude must still be one"
    )


def test_floats_either_side_of_a_rounding_midpoint_agree():
    """Bucketing always has a midpoint; comparing does not.

    `0.5e-9` and a value one ulp below it rounded to different multiples of
    1e-9, so two values closer than any tolerance compared unequal.
    """
    below, above = 0.4999999999e-9, 0.5000000001e-9
    assert round(below / 1e-9) != round(above / 1e-9), "not a midpoint any more"
    assert sweep.same_rows(sweep.normalised([(below,)]), sweep.normalised([(above,)]))


def test_nonfinite_values_sort_alongside_numbers():
    """`("number", "inf")` against `("number", 1.0)` compared str with float."""
    rows = sweep.normalised([(float("inf"),), (1.0,), (None,)])
    assert sweep.same_rows(rows, sweep.normalised([(1,), (None,), (float("inf"),)]))


def test_errors_and_unstable_are_disjoint():
    """One bucket per query, so no caller has to subtract one from the other.

    `unstable` used to be a subset of `errors`, and a caller that forgot to
    subtract counted an unstable query twice.
    """
    catalog = {"Q1": {"cypher": "unstable"}, "Q2": {"cypher": "refused"},
               "Q3": {"cypher": "fine"}}
    calls = []

    def samyama(cypher):
        calls.append(1)
        if cypher == "unstable":
            return [("row",)] * (3 if len(calls) % 2 else 1)
        if cypher == "refused":
            raise RuntimeError("SyntaxError")
        return [("row",)]

    def neo4j(_cypher):
        return [("row",)]

    out = sweep.compare_catalog(samyama, neo4j, catalog, repeats=3, warmup=1,
                                echo=lambda *a, **k: None)
    assert out["unstable"] == ["Q1"], out
    assert out["errors"] == ["Q2"], out
    assert not set(out["errors"]) & set(out["unstable"])


def test_a_query_never_asked_is_reported_by_the_sweep_itself(monkeypatch):
    """`asked` is checked where it is filled, not left to each caller.

    `timed` records a query only when its callable runs. One that returns
    without calling it -- the wiring mistake this guards against -- leaves the
    query with a fabricated row in the table and nothing in any bucket saying
    it was never put to that engine.
    """
    real_timed = sweep.timed

    def never_runs_q2(run, cypher, repeats, warmup=10):
        if cypher == "b":
            return 1.0, 1, sweep.normalised([("r",)])   # a result, `run` never called
        return real_timed(run, cypher, repeats, warmup)

    monkeypatch.setattr(sweep, "timed", never_runs_q2)
    catalog = {"Q1": {"cypher": "a"}, "Q2": {"cypher": "b"}}
    out = sweep.compare_catalog(lambda c: [("r",)], lambda c: [("r",)], catalog,
                                repeats=1, warmup=0, echo=lambda *a, **k: None)
    assert out["not_asked"] == {"samyama": ["Q2"], "neo4j": ["Q2"]}, out["not_asked"]


def _same(a, b):
    return sweep.same_rows(sweep.normalised(a), sweep.normalised(b))


def test_null_and_the_text_none_differ():
    """The reason `cell` exists: `str()` made these the same answer."""
    assert not _same([(None,)], [("None",)])
    assert _same([(None,)], [(None,)])


def test_a_bool_is_not_the_number_it_equals():
    """`True == 1` in Python; as answers they are different types."""
    assert not _same([(True,)], [(1,)])
    assert not _same([(False,)], [(0.0,)])


def test_nan_is_its_own_kind_and_agrees_with_itself():
    """`nan != nan`, so without its own kind a NaN never matched anything."""
    nan = float("nan")
    assert _same([(nan,), (1.0,)], [(1,), (nan,)])
    assert not _same([(nan,)], [(0.0,)])


def test_integers_are_compared_exactly():
    """Through `float()` and a relative tolerance, 10**9 equalled 10**9 + 1."""
    assert not _same([(10**9,)], [(10**9 + 1,)])
    assert _same([(3,)], [(3.0,)]), "1 and 1.0 are still the same answer"


def test_rows_within_tolerance_match_even_when_they_sort_apart():
    """Sort-position pairing broke on keys a few ulps apart.

    `0.1 + 0.2` sorts after `0.30000000000000001`-ish neighbours that `0.3`
    sorts before, so the same answer paired the wrong rows and read as two
    engines disagreeing.
    """
    left = [(0.1 + 0.2, "a"), (0.3, "b")]
    right = [(0.3, "a"), (0.1 + 0.2, "b")]
    assert sweep.normalised(left) != sweep.normalised(right)
    assert _same(left, right)
    assert not _same(left, [(0.3, "a"), (0.3, "c")])


def _quiet(**kw):
    return sweep.compare_catalog(echo=lambda *a, **k: None, **kw)


def test_an_error_on_one_side_outranks_instability_on_the_other():
    """A refusal is the stronger finding, so the query is `errors`, not `unstable`."""
    flips = iter(range(10**6))

    def unstable(_c):
        return [("r",)] * (1 + next(flips) % 2)

    def refuses(_c):
        raise RuntimeError("SyntaxError")

    for s, n in ((unstable, refuses), (refuses, unstable)):
        out = _quiet(samyama_run=s, neo4j_run=n, catalog={"Q": {"cypher": "q"}},
                     repeats=3, warmup=0)
        assert out["errors"] == ["Q"] and out["unstable"] == [], out


def test_two_empty_answers_get_no_verdict():
    """Timing two empty results against each other is not a win for either."""
    out = _quiet(samyama_run=lambda c: [], neo4j_run=lambda c: [],
                 catalog={"Q": {"cypher": "q"}}, repeats=3, warmup=0)
    assert out["empty"] == ["Q"], out
    assert not (out["wins"] or out["losses"] or out["parity"]), out


def test_a_list_is_compared_by_its_elements_not_its_repr():
    """`EA03` returns `collect(op.name)`; a repr comparison reads it as text."""
    assert _same([(["a", "b"],)], [(("a", "b"),)]), "list vs tuple is not an answer"
    assert _same([([1.0, 2],)], [([1, 2.0],)]), "1 and 1.0 inside a list, as outside"
    assert not _same([(["a", "b"],)], [(["b", "a"],)]), (
        "collect() order is part of the answer until the query sorts it")
    assert not _same([(["a"],)], [(["a", "b"],)])


def test_a_map_ignores_key_order_and_keeps_its_values():
    """Map keys are unordered; the values still get the number tolerance."""
    assert _same([({"a": 1, "b": 2.0},)], [({"b": 2, "a": 1.0},)])
    assert not _same([({"a": 1},)], [({"a": 2},)])
    assert not _same([({"a": 1},)], [({"a": 1, "b": 1},)])


def test_timed_returns_the_rows_it_already_normalised():
    """The third value is the last timed run's rows, not a second pass over them."""
    ms, count, rows = sweep.timed(lambda c: [(1, "x")], "q", repeats=2, warmup=0)
    assert count == 1 and rows == sweep.normalised([(1, "x")])
    assert ms >= 0
