"""The comparison harness's fairness guarantees, which are the part worth testing (#47).

The timings need two running engines and are not reproduced here. What is
tested is everything that decides whether the numbers mean anything:

**Both engines get the same indexes.** `neo4j_indexes` translates
`schema/edge_ai_kg.cypher` rather than restating it, so an index added there is
an index Neo4j gets. A comparison that silently gave one engine fewer indexes
would be measuring index availability, not engines -- and it would flatter us,
because every index in that file exists to serve one of our queries.

**Every catalog query is compared, or the run says so.** A sweep that quietly
dropped a query would drop exactly the ones that error, which are the ones a
reader most wants to see.

**A failure is a result.** `timed` returns the error rather than raising, so one
engine refusing a query does not end the run and hide the other fifteen.
"""
from __future__ import annotations

import math
import pathlib
import types

import pytest

from benchmarks import compare_neo4j, neo4j_client, sweep

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = str(ROOT / "schema" / "edge_ai_kg.cypher")


def test_every_schema_index_is_created_on_neo4j_too():
    """Compared against the repo's strict parser, not against the same regex.

    An earlier version re-declared `neo4j_indexes`'s own pattern here and
    compared the two. That cannot fail: a schema line the pattern misses -- no
    trailing `;`, leading whitespace, a two-property index -- is dropped from
    *both* sides, the counts still match, and Neo4j quietly runs the comparison
    with fewer indexes than Samyama. Which is the exact failure this test claims
    to prevent, and it flatters us.

    `declared_indexes()` is the strict one: it asserts every non-comment line in
    the schema parses, quoting any that does not.
    """
    from tests.test_schema_indexes import declared_indexes

    declared = declared_indexes()
    assert len(declared) == 22, (
        f"the schema declares {len(declared)} indexes, not 22. If that is "
        f"deliberate, update this number and the module docstring, which "
        f"publishes it as a fairness condition."
    )

    translated = compare_neo4j.neo4j_indexes(SCHEMA)
    assert len(translated) == len(declared), (
        f"{len(declared)} indexes declared, {len(translated)} translated -- Neo4j "
        f"would run the comparison with fewer indexes than Samyama, which "
        f"flatters us."
    )
    for label, prop in declared:
        expected = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
        assert expected in translated, f"missing Neo4j index for {label}({prop})"


def test_the_translation_is_neo4j_syntax_not_samyama_syntax():
    """`CREATE INDEX ON :L(p)` is Samyama's spelling and Neo4j 5 rejects it."""
    for statement in compare_neo4j.neo4j_indexes(SCHEMA):
        assert statement.startswith("CREATE INDEX IF NOT EXISTS FOR ("), statement
        assert " ON :" not in statement, f"Samyama spelling leaked through: {statement}"


def test_a_query_that_fails_on_one_engine_is_reported_not_raised():
    """One engine refusing a query must not end the run.

    Otherwise the first failure hides every result after it -- and the queries
    that fail are the ones #47 most wants published.
    """
    def always_fails(_cypher):
        raise RuntimeError("Query error: Type error: size() requires string")

    elapsed, count, rows = sweep.timed(always_fails, "MATCH (n) RETURN n",
                                               repeats=3)
    assert math.isnan(elapsed)
    assert isinstance(count, str) and count.startswith("ERR(warm-up)"), (
        f"a query refused outright fails in warm-up, and the table should say "
        f"so rather than leaving it indistinguishable from a timeout partway "
        f"through the timed runs. Got {count!r}"
    )
    assert "size()" in count, "the reason must survive into the table"
    assert rows == []


def test_timing_discards_warm_up_runs_and_reports_a_median():
    """A single reading on a JVM is not a measurement -- and nor is a single warm-up.

    The default is ten because one produced a wrong headline: a Neo4j that had
    been serving queries for an hour measured `EA01` at 6.0 ms, a freshly
    started one at 12.1 ms, and "2.6x faster" became "1.6x faster" on that alone.
    """
    calls = []

    def counting(_cypher):
        calls.append(1)
        return [("row",)]

    _elapsed, count, _rows = sweep.timed(counting, "MATCH (n) RETURN n",
                                                 repeats=5, warmup=10)
    assert count == 1
    assert len(calls) == 15, (
        f"expected 10 warm-up + 5 timed runs, got {len(calls)}. If warm-up runs "
        f"stop being discarded, Neo4j pays JIT costs that Samyama does not."
    )


def test_the_default_warm_up_is_enough_for_a_jvm():
    """Pinned so it cannot quietly drift back to one.

    This is a methodology constant, not a preference: the number exists because
    a smaller one changed the published conclusion.
    """
    import inspect

    # Both defaults, and neither may be 1. The first version of this test
    # asserted `>= 1` against a default that WAS 1 -- vacuously true, and
    # directly contradicting the docstring above it. A caller reaching `timed`
    # without going through the CLI would have got a single warm-up.
    function_default = inspect.signature(sweep.timed).parameters["warmup"].default
    assert function_default >= 10, (
        f"`timed(warmup=...)` defaults to {function_default}. Anything calling "
        f"it directly would then measure a barely-warmed JVM."
    )
    option = next(p for p in compare_neo4j.main.params if p.name == "warmup")
    assert option.default >= 10, (
        f"--warmup defaults to {option.default}; one warm-up moved the headline "
        f"ratio from 2.6x to 1.6x, so this must not shrink without a measurement "
        f"saying it can."
    )


def test_a_failure_partway_through_the_timed_runs_is_also_reported():
    """Not just the warm-up. A query can survive one call and die on the third.

    `timed` guarded only the warm-up once, so a mid-loop failure propagated out
    of the sweep and took every remaining query with it -- the opposite of what
    the test above claims the harness does.
    """
    calls = []

    warmup = 2

    def dies_during_the_timed_runs(_cypher):
        calls.append(1)
        # Survive every warm-up, then die on the second *timed* run. An earlier
        # version raised on call 3 with the default warmup of 10, so it died
        # during warm-up and never reached the loop it claimed to cover.
        if len(calls) > warmup + 1:
            raise RuntimeError("Neo4j.ClientError.Transaction.TransactionTimedOut")
        return [("a",)]

    elapsed, count, rows = sweep.timed(dies_during_the_timed_runs,
                                               "MATCH (n) RETURN n", repeats=5,
                                               warmup=warmup)
    assert len(calls) == warmup + 2, (
        f"expected {warmup} warm-ups, one timed run, then the failure; got "
        f"{len(calls)} calls. If this drops to {warmup + 1} the failure landed "
        f"in warm-up again and the timed loop is untested."
    )
    assert math.isnan(elapsed)
    assert isinstance(count, str) and count.startswith("ERR(timed)"), (
        f"this failure landed in the timed loop, not warm-up, and the two are "
        f"different problems. Got {count!r}"
    )
    assert "Neo4j.ClientError" in count, "the reason must survive into the table"
    assert rows == []


def test_rows_are_returned_so_content_can_be_compared_not_just_counts():
    """Two engines returning the same number of different rows are not equal.

    That exact failure hid on this repo before: seven catalog queries matched on
    count and differed on content between two builds, and a length comparison
    reported no disagreement. The harness compares content for the same reason.
    """
    def two_rows(_cypher):
        return [("a", 1), ("b", 2)]

    _elapsed, count, rows = sweep.timed(two_rows, "MATCH (n) RETURN n",
                                        repeats=2)
    assert count == 2
    # The property, not the representation: rows come back comparable, and
    # comparing them is not the same as comparing their length. Asserting the
    # exact tuples pinned `str(v)`, which was the bug -- `1` and `1.0` compared
    # unequal and `None` compared equal to the text "None".
    assert rows == sweep.normalised([("a", 1), ("b", 2)])
    assert rows != sweep.normalised([("a", 1), ("z", 2)]), (
        "two rows of different content must not compare equal, or the harness "
        "can only compare lengths"
    )
    assert rows == sweep.normalised([("a", 1.0), ("b", 2.0)]), (
        "and an integer against the same value as a float is the same answer, "
        "not a disagreement between engines"
    )


def test_every_catalog_query_is_asked_of_both_engines():
    """Driven, not grepped.

    An earlier version asserted `"BY_ID.items()"` appeared in `main`'s source --
    which passes on a comment and fails on a rename, and says nothing about what
    reached either engine. `compare_catalog` is separated out so it can be run
    with fakes that record every query they are asked.
    """
    from benchmarks.queries import BY_ID

    def samyama(_cypher):
        return [("row",)]

    def neo4j(_cypher):
        return [("row",)]

    out = sweep.compare_catalog(samyama, neo4j, BY_ID, repeats=1,
                                        warmup=1, echo=lambda *a, **k: None)
    for side in ("samyama", "neo4j"):
        asked = set(out["asked"][side])
        missing = set(BY_ID) - asked
        assert not missing, f"{side} was never asked {sorted(missing)}"
    assert len(BY_ID) >= 16, f"catalog unexpectedly small: {sorted(BY_ID)}"


def test_a_query_failing_on_one_engine_still_leaves_the_rest_compared():
    """The sweep must not stop at the first refusal.

    A query one engine refuses is not hypothetical -- a variable-length
    traversal raises on the 1.7.0 server (engine note 12) -- and if that ended
    the run, every result after it would vanish.
    """
    from benchmarks.queries import BY_ID

    def samyama(_cypher):
        return [("row",)]

    refused = min(BY_ID)            # any one query, refused by one engine

    def neo4j(cypher):
        if cypher == BY_ID[refused]["cypher"]:
            raise RuntimeError("Type error: size() requires string, list, or path")
        return [("row",)]

    out = sweep.compare_catalog(samyama, neo4j, BY_ID, repeats=1,
                                        warmup=1, echo=lambda *a, **k: None)
    counted = (len(out["wins"]) + len(out["losses"]) + len(out["parity"])
               + len(out["errors"]))
    assert counted == len(BY_ID), (
        f"{counted} of {len(BY_ID)} queries reached a verdict; a refusal ended "
        f"the sweep early"
    )


def test_a_near_parity_result_is_not_reported_as_a_winner():
    """The band exists because two runs of this command disagreed.

    EA13 measured 1.21x then 0.87x, EA15 1.07x then 0.81x -- pure run-to-run
    movement that a binary win/loss turns into a published result. Anything
    inside the band is reported as too close to call, which is a finding rather
    than a verdict.
    """
    band = sweep.PARITY_BAND
    assert 0 < band < 1, band
    for ratio in (1.0, 1.21, 0.87, 1.07, 0.81):
        assert 1 / (1 + band) <= ratio <= 1 + band, (
            f"{ratio}x should fall inside the parity band; it is one of the "
            f"readings that flipped between runs"
        )
    for ratio in (0.46, 0.49, 0.55, 10.0, 34.5):
        assert not (1 / (1 + band) <= ratio <= 1 + band), (
            f"{ratio}x is a real difference and must not be called parity"
        )


def test_the_index_parser_refuses_to_return_nothing(tmp_path):
    """An empty parse would benchmark an unindexed Neo4j and say nothing.

    `load_neo4j` creates whatever this returns and `--reuse-neo4j` verifies
    whatever this returns, so `[]` makes both no-ops: Neo4j runs unindexed
    against an indexed Samyama, and the guard reports success. That is exactly
    the defect this module was corrected for, reintroduced through a parser that
    can quietly find nothing.
    """
    changed = tmp_path / "edge_ai_kg.cypher"
    changed.write_text("CREATE INDEX FOR (n:Board) ON (n.id);\n", encoding="utf-8")
    with pytest.raises(neo4j_client.Neo4jLoadError) as raised:
        compare_neo4j.neo4j_indexes(str(changed))
    assert "no `CREATE INDEX" in str(raised.value)
    assert "unindexed" in str(raised.value), (
        "the message should say what continuing would cost, not just that the "
        "parse failed"
    )


def test_a_failed_timing_is_reported_as_a_failure_not_a_measurement():
    """The guard that stops `--natural` rating a run that errored.

    An earlier version of this test asserted `math.isnan(float("nan"))` and
    `assert float("nan")` -- true of Python, true without this repo existing, and
    it never touched `compare_neo4j`. It documented the *reason* for a guard
    while testing none of it.

    What matters is that `timed` returns NaN rather than a number, so the caller
    has something to test. NaN being truthy is why the caller must use `isnan`
    and not `if elapsed:` -- which is what the old guard did, and why a failed
    timing got a printed ratio.
    """
    def always_fails(_cypher):
        raise RuntimeError("Neo.ClientError.Statement.SyntaxError")

    elapsed, count, rows = sweep.timed(always_fails, "MATCH (n) RETURN n",
                                               repeats=2, warmup=1)
    assert math.isnan(elapsed), (
        f"a failed timing must come back as NaN so the caller can detect it; "
        f"got {elapsed!r}"
    )
    assert bool(elapsed), (
        "NaN is truthy, so `if elapsed:` passes for a failure -- the caller has "
        "to use `math.isnan`. This assertion exists to make that trap explicit "
        "rather than leaving it in a comment."
    )
    assert isinstance(count, str) and count.startswith("ERR")
    assert rows == []


def test_a_query_below_timer_resolution_is_not_a_win():
    """A zero median is a failure to measure, not an infinite speedup.

    `n_ms / s_ms if s_ms else float("inf")` put `inf` through `ratio > 1.25`,
    so the one query the timer could not resolve became the largest win in the
    table -- and the win count, which is the published headline, included it.
    """
    from benchmarks.queries import BY_ID

    # `timed` is replaced rather than driven with a fast fake: `perf_counter`
    # resolves to ~0.1 us, so even a `lambda: [("row",)]` measures non-zero and
    # the case never arises. Forcing the zero is the only way to assert what
    # happens at it.
    only = next(iter(BY_ID))
    real = sweep.timed
    sweep.timed = lambda run, cypher, *a, **k: (
        0.0 if run(cypher) == [("samyama",)] else 5.0, 1, [("r",)])
    try:
        out = sweep.compare_catalog(
            lambda _c: [("samyama",)], lambda _c: [("neo4j",)],
            dict(list(BY_ID.items())[:1]),
            repeats=1, warmup=0, echo=lambda *a, **k: None)
    finally:
        sweep.timed = real
    assert out["unresolved"] == [only], (
        f"a 0.0 ms median must be bucketed as unresolved, not divided by; "
        f"got {out}"
    )
    assert only not in out["wins"] and only not in out["losses"], (
        "an unmeasurable query must not count as either"
    )


def test_a_query_that_answers_differently_between_repeats_is_not_published():
    """The returned rows are one run's; the repeats have to agree on them.

    `timed` returns the rows from the last timed run, which was taken on trust.
    A query answering differently between repeats -- an unstable `LIMIT` over
    ties, a concurrent write -- would have had one arbitrary run published as
    "the" result and compared against the other engine's, and the row-content
    check downstream would compare against whichever came last.
    """
    calls = []

    def shrinking(_cypher):
        calls.append(1)
        return [("row",)] * (3 if len(calls) <= 2 else 1)

    _ms, count, rows = sweep.timed(shrinking, "MATCH (n) RETURN n",
                                           repeats=3, warmup=1)
    assert isinstance(count, str) and count.startswith("UNSTABLE"), (
        f"a query returning 3 rows then 1 must be reported unstable, not have "
        f"one of them published; got {count!r}"
    )
    assert rows == [], "an unstable result has no rows worth comparing"

    def steady(_cypher):
        return [("row",)] * 3

    _ms, count, rows = sweep.timed(steady, "MATCH (n) RETURN n",
                                           repeats=3, warmup=1)
    assert count == 3 and len(rows) == 3, (
        f"a stable query must still report normally, or this guard has "
        f"replaced the measurement; got {count!r}"
    )


def test_same_count_different_rows_gets_no_verdict():
    """Two engines answering differently is not a race one of them wins.

    `compare_catalog` bucketed the query into `content` **and** into
    wins/losses/parity, because the ratio was computed first. So a query where
    the two engines returned the same number of *different* rows was published
    with a speed verdict — timing two different answers against each other, the
    same mistake a row-count mismatch is already bucketed to avoid.
    """
    from benchmarks.queries import BY_ID

    one = dict(list(BY_ID.items())[:1])
    qid = next(iter(one))
    real = sweep.timed
    sweep.timed = lambda run, cypher, *a, **k: (
        (1.0, 2, [("a",), ("b",)]) if run(cypher) == "samyama"
        else (9.0, 2, [("c",), ("d",)]))
    try:
        out = sweep.compare_catalog(lambda _c: "samyama", lambda _c: "neo4j",
                                    one, repeats=1, warmup=0,
                                    echo=lambda *a, **k: None)
    finally:
        sweep.timed = real

    assert out["content"] == [qid], out
    assert qid not in out["wins"] and qid not in out["losses"] \
        and qid not in out["parity"], (
        f"a content mismatch must not also carry a speed verdict; got {out}"
    )


def test_stability_compares_answers_not_row_counts():
    """The docstring says "answering differently"; the check said "how many".

    Two runs returning the same number of different rows — an unstable `LIMIT`
    over ties, which is the commonest way this happens — passed a `len(rows)`
    comparison. That is exactly the case the row-content checks elsewhere in
    this module exist for.
    """
    calls = []

    def same_count_different_rows(_cypher):
        calls.append(1)
        return [("a",), ("b",)] if len(calls) <= 2 else [("a",), ("z",)]

    _ms, count, rows = sweep.timed(same_count_different_rows,
                                   "MATCH (n) RETURN n", repeats=3, warmup=1)
    assert isinstance(count, str) and count.startswith("UNSTABLE"), (
        f"two rows both times, different content the third: must be unstable, "
        f"got {count!r}"
    )
    assert "different content" in count, (
        f"and the message should say which kind of instability it was; "
        f"got {count!r}"
    )
    assert rows == []


def test_a_short_samyama_load_stops_the_run():
    """The Samyama side is counted too, as `load_neo4j` counts its own.

    A load that dropped rows would otherwise be timed: every query below would
    be fast for the wrong reason, and nothing would say so.
    """
    class DropsEverything:
        def query(self, statement, graph):
            return types.SimpleNamespace(records=[[0]])

    fleet = types.SimpleNamespace(nodes={"Board": [{"id": "b1"}]}, edges=[])
    with pytest.raises(SystemExit, match="expected 1"):
        compare_neo4j.load_samyama(DropsEverything(), fleet)
