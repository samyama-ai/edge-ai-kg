"""`NATURAL_EA01` and the guard that keeps it comparable with `EA01` (#47).

Split from `tests/test_neo4j_comparison.py`, which crossed the 500-line limit
the review harness skips a file at -- and a skipped file is an unreviewed one.
It mirrors the source split: `benchmarks/natural_ea01.py` holds the
hand-written hero query and the drift check, this holds their tests.

`--natural` prices our own query shapes by timing `EA01` both ways on Neo4j:
as we ship it, shaped around engine notes 3 and 5, and as a Neo4j author would
write it. That only means anything while the two ask the same question, which
is what the drift guard is for -- and what it could not previously detect.
"""
from __future__ import annotations

import math

from benchmarks import natural_ea01, sweep


def test_the_natural_hero_query_asks_the_same_question():
    """`--natural` is only meaningful if both spellings return the same answer.

    It drops the `WITH` projection and uses `NOT EXISTS` instead of
    `OPTIONAL MATCH ... count() = 0`, so it must still filter on the same model
    and accelerator and return the same three columns. The harness also compares
    the rows at runtime and refuses to print a ratio if they differ.
    """
    from benchmarks.queries import BY_ID

    natural = natural_ea01.NATURAL_EA01
    shipped = BY_ID["EA01"]["cypher"]
    for subject in ('"model:00000"', '"accel:00001"'):
        assert subject in natural, f"{subject} missing from the natural spelling"
        assert subject in shipped
    assert "NOT EXISTS" in natural, "the point is the negated pattern note 5 forbids"
    assert "OPTIONAL MATCH" not in natural
    for column in ("operator", "category", "opset"):
        assert f"AS {column}" in natural, f"natural spelling drops {column}"

def test_the_drift_guard_catches_a_retargeted_hero_query():
    """The case `natural_ea01_matches_the_catalog` exists for, and used to miss.

    It asked `if subject in shipped and subject not in NATURAL_EA01` over two
    hardcoded literals. Retarget `EA01` to `model:00042` and the first half is
    false, so nothing was reported -- and `--natural` priced two different
    questions against each other while printing a clean ratio. That is the only
    drift the guard exists to catch, so the guard was decorative.
    """
    import copy

    from benchmarks.queries import BY_ID

    assert not natural_ea01.natural_ea01_matches_the_catalog(BY_ID), (
        "the shipped pair must agree, or every case below is testing a "
        "mismatch against a mismatch"
    )

    retargeted = copy.deepcopy(BY_ID)
    retargeted["EA01"]["cypher"] = retargeted["EA01"]["cypher"].replace(
        "model:00000", "model:00042")
    problems = natural_ea01.natural_ea01_matches_the_catalog(retargeted)
    assert any("model:00042" in p for p in problems), (
        f"retargeting EA01 to a different model was not reported: {problems}"
    )

    renamed = copy.deepcopy(BY_ID)
    renamed["EA01"]["cypher"] = renamed["EA01"]["cypher"].replace(
        "RETURN operator, category, opset", "RETURN operator, category, opset AS v")
    assert any("columns differ" in p for p in
               natural_ea01.natural_ea01_matches_the_catalog(renamed))

    reordered = copy.deepcopy(BY_ID)
    reordered["EA01"]["cypher"] = reordered["EA01"]["cypher"].replace(
        "RETURN operator, category, opset", "RETURN category, operator, opset")
    assert any("columns differ" in p for p in
               natural_ea01.natural_ea01_matches_the_catalog(reordered)), (
        "same three columns in a different order is a different answer, and a "
        "set comparison would call it a match"
    )


def test_the_drift_guard_says_so_when_it_has_nothing_to_compare():
    """The branch that fires when `EA01` is parameterised, and was untested.

    Replace both literal ids with parameters and the two set differences above
    report only that NATURAL_EA01 names subjects EA01 lacks -- true, but it
    reads like an ordinary retarget. The guard has to say the stronger thing:
    it can no longer see EA01's subjects at all, so it has stopped guarding.
    """
    import copy

    from benchmarks.queries import BY_ID

    parameterised = copy.deepcopy(BY_ID)
    cypher = parameterised["EA01"]["cypher"]
    for literal, param in (('"model:00000"', "$model"), ('"accel:00001"', "$accel")):
        assert literal in cypher, f"EA01 no longer names {literal}; update this test"
        cypher = cypher.replace(literal, param)
    parameterised["EA01"]["cypher"] = cypher

    problems = natural_ea01.natural_ea01_matches_the_catalog(parameterised)
    assert any("stopped guarding" in p for p in problems), (
        f"a parameterised EA01 must be reported as unguardable, not as a "
        f"retarget: {problems}"
    )

def test_the_returned_columns_are_the_return_clause_not_every_alias():
    """Why the column check reads the `RETURN` and not every `AS`.

    `EA01` projects through two `WITH`s (engine note 3), so an `AS`-wide scan
    picks up its intermediate `count(k) AS kernels` -- a name neither spelling
    returns. That comparison reports drift on a pair that matches, which is how
    a guard gets deleted.
    """
    from benchmarks.queries import BY_ID

    shipped = BY_ID["EA01"]["cypher"]
    assert "AS kernels" in shipped, "EA01 no longer has the intermediate alias"
    assert natural_ea01.returned_columns(shipped) == ["operator", "category", "opset"]
    assert natural_ea01.returned_columns("MATCH (n) RETURN n.a, n.b AS c") == ["n.a", "c"]
    assert natural_ea01.returned_columns(
        "MATCH (n) RETURN n.a\nORDER BY n.a\nLIMIT 5") == ["n.a"]
    # Last `RETURN` wins, as the docstring promises: a subquery's inner
    # `RETURN` is not the shape the caller receives.
    assert natural_ea01.returned_columns(
        "MATCH (n) CALL { WITH n MATCH (n)--(m) RETURN count(m) AS inner }\n"
        "RETURN n.id AS id, inner") == ["id", "inner"]

def test_the_natural_path_refuses_a_ratio_over_a_non_result():
    """`--natural` has to recognise both of `timed`'s non-results, not one.

    `timed` signals a failure and an unstable query the same way: a *string*
    where the row count goes. `compare_catalog` tests for that with
    `isinstance(..., str)`; the `--natural` path tested for emptiness and NaN
    instead, and each string slipped through a different gap.

    `ERR(...)` is truthy, so the empty-rows check missed it and `isnan` caught
    it one branch later. `UNSTABLE ...` carries a **real median**, so `isnan`
    missed it too -- and when both spellings were unstable their empty row
    lists compared equal, so the row check passed and a ratio was printed over
    two measurements that had already declared themselves unusable.

    This pins the three properties that made the old guards insufficient. The
    surrounding path needs two live engines, so it is not driven here.
    """
    calls = []

    def alternating(_cypher):
        # Three rows then one, by parity. Any two consecutive calls disagree,
        # so the result does not depend on how many of them `timed` spends on
        # warm-up -- the previous fake switched after call 2 and was right only
        # for exactly one warm-up followed by timed runs.
        calls.append(1)
        return [("row",)] * (3 if len(calls) % 2 else 1)

    elapsed, count, rows = sweep.timed(alternating, "MATCH (n) RETURN n",
                                       repeats=3, warmup=1)
    assert isinstance(count, str) and count.startswith("UNSTABLE"), count
    assert not math.isnan(elapsed), (
        "the whole trap: an UNSTABLE result carries a real median, so an "
        "`isnan` guard does not see it"
    )
    assert rows == [], (
        "and no rows, so a row comparison against another unstable result "
        "finds them equal"
    )

    def failing(_cypher):
        raise RuntimeError("Neo.ClientError.Statement.SyntaxError")

    elapsed, count, _rows = sweep.timed(failing, "MATCH (n) RETURN n",
                                        repeats=2, warmup=1)
    assert isinstance(count, str) and count.startswith("ERR"), count
    # The two predicates, compared. `assert bool(count)` said nothing -- a
    # non-empty string is truthy by definition. What is worth asserting is
    # that the *old* guard misses this and the new one catches it, which is
    # the whole reason the check changed.
    old_guard_catches = not count               # `if not shaped_n or not nat_n`
    new_guard_catches = isinstance(count, str)  # what compare_catalog uses
    assert not old_guard_catches and new_guard_catches, (
        f"an ERR result must slip past the empty-rows guard and be caught by "
        f"the isinstance one; got old={old_guard_catches} "
        f"new={new_guard_catches} for {count!r}"
    )
