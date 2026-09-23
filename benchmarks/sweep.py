"""Timing a catalog on two engines, and refusing to call a non-measurement a result.

Split from `benchmarks/compare_neo4j.py`, which crossed the 500-line limit the
review harness skips a file at -- and a skipped file is an unreviewed one. This
half knows nothing about Neo4j: it takes two callables that run Cypher and
returns buckets.

Most of what is here exists because a plausible number came out of something
that had not measured anything:

- a query answering differently between repeats had one arbitrary run
  published;
- a zero median -- the timer failing to resolve a sub-millisecond query --
  divided into `inf` and was published as the largest win in the table;
- two engines both returning nothing were scored as a win for whichever
  returned nothing faster;
- a failure partway through took every remaining query's result with it.

Each is a bucket rather than a verdict now.
"""
from __future__ import annotations

import math
import statistics
import time
from collections import Counter

# How close two floats have to be to count as the same answer. Neo4j and
# Samyama compute the same aggregate in different orders, so a sum or an
# average can differ in the last bits -- `EA10` differed in the sixteenth
# significant digit and was reported as two engines disagreeing.
#
# Relative, and applied by comparison rather than by rounding. The first
# version rounded every float to a multiple of an absolute 1e-9, which was
# wrong at both ends: at 1e12 that is finer than a double resolves, so values a
# few ulps apart landed in different buckets; and two values either side of a
# rounding midpoint differed by one bucket however close they were. Any
# bucketing has that midpoint, so nothing is bucketed.
REL_TOLERANCE = 1e-9
ABS_TOLERANCE = 1e-12   # only for values at or near zero, where relative fails


def cell(value):
    """One result value, typed so equality means "the same answer".

    `str(v)` was neither necessary nor sufficient, and wrong in both
    directions:

    - `1` against `1.0` stringified to `"1"` and `"1.0"` and was reported as a
      content mismatch. The engines type integers differently and that is not a
      disagreement about the answer.
    - `None` against the string `"None"` stringified to the same thing and was
      reported as agreement. That one is a real difference and the more
      dangerous miss.

    A `(kind, value)` pair keeps null apart from the text "None". Numbers are
    kept exact here and given their tolerance in `same_rows`.

    Collections are typed through, not stringified. `EA03` returns
    `collect(op.name)`, and a list compared by `str()` reads two engines'
    *ordering* of the same names as a different answer -- the repr is the
    order. A list keeps its order (it is what `collect` returned) and a map
    does not (its keys are unordered), while both get their elements' tolerance
    rather than their repr.
    """
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        # Its own kind, above the number branch that would otherwise take it:
        # `True == 1` in Python, and "1" is not the answer "true".
        return ("bool", value)
    if isinstance(value, (list, tuple)):
        return ("list", tuple(cell(v) for v in value))
    if isinstance(value, dict):
        return ("map", tuple(sorted((str(k), cell(v)) for k, v in value.items())))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            # Its own kind: `("number", "inf")` sorted against
            # `("number", 1.0)` compared a str with a float and raised.
            return ("nonfinite", str(value))
        # One kind for both: `1` and `1.0` are the same answer. The value keeps
        # its type, so two integers are compared exactly in `_same_cell` --
        # through `float()` and a relative tolerance, 10**9 equalled 10**9 + 1.
        return ("number", value)
    return ("text", str(value))


def normalised(rows):
    """Rows as comparable tuples, order-insensitive within the row set."""
    return sorted(tuple(cell(v) for v in row) for row in rows)


def _same_cell(a, b) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == "list":
        return len(a[1]) == len(b[1]) and all(
            _same_cell(x, y) for x, y in zip(a[1], b[1]))
    if a[0] == "map":
        return len(a[1]) == len(b[1]) and all(
            ka == kb and _same_cell(va, vb)
            for (ka, va), (kb, vb) in zip(a[1], b[1]))
    if a[0] == "number":
        if isinstance(a[1], int) and isinstance(b[1], int):
            return a[1] == b[1]
        return math.isclose(a[1], b[1], rel_tol=REL_TOLERANCE,
                            abs_tol=ABS_TOLERANCE)
    return a == b


def _same_row(ra, rb) -> bool:
    return len(ra) == len(rb) and all(_same_cell(x, y) for x, y in zip(ra, rb))


def same_rows(a, b) -> bool:
    """Two `normalised` row sets are the same answer, floats within tolerance.

    Matched as multisets, not paired by sort position. Position pairing broke
    when two rows' float keys sat within a few ulps of each other: `0.1 + 0.2`
    and `0.3` sort in opposite orders against a neighbouring row, so two
    engines giving the same answer were reported as disagreeing.

    Two exact passes come first, and on identical answers -- which is every
    agreeing query, and this runs once per repeat inside `timed` -- they are
    all that runs. Only rows with no exact counterpart reach the pairwise
    search, so its quadratic cost is paid over the handful of rows that
    actually differ in their last bits rather than over the whole result.
    """
    if len(a) != len(b):
        return False
    if a == b:                                   # identical, the common case
        return True
    counts = Counter(b)
    left = []
    for ra in a:
        if counts[ra]:
            counts[ra] -= 1                      # an exact twin, wherever it sorted
        else:
            left.append(ra)
    spare = list(counts.elements())
    if len(left) != len(spare):
        return False
    for ra in left:
        hit = next((i for i, rb in enumerate(spare) if _same_row(ra, rb)), None)
        if hit is None:
            return False
        spare.pop(hit)
    return True

# Anything inside +/-25% is reported as parity rather than a win or a loss.
# Chosen from observed run-to-run movement, not from taste: in the two runs
# `docs/neo4j-comparison.md` publishes, EA04 moved 1.42x -> 0.94x on noise alone.
# A binary verdict at that sample size manufactures a winner.
PARITY_BAND = 0.25


def timed(run, cypher: str, repeats: int, warmup: int = 10) -> tuple[float, int | str, list]:
    """Median of `repeats`, after `warmup` discarded runs. Failures are a result.

    Returns the rows as well as their count. Comparing lengths alone calls two
    engines equal when they return the same number of *different* rows -- which
    is how a real difference hid on this repo once already: seven catalog
    queries matched on count and differed on content between two builds, and a
    length comparison reported no disagreement.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}; "
                         f"a median of nothing is not a measurement")
    samples = []
    phase = "warm-up"   # narrow labels: the reason needs the column width
    try:
        for _ in range(warmup):                  # discarded
            rows = run(cypher)
        phase = "timed"
        seen = []
        for _ in range(repeats):
            start = time.perf_counter()
            rows = run(cypher)
            samples.append((time.perf_counter() - start) * 1000)
            # The rows, not `len(rows)`. The docstring below says "answering
            # differently between repeats", and a count comparison does not see
            # that: two runs returning the same number of *different* rows --
            # an unstable `LIMIT` over ties, which is the commonest way this
            # happens -- passed it. Sorted, because row order is not the claim.
            seen.append(normalised(rows))
        # The returned rows are the last timed run's, and that was taken on
        # trust. A query answering differently between repeats -- an unstable
        # `LIMIT` over ties, a concurrent write -- would have one arbitrary run
        # published as "the" result and compared against the other engine.
        # Compared with the same tolerance as between engines: a parallel sum
        # that differs in its last bits from run to run is not instability.
        if not all(same_rows(seen[0], other) for other in seen[1:]):
            sizes = sorted({len(rows) for rows in seen})
            shape = (f"{sizes[0]}-{sizes[-1]} rows" if len(sizes) > 1
                     else f"{sizes[0]} rows, different content")
            return statistics.median(samples), f"UNSTABLE {shape}", []
    except Exception as exc:
        # The whole loop, not just the warm-up. A query that survives the warm-up
        # and then dies on run 3 -- a transaction timeout, memory pressure --
        # would otherwise propagate out of the sweep and take every remaining
        # query's result with it, which is exactly what this return value exists
        # to prevent.
        # Which phase matters when reading the table: a query refused outright
        # fails in warm-up, while one that dies partway through the timed runs
        # is a different problem -- a timeout, memory pressure -- and would
        # otherwise be indistinguishable.
        # `splitlines()[0]` raised IndexError on an exception with an empty
        # message -- inside the handler whose whole job is to stop one failure
        # killing the sweep.
        reason = (str(exc).splitlines() or [type(exc).__name__])[0]
        return float("nan"), f"ERR({phase}) {reason[:34]}", []
    # `seen[-1]`, not `normalised(rows)` again: the last timed run's rows were
    # normalised into `seen` above, and this is the same object by definition.
    return statistics.median(samples), len(rows), seen[-1]


def compare_catalog(samyama_run, neo4j_run, catalog, repeats, warmup, echo=print):
    """Time every query in `catalog` on both engines and bucket the results.

    Separated from `main` so it can be driven with fakes. An earlier test
    asserted the sweep covered the catalog by grepping `main`'s source for
    `"BY_ID.items()"` -- which passes on a comment and fails on a rename, and
    checks nothing about what was actually asked of either engine.
    """
    out = {"wins": [], "losses": [], "parity": [], "errors": [], "content": [],
           "counts": [], "unresolved": [], "empty": [], "unstable": [],
           "asked": {"samyama": set(), "neo4j": set()}}

    def ms(value):
        """Three decimals under 1 ms, one above.

        At 0.4 ms a single decimal is too coarse to reproduce the ratio from the
        printed figures -- `EA14`'s published 0.4 and 8.3 ms suggest 20.8x
        where the unrounded medians gave 18.99x -- and a published number a
        reader cannot check is one they have to trust.
        """
        return f"{value:.3f}" if value < 1 else f"{value:.1f}"

    def record(run, side, qid, statement):
        # A set: this is called once per warm-up *and* once per timed run, so a
        # list came back (warmup + repeats) times longer than the catalog and
        # could not be compared against its keys without deduplicating first.
        out["asked"][side].add(qid)
        return run(statement)

    for qid, spec in catalog.items():
        cypher = spec["cypher"]

        s_ms, s_n, s_rows = timed(
            lambda c, q=qid: record(samyama_run, "samyama", q, c),
            cypher, repeats, warmup)
        n_ms, n_n, n_rows = timed(
            lambda c, q=qid: record(neo4j_run, "neo4j", q, c),
            cypher, repeats, warmup)
        if isinstance(s_n, str) or isinstance(n_n, str):
            # One bucket each, never both. `unstable` used to be a subset of
            # `errors`, so every caller had to subtract one from the other to
            # count refusals -- and a caller that did not counted an unstable
            # query twice. A query that errored on one side is `errors` even if
            # the other side was unstable: a refusal is the stronger finding.
            errored = any(str(n).startswith("ERR") for n in (s_n, n_n))
            out["errors" if errored else "unstable"].append(qid)
            echo(f"{qid:<8}{s_n!s:>12}{n_n!s:>12}{'':>10}")
            continue
        if s_n != n_n:
            # Bucketed separately, not as a win or a loss. The two engines are
            # not answering the same question, so whichever is faster is faster
            # at something else.
            out["counts"].append(qid)
            echo(f"{qid:<8}{ms(s_ms):>10}ms{ms(n_ms):>10}ms{'':>10}"
                 f"   ROW COUNTS DIFFER {s_n} vs {n_n}")
            continue
        if not s_ms or not n_ms:
            # A zero median means the timer could not resolve the query, not
            # that it took no time. `n_ms / 0` gave `inf`, which sailed past
            # `ratio > 1.25` and was published as a win -- the largest win in
            # the table, from the one measurement that failed to measure.
            out["unresolved"].append(qid)
            echo(f"{qid:<8}{ms(s_ms):>10}ms{ms(n_ms):>10}ms{'':>10}"
                 f"   BELOW TIMER RESOLUTION, no ratio")
            continue
        if not s_n:
            # Both engines returned nothing, so this times two empty results
            # against each other. `--natural` already refuses to print a ratio
            # over an empty answer -- "a ratio over an empty result says
            # nothing about either" -- and the sweep was doing exactly that,
            # handing a win or a loss to whichever engine found nothing faster.
            out["empty"].append(qid)
            echo(f"{qid:<8}{ms(s_ms):>10}ms{ms(n_ms):>10}ms{'':>10}"
                 f"   BOTH EMPTY, no verdict")
            continue
        if not same_rows(s_rows, n_rows):
            # Before the verdict, not after it. The two engines returned the
            # same *number* of different rows, so whichever is faster is faster
            # at answering something else -- the same reason a row-count
            # mismatch is bucketed rather than scored. It was recorded in
            # `content` and given a win or a loss anyway.
            out["content"].append(qid)
            echo(f"{qid:<8}{ms(s_ms):>10}ms{ms(n_ms):>10}ms{'':>10}"
                 f"   {s_n}   SAME COUNT, DIFFERENT ROWS -- no verdict")
            continue
        ratio = n_ms / s_ms
        # Three buckets, not two. Between the two published runs EA04 moved
        # 1.42x -> 0.94x purely on noise, so a binary
        # win/loss over a 5-sample median invents a result for anything near
        # parity. PARITY is a finding: "too close to call at this sample size".
        if ratio > 1 + PARITY_BAND:
            out["wins"].append(qid)
        elif ratio < 1 / (1 + PARITY_BAND):
            out["losses"].append(qid)
        else:
            out["parity"].append(qid)
        echo(f"{qid:<8}{ms(s_ms):>10}ms{ms(n_ms):>10}ms{ratio:>9.2f}x"
             f"   {s_n}")
    # Checked here, where `asked` is filled, rather than left to each caller.
    # `timed` records a call only when its callable runs, so a lambda that was
    # never invoked -- a wiring mistake, or a sweep cut short -- would leave a
    # query with no row in the table and no error either. Every query that
    # reached no verdict is otherwise in some bucket; these are in none.
    out["not_asked"] = {side: sorted(set(catalog) - out["asked"][side])
                        for side in ("samyama", "neo4j")}
    return out
