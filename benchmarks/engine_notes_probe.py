"""Re-run the probeable engine notes against whatever engine is installed.

Notes 1-6, 8 and 9. **Note 7 has no probe**: "the `--graph` argument is
ignored" is a property of the OSS HTTP path, and there is no tenant boundary to
ignore on an embedded build, so there is nothing here to measure. It stands
un-re-measured, and the output says so.

`docs/engine-notes.md` records eleven engine behaviours, nine of which the
loader and the query catalog work around. Those nine were measured on the
**1.7.0 HTTP server** in August 2026 and written up in prose. Prose does not
re-measure itself: #56 pinned the embedded build to `samyama` 1.7.1 and the PR
initially asserted "notes 1-9 are unaffected -- they still reproduce", which was
never checked and turned out to be false for every one of them.

This module is the check that assertion should have been. Each probe is the
note's own minimal reproduction, and reports STILL REPRODUCES or FIXED against
the running engine rather than against a remembered result.

    python -m benchmarks.engine_notes_probe            # small fixtures
    python -m benchmarks.engine_notes_probe --scale 300  # note 1 at real cardinality

## Two limits worth stating before anyone acts on the output

**This runs embedded.** Notes 1-9 were measured against the HTTP server, so a
FIXED here says the *embedded 1.7.1* build no longer shows the behaviour. It
does not say the 1.7.0 server was fixed -- that is a different binary, and #56
is the standing reminder that assuming two builds agree is how this repo got
into trouble in the first place.

**Note 1 needs `--scale`.** Its own text warns that "a passing 6-node
reproduction proves nothing", because the join bug appears once cardinalities
are real. The default fixture is exactly such a 6-node reproduction and is
reported as INCONCLUSIVE unless `--scale` is given.
"""
from __future__ import annotations

import click

GRAPH = "default"


def _engine():
    from samyama import SamyamaClient

    return SamyamaClient.embedded()


class ResetFailed(RuntimeError):
    """The reset between probes did not run, so the next verdict is not trustworthy.

    Swallowing this is tempting and wrong. Note 8 measures whether a property
    survives `DETACH DELETE`, so its entire verdict *is* the reset: a silently
    failed delete leaves the old node in place and the probe reports on a graph
    it did not build. Every later probe inherits the contamination.
    """


def _reset(client):
    """Empty the graph, and verify it is empty.

    Raising only when `DETACH DELETE` *errors* was not enough: a delete that
    succeeds and leaves rows behind is the failure mode that matters here,
    because every probe opens with this call and would then measure a graph
    holding the previous probe's fixture. `note_8` detects its own case --
    `p1` surviving is reported UNSOUND rather than as note 8 -- but the other
    ten had nothing, and an inherited node shows up as a wrong row count, which
    is exactly what these probes read as a verdict.
    """
    try:
        client.query("MATCH (n) DETACH DELETE n", GRAPH)
        left = client.query("MATCH (n) RETURN count(n)", GRAPH).records[0][0]
    except Exception as exc:
        raise ResetFailed(f"reset failed; later verdicts are unsound: {exc}") from exc
    if left:
        raise ResetFailed(
            f"reset ran without error and left {left:,} nodes; later verdicts "
            f"are unsound, because every probe below would measure a graph "
            f"holding whatever the last one built")


def _rows(client, cypher):
    return client.query(cypher, GRAPH).records


def note_1(client, scale: int):
    """A trailing bound variable in a second MATCH is not joined -> cartesian product."""
    from etl.helpers import create_edges, create_nodes

    _reset(client)
    n = scale
    create_nodes(client, GRAPH, "RBd", [{"id": "B1"}])
    create_nodes(client, GRAPH, "RMd", [{"id": f"M{i}"} for i in range(n)])
    create_nodes(client, GRAPH, "RVr",
                 [{"id": f"M{i}{p}", "p": p} for i in range(n) for p in ("fp32", "int8")])
    create_nodes(client, GRAPH, "RDp",
                 [{"id": f"d{i}{p}"} for i in range(n) for p in ("fp32", "int8")])
    edges = []
    for i in range(n):
        for p in ("fp32", "int8"):
            edges += [("RDp", f"d{i}{p}", "R_ON", "RBd", "B1", None),
                      ("RDp", f"d{i}{p}", "R_OF", "RVr", f"M{i}{p}", None),
                      ("RVr", f"M{i}{p}", "R_VO", "RMd", f"M{i}", None)]
    create_edges(client, GRAPH, edges)
    got = _rows(client, (
        'MATCH (b:RBd)<-[:R_ON]-(d1:RDp)-[:R_OF]->(v1:RVr)-[:R_VO]->(m:RMd) '
        'WHERE v1.p="fp32" '
        'MATCH (b)<-[:R_ON]-(d2:RDp)-[:R_OF]->(v2:RVr)-[:R_VO]->(m) '
        'WHERE v2.p="int8" '
        'RETURN m.id, v1.id, v2.id'))
    mispaired = [r for r in got
                 if not (r[1] == f"{r[0]}fp32" and r[2] == f"{r[0]}int8")]
    ok = len(got) == n and not mispaired
    detail = (f"{len(got):,} rows for {n:,} models, {len(mispaired):,} mispaired "
              f"(a cartesian product would give {n * n:,})")
    return ok, detail


def note_2(client, scale):
    """`RETURN DISTINCT` is a no-op."""
    _reset(client)
    for i, f in enumerate(["A", "A", "B", "B", "B"]):
        _rows(client, f'CREATE (:Board {{id:"b{i}", form_factor:"{f}"}})')
    got = _rows(client, 'MATCH (b:Board) RETURN DISTINCT b.form_factor')
    return len(got) == 2, f"{len(got)} rows over 5 boards with 2 distinct values"


def note_3(client, scale):
    """`ORDER BY` on a RETURN-introduced alias is silently ignored."""
    _reset(client)
    for i, v in enumerate([10.9, 12.7, 7.5, 9.0, 3.2, 20.1]):
        _rows(client, f'CREATE (:Deployment {{id:"d{i}", latency_ms:{v}}})')
    got = [r[0] for r in
           _rows(client, 'MATCH (d:Deployment) RETURN d.latency_ms AS a ORDER BY a ASC')]
    return got == sorted(got), f"{got}"


def note_3b(client, scale):
    """Only the first `ORDER BY` key is honoured; later keys are ignored.

    A load-bearing CLAUDE.md rule in its own right ("One `ORDER BY` key only"),
    enforced across the catalog by `test_order_by_is_actually_applied`, so it
    belongs in the probe even though the notes file files it under note 3.
    """
    _reset(client)
    # Same first key, different second key -- so only the second key can order
    # these, and a dropped second key leaves them in insertion order.
    for i, (a, b) in enumerate([(1, 3), (1, 1), (1, 2)]):
        _rows(client, f'CREATE (:S {{id:"s{i}", a:{a}, b:{b}}})')
    got = [r[0] for r in _rows(
        client, 'MATCH (s:S) WITH s.a AS a, s.b AS b ORDER BY a ASC, b ASC RETURN b')]
    return got == sorted(got), f"second key gave {got}, sorted would be {sorted(got)}"


def note_4(client, scale):
    """`min()` mis-compares an integer sentinel against float values."""
    _reset(client)
    for i, (p, kb) in enumerate([("int8", 6.9), ("fp32", 27.6)]):
        _rows(client, f'CREATE (:V {{id:"v{i}", precision:"{p}", size_kb:{kb}}})')
    int_sentinel = _rows(client, 'MATCH (v:V) RETURN min(CASE WHEN v.precision = '
                                 '"int8" THEN v.size_kb ELSE 999999 END)')[0][0]
    float_sentinel = _rows(client, 'MATCH (v:V) RETURN min(CASE WHEN v.precision = '
                                   '"int8" THEN v.size_kb ELSE 999999.0 END)')[0][0]
    # Tolerance, not `== 6.9`: the value round-trips through the engine's
    # numeric handling, and note 4 is about a sentinel being returned *instead*
    # of the minimum -- a gap of 999,992 -- not about the last bits of a float.
    ok = all(abs(v - 6.9) < 1e-6 for v in (int_sentinel, float_sentinel))
    return ok, (f"min() with an int sentinel -> {int_sentinel}, with a float "
                f"sentinel -> {float_sentinel} (both should be 6.9)")


def note_4b(client, scale):
    """The same int/float weakness in the form the CLAUDE.md rule warns about.

    Probed separately from note 4 rather than `and`-ed into it: they are two
    behaviours, and one pass/fail over both cannot say which of them moved.
    """
    _reset(client)
    _rows(client, 'CREATE (:N {id:"n1", x:3})')
    got = _rows(client, 'MATCH (n:N) WHERE n.x > 0.5 RETURN n.id')
    return len(got) == 1, f"`WHERE n.x > 0.5` against int x=3 -> {len(got)} rows (want 1)"


def _does_not_parse(client, cypher: str):
    """Whether the engine refuses `cypher`, distinguishing refusal from breakage.

    Notes 5 and 6 are "this statement does not parse", so the probe reads a
    raised exception as the note reproducing. Any exception would do that --
    including a dropped connection or an engine that has stopped answering at
    all, which would report "STILL REPRODUCES" for a run that measured nothing.

    So after a failure the client is asked a statement that certainly parses.
    If that also fails, the engine is not refusing this Cypher, it is simply not
    working, and the verdict is `None` -- unsound -- rather than a reproduction.
    """
    try:
        _rows(client, cypher)
        return True, "parsed"
    except Exception as exc:  # the engine raises untyped errors
        refusal = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}"
    try:
        _rows(client, "RETURN 1")
    except Exception as alive:
        return None, (f"the statement failed ({refusal}) but so did `RETURN 1` "
                      f"({type(alive).__name__}), so the engine is not refusing "
                      f"this Cypher -- it is not answering at all")
    return False, refusal


def note_5(client, scale):
    """Negated pattern predicates do not parse."""
    _reset(client)
    _rows(client, 'CREATE (:N {id:"n1"})')
    return _does_not_parse(client, 'MATCH (n:N) WHERE NOT (n)-[:R]->() RETURN n.id')


def note_6(client, scale):
    """`CREATE CONSTRAINT ... REQUIRE ... IS UNIQUE` does not parse.

    If it ever *does* parse, the constraint outlives this probe: `_reset` is
    `DETACH DELETE`, which removes nodes and not schema. Every probe below
    would then run against a graph with a uniqueness constraint on `:N(id)`
    that nothing put there deliberately. So a parse is followed by a drop
    attempt, and a drop that fails is reported rather than left behind.
    """
    _reset(client)
    parsed, detail = _does_not_parse(
        client, 'CREATE CONSTRAINT FOR (n:N) REQUIRE n.id IS UNIQUE')
    if parsed is True:
        try:
            _rows(client, 'DROP CONSTRAINT FOR (n:N) REQUIRE n.id IS UNIQUE')
        except Exception as exc:
            return None, (f"the constraint parsed ({detail}) and could not be "
                          f"dropped ({type(exc).__name__}), so it outlives this "
                          f"probe and every verdict below is unsound")
    return parsed, detail


def note_8(client, scale):
    """Deleted property columns resurrect onto new nodes."""
    _reset(client)
    _rows(client, 'CREATE (:P {id:"p1", doomed:"ghost"})')
    _reset(client)                      # DETACH DELETE, which note 8 says is not a reset
    _rows(client, 'CREATE (:P {id:"p2"})')
    got = _rows(client, 'MATCH (p:P) RETURN p.id, p.doomed')
    # Every row, not `got[0]`: if the reset above did not actually delete `p1`
    # the survivor may sort either way, and reading one row would report the
    # behaviour of whichever happened to come first.
    resurrected = [list(r) for r in got if r[1] is not None]
    unexpected = [list(r) for r in got if r[0] != "p2"]
    detail = f"{[list(r) for r in got]}"
    if unexpected:
        # UNSOUND, not STILL REPRODUCES. The delete not running and the property
        # surviving the delete are different findings, and only the second is
        # note 8. Reporting the first as the second would be a false positive
        # for the exact behaviour this probe exists to detect.
        return None, (f"{detail}  <- `p1` survived the reset, so the delete did "
                      f"not run and this says nothing about note 8")
    return not resurrected, detail


def note_8b(client, scale):
    """`<>` matches a null property, so an inequality quietly includes absent keys.

    The CLAUDE.md rule is "use `IS NULL` / `IS NOT NULL`". Load-bearing: `EA11`
    filtered on `kind <> "MCU-CPU"` until #69, and a null `kind` matching would
    have counted a node the query meant to exclude.
    """
    _reset(client)
    _rows(client, 'CREATE (:Q {id:"has", kind:"NPU"})')
    _rows(client, 'CREATE (:Q {id:"null"})')           # no `kind` at all
    # Note 8 first. If a property column survived an earlier probe's delete
    # (that is note 8), `null` may come back carrying a `kind` it was never
    # given -- and then `kind <> "NPU"` matching it is note 8 showing through,
    # not note 8b. Different findings; reporting one as the other is the false
    # positive this probe is most exposed to, because 8 runs immediately above.
    planted = _rows(client, 'MATCH (q:Q) WHERE q.id = "null" RETURN q.kind')
    if planted and planted[0][0] is not None:
        return None, (f"`null` was created without `kind` and came back with "
                      f"{planted[0][0]!r} -- that is note 8 resurrecting a "
                      f"property column, so this run says nothing about 8b")
    got = {r[0] for r in _rows(client, 'MATCH (q:Q) WHERE q.kind <> "NPU" RETURN q.id')}
    return got == set(), f"`kind <> \"NPU\"` matched {sorted(got)}; want no rows"


def note_9(client, scale):
    """Aggregating a bare node variable over a multi-variable MATCH does not aggregate."""
    _reset(client)
    _rows(client, 'CREATE (:A {id:"a1"}), (:A {id:"a2"}), (:O {id:"o1"}), (:O {id:"o2"})')
    for a in ("a1", "a2"):
        for o in ("o1", "o2"):
            _rows(client, f'MATCH (x:A),(y:O) WHERE x.id="{a}" AND y.id="{o}" '
                          f'CREATE (x)-[:U]->(y)')
    bare = _rows(client, 'MATCH (a:A)-[:U]->(o:O) RETURN count(DISTINCT o)')
    prop = _rows(client, 'MATCH (a:A)-[:U]->(o:O) RETURN count(DISTINCT o.id)')
    ok = len(bare) == 1 and bare[0][0] == 2
    return ok, (f"count(DISTINCT o) -> {[list(r) for r in bare]}, "
                f"count(DISTINCT o.id) -> {[list(r) for r in prop]}")


# Note 7 (the tenant/graph argument is ignored) is deliberately absent: it is a
# property of the OSS *server's* HTTP path, and this probe runs embedded, where
# there is no tenant boundary to ignore in the first place. Reporting it either
# way from here would be a measurement of nothing.
PROBES = [
    (1, "trailing bound variable in a 2nd MATCH is not joined", note_1),
    (2, "RETURN DISTINCT is a no-op", note_2),
    (3, "ORDER BY on a RETURN-introduced alias is dropped", note_3),
    ("3b", "only the first ORDER BY key is honoured", note_3b),
    (4, "min() mis-compares an int sentinel against floats", note_4),
    ("4b", "an int property compared against a float literal", note_4b),
    (5, "negated pattern predicates do not parse", note_5),
    (6, "CREATE CONSTRAINT does not parse", note_6),
    (8, "deleted property columns resurrect", note_8),
    ("8b", "`<>` matches a null property", note_8b),
    (9, "aggregating a bare node variable does not aggregate", note_9),
]


@click.command()
@click.option("--scale", default=0, type=int,
              help="Models on one board for note 1. Its own text says a small "
                   "reproduction proves nothing, so note 1 is INCONCLUSIVE without this.")
def main(scale):
    client = _engine()
    # Reported, not relied on: `status()` is not a documented contract and an
    # engine that changes the field name should not stop the probe from running.
    # The version is a label on the output, and losing the label is not a reason
    # to lose the measurement.
    try:
        version = client.status().version
    except Exception as exc:
        version = f"unknown ({type(exc).__name__} reading client.status())"
    click.echo(f"embedded engine: {version}\n")
    fixed, reproduces, inconclusive, unsound = [], [], [], []
    for number, title, probe in PROBES:
        if number == 1 and not scale:
            # Skipped, not run and discarded. Building the fixture only to throw
            # the verdict away wastes the work and, worse, leaves a graph behind
            # that the next probe's reset is the only thing standing between.
            verdict, bucket, detail = "INCONCLUSIVE", inconclusive, "not run -- needs --scale"
        else:
            try:
                # `scale`, not `scale or 2`. Note 1 is the only probe that
                # reads it and it is skipped above when `scale` is 0, so the
                # fallback could never be reached -- and had it been, it would
                # have run note 1 at a size its own text calls proof of nothing.
                ok, detail = probe(client, scale)
            except ResetFailed as exc:
                verdict, bucket, detail = "UNSOUND", unsound, str(exc)
            except Exception as exc:
                # Per-probe, so one unexpected failure does not abort the run and
                # hide the eight verdicts after it. A probe that cannot run is a
                # result too -- it is just not a result about the engine note.
                verdict, bucket = "ERROR", unsound
                detail = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}"
            else:
                if ok is None:
                    verdict, bucket = "UNSOUND", unsound
                elif ok:
                    verdict, bucket = "FIXED", fixed
                else:
                    verdict, bucket = "STILL REPRODUCES", reproduces
        bucket.append(number)
        click.echo(f"note {number}  {verdict}")
        click.echo(f"          {title}")
        click.echo(f"          {detail}\n")

    # The denominator is what was actually measured. Counting an INCONCLUSIVE
    # note 1 among the probed would let "7 of 8" stand for a run in which the
    # single most important note was never really tested.
    measured = len(fixed) + len(reproduces)
    click.echo(f"{len(fixed)} of {measured} measured probes do not reproduce on {version}.")
    if reproduces:
        click.echo(f"still reproducing: {reproduces}")
    if unsound:
        click.echo(f"UNSOUND / ERROR, measuring nothing: {unsound} -- these are not "
                   "evidence\neither way, and are excluded from the count above.")
    if inconclusive:
        click.echo(f"NOT MEASURED: {inconclusive} -- note 1 needs --scale, because its "
                   "own\ntext says a small reproduction proves nothing. Re-run with "
                   "--scale 300.")
    click.echo("note 7 is not probed at all: it is a property of the OSS server's "
               "HTTP path,\nand there is no tenant boundary to ignore embedded.")
    click.echo("\nA FIXED here is about the EMBEDDED build. The notes were measured "
               "against the\n1.7.0 HTTP server, which is a different binary -- see "
               "#56 for what assuming\notherwise costs.")


if __name__ == "__main__":
    main()
