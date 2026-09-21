"""One minimal reproduction per probeable engine note in `docs/engine-notes.md`.

Notes 1-6, 8 and 9, plus 3b, 4b and 8b. Not every note: note 7 is a property
of the HTTP server (see the comment above `PROBES`), and notes 10 and 11 are
resolved and guarded elsewhere -- `tests/test_engine_version.py` re-runs note
11's reproduction on every test run, and note 10 is the second-`WITH` shape
`EA01`/`EA02` exercise in `tests/test_correctness.py`.

Each probe returns `(ok, detail)`: `True` when the running engine gives the
right answer (FIXED), `False` when it still shows the note's behaviour, `None`
when the run cannot tell. `verdict_for` in `engine_notes_harness.py` names
those three states, and `engine_notes_probe.py` runs `PROBES` and reports.

Every probe owns its labels. Note 8 is a *column* behaviour that no node-count
reset can detect, so a label shared between probes would let one probe's
resurrected property decide another's verdict.
"""
from __future__ import annotations

from benchmarks.engine_notes_harness import (
    GRAPH,
    built,
    parses,
    reset,
    rows,
)


def note_1(client, scale: int):
    """A trailing bound variable in a second MATCH is not joined -> cartesian product."""
    from etl.helpers import create_edges, create_nodes

    n = scale
    if n < 2:
        # At 0 there is nothing to join and `len(got) == n` holds for an empty
        # graph; at 1 the cartesian product *is* the right answer. The CLI
        # refuses these, but the function should not rely on its caller.
        return None, f"--scale {n} cannot tell a cartesian product from a join"
    reset(client)
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
    built(client, "RDp", 2 * n)
    got = rows(client, (
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
    reset(client)
    for i, f in enumerate(["A", "A", "B", "B", "B"]):
        rows(client, f'CREATE (:B2 {{id:"b{i}", form_factor:"{f}"}})')
    built(client, "B2", 5)
    got = rows(client, 'MATCH (b:B2) RETURN DISTINCT b.form_factor')
    return len(got) == 2, f"{len(got)} rows over 5 boards with 2 distinct values"


def note_3(client, scale):
    """`ORDER BY` on a RETURN-introduced alias is silently ignored."""
    reset(client)
    for i, v in enumerate([10.9, 12.7, 7.5, 9.0, 3.2, 20.1]):
        rows(client, f'CREATE (:D3 {{id:"d{i}", latency_ms:{v}}})')
    built(client, "D3", 6)
    got = [r[0] for r in
           rows(client, 'MATCH (d:D3) RETURN d.latency_ms AS a ORDER BY a ASC')]
    # Both halves. `[] == sorted([])` is true, so "is it sorted" alone reports
    # FIXED for an engine that returned nothing at all.
    if len(got) != 6:
        return None, f"expected 6 rows back, got {len(got)}: {got}"
    return got == sorted(got), f"{got}"


def note_3b(client, scale):
    """Only the first `ORDER BY` key is honoured; later keys are ignored.

    A load-bearing CLAUDE.md rule in its own right ("One `ORDER BY` key only"),
    enforced across the catalog by `test_order_by_is_actually_applied`, so it
    belongs in the probe even though the notes file files it under note 3.
    """
    reset(client)
    # Same first key, different second key -- so only the second key can order
    # these, and a dropped second key leaves them in insertion order.
    # Six rows, not three. With three, a dropped second key still comes back
    # sorted by luck one run in six, which is a false FIXED at a rate nobody
    # would notice. Six makes it 1 in 720, and the insertion order below is
    # deliberately the reverse of the sorted one.
    for i, (a, b) in enumerate([(1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1)]):
        rows(client, f'CREATE (:S3b {{id:"s{i}", a:{a}, b:{b}}})')
    built(client, "S3b", 6)
    got = [r[0] for r in rows(
        client, 'MATCH (s:S3b) WITH s.a AS a, s.b AS b ORDER BY a ASC, b ASC RETURN b')]
    if len(got) != 6:
        return None, f"expected 6 rows back, got {len(got)}: {got}"
    return got == sorted(got), f"second key gave {got}, sorted would be {sorted(got)}"


def note_4(client, scale):
    """`min()` mis-compares an integer sentinel against float values.

    One behaviour, one verdict. The note is about the *integer* sentinel, so
    that is the only thing the verdict reads. The float-sentinel query is the
    note's own workaround and serves as a control: if it is wrong too, `min()`
    is broken in some other way and this run says nothing about note 4.
    Folding both into one `all(...)` could not say which of them moved.
    """
    reset(client)
    for i, (p, kb) in enumerate([("int8", 6.9), ("fp32", 27.6)]):
        rows(client, f'CREATE (:V4 {{id:"v{i}", precision:"{p}", size_kb:{kb}}})')
    built(client, "V4", 2)

    def smallest(sentinel):
        got = rows(client, 'MATCH (v:V4) RETURN min(CASE WHEN v.precision = '
                           f'"int8" THEN v.size_kb ELSE {sentinel} END)')
        return got[0][0] if got else None

    def is_min(value):
        # Tolerance, not `== 6.9`: the value round-trips through the engine's
        # numeric handling, and note 4 is about a sentinel being returned
        # *instead* of the minimum -- a gap of 999,992 -- not about the last
        # bits of a float. A None or a string is not a number to compare, and
        # must read as "cannot tell" rather than raise TypeError into ERROR.
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        return numeric and abs(value - 6.9) < 1e-6

    control, probe = smallest("999999.0"), smallest("999999")
    if not is_min(control):
        return None, (f"the float-sentinel control returned {control!r}, want "
                      f"6.9 -- `min()` is wrong without the int/float mix, so "
                      f"this run says nothing about note 4")
    return is_min(probe), (f"min() with an int sentinel -> {probe!r} (want 6.9); "
                           f"float-sentinel control -> {control!r}")


def note_4b(client, scale):
    """The same int/float weakness in the form the CLAUDE.md rule warns about.

    Probed separately from note 4 rather than `and`-ed into it: they are two
    behaviours, and one pass/fail over both cannot say which of them moved.
    """
    reset(client)
    rows(client, 'CREATE (:N4b {id:"n1", x:3})')
    built(client, "N4b", 1)
    got = rows(client, 'MATCH (n:N4b) WHERE n.x > 0.5 RETURN n.id')
    return len(got) == 1, f"`WHERE n.x > 0.5` against int x=3 -> {len(got)} rows (want 1)"


def note_5(client, scale):
    """Negated pattern predicates do not parse."""
    reset(client)
    rows(client, 'CREATE (:N5 {id:"n1"})')
    built(client, "N5", 1)
    accepted, detail = parses(
        client, 'MATCH (n:N5) WHERE NOT (n)-[:R]->() RETURN n.id')
    if accepted is not True:
        return accepted, detail
    # Parsing is not the claim. This note is why the catalog writes every
    # anti-join as `OPTIONAL MATCH ... count() = 0`, and someone reading FIXED
    # will drop that workaround -- so the negated pattern has to return the
    # right rows, not merely be accepted. `n1` has no `:R`, so it must come
    # back.
    got = [r[0] for r in rows(
        client, 'MATCH (n:N5) WHERE NOT (n)-[:R]->() RETURN n.id')]
    if got != ["n1"]:
        return False, (f"the negated pattern parsed but answered {got}, want "
                       f"['n1'] -- accepted and wrong is worse than refused")
    return True, "parsed, and returned the one node with no :R edge"


def note_6(client, scale):
    """`CREATE CONSTRAINT ... REQUIRE ... IS UNIQUE` does not parse.

    On 1.7.1 it **does** parse, and the constraint is really created --
    `SHOW CONSTRAINTS` lists it. There is then no way to remove it: neither
    `DROP CONSTRAINT FOR (n:N6) REQUIRE ...` nor the older
    `DROP CONSTRAINT ON (n:N6) ASSERT ...` parses. `reset` is `DETACH DELETE`,
    which removes nodes and not schema, so the constraint outlives this probe
    -- but not the run: each `SamyamaClient.embedded()` is its own in-memory
    graph, and a constraint created on one is absent from the next (checked on
    1.7.1: `SHOW CONSTRAINTS` on a fresh client returns `[]`).

    Within the run it is contained rather than unsound, because `:N6` belongs
    to this probe alone: nothing below touches that label, so a uniqueness
    constraint on it cannot change another verdict. The drop is still
    attempted, and its failure is reported in the detail line rather than
    swallowed -- "created and cannot be removed" is worth knowing, and it is
    the reason per-probe labels are not decoration.
    """
    reset(client)
    parsed, detail = parses(
        client, 'CREATE CONSTRAINT FOR (n:N6) REQUIRE n.id IS UNIQUE')
    if parsed is True:
        # Parsing is not the claim: the note exists because `id` uniqueness is
        # a loader invariant rather than an enforced constraint, and someone
        # reading FIXED will stop maintaining the invariant. So the constraint
        # has to *reject a duplicate*. An engine that accepts the statement and
        # enforces nothing is the worst outcome here and the one this catches.
        # The first insert is fixture, not measurement: if even it is refused,
        # the constraint is not the thing being tested, so say so rather than
        # letting the raise land in ERROR.
        first, why_first = parses(client, 'CREATE (:N6 {id:"dup"})')
        if first is not True:
            return None, (f"the constraint was created, but a first `:N6` insert "
                          f"with no duplicate was refused too ({why_first[:60]})")
        inserted, why = parses(client, 'CREATE (:N6 {id:"dup"})')
        if inserted is True:
            held = rows(client, 'MATCH (n:N6) RETURN count(n.id)')
            return False, (f"the constraint was accepted and enforces nothing: "
                           f"a duplicate `id` inserted, {held[0][0] if held else 0} "
                           f"rows now hold it")
        detail = f"{detail}, and a duplicate id is rejected ({why[:40]})"
        try:
            rows(client, 'DROP CONSTRAINT FOR (n:N6) REQUIRE n.id IS UNIQUE')
        except Exception as exc:
            detail = (f"{detail}; created and left behind -- no DROP syntax "
                      f"parses ({type(exc).__name__}). Contained: `:N6` is "
                      f"used by no other probe")
    return parsed, detail


def note_8(client, scale):
    """Deleted property columns resurrect onto new nodes.

    `:P8` is this probe's own label, and every probe owns its labels -- which is
    what keeps a resurrected column from leaking into the next verdict. Note 8
    is a *column* behaviour, so `reset` counting nodes cannot detect it; the
    isolation is the guard. Each label ends in its probe's number (`:B2`,
    `:D3`, `:S3b`, `:V4`, `:N4b`, `:N5`, `:N6`, `:P8`, `:Q8b`, `:A9`/`:O9`;
    note 1's are `R`-prefixed) -- notes 2-4 used to build real schema labels
    (`:Board`, `:Deployment`) and 4b and 5 shared `:N`.
    """
    reset(client)
    rows(client, 'CREATE (:P8 {id:"p1", doomed:"ghost"})')
    reset(client)                      # DETACH DELETE, which note 8 says is not a reset
    rows(client, 'CREATE (:P8 {id:"p2"})')
    got = rows(client, 'MATCH (p:P8) RETURN p.id, p.doomed')
    # Every row, not `got[0]`. `p1` cannot be among them: `reset` above
    # verifies the graph is empty and raises `ResetFailed` otherwise, which
    # `tests/test_engine_notes_probe.py` pins -- so a surviving `p1` (a delete
    # that did not run, a different finding from note 8) never reaches here.
    resurrected = [list(r) for r in got if r[1] is not None]
    return not resurrected, f"{[list(r) for r in got]}"


def note_8b(client, scale):
    """`<>` matches a null property, so an inequality quietly includes absent keys.

    The CLAUDE.md rule is "use `IS NULL` / `IS NOT NULL`". Load-bearing: `EA11`
    filtered on `kind <> "MCU-CPU"` until #69, and a null `kind` matching would
    have counted a node the query meant to exclude.
    """
    reset(client)
    rows(client, 'CREATE (:Q8b {id:"has", kind:"NPU"})')
    rows(client, 'CREATE (:Q8b {id:"null"})')           # no `kind` at all
    # Note 8 first. If a property column survived an earlier probe's delete
    # (that is note 8), `null` may come back carrying a `kind` it was never
    # given -- and then `kind <> "NPU"` matching it is note 8 showing through,
    # not note 8b. Different findings; reporting one as the other is the false
    # positive this probe is most exposed to, because 8 runs immediately above.
    planted = rows(client, 'MATCH (q:Q8b) WHERE q.id = "null" RETURN q.kind')
    if planted and planted[0][0] is not None:
        return None, (f"`null` was created without `kind` and came back with "
                      f"{planted[0][0]!r} -- that is note 8 resurrecting a "
                      f"property column, so this run says nothing about 8b")
    # A positive control first: `<>` has to match something, or `got == set()`
    # below is satisfied by an engine that returns nothing for every
    # comparison, and the note would read FIXED on a build where `<>` is
    # broken outright.
    # Restricted to `has`: over the whole label, a build where note 8b
    # reproduces would match `null` too, and the control would call a working
    # `<>` broken -- reporting UNSOUND for exactly the case it should report.
    control = {r[0] for r in rows(
        client, 'MATCH (q:Q8b) WHERE q.id = "has" AND q.kind <> "GPU" RETURN q.id')}
    if control != {"has"}:
        return None, (f"`kind <> \"GPU\"` matched {sorted(control)}, want "
                      f"['has'] -- `<>` is not working at all here, so the "
                      f"null case below would say nothing about note 8b")
    got = {r[0] for r in rows(client, 'MATCH (q:Q8b) WHERE q.kind <> "NPU" RETURN q.id')}
    return got == set(), (f"`kind <> \"GPU\"` matched {sorted(control)} as it should; "
                          f"`kind <> \"NPU\"` matched {sorted(got)}, want none")


def note_9(client, scale):
    """Aggregating a bare node variable over a multi-variable MATCH does not aggregate."""
    reset(client)
    rows(client, 'CREATE (:A9 {id:"a1"}), (:A9 {id:"a2"}), (:O9 {id:"o1"}), (:O9 {id:"o2"})')
    built(client, "A9", 2)
    built(client, "O9", 2)
    for a in ("a1", "a2"):
        for o in ("o1", "o2"):
            rows(client, f'MATCH (x:A9),(y:O9) WHERE x.id="{a}" AND y.id="{o}" '
                          f'CREATE (x)-[:U]->(y)')
    bare = rows(client, 'MATCH (a:A9)-[:U]->(o:O9) RETURN count(DISTINCT o)')
    prop = rows(client, 'MATCH (a:A9)-[:U]->(o:O9) RETURN count(DISTINCT o.id)')
    # The property form is the note's workaround and the control. If it is
    # not 2 either, the edges did not land as built, and the bare form being
    # wrong says nothing about note 9.
    if [list(r) for r in prop] != [[2]]:
        return None, (f"control count(DISTINCT o.id) -> {[list(r) for r in prop]}, "
                      f"want [[2]] -- the fixture is not what this probe built")
    ok = len(bare) == 1 and bare[0][0] == 2
    return ok, (f"count(DISTINCT o) -> {[list(r) for r in bare]}, "
                f"count(DISTINCT o.id) -> {[list(r) for r in prop]}")


# Note 7 (the tenant/graph argument is ignored) is deliberately absent: it is a
# property of the OSS *server's* HTTP path, and this probe runs embedded, where
# there is no tenant boundary to ignore in the first place. Reporting it either
# way from here would be a measurement of nothing.
#
# Notes 10 and 11 are absent for a different reason: they are resolved (0.6.1
# against a 1.7.0 server, not a behaviour of either), and each already has a
# guard that runs on every `pytest` -- see the module docstring.
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
