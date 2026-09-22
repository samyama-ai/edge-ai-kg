# Engine notes -- Samyama Graph v1.7.0 (and the embedded build, notes 10-13b)

> **Partly superseded, 2026-09-21.** Notes 10 and 11 are **resolved**: they
> were `samyama` 0.6.1 against a 1.7.0 server, not two builds disagreeing, and
> `pyproject.toml` now floors the engine at 1.7.1 (#56). Notes 1-6, 8 and 9 do
> not reproduce on *embedded* 1.7.1 either -- `python -m
> benchmarks.engine_notes_probe --scale 300` -- but they were measured on the
> 1.7.0 *server*, which has not been re-probed, so their workarounds stay.
> Note 7 has no probe. The text below is the original record; #94 rewrites it.

Behaviour observed while building this KG.
**Notes 1-9 are filed upstream** — see the tracking issue
[samyama-graph#368](https://github.com/samyama-ai/samyama-graph/issues/368).

Notes 1-9 were observed on the OSS engine at v1.7.0
(`target/release/samyama --http-port 8080`). Every one of them is load-bearing:
the loader or the query catalog works around it. Verified 2026-08-14.

**Versions these describe.** Notes 1-9 are the **server** at 1.7.0
(`ghcr.io/samyama-ai/samyama-graph:1`), as the paragraph above says. Where an
embedded build is involved -- notes 10 and 11, which compare the two -- it was
`samyama` 0.6.1 from pip. **Notes 12, 13 and 13b were measured on embedded
1.7.1.** `pyproject.toml` asks for `samyama>=0.6.0` unpinned, so which embedded
build a fresh install gets is not decided here; run `pip show samyama` rather
than trusting a sentence on this page. Whether to pin it belongs with #56, which has not yet
decided which build the suite treats as authoritative -- pinning now would be
choosing that by the back door.

**Notes 10 and 11 are a different kind of entry.** Neither is a behaviour of the
server: both are disagreements between the server and the in-process embedded
build, neither is filed upstream, and neither is worked around in the catalog
today -- note 11 has a known workaround that is deliberately deferred to #56,
note 10 has none established. Between them
they are why three tests in `tests/test_correctness.py` are marked `xfail`:

| Test | Excused | Note |
|---|---|---|
| `test_every_catalog_query_runs_and_returns_rows` | `[EA01]`, `[EA02]` only | 10 |
| `test_order_by_is_actually_applied` | `[EA01]`, `[EA02]` only | 10 |
| `test_ea04_quantization_unlock_is_not_a_cartesian_product` | whole test | **11** |

Three test functions, **five xfailed parameters** in the run output. The two
sweeps are parametrised over the catalog so only the affected queries are
excused: marking either whole would excuse the other fourteen, and those two are
what `CLAUDE.md` calls the catalog-wide invariant enforcers. The marks are
applied with `request.applymarker`, not `pytest.xfail()` -- the imperative form
never runs the body, so a parameter could only ever report XFAIL and the XPASS
that says "the divergence is gone, remove the mark" would never arrive.

Verified 2026-08-28 and 2026-08-31, both tracked at #56.

---

## 1. A trailing bound variable in a second MATCH clause is not joined

> Filed upstream: [samyama-graph#360](https://github.com/samyama-ai/samyama-graph/issues/360)

**Severity: correctness. Silently returns wrong rows.**

When a second `MATCH` clause re-mentions a variable bound by an earlier `MATCH`,
and that variable appears in a **non-leading position** in the second pattern
while *another* shared variable leads it, the engine does not enforce the join.
It returns a cartesian product instead.

Minimal reproduction -- one board `B1`, two models `M1`/`M2`, each with an fp32
and an int8 variant deployed on `B1`:

```cypher
MATCH (b:RBd)<-[:R_ON]-(d1:RDp)-[:R_OF]->(v1:RVr)-[:R_VO]->(m:RMd) WHERE v1.p="fp32"
MATCH (b)<-[:R_ON]-(d2:RDp)-[:R_OF]->(v2:RVr)-[:R_VO]->(m)         WHERE v2.p="int8"
RETURN m.id, v1.id, v2.id
```

Expected 2 rows (`M1,M1fp32,M1int8` and `M2,M2fp32,M2int8`). **Got 4** -- the
cross pairs `M1,M2fp32,M1int8` and `M2,M1fp32,M2int8` are also returned. `v2` is
correctly constrained to `m`; `v1` is not.

Variants tested:

| Shape | Small graph | Full graph (24K nodes) |
|---|---|---|
| 2nd MATCH, shared var **trailing**, another shared var leading | **WRONG** | **WRONG** |
| 2nd MATCH, shared var **leading** | correct | correct |
| 2nd MATCH, only one shared variable | correct | correct |
| `WITH` between the two MATCH clauses | **WRONG** | **WRONG** |
| Single MATCH, comma-separated patterns | correct | **WRONG** |

**Note the last row.** The comma-separated single-`MATCH` form looks like a fix
on a toy graph and still breaks at scale -- presumably a different join plan is
chosen once the cardinalities and indexes are real. Any "workaround" for this
bug must be validated on a full-size graph; a passing 6-node reproduction proves
nothing.

**Workaround actually used here:** avoid the self-join entirely. `EA04` is
written as a single linear pattern plus conditional aggregation
(`sum(CASE WHEN ... )`), which has no second binding to lose:

```cypher
MATCH (m:Model)<-[:VARIANT_OF]-(v:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WITH m.name AS model, b.name AS board,
     sum(CASE WHEN v.precision = "fp32" AND d.fits = 0 THEN 1 ELSE 0 END) AS fp32_misses,
     sum(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN 1 ELSE 0 END) AS int8_hits,
     max(CASE WHEN v.precision = "fp32" THEN v.size_kb ELSE 0 END) AS fp32_kb,
     max(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 0 END) AS int8_kb
WHERE fp32_misses > 0 AND int8_hits > 0
RETURN model, board, fp32_kb, int8_kb
```

Because `int8_kb` is by construction exactly `fp32_kb / 4`, a cartesian product
is *detectable*: `tests/test_correctness.py` asserts that ratio, and
`test_ea04_shape_is_not_a_cartesian_product` pins the behaviour on a
purpose-built 4-deployment fixture.

**One query re-binds in trailing position deliberately, and is measured.**
`EA17` (#35) is five `OPTIONAL MATCH` legs that each re-bind a variable the
opening `MATCH` already bound. That is this shape, and there is no other way to
express "reachable from some *other* sensor" -- it is note 5's anti-join, which
needs the re-bind. So it is not taken on trust:
`test_ea17_matches_ground_truth_at_full_scale` compares every sensor, both
counts and the depth, against a Python breadth-first search at `--scale 1.0`,
where this note's failure appears if it appears. 0 disagreements.

**That test is opt-in.** It is gated on `--full-scale`, so a default `pytest`
run does not execute it -- loading a scale-1.0 fleet takes minutes. A green
default run is therefore *not* evidence about this note; `pytest --full-scale`
is. Re-run it after any edit to `EA17`'s patterns, because a small graph cannot
settle this and the failure is silent in the direction that looks like the
finding.

---

## 2. `RETURN DISTINCT` is a no-op

> Filed upstream: [samyama-graph#361](https://github.com/samyama-ai/samyama-graph/issues/361)

**Severity: correctness. Returns duplicate rows.**

```cypher
MATCH (b:Board) RETURN DISTINCT b.form_factor
```

returns 120 rows (one per board) across 8 distinct form factors. Multi-column
`RETURN DISTINCT` is equally ineffective.

`DISTINCT` *inside an aggregate* -- `count(DISTINCT op)` -- works correctly and
is used throughout the catalog.

**Workaround used here:** deduplicate with a `WITH`-grouping plus an aggregate,
which groups correctly:

```cypher
MATCH (b:Board) WITH b.form_factor AS f, b.year AS y, count(b) AS n RETURN f, y, n
```

---

## 3. `ORDER BY` on a RETURN-introduced alias is silently ignored

> Filed upstream: [samyama-graph#362](https://github.com/samyama-ai/samyama-graph/issues/362)

**Severity: correctness. Returns an arbitrary subset when combined with LIMIT.**

```cypher
MATCH (d:Deployment) RETURN d.latency_ms AS a ORDER BY a ASC LIMIT 6
-- returns 10.976, 12.771, 7.569, 9.003, ...   (unsorted)
```

An alias introduced in `RETURN` cannot be referenced by `ORDER BY`. The clause
is dropped without error. Combined with `LIMIT`, this is worse than an unsorted
result: `ORDER BY latency ASC LIMIT 12` returns *an arbitrary 12 rows*, not the
twelve fastest -- while looking exactly like a top-N.

| Form | Result |
|---|---|
| `RETURN d.latency_ms ORDER BY d.latency_ms` | correct |
| `RETURN d.latency_ms AS a ORDER BY d.latency_ms` | correct |
| `RETURN d.latency_ms AS a ORDER BY a` | **ignored** |
| `WITH d.latency_ms AS a RETURN a ORDER BY a` | correct |
| `WITH ... count(x) AS n RETURN n ORDER BY n` | correct |

Aliases introduced by `WITH` work; aliases introduced by `RETURN` do not. A
`RETURN` containing an aggregate happens to work, because the implicit grouping
pass establishes the alias.

**Workaround used here:** every catalog query projects through a `WITH` before
`RETURN`, and sorts on the `WITH` alias.

### 3b. Only the first `ORDER BY` key is honoured

`ORDER BY category, operator` sorts by `category` and leaves `operator`
unsorted within each group. Multi-key sorts are therefore avoided entirely;
`tests/test_correctness.py::test_order_by_is_actually_applied` asserts every
catalog query uses a single sort key *and* that the result really is sorted.

### 3c. `ORDER BY` after `UNION ALL` is dropped

Measured 2026-09-09 while writing `EA17` (#35), which is five legs joined by `UNION ALL`.
Appending `ORDER BY affected DESC` to the whole statement returned the rows in
leg order -- `16, 60, 1440, 120, 4` -- unchanged and unsorted. No error.

Same family as note 3 and the same severity: with `LIMIT` it would be an
arbitrary N dressed as a top-N. `test_order_by_is_actually_applied` cannot
assert this note -- `EA17` deliberately carries no `ORDER BY`, so there is no
sort to check applied. It no longer *skips* silently, though: an unsorted query
must name itself in that test's `NO_ORDER_BY` with its reason, so `EA17` is
covered by an explicit allowlist rather than by escaping the filter. Adding a
query with no `ORDER BY` and no entry there fails.

**Workaround used here:** `EA17` carries no `ORDER BY` at all, and returns five
rows keyed by `kind` so there is nothing an order would tell you. A `UNION`
result that genuinely needs sorting has no known workaround on this build --
sort it in the caller.

---

## 4. `min()` mis-compares an integer sentinel against float values

> Filed upstream: [samyama-graph#365](https://github.com/samyama-ai/samyama-graph/issues/365)

**Severity: correctness. Returns the sentinel instead of the minimum.**

```cypher
RETURN min(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 999999   END)  -- 999999  (wrong)
RETURN min(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 999999.0 END)  -- 6.9     (right)
```

With an `int` sentinel, `min()` returns the sentinel even though float values
compare smaller. Writing the sentinel as a float fixes it. This is the same
int/float coercion weakness that makes `WHERE n.x > 0.5` return nothing when
`x` was stored as an int -- **keep numeric literal types consistent with the
stored property type.**

---

## 5. Negated pattern predicates do not parse

> Filed upstream: [samyama-graph#367](https://github.com/samyama-ai/samyama-graph/issues/367)

**Severity: parse error. Fails loudly rather than silently.**

`WHERE NOT (:Acc)-[:SUPPORTS]->(op)` is a parse error.

**Workaround used here:** the standard anti-join, which works correctly:

```cypher
MATCH (op:Operator)
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op)
WITH op, count(k) AS kernels
WHERE kernels = 0
RETURN op.name
```

---

## 6. `CREATE CONSTRAINT ... REQUIRE ... IS UNIQUE` does not parse

> Filed upstream: [samyama-graph#367](https://github.com/samyama-ai/samyama-graph/issues/367)

**Severity: parse error. Uniqueness becomes a loader invariant instead.**

Only `CREATE INDEX ON :Label(prop)` is accepted. Uniqueness of `id` is therefore
a loader invariant, not an engine-enforced one -- ids are minted deterministically
in `etl/generate.py`.

---

## 7. The tenant / graph argument is ignored on the OSS HTTP path

> Filed upstream: [samyama-graph#366](https://github.com/samyama-ai/samyama-graph/issues/366)

**Severity: isolation. Two datasets loaded into different graphs merge silently.**

`client.query(cypher, "some_graph")` writes to, and reads from, the single
`default` graph regardless of the name passed. Writing to `graph_a` is visible
from `graph_b`, and `list_graphs()` only ever reports `default`.

This is consistent with multi-tenancy being an Enterprise Edition feature, but
it is worth knowing: **passing a graph name does not isolate anything on OSS.**
Two datasets loaded into "different" graphs will silently merge.

**Workaround used here:** the loader defaults to `default` and resets the graph
before loading, rather than relying on tenant isolation.

---

## 8. Deleted property columns resurrect onto new nodes

> Filed upstream: [samyama-graph#364](https://github.com/samyama-ai/samyama-graph/issues/364)

**Severity: data integrity. Stale values from deleted data appear on new data.**

```cypher
CREATE (:GhostProp {id: "a", ghost: "LEAKED"});
MATCH (n:GhostProp) DETACH DELETE n;          -- count is now 0
CREATE (:GhostProp {id: "b"});                -- note: no `ghost` property
MATCH (n:GhostProp) RETURN n.id, n.ghost;     -- ["b", "LEAKED"]   <-- expected null
```

A node created without a property inherits the value the *previous* generation
of nodes had for that column. A global `MATCH (n) DETACH DELETE n` does not help
either -- the columnar property store survives the delete.

We hit this for real: an internal `_chain` field briefly leaked onto `Sensor`
nodes, and after the generator was fixed and the graph reloaded, **every Sensor
still reported the stale blob** even though the source data no longer contained
it. It looked like the fix had failed.

**Workaround:** `DETACH DELETE` is not a reset. To genuinely reset a graph,
stop the server and start it against a fresh data directory:

```bash
pkill -f '[t]arget/release/samyama'
rm -rf <data-dir>/samyama_data
samyama --http-port 8080
```

**Corollary:** never rely on `DETACH DELETE` between experiments that change a
node's property *schema*. Changing values is fine; removing a property is not.

### 8b. `<>` against a null property matches

`WHERE s._chain <> ""` returns rows where `s._chain` is null. Standard Cypher
would treat `null <> ""` as null and filter the row out. Use an explicit
`IS NULL` / `IS NOT NULL` check instead of inequality when a property may be
absent.

---

## 9. Aggregating a bare node variable over a multi-variable MATCH does not aggregate

> Filed upstream: [samyama-graph#363](https://github.com/samyama-ai/samyama-graph/issues/363)

**Severity: correctness. Returns N rows of `1` instead of one total.**

```cypher
MATCH (k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WHERE k.execution_provider = "CPUExecutionProvider"
RETURN count(DISTINCT op) AS n
-- 293 rows, each containing 1      <-- expected one row with the operator count
```

Aggregating a *property* works; aggregating the bare node variable does not:

| Query | Result |
|---|---|
| `RETURN count(DISTINCT o)` over a 2-variable MATCH | **3 rows of `1`** |
| `RETURN count(o)` over a 2-variable MATCH | **3 rows of `1`** |
| `RETURN count(DISTINCT o.id)` | 1 row: `2` — correct |
| `WITH count(DISTINCT o.id) AS n RETURN n` | 1 row: `2` — correct |
| `RETURN count(DISTINCT n)` over a 1-variable MATCH | correct |

The aggregate appears to group implicitly by the other bound variable instead of
collapsing the whole result set.

**Rule adopted here:** always aggregate a property, never a bare node variable.
Every `count(DISTINCT x)` in the catalog is written `count(DISTINCT x.id)`.

---

## 10. The 0.6.1 embedded build does not register an alias introduced by a second `WITH`

> Not filed upstream. Tracked here as #56 — unlike notes 1-9 this is a
> disagreement between two builds, not a behaviour of the server.

**Severity: correctness. Raises on the embedded build, correct on the server.**

`SamyamaClient.embedded()` (`samyama` 0.6.1 from pip) and the HTTP server
(`ghcr.io/samyama-ai/samyama-graph:1`, labelled 1.7.0) do not answer the same
question. A second `WITH` that introduces a **new** alias is not registered on
the embedded build. Where the alias comes from does not matter -- a property
expression and an aggregate both fail:

| Statement | embedded | HTTP |
|---|---|---|
| `WITH n.name AS a RETURN a` | ok | ok |
| `WITH count(n.id) AS c RETURN c` | ok | ok |
| `WITH n, count(n.id) AS c RETURN n.name, c` | ok | ok |
| `WITH n WITH n.name AS a RETURN a` | **`Variable not found: a`** | ok |
| `WITH n, count(n.id) AS c WITH n.name AS a RETURN a` | **`Variable not found: a`** | ok |
| `WITH n, count(n.id) AS c WHERE c > 0 WITH n.name AS a RETURN a` | **`Variable not found: a`** | ok |
| `WITH n, count(n.id) AS c WITH n, c, count(n.id) AS d RETURN d` | **`Variable not found: d`** | ok |
| `WITH n, count(n.id) AS c WITH n, c RETURN n.name, c` | ok | ok |

Carrying existing variables through a second `WITH` is fine. It is introducing a
new one that fails, so the last row is what makes this narrow rather than
"chained `WITH` is broken" -- and the row above it is why the rule is about
*new aliases*, not about property expressions.

Minimal reproduction against an otherwise empty graph:

```python
from samyama import SamyamaClient
c = SamyamaClient.embedded()
c.query('CREATE (:Probe {id: "p1", name: "a"})', "default")
c.query("MATCH (n:Probe) WITH n WITH n.name AS a RETURN a", "default")
# RuntimeError: Query error: Variable not found: a
```

The same three lines against `SamyamaClient.connect("http://127.0.0.1:8080")`
return one row.

**Why it bites here.** Two catalog queries hit it, and for different reasons —
worth separating, because a reader fixing one should not assume the other has
the same shape:

| | second `WITH` | the alias it introduces | error |
|---|---|---|---|
| `EA01` | the note-3 projection: `WITH op.name AS operator, ...` | a **property expression** | `Variable not found: operator` |
| `EA02` | an aggregation step: `WITH op, models_using, count(k) AS kernel_count` | an **aggregate** | `Variable not found: kernel_count` |

`EA01` is the case where the workaround for one note triggers another: note 3
says project through a `WITH` before `RETURN` so `ORDER BY` is honoured, and
EA01 already needs a `WITH` for its anti-join, so the projection is a second one.

`EA02` is not that. Its second `WITH` carries `op` and `models_using` forward
*and* introduces `kernel_count` from `count(k)` — it is a genuine aggregation
step, not a projection, and nothing about note 3 is involved. It fails because
the alias is new, which is the rule above.

Both return rows against the server (`run_benchmark` reports 16/16, 0 failed)
and both fail under `pytest`, which uses the embedded build.

**No workaround adopted** — the queries are not rewritten to avoid the shape,
because which engine the suite should treat as authoritative is an open
decision (#56), and rewriting them now would encode a guess as a fix.

What *was* decided: `test_every_catalog_query_runs_and_returns_rows` and
`test_order_by_is_actually_applied` carry `xfail(strict=False)` naming this
note. Both sweep the whole catalog and so hit EA01 and EA02. `pytest` stays
green and the divergence stays visible in every run rather than as red lines
nobody reads; `strict=False` means an XPASS is not a failure, so if the embedded
build starts agreeing the run says so and the marks come off.

The third failing test, `test_ea04_quantization_unlock_is_not_a_cartesian_product`,
is **not** this note — EA04 has a single `WITH`. See note 11.

**`EA18` (#37) chains three `WITH`s, the second and third introducing all-new
aliases, and runs embedded without raising.** That is this note's shape, and on
`samyama` 1.7.1 it does not raise: `tests/test_latency_budget.py` passes
embedded, returning rows on a fixture and zero on the fleet.

That is a statement about `EA18` on 1.7.1, not a verdict on the note. Whether
the note is obsolete is #94's question -- it was written against 0.6.1, and #94
argues the whole embedded-versus-server split was a version skew -- and until
that lands, `tests/test_empty_answers.py::test_ea01_zero_row_case` is still
`xfail(strict=True)` naming this note and `EA01`'s second `WITH`. Declaring the
note false here while a strict test asserts it holds would put the two in
conflict with nothing resolving it. Recorded so the next reader does not
conclude `EA18` is untested against the note.

---

## 11. The two builds disagree on the type of `sum(CASE ... THEN <int> ... END)`, so a `WHERE` on it is dropped

> Not filed upstream. Tracked with note 10 under #56 — a second embedded/HTTP
> divergence, different shape. Verified 2026-08-31.

**Severity: correctness. Silently returns rows a WHERE should have removed.**

Four nodes, two groups, one of which should be filtered out:

```cypher
MATCH (g:WGrp)
WITH g.k AS k, sum(CASE WHEN g.k = "A" THEN 1 ELSE 0 END) AS hits
WHERE hits > 0
RETURN k, hits ORDER BY k
```

| | embedded (`samyama` 0.6.1) | HTTP (v1.7.0) |
|---|---|---|
| `sum(CASE ... THEN 1 ELSE 0 END)` returns | **float** — `2.0`, `0.0` | **int** — `2`, `0` |
| `WHERE hits > 0` (int literal) | **not applied** — both groups | applied — `A` only |
| `WHERE hits > 0.0` (float literal) | applied — `A` only | **not applied** — no rows at all |
| no `WHERE` | both groups | both groups |

The aggregate's *type* is what differs. Note 4 already records that this engine
mis-compares across int and float, so the predicate is silently dropped on
whichever build the literal does not match. Nothing errors either way.

**No bare literal is correct on both**, and that includes the obvious
rephrasings:

| Predicate | embedded | HTTP |
|---|---|---|
| `WHERE hits > 0` | **wrong** — not applied | correct |
| `WHERE hits >= 1` | **wrong** — not applied | correct |
| `WHERE hits <> 0` | **wrong** — not applied | correct |
| `WHERE hits > 0.0` | correct | **wrong** — returns nothing |
| **`WHERE toFloat(hits) > 0.0`** | **correct** | **correct** |

**A workaround does exist**, and it was tried rather than assumed: coercing the
aggregate with `toFloat()` before comparing is right on both builds, because it
removes the type disagreement rather than guessing which side of it the literal
should sit on.

It is not adopted here. Rewriting `EA04` changes the catalog, and #56 has not
decided which build is authoritative — but the decision there is now "adopt
`toFloat()`, or reconcile the builds", rather than "there is no way to write
this".

**What it costs.** `EA04` uses `WHERE fp32_misses > 0 AND int8_hits > 0`, so on
the server it filters and on the embedded build it does not. The unfiltered
groups are single-deployment ones where the only variant is `fp32`, so they
carry `int8_hits = 0.0` and `max(CASE WHEN v.precision = "int8" ... ELSE 0 END)`
correctly returns the `ELSE` sentinel:

```
EA04 on the HTTP server        EA04 on the embedded build
fp32_kb    int8_kb             fp32_kb    int8_kb   int8_hits
 3347.6      836.9              18125.6         0        0.0
  558.0      139.5              10224.4         0        0.0
  305.6       76.4               9929.2         0        0.0
```

The server's rows are right — `int8_kb` is exactly `fp32_kb / 4`. The embedded
rows are groups that should never have reached `RETURN`.

**Why `test_ea04_shape_is_not_a_cartesian_product` still passes** on the same
embedded build, despite the same four conditional aggregates and the same
`WHERE`: its purpose-built fixture is four deployments in two groups, and *every*
group satisfies `fp32_misses > 0 AND int8_hits > 0`. Whether the predicate is
applied or not, the same two rows come back. The divergence is invisible to any
query whose groups all pass — which is why it surfaces on the generated graph
and not on the fixture.

**A workaround is known and deliberately not adopted.** `toFloat()` on the
aggregate, above, is correct on both builds -- so unlike note 10 this is not
"there is no way to write this". It is not applied because rewriting `EA04`
changes the catalog, and #56 has not decided which build is authoritative;
adopting it now would settle that question by the back door. The distinction
matters: *fix deferred* and *no fix known* are different states, and someone
reading #56 should not re-derive `toFloat()` from scratch. See the mark on
`tests/test_correctness.py::test_ea04_quantization_unlock_is_not_a_cartesian_product`.

---

## 12. The 1.7.0 server does not traverse variable-length relationships; embedded 1.7.1 does

> Not filed upstream yet. Found 2026-09-10 while adding `EA17` (#35). Like
> notes 10 and 11 this is a build disagreement rather than a behaviour of one
> engine -- but unlike them it is the **server** that is wrong, and it is
> silent.
>
> **Version labels, because this file carries two vintages.** The rest of this
> page describes `samyama` **0.6.1** embedded against the **1.7.0** server,
> which is what `pyproject.toml`'s `samyama>=0.6.0` resolved when notes 1-11
> were written. This note was measured against **1.7.1** embedded, the floor
> #105 pins (landing with #104). So the comparison below is 1.7.0 server against 1.7.1 embedded, and
> the conclusion "the server is the one that is wrong" is really "the server at
> 1.7.0 does not do what the embedded build at 1.7.1 does". Whether 1.7.0
> *embedded* traverses has not been measured; there is no reason to think the
> defect is HTTP-specific rather than a fix that landed between the two.

**Severity: correctness. Returns fewer rows rather than erroring.**

A four-node chain `a -> b -> c -> d`, loaded identically into both builds, then
`MATCH (a:P)-[:R<pattern>]->(b:P) WHERE a.id = "a" RETURN b.id`:

| pattern | server 1.7.0 | embedded 1.7.1 |
|---|---|---|
| `*0..` | `['a']` | `['a', 'b', 'c', 'd']` |
| `*0..3` | `['a']` | `['a', 'b', 'c', 'd']` |
| `*1..3` | `[]` | `['b', 'c', 'd']` |
| `*` | `[]` | `['b', 'c', 'd']` |
| `*0..1` | `['a']` | `['a', 'b']` |
| `*2..2` | `[]` | `['c']` |

**The server matches the zero-length case and nothing else**, for every form,
bounded or not. It does not error; it returns a smaller answer.

`size(r)` on a variable-length relationship is a second, louder symptom on the
same build: `Type error: size() requires string, list, or path`. `length(p)`
and `size(relationships(p))` parse there, and then report the zero-length
result, which is worse.

**What it costs the catalog, measured rather than reasoned.** `EA07` and
`EA17` are the two queries with a variable-length pattern. `EA17` raises. **For
`EA07` the two builds return byte-identical rows** -- checked on one scale-1.0
graph loaded into both, comparing row content and not row counts.

That is not what this note predicts, and the prediction is what was wrong. An
earlier draft here said `EA07` "silently drops everything deeper", reasoned from
the table above and not measured. `EA07` ends `ORDER BY latency_ms ASC LIMIT
10`, and the ten lowest-latency paths are reachable at zero hops on this graph,
so the smaller candidate set produces the same answer. That is luck, not
robustness: it holds for this data and this `LIMIT`, and nothing enforces it.

**No workaround.** There is no way to express "walk a chain of unknown length"
that the 1.7.0 server executes. The options are an engine that does it
(embedded 1.7.1 does), or not asking the question over HTTP.

**`EA17` therefore depends on #105**, the floor raise that lands with #104. It needs an engine at 1.7.1, both for this
and because its per-leg second `WITH` introduces new aliases -- note 10's shape,
which 0.6.1 rejects. The floor in `pyproject.toml` is what decides this, and it
is `samyama>=0.6.0` until #104 merges -- so today it admits a build on which
`EA17` fails outright. In practice pip resolves 1.7.1 (published 2026-08-27),
which is why the suite is green.

When #104 merges, that floor becomes `>=1.7.1` and this paragraph's "today"
stops being true. Nothing in the suite reads the floor, so nothing will say so
-- re-read this note when #104 lands. (Naming a test here that pinned it would
be the better fix; there isn't one, and claiming otherwise is how a page starts
asserting enforcement it does not have.)

---

## 13. A `WITH` alias in an `OPTIONAL MATCH`'s `WHERE` turns it into an inner join

Measured 2026-09-15 while writing `EA18` (#37). **Silently returns wrong rows.**

An `OPTIONAL MATCH` whose `WHERE` compares against a **literal** behaves
correctly: rows with no match survive with `NULL`, and `count()` over them is 0.
An `OPTIONAL MATCH` whose `WHERE` mentions an alias introduced by an earlier
`WITH` drops the unmatched rows instead. The `OPTIONAL` is gone. No error.

```cypher
// two operators; only `op:c` has a kernel
MATCH (op:Operator)
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op)
WHERE k.name = "Relu-npu"                 // literal
WITH op.id AS o, count(k) AS n RETURN o, n
// -> [("op:a", 0), ("op:c", 1)]          correct

// one Kernel in this fixture, so `want` is a single row
MATCH (x:Kernel) WITH x.name AS want
MATCH (op:Operator)
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op)
WHERE k.name = want                       // the only change
WITH op.id AS o, count(k) AS n RETURN o, n
// -> [("op:c", 1)]                       `op:a` is gone
```

The single `Kernel` is deliberate: with more, that first `MATCH` yields one row
per kernel and the result multiplies, which obscures the drop rather than
changing it. The constant-alias form below shows the same thing with nothing to
multiply.

The trigger is an alias introduced by a **`WITH`**, and not what it holds:
carrying a plain constant through one (`WITH op, "accel:npu" AS accel_id`) and
comparing against that reproduces it exactly. The pattern shape does not matter
-- single-pattern and comma-separated `OPTIONAL MATCH` both do it.

**A `MATCH`-bound alias is safe**, which is a narrower rule than "anything that
is not a literal" and had to be measured rather than assumed:

```cypher
MATCH (t:T)-[:R]->(s:S) WHERE s.id = "s:1"
OPTIONAL MATCH (t)-[:R]->(o:O)
WHERE o.modality = s.modality        // `s` comes from the MATCH
WITH t.id AS task, count(o) AS alts RETURN task, alts
// -> [("t:1", 0), ("t:2", 1)]        correct

MATCH (t:T)-[:R]->(s:S) WHERE s.id = "s:1"
WITH t, s.modality AS mine           // the same value, through a WITH
OPTIONAL MATCH (t)-[:R]->(o:O)
WHERE o.modality = mine
WITH t.id AS task, count(o) AS alts RETURN task, alts
// -> [("t:2", 1)]                    `t:1` is gone
```

That matters for the catalog: `EA17`'s `ClinicalTask` leg filters
`o.modality = s.modality` inside an `OPTIONAL MATCH`, with `s` bound by the
opening `MATCH`. It is the first form, so it is correct -- and it is the reason
the rule has to be stated as "a `WITH` alias" rather than "a non-literal".

**Why this is the dangerous kind.** The whole point of the shape is the
anti-join: `OPTIONAL MATCH ... WITH count(k) AS n ... WHERE n = 0`, which note 5
forces on us because negated patterns do not parse. The rows this drops are
exactly the rows with no match -- the answer. A "what is missing" query comes
back saying nothing is missing, which is a plausible, reassuring, wrong answer.

**Workaround used here:** none is needed in the catalog. Its anti-joins compare
against a literal or against a `MATCH`-bound alias, and neither triggers this.
Where a `WITH` alias is genuinely required, collect and filter after the
aggregation, which is correct on this build:

```cypher
WITH op.id AS o, want, collect(k.name) AS names
WITH o, [n IN names WHERE n = want] AS hit
RETURN o, size(hit)
// -> [("op:a", 0), ("op:c", 1)]          correct
```

**This is why `EA18` re-binds `a` in trailing position** rather than carrying
`a.id` through a `WITH` and filtering on it. The second spelling is the one note
1 would prefer, and it is unavailable: it silently drops every deployment whose
operators are uncovered, which is the entire answer. Measured both ways on the
same fixture -- the trailing re-bind returns the two uncovered operators, the
`WITH`-alias form returns the deployment as though nothing were uncovered.

---

### 13b. A grouping key mixed with an aggregate in one projection is NULL

Same session, same query. Also silent.

```cypher
MATCH (n:N) WITH n.v AS v, min(n.v) AS lo, n.v - min(n.v) AS diff RETURN v, lo, diff
// -> [[250.0, 250.0, NULL], [90.0, 90.0, NULL]]      the arithmetic is gone

MATCH (n:N) WITH n.v AS v, min(n.v) AS lo WITH v, lo, v - lo AS diff RETURN v, lo, diff
// -> [[250.0, 250.0, 0.0], [90.0, 90.0, 0.0]]        correct
```

Aggregate-minus-aggregate in one projection is fine
(`max(n.v) - min(n.v)` returns 160.0). It is specifically a **grouping key**
referenced in the same projection as an aggregate over the group.

**Severity is lower than 13's** -- a NULL column is visible, where a dropped row
is not. But `ORDER BY` on that column then sorts on nothing, and with `LIMIT`
that is note 3's failure again: an arbitrary N dressed as a top-N.

**Workaround used here:** compute it one `WITH` later, which is what `EA18`
does for `over_by_ms`.

---

## What works well

Everything the catalog depends on, other than the above:
`OPTIONAL MATCH`, `WITH` + aggregation (`count`/`sum`/`avg`/`min`/`max`/`collect`)
— though see note 10 on chaining two of them against the embedded build —
variable-length paths (`-[:R*0..3]->`), `shortestPath`, `CASE`, `IN`, `SKIP` / `LIMIT`, string functions, `EXPLAIN`, and `CREATE INDEX`.

## Load throughput, measured (#10)

`python -m benchmarks.ingest` reports this, so it is a re-runnable artifact
rather than a sentence here. On this box (RTX 4050 laptop), embedded build,
`--scale 1.0`, 25,150 nodes and 76,303 edges, medians of 3:

| | rate | batch |
|---|---:|---|
| nodes | **~48K/s** on 1.7.1 (52-53K on 0.6.1) | 250 per `CREATE` |
| edges | **~3.1K/s** | 50 per statement |

**The per-item gap is ~17-21x, not the 11x this file used to imply.** The old
figures paired a stale node rate (~35K/s) with the edge rate, which understated
it. The gap moves with scale -- 18x at `--scale 0.3`, 21x at 1.0 -- because the
edge cost grows with graph size while the node cost does not.

### Why edges are slower, and what it is *not*

A node `CREATE` writes. An edge `CREATE` **looks up two endpoints and then
writes**: `etl/helpers.py:create_edges` emits one `MATCH` pattern per distinct
endpoint in the batch, constrains them in one `WHERE`, then creates the
relationships. A 50-edge batch is one statement carrying ~37 patterns.

That statement is the entire cost, and it is **superlinear in patterns per
statement**:

| edges/batch | patterns/stmt | ms/stmt | edges/s |
|---:|---:|---:|---:|
| 40 | 30 | 13.3 | 3,010 |
| 50 | 37 | 15.9 | **3,140** |
| 60 | 44 | 19.1 | **3,146** |
| 75 | 54 | 26.0 | 2,886 |
| 100 | 71 | 38.5 | 2,594 |
| 200 | 136 | 128.6 | 1,553 |

4.5x the patterns costs 9.7x the time, roughly `O(p^1.5)` steepening toward
`O(p^2)`. Node batching is **flat** by contrast -- 52-53K/s at every size from
100 to 2000 on `samyama` 0.6.1, ~48K/s across the same range on 1.7.1 -- which
is the control that isolates the cause to the lookup rather than to writing.
Flat is the finding; the absolute figure moved with the engine, as every other
number on this page did.

`create_edges` batches 50 for this reason; it was 100, which costs 21% of edge
throughput. Reproduce with `python -m benchmarks.ingest --sweep-edge-batch`.
This is a property of this build's join planning, so re-measure before assuming
the optimum has not moved.

**Two things that do not help.** Reordering edges: the `Fleet`'s natural order
already dedups endpoints well (56,823 patterns at batch 50 against a 152,606
ceiling), sorting by source is a wash, and sorting by target or relationship
type is 38% *worse* because it breaks up runs of edges sharing a source. Bigger
batches: total patterns actually *fall* as batches grow (51,112 at 250 vs 56,823
at 50) while time doubles, which is what rules out pattern count as the driver.

**The `id` indexes dominate everything above.** Without them the same edge load
takes 234s instead of 22s, a 10.6x penalty on `samyama` 0.6.1 -- re-measured at
**6.4x** on 1.7.1 (161s against 25s), reproducible with
`python -m benchmarks.ingest --no-indexes` -- because each endpoint lookup becomes
a label scan (#18). The numbers here all assume `schema/edge_ai_kg.cypher` has
been applied.
