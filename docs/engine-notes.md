# Engine notes -- Samyama Graph, measured on v1.7.0

> **Ten of the eleven notes were re-measured on `samyama` 1.7.1 (embedded) on
> 2026-09-08, and none of those ten reproduces.** Notes 10 and 11 were resolved
> by #56; notes 1-6, 8 and 9 were re-probed while checking that claim and are
> gone too, note 1 included at cardinality 300, where a cartesian product would
> return 90,000 rows instead of 300. The sub-behaviours the rules below depend
> on -- 3b (only the first `ORDER BY` key) and 8b (`<>` matching nulls) -- were
> probed separately and are also gone.
>
> **Note 7 is the exception, and is not measured either way.** It is a property
> of the OSS *server's* HTTP path, and the probe runs embedded, where there is
> no tenant boundary to ignore. Do not read "ten of eleven" as evidence about
> it. Re-run the rest yourself:
>
> ```bash
> python -m benchmarks.engine_notes_probe --scale 300
> ```
>
> `--scale` is not optional for a meaningful answer: without it note 1 reports
> INCONCLUSIVE, because its own text below says a 6-node reproduction proves
> nothing and the default fixture is exactly that.
>
> **The workarounds stay for now, and this file is not yet history.** Two
> reasons, both narrow. Notes 1-9 were measured against the **1.7.0 HTTP
> server**, and the probe runs **embedded** -- a different binary, which is
> precisely the assumption #56 punished. And unwinding the workarounds means
> editing the catalog, the loader and `etl/helpers.py`, which is a change that
> deserves its own review rather than riding along inside a version pin.
>
> So: read the rules below as still binding, and treat their *justification* as
> pending re-measurement against the server. Whoever picks that up should start
> from the probe, not from this prose -- asserting "notes 1-9 are unaffected"
> without running anything is the mistake that produced this banner.

Behaviour observed while building this KG.
**Notes 1-9 are filed upstream** — see the tracking issue
[samyama-graph#368](https://github.com/samyama-ai/samyama-graph/issues/368).

Notes 1-9 were observed on the OSS engine at v1.7.0
(`target/release/samyama --http-port 8080`). Every one of them is load-bearing:
the loader or the query catalog works around it. Verified 2026-08-14.

**Versions these describe.** The server is
`ghcr.io/samyama-ai/samyama-graph:1`, labelled 1.7.0; the embedded build is
`samyama` 1.7.1 from pip, pinned by `pyproject.toml` as `samyama>=1.7.1`.

**Notes 10 and 11 are resolved, and kept as history (#56).** They were never
behaviours of the server. They were disagreements between the server and the
in-process embedded build — and the cause turned out to be version skew rather
than a design difference: `pyproject.toml` asked for `samyama>=0.6.0` unpinned
and resolved **0.6.1**, an engine two minor versions behind the 1.7.0 server the
other notes were measured against. `SamyamaClient.embedded().status()` reports
the engine's own version, which is what made this visible.

Upgrading the embedded build to 1.7.1 makes both notes stop reproducing, and
`#56`'s question — *which build is authoritative?* — dissolves rather than gets
answered. There was never a disagreement between two modes of one engine to
adjudicate; there were two engines. The four tests they excused now pass
unmarked, and `tests/test_engine_version.py` holds the floor, re-running note
11's reproduction so a downgrade fails loudly instead of silently.

Four test functions, six xfail reports in the run output — the two sweeps were
parametrised, so each contributed two:

| Test | Was excused | Mark | Note | Now |
|---|---|---|---|---|
| `test_every_catalog_query_runs_and_returns_rows` | `[EA01]`, `[EA02]` | `strict=False` | 10 | passes |
| `test_order_by_is_actually_applied` | `[EA01]`, `[EA02]` | `strict=False` | 10 | passes |
| `test_ea04_quantization_unlock_is_not_a_cartesian_product` | whole test | `strict=True` | **11** | passes |
| `test_empty_answers.py::test_ea01_zero_row_case` | whole test | `strict=True` | 10 | passes, and gained the control it lacked |

**The two mark kinds reported the fix differently, and both were deliberate.**
The `strict=False` sweeps reported XPASS — a note, not a failure, so the suite
stayed green while the divergence stayed visible. The two `strict=True` tests
*failed* the run as `XPASS(strict)`, which is what strictness is for: each
guards a filter that silently drops rows, and a non-strict mark there could
outlive the fix and leave the answer unchecked. So the first pair said "you can
remove us" and the second pair said "you must".

Those marks are gone. The parametrised pair is described here because *how* it
was written is why this closed cleanly: the sweeps were parametrised over the
catalog so only the affected queries were excused, and the marks were applied
with `request.applymarker` rather than `pytest.xfail()`. The imperative form
never runs the body, so a parameter could only ever report XFAIL and the XPASS
that says "the divergence is gone, remove the mark" would never have arrived. It
did arrive, on the first run against 1.7.1.

Verified 2026-08-28 and 2026-08-31; resolved and re-verified 2026-09-08.

Notes 1-9 keep their workarounds, but "unaffected" would be too strong: none of
notes 1-6, 8 and 9 reproduces on embedded 1.7.1 either (see the banner at the
top of this file; note 7 is not probed either way). They were measured against
the 1.7.0 *server*, which has not been re-probed, so the rules stay binding
until it is.

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

## 10. `samyama` 0.6.1 did not register an alias introduced by a second `WITH`

> **RESOLVED by #56 — version skew, not a build difference. 1.7.1 is the floor.**
> Not filed upstream; there was nothing to file. This was an old embedded engine
> against a newer server, not a defect in either. Kept because it explains why
> `EA01` and `EA02` carried `xfail` marks, and because the *shape* below is worth
> knowing if anyone meets a 0.6.x build again.

**Severity when live: correctness. Raised on the embedded build, correct on the
server.** All eight rows below now agree; verified on 1.7.1 with the same
statements.

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

Both returned rows against the server (`run_benchmark` reported 16/16, 0 failed)
and both failed under `pytest`, which uses the embedded build. On 1.7.1 they
agree.

**No workaround was ever adopted**, and that turned out to be the right call —
the queries were not rewritten to avoid the shape, because which engine the
suite should treat as authoritative looked like an open decision (#56). Had
`EA01` and `EA02` been rewritten to appease 0.6.1, the catalog would now carry
two contorted queries working around an engine the repo no longer installs.

What *was* decided: `test_every_catalog_query_runs_and_returns_rows` and
`test_order_by_is_actually_applied` carried `xfail(strict=False)` naming this
note, on the two parameters only. `strict=False` meant an XPASS was not a
failure, so if the embedded build ever started agreeing, the run would say so
and the marks could come off.

**That is exactly what happened.** Upgrading to 1.7.1 turned all four
parameters XPASS in one run, which is how this note was closed rather than by
anyone re-deriving it. It is the argument for `strict=False` on a divergence you
do not control: a mark that can only ever report XFAIL tells you nothing on the
day the bug is fixed.

The third failing test, `test_ea04_quantization_unlock_is_not_a_cartesian_product`,
was **not** this note — EA04 has a single `WITH`. See note 11.

---

## 11. `samyama` 0.6.1 typed `sum(CASE ... THEN <int> ... END)` as float, so a `WHERE` on it was dropped

> **RESOLVED by #56 — version skew, not a build difference. 1.7.1 is the floor.**
> Same cause as note 10: version skew, not a design difference. Verified
> 2026-08-31; re-verified resolved 2026-09-08.
>
> **This is the one to remember.** Note 10 *raised*; note 11 returned confident
> wrong rows and nothing errored. `tests/test_engine_version.py` re-runs the
> reproduction below on every test run for exactly that reason — a silent bug
> needs a check that does not depend on someone noticing.

**Severity when live: correctness. Silently returned rows a WHERE should have
removed.** On 1.7.1 the aggregate is an `int` and `WHERE hits > 0` filters
correctly; `WHERE hits > 0.0` is also correct now, where the 1.7.0 server
returned no rows for it.

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

It was not adopted, and #56 took the second of its two options — *reconcile the
builds* rather than *adopt `toFloat()`*. That is the better outcome: `toFloat()`
would have left `EA04` carrying a coercion whose only purpose was to paper over
a version gap, and the catalog would read as though the engine still could not
type an aggregate.

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

**A workaround was known and deliberately not adopted.** `toFloat()` on the
aggregate, above, is correct on both builds -- so unlike note 10 this was never
"there is no way to write this". Holding it back kept `EA04` unmodified until
#56 resolved, and #56 resolved by upgrading the engine, so the coercion was
never needed. The distinction was the useful part: *fix deferred* and *no fix
known* are different states, and recording which one this was is what let #56
weigh "rewrite the query" against "reconcile the builds" instead of assuming the
query had to change.

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
