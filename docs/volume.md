# Volume above `--scale 1.0`, measured

Closes #11.

Every published figure in this repo is at `--scale 1.0` — 25,150 nodes and
76,303 edges. `etl/generate.py` accepts any scale and nothing recorded what
happens above one, while the pitch rests on volume being a strength.

So it was run. Embedded build, `samyama` 0.6.1, seed `20260814`, both layers.

**Which command produced which column:** the node/edge counts and load times come
from a scripted load equivalent to `python -m benchmarks.ingest`; the per-query
latencies are the median of 3 runs of each catalog query at 1.0/2.0/5.0, and of
**2** runs at 10.0, where a single load takes ten minutes. The `catalog total`
column is blank at 10.0 because only four queries were timed there, not all
sixteen — the four are listed in the EA11 table below.

## What happens as the fleet grows

| `--scale` | nodes | edges | load | edges/s | catalog total |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 25,150 | 76,303 | 19.5 s | 3,909 | 593 ms |
| 2.0 | 48,907 | 152,717 | 43.4 s | 3,518 | 1,132 ms |
| 5.0 | 155,660 | 505,419 | 190.0 s | 2,661 | 4,659 ms |
| 10.0 | 366,366 | 1,249,150 | 601.6 s | 2,076 | not run — see below |

Relative to scale 1.0:

| `--scale` | nodes | edges | load time | resident memory |
|---:|---:|---:|---:|---:|
| 2.0 | ×1.9 | ×2.0 | ×2.2 | — |
| 5.0 | ×6.2 | ×6.6 | **×9.7** | — |
| 10.0 | ×14.6 | ×16.4 | **×30.9** | ×12.1 |

**A million edges load and answer.** At 10.0 the graph holds 366,366 nodes and
1,249,150 edges — 16x the published size — and `count(n.id)` returns exactly the
366,366 the `Fleet` says. It takes ten minutes.

**`--scale` is not linear in either count.** 5.0 produces 6.2x the nodes and
6.6x the edges, because `etl/generate.py` multiplies several independent
dimensions. Read the node and edge columns, not the scale number.

### Loading degrades; querying roughly tracks size

Ingest throughput falls **3,909 → 2,661 → 2,076 edges/s**, a 47% loss by 10.0,
so load time grows much faster than the graph — ×30.9 for ×16.4 edges. That is
consistent with the endpoint-lookup cost in #10: every edge is two indexed
lookups, and the index keeps growing.

Node loading stays cheap throughout (7.2 s for 366,366 nodes at 10.0), which is
the same nodes-write-vs-edges-look-up split #10 isolated.

At 5.0 the catalog as a whole grows about with the edge count (×7.9 for ×6.6),
which is the unexciting and reassuring result.

### Except EA11, which is the one to watch

`benchmarks/README.md` named `EA08` and `EA11` as the queries touching every
kernel. They diverge:

| query | 1.0 | 2.0 | 5.0 | 10.0 | growth vs ×16.4 edges |
|---|---:|---:|---:|---:|---|
| `EA08` | 118 ms | 220 ms | 664 ms | 1,365 ms | ×11.6 — sub-linear |
| `EA11` | 118 ms | 264 ms | 1,376 ms | **4,426 ms** | **×37.5 — superlinear** |
| `EA05` | — | — | — | 517 ms | |
| `EA16` | — | — | — | 250 ms | |

`EA11` is the anti-join written as
`OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)`
— the comma-separated multi-variable shape `docs/engine-notes.md` note 1 warns
about. Until `EA17` arrived it was the only catalog query whose cost grows
faster than the graph, and the shape is the plausible reason. `EA17` is now
steeper still — see below.

**At 10.0 it is 4.4 s** — slow but usable, and the gap to `EA08` is now 3.2x
where at scale 1.0 the two were identical. This is a trajectory rather than a
problem today; extrapolating the same exponent to 100x puts `EA11` in minutes
while `EA08` stays in seconds.

### `EA17` is the most expensive query in this table, and grows faster than `EA11`

> **Everything in this section predates the bound.** `EA17`'s walks carried
> no upper hop limit when these were taken; they carry `MAX_STAGE_HOPS` now
> (engine note 14, #126) and have not been re-timed. That covers the table
> below, the ×4.5 per-doubling rate, and everything derived from it — the
> "passes a second just under scale 3" and "reaches `EA11`'s figure near
> scale 6" extrapolations, and `EA17`'s share of the catalog total.
>
> What the bound does to these numbers is **not** measured. The argument that
> it can only remove paths rests on depth, and depth was measured at both of
> this table's scales: **5 hops at `--scale 1.0`, 4 at 2.0** — so 5 is the
> deepest either column reaches, well inside a bound of 8. It does not grow
> with `--scale`, because the stage pool does not: 16 stages and 14 sensors at
> 0.3, 1.0 and 2.0 alike.
>
> That says the *answers* are unchanged. It does not say what the planner does
> with a bounded pattern, and one that takes a different strategy when given an
> upper limit would make these figures wrong in either direction.
> `tests/test_bounded_walks.py::test_bounding_changes_no_answer` checks the
> answers, but on a `--scale 0.3` fleet — **not** the fleets timed here — so it
> is evidence about the shape of the claim, not about these two columns.
> Re-running `python -m benchmarks.run_benchmark` at both scales is what would
> settle the timings.

Added with `EA17` (issue #35, PR #96) and measured separately, because the run
above predates it. Same machine, **embedded**, schema applied, catalog warmed,
median of 5, 2026-09-10.

Run mode is part of the result rather than a detail: on the **1.7.0** server
`EA17` raises as soon as its traversal binds anything (engine note 12), so
these figures cannot be reproduced over `--url` against that build. Where
nothing binds — `--layers real`, which has no `Sensor` — it returns 0 rows
*without* erroring, so "cannot run over HTTP at all" would be too strong.

Whether a **1.7.1 server** still refuses it is unmeasured: the published image
`ghcr.io/samyama-ai/samyama-graph:1` is 1.7.0, and no 1.7.1 server has been
run here. Note 12's title names 1.7.0 for that reason, and this paragraph
should not be read as covering a build nobody has tested.

**Different engine build from the rest of this page.** Everything above was
measured on `samyama` **0.6.1**, the build in use when that table was taken;
this one on **1.7.1**, which `pyproject.toml` has floored since #104.
Absolute milliseconds across the two tables are not comparable — `EA11` reads
33 ms here against 118 ms above, `EA08` 48 ms against 118 ms — and **the growth
columns are not directly comparable either**, because the two tables span
different ranges: ×16.4 edges above, ×2 here. What carries across is the
*direction*, and the per-doubling rate once normalised: `EA11` grows ×2.45 per
doubling above and ×2.97 here, `EA08` ×1.83 and ×2.06. Both tables agree that
`EA11` is superlinear and `EA08` is close to linear; neither licenses reading
×37.5 against ×3.0.

| query | 1.0 | 2.0 | growth |
|---|---:|---:|---|
| `EA17` | **95 ms** | **431 ms** | **×4.5 — superlinear** |
| `EA11` | 33 ms | 98 ms | ×3.0 — superlinear |
| `EA08` | 48 ms | 99 ms | ×2.1 — about linear |
| catalog total (the 17 queries that existed then, `EA17` included) | 270 ms | 866 ms | ×3.2 — superlinear, but see below |

The graph doubles between those columns — 76,303 edges at 1.0 against 152,717
at 2.0, and 25,150 nodes against 48,907 — so ×2 is the linear line. Everything
above it is superlinear in the graph, which is what the column is for.

**The catalog total is mostly one query.** Of the 596 ms it gains between the
columns, `EA17` accounts for 336 — **56%** — so ×3.2 describes a catalog
carrying `EA17` rather than sixteen queries each growing that way. Remove it
and the remaining sixteen go 175 ms → 435 ms, ×2.5.

**An earlier version of this table was measured without indexes** and is
withdrawn: it read `EA17` 105/480 ms and a 340 ms catalog total, and had
`EA13` at 22.8 ms where the indexed figure is 0.7 ms. The load helper used here
did not call `apply_schema`, the same omission that invalidated the first Neo4j
comparison in #47. Both tables above are indexed.

`EA17` is the slowest of the queries in this table at both sizes, and the only
one whose cost grows faster than `EA11`. `EA18` and `EA19` were added after
this sweep and are not in it, so "slowest in the catalog" is a claim this page
cannot make about them. Both are cheap *relative to `EA17`*, measured in one
run on the shipped fleet at scale 1.0 (median of 5 after 3 warm-ups, same
process): `EA17` 691 ms, `EA18` 65 ms, `EA19` 0.1 ms, returning 5, 0 and 4 rows
respectively. Those absolutes come from a different machine than the tables
above and are not comparable with them — the ratio within the one run is what
"cheap" means here. Neither has been run at 2.0.

**`EA17`'s shape is the candidate explanation** for its growth: **four of its
five legs** carry a variable-length `NEXT_STAGE` walk in the main pattern
*and* another inside an `OPTIONAL MATCH` — the fifth,
`(:ClinicalTask)-[:REQUIRES_SENSOR]->(:Sensor)`, has no variable-length hop at
all — over a `NEXT_STAGE` graph that is cyclic:
`etl/generate.py` samples each sensor's chain from one shared pool in random
order, so one sensor contributes `s7 -> s1` and another `s1 -> s7`.
Relationship-uniqueness stops it looping forever. **Why that is expensive is
inferred, not measured**: `min(size(r))` asks for one shortest path per
endpoint, and a planner doing a breadth-first search would not need to see the
others — so the ×4.5 growth is consistent with enumerating them, but nothing
here inspects a plan. The engine exposes no `EXPLAIN`, so confirming it would
mean instrumenting the engine rather than the query.

**Not run above 2.0, and the extrapolation rests on two points.** Both figures
are well inside a demo's patience. Taking `EA17`'s ×4.5 per doubling — which is
one interval, 1.0 to 2.0, so a rate and not a curve — 431 ms at 2.0 passes a
second just under scale 3, and reaches `EA11`'s 4.4-s-at-10.0 figure near scale
6. Both of those are arithmetic on a single measured ratio, not
measurements: a planner that changes strategy at some cardinality would break
them in either direction. Anyone loading a larger fleet should time it rather
than trust this paragraph.

## Correctness at scale — the half that matters more

Note 1 says a join bug "only appears once the cardinalities are real" and that
"a passing 6-node reproduction proves nothing". Timing a query that returns wrong
rows is worthless, so `EA11` was checked against ground truth recomputed in
Python from the `Fleet` at every scale:

| `--scale` | nodes | edges | EA11 row counts | EA11 operator lists | EA04 canary (int8 = ¼ fp32) |
|---:|---:|---:|---|---|---|
| 1.0 | 25,150 | 76,303 | match | match | 0 violations |
| 2.0 | 48,907 | 152,717 | match | match | 0 violations |
| 5.0 | 155,660 | 505,419 | **match** | **match** | 0 violations |

**The join holds at 505,419 edges.** Both the ordered counts and the collected
operator names agree with Python, and the `EA04` canary — int8 variants are by
construction exactly a quarter of their fp32 size, so a cartesian product is
detectable — shows no violation at any scale.

That is evidence against the specific fear, not a general guarantee: note 1's
bug was found in a *different* query shape, and nothing here exercises the
`WITH`-between-two-`MATCH` form it describes.

## What this does and does not establish

**Established:** the graph **loads and stays queryable at 16x the published
size** — 366,366 nodes, 1,249,150 edges, on a laptop in one process — and is
**verified correct against Python ground truth to 505,419 edges** (x6.6).

### Memory is the real ceiling, and it qualifies the footprint claim

At 10.0 the process holds **2,415 MB** resident, against ~199 MB at scale 1.0
(`docs/footprint.md`) — ×12.1 for ×16.4 edges, so roughly 2 KB per edge and
mildly *sub*-linear.

That matters because `docs/footprint.md` argues the footprint win against
**Neo4j's 2 GB heap guidance**. At 16x the edge count this engine is *past*
that line. The
honest statement is therefore narrower than that page alone implies:

> The footprint advantage is real **at the size this repo ships**. It is gone by
> 16x the published edge count, where the graph needs more than the Neo4j heap
> it is being compared against.

Nothing here says Neo4j would fit 1.25M edges in 2 GB either — that comparison
has not been run (#47). What it does say is that "small footprint" is a claim
about scale 1.0 and should not be quoted without one.

**Not established:**

- **100x.** Not run. The largest measured point is 10.0, at 1.25M edges.
- **Correctness above 5.0.** The ground-truth comparison above was run to 5.0.
  At 10.0 only the shape was checked — `EA11` returns its 10 rows and the node
  count is exact — not the row contents.
- **Any figure on the HTTP build.** All of this is embedded. Note 8 means a
  server needs a wiped data directory between configurations to be comparable
  (see `benchmarks/ingest.py`).
- **That EA11 is *wrong* at scale.** It is slower, and correct.

## Reproducing

```bash
python -m benchmarks.ingest --scale 5.0 --repeats 3   # nodes/s, edges/s, load
python -m benchmarks.run_benchmark --repeats 3       # per-query latency
```

`etl.download_data --scale 5.0` rebuilds `data/` at another size; note that the
published counts in the other documents are all scale 1.0, and
`tests/test_published_counts.py` compares against a scale-1.0 rebuild.
