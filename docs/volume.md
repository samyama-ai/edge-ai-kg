# Volume above `--scale 1.0`, measured

Closes #11.

Every published figure in this repo is at `--scale 1.0` — 25,150 nodes and
76,303 edges. `etl/generate.py` accepts any scale and nothing recorded what
happens above one, while the pitch rests on volume being a strength.

So it was run. Embedded build, `samyama` 0.6.1, seed `20260814`, both layers,
median of 3 per query.

## What happens as the fleet grows

| `--scale` | nodes | edges | load | edges/s | catalog total |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 25,150 | 76,303 | 19.5 s | 3,909 | 593 ms |
| 2.0 | 48,907 | 152,717 | 43.4 s | 3,518 | 1,132 ms |
| 5.0 | 155,660 | 505,419 | 190.0 s | 2,661 | 4,659 ms |
| 10.0 | 366,366 | 1,249,150 | 601.6 s | 2,076 | — |

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
about. It is the only catalog query whose cost grows faster than the graph, and
the shape is the plausible reason.

**At 10.0 it is 4.4 s** — slow but usable, and the gap to `EA08` is now 3.2x
where at scale 1.0 the two were identical. This is a trajectory rather than a
problem today; extrapolating the same exponent to 100x puts `EA11` in minutes
while `EA08` stays in seconds.

## Correctness at scale — the half that matters more

Note 1 says a join bug "only appears once the cardinalities are real" and that
"a passing 6-node reproduction proves nothing". Timing a query that returns wrong
rows is worthless, so `EA11` was checked against ground truth recomputed in
Python from the `Fleet` at every scale:

| `--scale` | nodes | edges | EA11 row counts | EA11 operator lists | int8 = ¼ fp32 |
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
**verified correct against Python ground truth to 6x** (505,419 edges).

### Memory is the real ceiling, and it qualifies the footprint claim

At 10.0 the process holds **2,415 MB** resident, against ~199 MB at scale 1.0
(`docs/footprint.md`) — ×12.1 for ×16.4 edges, so roughly 2 KB per edge and
mildly *sub*-linear.

That matters because `docs/footprint.md` argues the footprint win against
**Neo4j's 2 GB heap guidance**. At 10x this engine is *past* that line. The
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
python -m benchmarks.ingest --scale 5.0 --repeats 1
python -m benchmarks.run_benchmark          # per-query latency
```

`etl.download_data --scale 5.0` rebuilds `data/` at another size; note that the
published counts in the other documents are all scale 1.0, and
`tests/test_published_counts.py` compares against a scale-1.0 rebuild.
