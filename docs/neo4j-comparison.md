# Head to head against Neo4j, losses first

`benchmarks/run_benchmark.py` has always timed the catalog on this engine. A
number with nothing beside it is a number, not a claim. This page is the same
16 queries on Neo4j 5.26.30 and Samyama 1.7.1, on one machine, over one graph,
with both engines indexed and both warmed.

**Two runs are published, not one.** Getting here took four attempts and two
withdrawn headlines — the first right by accident and for the wrong reason, the
second simply wrong — and a single reading taken after a methodology change is
what produced both. Both are marked in the table below. Where the two published
runs disagree, that disagreement is the result.

```bash
export NEO4J_PASSWORD=<any local password>   # read by both lines below
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/$NEO4J_PASSWORD \
  -e NEO4J_server_memory_heap_max__size=2G neo4j:5-community
python -m benchmarks.compare_neo4j --repeats 15 --warmup 10 --natural
```

Re-running against a server that already holds a fleet needs `--force-wipe`,
which deletes everything in the database first. `--reuse-neo4j` is the other
option and keeps what is there, but only if it is byte-for-byte this fleet: it
verifies the node and edge counts and all 22 indexes, and refuses otherwise.

## Where Neo4j wins, including the hero query

| query | what it asks | samyama | neo4j | |
|---|---|---:|---:|---|
| **`EA01`** | **the fallback audit — the hero question** | 34.8 / 35.8 ms | **8.1 / 7.7 ms** | **neo4j 4.3-4.7x** |
| `EA15` | operators on only one execution provider | 23.5 / 22.5 ms | **13.0 / 14.2 ms** | neo4j 1.6-1.8x |

Two figures per cell: the two warmed runs, in the same order everywhere on this
page. `EA01` is the query in the README, in the demo, in the pitch, and **Neo4j
answers it more than four times as fast.**

**This replaces an earlier version of this table that also listed `EA02` and
`EA11` as Neo4j wins.** They are not wins; they are queries neither engine
answers consistently. Both sort on a column that ties and then take a `LIMIT`,
so each *repeat within one engine* returns a different arbitrary N of equally
ranked rows — see "Answers that are not stable" below. They were scored because
the harness compared row content only between engines, never between repeats of
the same one.

## Everything, in order

Median of 15 after a 10-pass catalog warm-up plus 10 per-query runs, both
engines indexed. Ratio is neo4j / samyama; above 1.0 means Samyama is faster.

Every cell is `run 1 / run 2`, in that order everywhere on this page. An earlier
version averaged the two runs' milliseconds while printing both ratios, which
made them irreconcilable.

**The sub-millisecond rows will not reconcile by hand, and that is a rounding
artifact rather than an error.** Ratios come from unrounded medians and the
table shows one decimal: `EA14`'s displayed 8.3 / 0.4 suggests 20.8x, while the
published 18.99x puts its Samyama median nearer 0.44 ms. The tool prints three
decimals below 1 ms so a future run is checkable from its own output; this
table is transcribed at one decimal and left as measured rather than re-rounded.

| query | samyama | neo4j | ratio | verdict |
|---|---:|---:|---:|---|
| `EA14` | 0.4 / 0.4 ms | 8.3 / 7.8 ms | 18.99x / 17.37x | samyama |
| `EA13` | 2.8 / 2.2 ms | 47.0 / 45.8 ms | 16.89x / 21.24x | samyama |
| `EA12` | 2.1 / 1.9 ms | 21.5 / 17.6 ms | 10.41x / 9.16x | samyama |
| `EA06` | 1.2 / 1.8 ms | 9.6 / 12.2 ms | 8.05x / 6.84x | samyama |
| `EA03` | 1.5 / 1.5 ms | 10.9 / 9.1 ms | 7.34x / 6.27x | samyama |
| `EA10` | 2.1 / 2.4 ms | 12.2 / 11.6 ms | 5.92x / 4.82x | samyama |
| `EA16` | 35.9 / 34.3 ms | 79.1 / 79.1 ms | 2.21x / 2.31x | samyama |
| `EA05` | 62.5 / 72.8 ms | 112.9 / 117.1 ms | 1.81x / 1.61x | samyama |
| `EA04` | 18.0 / 30.2 ms | 25.6 / 28.6 ms | 1.42x / 0.94x | **unresolved — samyama, then parity** |
| `EA15` | 23.5 / 22.5 ms | 13.0 / 14.2 ms | 0.55x / 0.63x | **neo4j** |
| `EA01` | 34.8 / 35.8 ms | 8.1 / 7.7 ms | 0.23x / 0.21x | **neo4j** |
| `EA07` | 16.7 / 15.2 ms | 9.5 / 9.1 ms | — | no verdict: same count, different rows |
| `EA09` | 1.1 / 1.3 ms | 7.7 / 8.0 ms | — | no verdict: same count, different rows |
| `EA02`, `EA08`, `EA11` | — | — | — | no verdict: unstable across repeats |

**Samyama 8, Neo4j 2, unresolved 1, no verdict 5.**

`EA04` sits on the band edge and lands differently in the two runs — 1.42x then
0.94x — so it is reported as unresolved rather than picked. Anything inside
0.80x–1.25x is parity rather than a winner: naming one at that spread would be
inventing it.

**Five queries get no verdict at all**, which is new on this page and is the
main thing that changed. Three of them (`EA02`, `EA08`, `EA11`) do not answer
consistently between repeats of the *same* engine; two (`EA07`, `EA09`) answer
the same question differently on each. Timing either against the other compares
two different answers, so the harness refuses rather than scoring them.

## What the win actually is

Six of Samyama's eight wins are queries it answers in **under three milliseconds** —
`EA14` at 0.4 ms, `EA03` and `EA06` at 1.2-1.5 ms — against a Neo4j floor of
about 8 ms that barely moves with the query. That floor is a client, a socket
and a transaction, not a planner.

> **On cheap queries this measures in-process against over-the-wire, and
> in-process wins by a mile. On the expensive ones it measures the two engines,
> and there Neo4j takes the hero query.**

Above ~20 ms of real work: `EA05` and `EA16` to Samyama, `EA01` and `EA15` to
Neo4j. An HTTP-mode Samyama would pay a wire cost too, and that has not been
run. Nothing here says Samyama's *planner* is faster; the fast queries say its
**call overhead** is far lower, which is a claim about deployment shape — the
one `docs/why-this-engine.md` actually makes.

## How this page was wrong twice, from three defects

Worth reading before trusting any benchmark, including this one. Three
independent defects, each of which changed the headline, each found only after
the previous was fixed.

**1. Neo4j's edges loaded without indexes.** `load_neo4j` matched endpoints with
a label-free `MATCH (a) WHERE a.id = ...`; every index in
`schema/edge_ai_kg.cypher` is per-label, so none applied. From Neo4j's own plan:

```
MATCH (a)       WHERE a.id = "x"  ->  Filter, AllNodesScan
MATCH (a:Board) WHERE a.id = "x"  ->  NodeIndexSeek
```

Caught in review.

**2. Then Samyama's whole comparison ran without indexes.** Fixing the first
exposed the mirror image: `load_samyama` never called `apply_schema`, which
`etl/loader.py` and `benchmarks/ingest.py` both do. Not merely a slow load —
**every Samyama query timing was measured on an unindexed graph while Neo4j had
all 22.**

| | without `apply_schema` | with it |
|---|---:|---:|
| Samyama load | 157 s (486 edges/s) | **25.6 s** |
| `EA13` | 18.0 ms | **0.7 ms** |

Both columns are from the earlier machine, before the fourth correction below,
so `EA13`'s 0.7 ms here is not comparable with the 2.8 / 2.2 ms in the tables
above; what this table shows is the 25x step between its own two columns.

**3. Per-query warm-up does not warm a JVM.** With 10 warm-up runs *per query*,
`EA01` came back at 11.6 ms in the sweep and 6.2 ms when `--natural` re-ran it
at the end — 1.9x apart, same query, same process, differing only in how much
work the JVM had done by then. A whole-catalog warm-up pass before anything is
timed closes that to 1.2x.

**What each defect did to the headline:**

| conditions | `EA01` samyama / neo4j | published as |
|---|---:|---|
| Samyama unindexed, Neo4j warm from an hour of use | 15.9 / 6.0 ms | "Neo4j 2.6x faster" — **withdrawn** |
| both indexed, Neo4j freshly started | 13.2 / 11.6 ms | "a draw" — **withdrawn** |
| both indexed, catalog-warmed | 13.7 / 6.5 ms | 2.1x |
| both indexed, catalog-warmed, second run | 15.8 / 5.9 ms | 2.6x |
| **type-aware row comparison, instability check** | **34.8 / 8.1 ms** (run 1) | **4.3-4.7x — current** |

The "draw" was the outlier: Neo4j was under-warmed, not faster or slower than it
really is. The lesson is not any single bug — it is that **one run after a
methodology change is not evidence**, which is why two are published here.

**The last row is this page's fourth correction and the largest.** Nothing about
either engine changed; the comparison did. Queries that could not agree with
themselves between repeats stopped being scored, and float and integer values
stopped counting as disagreements. `EA01`'s loss got wider, two of Neo4j's
three published wins turned out to be unstable queries, and the machine these
ran on is not the one the earlier rows used — which is why the absolute
milliseconds moved as well as the ratio.

## Our query shapes do not handicap Neo4j

The obvious objection to running our Cypher on Neo4j is that the catalog is
written around Samyama's limits — projected through `WITH` before `RETURN`
(note 3), single `ORDER BY` key (note 3b), `OPTIONAL MATCH ... count() = 0`
instead of a negated pattern (note 5), conditional aggregation instead of a
self-join (note 1). Neo4j needs none of them.

**Measured, both spellings on Neo4j:**

| `EA01` on Neo4j | run 1 | run 2 |
|---|---:|---:|
| our shape, as it ships | 6.7 ms | 6.2 ms |
| idiomatic, `NOT EXISTS { }` | 9.9 ms | 11.4 ms |
| | our shape 1.48x faster | 1.84x faster |

**These are not the 8.1 / 7.7 ms the tables above report for the same query**,
and the difference is measurement position, not disagreement: `--natural` runs
after the whole sweep, so the JVM has done sixteen queries more work by then.
That is the same effect described as defect 3 above, and it is why the
two spellings are compared *against each other within this block* rather than
against the main table.

Our shape is faster on Neo4j in both runs, by a margin that is itself unstable
(1.48x to 1.84x) — so read the direction, not the number. The workarounds are not
a thumb on the scale against Neo4j on this query, and it still wins.

One query. It does not license "our shapes are faster everywhere". It does
dispose of the specific objection that these timings are rigged.

## Same answers, not always the same output

`EA07` and `EA09` return the same row count and different rows on the two
engines. Both sort on a column that ties — `kernel_count` 79, `power_mw` 15 —
and then take a `LIMIT`, so each returns a different arbitrary N of many
equal-ranked rows.

Not different answers. Also not identical output, and the harness now refuses a
speed verdict on them rather than timing two different answers against each
other. It used to record the mismatch *and* score the query.

`EA10` was on this list and is not any more. It differed in the sixteenth
significant digit of a float, which is the two engines summing in different
orders — the comparison is now type- and tolerance-aware, so that reads as
agreement, which it is. The same fix stopped `1` and `1.0` counting as a
disagreement, and stopped `null` counting as equal to the text `"None"`.

## Answers that are not stable

**`EA02`, `EA08` and `EA11` answer differently between repeats of the same
engine.** Not between engines — between runs of one query on one engine, inside
a single invocation of this command. Each sorts on a tied column and takes a
`LIMIT`, so which equally ranked rows come back is arbitrary and not stable.

There is no single result to compare, so they get no verdict. Two of them,
`EA02` and `EA11`, were previously published as Neo4j wins.

That is worth stating plainly: **the harness compared row content between the
two engines and never between repeats of the same one**, so a query that could
not agree with itself was scored as though it had a definite answer. The
instability check runs over the repeats now, which is what surfaced it.

The queries themselves are not wrong — an arbitrary N of equally ranked rows is
what `ORDER BY ... LIMIT` over ties means, on any engine. What was wrong was
timing them as if the answer were fixed.

## The engine version these figures need

Samyama here is **embedded 1.7.1**, which is what `pyproject.toml` floors the
engine at since #104. (`CLAUDE.md` describes the *server* image as 1.7.0 — a
different build, not a contradiction.)

It matters for three of these queries. `EA01`, `EA02` and `EA04` do not run
correctly on 0.6.1: `EA01` and `EA02` raise `Variable not found` where a second
`WITH` introduces an alias, and `EA04`'s `WHERE` is silently dropped. That is
version skew, **not** the embedded-versus-HTTP difference engine notes
[10 and 11](engine-notes.md) originally claimed — those notes compared a 0.6.1
embedded build against a 1.7.0 server and read the version gap as a build gap,
a conclusion the notes themselves now withdraw. On 1.7.1 the three run, which
is why they are in the table at all, so **below the floor this comparison would
not reproduce** for them.

## Conditions

| | |
|---|---|
| machine | 11th Gen Intel i5-1135G7 @ 2.40 GHz, 8 cores, 31 GB |
| neo4j | 5.26.30 Community, Docker, 2 GB heap, HTTP transactional endpoint |
| samyama | 1.7.1, embedded, in-process — the floor `pyproject.toml` declares since #104 |
| data | 25,150 nodes / 76,303 edges, seed 20260814, scale 1.0, both layers |
| indexes | the same 22 on both, translated from `schema/edge_ai_kg.cypher` |
| method | 10-pass catalog warm-up, then 10 per-query warm-ups, then median of 15 |
| runs | two, both published |
| queries | identical Cypher text; all 16 parse on Neo4j 5 unchanged |

## What this does not measure

- **One machine, one dataset, one version of each engine.**
- **No concurrency.** Single client, one query at a time. Neo4j is built for a
  workload this does not exercise.
- **No cold cache**, no memory ceiling, no persistence, no failover. See
  `docs/why-this-engine.md` for the operational axis, where Neo4j wins outright.
- **Not Memgraph**, which `docs/why-this-engine.md` names as the likelier winner
  on this axis and which remains unmeasured.
- **Load time is not compared.** Samyama's 25.6 s and Neo4j's 7.6 s were
  measured before the two were made symmetric — Samyama's then included its
  `apply_schema`, Neo4j's already excluded its index build; both exclude index
  creation now, and neither figure has been re-run since. "Both exclude index
  creation" is not the same as like-for-like: Samyama maintains its indexes
  during the node inserts, while Neo4j builds them afterwards and that build is
  subtracted, so Samyama's figure still carries index upkeep that Neo4j's does
  not. Load time is not scored for that reason too. They also come from
  different paths — batched `CREATE` over a Python client versus parameterised
  `UNWIND` over HTTP — and neither is that engine's fastest bulk path
  (`.sgsnap` import and `neo4j-admin import`). The figures are here for scale.
