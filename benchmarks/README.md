# Benchmarks

17 queries covering the questions an edge-AI team asks while landing a model on
custom silicon. Each entry in [`queries.py`](queries.py) carries the plain
question, the Cypher, and a `why_graph` note explaining why the question is
awkward in a relational or document store.

```bash
python -m benchmarks.run_benchmark --url http://127.0.0.1:8080   # all 17
python -m benchmarks.run_benchmark --only EA01 --rows 20         # one
python -m benchmarks.run_benchmark --json-out results.json       # machine-readable
```

Omit `--url` to run against an in-process embedded engine.

## Current results

> **Stale — measured on the 12-query catalog, before EA13-EA16 and EA20 were
> added.** Re-running it needs the `samyama` server binary, which lives in the
> engine repo, not here; the embedded engine is in-process and cannot be loaded
> by one command and queried by the next. The figures below are kept as the last
> recorded run rather than silently restated for 17 queries.
> `README.md` quotes a later 16-query recorded run: all 16 returned rows, median
> 14 ms, slowest 73 ms, on a recorded 25,150 nodes / 76,303 edges.

**Today's build is 24,127 nodes / 75,265 edges generated, and 25,162 nodes /
77,743 edges with the real layer** — bigger than either run below, because
`Site` and its 1,440 `DEPLOYED_AT` edges landed with #34 after both were taken.

Samyama Graph v1.7.0 OSS, a recorded 24,115 nodes / 73,825 edges, server on
localhost:

**12/12 queries return rows, 0 empty, 0 failed. Median 15 ms, slowest 71 ms.**

The slow end are the queries that touch all 21,844 kernels (EA08, EA11); the
anti-joins that anchor on a single model run in 30-40 ms.

## A warning about "it returned rows"

Three of the engine behaviours in [`../docs/engine-notes.md`](../docs/engine-notes.md)
return **wrong rows rather than errors**. EA04 originally returned 12
confident-looking rows that paired one model's fp32 variant with another
model's int8 variant. Treat a green run as necessary, not sufficient —
[`../tests/test_correctness.py`](../tests/test_correctness.py) is what actually
checks the answers.
