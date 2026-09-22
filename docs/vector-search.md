# The vector half: what it is for here, and why it did not work on 0.6.1

> **Superseded in part by #56, 2026-09-08. The panic below no longer
> reproduces.** Everything on this page was measured on `samyama` 0.6.1. #56
> raised the floor to **1.7.1**, and on that build the two-call reproduction
> succeeds, and the hold-out experiment — the one this page's argument rests on
> — runs to completion:
>
> ```
> $ python -m benchmarks.vector_probe --repro
> index accepted: True
> same normalised vector added twice -> ok:
> that is the 0.6.1 panic gone -- it raised PanicException (assertion failed: c.dist_to_ref <= 0.) until 1.7.1; see #56
>
> $ python -m benchmarks.vector_probe --holdout
> held out 40; 336 remain -> 307 distinct embeddings
> add_vector: 307 accepted, none failed
> vector_search: 40 of 40 unseen operators queried, none failed
> ```
>
> Against 0.6.1 that same run panicked on the 9th of 40. **So the blocker is
> gone and #48's nearest-unknown-operator query is buildable.** What this page
> says about *the engine* is now history; what it says about the **embedding**
> is not — the 26 collision groups are a property of a name catalogue, not of
> any engine version, and still shape whatever gets built.
>
> The two tables below now carry a **1.7.1 column**, re-measured on
> 2026-09-21: every trigger and every metric completes. What does *not* change
> is that `create_vector_index` accepts `dot`, `inner_product` and `manhattan`
> without complaint — the metric is still not validated. Building the query
> itself is #48's remaining work, deliberately not folded into #56: this banner
> corrects a claim #56's version bump falsified, and stops there.

The convergence pitch is graph traversal and vector search in one binary. This
KG uses the graph half and nothing else, and #48 asks whether the obvious
application works: **a model arrives carrying an operator this graph has never
seen.** Exact matching says "unknown". A vector search over operator names could
say "nearest known operator — and here is whether *it* has a kernel", which is a
converged answer a graph-only engine cannot give without a second system.

#48 accepts either outcome, as long as it is written down. The outcome is:

> **The API exists and the query cannot be built on `samyama` 0.6.1, because the
> HNSW index panics on vectors that are identical or nearly so — which is what
> an embedding of a name catalogue produces.**

## The API is real

`create_vector_index`, `add_vector` and `vector_search` all work on small,
well-separated inputs (see [`client-api.md`](client-api.md)):

```python
client.create_vector_index("Operator", "emb", dimensions=64, metric="cosine")
client.add_vector("Operator", "emb", node_id, vector)   # node_id is the INTERNAL int
client.vector_search("Operator", "emb", query_vector, k=3)
# -> [(node_id, distance), ...]
```

`id()` bridges the internal integer ids to this repo's string `id`s.

## The defect, as measured on `samyama` 0.6.1

> Everything from here to the end of this section is a record of **0.6.1**.
> It is kept because the reasoning is what makes the banner above checkable,
> not because it still happens. On the pinned floor (1.7.1) none of these
> panics reproduces — the tables carry a 1.7.1 column saying so, re-measured
> 2026-09-21, and the commands below re-run them.


Adding the **same vector twice** panics the Rust extension:

```
thread '<unnamed>' panicked at hnsw_rs-0.2.1/src/hnsw.rs:938:13:
assertion failed: c.dist_to_ref <= 0.
pyo3_runtime.PanicException
```

It surfaces as `pyo3_runtime.PanicException`, which is **not** an `Exception`
subclass — so an `except Exception` around the call does not catch it.

### Minimal reproduction — two calls

```python
import math, random
from samyama import SamyamaClient

client = SamyamaClient.embedded()
client.query('CREATE (:T {id: "a"}), (:T {id: "b"})', "default")
ids = [r[0] for r in client.query("MATCH (t:T) RETURN id(t)", "default").records]

v = [random.random() for _ in range(8)]
n = math.sqrt(sum(x * x for x in v))
v = [x / n for x in v]                       # a normalised random vector

client.create_vector_index("T", "emb", dimensions=8, metric="cosine")
client.add_vector("T", "emb", ids[0], v)
client.add_vector("T", "emb", ids[1], v)     # PANIC
```

Ran as pasted **on 0.6.1**. `python -m benchmarks.vector_probe --repro` is the
same thing as a command; on the pinned floor it completes instead, and prints
so. `--collisions` reproduces the embedding table below on either build --
the collisions are a property of the embedding, not of the engine.

### What triggers it, measured

The 0.6.1 column is the original measurement. The 1.7.1 column was
re-measured on 2026-09-21: each row in a fresh process, every vector added
*and* searched (`k=3`) under `metric="cosine"`, three random seeds per row. No
row raised on any seed. The script was a throwaway rather than a
`vector_probe` flag -- the two rows that matter most, a random vector added
twice and the hold-out, are `--repro` and `--holdout`, which anyone can re-run.

| input | 0.6.1 | 1.7.1 |
|---|---|---|
| 336 **distinct** vectors | OK | OK |
| 50 copies of one *sparse* vector | OK | OK |
| `[1,0,0,0,0,0,0,0]` added twice | OK | OK |
| a normalised **random** vector added twice | **PANIC** | OK |
| `[0.5] * 8` added twice | **PANIC** | OK |
| 40 distinct + 2 duplicates | OK | OK |
| 40 distinct + 4 duplicates | **PANIC** | OK |
| 40 distinct, each duplicated once | **PANIC** | OK |

The assertion is on the **sign of a distance** (`c.dist_to_ref <= 0.`, and
`f.dist_to_ref >= 0.` in a neighbouring case). That is consistent with a
self-distance computing to ±1e-8 rather than exactly `0.0`: the cases that
survive are the ones whose float arithmetic lands on an exact zero
(`[1,0,0,…]`), and the cases that panic are the ones that do not. It is
therefore **value-dependent, not count-dependent** — which is why "40 + 2
duplicates" passes and "40 + 4" does not.

Volume alone is fine. 336 distinct vectors index and search without complaint.

## The metric argument is not validated (measured on 0.6.1)

This is a separate finding from the panic, and useful on its own.

On 0.6.1 every metric panicked on the same two-call reproduction, so the
panic was not metric-specific. Each row below runs in a **fresh process** -- recoverability
after a panic is uninvestigated, so sharing one would make rows 2-6
order-dependent -- and `index accepted` is recorded separately from the panic,
so "accepted without complaint" is a measurement rather than an inference:

`python -m benchmarks.vector_probe --metrics` produces both columns. The
0.6.1 column is the original run; the 1.7.1 column is the run of 2026-09-21,
whose output was:

```
metric            index accepted  add + search
cosine            True            ok
euclidean         True            ok
l2                True            ok
dot               True            ok
inner_product     True            ok
manhattan         True            ok
```

| `metric=` | index accepted (both) | duplicate add + search, 0.6.1 | 1.7.1 |
|---|---|---|---|
| `cosine` | yes | **PanicException** | ok |
| `euclidean` | yes | **PanicException** | ok |
| `l2` | yes | **PanicException** | ok |
| `dot` | yes | **PanicException** | ok |
| `inner_product` | yes | **PanicException** | ok |
| `manhattan` | yes | **PanicException** | ok |

Note the last three: `create_vector_index` accepted `dot`, `inner_product` and
`manhattan` without complaint, and there is no indication any of them is a
supported metric. **The `metric` argument is not validated** — a typo would be
accepted silently and you would not learn which distance you actually got.
That is worth knowing independently of the panic, and it is the one finding
on this page that 1.7.1 did **not** change: all six are still accepted.

## Why that blocked this use case (0.6.1)

An embedding of a **name catalogue** produces near-duplicates by construction.
Measured on this graph's 376 operators with a 64-dimension character 3- and
4-gram embedding:

```
376 operators -> 344 distinct embeddings
                 26 collision groups, covering 58 operators
  collision: ['AveragePool', 'AveragePool', 'AveragePool']
  collision: ['BatchNormalization', 'BatchNormalization']
  collision: ['Conv', 'Conv', 'Conv']
```

Every one of the 376 rows has a **distinct `id`** -- `--collisions` prints that
alongside the counts -- so a shared *name* is a real collision across domains,
not a row counted twice. `AveragePool` appears as `op:ai.onnx:averagepool`,
`op:com.microsoft.nchwc:averagepool` and `op:com.ms.internal.nhwc:averagepool`.

`376 - 344 = 32` is the number of *surplus* rows; the 58 is how many operators
sit in a group with at least one other. Those are the same operator name at
several opset versions and domains — real rows in this graph, not an artefact of
the embedding.

**Deduplicating exact vectors was not enough** — on 0.6.1. The experiment
holds out 40 operators as "unseen", leaving 336 to index; those 336 collapse to
**307** distinct embeddings. All 307 `add_vector` calls succeeded, and
`vector_search` then panicked on the 9th unseen operator queried, so vectors
that are merely *close* tripped the same assertion.

The transcript below is that 0.6.1 run. `python -m benchmarks.vector_probe
--holdout` is the same command; on 1.7.1 it reports `40 of 40 ... none failed`,
which is the banner at the top of this page:

```
held out 40; 336 remain -> 307 distinct embeddings
add_vector: 307 accepted, none failed
vector_search: 8 of 40 unseen operators queried, then
               PanicException: assertion failed: c.dist_to_ref <= 0.
```

(344 and 307 are counts of different sets: 344 distinct embeddings across all
376 operators, 307 across the 336 that remain after the hold-out. Both are
reproducible with `python -m benchmarks.vector_probe --collisions`.)

Any embedding worth using puts `Conv` and `ConvTranspose` close together; that
is what an embedding is for.

## So the honest position

- **Convergence is available, not hypothetical.** The engine really does carry
  graph and vector in one binary, and the API is coherent.
- **It cannot serve this repo's obvious use case on 0.6.1.** The
  nearest-unknown-operator query is not blocked by design or by effort; it is
  blocked by a panic in `hnsw_rs 0.2.1`.
- **Nothing in this repo should claim the vector half is in use.** It is not.
  `docs/why-this-engine.md` says "the engine can, this KG does not"; this page
  is why the second half has not changed.

## What would have unblocked it, and what did

The panic was in a third-party crate (`hnsw_rs 0.2.1`) rather than in Samyama's
own code, so the likely fixes were upstream or a version bump. **It was the
version bump**: #56 raised the floor to 1.7.1 and the reproduction stops
reproducing, so nothing was ever filed against
`samyama-ai/samyama-graph`.

**Not investigated:** whether the panic leaves the index corrupt or is
recoverable within the same process. Cheap follow-up for whoever picks this up.
