# The vector half: what it is for here, and why it does not work yet

Closes #48.

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

## The defect

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

v = [random.random() for _ in range(8)]
n = math.sqrt(sum(x * x for x in v))
v = [x / n for x in v]                      # a normalised random vector

client = SamyamaClient.embedded()
# ... create two nodes labelled T, take their id() values ...
client.create_vector_index("T", "emb", dimensions=8, metric="cosine")
client.add_vector("T", "emb", id_a, v)
client.add_vector("T", "emb", id_b, v)      # PANIC
```

### What triggers it, measured

| input | result |
|---|---|
| 336 **distinct** vectors | OK |
| 50 copies of one *sparse* vector | OK |
| `[1,0,0,0,0,0,0,0]` added twice | OK |
| a normalised **random** vector added twice | **PANIC** |
| `[0.5] * 8` added twice | **PANIC** |
| 40 distinct + 2 duplicates | OK |
| 40 distinct + 4 duplicates | **PANIC** |
| 40 distinct, each duplicated once | **PANIC** |

The assertion is on the **sign of a distance** (`c.dist_to_ref <= 0.`, and
`f.dist_to_ref >= 0.` in a neighbouring case). That is consistent with a
self-distance computing to ±1e-8 rather than exactly `0.0`: the cases that
survive are the ones whose float arithmetic lands on an exact zero
(`[1,0,0,…]`), and the cases that panic are the ones that do not. It is
therefore **value-dependent, not count-dependent** — which is why "40 + 2
duplicates" passes and "40 + 4" does not.

Volume alone is fine. 336 distinct vectors index and search without complaint.

### It is not the metric

Every metric panics on the same two-call reproduction:

| `metric=` | duplicate add + search |
|---|---|
| `cosine` | **PANIC** |
| `euclidean` | **PANIC** |
| `l2` | **PANIC** |
| `dot` | **PANIC** |
| `inner_product` | **PANIC** |
| `manhattan` | **PANIC** |

Note the last three: `create_vector_index` accepted `dot`, `inner_product` and
`manhattan` without complaint, and there is no indication any of them is a
supported metric. **The `metric` argument is not validated** — a typo would be
accepted silently and you would not learn which distance you actually got.
That is worth knowing independently of the panic.

## Why that blocks this use case specifically

An embedding of a **name catalogue** produces near-duplicates by construction.
Measured on this graph's 376 operators with a 64-dimension character-trigram
embedding:

```
376 operators -> 344 distinct embeddings; 26 collide
  collision: ['AveragePool', 'AveragePool', 'AveragePool']
  collision: ['BatchNormalization', 'BatchNormalization']
  collision: ['Conv', 'Conv', 'Conv']
```

Those are the same operator name at several opset versions and domains — real
rows in this graph, not an artefact of the embedding.

**Deduplicating exact vectors is not enough.** After collapsing the 376 to 307
distinct embeddings, every `add_vector` succeeded and **`vector_search` then
panicked** — so vectors that are merely *close* trip the same assertion. Any
embedding worth using puts `Conv` and `ConvTranspose` close together; that is
what an embedding is for.

## So the honest position

- **Convergence is available, not hypothetical.** The engine really does carry
  graph and vector in one binary, and the API is coherent.
- **It cannot serve this repo's obvious use case on 0.6.1.** The
  nearest-unknown-operator query is not blocked by design or by effort; it is
  blocked by a panic in `hnsw_rs 0.2.1`.
- **Nothing in this repo should claim the vector half is in use.** It is not.
  `docs/why-this-engine.md` says "the engine can, this KG does not"; this page
  is why the second half has not changed.

## What would unblock it

The panic is in a third-party crate (`hnsw_rs 0.2.1`) rather than in Samyama's
own code, so the likely fixes are upstream or a version bump. Worth reporting to
`samyama-ai/samyama-graph` with the two-call reproduction above — it needs no
graph, no data and no scale.

**Not investigated:** whether the panic leaves the index corrupt or is
recoverable within the same process. Cheap follow-up for whoever picks this up.
