# The client's non-Cypher surface

Closes #81.

`docs/engine-notes.md` documents the *query* path and its bugs. The Python
client also exposes graph algorithms and a vector index, and **nothing in this
repo mentioned them** — which is how a draft of `docs/why-this-engine.md` came
to assert that centrality and embeddings were "not merely unused, unavailable".
They are available. That claim was derived from grepping this repo rather than
the client, and it was false.

Everything below was run against a 3,896-node / 11,306-edge load
(`--scale 0.15`, seed 4242) on `samyama` 0.6.1, embedded.

## Two traps, before the list

**1. The first positional argument is `label`, not the graph name.**

```python
client.page_rank("default")   # -> {}   <- reads "default" as a LABEL
client.page_rank()            # -> 3,896 scored nodes
```

It does not raise. `"default"` is taken as a label that does not exist, so the
result is an empty dict. This is the same *returns nothing rather than erroring*
shape `engine-notes.md` catalogues on the query path, and it produced a wrong
conclusion here before the signature was checked.

**None of the algorithm methods take a `graph` argument at all** — only `query`,
`query_readonly` and `delete_graph` do. There is nothing to pass.

**2. Results key on the engine's internal integer node id, not on `id`.**

```python
client.page_rank(label="Operator", edge_type="IMPLEMENTS")
# {55: 0.0129, 226: 0.0107, ...}      not  {"op:ai.onnx:abs": ...}
```

The bridge is `id()`, which parses on this engine:

```cypher
MATCH (n:Operator) RETURN id(n), n.id
-- [[53, 'op:ai.onnx:abs'], [54, 'op:ai.onnx:acos'], ...]
```

`elementId()` does **not** parse (`Unknown function: elementId`). Nothing in
this repo builds that mapping today, so any use of these results needs it first.

## What exists

| call | signature | measured on the fixture above |
|---|---|---|
| `page_rank` | `(label=None, edge_type=None, damping=0.85, iterations=20, tolerance=1e-06)` | 3,896 nodes scored; `label="Operator", edge_type="IMPLEMENTS"` → 205 |
| `wcc` | `(label=None, edge_type=None)` | `{'components': {...}, 'component_count': 3}` |
| `scc` | `(label=None, edge_type=None)` | `component_count: 3833` |
| `triangle_count` | `(label=None, edge_type=None)` | `3523` |
| `bfs` | `(source, target, label=None, edge_type=None)` | `{'path': [258, 15], 'cost': 1.0}` |
| `dijkstra` | `(source, target, label=None, edge_type=None, weight_property=None)` | same shape, plus `weight_property` |
| `pca` | `(properties, label=None, n_components=2)` | `{'components': [[1.0]], 'explained_variance': [...]}` |
| `create_vector_index` | `(label, property, dimensions, metric='cosine')` | succeeds |
| `add_vector` | `(label, property, node_id, vector)` | `node_id` is the **internal integer** |
| `vector_search` | `(label, property, query_vector, k=10)` | `[(1, 0.0), (0, 0.0)]` — `(node_id, distance)` |

`page_rank` and the component algorithms take a **projection**: `label` and
`edge_type` restrict which subgraph is walked, which is the useful part —
`page_rank(label="Operator", edge_type="IMPLEMENTS")` ranks operators by kernel
coverage rather than ranking the whole mixed graph.

### `bfs` / `dijkstra` return `None` for "no path"

Not an error, and easy to misread as a failure:

```python
client.bfs(kernel_a, kernel_b)   # -> None   two Kernels have no path
client.bfs(kernel, accelerator)  # -> {'path': [258, 15], 'cost': 1.0}
client.bfs(node, node)           # -> {'path': [258], 'cost': 0.0}
```

## What this does not claim

**No comparison against the Cypher equivalents.** A naive
`MATCH (k:Kernel), (a:Accelerator) WHERE id(k) = ... MATCH p = shortestPath(...)`
timed out (`Query timed out after 1 rows`) while `bfs()` answered immediately —
but that Cypher is the comma-separated multi-variable shape engine note 1 warns
about, so it is a bad query rather than evidence the API is faster. A fair
comparison has not been run.

**Whether the catalog should use any of this is out of scope.** #48 covers the
vector half. This page exists because the surface was undocumented, not because
it is unused.

**Not fault-tested.** These are one measured call each, on one build, on a small
fixture. They are enough to say what exists and what shape it returns; they are
not enough to say the algorithms are correct.
