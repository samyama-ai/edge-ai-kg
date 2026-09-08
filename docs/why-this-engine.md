# Why this engine, and where it loses

Part of the differentiation work tracked in #43.

Neo4j, Memgraph, KuzuDB, ArangoDB and TigerGraph all exist and are mature.
"Why not just use Neo4j" is the first question a technical reader asks, and a
feature table with ticks is not an answer — anyone can write one and nobody
believes one.

So this page is organised the other way round. **The limitations come first**,
because they are the part that can be checked, and a reader who finds them
honest has reason to believe the rest.

Every claim here is one of three kinds, and each is labelled:

- **Measured** — a command in this repo produces the number.
- **Quoted** — taken verbatim from the other project's own licence file, linked.
- **Unmeasured** — believed, not tested. Named as such, with the open issue.

---

## 1. What the others do better (#50)

### Cypher completeness — they win, and it is not close

`docs/engine-notes.md` documents **eleven** behaviours of this engine that
Neo4j does not have. Nine are v1.7.0 semantics; two are disagreements between
the embedded and HTTP builds of the same version.

Several **return wrong rows rather than erroring**, which is the worst failure
mode a database can have:

| | |
|---|---|
| `RETURN DISTINCT` | silently a no-op (note 2) |
| `ORDER BY` on a `RETURN` alias | silently ignored — with `LIMIT`, an arbitrary N rows dressed as a top-N (note 3) |
| Second `ORDER BY` key | silently ignored (note 3b) |
| `count(DISTINCT node)` over a multi-variable `MATCH` | returns N rows of `1` (note 9) |
| `<>` against a null property | matches (note 8b) |
| A trailing bound variable in a second `MATCH` | not joined — a cartesian product (note 1) |

**Measured.** The hero query is written *around* note 1, which is why `EA04`
uses conditional aggregation instead of the natural self-join. Three tests are
marked `xfail` because the two builds disagree (#56).

Neo4j has none of these. An engineer who knows Cypher can write Cypher against
Neo4j; against this engine they must read an eleven-item notes file first. That
is a real cost and it is paid on day one.

### Operations and maturity — they win

Backup, replication, failover, monitoring, rolling upgrade, and twenty years of
production scar tissue. **Unmeasured on our side because it does not exist to
measure.** This repo runs a single process against a single graph.

Note also that `--graph` looks like tenant isolation and **is ignored** on the
OSS HTTP path (note 7) — everything lands in `default`. Multi-tenancy is not a
thing you have here.

### Graph algorithms — they win, but by less than this page first claimed

**Correction.** An earlier draft of this page said centrality and community
detection were "not merely unused, unavailable". That is false, and it is the
kind of checkable claim #50 exists to prevent. The Python client exposes:

```
page_rank  wcc  scc  triangle_count  bfs  dijkstra  pca
create_vector_index  add_vector  vector_search
```

**Measured** on a 3,896-node / 11,306-edge load (`--scale 0.15`, seed 4242):

| call | result |
|---|---|
| `page_rank()` | 3,896 nodes scored |
| `page_rank(label="Operator", edge_type="IMPLEMENTS")` | 205 scored — it takes projections |
| `wcc()` | 3 components |
| `scc()` | 3,833 components |
| `triangle_count()` | 3,523 |

So the honest comparison is **breadth, not presence**. Neo4j GDS ships dozens of
algorithms with tuning, streaming and write-back modes; this engine ships about
seven with none of that scaffolding. That is still a clear win for GDS, and a
much smaller one than "they have algorithms and we have none".

Two practical notes, both measured:

- Results are keyed by the engine's **internal integer node id**, not by our
  `id` property. `MATCH (n:Operator) RETURN id(n), n.id` is the bridge, and
  nothing in this repo does it yet.
- The first argument is `label`, not the graph name. Passing `"default"`
  positionally silently scores **zero** nodes, because it is read as a label
  that does not exist. That cost me a wrong conclusion before I checked the
  signature, and it is exactly the "returns nothing rather than erroring" shape
  `engine-notes.md` catalogues.

**Measured by absence, still true:** of the 16 catalog queries, exactly one
(`EA07`) uses a variable-length path and none uses `shortestPath` or any of the
algorithms above. They are available and unused.

### Ecosystem and hiring — they win

Drivers in every language, Bloom, an enormous body of answered questions, and
people who already know both Cypher and Neo4j operationally. There is no
contest here and it would be silly to claim one.

### Scale ceiling — unknown, which is worse than losing

The largest graph anyone has loaded here is **25,150 nodes and 76,303 edges**
(`--scale 1.0`). There is no evidence at 10x or 100x that (#11). Neo4j runs
graphs of billions of nodes in production.

**The honest statement is not "we are smaller" — it is "we have not tested past
25K nodes and therefore do not know."** Anyone choosing this engine for a
larger graph is doing something nobody in this repo has done.

---

## 2. Licence and cost (#51)

Every line below is **quoted from the project's own licence file**, linked. This
matters more here than in most comparisons: for a repo about *edge deployment*,
where the database may ship on or near the device, licence terms are a
first-order question, not a footnote.

| Engine | Licence | Can you embed it in a product you ship? |
|---|---|---|
| **Samyama Graph** | [Apache-2.0](../LICENSE) | **Yes**, including redistribution, without copyleft |
| Neo4j Community | [GPLv3](https://github.com/neo4j/neo4j/blob/dev/LICENSE.txt) | Only if your product is also GPLv3 |
| Neo4j Enterprise | commercial | Negotiate |
| **Memgraph** | [BSL 1.1](https://github.com/memgraph/memgraph/blob/master/licenses/BSL.txt) | **No** — see below |
| **KuzuDB** | [MIT](https://github.com/kuzudb/kuzu/blob/master/LICENSE) | **Yes** |
| ArangoDB | [BSL 1.1](https://github.com/arangodb/arangodb/blob/devel/LICENSE) | Internal use only |
| DuckDB | [MIT](https://github.com/duckdb/duckdb/blob/main/LICENSE) | **Yes** |

**Neo4j Community is GPLv3**, quoted: *"The software … is licensed under the
GNU GENERAL PUBLIC LICENSE Version 3."* Shipping a GPLv3 database inside a
device is a legal conversation, not a technical one. Clustering and hot backup
are Enterprise-only.

**Memgraph's BSL forbids exactly our use case.** Its Additional Use Grant
permits internal production use but explicitly forbids you to *"embed or
otherwise distribute the Licensed Work to third parties"*. Change Date 2030,
converting to Apache-2.0. For an edge product that ships, Memgraph is not
available on its open licence at all — which is worth knowing before it is
benchmarked, not after.

**ArangoDB** is likewise BSL 1.1: internal production use, but not *"a
commercial offering that allows one or more third parties … to access, create
or manage databases."* Change Date is four years from its March 2024 release.

**TigerGraph** is commercially licensed. Its terms are not quoted here because
they are not in a public licence file, and this page does not paraphrase terms
it has not read.

### So the licence claim is real, but shared

Apache-2.0 is a genuine advantage over Neo4j, Memgraph and ArangoDB for anything
that ships. It is **not** an advantage over KuzuDB or DuckDB, both MIT and both
more permissive in practice than Apache-2.0's patent and notice requirements.

---

## 3. Which competitor is actually closest (#52)

"Neo4j and others" flattens a field that is not flat, and naming only the
biggest incumbent looks like the field has not been surveyed.

### KuzuDB — the direct challenge, and it should be said plainly

KuzuDB is an **embedded** graph database that calls itself "the SQLite for
graphs". It is MIT, in-process, columnar, and speaks Cypher.

**"We run embedded" is therefore not a differentiator.** Kuzu does that too,
under a more permissive licence, with a query engine built for analytics. Any
page that presents embedding alone as the reason to choose this engine is
weaker for pretending Kuzu does not exist.

What is left after conceding that: the *convergence* — graph traversal, vector
search and an MCP surface in one embedded binary.

**The vector half exists and works.** Measured: `create_vector_index`,
`add_vector` and `vector_search` all succeed on the embedded build, and a search
returns ranked `(node_id, distance)` pairs. So convergence is **available**, not
hypothetical.

**But this repo does not use it.** `grep` for vector/embedding/hnsw across the
source returns nothing; the catalog is 16 Cypher queries. So the claim today is
"the engine can, this KG does not" — #48 is the issue that closes that gap, and
until it lands the convergence differentiator is real in the engine and
undemonstrated here. #49 asks the same of the MCP surface.

### Memgraph — closest on the performance axis, unavailable on licence

In-memory, C++, Bolt and Cypher, positions explicitly on speed, and is the one
most likely to beat this engine on a head-to-head benchmark (#47, not yet run).

But per its own BSL, it cannot be embedded in a shipped product. So it is the
strongest technical competitor and simultaneously not a competitor for the
deployment shape this repo is about. Both halves are true and the page is worse
if it states only one.

### DuckDB — increasingly the real answer to a related question

MIT, embedded, and with property-graph extensions maturing. For "I want
analytics with some traversal, in-process", DuckDB is a serious answer and is
getting more so. It is not a graph database, and the hero question here —
multi-hop coverage gaps across kernels, operators and accelerators — is not
what it is for. **Unmeasured**; no comparison has been run.

### ArangoDB and TigerGraph — different shapes

ArangoDB is multi-model (document + graph), server-based, BSL. TigerGraph is
distributed, proprietary, aimed at very large analytical graphs. Neither is
embedded, so neither competes on the axis this repo cares about.

---

## 4. Embedded and in-process — the claim worth leading with (#44)

This is the one differentiator that is **architectural rather than a benchmark**,
so no tuning flag on the other side overturns it.

**Measured**, median of 5 cold processes — a fresh interpreter each time:

| step | median |
|---|---:|
| `from samyama import SamyamaClient` | 1.9 ms |
| `SamyamaClient.embedded()` | 0.0 ms |
| first query answered | 0.6 ms |
| **cold interpreter to first answer** | **2.5 ms** |

No server, no JVM, no Docker, no port, nothing to start. Neo4j cannot do this at
any speed: it is a server process, and nothing in that family runs *inside* the
Python process that is also doing the inference. For edge AI that is the whole
difference between a graph that can sit next to the model and one that cannot.

### What embedded mode gives up, in the same breath

**It is in-memory and it does not persist.** A second Python process sees an
empty graph. So `python -m etl.loader` with no `--url` loads a graph and then
throws it away — it is a timing exercise, not a way to prepare data (#4).

That reframes the 2.5 ms honestly: it is the time to a *usable engine*, not to
a *loaded graph*. Querying **this** graph in a fresh process means loading it
first, which is ~25 s at `--scale 1.0` — the figure `python -m etl.loader`
prints on its `[4/4] loading edges` line. The options
are to build the graph inside your process — as every demo and test here does —
or to run the HTTP server and pay a network hop instead.

So the fair statement is: **zero-install and instant to start, and you pay for
the data every process.** A snapshot import is the escape hatch from that and is
still unmeasured (#45).

## 5. What this engine actually has, measured

Kept short deliberately: the claims that can be reproduced today.

| Claim | Status | How |
|---|---|---|
| Runs in-process, no server, no install beyond `pip` | **Measured** | 2.5 ms cold to first query; every test and demo does it |
| Embedded mode does not persist — every process reloads | **Measured** | #4; ~25 s for this graph |
| Apache-2.0, embeddable and redistributable | **Quoted** | [`LICENSE`](../LICENSE) |
| ~52K nodes/s, ~3.1K edges/s ingest | **Measured, no command on this branch** | figures from #10, which adds the `benchmarks.ingest` module that reproduces them |
| `id` indexes are load-critical — 10.6x | **Measured** | #18 |
| 16-query catalog, ground-truthed in Python | **Measured** | `pytest tests/test_correctness.py` |
| Snapshot import "well under a second" | **Unmeasured** | #45 |
| Footprint on a shared machine | **Unmeasured** | #46 |
| Faster than Neo4j on the hero query | **Unmeasured** | #47 — and Memgraph is the likelier winner |
| Graph + vector in one binary | **Available, measured; unused in this repo** | #48 |
| PageRank / WCC / SCC / triangle count | **Measured** — available, unused by the catalog | — |

---

## The short version

If you are shipping a graph inside a product, Neo4j Community's GPLv3 and
Memgraph's BSL are disqualifying and Apache-2.0 is not — but **KuzuDB is MIT and
also embedded**, so licence alone does not pick this engine.

If you need operations, algorithms, ecosystem, or a graph larger than 25K nodes,
the honest answer today is that Neo4j is the better choice and this repo has no
evidence to argue otherwise.

The defensible position is narrower than "why not Neo4j" suggests: an
Apache-2.0, in-process graph with a ground-truthed query catalog for a
domain-specific question, whose main convergence claim is **not yet
demonstrated in this repo**.

A page that said more than that would be one an engineer could catch out, which
per #50 is exactly the bar this is trying to clear.
