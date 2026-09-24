# Edge AI Deployment Knowledge Graph

**25,150 nodes. 76,303 edges. Boards, kernels and neural networks in one graph — so you can ask what actually runs on your silicon.**

Real ONNX + ONNX Runtime + MLPerf Tiny data, plus a generated fleet for scale. Every node is stamped `real` or `synthetic`.

> Part of the **Samyama** ecosystem — loaded into and queried via the graph engine at [samyama-ai/samyama-graph](https://github.com/samyama-ai/samyama-graph).
> This repo holds the loader, the generator and the query catalog for the KG.

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue" alt="License"></a>

![Edge AI KG — the catalog questions answered](demo/edgeai-questions.gif)

*16 of the 20 [catalog queries](benchmarks/queries.py) run end to end — each question, the Cypher it becomes, and the answer. The missing ones are `EA17` (#35), `EA18` (#37), `EA19` (#40) and `EA21` (#36), added after this was recorded rather than left out of it. `EA13`-`EA16` run on real ONNX Runtime and MLPerf Tiny data. Long-form: the whole run in one image, nothing scrolled off.*

*Recorded 2026-08-14 at `--scale 1.0`, seed `20260814`. **Some figures in it have since moved** — the node count was corrected in #17 and ONNX Runtime has published since — so read it for the shape of the answers, not the numbers. Re-record with [`scripts/record_gif.sh`](scripts/record_gif.sh); `tests/test_demo_recording.py` compares it to the current build.*

---

## The question this exists to answer

Deploying a model onto custom edge hardware fails in a specific, boring way:
**one operator has no kernel on your NPU, silently falls back to the CPU, and
your latency budget is gone.** Finding out which operator, on which board,
under which runtime, is a graph traversal — a model's operator surface joined
against a kernel library joined against a hardware fleet.

```cypher
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.name = "depthwise-cnn-neuro-043"
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.kind = "NPU-Lite"
WITH op.name AS operator, op.category AS category,
     op.since_version AS opset, count(k) AS kernels
WHERE kernels = 0
RETURN operator, category, opset
ORDER BY category
```

```
operator            category       opset
PRelu               activation     16
HardSwish           activation     22
ThresholdedRelu     activation     22
Col2Im              convolution    18
Cos                 elementwise    22
LayerNormalization  normalization  17
RMSNormalization    normalization  23
ReduceL1            reduction      18
...                                          17 rows in 24 ms
```

Seventeen operators on the CPU instead of the NPU — including `PRelu` and
`LayerNormalization`, which you would have assumed were accelerated.
**One query, no ETL.**

(The projection goes through `WITH` and sorts on a single key deliberately —
see [engine notes](docs/engine-notes.md).)
Flatten this into JSON and it becomes a script you maintain forever.

---

## What is in it

Two spines that meet in the middle:

```
Vendor <- SoC <- Board                        Sensor -> SignalStage -> ... -> Model
           |                                                                   |
           +-> Accelerator <- Kernel -> Operator <---- USES_OPERATOR -----------+
                    ^            |                                             |
                 TARGETS    PROVIDED_BY                                  ModelVariant
                    +--------- Runtime                                         |
                                                                          Deployment -> Board
```

*The diagram is an orientation sketch, not the schema.* It shows 12 of the 16
node labels and names 3 of the 22 edge types; it **omits** `ClinicalTask`,
`Certification`, `Dataset` and `BenchmarkTask`, so the clinical spine appears to
stop at `Model` when it actually continues to a task and its regulatory
posture. [`docs/schema.md`](docs/schema.md) is the full picture, and
`tests/test_readme_diagram.py` fails if the two drift apart.

**Hardware**: 15 vendors, 52 SoCs, 91 accelerators -- 85 generated across five
archetypes (MCU-CPU / DSP / NPU-Lite / NPU-Pro / GPU-Embedded) plus 6 real ones
carrying four further kinds (NPU / CPU / GPU-CUDA / GPU-DirectML), 134 boards,
13 runtimes.
**Software**: 375 operators, 22,578 kernels, 64 models, 240 quantized variants,
1,513 deployments (73 of them real MLPerf Tiny measurements).
**Clinical**: 14 biosignal sensors, 16 DSP stages, 18 clinical tasks, 4 MLPerf
benchmark tasks, 12 datasets, 6 certifications.

Full detail in [`docs/schema.md`](docs/schema.md).

## Data: what's real, what's synthetic

| Source | License | What it contributes |
|---|---|---|
| [onnx/onnx](https://github.com/onnx/onnx) | Apache-2.0 | **205 real operators** — names, domains, opset versions |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | **734 real kernel registrations** across CPU / CUDA / DirectML execution providers |
| [mlcommons/tiny_results_v1.2](https://github.com/mlcommons/tiny_results_v1.2) | Apache-2.0 | **73 measured submissions** — real boards from Qualcomm, Renesas, ST, Syntiant, Bosch, with real throughput, accuracy and energy |
| generated | — | **The fleet**: 120 boards, 85 accelerators, 21,844 kernels, 1,440 deployments. Vendor and board names deliberately fictional (`Corvid Silicon`, `Tessera Labs`, …) |

**1,035 nodes are real; 24,115 are generated.** The split is queryable, not just
documented — every node carries `provenance` and `source`:

```bash
python -m etl.loader --layers real        # public-source subgraph only
```

#### What the real layer alone can answer

Measured, not assumed (#21). `--layers real` is **connected** — it is not a set
of islands sharing a database:

| | |
|---|---:|
| nodes | 1,240 |
| edges | 2,478 |
| labels with nodes | 10 of 16 |
| edge types present | 11 of 22 |
| orphaned nodes | **18**, all `Operator`s no ONNX Runtime kernel registers |

What it lacks is a *half*, not the joins. The real layer is the hardware and
kernel spine plus the MLPerf submissions; the clinical spine is entirely
generated, so `ModelVariant`, `Sensor`, `SignalStage`, `ClinicalTask`, `Dataset`
and `Certification` are empty.

**Against the HTTP server, 6 of the 20 catalog queries return rows**, 14 come
back empty, none error:

| | Queries |
|---|---|
| Return rows | `EA05`, `EA08`, `EA13`, `EA14`, `EA15`, `EA16` |
| Empty | `EA01`, `EA02`, `EA03`, `EA04`, `EA06`, `EA07`, `EA09`, `EA10`, `EA11`, `EA12`, `EA17`, `EA18`, `EA19`, `EA21` |

`EA18`, `EA19` and `EA21` are the three entries here **not** from the server
run, which predates all of them. Their place in the table is **measured
embedded**, against the
real layer, by `tests/test_real_layer_shape.py`, which executes every catalog
query and compares the result to this table. **Over HTTP it is expected, not
measured**: the real layer has no `ClinicalTask` and no `Sensor` (it does have
`Deployment`s, the MLPerf rows), and `EA18` opens on a `ClinicalTask` and
`EA19` on a `Sensor`, so neither's opening `MATCH` binds anything on either
build. `EA21` opens on a `Sensor` too, and carries a second reason it cannot
be read off the server run: it walks `NEXT_STAGE*0..`, the variable-length
shape the 1.7.0 server rejects outright for `EA17` (engine note 12), and
nothing here has run it there. The distinction between measured and expected
is kept rather than smoothed over, because on this page it has mattered
before.

Re-measured 2026-09-09, server 1.7.0 against embedded 1.7.1 over the same real
layer: the partition above is identical on both, for the seventeen queries that
existed then — `EA18`, `EA19` and `EA21` are covered by the embedded check
only. `EA10` and `EA12` are empty on *both*: they lost their rows to an
upstream ONNX Runtime refresh, not to a build difference. Reading that as a
divergence is what comparing a fresh embedded run against a recorded server
figure produces, and it is the trap this paragraph exists to mark.

**"Identical" is a claim about this table, not about the two builds.** On the
**full** graph, measured 2026-09-10 by loading one scale-1.0 fleet into both and
comparing row *content*: row counts match for 16 of the 17 in the catalog on
that date (`EA17` raises on the server, see below; `EA18` and `EA19` were added
afterwards and are not in this run), and **seven queries return the same number
of different rows** — `EA01`, `EA02`, `EA08`, `EA09`, `EA10`, `EA11`, `EA13`.
Every one is a tie under `ORDER BY … LIMIT`, where an arbitrary N of many
equal-ranked rows comes back, except `EA10`, which differs in the sixteenth
significant digit of a float. Those are not different answers, but they are not
"no disagreements" either, and comparing lengths would have hidden all seven.

**This does not contradict the engine-notes section below.** That section
describes `samyama` 0.6.1, where `EA01`, `EA02` and `EA04` were wrong against
the engine `pytest` used and carried `xfail`s. The comparison above was run at
1.7.1, where notes 10 and 11 do not reproduce and those marks are gone (#104) —
which is #56's finding, that
the "embedded versus server" disagreement was a version skew (0.6.1 against a
1.7.0 server) rather than a difference between the two builds.

`pyproject.toml` declares `samyama>=1.7.1` since #104, so the build these
pages describe is the build you get.

`EA17` is empty here because the real layer has no `Sensor` — the clinical spine
is entirely generated.

**That is why "none error" above is true and still consistent with engine note
12**, which says `EA17` raises on the 1.7.0 server. It raises only when there is
something to traverse: with no `Sensor` nodes the opening `MATCH` binds nothing,
`size(r)` is never evaluated, and the query returns empty. Measured on the
server with `--layers real`: 0 rows, no error. On the **full** graph it raises,
and is the one catalog query that cannot be asked over HTTP.

`EA07` walks the same `NEXT_STAGE` chain with a fixed bound, `*0..3`, and the
bound does not exempt it: note 12 measured the server matching only the
zero-length case for **every** form, bounded or not. It does not raise, because
`EA07` never calls `size(r)` — that type error is what makes `EA17` fail loudly.
Measured on a scale-1.0 graph loaded into both builds, `EA07` returns
byte-identical rows, because its `ORDER BY latency_ms ASC LIMIT 10` is
satisfied by paths at zero hops. That is luck rather than robustness; note 12
says so. Separately, the bound costs completeness on the embedded build — the
longest chain at scale 1.0 is 13 hops — which is written down on `EA07` in
[`benchmarks/queries.py`](benchmarks/queries.py) and pinned by
`tests/test_blast_radius_semantics.py`.

(On the real layer, which this section is about, `EA07` is simply empty like
most of the catalog — there are no `Sensor` nodes for it to start from.)

`EA01` and `EA02` are empty here rather than erroring, on both builds, because
the real layer has no `USES_OPERATOR` edges for their opening `MATCH` to bind.

**The hero question is one of the empty ones.** It walks
`Model -[:USES_OPERATOR]-> Operator`, and the real layer has no `USES_OPERATOR`
edges at all: it records which kernels implement which operators, but nothing
records which operators a model uses. `EA01`, `EA02` and `EA11` therefore cannot
be asked of the public-source subgraph — use the full graph for those.

`tests/test_real_layer_shape.py` pins the shape above, so this section fails a
test rather than going quietly stale. Counts are deliberately not pinned; they
move whenever ONNX Runtime publishes.

No **generated** number is a claim about any real product — attaching invented
latency figures to real part numbers would produce a dataset that looks
authoritative and isn't. The real layer, by contrast, is checkable line by line
against its upstream sources, and `tests/test_real_layer.py` does exactly that. Deployment metrics are *derived* from a documented cost model rather
than drawn at random, so a board missing a kernel really does pay for it.
Read the **[dataset card](DATASET_CARD.md)** — which covers intended and
out-of-scope uses, known limitations and biases — plus
[`docs/data-provenance.md`](docs/data-provenance.md) for the cost model, before
quoting anything.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m etl.download_data     # fetch 3 public sources + generate the fleet
python -m demo.demo             # narrated walkthrough, in-process, no server
```

### On Linux, install these first

`samyama` publishes a macOS wheel and an sdist, so on Linux `pip` builds the
Rust extension from source. maturin fetches its own Rust toolchain, but the
system still has to supply venv support, a C compiler and clang's builtin
headers — a stock Ubuntu 24.04 image has none of the three:

```bash
sudo apt install -y python3-venv build-essential python3-dev
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)"
pip install -e ".[dev]"         # ~3 min of cargo build
```

Without them the install fails three times, and no message names its real cause:

| Failure | Actually missing |
|---|---|
| `ensurepip is not available` from `python -m venv` | `python3-venv` |
| `could not compile 'proc-macro2' (build script)` … `No such file or directory (os error 2)` | a C linker (`cc`) |
| `zstd.h:16:10: fatal error: 'stddef.h' file not found` | clang's builtin headers |

Installing `clang` and `libclang-dev` supplies those headers directly and should
remove the need for `BINDGEN_EXTRA_CLANG_ARGS`.

`demo.demo` runs the engine **embedded** — no server, no Docker, nothing to
start. To use a running server instead:

```bash
# from the samyama-graph checkout
./target/release/samyama --http-port 8080

python -m etl.loader --url http://127.0.0.1:8080        # ~13s for 25K/76K
python -m benchmarks.run_benchmark --url http://127.0.0.1:8080
python -m mcp_server.server                             # expose over MCP
pytest                                                  # the whole suite
```

Scale the fleet with `--scale` (`1.0` ≈ 24K nodes) and change the world with
`--seed`. Same seed, same graph, every time.

After loading, the loader counts edges per type against what it intended and
reports `verified: N of N intended edges across 22 types`. It exits non-zero if
the graph holds fewer (an endpoint id did not resolve -- edges are created in
batches sharing one `MATCH`, so one bad id drops its whole batch) or more (the
graph was not empty, or two nodes share an `id`). `--no-verify` skips the
check.

## Load it without building it

A prebuilt `.sgsnap` snapshot of the graph (both layers, 992 KB — the file is
itself gzip, 10.8 MB uncompressed) is published on the engine repo's releases,
so you can skip the ETL entirely:

```bash
# 1. start the engine (from a samyama-graph checkout)
./target/release/samyama --http-port 8080

# 2. fetch and import the snapshot
curl -fL -o edge-ai-kg.sgsnap \
  https://github.com/samyama-ai/samyama-graph/releases/download/kg-snapshots-v9/edge-ai-kg.sgsnap
curl -X POST -F "file=@edge-ai-kg.sgsnap" http://127.0.0.1:8080/api/snapshot/import

# 3. ask it something
python -m benchmarks.run_benchmark --url http://127.0.0.1:8080
```

**Import takes 0.31 s** — median of 5 runs against a fresh server 1.7.0,
`kg-snapshots-v9`, range 0.225–0.396 s, measured 2026-09-09 (#45). Building the
same graph with `python -m etl.loader` takes **24.4 s**, so the snapshot is
about **80× faster**.

That comparison is only fair if you say what each one does: a `.sgsnap` is
**serialised internal state**, and the loader is a **build** — it renders
Cypher, parses it, mints ids and constructs indexes. The snapshot is faster
because it skips all of that, which also means it can only reproduce a graph
someone already built. The honest counterpart on the Neo4j side is restoring a
backup, not `LOAD CSV`; that has not been measured (#47).

Two things worth knowing before you quote the number:

- **The download is slower than the import.** Fetching the 992 KB file took
  1.14 s here — about four times the import it precedes.
- **The published snapshot holds 25,145 nodes / 76,291 edges**, not the 25,150 /
  76,303 a fresh build produces. It was exported from a slightly earlier build,
  and `data/` is not pinned (see `docs/build-manifest.json`).

16 of the 17 catalog queries then present were verified to return rows against
the imported snapshot, not just against a freshly-loaded graph — re-check with
`--verify-queries` below. `EA18` and `EA19` postdate that run. The exception is
`EA17`, which needs an engine that walks variable-length paths; the 1.7.0 server
does not (engine note 12), and it raises there rather than quietly answering one
hop deep.

Reproduce, including the export side (0.63 s, 989 KB):

```bash
python -m benchmarks.snapshot --url http://127.0.0.1:8080 \
  --file edge-ai-kg.sgsnap --repeats 5 --verify-queries \
  --restart-cmd '<command that restarts your server with an empty data dir>'
```

**Import appends, it does not replace.** Running the import twice against one
server leaves both copies — 76,303 edges became 152,606. The flow above starts
from a fresh server so it is correct as written; the benchmark refuses to time
an import into a non-empty graph for the same reason.

Export your own after any change:

```bash
curl -X POST -o edge-ai-kg.sgsnap http://127.0.0.1:8080/api/snapshot/export
```

## The query catalog

20 queries in [`benchmarks/queries.py`](benchmarks/queries.py), each recording
the question it answers and why it's awkward without a graph. On the
**embedded** build, **19 of the 20 return rows** against the **full** graph at
`--scale 1.0` — `EA21` was run there and answers in 0.1 ms, median of five
after one warm-up (its first, cold call is 23 ms). The timings —
median 5.8 ms, slowest `EA17` at 95 ms — are from the sweep of the **17**
queries that existed when it was run; `EA18`, `EA19` and `EA21` post-date it
and are not in that median.

**Over HTTP, that is two fewer — 17 — and it is an inference, not a sweep.**
`EA17` raises on the 1.7.0 server, which *is* measured (engine note 12);
`EA21` walks the same variable-length shape and has not been run there at all.
The count is those subtracted from the embedded result. No full-graph HTTP
sweep has been run since `EA18`, `EA19` and `EA21` were added.

Both counts above describe the **full** graph. The only recorded HTTP run is
over the **real layer**, in the section above, which is where the "6 of the
20" figure comes from — two different graphs, not two readings of one.

The other one is `EA18`, which asks which deployments miss a clinical task's
latency budget: **none do**, on either build. All 1,440 (deployment, task) pairs
are inside budget, the worst at 54.5% of it — that is a property of the data
rather than of the engine, so it holds wherever the query runs. The empty result
is the answer rather than a gap, and `tests/test_latency_budget.py` pins both
the zero and the reason for it.

(Re-measured at 1.7.1 over 17 queries with
`python -m benchmarks.run_benchmark`; the previous 14 ms / 73 ms pair was 16
queries at 0.6.1 and is not comparable — the engine moved and so did the
catalog. `EA18` and `EA19` postdate that run. Timed separately on 2026-09-21,
embedded 1.7.1 at `--scale 1.0`, median of five after one warm-up: `EA18` 37.5
ms (36.9 ms when first recorded), above the median and well under `EA17`, and
`EA19` 0.1 ms. `EA21` was timed the same way on 2026-09-24: 0.1 ms. These figures are hand-recorded and **not pinned by a test**,
unlike the node and edge counts on this page, which
`tests/test_published_counts.py` checks: they are machine-dependent, so the
command is the thing to trust, not the numbers. Expect them to drift.)

Against `--layers real` only 6 return rows. Which six is
[tabulated under Data](#data-whats-real-whats-synthetic) — that table is the
**server** run, and this paragraph's timings are embedded, so it is the
partition being shared between them and not the measurement. The cause is the
same either way: that subgraph has no clinical spine. **EA13–EA16 run entirely on real data**, so
their answers can be checked against the upstream sources; **EA17 needs an
engine that walks variable-length paths**, which the 1.7.0 server does not
(engine note 12).

| id | Question |
|---|---|
| EA01 | Which operators fall back to CPU for this model on this accelerator? |
| EA02 | Which operators have the fewest kernels fleet-wide? |
| EA03 | Which boards meet this clinical task's latency budget? |
| EA04 | What does int8 quantization unlock that fp32 can't fit? |
| EA05 | How much of the ONNX surface does each accelerator class cover? |
| EA06 | If a vendor drops one kernel, what breaks? |
| EA07 | Trace electrode → DSP pipeline → model → board |
| EA08 | Which runtime gives the widest coverage per accelerator? |
| EA09 | Which battery-powered boards are certified for regulated tasks? |
| EA10 | What does CPU fallback actually cost in latency? |
| EA11 | Which models are CPU-only no matter which board you pick? |
| EA12 | How concentrated is the fleet on one silicon vendor? |
| **EA13** | **REAL:** which ai.onnx operators does ONNX Runtime implement on CPU but not CUDA? |
| **EA14** | **REAL:** MLPerf Tiny v1.2 throughput leaders per benchmark task |
| **EA15** | **REAL:** which operators are registered on only one execution provider? |
| **EA16** | **REAL vs SYNTHETIC:** what is measured and what is generated |
| **EA17** | **EMBEDDED-ONLY** (its `*0..` walk; note 12): this sensor stops — what stops with it, and what stops *only* because of it? |
| **EA18** | **EMPTY ON THIS FLEET:** which deployments miss a clinical task's latency budget, and which operators have no kernel on their accelerator? |
| **EA19** | **COMPLIANCE:** this sensor fails — which certifications does that touch, through the tasks that require it? |

## Engine notes

This KG is built against **Samyama Graph v1.7.0 (OSS)**. Building it surfaced
several engine behaviours that the loader and queries work around — including
two that **silently return wrong rows** rather than erroring:

- a bound variable re-used in a later `MATCH` is not always joined, producing a cartesian product ([#360](https://github.com/samyama-ai/samyama-graph/issues/360));
- `RETURN DISTINCT` is a no-op ([#361](https://github.com/samyama-ai/samyama-graph/issues/361));
- `ORDER BY` on a `RETURN` alias is dropped, and only the first sort key applies ([#362](https://github.com/samyama-ai/samyama-graph/issues/362));
- aggregating a bare node variable returns N rows of `1` ([#363](https://github.com/samyama-ai/samyama-graph/issues/363));
- `DETACH DELETE` doesn't clear property columns, so deleted values resurrect ([#364](https://github.com/samyama-ai/samyama-graph/issues/364));
- `min()` mis-compares an int sentinel against float values ([#365](https://github.com/samyama-ai/samyama-graph/issues/365));
- the tenant/graph argument is ignored on the OSS HTTP path ([#366](https://github.com/samyama-ai/samyama-graph/issues/366));
- negated pattern predicates and `CREATE CONSTRAINT` don't parse ([#367](https://github.com/samyama-ai/samyama-graph/issues/367)).

All nine are filed upstream — tracking issue [samyama-graph#368](https://github.com/samyama-ai/samyama-graph/issues/368).

**Notes 10 to 13b are recorded but not filed**, for two different reasons.
Notes 10 and 11 were **version skew** — `samyama` 0.6.1 against a 1.7.0 server,
not a defect in either — which is what #56 settled, so there is nothing to
file. Notes 12, 13 and 13b *are* engine behaviours: note 12 is a capability the
1.7.0 OSS server lacks and the embedded 1.7.1 build has, which reads as a
version gap rather than a defect, and notes 13 and 13b were found on embedded
1.7.1 while writing `EA18` and are not yet filed.

- `samyama` 0.6.1 did not register a second `WITH` that introduces a new
  alias, where the 1.7.0 server did ([note 10](docs/engine-notes.md)); it does
  not reproduce on the `>=1.7.1` floor;
- the same for the type `sum(CASE ...)` returns, which silently dropped a
  `WHERE` on it ([note 11](docs/engine-notes.md));
- the 1.7.0 **server** does not traverse a variable-length relationship,
  bounded or not — it returns only the zero-length match, and rejects `size(r)`
  on one — where the embedded 1.7.1 build walks it
  ([note 12](docs/engine-notes.md)). That is why `EA17` is embedded-only, and
  why it needs the `samyama>=1.7.1` floor #104 landed.

Notes 10 and 11 need no workaround in the catalog: #56 resolved both by
raising the floor, and neither reproduces on `samyama>=1.7.1`. Note 12 has no
workaround either, and one is not possible: there is no way to write "walk a
chain of unknown length" that the 1.7.0 server executes, so `EA17` is
embedded-only rather than reshaped.

`EA01`, `EA02` and `EA04` used to carry `xfail` marks for notes 10 and 11 —
four test functions, six reported outcomes, since two of them are parametrised
sweeps. Those marks were written against `samyama` 0.6.1. **#105 raised the floor to
1.7.1 and removed them, reaching `main` with #104**, and the three queries now
pass unmarked under `pytest` and under `run_benchmark`.

Each is documented with a minimal reproduction and the workaround used in
[`docs/engine-notes.md`](docs/engine-notes.md). Because of these,
[`tests/test_correctness.py`](tests/test_correctness.py) validates query
**results** against ground truth computed in Python — a query that runs and
returns plausible rows is not evidence that it is right.

## Structure

```
etl/          onnx_catalog.py, ort_kernels.py, mlperf_tiny.py, real_layer.py (real)
              generate.py (synthetic) + loader.py
schema/       edge_ai_kg.cypher — indexes and documented relationship shapes
benchmarks/   the 19-query catalog + runner
mcp_server/   7 MCP tools shaped around deployment questions
demo/         two walkthroughs (question-driven + 6-beat story) + recorded gif
scripts/      record_gif.sh — long-form demo recording
docs/         schema, data provenance, engine notes, scope decisions
DATASET_CARD.md  HF-style card: structure, provenance, intended + out-of-scope uses
tests/        ~100 tests: parsing, fleet + real-layer invariants, query correctness
```

## License

Apache-2.0. The ONNX operator catalog is Apache-2.0 from the ONNX project;
everything else is generated.
