# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A knowledge graph of edge-AI deployment — boards, SoCs, accelerators, runtimes,
ONNX operators, kernels, quantized model variants and biosignal pipelines — built
on the **Samyama Graph** engine (server 1.7.0; embedded `samyama>=1.7.1`). The
repo holds the loader, the synthetic generator and the query catalog; the
engine itself lives in `samyama-ai/samyama-graph`.

The hero question it exists to answer: *which operators in this model have no
kernel on this accelerator, and therefore silently fall back to the CPU?*

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m etl.download_data      # REQUIRED first — builds ./data (gitignored)
```

`etl.download_data` fetches three public sources and generates the fleet into
`data/`. **Nothing else works until it has run** — the demos and
`tests/test_correctness.py` call `load_cached()` and skip or fail without it.
Add `--force` to re-fetch upstreams, `--seed` / `--scale` to change the fleet.

### Installing `samyama` on Linux

`samyama` publishes only a **macOS x86_64 wheel** plus an sdist, so Linux builds
the Rust extension from source. maturin auto-downloads a Rust toolchain, but the
host must supply a C compiler and clang's builtin headers. On a bare Ubuntu box
the build fails twice — first on a missing linker, then on `zstd-sys` bindgen not
finding `stddef.h`. Fix:

```bash
sudo apt install -y build-essential python3-dev
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"
export LIBCLANG_PATH=/usr/lib/llvm-18/lib     # only if libclang isn't found
pip install -e ".[dev]"                        # ~3 min of cargo build
```

(Installing `clang`/`libclang-common-*-dev` is the cleaner fix if you have sudo;
`BINDGEN_EXTRA_CLANG_ARGS` is the workaround when you only have gcc.)

## Commands

```bash
pytest                                  # the whole suite, against an embedded engine
pytest tests/test_correctness.py -x     # the ones that matter most
pytest tests/test_correctness.py::test_ea01_fallback_audit_matches_ground_truth
ruff check .                            # clean; config in pyproject.toml

python -m demo.demo --fast              # 6-beat story, self-contained
python -m demo.questions --only EA01 EA06 --fast
python -m benchmarks.run_benchmark --only EA01 --rows 20
python -m benchmarks.compare_neo4j --repeats 15 --natural   # head-to-head vs Neo4j (needs one running and NEO4J_PASSWORD set)
python -m benchmarks.compare_neo4j --repeats 15 --force-wipe # same, against a server that already holds a fleet
python -m etl.loader --layers real      # load only the public-source subgraph
python -m etl.loader --no-verify        # skip the post-load edge count

python -m etl.manifest --check          # has the build moved since docs/build-manifest.json?
python -m etl.manifest --write          # re-record it, then commit the diff
python -m mcp_server.server             # 7 MCP tools over the graph
```

### Embedded vs server — the thing that trips people up

`SamyamaClient.embedded()` runs the engine **in-process and in-memory**. It does
**not** persist: a second Python process sees an empty graph (verified). So:

- `python -m etl.loader` with no `--url` loads a graph and then throws it away —
  it is only a timing/verification exercise. A following
  `python -m benchmarks.run_benchmark` (embedded) will find nothing.
- `demo/demo.py`, `demo/questions.py` and `tests/test_correctness.py` each build
  the graph **inside their own process**, which is why they are self-contained.
- To load once and query repeatedly, run the engine over HTTP and pass `--url`
  to both commands:

  ```bash
  ./target/release/samyama --http-port 8080   # from a samyama-graph checkout
  python -m etl.loader --url http://127.0.0.1:8080
  python -m benchmarks.run_benchmark --url http://127.0.0.1:8080
  ```

`--graph` looks like tenant isolation but is ignored on OSS (engine note 7 below);
everything lands in `default`. Every entry point — both CLIs, `mcp_server/config.yaml`
and the four `GRAPH` constants — now defaults to `default`, pinned by
`tests/test_cli_defaults.py`. They did not always agree, and nothing noticed,
because the argument being ignored made the disagreement invisible.

## Architecture

### The `Fleet` intermediate representation

Nothing writes Cypher directly from source data. Every path funnels through
`etl.generate.Fleet` — a dataclass holding `nodes: dict[label, list[dict]]` and
`edges: list[tuple]` — which `etl/helpers.py` then renders into batched `CREATE`
statements. Two independent producers fill the same Fleet:

- **`etl/generate.py`** (synthetic): deterministic from `--seed` (default
  `20260814`); `--scale` multiplies fleet size. Same seed → same graph, always.
  Vendor/board names are deliberately fictional so no generated number can be
  read as a claim about a real product.
- **`etl/real_layer.py`** (real): stitches `onnx_catalog.py` (ONNX operator
  catalog), `ort_kernels.py` (ONNX Runtime kernel registrations) and
  `mlperf_tiny.py` (MLPerf Tiny v1.2 measurements) onto an existing — possibly
  empty — Fleet.

Each `etl/*.py` source module follows the same shape: `download()` → `parse_*()`
→ `build()` writes `data/<source>/*.json` → `load_cached()` reads it back.
Parsers are tested against fixture text, so they can be edited without network.

`Fleet.add_nodes()` does two load-bearing things: it **copies** each row (the
generator keeps mutating the originals afterward, attaching `_`-prefixed
internals) and stamps `provenance` (`"real"` | `"synthetic"`) plus `source` on
every node. `_`-prefixed keys are stripped and never reach the graph. The
real/synthetic split is therefore *queryable* (see EA16), not a README claim.

### The query catalog is the single source of truth

`benchmarks/queries.py` holds 20 entries (`EA01`–`EA19` and `EA21`), each with
`question` / `why_graph` / `cypher`, exported as `QUERIES` and `BY_ID`. It is
consumed by `benchmarks/run_benchmark.py`, `demo/questions.py` and
`tests/test_correctness.py`. Editing a query changes the benchmark, the demo and
the tests at once. `EA13`–`EA16` run entirely on the real layer, so the
`test_correctness.py` fixture must load **both** layers or they come back empty.

`mcp_server/server.py` deliberately re-states rather than imports these queries —
its tools take typed parameters and interpolate them, so they are parameterized
variants, not the catalog verbatim. Keep the two in sync by hand when the graph
shape changes.

## Engine constraints — read `docs/engine-notes.md` before writing Cypher

Notes 1-9 document nine v1.7.0 behaviours, several of which **return wrong rows
rather than erroring**. The loader and every catalog query work around them, so
these are not trivia — breaking one of these rules produces confident,
plausible, wrong output.

`python -m benchmarks.engine_notes_probe --scale 300` re-runs notes 1-6, 8 and
9 against the installed engine, and on **embedded** 1.7.1 none of them
reproduces. That does not retire a rule: the notes were measured on the 1.7.0
**HTTP server**, a different binary that has not been re-probed, and note 7
(no tenant boundary on that server) has no probe at all. The rules below stay
binding until the server is measured too.

**Notes 10 and 11 are resolved, and were never what they said they were.** They
read as disagreements between the embedded build and the HTTP server -- a
second `WITH` introducing a new alias failing embedded (note 10), and
`sum(CASE ...)` returning a different type on each build and silently dropping
a `WHERE` on it (note 11). They are neither. They are one pip install against
another: `pyproject.toml` then asked for `samyama>=0.6.0` (the old floor), pip
resolved 0.6.1, and the notes were measured against a 1.7.0 server.

`pyproject.toml` now floors the engine at `samyama>=1.7.1`, on which neither
reproduces embedded -- the same caveat as above: nothing here re-probed the
server. `EA01`, `EA02` and `EA04` are correct under `pytest` and under
`run_benchmark`, no test carries a #56 `xfail`, and #56's code half is closed.
The two notes stay in `docs/engine-notes.md` as history, because the wrong
conclusion is the useful part: two builds were assumed to differ for three
weeks when the difference was a version.
`docs/engine-notes.md` carries a banner saying the same; rewriting the notes
themselves is #109 (which replaces #94, closed unmerged).

`tests/test_engine_version.py` keeps the floor honest -- it asserts the
declared dependency, the running engine, **and** re-runs note 11's own
reproduction, because note 11 does not raise. On a downgraded build it makes
`EA04` return confident extra rows rather than fail.

**Notes 12, 13 and 13b belong with 1-9, not with the carve-out above** -- all
three return wrong rows rather than erroring. Note 12: the 1.7.0 server does
not *traverse* a variable-length relationship the embedded 1.7.1 build walks,
and it does not error, it returns fewer rows -- which is why `EA17` is
embedded-only rather than reshaped. Notes 13 and 13b were measured on embedded
1.7.1 while writing `EA18`: a `WHERE` on an `OPTIONAL MATCH` mentioning a
**`WITH`-introduced** alias drops the unmatched rows, and an expression mixing
a grouping key with an aggregate in one projection returns `NULL`. Note 13 has
a rule in the list below; 13b's workaround is to compute the expression one
`WITH` later.

The rules that follow from notes 1-9 and 13:

- **Project through `WITH` before `RETURN`, and sort on the `WITH` alias.**
  `ORDER BY` on a `RETURN`-introduced alias is silently dropped. With `LIMIT`
  that yields an arbitrary N rows dressed up as a top-N.
- **One `ORDER BY` key only.** Later keys are ignored.
  `test_order_by_is_actually_applied` enforces this across the whole catalog.
- **Aggregate a property, never a bare node variable.** `count(DISTINCT op)` over
  a multi-variable `MATCH` returns N rows of `1`; `count(DISTINCT op.id)` is
  correct. Every catalog aggregate is written the second way.
- **Never re-bind a variable in a trailing position of a second `MATCH`.** The
  join isn't enforced and you get a cartesian product — at scale, even the
  comma-separated single-`MATCH` form breaks. Use one linear pattern plus
  `sum(CASE WHEN ...)` conditional aggregation instead (this is why EA04 looks
  the way it does).
- **`RETURN DISTINCT` is a no-op.** `DISTINCT` inside an aggregate works. Group
  with `WITH` + an aggregate to deduplicate.
- **No negated pattern predicates.** Anti-joins are
  `OPTIONAL MATCH ... WITH ... count(k) AS n ... WHERE n = 0`.
- **Never filter an `OPTIONAL MATCH` on a `WITH`-introduced alias.** It drops
  the unmatched rows -- the `OPTIONAL` becomes an inner join, silently, which
  turns the anti-join above into the opposite of what it is for (note 13). A
  literal or a `MATCH`-bound alias is safe; if a `WITH` alias is unavoidable,
  `collect` and filter after the aggregation.
- **Keep numeric literal types matching the stored property type.** `WHERE x > 0.5`
  against an int-typed property returns nothing; `min(CASE ... ELSE 999999 END)`
  returns the int sentinel while `999999.0` works.
- **`DETACH DELETE` is not a reset.** The columnar property store survives it, so
  a property removed from the source data resurrects onto newly created nodes.
  A genuine reset means stopping the server and deleting its data directory.
- **`<>` matches null properties.** Use `IS NULL` / `IS NOT NULL`.
- **No `CREATE CONSTRAINT`.** Only `CREATE INDEX ON :Label(prop)`, as in
  `schema/edge_ai_kg.cypher`. `id` uniqueness is a loader invariant, guaranteed by
  deterministic id minting in `etl/generate.py`.

`etl/helpers.py` encodes the write-side equivalents: strings are double-quoted
with quotes/newlines *stripped* (the parser has no escape syntax), floats are
forced to fixed notation (`1e-05` is a parse error), and edge endpoints are
matched with `WHERE id = ...` rather than inline map properties, because inline
properties don't trigger an index scan on this build.

## Testing philosophy

`tests/test_correctness.py` recomputes expected answers **in Python from the
Fleet** and asserts the engine agrees. This exists because a query that runs and
returns plausible rows is not evidence it is right — EA04 once returned 12
confident rows pairing one model's fp32 variant with another model's int8. The
canary there is that int8 size is by construction exactly ¼ of fp32 size, so a
cartesian product is detectable; `test_ea04_shape_is_not_a_cartesian_product`
pins it on a purpose-built 4-deployment fixture.

When adding a query to the catalog, add a ground-truth test alongside it. When a
result looks suspicious, check `docs/engine-notes.md` before assuming the data is
wrong — and validate any proposed workaround on a full-size graph, since at least
one bug only appears once cardinalities are real.
