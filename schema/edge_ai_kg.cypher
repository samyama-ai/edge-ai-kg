// Edge AI Deployment Knowledge Graph -- schema
//
// Node labels: Vendor, SoC, Accelerator, Board, Runtime, Operator, Kernel,
//              Model, ModelVariant, Sensor, SignalStage, ClinicalTask,
//              BenchmarkTask, Dataset, Certification, Deployment, Site
//
// Every node carries `provenance` ("real" | "synthetic") and `source`.
//
// Edge types:  MADE_BY, HAS_SOC, HAS_ACCELERATOR, TARGETS, IMPLEMENTS,
//              RUNS_ON, PROVIDED_BY, USES_OPERATOR, VARIANT_OF, SOLVES,
//              TRAINED_ON, REQUIRES_SENSOR, FEEDS, NEXT_STAGE, PRECEDES,
//              OF_VARIANT, ON_BOARD, VIA_RUNTIME, USES_ACCELERATOR,
//              CERTIFIED_FOR, GOVERNED_BY, MEASURES, DEPLOYED_AT
//
// This engine accepts `CREATE INDEX ON :Label(prop)`. It does NOT parse
// `CREATE CONSTRAINT ... REQUIRE ... IS UNIQUE`; uniqueness of `id` is
// guaranteed by the loader, which mints ids deterministically.
//
// --- What the indexes cost, measured (#18) ---
//
// The `id` indexes are not an optimisation, they are what makes a load
// finish. Embedded engine, --scale 1.0, 25,150 nodes and 76,303 edges:
//
//     16 id indexes      nodes 0.35s   edges  22.1s
//     no indexes at all  nodes 0.38s   edges 233.7s      <- 10.6x slower
//
// `etl/helpers.py` resolves every edge endpoint with `WHERE v.id = ...`, so
// without the index each of the 76,303 edges costs a label scan. Node creation
// is unaffected because nothing is looked up to create a node.
//
// The non-id indexes are the opposite: they cost load time and buy nothing at
// load. Median of 4 runs -- 27 indexes 24.68s, 16 id only 23.13s -- so the
// extras are ~1.5s, about 6% of the load, for queries that run in milliseconds.
// Five that nothing filtered on were removed; removing them changed no catalog
// row and no query time outside run-to-run noise (+9ms across all 16, on 562ms).
//
// Each surviving non-id index names what needs it. An index only helps a
// lookup: a property that is merely projected or sorted on a `WITH` alias
// cannot use one, which is why `Operator(category)` and `Model(family)` went.
//
// That guarantee is per load, not per graph. The loader resets the graph first
// unless told not to; `--no-reset` against a populated graph mints every id a
// second time and nothing rejects the write. The run does not survive it: with
// one id bound to two nodes, an edge batch's MATCH binds both ends more than
// once -- one submitted edge becomes four -- and the engine is OOM-killed
// during edge creation. See tests/test_id_uniqueness.py.

// --- id indexes: one per label, and load-critical (see above) ---
CREATE INDEX ON :Vendor(id);
CREATE INDEX ON :SoC(id);
CREATE INDEX ON :Accelerator(id);
CREATE INDEX ON :Board(id);
CREATE INDEX ON :Runtime(id);
CREATE INDEX ON :Operator(id);
CREATE INDEX ON :Kernel(id);
CREATE INDEX ON :Model(id);
CREATE INDEX ON :ModelVariant(id);
CREATE INDEX ON :Sensor(id);
CREATE INDEX ON :SignalStage(id);
CREATE INDEX ON :ClinicalTask(id);
CREATE INDEX ON :Dataset(id);
CREATE INDEX ON :Certification(id);
CREATE INDEX ON :Deployment(id);
CREATE INDEX ON :BenchmarkTask(id);
CREATE INDEX ON :Site(id);

// --- lookup indexes: each one is filtered on by something ---
// Traced to the queries that need them; tests/test_schema_indexes.py fails if
// an index is added here that nothing filters on.
CREATE INDEX ON :Operator(name);              // EA06; mcp kernel_blast_radius
CREATE INDEX ON :Accelerator(kind);           // mcp fallback_audit, operator_coverage
CREATE INDEX ON :ModelVariant(precision);     // EA07; mcp boards_for_task, device_path
CREATE INDEX ON :Deployment(provenance);      // EA14
CREATE INDEX ON :Kernel(provenance);          // EA15
CREATE INDEX ON :Kernel(execution_provider);  // EA13
//
// Only queries that put the property in a *predicate* are listed. Several more
// read these properties through `WITH x.p AS ...` -- EA01/EA13/EA15 project
// `op.name`, EA05/EA08 project `a.kind`, EA16 projects `k.provenance` -- and a
// projection cannot use an index, so naming them here would be the same
// overclaim this file removes five indexes for.
// `sum(CASE WHEN v.precision = ...)` in EA04 is aggregation over rows the MATCH
// already produced, so it is not a user either.
//
// `Accelerator(kind)` is the weakest of the six. Its only catalog use is EA11's
// `WHERE a.kind <> "MCU-CPU"`, and an inequality is not served by an index scan
// -- `<>` also matches nulls on this engine (engine note 8b). The index is
// earned by `operator_coverage` and `fallback_audit`, which do `a.kind = ...`;
// EA11 is listed as a user of the property, not as a beneficiary of the index.

// Removed as unused (#18): nothing filtered on any of these.
//   Operator(category)       projected and ORDER BY'd, never a predicate
//   Model(family)            projected by EA11, never a predicate
//   ClinicalTask(category)   read by nothing at all
//   Board(provenance)        read by nothing at all
//   Accelerator(provenance)  read by no query; one test reads it unfiltered
//
// Deliberately NOT added. Seven properties are filtered on somewhere and carry
// no index. All were measured together: adding all seven moved the whole
// 16-query catalog by +0.9ms on 575ms, individual queries swinging -8.6ms to
// +3.0ms. Noise. Every one sits on a small label -- Deployment 1,513 rows,
// Operator 376, Board 134, ClinicalTask 18 -- where the scan is already cheap.
//
//   ClinicalTask(name)               mcp boards_for_task
//   ClinicalTask(latency_budget_ms)  EA09
//   Model(name)                      mcp fallback_audit
//   Operator(domain)                 EA13, `AND op.domain = "ai.onnx"`
//   Deployment(fits)                 EA06, EA07, demo/demo.py
//   Deployment(latency_ms)           EA03
//   Board(battery_powered)           demo/demo.py
//   Accelerator(is_cpu_fallback)     EA11 -- see below
//
// `Accelerator(is_cpu_fallback)` is the newest of these and was measured on its
// own (#69): adding it moved EA11 from 105.6ms to 115.5ms and the whole catalog
// from 541ms to 615ms -- *slower*, because `Accelerator` has 91 rows and the
// index is overhead the scan does not need. EA11 stopped filtering
// `Accelerator(kind)` at the same time, which is why that index now cites only
// the two MCP tools.
//
// `tests/test_schema_indexes.py` holds this list to the queries, in both
// directions: an index nothing filters on fails, and a *new* filtered property
// with neither an index nor a line here fails too.

// --- Relationship shapes (documentation only) ---
// (:Board)-[:HAS_SOC]->(:SoC)-[:HAS_ACCELERATOR]->(:Accelerator)
// (:Board)-[:MADE_BY]->(:Vendor)          (:SoC)-[:MADE_BY]->(:Vendor)
// (:Board)-[:CERTIFIED_FOR]->(:Certification)
// (:Runtime)-[:TARGETS]->(:Accelerator)
// (:Kernel)-[:IMPLEMENTS]->(:Operator)
// (:Kernel)-[:RUNS_ON]->(:Accelerator)
// (:Kernel)-[:PROVIDED_BY]->(:Runtime)
// (:Model)-[:USES_OPERATOR {count}]->(:Operator)
// (:ModelVariant)-[:VARIANT_OF]->(:Model)
// (:Model)-[:SOLVES]->(:ClinicalTask)     (:Model)-[:TRAINED_ON]->(:Dataset)
// (:ClinicalTask)-[:REQUIRES_SENSOR]->(:Sensor)
// (:ClinicalTask)-[:GOVERNED_BY]->(:Certification)
// (:Sensor)-[:FEEDS]->(:SignalStage)-[:NEXT_STAGE]->(:SignalStage)
// (:SignalStage)-[:PRECEDES]->(:Model)
// (:SignalStage)-[:USES_OPERATOR]->(:Operator)
// (:Deployment)-[:OF_VARIANT]->(:ModelVariant)
// (:Deployment)-[:ON_BOARD]->(:Board)
// (:Deployment)-[:VIA_RUNTIME]->(:Runtime)
// (:Deployment)-[:USES_ACCELERATOR]->(:Accelerator)
// (:Deployment)-[:MEASURES]->(:Model)          -- real MLPerf Tiny submissions
// (:Model)-[:SOLVES]->(:BenchmarkTask)        -- real MLPerf Tiny tasks
