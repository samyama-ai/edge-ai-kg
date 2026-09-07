// Edge AI Deployment Knowledge Graph -- schema
//
// Node labels: Vendor, SoC, Accelerator, Board, Runtime, Operator, Kernel,
//              Model, ModelVariant, Sensor, SignalStage, ClinicalTask,
//              BenchmarkTask, Dataset, Certification, Deployment
//
// Every node carries `provenance` ("real" | "synthetic") and `source`.
//
// Edge types:  MADE_BY, HAS_SOC, HAS_ACCELERATOR, TARGETS, IMPLEMENTS,
//              RUNS_ON, PROVIDED_BY, USES_OPERATOR, VARIANT_OF, SOLVES,
//              TRAINED_ON, REQUIRES_SENSOR, FEEDS, NEXT_STAGE, PRECEDES,
//              OF_VARIANT, ON_BOARD, VIA_RUNTIME, USES_ACCELERATOR,
//              CERTIFIED_FOR, GOVERNED_BY, MEASURES
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

// --- lookup indexes: each one is filtered on by something ---
// Traced to the queries that need them; tests/test_schema_indexes.py fails if
// an index is added here that nothing filters on.
CREATE INDEX ON :Operator(name);              // EA01, EA06, EA13, EA15; mcp operator_risk
CREATE INDEX ON :Accelerator(kind);           // EA05, EA08, EA11; mcp coverage_by_kind
CREATE INDEX ON :ModelVariant(precision);     // EA07; mcp boards_for_task, device_path
CREATE INDEX ON :Deployment(provenance);      // EA14
CREATE INDEX ON :Kernel(provenance);          // EA15, EA16
CREATE INDEX ON :Kernel(execution_provider);  // EA13

// Removed as unused (#18): nothing filtered on any of these.
//   Operator(category)       projected and ORDER BY'd, never a predicate
//   Model(family)            projected by EA11, never a predicate
//   ClinicalTask(category)   read by nothing at all
//   Board(provenance)        read by nothing at all
//   Accelerator(provenance)  read by no query; one test reads it unfiltered
//
// Deliberately NOT added: `mcp_server` filters `ClinicalTask(name)` and
// `Model(name)`, which have no index. Measured on the shape
// `mcp_server.boards_for_task` uses, adding `ClinicalTask(name)` moved a 5.99ms
// query to 5.55ms -- inside the noise, so it would be load cost for nothing.

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
