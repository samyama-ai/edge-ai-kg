"""Benchmark / demo query catalog for the Edge AI deployment KG.

Each entry is a question an edge-AI team actually asks while trying to land a
model on custom silicon. The `why_graph` field records why the question is
awkward in a relational or document store -- most of these are variable-depth
joins across hardware, kernel-library and model structure, or anti-joins
("which operator has NO kernel here").

Engine notes:
  * `NOT (pattern)` does not parse on this build; anti-joins are written as
    `OPTIONAL MATCH ... WITH ... count(x) AS n ... WHERE n = 0`.
  * String literals must be double-quoted.
"""
from __future__ import annotations

# `EA17`'s subject, in one place. The query names it ten times -- twice in each
# of five legs -- and a retarget that rewrote only some of them would leave one
# leg answering about a different sensor than the rest, which reads as a real
# finding rather than an editing mistake.
EA17_SUBJECT = "sensor:00000"

QUERIES: list[dict] = [
    {
        "id": "EA01",
        "title": "CPU fallback audit for one model on one accelerator",
        "question": ("Which operators in this model have no kernel on this "
                     "accelerator, and therefore silently fall back to the CPU?"),
        "why_graph": ("The anti-join is over a 3-hop path "
                      "(Model->Operator<-Kernel->Accelerator). In SQL this is a "
                      "NOT EXISTS over a join of four tables, re-written per "
                      "accelerator; here it is one pattern."),
        "cypher": """
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.id = "model:00000"
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.id = "accel:00001"
WITH op, count(k) AS kernels
WHERE kernels = 0
WITH op.name AS operator, op.category AS category, op.since_version AS opset
RETURN operator, category, opset
ORDER BY category
""",
    },
    {
        "id": "EA02",
        "title": "Fleet-wide operator coverage gaps",
        "question": ("Across the whole accelerator fleet, which operators used "
                     "by our models have the fewest kernel implementations?"),
        "why_graph": ("Ranks a join fan-out; the answer tells you which "
                      "operator to avoid at architecture-design time."),
        "cypher": """
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WITH op, count(DISTINCT m.id) AS models_using
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op)
WITH op, models_using, count(k) AS kernel_count
RETURN op.name AS operator, op.category AS category,
       models_using, kernel_count
ORDER BY kernel_count ASC
LIMIT 15
""",
    },
    {
        "id": "EA03",
        "title": "Boards that meet a clinical task's latency budget",
        "question": ("For a given clinical task, which boards run a variant of "
                     "a solving model inside the task's latency budget?"),
        "why_graph": ("Six hops: Task<-Model<-Variant<-Deployment->Board, with a "
                      "predicate that compares a measured value against a "
                      "budget stored on the task."),
        "cypher": """
MATCH (t:ClinicalTask)<-[:SOLVES]-(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE t.name = "Fall detection" AND d.fits = 1
  AND d.latency_ms < t.latency_budget_ms
WITH b.name AS board, b.form_factor AS form, b.power_budget_mw AS power_mw,
     v.precision AS precision, d.latency_ms AS latency_ms,
     d.fallback_op_count AS fallback_ops
RETURN board, form, power_mw, precision, latency_ms, fallback_ops
ORDER BY latency_ms ASC
LIMIT 12
""",
    },
    {
        "id": "EA04",
        "title": "What quantization unlocks",
        "question": ("Which board/model pairs do not fit at fp32 but do fit "
                     "once quantized to int8?"),
        "why_graph": ("Self-join across two variants of the same model on the "
                      "same board -- the classic 'what changed' question."),
        # IMPORTANT: written as a single linear pattern + conditional aggregation
        # rather than as a self-join. Both the two-MATCH form AND the
        # comma-separated single-MATCH form return a cartesian product on
        # v1.7.0 at this scale -- see docs/engine-notes.md item 1. The
        # invariant `int8_kb == fp32_kb / 4` makes the breakage detectable, and
        # tests/test_correctness.py asserts it.
        "cypher": """
MATCH (m:Model)<-[:VARIANT_OF]-(v:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)
      -[:ON_BOARD]->(b:Board)
WITH m.name AS model, b.name AS board, b.ram_kb AS board_ram_kb,
     sum(CASE WHEN v.precision = "fp32" AND d.fits = 0 THEN 1 ELSE 0 END) AS fp32_misses,
     sum(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN 1 ELSE 0 END) AS int8_hits,
     max(CASE WHEN v.precision = "fp32" THEN v.size_kb ELSE 0 END) AS fp32_kb,
     max(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 0 END) AS int8_kb,
     min(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN d.latency_ms ELSE 999999.0 END) AS int8_latency_ms
WHERE fp32_misses > 0 AND int8_hits > 0
RETURN model, board, board_ram_kb, fp32_kb, int8_kb, int8_latency_ms
ORDER BY fp32_kb DESC
LIMIT 12
""",
    },
    {
        "id": "EA05",
        "title": "Kernel coverage by accelerator archetype",
        "question": ("How much of the ONNX operator surface does each class of "
                     "accelerator actually implement?"),
        "why_graph": "Aggregation over a 2-hop path, grouped by a node property.",
        "cypher": """
MATCH (a:Accelerator)<-[:RUNS_ON]-(k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WITH a.kind AS accelerator_kind, count(DISTINCT op.id) AS operators_covered,
     count(DISTINCT a.id) AS accelerators
RETURN accelerator_kind, accelerators, operators_covered
ORDER BY operators_covered DESC
""",
    },
    {
        "id": "EA06",
        "title": "Blast radius of losing one kernel",
        "question": ("If a vendor drops the kernel for this operator, which "
                     "deployments regress to CPU fallback?"),
        "why_graph": ("Impact analysis -- reachability from one node out to "
                      "every affected deployment. This is the question that is "
                      "genuinely painful without a graph."),
        "cypher": """
MATCH (op:Operator)<-[:USES_OPERATOR]-(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE op.name = "Conv" AND d.fits = 1
RETURN op.name AS operator, count(DISTINCT d.id) AS deployments_at_risk,
       count(DISTINCT b.id) AS boards_affected, count(DISTINCT m.id) AS models_affected
""",
    },
    {
        "id": "EA07",
        "title": "End-to-end on-device path: sensor to board",
        "question": ("Trace one complete on-device path: sensor, signal "
                     "pipeline, model, quantized variant, deployment, board."),
        "why_graph": ("A path query. The whole point of the KG -- the physical "
                      "chain from electrode to silicon is a path, not a table."),
        # `*0..3` is a fixed bound on a chain of unknown length, and it is
        # lossy **today**. Measured at `--scale 1.0`: the longest `NEXT_STAGE`
        # chain is 13 hops, and lifting the bound to `*0..` changes the top ten
        # -- `Chest mic array` and `Thermistor array` drop out, `ECG 3-lead` and
        # `ECG 12-lead` come in. The row count is 10 either way, which is why
        # the difference is easy to miss.
        #
        # It stays bounded anyway. The 1.7.0 server traverses neither form --
        # it matches only the zero-length case, bounded or not (note 12) -- and
        # this bounded form is the one measured to return byte-identical rows
        # on both builds, because its ten lowest-latency paths sit at zero hops.
        # That is luck of this data and this `LIMIT`, not robustness, and
        # lifting the bound would be an unmeasured change to the one
        # variable-length query that currently answers the same over HTTP.
        #
        # `tests/test_blast_radius_semantics.py::test_ea07s_fixed_bound_is_lossy
        # _and_that_is_a_known_trade` pins the cost on a purpose-built chain, so
        # this stops being prose the moment it stops being true.
        "cypher": """
MATCH (s:Sensor)-[:FEEDS]->(st:SignalStage)-[:NEXT_STAGE*0..3]->(last:SignalStage)
      -[:PRECEDES]->(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE v.precision = "int8" AND d.fits = 1
WITH s.name AS sensor, s.sample_rate_hz AS hz, last.name AS final_stage,
     m.name AS model, b.name AS board, d.latency_ms AS latency_ms
RETURN sensor, hz, final_stage, model, board, latency_ms
ORDER BY latency_ms ASC
LIMIT 10
""",
    },
    {
        "id": "EA08",
        "title": "Best runtime per accelerator",
        "question": ("For each accelerator, which runtime gives the widest "
                     "operator coverage?"),
        "why_graph": ("Groups a 3-way relationship (kernel joins operator, "
                      "accelerator and runtime) that has no natural table."),
        "cypher": """
MATCH (a:Accelerator)<-[:RUNS_ON]-(k:Kernel)-[:PROVIDED_BY]->(r:Runtime)
MATCH (k)-[:IMPLEMENTS]->(op:Operator)
WITH a.kind AS accelerator_kind, r.name AS runtime,
     count(DISTINCT op.id) AS operators
WITH accelerator_kind, runtime, operators
RETURN accelerator_kind, runtime, operators
ORDER BY operators DESC
LIMIT 20
""",
    },
    {
        "id": "EA09",
        "title": "Battery-powered boards for regulated tasks",
        "question": ("Which battery-powered boards are certified for the same "
                     "standard a regulated clinical task demands?"),
        "why_graph": ("Joins two independent subgraphs (regulatory and "
                      "hardware) through a shared certification node."),
        "cypher": """
MATCH (t:ClinicalTask)-[:GOVERNED_BY]->(c:Certification)<-[:CERTIFIED_FOR]-(b:Board)
WHERE b.battery_powered = 1
WITH c.name AS certification, t.name AS task, b.name AS board,
     b.power_budget_mw AS power_mw, b.form_factor AS form
RETURN certification, task, board, power_mw, form
ORDER BY power_mw ASC
LIMIT 15
""",
    },
    {
        "id": "EA10",
        "title": "Fallback cost distribution by accelerator kind",
        "question": ("How much latency does CPU fallback actually cost, per "
                     "accelerator class?"),
        "why_graph": ("Correlates a structural property (missing kernels) with "
                      "a measured one (latency) in a single pass."),
        "cypher": """
MATCH (d:Deployment)
WHERE d.fits = 1
RETURN d.accelerator_kind AS accelerator_kind,
       count(d) AS deployments,
       avg(d.fallback_fraction) AS avg_fallback_fraction,
       avg(d.latency_ms) AS avg_latency_ms,
       max(d.latency_ms) AS worst_latency_ms
ORDER BY avg_fallback_fraction DESC
""",
    },
    {
        "id": "EA11",
        "title": "Models whose operators no accelerator can fully run",
        "question": ("Which models depend on operators that NO accelerator in "
                     "the fleet implements -- i.e. CPU-only no matter what "
                     "board you pick?"),
        "why_graph": ("Fleet-wide anti-join. Every SoC has a CPU that runs "
                      "everything, so the question that matters is not 'is it "
                      "runnable' but 'is it ever *accelerated*'. Catches the "
                      "architecture mistake before tape-out, not after."),
        "cypher": """
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.is_cpu_fallback = 0
WITH m, op, count(k) AS accel_kernels
WHERE accel_kernels = 0
RETURN m.name AS model, m.family AS family,
       count(op) AS cpu_only_ops, collect(op.name) AS operators
ORDER BY cpu_only_ops DESC
LIMIT 10
""",
    },
    {
        "id": "EA12",
        "title": "Vendor concentration in feasible deployments",
        "question": ("If we shipped every deployment that fits, how much of "
                     "the fleet would sit on a single silicon vendor?"),
        "why_graph": ("Supply-chain concentration is a 4-hop rollup "
                      "(Deployment->Board->SoC->Vendor)."),
        "cypher": """
MATCH (d:Deployment)-[:ON_BOARD]->(b:Board)-[:HAS_SOC]->(s:SoC)
      -[:MADE_BY]->(vendor:Vendor)
WHERE d.fits = 1
RETURN vendor.name AS vendor, vendor.country AS country,
       count(DISTINCT b.id) AS boards, count(d) AS deployments
ORDER BY deployments DESC
""",
    },
    # ------------------------------------------------------------------
    # EA13-EA16 run entirely on the REAL layer (provenance = "real"):
    # ONNX Runtime kernel registrations and MLPerf Tiny v1.2 submissions.
    # Their answers are checkable against the upstream sources.
    # ------------------------------------------------------------------
    {
        "id": "EA13",
        "title": "REAL: operators the CUDA provider does not implement",
        "question": ("Which ai.onnx operators does ONNX Runtime implement on "
                     "CPU but NOT on CUDA, so a GPU graph would break or fall "
                     "back?"),
        "why_graph": ("A real coverage gap between two execution providers, "
                      "expressed as an anti-join over the same operator node. "
                      "Answer is verifiable against onnxruntime's "
                      "docs/OperatorKernels.md."),
        "cypher": """
MATCH (k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WHERE k.execution_provider = "CPUExecutionProvider" AND op.domain = "ai.onnx"
OPTIONAL MATCH (k2:Kernel)-[:IMPLEMENTS]->(op)
WHERE k2.execution_provider = "CUDAExecutionProvider"
WITH op.name AS operator, op.category AS category, count(k2) AS cuda_kernels
WHERE cuda_kernels = 0
RETURN operator, category, cuda_kernels
ORDER BY category
LIMIT 20
""",
    },
    {
        "id": "EA14",
        "title": "REAL: MLPerf Tiny throughput leaders",
        "question": ("On the real MLPerf Tiny v1.2 submissions, which board "
                     "posted the highest throughput for each benchmark task?"),
        "why_graph": ("Joins measured submissions to the board, the reference "
                      "model and the benchmark task in one linear path."),
        "cypher": """
MATCH (b:Board)<-[:ON_BOARD]-(d:Deployment)-[:MEASURES]->(m:Model)
      -[:SOLVES]->(t:BenchmarkTask)
WHERE d.provenance = "real"
WITH t.name AS task, b.name AS board, d.throughput_inf_s AS throughput_inf_s,
     d.accuracy AS accuracy
RETURN task, board, throughput_inf_s, accuracy
ORDER BY throughput_inf_s DESC
LIMIT 12
""",
    },
    {
        "id": "EA15",
        "title": "REAL: operators available on only one execution provider",
        "question": ("Which operators are registered on exactly one ONNX "
                     "Runtime execution provider -- i.e. using them pins you "
                     "to that backend?"),
        "why_graph": ("Portability risk as a degree count over the real kernel "
                      "registration graph."),
        "cypher": """
MATCH (k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WHERE k.provenance = "real"
WITH op.name AS operator, op.domain AS domain,
     count(DISTINCT k.execution_provider) AS providers
WHERE providers = 1
RETURN operator, domain, providers
ORDER BY operator
LIMIT 20
""",
    },
    {
        "id": "EA16",
        "title": "REAL vs SYNTHETIC: what is measured and what is generated",
        "question": ("How much of this graph is real public data versus the "
                     "generated fleet, per source?"),
        "why_graph": ("Provenance is a first-class property, so the split is "
                      "one aggregation -- not a README claim you have to "
                      "trust."),
        "cypher": """
MATCH (k:Kernel)
WITH k.provenance AS provenance, k.source AS source, count(k.id) AS kernels
RETURN provenance, source, kernels
ORDER BY kernels DESC
""",
    },
    {
        "id": "EA17",
        "title": "BLAST RADIUS: this sensor stops -- what stops with it?",
        "question": (f"Sensor `{EA17_SUBJECT}` fails. What depends on it, and how "
                     "much of that stops *only* because of it?"),
        "why_graph": (
            "Two numbers, because reachable is not the same as stopped. "
            "`affected` is everything downstream; `only_via_me` is the subset "
            "with no other live feeder -- the part that actually goes dark. A "
            "stage fed by three sensors is a firebreak, not a casualty, and a "
            "reachability query alone reports it as one. The `NEXT_STAGE` chain "
            "is of unknown length, so both need `*0..`."),
        # `only_via_me` is the second number because reachability is not the
        # answer: `etl/generate.py:434` samples every sensor's chain from one
        # shared 16-stage pool, so an unbounded walk reaches most of the fleet.
        # The ClinicalTask leg matches `o.modality = s.modality` for the same
        # reason -- a task's other sensors replace this one only if they supply
        # the same modality.
        #
        # Six constraints, each written up where it can be checked rather than
        # repeated here. `tests/test_blast_radius.py`'s module docstring is the
        # long form:
        #
        #   1. The `OPTIONAL MATCH` legs use note 1's trailing-rebind shape.
        #      Validated at `--scale 1.0`, 0 disagreements. Do not edit those
        #      patterns without re-running `pytest --full-scale`.
        #   2. Needs the `samyama>=1.7.1` floor #104 landed (note 10's shape,
        #      a per-leg second `WITH`, which 0.6.x rejects).
        #   3. `+1/+2/+4/+5` are schema-fixed hops, not a depth bound. The
        #      variable part is `size(r)`, which is what `*0..` is for.
        #   4. The ClinicalTask leg's `depth` is 1 by construction, not a
        #      measured hop count like the other legs': a task points *at* the
        #      sensor, so it is adjacent. `nearest` is therefore comparable
        #      within a kind and not across them.
        #   5. The sensor id is `EA17_SUBJECT`, written once above and
        #      interpolated into all ten places the query names it -- five
        #      legs, each naming it twice. Retarget with `retargeted_ea17`,
        #      never by hand. An unknown id gives an empty blast radius rather
        #      than an error, as `EA01` and `EA06` do -- measured by
        #      `test_an_unknown_sensor_gives_an_empty_blast_radius`, because
        #      aggregate-only legs could as easily have returned a row of
        #      zeros.
        #   6. No `ORDER BY` (note 3c) and no `WHERE` on `only_via_me`
        #      (note 11); `o.id IS NOT NULL` guards note 8b -- `<>` against a
        #      null property matches.
        #
        # **Embedded only.** The 1.7.0 server does not traverse the
        # variable-length walk and rejects `size(r)` on it (note 12), so
        # `run_benchmark` over HTTP records a per-query failure -- a standing
        # one, not a regression, and `docs/engine-notes.md` note 12 is what a
        # reader of that output is sent to. Constraint 2 means embedded 0.6.1
        # rejects it too, and `pyproject.toml`'s `samyama>=1.7.1` floor (#104)
        # is what excludes that build.
        "cypher": """
MATCH (s:Sensor)-[:FEEDS]->(:SignalStage)-[r:NEXT_STAGE*0..]->(x:SignalStage)
WHERE s.id = {subject}
OPTIONAL MATCH (o:Sensor)-[:FEEDS]->(:SignalStage)-[:NEXT_STAGE*0..]->(x)
WHERE o.id IS NOT NULL AND o.id <> {subject}
WITH x.id AS thing, min(size(r)) + 1 AS depth, count(DISTINCT o.id) AS others
WITH "SignalStage" AS kind, count(thing) AS affected,
     sum(CASE WHEN others = 0 THEN 1 ELSE 0 END) AS only_via_me, min(depth) AS nearest
RETURN kind, affected, only_via_me, nearest
UNION ALL
MATCH (s:Sensor)-[:FEEDS]->(:SignalStage)-[r:NEXT_STAGE*0..]->(:SignalStage)-[:PRECEDES]->(m:Model)
WHERE s.id = {subject}
OPTIONAL MATCH (o:Sensor)-[:FEEDS]->(:SignalStage)-[:NEXT_STAGE*0..]->(:SignalStage)-[:PRECEDES]->(m)
WHERE o.id IS NOT NULL AND o.id <> {subject}
WITH m.id AS thing, min(size(r)) + 2 AS depth, count(DISTINCT o.id) AS others
WITH "Model" AS kind, count(thing) AS affected,
     sum(CASE WHEN others = 0 THEN 1 ELSE 0 END) AS only_via_me, min(depth) AS nearest
RETURN kind, affected, only_via_me, nearest
UNION ALL
MATCH (s:Sensor)-[:FEEDS]->(:SignalStage)-[r:NEXT_STAGE*0..]->(:SignalStage)-[:PRECEDES]->(:Model)<-[:VARIANT_OF]-(:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)
WHERE s.id = {subject}
OPTIONAL MATCH (o:Sensor)-[:FEEDS]->(:SignalStage)-[:NEXT_STAGE*0..]->(:SignalStage)-[:PRECEDES]->(:Model)<-[:VARIANT_OF]-(:ModelVariant)<-[:OF_VARIANT]-(d)
WHERE o.id IS NOT NULL AND o.id <> {subject}
WITH d.id AS thing, min(size(r)) + 4 AS depth, count(DISTINCT o.id) AS others
WITH "Deployment" AS kind, count(thing) AS affected,
     sum(CASE WHEN others = 0 THEN 1 ELSE 0 END) AS only_via_me, min(depth) AS nearest
RETURN kind, affected, only_via_me, nearest
UNION ALL
MATCH (s:Sensor)-[:FEEDS]->(:SignalStage)-[r:NEXT_STAGE*0..]->(:SignalStage)-[:PRECEDES]->(:Model)<-[:VARIANT_OF]-(:ModelVariant)<-[:OF_VARIANT]-(:Deployment)-[:ON_BOARD]->(b:Board)
WHERE s.id = {subject}
OPTIONAL MATCH (o:Sensor)-[:FEEDS]->(:SignalStage)-[:NEXT_STAGE*0..]->(:SignalStage)-[:PRECEDES]->(:Model)<-[:VARIANT_OF]-(:ModelVariant)<-[:OF_VARIANT]-(:Deployment)-[:ON_BOARD]->(b)
WHERE o.id IS NOT NULL AND o.id <> {subject}
WITH b.id AS thing, min(size(r)) + 5 AS depth, count(DISTINCT o.id) AS others
WITH "Board" AS kind, count(thing) AS affected,
     sum(CASE WHEN others = 0 THEN 1 ELSE 0 END) AS only_via_me, min(depth) AS nearest
RETURN kind, affected, only_via_me, nearest
UNION ALL
MATCH (t:ClinicalTask)-[:REQUIRES_SENSOR]->(s:Sensor)
WHERE s.id = {subject}
OPTIONAL MATCH (t)-[:REQUIRES_SENSOR]->(o:Sensor)
WHERE o.id IS NOT NULL AND o.id <> {subject} AND o.modality = s.modality
WITH t.id AS thing, 1 AS depth, count(DISTINCT o.id) AS others
WITH "ClinicalTask" AS kind, count(thing) AS affected,
     sum(CASE WHEN others = 0 THEN 1 ELSE 0 END) AS only_via_me, min(depth) AS nearest
RETURN kind, affected, only_via_me, nearest
""".replace("{subject}", f'"{EA17_SUBJECT}"'),
    },
    {
        "id": "EA18",
        "title": "SILENT DEGRADATION: over the latency budget, and why",
        "question": ("Which deployments miss the latency budget of a clinical "
                     "task they serve, and which of their operators have no "
                     "kernel on the accelerator they run on?"),
        "why_graph": (
            "The alert nobody can raise today. Nothing about the sensor "
            "changes and nothing about the reading changes -- the device keeps "
            "reporting healthy -- but an operator with no kernel falls back to "
            "the CPU and the budget is gone. The measurement and the budget sit "
            "on different labels three hops apart (`Deployment` to "
            "`ModelVariant` to `Model` to `ClinicalTask`), and the cause sits "
            "on a fourth branch through `USES_ACCELERATOR`. A threshold system "
            "watching latency sees the symptom; only the graph names the "
            "operators responsible in the same row."),
        # `collect(CASE ... ELSE NULL END)` drops the nulls -- asserted on
        # every call by `tests/test_latency_budget.py::run_ea18`, not assumed --
        # so a deployment over budget with every kernel present comes back with
        # `[]` rather than being filtered out. That case is the point: over budget and
        # *not* because of fallback is a different alert, and dropping those
        # rows would report the cause as universal.
        #
        # `d.latency_ms > t.latency_budget_ms` compares a float property with an
        # int one. Engine note 4 is the int/float coercion weakness -- a literal
        # whose type does not match the column, or an int sentinel in `min()`;
        # this is property-to-property, and it was checked rather than assumed
        # -- 309.291 > 200 is true on 1.7.1, as are the `toFloat()` and literal
        # spellings.
        #
        # `sum(CASE ...)` is engine note 11's shape. That note read as two
        # builds disagreeing and turned out to be version skew -- 0.6.1 against
        # a 1.7.0 server (#56) -- so on the `>=1.7.1` floor it does not
        # reproduce at all. Nothing here filters on `fallback_ops`, it is only
        # ordered by, so the note's silently-dropped `WHERE` cannot apply. Do
        # not add `WHERE fallback_ops > 0` without reading that note.
        #
        # The second and third `WITH`s introduce new aliases -- engine note
        # 10's shape. Like note 11 that was version skew rather than a build
        # difference: it raised on `samyama` 0.6.1 and does not on 1.7.1:
        # `tests/test_latency_budget.py` runs this embedded and passes. So
        # `EA18` is deliberately **not** in `NOTE_10_QUERIES`: the mark would
        # XPASS on every run and excuse nothing, and on 0.6.1 this query was
        # never run at all. It depends on the `samyama>=1.7.1` floor that #104
        # landed, the same one `EA17` names -- #105 was the PR that wrote the
        # pin, and it reached `main` inside #104, so one number for one floor;
        # the note records this under "EA18 (#37)".
        #
        # `over_by_ms` is computed in its own `WITH`, not beside the
        # aggregates. Engine note 13b: an expression mixing a grouping key with
        # an aggregate in the same projection returns NULL, silently -- so
        # `d.latency_ms - min(budget_ms)` alongside them gave a null column and
        # an ORDER BY on nothing.
        #
        # **One accelerator per deployment is load-bearing here.** `a` binds
        # the deployment's accelerator, and an operator is called a fallback
        # when no kernel implements it *and* runs on that accelerator. Two
        # accelerators on one deployment would pool their kernels, and an
        # operator covered on one but not the other would stop counting as a
        # fallback -- the query under-reporting the very thing it looks for.
        # Measured rather than assumed: one `USES_ACCELERATOR` edge per
        # deployment across both layers at `--scale 1.0`, 1,451 deployments,
        # and `tests/test_generate.py::test_a_deployment_uses_exactly_one_accelerator`
        # fails (naming `EA18`) if a future fleet relaxes that.
        #
        # A deployment whose model solves several breached tasks does not get
        # an inflated `fallback_ops`, but not because the duplicate task paths
        # collapse -- they do not. The first `WITH` groups by the deployment
        # and operator *properties* (never the bare nodes, per note 9) while
        # `t` is still bound, so `count(k.id)` is multiplied by the number of
        # breached tasks: one operator with one kernel and two breached tasks
        # gives `kernels_here = 2`. What saves the count is that the second
        # `WITH` only asks whether it is **zero**, and a multiple of zero is
        # zero -- so the operator is either covered or it is not, whatever the
        # multiplier. `operators` is a `collect` over the same grouping, so it
        # is not duplicated either. The generator emits one `SOLVES` per model
        # today, so none of this can arise yet, which is why it is written down
        # rather than tested.
        #
        # `min(budget_ms)` across several breached tasks takes the **tightest**
        # budget, which maximises `over_by_ms`. That is the intended reading --
        # the deployment has to satisfy every task it serves, so the strictest
        # one is the binding constraint -- and it is stated here because the
        # alternative (the budget of the worst-served task) is just as
        # plausible to a reader and would give a different number.
        #
        # Ordered by `over_by_ms`, not by `fallback_ops`. Sorting on the
        # fallback count puts the zero-fallback rows last, so on a fleet with
        # more than 20 breaches `LIMIT 20` would truncate exactly the rows that
        # are over budget for some *other* reason -- the more surprising alert,
        # and the one this query is careful to keep. Overshoot is also the
        # ordering an operator wants: worst breach first. One key, because
        # engine note 3b drops every key after the first.
        #
        # The trailing re-bind of `a` is note 1's shape and is validated at
        # `--scale 1.0` by
        # `tests/test_latency_budget.py::test_ea18_matches_ground_truth_at_full
        # _scale_with_an_injected_breach`, which raises one deployment's latency
        # over its task's budget so the query has rows at that cardinality and
        # compares them against Python. Removing the join fails it.
        #
        # On the shipped fleet this returns **no rows**: measured at
        # `--scale 1.0`, all 1,440 (deployment, task) pairs are inside budget,
        # the worst at 54.5% of it. That is the honest answer, not a broken
        # query -- `tests/test_latency_budget.py` proves it fires where a
        # breach exists, and pins the zero.
        "cypher": """
MATCH (t:ClinicalTask)<-[:SOLVES]-(m:Model)<-[:VARIANT_OF]-(:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)-[:USES_ACCELERATOR]->(a:Accelerator)
WHERE d.latency_ms > t.latency_budget_ms
MATCH (m)-[:USES_OPERATOR]->(op:Operator)
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a)
WITH d.id AS deployment, d.latency_ms AS latency_ms, op.id AS op_id,
     op.name AS op_name, min(t.latency_budget_ms) AS budget_ms, count(k.id) AS kernels_here
WITH deployment, latency_ms, min(budget_ms) AS budget_ms,
     sum(CASE WHEN kernels_here = 0 THEN 1 ELSE 0 END) AS fallback_ops,
     collect(CASE WHEN kernels_here = 0 THEN op_name ELSE NULL END) AS operators
WITH deployment, latency_ms, budget_ms, latency_ms - budget_ms AS over_by_ms,
     fallback_ops, operators
RETURN deployment, latency_ms, budget_ms, over_by_ms, fallback_ops, operators
ORDER BY over_by_ms DESC
LIMIT 20
""",
    },
    {
        "id": "EA19",
        "title": "COMPLIANCE: this sensor fails -- which certifications does that touch?",
        "question": (f"Sensor `{EA17_SUBJECT}` fails. Which certifications are "
                     "implicated, through the clinical tasks that require it?"),
        "why_graph": (
            "The difference between an ops ticket and a reportable event. "
            "`Certification` and `Sensor` share no edge: they meet only through "
            "`ClinicalTask`, which `REQUIRES_SENSOR` on one side and is "
            "`GOVERNED_BY` on the other. A table of sensors cannot answer it "
            "without knowing to join through tasks, and for a medical or "
            "industrial device that join decides who has to be told, and how "
            "fast."),
        # Deliberately one linear pattern, no `OPTIONAL MATCH` and no
        # re-binding. `EA17` needs note 1's trailing-rebind shape because an
        # anti-join has no other spelling on this build; this question does
        # not, so it does not carry the risk. That is worth stating rather
        # than leaving the next reader to wonder why two neighbouring alerting
        # queries look so different.
        #
        # Grouped by the three `Certification` properties rather than by the
        # node: note 9 says `count(DISTINCT cert)` over a multi-variable MATCH
        # returns rows of 1, so the grouping key has to be properties.
        #
        # The `DISTINCT` inside `count(DISTINCT t.id)` is **not** load-bearing
        # on today's data and was checked rather than assumed: the pattern
        # yields one row per (task, certification) pair, so `count(t.id)`
        # returns the same number, and removing it fails nothing. It stays as
        # insurance against a second `REQUIRES_SENSOR` edge between the same
        # pair, which the loader does not currently produce -- said plainly,
        # because a comment claiming a guard is load-bearing when it is not is
        # how the next person leaves a real one out.
        #
        # An unknown sensor id yields no rows rather than an error, the same
        # shape `EA01` and `EA17` have.
        #
        # `ORDER BY tasks_affected DESC LIMIT 20` has no tiebreaker, and note
        # 3b forbids a second key. It does not matter here: the generator emits
        # six certifications at every scale, so a sensor implicates at most six
        # rows and `LIMIT 20` never truncates -- ties change the order, never
        # the set. `tests/test_certification_alerts.py` compares sorted rows and
        # pins the six, so a catalog that outgrows the limit fails there first.
        #
        # The subject is `EA17_SUBJECT`, the same constant `EA17` uses and
        # interpolated the same way -- these two queries are asked about the
        # same failing sensor in the same breath ("what stops" then "what does
        # that implicate"), so two literals that could drift apart would make
        # the pair answer about different sensors while reading as one story.
        # Neither has a parameterised tool in `mcp_server/server.py`, and that
        # is a gap rather than an oversight to hide: the MCP surface covers
        # none of the alerting queries yet, and exposing them is #49's scope.
        "cypher": """
MATCH (s:Sensor)<-[:REQUIRES_SENSOR]-(t:ClinicalTask)-[:GOVERNED_BY]->(cert:Certification)
WHERE s.id = {subject}
WITH cert.name AS certification, cert.body AS body, cert.class AS cert_class,
     count(DISTINCT t.id) AS tasks_affected
RETURN certification, body, cert_class, tasks_affected
ORDER BY tasks_affected DESC
LIMIT 20
""".replace("{subject}", f'"{EA17_SUBJECT}"'),
    },
]

BY_ID = {q["id"]: q for q in QUERIES}
