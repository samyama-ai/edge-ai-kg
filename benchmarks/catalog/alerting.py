"""`EA17`-`EA18`: what has broken, and what is breaking quietly.

The blast radius of a failing sensor, and the deployments that miss a
clinical task's latency budget while nothing is down at all.

Only `EA17` is embedded-only: it walks `NEXT_STAGE*0..` unbounded and calls
`size(r)` on the result, which the 1.7.0 server rejects
(`docs/engine-notes.md` note 12). `EA18`'s walk is bounded, so it runs on
both builds, and `tests/test_correctness.py::EMBEDDED_ONLY` -- derived from
the Cypher rather than from prose -- holds `EA17` and `EA21`, not `EA18`.
"""
from __future__ import annotations

from benchmarks.catalog.subjects import EA17_SUBJECT

ALERTING: list[dict] = [
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
        # answer: `etl/generate.py`'s sensor-pipeline section (`chain =
        # rng.sample(stages, ...)`) samples every sensor's chain from one
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
        #      Validated in two tiers, because one of them is opt-in: the
        #      fixture-scale ground-truth comparison in
        #      `tests/test_blast_radius.py` runs on every `pytest` and would
        #      catch a leg that counts the wrong set, while
        #      `test_ea17_matches_ground_truth_at_full_scale` -- gated on
        #      `--full-scale`, minutes to load -- is what settles *cardinality*,
        #      which note 1 says a small graph cannot. 0 disagreements at
        #      `--scale 1.0`. Do not edit those patterns without re-running the
        #      full-scale one.
        #   2. Needs the `samyama>=1.7.1` floor #104 landed (note 10's shape,
        #      a per-leg second `WITH`, which 0.6.x rejects).
        #   3. `+1/+2/+4/+5` are schema-fixed hops, not a depth bound. The
        #      variable part is `size(r)`, which is what `*0..` is for.
        #      **The walk has no cost bound, by design and not by oversight.**
        #      A depth cap would cap the answer -- the chain's length is what
        #      is being asked -- so what exists instead is a measurement:
        #      `docs/volume.md` puts `EA17` at 95 ms at `--scale 1.0` and
        #      431 ms at 2.0, x4.5 per doubling, the steepest in the catalog.
        #      Relationship-uniqueness is what stops it looping on the cyclic
        #      `NEXT_STAGE` graph, pinned by
        #      `test_the_unbounded_walk_terminates_on_a_cyclic_chain`.
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
        # `EA18` carries no note-10 `xfail`: there is no such mark left in the
        # suite to join -- #105 removed the last of them when it raised the
        # floor -- and one here would XPASS on every run and excuse nothing.
        # It depends on the `samyama>=1.7.1` floor #105 wrote and #104 landed,
        # the same one `EA17` names; the note records this under "EA18 (#37)".
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
        # Measured rather than assumed: across both layers at `--scale 1.0`,
        # 1,513 deployments hold 1,451 `USES_ACCELERATOR` edges, never two --
        # the other 62 are real-layer MLPerf rows with none, which this
        # query's opening `MATCH` does not bind at all.
        # `tests/test_generate.py::test_no_deployment_uses_more_than_one_accelerator`
        # fails (naming `EA18`) if a future fleet relaxes that.
        #
        # The operator leg is an inner `MATCH`, so a breached deployment whose
        # model has no operators is dropped rather than reported with zero
        # fallbacks. Four models do lack operators -- the real-layer MLPerf
        # ones -- and none is reachable from a deployment that has an
        # accelerator, so the hole is closed by the *accelerator* hop rather
        # than by the operators. That is the invariant
        # `tests/test_generate.py::test_every_model_ea18_can_reach_has_operators`
        # pins, measured at seeds 20260814/1234/999 and scales 0.25 and 1.0.
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
        # `tests/test_latency_budget.py`'s
        # `test_ea18_matches_ground_truth_at_full_scale_with_an_injected_breach`,
        # which raises one deployment's latency
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
]
