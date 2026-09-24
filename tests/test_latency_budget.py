"""`EA18`, the silent-degradation alert (#37).

The hero question and the alerting theme in one row: an operator with no kernel
falls back to the CPU, the device keeps reporting healthy, and the clinical
task's latency budget is gone. Nothing about the sensor changes, so nothing a
threshold system watches changes either.

**On the shipped fleet the answer is zero, and that is the finding.** Measured
at `--scale 1.0`: 1,440 (deployment, task) pairs, none over budget, the worst at
54.5% of it. So the query cannot be validated against the generated graph -- an
empty result is consistent with a correct query and with a query that matches
nothing at all, and the two are indistinguishable from the outside. That is the
same trap `EA01`'s zero-row case fell into (#66), so this module does both
halves:

- `test_it_fires_on_a_graph_where_a_deployment_is_over_budget` builds the breach
  the fleet does not contain, on three deployments that differ in one way each.
- `test_the_shipped_fleet_has_no_breach_and_that_is_why_ea18_is_empty` pins the
  zero *and* its cause, so "empty" cannot quietly become "empty because the
  pattern stopped matching". It asserts the fleet has the pairs the query needs
  and that every one is inside budget.

The three fixture deployments are the three cases that must not be confused:

| deployment | latency vs budget | kernels on its accelerator | expected |
|---|---|---|---|
| `dep:over-fallback` | over | two operators uncovered | row, 2 operators named |
| `dep:over-clean` | over | all covered | row, **empty** operator list |
| `dep:within` | inside | two operators uncovered | no row |

`dep:over-clean` is the one a narrower query loses. Filtering the operator leg
to uncovered operators drops the deployment entirely, which reports "over budget
implies fallback" -- a claim this graph does not support and the fleet would
never contradict, because the fleet has no breaches at all.
"""
from __future__ import annotations

import pytest

from benchmarks.queries import BY_ID
from etl.helpers import create_edges, create_nodes

GRAPH = "default"
BUDGET_MS = 100


def breach_fixture(client):
    """Three deployments against one 100 ms task, differing in one way each."""
    create_nodes(client, GRAPH, "ClinicalTask",
                 [{"id": "task:tight", "name": "tight", "category": "monitoring",
                   "latency_budget_ms": BUDGET_MS, "min_sensitivity": 0.9}])
    create_nodes(client, GRAPH, "Model",
                 [{"id": f"model:{n}", "name": n, "family": "cnn"}
                  for n in ("slow", "clean", "ok")])
    create_nodes(client, GRAPH, "ModelVariant",
                 [{"id": f"var:{n}", "precision": "int8"}
                  for n in ("slow", "clean", "ok")])
    # Two accelerators, and the one the deployments do *not* use carries
    # kernels for both uncovered operators. Engine note 1 warns that a
    # trailing re-bind can lose its join; with a single accelerator in the
    # graph a lost join is undetectable by construction, because "kernels on
    # this accelerator" and "kernels anywhere" are the same set. Here they are
    # not: if `(k)-[:RUNS_ON]->(a)` stops joining, `fallback_ops` for
    # `dep:over-fallback` goes from 2 to 0 and the assertion below fails.
    create_nodes(client, GRAPH, "Accelerator",
                 [{"id": "accel:npu", "name": "npu", "kind": "npu",
                   "is_cpu_fallback": 0},
                  {"id": "accel:gpu", "name": "gpu", "kind": "gpu",
                   "is_cpu_fallback": 0}])
    create_nodes(client, GRAPH, "Operator",
                 [{"id": "op:a", "name": "Conv", "category": "nn", "since_version": 1},
                  {"id": "op:b", "name": "GRU", "category": "rnn", "since_version": 7},
                  {"id": "op:c", "name": "Relu", "category": "nn", "since_version": 6}])
    create_nodes(client, GRAPH, "Kernel",
                 [{"id": "k:relu", "name": "Relu-npu"},
                  {"id": "k:conv-gpu", "name": "Conv-gpu"},
                  {"id": "k:gru-gpu", "name": "GRU-gpu"}])
    create_nodes(client, GRAPH, "Deployment",
                 [{"id": "dep:over-fallback", "latency_ms": 250.0, "fits": 1},
                  {"id": "dep:over-clean", "latency_ms": 300.0, "fits": 1},
                  {"id": "dep:within", "latency_ms": 50.0, "fits": 1}])

    edges = []
    # `dep:within` gets `var:ok`, its own variant. It shared `var:slow` before,
    # which made that variant `VARIANT_OF` two models -- so `dep:over-fallback`
    # reached both, both solving the same task with the same operators. The
    # assertions still held, because grouping by `op` in the intermediate `WITH`
    # collapses the duplicate paths, but that made the collapse load-bearing
    # while the table above describes one model per deployment.
    for model, variant, deployment in (("slow", "slow", "dep:over-fallback"),
                                       ("clean", "clean", "dep:over-clean"),
                                       ("ok", "ok", "dep:within")):
        edges += [
            ("ModelVariant", f"var:{variant}", "VARIANT_OF", "Model", f"model:{model}", None),
            ("Deployment", deployment, "OF_VARIANT", "ModelVariant", f"var:{variant}", None),
            ("Deployment", deployment, "USES_ACCELERATOR", "Accelerator", "accel:npu", None),
            ("Model", f"model:{model}", "SOLVES", "ClinicalTask", "task:tight", None),
        ]
    # `slow` and `ok` use two operators the npu has no kernel for; `clean` uses
    # only the covered one. `ok` is inside budget, so its uncovered operators
    # must not put it in the answer -- fallback alone is not the alert.
    edges += [
        ("Model", "model:slow", "USES_OPERATOR", "Operator", "op:a", None),
        ("Model", "model:slow", "USES_OPERATOR", "Operator", "op:b", None),
        ("Model", "model:ok", "USES_OPERATOR", "Operator", "op:a", None),
        ("Model", "model:ok", "USES_OPERATOR", "Operator", "op:b", None),
        ("Model", "model:clean", "USES_OPERATOR", "Operator", "op:c", None),
        ("Kernel", "k:relu", "IMPLEMENTS", "Operator", "op:c", None),
        ("Kernel", "k:relu", "RUNS_ON", "Accelerator", "accel:npu", None),
        # On the gpu, which nothing here deploys to. These exist so that
        # "uncovered on this accelerator" and "uncovered anywhere" differ.
        ("Kernel", "k:conv-gpu", "IMPLEMENTS", "Operator", "op:a", None),
        ("Kernel", "k:conv-gpu", "RUNS_ON", "Accelerator", "accel:gpu", None),
        ("Kernel", "k:gru-gpu", "IMPLEMENTS", "Operator", "op:b", None),
        ("Kernel", "k:gru-gpu", "RUNS_ON", "Accelerator", "accel:gpu", None),
    ]
    create_edges(client, GRAPH, edges)
    return client


def run_ea18(client):
    """`{deployment: (latency_ms, budget_ms, over_by_ms, fallback_ops, ops)}`.

    Operator names are sorted here: `collect` order is not specified and was
    observed to differ between two spellings of the same query on one build, so
    asserting on the raw order would pin an accident.

    `EA18` relies on `collect(CASE ... ELSE NULL END)` dropping the nulls. That
    is checked here rather than assumed: a null left in the list would make
    `sorted` raise `TypeError` on a mixed list, or turn `[]` into `[None]`,
    and neither failure would name the cause.
    """
    records = client.query(BY_ID["EA18"]["cypher"], GRAPH).records
    kept = [row[0] for row in records if None in row[5]]
    assert not kept, (
        f"`collect` kept a NULL for {kept}. The query's empty-list case depends "
        f"on nulls being dropped; filter them in the Cypher if the engine no "
        f"longer does.")
    return {row[0]: (row[1], row[2], round(row[3], 9), row[4], sorted(row[5]))
            for row in records}


@pytest.fixture(scope="module")
def operators():
    """The ONNX catalogue, skipping in **setup** so `--no-skips` sees it.

    A body skip is one `--no-skips` cannot convert (`test_environment_skips`).
    """
    from etl import onnx_catalog as oc

    try:
        return oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


def test_it_fires_on_a_graph_where_a_deployment_is_over_budget(engine_factory):
    """The case the shipped fleet does not contain, built to have exactly one."""
    got = run_ea18(breach_fixture(engine_factory()))

    assert got.get("dep:over-fallback") == (250.0, BUDGET_MS, 150.0, 2, ["Conv", "GRU"]), (
        f"a deployment 250 ms against a 100 ms budget, with two operators its "
        f"accelerator has no kernel for, must be reported with both named. "
        f"Got {got.get('dep:over-fallback')}"
    )
    assert "dep:within" not in got, (
        f"`dep:within` is inside its budget. It has the same two uncovered "
        f"operators as `dep:over-fallback`, so reporting it would mean the "
        f"query is answering 'has CPU fallback' rather than 'is over budget'. "
        f"Got {got.get('dep:within')}"
    )


def test_over_budget_without_fallback_is_still_reported(engine_factory):
    """The row a narrower query loses, and the wrong claim losing it would make.

    `dep:over-clean` is over budget with every operator covered. Filtering the
    operator leg to uncovered operators drops it, and the answer then reads as
    "everything over budget is over budget because of fallback" -- which this
    graph does not show and the generated fleet could never contradict, having
    no breach at all. An empty operator list is a different alert, not an
    absent one.
    """
    got = run_ea18(breach_fixture(engine_factory()))

    assert got.get("dep:over-clean") == (300.0, BUDGET_MS, 200.0, 0, []), (
        f"a deployment over budget with no uncovered operators must still be "
        f"reported, with an empty operator list. Got "
        f"{got.get('dep:over-clean')}"
    )


def test_the_shipped_fleet_has_no_breach_and_that_is_why_ea18_is_empty(operators):
    """Pin the zero *and* its cause, so the two ways of being empty differ.

    A query that matches nothing and a query whose answer is genuinely none
    return the same thing. This asserts the fleet contains the pairs `EA18`
    traverses -- so the pattern is exercised -- and that every one of them is
    inside budget, which is the reason the catalog run shows no rows.

    If the generator ever produces a breach this fails, and the right response
    is to update the count here and the sentence in `benchmarks/queries.py`,
    not to relax the check.
    """
    from etl import generate as gen
    from etl import real_layer

    fleet = gen.generate(seed=20260814, scale=1.0, operators=operators)
    real_layer.build_real(fleet, operators)

    out = {}
    for _sl, src, rel, _tl, tgt, _p in fleet.edges:
        out.setdefault((rel, src), []).append(tgt)
    budget = {r["id"]: r["latency_budget_ms"] for r in fleet.nodes["ClinicalTask"]}

    pairs, over, missing = 0, [], []
    # `EA18` also needs `USES_ACCELERATOR` and `USES_OPERATOR`; this walk used
    # only OF_VARIANT/VARIANT_OF/SOLVES. If the generator stopped emitting
    # either of the other two the query would return zero for a reason this
    # pin explicitly claims to exclude, and the pin would still pass.
    no_accelerator, no_operators = [], []
    for deployment in fleet.nodes["Deployment"]:
        latency = deployment.get("latency_ms")
        if latency is None:              # the real MLPerf rows carry throughput
            continue
        models = [m for v in out.get(("OF_VARIANT", deployment["id"]), ())
                  for m in out.get(("VARIANT_OF", v), ())]
        if models and not out.get(("USES_ACCELERATOR", deployment["id"])):
            no_accelerator.append(deployment["id"])
        for model in models:
            if not out.get(("USES_OPERATOR", model)):
                no_operators.append(model)
            for task in out.get(("SOLVES", model), ()):
                if task not in budget:
                    # Diagnostic rather than a bare KeyError: a `SOLVES` target
                    # that is not in `fleet.nodes["ClinicalTask"]` means the
                    # generator now points that edge somewhere else, and the
                    # traceback would say nothing about which edge or why.
                    missing.append(f"{model} SOLVES {task}, which is not a "
                                   f"ClinicalTask in this fleet")
                    continue
                pairs += 1
                if latency > budget[task]:
                    over.append(f"{deployment['id']} {latency}ms > {budget[task]}ms")

    assert not no_accelerator, (
        f"{len(no_accelerator)} deployments have a model but no "
        f"`USES_ACCELERATOR` edge ({no_accelerator[:3]}). `EA18` traverses it, "
        f"so it would return zero for that reason rather than for the reason "
        f"this test claims -- and the claim would still read as verified."
    )
    assert not no_operators, (
        f"{len(set(no_operators))} models on a deployment have no "
        f"`USES_OPERATOR` edge ({sorted(set(no_operators))[:3]}). `EA18`'s "
        f"second `MATCH` needs it; without it every breach row is dropped "
        f"before the operator count is reached."
    )
    assert not missing, (
        f"`SOLVES` edges pointing at something that is not a ClinicalTask: "
        f"{missing[:3]}. This test reads the budget off the target, so the "
        f"spine it walks is not the one it thinks."
    )
    # 1,440 at the moment, and the floor goes with the `scale=1.0` above: this
    # test hardcodes that scale, so the number is a property of one fleet, not
    # a general invariant. Change both together or neither.
    assert pairs >= 1000, (
        f"only {pairs} (deployment, task) pairs at scale 1.0 -- `EA18`'s "
        f"pattern is barely exercised, so its empty answer would say nothing. "
        f"The spine this query walks has changed shape."
    )
    assert not over, (
        f"the generated fleet now contains {len(over)} budget breaches: "
        f"{over[:3]}. `EA18` is documented as returning no rows on this fleet "
        f"and that is no longer true -- update the comment in "
        f"benchmarks/queries.py and the README."
    )


def test_the_accelerator_join_is_enforced(engine_factory):
    """Engine note 1's risk, made detectable (#37 review).

    `EA18`'s kernel leg re-binds `a` -- bound by the opening `MATCH` -- in the
    trailing position of a comma-separated `OPTIONAL MATCH`. That is the shape
    note 1 says can silently lose its join. `EA01` and `EA11` look identical but
    are not: they introduce `(a:Accelerator)` *inside* the `OPTIONAL MATCH`, so
    there is no outer variable to lose.

    If the join is lost, `count(k)` counts kernels on **any** accelerator,
    `kernels_here` is never 0, and every row comes back with `fallback_ops = 0`
    and an empty operator list -- indistinguishable from the honest
    `dep:over-clean` answer this module exists to preserve.

    The fixture makes that detectable by having a second accelerator carry
    kernels for both operators the deployment's accelerator lacks. This asserts
    the distinction directly rather than relying on it as a side effect.

    Note 1 says only a full-size graph settles this shape, and that is done
    separately by
    `test_ea18_matches_ground_truth_at_full_scale_with_an_injected_breach`,
    which injects one breach into a scale-1.0 fleet so the query has rows to
    return there. This fixture is the fast, deterministic version of the same
    check; the full-scale one is the one note 1 asks for.
    """
    client = breach_fixture(engine_factory())

    anywhere = client.query(
        'MATCH (k:Kernel)-[:IMPLEMENTS]->(op:Operator) WHERE op.id = "op:a" '
        'WITH count(k) AS n RETURN n', GRAPH).records[0][0]
    assert anywhere == 1, (
        f"`op:a` must have a kernel *somewhere* for this test to mean anything "
        f"-- otherwise a lost join and an enforced one give the same answer. "
        f"Got {anywhere}"
    )

    got = run_ea18(client)
    assert got["dep:over-fallback"][3] == 2, (
        f"`Conv` and `GRU` have kernels on `accel:gpu` but not on `accel:npu`, "
        f"which is what `dep:over-fallback` runs on. `fallback_ops` must be 2. "
        f"0 here means `(k)-[:RUNS_ON]->(a)` stopped joining and the query is "
        f"counting kernels on any accelerator -- engine note 1. Got "
        f"{got['dep:over-fallback']}"
    )


def test_note_13_would_break_the_obvious_rewrite(engine_factory):
    """Why `EA18` re-binds `a` instead of carrying `a.id` through a `WITH`.

    The rewrite note 1 would prefer -- introduce the accelerator inside the
    `OPTIONAL MATCH` and filter it against an id carried forward -- is not
    available on this build. Engine note 13: a `WHERE` on an `OPTIONAL MATCH`
    that mentions a `WITH`-introduced alias drops the unmatched rows, turning
    the anti-join into an inner join, with no error.

    Pinned here rather than only in the note, because the next person to read
    note 1 against `EA18` will propose exactly this rewrite. If a future engine
    fixes note 13 this test fails, which is the signal to reconsider the shape.

    Exact, against a literal control differing only in the alias, so nothing
    else satisfies it: the rewrite ran (not note 10 raising) and lost exactly
    the row whose kernel leg had no match.
    """
    client = breach_fixture(engine_factory())
    rewritten = """
MATCH (t:ClinicalTask)<-[:SOLVES]-(m:Model)<-[:VARIANT_OF]-(:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)-[:USES_ACCELERATOR]->(a:Accelerator)
WHERE d.latency_ms > t.latency_budget_ms
WITH d, m, a.id AS accel_id, min(t.latency_budget_ms) AS budget_ms
MATCH (m)-[:USES_OPERATOR]->(op:Operator)
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(ka:Accelerator)
WHERE ka.id = accel_id
WITH d, budget_ms, op, count(k) AS kernels_here
WITH d.id AS deployment, sum(CASE WHEN kernels_here = 0 THEN 1 ELSE 0 END) AS fallback_ops
RETURN deployment, fallback_ops
"""
    control = rewritten.replace("WHERE ka.id = accel_id",
                                'WHERE ka.id = "accel:npu"')
    assert control != rewritten
    controlled = {r[0]: r[1] for r in client.query(control, GRAPH).records}
    assert controlled == {"dep:over-fallback": 2, "dep:over-clean": 0}, (
        f"the literal control should be correct on this fixture; got "
        f"{controlled}. Without it the assertion below proves nothing."
    )
    rows = {r[0]: r[1] for r in client.query(rewritten, GRAPH).records}
    assert rows == {"dep:over-clean": 0}, (
        f"expected note 13's signature -- only the row whose kernel leg "
        f"matched survives -- and got {rows}. If `dep:over-fallback` is back "
        f"with 2, note 13 is fixed: re-read it with note 1, and that rewrite "
        f"becomes the better shape for EA18."
    )
    assert run_ea18(client)["dep:over-fallback"][3] == 2, (
        "the shipped spelling must still be right; this test is only "
        "meaningful as a contrast with it"
    )


def test_ea18_matches_ground_truth_at_full_scale_with_an_injected_breach(
        request, engine_factory, operators):
    """Note 1's shape, validated at real cardinality (#37 review).

    `EA18`'s kernel leg re-binds `a` in the trailing position of a
    comma-separated `OPTIONAL MATCH`, and `CLAUDE.md` says to validate any
    workaround for that on a full-size graph "since at least one bug only
    appears once cardinalities are real". The fixture above cannot do it: three
    deployments, three operators, two accelerators.

    The shipped fleet has no breach, so `EA18` returns nothing there -- but that
    is a property of the latency and budget *values*, not of the spine, which
    all 1,440 pairs traverse. So one breach is injected into the `Fleet` before
    loading: a single deployment's `latency_ms` is raised above the budget of a
    task its model serves. Everything else is the real graph -- 22,582 kernels,
    real operator counts, real accelerators -- which is the condition under
    which note 1 is documented to fail.

    If the join is lost, `count(k)` counts kernels on any of the fleet's
    accelerators rather than on this deployment's, `fallback_ops` collapses to
    0, and the row comes back looking like the honest `dep:over-clean` case.
    Ground truth is recomputed in Python from the same `Fleet`, so that is
    caught rather than assumed.

    Opt-in via `--full-scale`: the load costs about three minutes.
    """
    if not request.config.getoption("--full-scale"):
        pytest.skip("needs --full-scale (about three minutes to build the graph)")

    from etl import generate as gen
    from etl import real_layer
    from etl.loader import NODE_LABELS

    fleet = gen.generate(seed=20260814, scale=1.0, operators=operators)
    real_layer.build_real(fleet, operators)

    out = {}
    for _sl, src, rel, _tl, tgt, _p in fleet.edges:
        out.setdefault((rel, src), []).append(tgt)
    budget = {r["id"]: r["latency_budget_ms"] for r in fleet.nodes["ClinicalTask"]}
    opname = {r["id"]: r["name"] for r in fleet.nodes["Operator"]}
    covered = {}                       # operator id -> accelerators with a kernel
    for kernel in fleet.nodes["Kernel"]:
        for operator in out.get(("IMPLEMENTS", kernel["id"]), ()):
            covered.setdefault(operator, set()).update(
                out.get(("RUNS_ON", kernel["id"]), ()))

    def models_of(deployment_id):
        return [m for v in out.get(("OF_VARIANT", deployment_id), ())
                for m in out.get(("VARIANT_OF", v), ())]

    # The one edit: pick a deployment whose model serves a task, and push it
    # over that task's budget. Chosen by id so the same deployment is picked on
    # every run -- `min` over a dict's insertion order would not be.
    candidates = sorted(
        d["id"] for d in fleet.nodes["Deployment"]
        if d.get("latency_ms") is not None
        and any(out.get(("SOLVES", m)) for m in models_of(d["id"])))
    assert candidates, "no deployment serves a clinical task; the spine has moved"
    victim_id = candidates[0]
    victim = next(d for d in fleet.nodes["Deployment"] if d["id"] == victim_id)
    tightest = min(budget[t] for m in models_of(victim_id)
                   for t in out.get(("SOLVES", m), ()))
    victim["latency_ms"] = float(tightest) + 50.0

    client = engine_factory()
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)

    # Ground truth for the whole fleet, from the Fleet rather than from the query.
    expected = {}
    for deployment in fleet.nodes["Deployment"]:
        latency = deployment.get("latency_ms")
        if latency is None:
            continue
        accelerators = out.get(("USES_ACCELERATOR", deployment["id"]), ())
        # Only models that breach a task *and* have a `USES_OPERATOR` edge:
        # `EA18`'s second `MATCH` drops a row whose model has no operators, so
        # a walk that ignored that precondition would expect rows the query
        # correctly never returns -- the same condition the zero-pin above
        # asserts on the unmodified fleet.
        models = [m for m in models_of(deployment["id"])
                  if out.get(("USES_OPERATOR", m))
                  and any(latency > budget[t] for t in out.get(("SOLVES", m), ()))]
        breached = [budget[t] for m in models
                    for t in out.get(("SOLVES", m), ()) if latency > budget[t]]
        if not breached or not accelerators:
            continue
        assert len(accelerators) == 1, (
            f"{deployment['id']} has {len(accelerators)} `USES_ACCELERATOR` "
            f"edges. Both this walk and `EA18` assume one: the query joins "
            f"kernels to `a` and would count each operator once per "
            f"accelerator, and this ground truth reads `accelerators[0]`. "
            f"Whichever is right, they would no longer agree by accident."
        )
        accelerator = accelerators[0]
        uncovered = {o for m in models for o in out.get(("USES_OPERATOR", m), ())
                     if accelerator not in covered.get(o, ())}
        tight = min(breached)
        # `over_by_ms` rounded on both sides. The engine computes it as
        # `latency_ms - budget_ms` in Cypher and this computes it in Python;
        # they agree today, but pinning an exact float across two
        # implementations is a test that fails for a reason no one will enjoy
        # reading. Nine decimals is far finer than anything this asserts.
        expected[deployment["id"]] = (latency, tight, round(latency - tight, 9),
                                      len(uncovered),
                                      sorted(opname[o] for o in uncovered))

    assert len(expected) == 1 and victim_id in expected, (
        f"the injection should create exactly one breach; Python finds "
        f"{sorted(expected)}"
    )
    assert expected[victim_id][3] > 0, (
        f"{victim_id}'s model has no operator uncovered on its accelerator, so "
        f"a lost join and an enforced one would give the same answer. Pick a "
        f"different victim rather than deleting this check."
    )

    got = run_ea18(client)
    assert got == expected, (
        f"EA18 disagrees with the Fleet at full cardinality, which is where the "
        f"trailing-rebind join fails if it fails.\n  EA18:   {got}\n  Python: "
        f"{expected}"
    )
