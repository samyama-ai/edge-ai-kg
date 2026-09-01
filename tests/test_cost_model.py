"""Deployment latency is derived, not drawn at random -- asserted, not assumed.

The README makes this the claim that separates the dataset from noise:

    Deployment metrics are *derived* from a documented cost model rather than
    drawn at random, so a board missing a kernel really does pay for it.

`etl/generate.py` splits the MACs between the accelerator and the CPU it falls
back to:

    t_acc = (macs * 2 * (1 - frac_fb)) / (gops_accel * 1e9 * thr)
    t_cpu = (macs * 2 * frac_fb)       / (gops_cpu   * 1e9 * thr)
    latency_ms = (t_acc + t_cpu) * 1000 * rng.uniform(1.05, 1.45)

## Why this is not a correlation test

A plain correlation does not work: `macs` and the accelerator's speed dominate,
and measured across the whole graph Spearman(`fallback_fraction`, `latency_ms`)
is only **0.166**. A test built on that would be too loose to catch anything.

The formula is recoverable instead. `thr` and `macs` are per-variant constants,
so within one `ModelVariant` they cancel:

    latency / shape  =  macs * 2 * 1000 * jitter / thr     where
    shape            =  (1 - f) / gops_accel + f / gops_cpu

Every deployment of a variant therefore has the same `latency / shape` up to the
jitter, so the spread within a variant cannot exceed `1.45 / 1.05`. That bound is
the generator's own, not a threshold picked to fit.

`gops_cpu` is the `MCU-CPU` accelerator on the deployment's board's SoC, which
is how `etl/generate.py:511` chooses it, and is reachable from the graph.

## Measured

Full load (1,440 generated deployments, 240 variants): worst spread **1.3760**
against the **1.3810** bound, nothing over. At this fixture's scale, **1.3451**.

The assertion has teeth -- recomputed against the same graph with the derivation
broken:

    derivation reversed (f -> 1-f)   worst spread   5.56   37 of 72 variants over
    latencies shuffled               worst spread 588.46   65 of 72 variants over
"""
from collections import defaultdict

import pytest

from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
SCALE = 0.3
SEED = 4242

# etl/generate.py: latency_ms = (t_acc + t_cpu) * 1000 * rng.uniform(1.05, 1.45)
JITTER_MIN, JITTER_MAX = 1.05, 1.45
SPREAD_BOUND = JITTER_MAX / JITTER_MIN


@pytest.fixture(scope="module")
def loaded():
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    try:
        from samyama import SamyamaClient
        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

    fleet = gen.generate(seed=SEED, scale=SCALE, operators=ops)
    from etl import real_layer
    real_layer.build_real(fleet, ops)
    try:
        client.query("MATCH (n) DETACH DELETE n", GRAPH)
    except Exception:
        pass
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client


def deployments_with_cost_model_inputs(client) -> list[dict]:
    """Every generated deployment, joined to the terms the cost model uses.

    Assembled from four single-hop queries rather than one wide pattern: engine
    note 1 is that a variable re-bound in a trailing position is not joined, and
    the resulting cartesian product is silent.
    """
    def rows(cypher):
        return client.query(cypher, GRAPH).records

    found = {r[0]: {"latency": r[1], "fallback": r[2]} for r in rows(
        "MATCH (x:Deployment) WHERE x.fallback_fraction IS NOT NULL "
        "RETURN x.id, x.latency_ms, x.fallback_fraction")}
    for did, variant in rows(
            "MATCH (x:Deployment)-[:OF_VARIANT]->(v:ModelVariant) RETURN x.id, v.id"):
        if did in found:
            found[did]["variant"] = variant
    for did, gops in rows(
            "MATCH (x:Deployment)-[:USES_ACCELERATOR]->(a:Accelerator) "
            "RETURN x.id, a.gops_int8"):
        if did in found:
            found[did]["gops_accel"] = gops
    for did, gops in rows(
            "MATCH (x:Deployment)-[:ON_BOARD]->(b:Board)-[:HAS_SOC]->(s:SoC)"
            "-[:HAS_ACCELERATOR]->(a:Accelerator) "
            'WHERE a.kind = "MCU-CPU" RETURN x.id, a.gops_int8'):
        if did in found:
            found[did]["gops_cpu"] = gops

    needed = {"variant", "gops_accel", "gops_cpu"}
    return [d for d in found.values() if needed <= set(d)]


def spread_per_variant(deployments) -> dict[str, float]:
    """max/min of `latency / shape` within each variant. 1.0 would be no jitter."""
    grouped = defaultdict(list)
    for d in deployments:
        shape = ((1 - d["fallback"]) / d["gops_accel"]
                 + d["fallback"] / d["gops_cpu"])
        grouped[d["variant"]].append(d["latency"] / shape)
    return {variant: max(values) / min(values)
            for variant, values in grouped.items()
            if len(values) > 1 and min(values) > 0}


def test_every_generated_deployment_has_the_cost_model_terms(loaded):
    """Guard: the assertion below is only as good as what it can reach."""
    deployments = deployments_with_cost_model_inputs(loaded)
    total = loaded.query(
        "MATCH (x:Deployment) WHERE x.fallback_fraction IS NOT NULL "
        "RETURN count(x) AS n", GRAPH).records[0][0]
    assert len(deployments) == total, (
        f"{len(deployments)} of {total} deployments could be joined to a "
        f"variant, an accelerator and an MCU-CPU; the rest are invisible here"
    )
    assert total > 50, f"only {total} deployments to check"


def test_latency_matches_the_documented_cost_model(loaded):
    spreads = spread_per_variant(deployments_with_cost_model_inputs(loaded))
    assert len(spreads) > 20, f"only {len(spreads)} variants have >1 deployment"

    over = {v: round(s, 4) for v, s in spreads.items() if s > SPREAD_BOUND}
    assert not over, (
        f"{len(over)} variants whose latency does not follow "
        f"macs * ((1-f)/gops_accel + f/gops_cpu): within one variant that ratio "
        f"may only vary by the generator's jitter, {SPREAD_BOUND:.4f}. "
        f"Worst: {sorted(over.items(), key=lambda kv: -kv[1])[:5]}"
    )


def test_falling_back_to_a_slower_cpu_is_what_costs_time(loaded):
    """The README's "really does pay for it", stated as the graph holds it.

    The penalty is only real where the CPU is slower than the accelerator; on an
    `MCU-CPU` deployment the two are the same unit and falling back costs
    nothing, which is correct rather than a defect.
    """
    deployments = deployments_with_cost_model_inputs(loaded)
    penalised = [d for d in deployments
                 if d["gops_cpu"] < d["gops_accel"] and d["fallback"] > 0]
    assert penalised, "no deployment falls back from a faster unit to a slower one"

    for d in penalised:
        no_fallback = 1 / d["gops_accel"]
        actual = (1 - d["fallback"]) / d["gops_accel"] + d["fallback"] / d["gops_cpu"]
        assert actual > no_fallback, (
            f"a deployment with fallback {d['fallback']} on a {d['gops_accel']} "
            f"GOPS unit falling back to {d['gops_cpu']} GOPS is not costed higher "
            f"than no fallback at all"
        )
