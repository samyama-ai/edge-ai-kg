"""Deployment latency is derived, not drawn at random -- asserted, not assumed.

The README makes this the claim that separates the dataset from noise:

    Deployment metrics are *derived* from a documented cost model rather than
    drawn at random, so a board missing a kernel really does pay for it.

`etl/generate.py` splits the MACs between the accelerator and the CPU it falls
back to:

    t_acc = (macs * 2 * (1 - frac_fb)) / (gops_accel * 1e9 * thr)
    t_cpu = (macs * 2 * frac_fb)       / (gops_cpu   * 1e9 * thr)
    latency_ms = round((t_acc + t_cpu) * 1000 * uniform(*LATENCY_JITTER), 3)

## Why this is not a correlation test

A plain correlation does not work: `macs` and the accelerator's speed dominate,
and measured across the whole graph Spearman(`fallback_fraction`, `latency_ms`)
is only **0.166**. An assertion loose enough to pass on that would not catch a
broken derivation.

The formula is recoverable instead. `thr` and `macs` are per-variant constants,
so within one `ModelVariant` they cancel:

    latency / shape  =  macs * 2 * 1000 * jitter / thr     where
    shape            =  (1 - f) / gops_accel + f / gops_cpu

Every deployment of a variant therefore shares that ratio up to the jitter, so
its spread within a variant cannot exceed `max(JITTER) / min(JITTER)`. That
bound is imported from the generator rather than restated here, so a change to
the spread cannot leave this test passing against a stale number.

`gops_cpu` is the `MCU-CPU` accelerator on the deployment's board's SoC, which is
how `etl/generate.py` chooses it, and is reachable from the graph.

## Rounding

`latency_ms` is rounded to `LATENCY_DECIMALS`, so a stored value carries up to
half a unit in the last place of absolute error -- 1.4% of the smallest latency
in the graph today. That is larger than the headroom between the observed worst
spread and the bound, so the bound is widened per variant by what rounding can
actually do at that variant's smallest latency, rather than being applied flat.

A latency that rounds to zero cannot be normalised at all. None occur at the
seed and scale used here, and the test asserts that rather than filtering them
away silently -- a quiet `min(values) > 0` guard would let a generator change
shrink the checked population without anyone noticing.

## A note on MCU-CPU deployments

`etl/generate.py` picks the fallback CPU as `next((a for a in accels if
a["kind"] == "MCU-CPU"), accel)`. On a deployment whose accelerator *is* the
MCU-CPU, the two are the same unit, `shape` does not vary with `f`, and falling
back costs nothing. That is correct rather than a defect: there is nothing
faster to fall back from.

## Measured

Full load (1,440 generated deployments, 240 variants): worst spread **1.3760**
against the **1.3810** bound, nothing over. At this fixture's scale, **1.3451**.

The assertion has teeth -- recomputed against the same graph with the derivation
broken:

    derivation reversed (f -> 1-f)   37 of 72 variants over the bound
    latencies shuffled               65 of 72 variants over the bound
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

# Imported, not restated: a change to the generator's spread must reach here.
SPREAD_BOUND = max(gen.LATENCY_JITTER) / min(gen.LATENCY_JITTER)
# round(x, n) moves a value by at most half a unit in the last place.
ROUNDING_ERROR_MS = 0.5 * 10 ** -gen.LATENCY_DECIMALS


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


def normalised_by_variant(deployments) -> dict[str, list[tuple[float, float]]]:
    """(latency / shape, latency) per variant. The ratio is constant but for jitter."""
    grouped = defaultdict(list)
    for d in deployments:
        shape = ((1 - d["fallback"]) / d["gops_accel"]
                 + d["fallback"] / d["gops_cpu"])
        grouped[d["variant"]].append((d["latency"] / shape, d["latency"]))
    return grouped


def rounding_slack(smallest_latency: float) -> float:
    """How much of the observed spread `round(latency, 3)` alone can explain."""
    relative = ROUNDING_ERROR_MS / smallest_latency
    return (1 + relative) / (1 - relative)


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


def test_no_latency_rounds_away_to_zero(loaded):
    """A zeroed latency cannot be normalised, so it would leave the check.

    Asserted rather than filtered: `latency_ms` is rounded, and a faster
    generated fleet -- bigger GOPS ranges, smaller models -- would push the
    fastest deployments under half a millisecond's last place. If that happens
    this fails and names it, instead of the population quietly shrinking.
    """
    deployments = deployments_with_cost_model_inputs(loaded)
    zeroed = [d for d in deployments if d["latency"] <= 0]
    assert not zeroed, (
        f"{len(zeroed)} deployments have a latency of 0 after rounding to "
        f"{gen.LATENCY_DECIMALS} decimals; they cannot be normalised against "
        f"the cost model and would silently leave the assertion"
    )


def test_latency_matches_the_documented_cost_model(loaded):
    grouped = normalised_by_variant(deployments_with_cost_model_inputs(loaded))
    checkable = {v: vals for v, vals in grouped.items() if len(vals) > 1}
    assert len(checkable) > 20, f"only {len(checkable)} variants have >1 deployment"

    over = {}
    for variant, values in checkable.items():
        ratios = [r for r, _ in values]
        allowed = SPREAD_BOUND * rounding_slack(min(lat for _, lat in values))
        spread = max(ratios) / min(ratios)
        if spread > allowed:
            over[variant] = (round(spread, 4), round(allowed, 4))

    assert not over, (
        f"{len(over)} of {len(checkable)} variants whose latency does not follow "
        f"macs * ((1-f)/gops_accel + f/gops_cpu). Within one variant that ratio "
        f"may only vary by the generator's jitter ({SPREAD_BOUND:.4f}), widened "
        f"by what rounding can explain. Worst (spread, allowed): "
        f"{sorted(over.items(), key=lambda kv: -kv[1][0])[:5]}"
    )
