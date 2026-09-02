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

## Rounding -- two sources, both paid for

`latency_ms` is stored rounded, so it carries up to half a unit in the last
place. That is an *absolute* error: 1.4% of the smallest latency in the graph
today, shrinking as latency grows.

`fallback_fraction` is stored rounded too -- and `latency_ms` was computed from
the **unrounded** value. So rebuilding `shape` from what the graph holds is
wrong by `d(shape)/df * delta_f`. That error is *relative*: it does not shrink
with latency, so it cannot be folded into the first, and at the slow end of the
fleet -- where the latency term has decayed to nothing -- it is all that is
left. Measured worst on a full load: 0.06% from the fraction, 1.4% from the
latency.

The bound is therefore widened per variant by the sum of both, and the widening
is itself bounded: `allowed_spread` grows without limit as a latency shrinks,
and at 0.001 ms -- the smallest value surviving the rounding -- it reaches 4.14,
wide enough to pass a broken derivation.
`test_rounding_never_swallows_the_tolerance` fails before that, so the tolerance
cannot quietly stop being one.

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

# round(x, n) moves a value by at most half a unit in the last place. There are
# two such roundings between the cost model and what this test can recompute:
# `latency_ms` is stored rounded, and `fallback_fraction` is stored rounded
# while the latency was computed from the *unrounded* fraction.
LATENCY_ROUNDING_MS = 0.5 * 10 ** -gen.LATENCY_DECIMALS
FRACTION_ROUNDING = 0.5 * 10 ** -gen.FALLBACK_FRACTION_DECIMALS

# Above this, the allowance is wide enough to pass a broken derivation, so the
# tolerance has stopped being a tolerance. Measured worst today: 0.4%.
MAX_TOLERABLE_ERROR = 0.05
# Nearly every variant carries several deployments; a collapse in that is a
# generator change hiding most of the population from the assertion.
MIN_CHECKABLE_FRACTION = 0.8


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

    # Filtered on `provenance`, not on `fallback_fraction IS NOT NULL`. The
    # real MLPerf deployments are appended to the same node list and created in
    # the same batched CREATE, and engine note 8 records that a node created
    # without a property can inherit the previous generation's column value.
    # `provenance` is stamped by `Fleet.add_nodes` and cannot appear that way.
    found = {r[0]: {"latency": r[1], "fallback": r[2]} for r in rows(
        'MATCH (x:Deployment) WHERE x.provenance = "synthetic" '
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


def recoverable_error(d: dict) -> float:
    """Relative error in `latency / shape` from what the graph stores rounded.

    Two independent sources, and both have to be paid for:

    * `latency_ms` is stored rounded -- an absolute error, so its relative size
      shrinks as latency grows.
    * `fallback_fraction` is stored rounded *and the latency was computed from
      the unrounded value*, so rebuilding `shape` from the stored one is wrong
      by `d(shape)/df * delta_f`. That error is relative and does not shrink
      with latency, which is exactly why it cannot be folded into the first.
    """
    shape = shape_of(d)
    from_latency = LATENCY_ROUNDING_MS / d["latency"]
    from_fraction = (FRACTION_ROUNDING
                     * abs(1 / d["gops_cpu"] - 1 / d["gops_accel"]) / shape)
    return from_latency + from_fraction


def shape_of(d: dict) -> float:
    return (1 - d["fallback"]) / d["gops_accel"] + d["fallback"] / d["gops_cpu"]


def normalised_by_variant(deployments) -> dict[str, list[tuple[float, float]]]:
    """(latency / shape, recoverable error) per variant."""
    grouped = defaultdict(list)
    for d in deployments:
        grouped[d["variant"]].append((d["latency"] / shape_of(d), recoverable_error(d)))
    return grouped


def allowed_spread(error: float) -> float:
    """The jitter bound, widened by what rounding alone can explain."""
    return SPREAD_BOUND * (1 + error) / (1 - error)


def test_every_generated_deployment_has_the_cost_model_terms(loaded):
    """Guard: the assertion below is only as good as what it can reach."""
    deployments = deployments_with_cost_model_inputs(loaded)
    total = loaded.query(
        'MATCH (x:Deployment) WHERE x.provenance = "synthetic" '
        "RETURN count(x) AS n", GRAPH).records[0][0]
    assert len(deployments) == total, (
        f"{len(deployments)} of {total} deployments could be joined to a "
        f"variant, an accelerator and an MCU-CPU; the rest are invisible here"
    )
    assert total > 50, f"only {total} deployments to check"


def test_rounding_never_swallows_the_tolerance(loaded):
    """The allowance must stay an allowance.

    `allowed_spread` grows without limit as a latency shrinks: the smallest
    value surviving `round(x, 3)` is 0.001 ms, which alone gives a relative
    error of 0.5 and an allowance of 4.14 -- wide enough for a completely broken
    derivation to pass. Checking only for a zero latency does not catch that, so
    the error itself is bounded rather than its most extreme cause.
    """
    deployments = deployments_with_cost_model_inputs(loaded)
    zeroed = [d for d in deployments if d["latency"] <= 0]
    assert not zeroed, (
        f"{len(zeroed)} deployments have a latency of 0 after rounding to "
        f"{gen.LATENCY_DECIMALS} decimals and cannot be normalised at all"
    )
    worst = max(recoverable_error(d) for d in deployments)
    assert worst < MAX_TOLERABLE_ERROR, (
        f"rounding alone explains {worst:.1%} of the ratio, so the bound widens "
        f"to {allowed_spread(worst):.2f} against a real jitter of "
        f"{SPREAD_BOUND:.4f}; at that width the assertion below proves nothing"
    )


def test_latency_matches_the_documented_cost_model(loaded):
    deployments = deployments_with_cost_model_inputs(loaded)
    grouped = normalised_by_variant(deployments)
    checkable = {v: vals for v, vals in grouped.items() if len(vals) > 1}

    # Proportional, not absolute: a generator change leaving most variants with
    # a single deployment would drop them from the check while a fixed floor
    # like "more than 20" stayed satisfied.
    assert len(checkable) >= MIN_CHECKABLE_FRACTION * len(grouped), (
        f"only {len(checkable)} of {len(grouped)} variants carry more than one "
        f"deployment, so most of the population is outside this assertion"
    )

    over = {}
    for variant, values in checkable.items():
        ratios = [ratio for ratio, _ in values]
        allowed = allowed_spread(max(error for _, error in values))
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
