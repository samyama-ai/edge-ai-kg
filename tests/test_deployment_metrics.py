"""`fallback_op_count` and `fallback_fraction` describe one fact twice.

`etl/generate.py` derives both from the same list:

    fallback_ops = [o for o in model_ops if o.id not in covered]
    frac_fb      = len(fallback_ops) / max(1, len(model_ops))

so the denominator is the model's operator surface, which the graph still
holds as `USES_OPERATOR` edges. That makes the pair checkable rather than
merely consistent-looking: a generator change touching one and not the other
leaves the audit and the summary disagreeing, both plausibly (#28).

Two things worth knowing about scope:

* `fallback_fraction` is stored as `round(frac, 4)`, so the two agree to within
  half of the last retained digit -- 5e-5 -- and no closer. The tolerance below
  is that bound, not a guess.
* The 73 real MLPerf Tiny deployments carry neither property. They are measured
  submissions, not derived ones, so there is no cost model behind them to check.
  The test asserts it is looking at the generated layer rather than silently
  finding nothing.
"""
import pytest

from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
SCALE = 0.3
SEED = 4242

# `fallback_fraction` is stored as round(x, 4); half the last digit is the
# tightest agreement the stored value can express.
ROUNDING_TOLERANCE = 5e-5

PAIRED_WITH_OPERATOR_SURFACE = """
MATCH (d:Deployment)-[:OF_VARIANT]->(v:ModelVariant)-[:VARIANT_OF]->(m:Model)-[:USES_OPERATOR]->(op:Operator)
WITH d.id AS deployment, d.fallback_op_count AS count, d.fallback_fraction AS fraction,
     count(op.id) AS total_ops
RETURN deployment, count, fraction, total_ops
"""


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


def paired_rows(client):
    """One row per deployment reachable through its model's operator surface."""
    return client.query(PAIRED_WITH_OPERATOR_SURFACE.strip(), GRAPH).records


def test_the_fixture_actually_has_deployments_to_check(loaded):
    """Guard against the assertions below passing on an empty result."""
    rows = paired_rows(loaded)
    assert len(rows) > 100, (
        f"only {len(rows)} deployments reachable through USES_OPERATOR; the "
        f"consistency assertions would be close to vacuous"
    )


def test_fraction_equals_count_over_the_models_operator_surface(loaded):
    disagreeing = []
    for deployment, count, fraction, total_ops in paired_rows(loaded):
        if count is None or fraction is None or not total_ops:
            continue
        delta = abs(fraction - count / total_ops)
        if delta > ROUNDING_TOLERANCE:
            disagreeing.append((deployment, count, fraction, total_ops, round(delta, 6)))
    assert not disagreeing, (
        f"{len(disagreeing)} deployments where fallback_fraction is not "
        f"fallback_op_count / (operators on the model), beyond the {ROUNDING_TOLERANCE} "
        f"rounding bound: {disagreeing[:5]}"
    )


def test_the_pair_is_present_or_absent_together(loaded):
    """Neither half should exist without the other."""
    def missing(prop_present, prop_absent):
        result = loaded.query(
            f"MATCH (d:Deployment) WHERE d.{prop_present} IS NOT NULL "
            f"AND d.{prop_absent} IS NULL RETURN count(d) AS n", GRAPH)
        return result.records[0][0] if result.records else 0

    half = (missing("fallback_op_count", "fallback_fraction"),
            missing("fallback_fraction", "fallback_op_count"))
    assert half == (0, 0), (
        f"{half[0]} deployments carry a count with no fraction, "
        f"{half[1]} carry a fraction with no count"
    )


def test_the_real_layer_carries_neither(loaded):
    """MLPerf submissions are measured, not derived -- there is no cost model.

    Asserted so that the split is a recorded decision rather than something a
    reader has to infer from a null.
    """
    result = loaded.query(
        'MATCH (d:Deployment) WHERE d.provenance = "real" '
        "AND d.fallback_op_count IS NOT NULL RETURN count(d) AS n", GRAPH)
    leaked = result.records[0][0] if result.records else 0
    assert leaked == 0, (
        f"{leaked} real deployments carry fallback_op_count; measured "
        f"submissions have no cost model behind them"
    )
