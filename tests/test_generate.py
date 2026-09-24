"""Invariants of the synthetic fleet. These are what the query tests lean on."""
import collections

import pytest

from etl import generate as gen
from etl import onnx_catalog as oc


@pytest.fixture(scope="module")
def fleet():
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    return gen.generate(seed=1234, scale=0.25, operators=ops)


@pytest.fixture(scope="module")
def both_layers():
    """The generated fleet plus the real layer, for invariants that span both.

    Separate from `fleet` rather than replacing it: every other test here is
    about what the *generator* guarantees, and folding public-source rows into
    that would make a failure ambiguous between the two.
    """
    from etl import real_layer

    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    fleet = gen.generate(seed=1234, scale=1.0, operators=ops)
    real_layer.build_real(fleet, ops)
    return fleet


def test_deterministic_for_a_given_seed(fleet):
    ops = oc.load_cached()
    again = gen.generate(seed=1234, scale=0.25, operators=ops)
    assert again.node_count == fleet.node_count
    assert again.edge_count == fleet.edge_count
    assert again.nodes["Board"] == fleet.nodes["Board"]
    assert again.edges == fleet.edges


def test_different_seeds_differ(fleet):
    other = gen.generate(seed=999, scale=0.25, operators=oc.load_cached())
    assert other.nodes["Board"] != fleet.nodes["Board"]


def test_node_ids_unique_per_label(fleet):
    for label, rows in fleet.nodes.items():
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate ids in {label}"


def test_every_edge_endpoint_exists(fleet):
    by_label = {label: {r["id"] for r in rows} for label, rows in fleet.nodes.items()}
    for src_label, src_id, rel, tgt_label, tgt_id, _ in fleet.edges:
        assert src_id in by_label[src_label], f"{rel}: missing source {src_id}"
        assert tgt_id in by_label[tgt_label], f"{rel}: missing target {tgt_id}"


def test_each_variant_belongs_to_exactly_one_model(fleet):
    counts = {}
    for src_label, src_id, rel, _, _, _ in fleet.edges:
        if rel == "VARIANT_OF":
            counts[src_id] = counts.get(src_id, 0) + 1
    assert counts and set(counts.values()) == {1}


def test_int8_is_exactly_a_quarter_of_fp32(fleet):
    """The query tests use this ratio to detect cartesian-product breakage."""
    by_model = {}
    variant_by_id = {v["id"]: v for v in fleet.nodes["ModelVariant"]}
    for src_label, src_id, rel, _, tgt_id, _ in fleet.edges:
        if rel == "VARIANT_OF":
            by_model.setdefault(tgt_id, []).append(variant_by_id[src_id])
    for model_id, variants in by_model.items():
        sizes = {v["precision"]: v["size_kb"] for v in variants}
        assert abs(sizes["fp32"] / 4.0 - sizes["int8"]) < 0.01, model_id


def test_every_clinical_task_has_a_model(fleet):
    solved = {tgt for _, _, rel, _, tgt, _ in fleet.edges if rel == "SOLVES"}
    all_tasks = {t["id"] for t in fleet.nodes["ClinicalTask"]}
    assert all_tasks == solved, "task-anchored queries would return empty"


def test_every_soc_has_a_cpu_fallback(fleet):
    """Every SoC must have an MCU-CPU, or 'falls back to CPU' is meaningless."""
    accel_by_id = {a["id"]: a for a in fleet.nodes["Accelerator"]}
    per_soc = {}
    for src_label, src_id, rel, _, tgt_id, _ in fleet.edges:
        if rel == "HAS_ACCELERATOR":
            per_soc.setdefault(src_id, []).append(accel_by_id[tgt_id]["kind"])
    assert per_soc
    for soc_id, kinds in per_soc.items():
        assert "MCU-CPU" in kinds, soc_id


def test_kernels_respect_opset_ceiling(fleet):
    """A kernel must never exist for an operator above its accelerator's opset."""
    accel = {a["id"]: a for a in fleet.nodes["Accelerator"]}
    op = {o["id"]: o for o in fleet.nodes["Operator"]}
    kernel_op, kernel_accel = {}, {}
    for src_label, src_id, rel, _, tgt_id, _ in fleet.edges:
        if rel == "IMPLEMENTS":
            kernel_op[src_id] = tgt_id
        elif rel == "RUNS_ON":
            kernel_accel[src_id] = tgt_id
    checked = 0
    for kid, op_id in kernel_op.items():
        a = accel[kernel_accel[kid]]
        assert op[op_id]["since_version"] <= a["opset_ceiling"], kid
        checked += 1
    assert checked > 0


def test_fallback_fraction_is_a_fraction(fleet):
    for d in fleet.nodes["Deployment"]:
        assert 0.0 <= d["fallback_fraction"] <= 1.0
        assert d["latency_ms"] > 0


def test_no_internal_fields_leak_into_node_properties(fleet):
    """Generator internals are `_`-prefixed; none may reach the graph.

    Regression: Sensor nodes were handed to add_nodes() and then mutated with a
    `_chain` field, which serialised into the graph as a stringified blob.
    """
    for label, rows in fleet.nodes.items():
        for row in rows:
            leaked = [k for k in row if k.startswith("_")]
            assert not leaked, f"{label} leaks internal fields: {leaked}"


def test_node_property_values_are_scalars(fleet):
    """Cypher properties must be scalars -- a dict or list silently stringifies."""
    for label, rows in fleet.nodes.items():
        for row in rows:
            for key, value in row.items():
                assert isinstance(value, (str, int, float, bool)) or value is None, (
                    f"{label}.{key} is {type(value).__name__}, not a scalar"
                )


def test_no_deployment_uses_more_than_one_accelerator(both_layers):
    """`EA18` reads this as an invariant, so it is checked rather than assumed.

    `EA18` counts, per (deployment, operator), the kernels that implement the
    operator *and* run on the deployment's accelerator, and calls the operator
    a fallback when that count is zero. With two accelerators on one
    deployment, `a` binds to either, kernels on **either** are counted
    together, and an operator covered on one accelerator but not the other
    stops looking like a fallback -- the query would under-report exactly the
    thing it exists to find.

    **Not "exactly one":** measured over both layers at `--scale 1.0`, 1,513
    deployments hold 1,451 `USES_ACCELERATOR` edges, one each. The other 62
    are real-layer MLPerf Tiny rows carrying no accelerator at all, and
    `EA18`'s opening `MATCH` simply does not bind them -- they are absent from
    its result rather than under-reported, which is a different thing and a
    harmless one. Both halves are asserted below, because the dangerous
    direction is two and the surprising direction is none.

    Over both layers deliberately: the zero-accelerator rows live in the real
    one, so a generated-only fleet cannot see them and a docstring claiming
    both layers would be describing a run this test never made.
    """
    deployments = {row["id"] for row in both_layers.nodes.get("Deployment", ())}
    per_deployment = collections.Counter(
        src for _sl, src, rel, _tl, _tgt, _p in both_layers.edges
        if rel == "USES_ACCELERATOR")
    # Before `max()`, which raises ValueError on an empty counter -- so a
    # sweep that found no edges at all used to abort here as an error rather
    # than fail as the assertion written for it, and the assertion placed
    # after `max()` could never run.
    assert per_deployment, "no USES_ACCELERATOR edges at all; EA18 cannot mean anything"

    worst = max(per_deployment.values())
    assert worst == 1, (
        f"a deployment has {worst} USES_ACCELERATOR edges. EA18 counts kernels "
        f"on 'the' accelerator and reads zero as a CPU fallback, so a second "
        f"one hides fallback rather than reporting it. Fix EA18 before "
        f"relaxing this.")

    without = deployments - set(per_deployment)
    provenance = {row["id"]: row.get("provenance")
                  for row in both_layers.nodes["Deployment"]}
    generated_without = sorted(d for d in without if provenance.get(d) != "real")
    assert not generated_without, (
        f"{len(generated_without)} generated deployments have no accelerator "
        f"({generated_without[:3]}). EA18 cannot see them at all, so a fleet "
        f"that stopped attaching accelerators would shrink EA18's scope "
        f"silently rather than change its answer.")


def test_every_model_ea18_can_reach_has_operators(both_layers):
    """`EA18` drops a breached deployment whose model has no operators.

    The `MATCH (m)-[:USES_OPERATOR]->(op:Operator)` leg is an inner match, so
    a model with no operator edges takes its deployment out of the result
    entirely -- not with a zero fallback count, but absent. For a query whose
    whole job is finding a breach nothing else would catch, silently dropping
    one is the worst failure it has.

    **It cannot arise today, and not for the reason the shape suggests.**
    There *are* operator-less models: the four real-layer MLPerf ones
    (`model:mlperf-ad`, `-ic`, `-kws`, and one more), stable at every seed and
    scale measured. What keeps them out of `EA18` is the *earlier*
    `USES_ACCELERATOR` hop -- the MLPerf deployments that reach them carry no
    accelerator, so the opening `MATCH` has already dropped them before the
    operator leg is reached. Give those deployments an accelerator and the
    operator hole opens.

    So the invariant worth pinning is the reachable one: every model a
    deployment *with an accelerator* can reach has at least one operator.
    Measured over both layers at seeds 20260814, 1234 and 999 and scales 0.25
    and 1.0: 60 reachable models at scale 1.0, none without operators.
    """
    edges = both_layers.edges
    with_operators = {src for label, src, rel, _tl, _tgt, _p in edges
                      if rel == "USES_OPERATOR" and label == "Model"}
    variant_of = collections.defaultdict(set)
    of_variant = collections.defaultdict(set)
    has_accelerator = set()
    for _sl, src, rel, _tl, tgt, _p in edges:
        if rel == "VARIANT_OF":
            variant_of[src].add(tgt)
        elif rel == "OF_VARIANT":
            of_variant[src].add(tgt)
        elif rel == "USES_ACCELERATOR":
            has_accelerator.add(src)

    reachable = {model
                 for deployment in has_accelerator
                 for variant in of_variant.get(deployment, ())
                 for model in variant_of.get(variant, ())}
    assert reachable, (
        "no model is reachable from a deployment with an accelerator, so "
        "EA18 cannot return anything and this guard measures nothing")

    without = sorted(reachable - with_operators)
    assert not without, (
        f"{len(without)} model(s) reachable from an accelerator-carrying "
        f"deployment have no USES_OPERATOR edges ({without[:3]}). EA18's "
        f"operator leg is an inner MATCH, so a breached deployment on one of "
        f"these is dropped from the result rather than reported with zero "
        f"fallback operators. Make that leg OPTIONAL before relaxing this.")


def test_generated_labels_matches_what_the_generator_emits(fleet):
    """`GENERATED_LABELS` is what `load()` refuses a cache for; keep it honest.

    A hand-maintained list beside the code it describes drifts the moment
    someone adds a label and forgets it -- which is the same failure it exists
    to catch, one level up.

    Through the module's `fleet` fixture rather than generating inline: its
    skip happens in **setup**, where `--no-skips` can convert it, and a skip
    written in a test body cannot be (`tests/test_environment_skips.py`). The
    label *set* does not depend on the fixture's scale, only the row counts do.
    """
    emitted = set(fleet.nodes)
    assert emitted == set(gen.GENERATED_LABELS), (
        f"the generator emits {sorted(emitted - set(gen.GENERATED_LABELS))} that "
        f"GENERATED_LABELS omits, and lists "
        f"{sorted(set(gen.GENERATED_LABELS) - emitted)} it does not emit. "
        f"`etl.generate.load()` refuses a cached fleet missing any of these, "
        f"so a stale entry here either rejects every cache or lets a stale one "
        f"through.")


def test_a_fleet_cache_written_before_a_label_existed_is_refused(tmp_path):
    """The stale-cache case, on a throwaway file rather than the shared `data/`.

    `data/` is gitignored, so a fresh clone is always current. An existing
    checkout is where this bites: `git pull` brings a new label, `fleet.json`
    keeps the old shape, and the catalog query over that label returns zero
    rows -- indistinguishable from "the answer is none". `--verify` cannot
    catch it either, because it compares the load against this same cache.
    """
    import json

    from etl import generate as gen

    complete = {label: [{"id": f"{label.lower()}:00000"}]
                for label in gen.GENERATED_LABELS}
    path = tmp_path / "fleet.json"

    def write(nodes):
        path.write_text(json.dumps(
            {"seed": gen.DEFAULT_SEED, "scale": 1.0, "node_count": len(nodes),
             "edge_count": 0, "nodes": nodes, "edges": []}), encoding="utf-8")

    write(complete)
    assert set(gen.load(path).nodes) == set(gen.GENERATED_LABELS), (
        "a complete cache must load; otherwise this guard rejects everything")

    stale = {k: v for k, v in complete.items() if k != "Site"}
    write(stale)
    with pytest.raises(gen.StaleFleetCache, match="Site"):
        gen.load(path)

    # A cache from a *newer* checkout is not this function's business.
    write({**complete, "Warehouse": [{"id": "warehouse:00000"}]})
    gen.load(path)
