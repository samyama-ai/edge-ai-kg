"""Invariants of the synthetic fleet. These are what the query tests lean on."""
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


def test_generated_labels_matches_what_the_generator_emits(tmp_path):
    """`GENERATED_LABELS` is what `load()` refuses a cache for; keep it honest.

    A hand-maintained list beside the code it describes drifts the moment
    someone adds a label and forgets it -- which is the same failure it exists
    to catch, one level up. Generated at the smallest useful scale: the label
    *set* does not depend on scale, only the row counts do.
    """
    from etl import generate as gen
    from etl import onnx_catalog as oc

    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")

    emitted = set(gen.generate(seed=gen.DEFAULT_SEED, scale=0.1, operators=ops).nodes)
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
