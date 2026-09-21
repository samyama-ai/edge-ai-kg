"""Validate catalog queries against ground truth computed in Python.

These tests exist because engine v1.7.0 has query-shape bugs that return
*plausible but wrong* rows (see docs/engine-notes.md). A query that runs and
returns rows is not evidence that it is right, so every structural claim the
demo makes is checked here against the generator's own data.

Runs against an in-process embedded engine, so no server is needed.

## This file is deliberately not split (#31)

#31 asked whether this file should be split by subject, expecting the batch of
new tests around it to land here and push it past what a reviewer reads in one
pass. **That did not happen** -- every one of those went into its own file:

    $ ls tests/test_*.py | grep -v test_correctness | wc -l

No number is quoted, only the command. A list went stale twice before this
comment shipped, and the number that replaced it was wrong on arrival -- taken
from a working tree holding two other branches' untracked files. The count also
moves with every merge, so any literal here is stale by design. The argument
does not rest on the answer being 20 or 22; it rests on it being large, and the
command carries that.

So the split happened, by subject, without touching this file. What is left here
is one subject -- catalog answers checked against the `Fleet` -- plus the fixture
they share.

**Splitting what remains would cost more than it buys.** Measured: of this
module's 24.85s, **24.06s is the `loaded` fixture** building and loading a graph,
and the eleven tests themselves total about 0.3s. Ten of the eleven use that
fixture. Two files means two module-scoped fixtures and two loads, roughly
doubling this area's runtime to move ~130 lines.

The seam #31 proposes -- graph integrity versus catalog answers -- also yields a
one-test file: `test_graph_loaded_completely` is the only integrity assertion
here, and the rest of that subject already lives in `test_id_uniqueness.py`,
`test_provenance.py` and `test_edge_verification.py`.

**Decision: one file, and the size limit does not apply here.** Revisit if a
second subject arrives, or if a way to share the fixture across modules without
reloading appears -- `SamyamaClient.embedded()` is per-process and in-memory, so
today there is none.
"""
from __future__ import annotations

import pytest

from benchmarks.queries import BY_ID
from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
SCALE = 0.3
SEED = 4242


@pytest.fixture(scope="module")
def loaded():
    """Generate a small fleet, load it into an embedded engine, return both."""
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
    # The catalog includes EA13-EA16, which run on the real public-source layer,
    # so the fixture must load both layers or those queries come back empty.
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
    return client, fleet


def rows(client, cypher):
    result = client.query(cypher.strip(), GRAPH)
    return list(result.columns), result.records


def index(fleet):
    """Build lookup tables from the fleet for ground-truth computation."""
    out = {
        "node": {label: {r["id"]: r for r in rs} for label, rs in fleet.nodes.items()},
        "out": {},
    }
    for src_label, src_id, rel, tgt_label, tgt_id, props in fleet.edges:
        out["out"].setdefault(rel, []).append((src_id, tgt_id, props))
    return out


# --------------------------------------------------------------------------


def test_graph_loaded_completely(loaded):
    client, fleet = loaded
    _, recs = rows(client, "MATCH (n) RETURN count(n) AS n")
    assert recs[0][0] == fleet.node_count
    _, recs = rows(client, "MATCH ()-[r]->() RETURN count(r) AS n")
    assert recs[0][0] == fleet.edge_count


def test_ea11_cpu_only_models_match_ground_truth(loaded):
    """EA11's anti-join, against Python, on a real-cardinality graph.

    `EA11` is written as `OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op),
    (k)-[:RUNS_ON]->(a:Accelerator)` -- the comma-separated single-`MATCH` shape
    that `docs/engine-notes.md` records as correct on a toy graph and **wrong**
    at scale, with the explicit warning that "a passing 6-node reproduction
    proves nothing". `tests/test_empty_answers.py` builds exactly such a
    reproduction to pin the zero-vs-one distinction, so the shape itself is
    checked here instead.

    Verified once by hand at full `--scale 1.0` (24,115 nodes), which is what
    the engine note asks for: 60 of 60 models are CPU-only somewhere and EA11's
    top ten counts `[12, 11, 10, 10, 10, 9, 9, 9, 9, 9]` matched Python exactly.
    This runs at `SCALE` so the suite stays fast; if the join ever breaks it is
    the ordered counts that go wrong, and those are compared in full.

    Note `SignalStage` also emits `USES_OPERATOR`. EA11 filters on `:Model`, so
    the ground truth below must too, or it counts operators no model uses.
    """
    client, fleet = loaded
    idx = index(fleet)
    accel = {a["id"]: a for a in fleet.nodes["Accelerator"]}
    name = {m["id"]: m["name"] for m in fleet.nodes["Model"]}
    op_name = {o["id"]: o["name"] for o in fleet.nodes["Operator"]}

    # Derived from the source definitions, NOT from `is_cpu_fallback` -- the
    # property EA11 filters on. An earlier version computed truth with the same
    # `kind != "MCU-CPU"` rule the query used, so the two agreed with each other
    # while both treating ONNX Runtime's CPU provider as an accelerator (#69).
    # A ground truth that restates the query cannot check the query's premise,
    # so this rebuilds the notion of "is a CPU" from `ORT_DEVICES` and checks
    # the flag as well as the query.
    #
    # Not fully independent, and worth not overselling: `cpu_kinds` is the union
    # of the same two rules the implementation applies, so it agrees by
    # construction unless one side is edited. What it does catch is exactly
    # that -- one side drifting, including `add_mlperf_layer`'s hardcoded
    # `kind: "NPU"` -- which is the realistic failure, not a conspiracy.
    from etl.real_layer import ORT_DEVICES
    # Read from the source tables, not from `is_cpu_fallback` -- the property
    # EA11 filters on -- so this checks the flag as well as the query.
    #
    # An earlier version wrote `{kind for _, kind in ORT_DEVICES.values()
    # if kind == "CPU"}` and claimed it would pick up a future CPU-ish entry.
    # It would not: that reduces to {"CPU"}, so a `("OpenVINO CPU EP",
    # "CPU-OpenVINO")` would be classified as an accelerator by the test and by
    # the implementation alike, agreeing with each other and both wrong.
    # `ORT_DEVICES` now carries the verdict as its third element, so this reads
    # the decision instead of re-deriving it from a spelling.
    cpu_kinds = {"MCU-CPU"} | {kind for _name, kind, fallback in ORT_DEVICES.values()
                               if fallback}

    def is_fallback_target(accelerator_id: str) -> bool:
        row = accel.get(accelerator_id)
        return bool(row) and row["kind"] in cpu_kinds

    runs = {}
    for k, a, _ in idx["out"]["RUNS_ON"]:
        runs.setdefault(k, set()).add(a)
    accelerated = {
        op for k, op, _ in idx["out"]["IMPLEMENTS"]
        if any(not is_fallback_target(a) for a in runs.get(k, ()))
    }

    # The flag EA11 reads must agree with that independent derivation, or the
    # comparison below is checking the query against itself again.
    disagreeing = [a["id"] for a in fleet.nodes["Accelerator"]
                   if bool(a.get("is_cpu_fallback")) != (a["kind"] in cpu_kinds)]
    assert not disagreeing, (
        f"`is_cpu_fallback` disagrees with the kind it is derived from for "
        f"{disagreeing[:5]}. EA11 filters on the flag, so a wrong flag is a "
        f"wrong answer that this test would otherwise inherit."
    )
    uses = {}
    for m, op, _ in idx["out"]["USES_OPERATOR"]:
        if m in name:                      # :Model only -- SignalStage also uses operators
            uses.setdefault(m, set()).add(op)
    assert uses, "no model has an operator surface"

    truth = {}
    for m, ops in uses.items():
        cpu_only = sorted(op_name[o] for o in ops if o not in accelerated)
        if cpu_only:
            truth[name[m]] = cpu_only

    _, recs = rows(client, BY_ID["EA11"]["cypher"])
    assert recs, "EA11 returned nothing; every model should have some CPU-only operator"
    expected = sorted(((len(v), k) for k, v in truth.items()), reverse=True)[:len(recs)]
    assert [r[2] for r in recs] == [n for n, _ in expected], (
        f"EA11's counts disagree with the Fleet. Got {[(r[0], r[2]) for r in recs]}, "
        f"expected the top {len(recs)} of {len(truth)} to be {expected}. This is the "
        f"shape docs/engine-notes.md warns breaks once cardinalities are real."
    )
    for model, _family, count, operators in recs:
        assert sorted(operators) == truth[model], (
            f"EA11 named the wrong operators for {model}: got {sorted(operators)}, "
            f"expected {truth[model]}"
        )


def test_ea01_fallback_audit_matches_ground_truth(loaded):
    """The hero query: operators with no kernel on a chosen accelerator."""
    client, fleet = loaded
    idx = index(fleet)
    model = fleet.nodes["Model"][0]
    accel = fleet.nodes["Accelerator"][0]

    model_ops = {t for s, t, _ in idx["out"]["USES_OPERATOR"] if s == model["id"]}
    assert model_ops, "model has no operators"
    kernels_on_accel = {k for k, a, _ in idx["out"]["RUNS_ON"] if a == accel["id"]}
    covered = {t for s, t, _ in idx["out"]["IMPLEMENTS"] if s in kernels_on_accel}
    expected = {idx["node"]["Operator"][o]["name"] for o in model_ops - covered}

    cypher = f"""
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.id = "{model['id']}"
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.id = "{accel['id']}"
WITH op, count(k) AS kernels
WHERE kernels = 0
RETURN op.name AS operator
"""
    _, recs = rows(client, cypher)
    assert {r[0] for r in recs} == expected


def test_ea04_quantization_unlock_is_not_a_cartesian_product(loaded):
    """int8 size must be exactly a quarter of fp32 size for the SAME model.

    This is the canary for the v1.7.0 self-join bug: a cartesian product pairs
    variants from different models and the ratio drifts away from 4.
    """
    client, _ = loaded
    cols, recs = rows(client, BY_ID["EA04"]["cypher"])
    i_fp32, i_int8 = cols.index("fp32_kb"), cols.index("int8_kb")
    for rec in recs:
        fp32, int8 = rec[i_fp32], rec[i_int8]
        assert int8 > 0
        assert abs(fp32 / int8 - 4.0) < 0.05, (
            f"fp32={fp32} int8={int8} -- variants came from different models"
        )


def test_ea05_operator_coverage_matches_ground_truth(loaded):
    client, fleet = loaded
    idx = index(fleet)
    accel_of_kernel = {k: a for k, a, _ in idx["out"]["RUNS_ON"]}
    op_of_kernel = {k: o for k, o, _ in idx["out"]["IMPLEMENTS"]}
    expected: dict[str, set] = {}
    for kernel, accel_id in accel_of_kernel.items():
        kind = idx["node"]["Accelerator"][accel_id]["kind"]
        expected.setdefault(kind, set()).add(op_of_kernel[kernel])

    cols, recs = rows(client, BY_ID["EA05"]["cypher"])
    got = {r[cols.index("accelerator_kind")]: r[cols.index("operators_covered")]
           for r in recs}
    assert got == {k: len(v) for k, v in expected.items()}


def test_cpu_covers_every_operator(loaded):
    """Every SoC has an MCU-CPU that runs everything -- the fallback premise.

    Scoped to the ONNX catalog: the real ONNX Runtime layer adds operators from
    other domains (com.microsoft, ai.onnx.ml) that the synthetic fleet never
    models, and the synthetic CPU is not expected to cover those.
    """
    client, fleet = loaded
    catalog_ops = [o for o in fleet.nodes["Operator"] if o["source"] == "onnx"]
    _, recs = rows(client, """
MATCH (a:Accelerator)<-[:RUNS_ON]-(k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WHERE a.kind = "MCU-CPU" AND op.source = "onnx"
RETURN count(DISTINCT op.id) AS n
""")
    assert recs[0][0] == len(catalog_ops)


def test_ea06_blast_radius_matches_ground_truth(loaded):
    client, fleet = loaded
    idx = index(fleet)
    op = next(o for o in fleet.nodes["Operator"] if o["name"] == "Conv")

    models = {s for s, t, _ in idx["out"]["USES_OPERATOR"] if t == op["id"]}
    variants = {s for s, t, _ in idx["out"]["VARIANT_OF"] if t in models}
    deploys = {s for s, t, _ in idx["out"]["OF_VARIANT"] if t in variants}
    fitting = {d for d in deploys if idx["node"]["Deployment"][d]["fits"] == 1}

    cols, recs = rows(client, BY_ID["EA06"]["cypher"])
    assert recs, "EA06 returned nothing"
    assert recs[0][cols.index("deployments_at_risk")] == len(fitting)


def test_ea12_vendor_totals_match_ground_truth(loaded):
    client, fleet = loaded
    idx = index(fleet)
    soc_of_board = {b: s for b, s, _ in idx["out"]["HAS_SOC"]}
    vendor_of = {s: v for s, v, _ in idx["out"]["MADE_BY"]}
    board_of_deploy = {d: b for d, b, _ in idx["out"]["ON_BOARD"]}

    expected: dict[str, int] = {}
    for deploy, board in board_of_deploy.items():
        # EA12 filters on `fits`, which only synthetic deployments carry; the
        # real MLPerf submissions have no such notion and are excluded.
        if idx["node"]["Deployment"][deploy].get("fits") != 1:
            continue
        if board not in soc_of_board:
            continue
        vendor = vendor_of[soc_of_board[board]]
        name = idx["node"]["Vendor"][vendor]["name"]
        expected[name] = expected.get(name, 0) + 1

    cols, recs = rows(client, BY_ID["EA12"]["cypher"])
    got = {r[cols.index("vendor")]: r[cols.index("deployments")] for r in recs}
    assert got == expected


# EA01 and EA02 each introduce a new alias in a second WITH -- EA01 from a
# property expression (`op.name AS operator`), EA02 from an aggregate
# (`count(k) AS kernel_count`). Both were excused here as `xfail` until #56:
# the 0.6.x embedded build did not register such an alias and raised
# `Variable not found`, while the 1.7.0 server answered correctly. Pinning the
# embedded build to `samyama>=1.7.1` removed the divergence, both parameters
# XPASSed, and the marks came off. Engine note 10 keeps the history.


@pytest.mark.parametrize("qid", list(BY_ID))
def test_every_catalog_query_runs_and_returns_rows(loaded, qid):
    """Parametrised rather than one sweep, so a single query can be excused.

    Marking the whole sweep would excuse the other fifteen too: EA07 could stop
    returning rows and the run would still be green.
    """
    client, _ = loaded
    _, recs = rows(client, BY_ID[qid]["cypher"])
    # EA04 needs a model that misses at fp32 but fits at int8 on the same board;
    # at this small scale that combination may legitimately not occur. Its
    # correctness is pinned by test_ea04_shape_is_not_a_cartesian_product below.
    if qid != "EA04":
        assert recs, f"{qid} returned no rows"


@pytest.fixture
def embedded_client():
    """A fresh embedded client, skipping in **setup**.

    The purpose-built fixture tests below constructed their own and called
    `pytest.skip` from the body. `conftest.py` converts only setup-phase skips
    under `--no-skips`, so on a machine without the extension those regressions
    were skipped and the suite reported green -- which is the state that option
    exists to make impossible.
    """
    try:
        from samyama import SamyamaClient

        return SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")


def test_ea04_shape_is_not_a_cartesian_product(embedded_client):
    """Purpose-built regression test for the v1.7.0 self-join bug.

    One board carries an fp32 (does not fit) and an int8 (fits) deployment for
    EACH of two models. The correct answer pairs each model with its OWN
    variants -- 2 rows. The buggy join shapes return 4, pairing model A's fp32
    with model B's int8. See docs/engine-notes.md item 1.
    """
    client = embedded_client
    g = "default"
    for label in ("XBoard", "XModel", "XVariant", "XDeploy"):
        try:
            client.query(f"MATCH (n:{label}) DETACH DELETE n", g)
        except Exception:
            pass

    client.query(
        'CREATE (:XBoard {id: "B1", ram_kb: 1024}), '
        '(:XModel {id: "MA"}), (:XModel {id: "MB"}), '
        '(:XVariant {id: "MA32", precision: "fp32", size_kb: 4000.0}), '
        '(:XVariant {id: "MA8",  precision: "int8", size_kb: 1000.0}), '
        '(:XVariant {id: "MB32", precision: "fp32", size_kb: 8000.0}), '
        '(:XVariant {id: "MB8",  precision: "int8", size_kb: 2000.0}), '
        '(:XDeploy {id: "D1", fits: 0}), (:XDeploy {id: "D2", fits: 1}), '
        '(:XDeploy {id: "D3", fits: 0}), (:XDeploy {id: "D4", fits: 1})', g)
    for deploy, variant in (("D1", "MA32"), ("D2", "MA8"), ("D3", "MB32"), ("D4", "MB8")):
        client.query(f'MATCH (d:XDeploy), (b:XBoard) WHERE d.id = "{deploy}" '
                     f'AND b.id = "B1" CREATE (d)-[:X_ON]->(b)', g)
        client.query(f'MATCH (d:XDeploy), (v:XVariant) WHERE d.id = "{deploy}" '
                     f'AND v.id = "{variant}" CREATE (d)-[:X_OF]->(v)', g)
    for variant, model in (("MA32", "MA"), ("MA8", "MA"), ("MB32", "MB"), ("MB8", "MB")):
        client.query(f'MATCH (v:XVariant), (m:XModel) WHERE v.id = "{variant}" '
                     f'AND m.id = "{model}" CREATE (v)-[:X_VO]->(m)', g)

    result = client.query("""
MATCH (m:XModel)<-[:X_VO]-(v:XVariant)<-[:X_OF]-(d:XDeploy)-[:X_ON]->(b:XBoard)
WITH m.id AS model, b.id AS board,
     sum(CASE WHEN v.precision = "fp32" AND d.fits = 0 THEN 1 ELSE 0 END) AS fp32_misses,
     sum(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN 1 ELSE 0 END) AS int8_hits,
     max(CASE WHEN v.precision = "fp32" THEN v.size_kb ELSE 0 END) AS fp32_kb,
     max(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 0 END) AS int8_kb
WHERE fp32_misses > 0 AND int8_hits > 0
RETURN model, board, fp32_kb, int8_kb
ORDER BY model
""", g)

    got = {r[0]: (r[2], r[3]) for r in result.records}
    assert got == {"MA": (4000.0, 1000.0), "MB": (8000.0, 2000.0)}, (
        f"expected each model paired with its own variants, got {result.records}"
    )


@pytest.mark.parametrize("qid", list(BY_ID))
def test_order_by_is_actually_applied(loaded, qid):
    """ORDER BY on a RETURN-introduced alias is silently ignored on v1.7.0, and
    only the first sort key is honoured. Every catalog query must therefore
    project through WITH and sort on a single key -- assert it really sorts.

    Parametrised for the same reason as the sweep above: excusing the whole test
    for a single query would excuse the other fifteen queries' sort order too.
    That is what kept EA01 and EA02's note-10 marks from hiding anything, and it
    is why the parametrisation stays now that the marks are gone.
    """
    import re

    cypher = BY_ID[qid]["cypher"].strip()
    match = re.search(r"ORDER BY\s+(.+?)(?:\s+LIMIT|\s*$)", cypher, re.DOTALL)
    if not match:
        pytest.skip(f"{qid} has no ORDER BY")
    keys = [k.strip() for k in match.group(1).split(",")]
    assert len(keys) == 1, f"{qid}: multi-key ORDER BY is not honoured by the engine"

    client, _ = loaded
    cols, recs = rows(client, cypher)
    key = keys[0].split()[0]
    descending = keys[0].lower().endswith("desc")
    values = [r[cols.index(key)] for r in recs]
    assert values == sorted(values, reverse=descending), f"ORDER BY not applied for {qid} ({key})"
