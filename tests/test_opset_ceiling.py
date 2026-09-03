"""`opset_ceiling` is a hard cut on kernel existence, so the graph is held to it.

`Accelerator.opset_ceiling` sat on the node the hero question turns on, with
nothing saying what it meant operationally (#25). `etl/generate.py` answers it:
a `Kernel` is created only when, among other conditions,

    if o.since_version > a["opset_ceiling"]:
        continue

so it is a rule, not a hint. `docs/schema.md` now states it; this file holds the
graph to it.

**`since_version` is the operator's latest revision, not its introduction.**
`etl/onnx_catalog.py` sets it to `max(versions)`, so `Pad` -- introduced at
opset 1 and revised ten times through 25 -- carries 25 and is excluded from
every band below that. The rule is therefore "no kernel for an operator whose
*latest revision* is above the ceiling", which excludes a great many old
operators and is easy to read backwards.

**`99` is a sentinel, not a ceiling.** `MCU-CPU` carries it and no ONNX operator
reaches it, so it excludes nothing -- which is what makes the MCU-CPU the
universal fallback, and why the hero question always has an answer. It is
derived from `ACCEL_ARCHETYPES` rather than restated here, and skipped wherever
a real band is meant.

**The real layer has no ceiling at all.** Real accelerators carry none, because
their kernel registrations are read from upstream rather than derived from a
rule, so their placements are outside the invariant: not violating it, not
covered by it.

`tests/test_generate.py::test_kernels_respect_opset_ceiling` already asserts the
core rule on the `Fleet`, with no engine and no downloaded data. That is the
stronger place for it and the two should change together. What this file adds is
the load round-trip, the per-band form the generator actually applies, and the
anti-vacuity and provenance facts above.

Comparisons run in Python rather than in a `WHERE` clause, following
`tests/test_correctness.py`. It also sidesteps a `Type error: AND requires
boolean operands` this shape raised on the embedded build with a null ceiling in
play -- not reduced to a minimal case, so avoided rather than relied upon.
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

# Derived, not restated: if MCU-CPU's ceiling changes in ACCEL_ARCHETYPES, the
# sentinel follows instead of this file quietly treating it as a real band.
ARCHETYPES = {kind: (cats, ceiling)
              for kind, cats, ceiling, *_ in gen.ACCEL_ARCHETYPES}
NO_CEILING_SENTINEL = ARCHETYPES["MCU-CPU"][1]


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


def placements(client):
    """(kernel id, operator since_version, accelerator ceiling) per placement.

    One linear pattern through `Kernel`: engine note 1 is that a variable
    re-bound in a trailing position of a second `MATCH` is silently not joined.
    """
    return client.query(
        "MATCH (op:Operator)<-[:IMPLEMENTS]-(k:Kernel)-[:RUNS_ON]->(a:Accelerator) "
        "RETURN k.id, op.since_version, a.opset_ceiling", GRAPH).records


def accelerators(client):
    return client.query(
        "MATCH (a:Accelerator) RETURN a.id, a.opset_ceiling, a.provenance, a.kind",
        GRAPH).records


def test_no_kernel_exceeds_its_accelerators_ceiling(loaded):
    """The rule `docs/schema.md` states, checked after the load round-trip.

    `tests/test_generate.py::test_kernels_respect_opset_ceiling` already asserts
    this on the `Fleet`, with no engine and no downloaded data, and that is the
    stronger place for the generator's own invariant. What this adds is that the
    round-trip preserves it: Fleet -> batched Cypher -> graph -> query. The two
    should be changed together.
    """
    over = [(kid, since, ceiling) for kid, since, ceiling in placements(loaded)
            if ceiling is not None and since is not None and since > ceiling]
    unversioned = [kid for kid, since, _ in placements(loaded) if since is None]
    assert not unversioned, (
        f"{len(unversioned)} kernels implement an operator with no "
        f"since_version, e.g. {unversioned[:3]}; the comparison above would "
        f"raise TypeError rather than assert"
    )
    assert not over, (
        f"{len(over)} kernels implement an operator above their accelerator's "
        f"opset_ceiling, e.g. {over[:5]}. The generator excludes these, so the "
        f"graph disagreeing with it means one of the two has changed."
    )


def operators(client):
    return client.query(
        "MATCH (op:Operator) RETURN op.name, op.category, op.since_version",
        GRAPH).records


def test_each_band_excludes_operators_it_otherwise_covers(loaded):
    """Anti-vacuity, per archetype and category-aware.

    An earlier version compared each ceiling against *every* operator in the
    graph, which does not establish what it claims: an archetype only ever sees
    operators in its own categories (`etl/generate.py`), so if everything above
    a ceiling sat in categories that archetype does not cover, the ceiling check
    could be deleted from the generator with no effect and the test would still
    pass -- the exact failure it exists to prevent.

    Scoped to the covering archetype's categories, it establishes the thing:
    each band has operators it covers *and* excludes, so removing the rule would
    change the graph.
    """
    ops = [(name, cat, sv) for name, cat, sv in operators(loaded) if sv is not None]
    assert ops, "no operators loaded"

    for kind, (categories, ceiling) in sorted(ARCHETYPES.items()):
        if ceiling == NO_CEILING_SENTINEL:
            continue
        covered = [o for o in ops if o[1] in categories]
        above = [o for o in covered if o[2] > ceiling]
        assert covered, f"{kind} covers no operator category present in the graph"
        assert above, (
            f"{kind} (ceiling {ceiling}) covers {len(covered)} operators and "
            f"excludes none of them, so the ceiling changes nothing for this "
            f"band and the invariant holds vacuously here"
        )


def test_no_kernel_exists_for_an_operator_its_band_excludes(loaded):
    """The rule stated per band, which is how the generator applies it.

    Deliberately not the exact-equality form this file used to carry -- that
    required some operator to sit *exactly* on 13, 17, 19 and 21, and
    `since_version` is `max(versions)`, so an upstream revision moving the
    single operator holding a boundary would fail this for ONNX's reasons
    rather than ours.
    """
    excluded_by_band = {}
    for kind, (categories, ceiling) in ARCHETYPES.items():
        if ceiling == NO_CEILING_SENTINEL:
            continue
        excluded_by_band[kind] = {
            name for name, cat, sv in operators(loaded)
            if sv is not None and cat in categories and sv > ceiling
        }

    kind_of = {aid: kind for aid, _, _, kind in accelerators(loaded)}
    violations = []
    for name, kind in loaded.query(
            "MATCH (op:Operator)<-[:IMPLEMENTS]-(k:Kernel)-[:RUNS_ON]->(a:Accelerator) "
            "RETURN op.name, a.id", GRAPH).records:
        band = kind_of.get(kind)
        if band in excluded_by_band and name in excluded_by_band[band]:
            violations.append((name, band))
    assert not violations, (
        f"{len(violations)} kernels implement an operator their band excludes, "
        f"e.g. {violations[:5]}"
    )


def test_every_band_present_in_the_graph_is_actually_checked(loaded):
    """A band with no kernels is a band nobody checked.

    At `SCALE = 0.3` each SoC draws its extra accelerator kinds at random, so an
    archetype can end up with no placements. If that happens the assertions
    above skip it silently, which is worth knowing rather than passing.
    """
    kinds_with_accelerators = {kind for _, ceiling, _, kind in accelerators(loaded)
                               if ceiling is not None}
    kinds_with_kernels = {kind for kind, in loaded.query(
        "MATCH (k:Kernel)-[:RUNS_ON]->(a:Accelerator) RETURN a.kind", GRAPH).records}
    unchecked = kinds_with_accelerators - kinds_with_kernels
    assert not unchecked, (
        f"archetypes present in the graph but carrying no kernels, so their "
        f"ceiling is never exercised: {sorted(unchecked)}"
    )


def test_real_accelerators_carry_no_ceiling(loaded):
    """The documented gap, pinned.

    Real kernel registrations come from upstream rather than a rule, so real
    accelerators have no ceiling and their placements sit outside the invariant.
    A query filtering on `opset_ceiling` drops them silently, which
    `docs/schema.md` says and this keeps true.

    How many real accelerators there are is not asserted: `add_ort_layer`
    derives them from the device ids present in the dump and `add_mlperf_layer`
    from the distinct accelerator strings, so refreshing either changes the
    count. What is asserted is that whatever is real lacks a ceiling.
    """
    by_provenance = defaultdict(lambda: [0, 0])
    for _, ceiling, provenance, _ in accelerators(loaded):
        by_provenance[provenance][0] += 1
        if ceiling is None:
            by_provenance[provenance][1] += 1

    synthetic_total, synthetic_without = by_provenance["synthetic"]
    real_total, real_without = by_provenance["real"]
    assert synthetic_total > 0, (
        f"no generated accelerators in the load, so the check below is vacuous; "
        f"provenance values seen: {sorted(by_provenance)}"
    )
    assert set(by_provenance) <= {"real", "synthetic"}, (
        f"unexpected provenance values, silently uninspected: "
        f"{sorted(set(by_provenance) - {'real', 'synthetic'})}"
    )
    assert synthetic_without == 0, (
        f"{synthetic_without} of {synthetic_total} generated accelerators have "
        f"no opset_ceiling; the generator gives every archetype one"
    )
    assert real_without == real_total and real_total > 0, (
        f"{real_without} of {real_total} real accelerators lack a ceiling; "
        f"docs/schema.md says all of them do"
    )
