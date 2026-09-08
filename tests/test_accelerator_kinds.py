"""`kind` is an open vocabulary, and the archetype table describes only half of it.

`docs/schema.md` gives five accelerator archetypes and #26 read the `Opset
ceiling` column as a count -- `99 + 17 + 13 + 19 + 21 = 169` against an
`Accelerator` count of 85. It is not a count. `99` is `MCU-CPU`'s ceiling
sentinel, documented in the section #25 added.

The real disagreement the issue found is the one underneath: **85 and 91 are
both right, for different layers**, and neither page said which it meant. The
generated layer holds 85 accelerators across the five archetypes; loading the
real layer as well makes it 91.

## The finding

The six extra are not more of the same five kinds. They carry **four kinds that
have no archetype at all**:

    CPU  GPU-CUDA  GPU-DirectML  NPU

`etl/real_layer.py` names them from `ORT_DEVICES` and the MLPerf accelerator
strings, so `kind` is not the closed set of five the table implies -- and
`ORT_DEVICES` also holds `GPU-ROCm` and `GPU-TensorRT`, which appear the moment
a dump mentions them. Having no archetype, they carry no `opset_ceiling`, no
category list, no `gops_int8` and no `energy_factor`.

**This changed catalog answers, and #69 fixed it.** `EA11` asks which models
fall back to the CPU. It used to filter `WHERE a.kind <> "MCU-CPU"`, so ONNX
Runtime's CPU execution provider -- spelled `CPU` -- did not match and its
kernels counted as *acceleration*. Measured at `--scale 1.0`, seed `20260814`:

| models EA11 reports CPU-only | generated only | both layers |
|---|---:|---:|
| old rule, `kind <> "MCU-CPU"` | 60 | **12** |
| new rule, `is_cpu_fallback = 0` | 60 | **20** |

The generated layer is unchanged, because `MCU-CPU` is its only CPU. The real
layer recovers 8 models that the spelling had hidden. Not all 60: ONNX Runtime's
CUDA and DirectML providers are genuine accelerators, so operators they
implement are correctly not CPU-only.

`etl/generate.py` and `etl/real_layer.py` now set `is_cpu_fallback` explicitly
rather than letting the catalog infer it from a string, so a new `ORT_DEVICES`
entry is classified where it is created. This file pins the vocabulary so the
next kind cannot arrive silently, and that the flag matches it.

Counts are asserted as *relationships* rather than figures. The published 85 and
91 are `--scale 1.0`; this runs smaller, and the facts worth holding -- that the
per-kind counts sum to the total, that the generated layer uses exactly the
archetype kinds, and that the real layer uses none of them -- are true at every
scale. `tests/test_opset_ceiling.py` makes the same choice for the same reason.
"""
from collections import Counter

import pytest

from etl import generate as gen
from etl import onnx_catalog as oc

SEED = 4242
SCALE = 0.3

# Derived, so a sixth archetype is picked up here rather than silently ignored.
ARCHETYPE_KINDS = {kind for kind, *_ in gen.ACCEL_ARCHETYPES}

# Every kind `etl/real_layer.py` can mint: the ORT_DEVICES table plus the "NPU"
# it gives MLPerf submitters. Listed in full rather than as the subset today's
# dumps happen to use -- a dump mentioning ROCm should not fail this test, but a
# genuinely new spelling should.
REAL_KINDS = {"CPU", "GPU-CUDA", "GPU-DirectML", "GPU-ROCm", "GPU-TensorRT", "NPU"}


@pytest.fixture(scope="module")
def fleets():
    """The generated fleet, and the same fleet with the real layer added."""
    from etl import real_layer
    # `build_real` calls `ort.load_cached()` and `tiny.load_cached()` of its own,
    # so guarding only `oc.load_cached()` would turn a partial `data/` into an
    # error instead of a skip.
    try:
        ops = oc.load_cached()
        generated = gen.generate(seed=SEED, scale=SCALE, operators=ops)
        # Snapshot before build_real mutates the same Fleet in place.
        generated_accelerators = [dict(a) for a in generated.nodes["Accelerator"]]
        real_layer.build_real(generated, ops)
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    return generated_accelerators, generated.nodes["Accelerator"]


def test_the_generated_layer_uses_exactly_the_archetype_kinds(fleets):
    """The table's five kinds, and no others, in the layer it describes.

    That all five *appear* at `SCALE = 0.3` is the pinned `SEED` doing work:
    `etl/generate.py` draws each SoC's extra accelerator kinds at random, so a
    smaller fleet on another seed could omit one and this would fail for the
    sampling rather than for a documentation drift. The equality is still the
    right assertion -- a sixth archetype must fail here -- but read a failure
    naming a *missing* kind as "check the seed and scale" first.
    """
    generated, _ = fleets
    kinds = {a["kind"] for a in generated}
    assert kinds == ARCHETYPE_KINDS, (
        f"generated accelerators use kinds the archetype table does not list: "
        f"extra {sorted(kinds - ARCHETYPE_KINDS)}, "
        f"missing {sorted(ARCHETYPE_KINDS - kinds)}. docs/schema.md documents "
        f"one row per archetype, so the table needs a row added or removed."
    )


def test_per_kind_counts_sum_to_the_accelerator_count(fleets):
    """#26's arithmetic, on the column that is actually a count.

    The issue summed the `Opset ceiling` column to 169 against 85. Summing the
    counts is the check it was reaching for, and it holds per layer.
    """
    generated, both = fleets
    for label, rows in (("generated layer", generated), ("both layers", both)):
        counts = Counter(a["kind"] for a in rows)
        assert sum(counts.values()) == len(rows), (
            f"{label}: per-kind counts sum to {sum(counts.values())} against "
            f"{len(rows)} Accelerator nodes"
        )
    assert len(both) > len(generated), (
        "the real layer added no accelerators, so the 85-vs-91 split this test "
        "documents no longer exists"
    )


def test_the_real_layer_introduces_kinds_that_have_no_archetype(fleets):
    """The finding. If this ever passes trivially, the docs are stale.

    Not asserted as a fixed set: `add_ort_layer` names kinds from the device ids
    a dump happens to mention, so today's three ORT kinds could become five. What
    is asserted is that they are all known spellings and that none of them
    collides with an archetype -- a real accelerator answering to `MCU-CPU`
    would quietly join the generated fleet in every query that groups by kind.
    """
    _, both = fleets
    real = [a for a in both if a.get("provenance") == "real"]
    assert real, "no real accelerators loaded, so this test proves nothing"

    real_kinds = {a["kind"] for a in real}
    assert real_kinds, "real accelerators carry no kind at all"
    # Checked before the vocabulary assertion below: an archetype kind is also
    # an unrecognised one, and this is the diagnosis worth reading first.
    assert not (real_kinds & ARCHETYPE_KINDS), (
        f"a real accelerator now answers to an archetype kind: "
        f"{sorted(real_kinds & ARCHETYPE_KINDS)}. Everything grouping by kind "
        f"-- EA05, EA08, EA11 -- would silently mix the two layers."
    )
    assert real_kinds <= REAL_KINDS, (
        f"etl/real_layer.py minted an unrecognised kind: "
        f"{sorted(real_kinds - REAL_KINDS)}. docs/schema.md lists which kinds "
        f"exist outside the archetypes; add it there and here."
    )


def test_kinds_without_an_archetype_carry_no_archetype_properties(fleets):
    """No ceiling, no GOPS, no energy factor -- so no query may assume them."""
    _, both = fleets
    real = [a for a in both if a.get("provenance") == "real"]
    assert real, "no real accelerators loaded, so this test proves nothing"
    derived = ("opset_ceiling", "gops_int8", "energy_factor")
    offenders = [
        (a["id"], a["kind"], {p: a[p] for p in derived if a.get(p) is not None})
        for a in both
        if a.get("provenance") == "real"
        and any(a.get(p) is not None for p in derived)
    ]
    assert not offenders, (
        f"real accelerators carrying generator-derived properties: {offenders}. "
        f"These come from an archetype, and kinds outside the table have none."
    )


def test_every_cpu_is_marked_as_a_fallback_target(fleets):
    """The fix for #69, pinned in the direction that matters.

    `EA11` used to filter `WHERE a.kind <> "MCU-CPU"`, so ONNX Runtime's CPU
    execution provider -- spelled `CPU` -- passed the filter and its kernels
    counted as *acceleration*. Loading the real layer dropped the models EA11
    reported as CPU-only from 60 to 12.

    It now filters `WHERE a.is_cpu_fallback = 0`, an explicit property set in
    both layers, and the figure is 20. Not 60: ONNX Runtime's CUDA and DirectML
    providers are genuine accelerators, so an operator they implement is
    correctly not CPU-only.

    Asserted as a property of every CPU rather than as a row count, because the
    counts are scale- and dump-dependent while the classification is ours. The
    previous version of this test pinned the *old* behaviour -- that a real CPU
    provider does not match the filter -- which is exactly the assertion #69
    exists to overturn.
    """
    _, both = fleets

    # Checked first, and over *every* accelerator: EA11 filters
    # `is_cpu_fallback = 0`, so a missing flag is null, does not equal 0, and is
    # therefore treated as a fallback target -- its kernels stop counting as
    # acceleration and the models depending on it are wrongly reported CPU-only.
    # The two assertions below cannot see that: `a.get(...) == 1` is False for a
    # null, so a flagless GPU passes both halves. A fourth creation site added
    # tomorrow would land nulls with the suite still green.
    missing = [(a["id"], a["kind"]) for a in both
               if a.get("is_cpu_fallback") not in (0, 1)]
    assert not missing, (
        f"accelerators with no usable is_cpu_fallback: {missing}. EA11 filters "
        f"`is_cpu_fallback = 0`, so null is counted as a CPU and every model "
        f"depending on these is wrongly reported CPU-only."
    )

    # Derived from the source tables rather than the literal `"CPU"`: an
    # `ORT_DEVICES` entry spelled `CPU-OpenVINO` is still a CPU, and a test that
    # partitions on the spelling would miss it in the same way the old query did.
    from etl.real_layer import ORT_DEVICES
    cpu_kinds = {"MCU-CPU"} | {kind for _n, kind, fb in ORT_DEVICES.values() if fb}
    cpus = [a for a in both if a["kind"] in cpu_kinds]
    assert cpus, (
        "no accelerator of kind CPU or MCU-CPU at all; EA11's fallback question "
        "has no target and this test proves nothing"
    )
    not_marked = [(a["id"], a["kind"]) for a in cpus if a.get("is_cpu_fallback") != 1]
    assert not not_marked, (
        f"CPUs not marked as fallback targets: {not_marked}. EA11 filters on "
        f"`is_cpu_fallback`, so an unmarked CPU is counted as acceleration and "
        f"the models depending on it silently disappear from the answer."
    )
    accelerators = [a for a in both if a["kind"] not in cpu_kinds]
    wrongly_marked = [(a["id"], a["kind"]) for a in accelerators
                      if a.get("is_cpu_fallback") == 1]
    assert not wrongly_marked, (
        f"non-CPU accelerators marked as fallback targets: {wrongly_marked}. "
        f"That would report models as CPU-only when they are accelerated."
    )


def test_the_catalog_no_longer_infers_cpu_from_the_kind_spelling(fleets):
    """The mechanism, not just the data -- so the query cannot regress quietly.

    A future edit restoring `kind <> "MCU-CPU"` would pass every other test
    here: the flag would still be correct, it would simply stop being consulted.
    """
    from benchmarks.queries import BY_ID
    cypher = BY_ID["EA11"]["cypher"]
    assert "is_cpu_fallback" in cypher, (
        "EA11 no longer filters on `is_cpu_fallback`. If it has gone back to "
        "comparing `kind`, ONNX Runtime's CPU provider counts as acceleration "
        "again (#69)."
    )
    assert '<> "MCU-CPU"' not in cypher, (
        'EA11 compares `kind <> "MCU-CPU"` again. That treats any CPU not '
        'spelled MCU-CPU as an accelerator, and on this engine `<>` also '
        "matches a null kind (engine note 8b)."
    )


def test_the_covers_column_matches_the_archetypes_it_describes():
    """`docs/schema.md`'s `Covers` cells, against `ACCEL_ARCHETYPES`.

    Added because the column was wrong. It used to be written incrementally --
    `NPU-Pro` as "+ reduction, attention, shape" over the row above -- which
    assumes the category sets form a subset chain. They do not: `DSP` carries
    `signal`, which no row below it has, and `NPU-Lite` drops `reduction` and
    `shape` that `DSP` holds. Four of the five rows understated themselves and
    nothing noticed, because no test read the table.

    Parses the document rather than the graph, so it needs no engine and no
    downloaded data.
    """
    import re
    from pathlib import Path

    doc = (Path(__file__).resolve().parent.parent / "docs" / "schema.md").read_text()
    section = re.search(r"^## Accelerator archetypes$(.*?)^### ", doc,
                        re.DOTALL | re.MULTILINE)
    assert section, "no `## Accelerator archetypes` section in docs/schema.md"

    documented = {}
    for row in re.finditer(r"^\|\s*`(\w[\w-]*)`\s*\|[^|]*\|[^|]*\|([^|]*)\|",
                           section.group(1), re.MULTILINE):
        documented[row.group(1)] = row.group(2).strip()

    actual = {kind: set(cats) for kind, cats, *_ in gen.ACCEL_ARCHETYPES}
    assert set(documented) == set(actual), (
        f"the archetype table and ACCEL_ARCHETYPES disagree on which kinds "
        f"exist: only in docs {sorted(set(documented) - set(actual))}, "
        f"only in code {sorted(set(actual) - set(documented))}"
    )

    for kind, cell in documented.items():
        if "every category" in cell:                 # MCU-CPU's shorthand
            assert actual[kind] == max(actual.values(), key=len), (
                f"{kind} is described as covering every category but does not "
                f"hold the largest set"
            )
            continue
        assert not cell.startswith("+"), (
            f"{kind}'s Covers cell is incremental ('{cell}'). The category sets "
            f"are not a subset chain, so '+' cannot be read against the row "
            f"above -- spell the categories out."
        )
        listed = {c.strip() for c in cell.split(",") if c.strip()}
        assert listed == actual[kind], (
            f"{kind}'s Covers cell disagrees with ACCEL_ARCHETYPES: "
            f"missing {sorted(actual[kind] - listed)}, "
            f"not in the archetype {sorted(listed - actual[kind])}"
        )
