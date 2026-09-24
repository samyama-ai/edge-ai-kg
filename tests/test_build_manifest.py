"""The committed manifest still describes the graph the generator produces (#29).

`docs/build-manifest.json` records per-label node counts and per-type edge
counts so two builds can be diffed. It is only worth committing if it is kept
true, and only worth trusting if the check distinguishes *our* drift from
upstream's.

**The generated half is asserted; the real half is not.** The generated layer is
deterministic from the seed, so a mismatch there means the generator changed and
the five documents quoting those figures need updating. The real layer comes
from ONNX Runtime and MLPerf, which publish on their own schedule -- 734 kernel
registrations became 738 during one week of this backlog -- and a test that
fails for upstream's reasons is one people learn to ignore.

`python -m etl.manifest --check` reports both, so a person still sees the
upstream move; only the test is selective.

Needs `data/`, so it skips without it. No engine: everything here is computed
from the `Fleet`.
"""
import json

import pytest

from etl import manifest


@pytest.fixture(scope="module")
def recorded():
    if not manifest.MANIFEST_PATH.exists():
        pytest.fail(
            f"{manifest.MANIFEST_PATH} is missing. It is committed on purpose -- "
            f"run `python -m etl.manifest --write` and commit the result."
        )
    return json.loads(manifest.MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh(recorded):
    try:
        return manifest.build_manifest(recorded["seed"], recorded["scale"])
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


@pytest.fixture(scope="module")
def comparable(fresh, recorded):
    """`fresh`, but only when it was built from the recorded catalogue.

    The skip lives **in a fixture** rather than in the test body, and that is
    the whole point of this fixture existing. `conftest.py` converts only
    *setup-phase* skips under `--no-skips`, so a call-phase skip stays a skip
    in CI -- measured, not assumed: a body skip survives `--no-skips` as
    `1 skipped` while a fixture skip becomes an error. The comparison below is
    the one that pins five documents' published figures, so a checkout whose
    `data/` had drifted would have shown `1 skipped`, gone green, and asserted
    nothing about the manifest at all.

    Still a skip rather than a failure locally, which is the judgement the
    earlier version made and it was right: a red test on a developer's machine
    whose only fault is a fresher `data/` reads as the repo's fault. `--no-skips`
    is what separates the two audiences -- a person sees a skip and a reason,
    CI sees a failure.
    """
    mine = (recorded.get("inputs") or {}).get("onnx_catalogue")
    theirs = (fresh.get("inputs") or {}).get("onnx_catalogue")
    if mine != theirs:
        pytest.skip(
            f"the ONNX operator catalogue moved since the manifest was written "
            f"({mine} -> {theirs}), so the generated layer is not comparable. "
            f"Run `python -m etl.manifest --check` for the full report, and "
            f"--write only if the published figures should follow upstream."
        )
    return fresh


def test_the_generated_layer_matches_the_manifest(comparable, recorded):
    """Deterministic from the seed *given the same operator catalogue*.

    That qualification is the whole of it, and the first version of this test
    lacked it. `etl/generate.py` builds Kernels from the ONNX catalogue and
    `data/` is gitignored, so a refresh moves `Kernel` -- and `generated_total`,
    and everything derived from them -- with nothing in this repo changing.

    Skipped rather than failed when the fingerprint differs: with a different
    input this is not testing what its name says, and a red test on a checkout
    whose only fault is a fresher `data/` reads as the repo's fault. The CLI
    reports the upstream move loudly, which is where a person should see it.
    That skip is raised by the `comparable` fixture, in setup, so `--no-skips`
    turns it into a CI failure -- otherwise a drifted catalogue would skip this
    comparison and leave the suite green.

    Fails with the same per-key diff the CLI prints, so a genuine generator
    change names the published figures that need updating.
    """
    fresh = comparable
    # `both_layers_total` is excluded as well as `added_by_real_layer`: it is
    # generated + real, so pinning it pins the upstream half by the back door.
    # Verified rather than assumed -- simulating an upstream-only move (four
    # more ONNX Runtime kernels) failed this test while it was compared.
    upstream = {"added_by_real_layer", "both_layers_total"}
    for section in ("nodes", "edges"):
        mine = {k: v for k, v in recorded[section].items() if k not in upstream}
        theirs = {k: v for k, v in fresh[section].items() if k not in upstream}
        diff = manifest.differences(mine, theirs, section)
        assert not diff, (
            "the generated layer no longer matches docs/build-manifest.json:\n"
            + "\n".join(diff)
            + "\n\nIf the generator changed on purpose, run "
              "`python -m etl.manifest --write` and commit it -- the diff above "
              "is the list of published figures that need updating."
        )


def test_the_real_layer_is_recorded_but_not_pinned(fresh, recorded):
    """The upstream half must be *present*, so a diff shows it, and not asserted.

    Checking the keys rather than the values: if the real layer stopped adding
    a whole edge type that is a structural change worth failing on, while the
    counts moving is upstream doing its job.
    """
    for section in ("nodes", "edges"):
        mine = set(recorded[section]["added_by_real_layer"])
        theirs = set(fresh[section]["added_by_real_layer"])
        assert mine == theirs, (
            f"the real layer's {section} changed shape: "
            f"no longer adding {sorted(mine - theirs)}, newly adding "
            f"{sorted(theirs - mine)}. Counts moving is upstream publishing; "
            f"a type appearing or vanishing is not -- check "
            f"etl/real_layer.py before regenerating."
        )


def test_the_manifest_is_written_in_the_stable_form(recorded):
    """Byte-for-byte what `render()` produces.

    Formatting is part of the contract: a manifest whose key order or spacing
    wobbles produces diff noise that hides the counts, which is the one thing it
    exists to show. This fails if someone hand-edits it.
    """
    on_disk = manifest.MANIFEST_PATH.read_text(encoding="utf-8")
    assert on_disk == manifest.render(recorded), (
        "docs/build-manifest.json is not in the canonical form. Do not edit it "
        "by hand -- run `python -m etl.manifest --write`."
    )


def test_every_declared_label_appears_somewhere(recorded):
    """A label missing from both maps would be invisible in a diff.

    `BenchmarkTask` is the case: generated-only-empty, so it appears in neither
    count map, and only `labels_declared` records that it exists at all.
    """
    from etl.loader import NODE_LABELS
    assert recorded["labels_declared"] == sorted(NODE_LABELS), (
        f"labels_declared has drifted from the loader's NODE_LABELS: "
        f"only in manifest {sorted(set(recorded['labels_declared']) - set(NODE_LABELS))}, "
        f"only in loader {sorted(set(NODE_LABELS) - set(recorded['labels_declared']))}"
    )


def test_the_totals_agree_with_the_per_key_counts(recorded):
    """Cheap, and it catches a hand-edit that changes one without the other."""
    for section in ("nodes", "edges"):
        block = recorded[section]
        generated = sum(block["generated"].values())
        assert generated == block["generated_total"], (
            f"{section}: per-key generated counts sum to {generated:,} but "
            f"generated_total says {block['generated_total']:,}"
        )
        added = sum(block["added_by_real_layer"].values())
        assert block["generated_total"] + added == block["both_layers_total"], (
            f"{section}: generated {block['generated_total']:,} + real {added:,} "
            f"!= both_layers_total {block['both_layers_total']:,}"
        )


def test_the_manifest_records_the_input_it_was_derived_from(recorded):
    """Without this the diff cannot say which side moved (#29, Tarun's review).

    The generated counts depend on the ONNX operator catalogue, which is
    re-fetched rather than pinned. A manifest recording only `seed` and `scale`
    claims a determinism it does not have.
    """
    catalogue = (recorded.get("inputs") or {}).get("onnx_catalogue")
    assert catalogue, (
        "the manifest records no `inputs.onnx_catalogue`. The generated layer "
        "is built from that catalogue and `data/` is gitignored, so without a "
        "fingerprint a diff cannot distinguish a generator change from an "
        "upstream refresh."
    )
    assert catalogue.get("fingerprint", "").startswith("sha256:"), catalogue
    assert isinstance(catalogue.get("operator_count"), int), catalogue


def test_every_operator_field_moves_the_fingerprint():
    """Changing *any* field of any operator must change the fingerprint.

    This is the guard the first version needed and did not have. It
    fingerprinted `(id, since_version, category)` -- the fields
    `etl/generate.py` was known to read -- and had already missed
    `is_control_flow`, which `generate.py:370` gates kernel creation on. Nothing
    failed, because no test asserted the coupling.

    Which fields the generator consults is not knowable from `manifest.py`, so
    the assertion here is deliberately not "the load-bearing fields are covered"
    -- it is "all of them are". A field added to `Operator` tomorrow is covered
    without anyone remembering this file exists; a fingerprint narrowed back to
    a hand-picked subset fails here.

    The failure mode being prevented is the confident-and-wrong one: an
    unfingerprinted field moves, `--check` reports "the upstream input is
    unchanged, so the generator changed", and someone re-baselines five
    documents' published figures to an upstream refresh.
    """
    import dataclasses

    from etl.onnx_catalog import Operator

    base = Operator(id="op:ai.onnx:conv", name="Conv", domain="ai.onnx",
                    since_version=11, version_count=3, category="convolution",
                    is_control_flow=False)
    original = manifest.catalogue_fingerprint([base])["fingerprint"]

    mutations = {
        "id": "op:ai.onnx:convtranspose",
        "name": "ConvTranspose",
        "domain": "com.microsoft",
        "since_version": 12,
        "version_count": 4,
        "category": "activation",
        "is_control_flow": True,
    }
    fields = [f.name for f in dataclasses.fields(Operator)]
    assert set(mutations) == set(fields), (
        f"`Operator` gained or lost a field: {sorted(set(fields) ^ set(mutations))}. "
        f"Add it to `mutations` with a value different from the base operator, so "
        f"this test keeps covering every field."
    )

    for field, new_value in mutations.items():
        moved = dataclasses.replace(base, **{field: new_value})
        assert manifest.catalogue_fingerprint([moved])["fingerprint"] != original, (
            f"changing `{field}` left the catalogue fingerprint unchanged. "
            f"`--check` would then report an upstream move as a generator change "
            f"and advise re-baselining the published figures."
        )


def test_the_fingerprint_is_stable_across_operator_ordering():
    """Otherwise it moves when upstream reorders a table, which is not a change."""
    import dataclasses

    from etl.onnx_catalog import Operator

    a = Operator(id="op:ai.onnx:abs", name="Abs", domain="ai.onnx",
                 since_version=13, version_count=2, category="elementwise",
                 is_control_flow=False)
    b = dataclasses.replace(a, id="op:ai.onnx:conv", name="Conv",
                            category="convolution")
    assert (manifest.catalogue_fingerprint([a, b])["fingerprint"]
            == manifest.catalogue_fingerprint([b, a])["fingerprint"])


def test_the_drift_skip_is_raised_in_setup_so_no_skips_can_convert_it():
    """The catalogue-drift skip must stay in a fixture, not move into the body.

    `conftest.py` converts only *setup-phase* skips under `--no-skips`. Measured
    rather than reasoned about: with a drifted manifest, a body skip survives
    `--no-skips` as `1 skipped` and the suite goes green, while the same skip
    raised from a fixture becomes an error. So the comparison that pins five
    documents' published figures was skipped silently on any checkout whose
    `data/` had moved -- which is how a stale catalogue produced green suites
    while proving nothing.

    Asserted structurally because the behavioural version costs a scale-1.0
    rebuild in a subprocess for a property two lines of source already fix: the
    test takes the gating fixture, and raises no skip of its own.
    """
    import inspect

    source = inspect.getsource(test_the_generated_layer_matches_the_manifest)
    assert "pytest.skip" not in source, (
        "the catalogue-drift skip has moved back into the test body. "
        "`--no-skips` converts setup-phase skips only, so a body skip leaves "
        "CI green on a drifted catalogue -- raise it from the `comparable` "
        "fixture instead."
    )
    assert "comparable" in inspect.signature(
        test_the_generated_layer_matches_the_manifest).parameters, (
        "the test no longer requests `comparable`, the fixture that gates it "
        "on the recorded catalogue. Without it the comparison runs against a "
        "catalogue the manifest was not written on, and fails for upstream's "
        "reasons rather than the generator's."
    )
    assert "pytest.skip" in inspect.getsource(comparable), (
        "`comparable` no longer skips, so a drifted catalogue would be "
        "compared rather than reported."
    )
