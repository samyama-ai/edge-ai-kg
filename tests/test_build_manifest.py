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


def test_the_generated_layer_matches_the_manifest(fresh, recorded):
    """Ours, deterministic from the seed. A mismatch is a generator change.

    Fails with the same per-key diff the CLI prints, so the failure names the
    published figures that need updating rather than only that something moved.
    """
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
