"""A committable record of what a build contains, so two builds can be diffed.

`--seed` and `--scale` change the world and the README promises "same seed, same
graph, every time". That is determinism. The missing half is what happens when
the seed or the generator *does* change: nothing said which figures moved (#29).

This repo quotes counts in five documents. When they drifted, nothing noticed
until a test was written to compare them against a rebuild -- and even then the
failure said *that* a number was wrong, not *which regeneration* moved it. A
committed manifest makes the move itself reviewable:

    python -m etl.manifest --write     # regenerate docs/build-manifest.json
    python -m etl.manifest --check     # exit 1 with a diff if it has moved
    git diff docs/build-manifest.json  # exactly which counts changed

## Why a file rather than a command that prints

Because the point is the *diff*. A printed table tells you today's numbers; a
committed file tells you what changed and puts it in the review, next to the
generator change that caused it. The documents quoting those counts can then be
updated from the diff rather than from memory.

## The generated layer is not ours alone

The obvious framing -- generated is deterministic from the seed, so a mismatch
means the generator changed -- is **wrong**, and the manifest was built on it at
first. `etl/generate.py` creates Kernels from the ONNX operator catalogue, and
`data/` is gitignored, so each checkout holds whatever it last fetched. The
catalogue is pinned to `etl.onnx_catalog.ONNX_REF` now, which makes a fresh
build reproducible -- but a cache predating the pin, or a deliberate move of
it, still moves `Kernel` and every total derived from it while nothing in this
repo changed.

So the manifest records the catalogue's **content fingerprint** alongside `seed`
and `scale`, and `--check` reports an upstream move as its own case rather than
advising a re-baseline. Without that, the diff cannot answer the one question it
exists for: did *we* change, or did upstream?

## What is deliberately not in it

**No timings and no environment.** Those move for reasons unrelated to the graph
-- a slower laptop, a different Python -- and a manifest that churns on noise is
one people regenerate without reading. The only non-count recorded is the
fingerprint of an input the counts genuinely depend on.

**The real layer is recorded but not pinned by a test.** ONNX Runtime and MLPerf
publish on their own schedule; 734 kernel registrations became 738 during one
week of this backlog. `--check` reports those moves so a person can see them.
`tests/test_build_manifest.py` asserts the generated half **only when the
catalogue fingerprint matches**, and skips otherwise -- because with a different
input it is not testing what its name says.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path

import click

from etl import generate as gen
from etl.loader import NODE_LABELS

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "docs" / "build-manifest.json"


def catalogue_fingerprint(operators) -> dict:
    """Identify the ONNX operator catalogue this build was derived from.

    The generated layer is deterministic from the seed **given the same
    operators** -- `etl/generate.py` builds Kernels from this catalogue. The
    pin (`etl.onnx_catalog.ONNX_REF`) makes that hold for a fresh build; it
    does not hold for a `data/` cached before the pin, and it stops holding
    the moment the pin is moved. Without this, a diff cannot tell "our
    generator changed" from "the catalogue did", which is the one distinction
    this file exists to make.

    A content fingerprint rather than a version: the cache records
    `license`, `operator_count`, `operators` and `source`, and no version,
    commit or fetch date to read.

    **Every field, not the ones the generator reads.** An earlier version
    fingerprinted `(id, since_version, category)`, which were the fields
    `etl/generate.py` was known to consult. That is the wrong coupling: which
    fields are load-bearing is a property of `generate.py`, invisible from here,
    and free to change without anything failing. It had already drifted --
    `generate.py:370` gates kernel creation on `is_control_flow`, which was not
    in the material, so a catalogue that flipped it would move `Kernel` and
    every total derived from it while the fingerprint stayed byte-identical.

    That is worse than recording no fingerprint at all. `--check` would report
    *"the upstream input is unchanged, so the generator changed"* and advise a
    re-baseline, and someone would rewrite five documents' published figures to
    an upstream move believing it was theirs. A missing fingerprint asserts no
    cause; a wrong one asserts the wrong cause confidently.

    Hashing the whole record removes the coupling instead of tracking it. The
    cost is fingerprint churn on fields the generator ignores today
    (`version_count`, `name`, `domain`) -- which is the safe direction: it
    over-reports upstream movement, and over-reporting sends a human to look
    rather than sending them to re-baseline.
    """
    material = "\n".join(
        "\t".join(str(v) for v in dataclasses.astuple(o))
        for o in sorted(operators, key=lambda o: o.id))
    return {
        "method": FINGERPRINT_METHOD,
        "operator_count": len(operators),
        "fingerprint": "sha256:" + hashlib.sha256(material.encode()).hexdigest()[:16],
    }


# Bump when `catalogue_fingerprint`'s material changes.
#
# Recorded because a change of *method* is otherwise indistinguishable from a
# change of *input*: both are "the hash differs". Widening the material from
# three fields to all seven -- the fix for the review on #92 -- moved the
# fingerprint on an identical 205-operator catalogue, and `--check` duly
# announced "THE UPSTREAM INPUT MOVED", which was false. Without this the same
# false report would recur on every future change to the material, and the one
# distinction this module exists to make would be wrong in a third way.
FINGERPRINT_METHOD = "sha256/all-fields/1"


def build_manifest(seed: int, scale: float) -> dict:
    """Per-label node counts and per-type edge counts, generated and real.

    Both layers are built from one `Fleet` so the real figures are the *delta*
    the real layer adds, which is what the documents' `N (+M real)` convention
    means. Building them separately would report the real layer standalone --
    a different number, and the one that already confused two pages.
    """
    from etl import onnx_catalog as oc
    from etl import real_layer

    ops = oc.load_cached()
    fleet = gen.generate(seed=seed, scale=scale, operators=ops)
    generated_nodes = {label: len(rows) for label, rows in fleet.nodes.items() if rows}
    generated_edges = Counter(rel for _s, _si, rel, _t, _ti, _p in fleet.edges)
    generated_totals = (fleet.node_count, fleet.edge_count)

    real_layer.build_real(fleet, ops)
    both_nodes = {label: len(rows) for label, rows in fleet.nodes.items() if rows}
    both_edges = Counter(rel for _s, _si, rel, _t, _ti, _p in fleet.edges)

    def delta(both: dict, generated: dict) -> dict:
        return {k: both[k] - generated.get(k, 0)
                for k in sorted(both) if both[k] - generated.get(k, 0)}

    return {
        "seed": seed,
        "scale": scale,
        # The upstream input the generated layer depends on. Recorded so a diff
        # can say which of the two things moved.
        "inputs": {"onnx_catalogue": catalogue_fingerprint(ops)},
        "nodes": {
            "generated": {k: generated_nodes[k] for k in sorted(generated_nodes)},
            "added_by_real_layer": delta(both_nodes, generated_nodes),
            "generated_total": generated_totals[0],
            "both_layers_total": fleet.node_count,
        },
        "edges": {
            "generated": {k: generated_edges[k] for k in sorted(generated_edges)},
            "added_by_real_layer": delta(both_edges, generated_edges),
            "generated_total": generated_totals[1],
            "both_layers_total": fleet.edge_count,
        },
        # Recorded so a label appearing or vanishing is visible in the diff even
        # when its count is zero -- `BenchmarkTask` is generated-only-empty and
        # would otherwise be absent from both maps.
        "labels_declared": sorted(NODE_LABELS),
    }


def render(manifest: dict) -> str:
    """Stable text: sorted keys, two-space indent, one trailing newline.

    Formatting is part of the contract. A manifest whose key order or spacing
    wobbles produces diff noise that hides the counts, which is the one thing it
    exists to show.
    """
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def differences(old: dict, new: dict, path: str = "") -> list[str]:
    """Human-readable lines describing what moved, deepest key first."""
    out: list[str] = []
    for key in sorted(set(old) | set(new)):
        here = f"{path}.{key}" if path else key
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out.extend(differences(a, b, here))
        elif a != b:
            if a is None:
                out.append(f"  + {here}: {b}")
            elif b is None:
                out.append(f"  - {here}: {a} (gone)")
            else:
                sign = ""
                if isinstance(a, int) and isinstance(b, int):
                    sign = f"  ({b - a:+d})"
                out.append(f"  ~ {here}: {a} -> {b}{sign}")
    return out


@click.command()
@click.option("--write", is_flag=True, help="Regenerate the manifest on disk.")
@click.option("--check", is_flag=True,
              help="Exit 1 and print a diff if the manifest has moved.")
@click.option("--seed", default=gen.DEFAULT_SEED, show_default=True, type=int)
@click.option("--scale", default=1.0, show_default=True, type=float)
def main(write, check, seed, scale):
    if write == check:
        raise SystemExit("give exactly one of --write or --check")
    try:
        current = build_manifest(seed, scale)
    except FileNotFoundError:
        raise SystemExit("run `python -m etl.download_data` first") from None

    if write:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(render(current), encoding="utf-8")
        click.echo(f"wrote {MANIFEST_PATH.relative_to(MANIFEST_PATH.parent.parent)}")
        click.echo(f"  {current['nodes']['both_layers_total']:,} nodes / "
                   f"{current['edges']['both_layers_total']:,} edges "
                   f"(seed {seed}, scale {scale})")
        click.echo("  commit it, so the next regeneration shows up as a diff")
        return

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} does not exist; run --write first")
    recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # `--check` compares against a manifest built at a *recorded* seed and
    # scale, so being handed different ones is not a comparison -- it is two
    # different worlds, and every count differs. The diff would be enormous,
    # headed by a `~ seed:` line that explains it only if you read to the end.
    # Refuse instead of printing it.
    for name, given, was in (("seed", seed, recorded.get("seed")),
                             ("scale", scale, recorded.get("scale"))):
        if was is not None and given != was:
            raise SystemExit(
                f"--check was given --{name} {given}, but the manifest records "
                f"{name} {was}. Those describe different graphs, so the diff "
                f"would be noise. Drop the flag to check the recorded build, or "
                f"run --write --{name} {given} to re-record at this {name}."
            )

    if recorded == current:
        click.echo(f"manifest matches (seed {seed}, scale {scale})")
        return

    old_cat = (recorded.get("inputs") or {}).get("onnx_catalogue") or {}
    new_cat = (current.get("inputs") or {}).get("onnx_catalogue") or {}
    # A method change moves the hash on an identical catalogue, so it must be
    # ruled out before the difference can be read as upstream movement.
    method_changed = old_cat.get("method") != new_cat.get("method")
    upstream_moved = (not method_changed
                      and recorded.get("inputs") != current.get("inputs"))

    click.echo("the build no longer matches the committed manifest:", err=True)
    for line in differences(recorded, current):
        click.echo(line, err=True)
    click.echo("", err=True)

    if method_changed:
        # Third case, and it must come first: a method change moves the hash on
        # an identical catalogue, so reading it as upstream movement would be
        # wrong in the same way the missing fingerprint was.
        click.echo("  THE FINGERPRINT METHOD CHANGED, so the hashes are not "
                   "comparable.", err=True)
        click.echo(f"    method: {old_cat.get('method') or '(none recorded)'} "
                   f"-> {new_cat.get('method')}", err=True)
        click.echo(f"    operator_count: {old_cat.get('operator_count')} -> "
                   f"{new_cat.get('operator_count')}"
                   + ("  (unchanged, so the catalogue itself probably did not "
                      "move)" if old_cat.get("operator_count")
                      == new_cat.get("operator_count") else ""), err=True)
        click.echo("    Re-record with --write. Any *count* differences above "
                   "are still real and\n    still need reading -- only the "
                   "fingerprint line is explained by this.", err=True)
    elif upstream_moved:
        # Reported as its own case. Telling someone to `--write` here would
        # re-baseline to whatever upstream happened to be today and rewrite
        # published figures that were correct for the build they describe.
        old = (recorded.get("inputs") or {}).get("onnx_catalogue", {})
        new = (current.get("inputs") or {}).get("onnx_catalogue", {})
        click.echo("  THE UPSTREAM INPUT MOVED, so this is not necessarily a "
                   "generator change.", err=True)
        click.echo(f"    ONNX operator catalogue: "
                   f"{old.get('operator_count')} operators "
                   f"({old.get('fingerprint')})\n"
                   f"                          -> "
                   f"{new.get('operator_count')} operators "
                   f"({new.get('fingerprint')})", err=True)
        click.echo("    The catalogue is pinned to `etl.onnx_catalog."
                   "ONNX_REF`, so the likeliest cause is a\n"
                   "    `data/` cached before the pin: run `python -m "
                   "etl.download_data --force` first.\n"
                   "    If the pin was moved deliberately, re-baselining with "
                   "--write is correct only\n"
                   "    if you also intend the published figures to follow "
                   "it.", err=True)
    else:
        click.echo("  The upstream input is unchanged, so the generator changed.\n"
                   "  Run `python -m etl.manifest --write` and commit the result "
                   "-- the diff is\n  the list of published figures that need "
                   "updating.", err=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
