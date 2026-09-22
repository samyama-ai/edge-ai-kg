"""Reproduce the findings in `docs/vector-search.md`, so a retest is a re-run.

The page records that `samyama` 0.6.1's HNSW index panics on vectors that are
identical or nearly so (#48). A defect record is only useful if someone can
re-run it against a bumped engine and see whether it still holds, so both
measurements behind that page live here rather than in prose.

**That is what happened** -- #56 raised the floor to `samyama` 1.7.1, and on the
build measured there the panic did not reproduce. That result is recorded in
`docs/vector-search.md`, deliberately not here: this module reports what it
observes on the engine you are running and names the 0.6.1 behaviour only as the
contrast. A docstring asserting today's answer would be the same overclaim the
output was just fixed to avoid, one file away from the code that avoids it.

    python -m benchmarks.vector_probe --repro        # the two-call duplicate add (panicked on 0.6.1)
    python -m benchmarks.vector_probe --metrics      # the same, once per metric
    python -m benchmarks.vector_probe --collisions   # the embedding collisions
    python -m benchmarks.vector_probe --holdout      # the decisive dedupe-and-search run

`--repro` and `--metrics` need nothing but the engine. `--collisions` and
`--holdout` both need `data/` -- they build a fleet from the ONNX catalogue.

**The panic is a Rust `PanicException`**, which is not an `Exception` subclass,
so every guard here catches `BaseException`. A plain `except Exception` lets it
through -- which is itself part of the finding.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import click

GRAPH = "default"
DIM = 64


def normalised_random(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    v = [rng.random() for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def two_nodes(client):
    """Two `:T` nodes and their internal integer ids.

    `add_vector` keys on the engine's own node id, not on this repo's string
    `id` property; `id()` is the bridge and `elementId()` does not parse.
    """
    client.query('CREATE (:T {id: "a"}), (:T {id: "b"})', GRAPH)
    return [r[0] for r in client.query("MATCH (t:T) RETURN id(t)", GRAPH).records]


def embed(name: str, dim: int = DIM) -> tuple[float, ...]:
    """Character 3- and 4-gram hashing, L2-normalised.

    Deliberately crude: the point is not embedding quality but that *any*
    embedding of a name catalogue produces near-duplicates, which is what the
    index cannot take.
    """
    s = f"^{name.lower()}$"
    v = [0.0] * dim
    for n in (3, 4):
        for i in range(len(s) - n + 1):
            v[int(hashlib.md5(s[i:i + n].encode()).hexdigest()[:8], 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / norm for x in v)


def run_repro(metric: str = "cosine") -> dict:
    """Add one vector twice, with the phases kept apart.

    `create_vector_index` is timed separately from the adds because the metrics
    table makes two different claims: that an unknown metric is *accepted*, and
    that the duplicate add panics. Catching both in one `try` would report
    "PANIC" for a metric the engine had cleanly rejected, and the table would
    read the same either way.

    The exception *type* is returned rather than asserted, so a reader sees
    `PanicException` instead of taking it on trust -- and so an ordinary
    `ValueError` is visibly not a panic.
    """
    from samyama import SamyamaClient
    client = SamyamaClient.embedded()
    ids = two_nodes(client)
    v = normalised_random(8, seed=11)
    out = {"metric": metric, "index_accepted": None, "outcome": None, "detail": ""}
    try:
        client.create_vector_index("T", "emb", dimensions=8, metric=metric)
        out["index_accepted"] = True
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:           # noqa: BLE001 - PanicException is not Exception
        out["index_accepted"] = False
        out["outcome"] = type(exc).__name__
        out["detail"] = (str(exc).splitlines() or [""])[0]
        return out
    try:
        client.add_vector("T", "emb", ids[0], v)
        client.add_vector("T", "emb", ids[1], v)
        client.vector_search("T", "emb", v, k=2)
        out["outcome"] = "ok"
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:           # noqa: BLE001
        out["outcome"] = type(exc).__name__
        out["detail"] = (str(exc).splitlines() or [""])[0]
    return out


@click.command()
@click.option("--repro", is_flag=True, help="The two-call panic.")
@click.option("--metrics", is_flag=True, help="Every metric, same reproduction.")
@click.option("--collisions", is_flag=True, help="Embedding collisions on the operators.")
@click.option("--holdout", is_flag=True,
              help="The decisive run: dedupe, index, then search until it panics.")
@click.option("--seed", default=20260814, show_default=True, type=int)
@click.option("--scale", default=0.3, show_default=True, type=float)
def main(repro, metrics, collisions, holdout, seed, scale):
    if not (repro or metrics or collisions or holdout):
        raise SystemExit("give --repro, --metrics, --collisions or --holdout")

    if repro:
        r = run_repro()
        click.echo(f"index accepted: {r['index_accepted']}")
        click.echo(f"same normalised vector added twice -> {r['outcome']}: {r['detail']}")
        # Conditional on the outcome, and on whether the index was accepted
        # at all. Printing "fixed as of 1.7.1" unconditionally would declare
        # the panic gone on a run that had just reproduced it -- the probe
        # asserting a version claim over its own measurement, which is the one
        # thing it exists not to do.
        if r["outcome"] == "ok":
            # What this run shows, not what any version does. The module's own
            # docstring refuses to make version claims, and "fixed as of 1.7.1"
            # is one -- it would also be asserted on a run of some other build
            # entirely. `docs/vector-search.md` records the 0.6.1 measurement;
            # this line records only that this build did not reproduce it.
            click.echo("no panic on this build. docs/vector-search.md records "
                       "0.6.1 raising PanicException (assertion failed: "
                       "c.dist_to_ref <= 0.) here; see #56.")
        elif r["outcome"] == "PanicException":
            click.echo("this is the 0.6.1 failure mode -- the same "
                       "PanicException -- reproducing on this build. Check "
                       "`pip show samyama` before reading further.")
        elif r["index_accepted"]:
            # Named for what it is. Calling any exception "the 0.6.1 failure
            # mode" is the mislabelling this probe exists to avoid: a
            # connection error and an HNSW assertion are different findings.
            click.echo(f"the duplicate add failed with {r['outcome']}, which is "
                       f"not the 0.6.1 panic (that is a PanicException). A "
                       f"different fault, and this run says nothing about the "
                       f"one docs/vector-search.md records.")
        else:
            click.echo("the index itself was rejected, so this run says nothing "
                       "about the duplicate-add panic.")

    if metrics:
        # Each metric in its own process. The page records that recoverability
        # after a panic is uninvestigated, so running six in one process would
        # make rows 2-6 order-dependent and arguably meaningless.
        import subprocess
        import sys
        click.echo(f"\n{'metric':<18}{'index accepted':<16}{'add + search'}")
        for metric in ("cosine", "euclidean", "l2", "dot", "inner_product", "manhattan"):
            proc = subprocess.run(
                [sys.executable, "-c",
                 ("import json;from benchmarks.vector_probe import run_repro;"
                  f"print(json.dumps(run_repro({metric!r})))")],
                capture_output=True, text=True, check=False)
            line = [x for x in proc.stdout.splitlines() if x.startswith("{")]
            if not line:
                click.echo(f"{metric:<18}{'?':<16}subprocess died: "
                           f"{(proc.stderr or '').splitlines()[-1][:40]}")
                continue
            r = json.loads(line[-1])
            click.echo(f"{metric:<18}{r['index_accepted']!s:<16}"
                       f"{r['outcome']}  {r['detail'][:34]}")
        click.echo("\nEach row ran in a fresh process, so none is contaminated by "
                   "the previous panic.\n`dot`, `inner_product` and `manhattan` are "
                   "accepted by create_vector_index without\ncomplaint and are not "
                   "documented as supported -- the metric is not validated.")

    if collisions:
        try:
            from etl import generate as gen
            from etl import onnx_catalog as oc
            from etl import real_layer
            ops = oc.load_cached()
            fleet = gen.generate(seed=seed, scale=scale, operators=ops)
            real_layer.build_real(fleet, ops)
        except FileNotFoundError:
            raise SystemExit("run `python -m etl.download_data` first") from None

        rows = fleet.nodes["Operator"]
        distinct_ids = len({r["id"] for r in rows})
        click.echo(f"\nOperator rows: {len(rows)}   distinct ids: {distinct_ids}")
        if distinct_ids != len(rows):
            click.echo("  !! duplicate ids -- the counts below would measure "
                       "fleet assembly, not the embedding")
        else:
            click.echo("  every row is a distinct operator, so a shared name is a "
                       "real name collision\n  across domains, not a repeated row")
        names = [o["name"] for o in rows]
        groups: dict[tuple, list[str]] = {}
        for name in names:
            groups.setdefault(embed(name), []).append(name)
        colliding = [v for v in groups.values() if len(v) > 1]
        click.echo(f"\n{len(names)} operators -> {len(groups)} distinct embeddings")
        click.echo(f"{len(colliding)} collision groups, "
                   f"covering {sum(len(v) for v in colliding)} operators")
        for group in colliding[:3]:
            click.echo(f"  collision: {group}")

        # The hold-out the experiment uses: 40 operators treated as unseen.
        rng = random.Random(7)
        held = set(rng.sample(range(len(names)), min(40, len(names))))
        kept = [n for i, n in enumerate(names) if i not in held]
        kept_groups = {embed(n) for n in kept}
        click.echo(f"\nafter holding out {len(held)} as 'unseen': {len(kept)} indexed "
                   f"-> {len(kept_groups)} distinct embeddings")
        click.echo("(the two distinct-embedding counts in docs/vector-search.md are "
                   "these two:\n all operators, and the subset left after the "
                   "hold-out)")

    if holdout:
        from samyama import SamyamaClient
        try:
            from etl import generate as gen
            from etl import onnx_catalog as oc
            from etl import real_layer
            ops = oc.load_cached()
            fleet = gen.generate(seed=seed, scale=scale, operators=ops)
            real_layer.build_real(fleet, ops)
        except FileNotFoundError:
            raise SystemExit("run `python -m etl.download_data` first") from None

        rows = fleet.nodes["Operator"]
        rng = random.Random(7)
        held = set(rng.sample(range(len(rows)), min(40, len(rows))))
        kept = [r for i, r in enumerate(rows) if i not in held]

        client = SamyamaClient.embedded()
        from etl.helpers import create_nodes
        create_nodes(client, GRAPH, "Operator", rows)
        ids = {r[1]: r[0] for r in client.query(
            "MATCH (o:Operator) RETURN id(o), o.id", GRAPH).records}
        client.create_vector_index("Operator", "emb", dimensions=DIM, metric="cosine")

        seen, added, add_failed = set(), 0, None
        for r in kept:
            vec = embed(r["name"])
            if vec in seen:                  # exact duplicates removed
                continue
            seen.add(vec)
            try:
                client.add_vector("Operator", "emb", ids[r["id"]], list(vec))
                added += 1
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:     # noqa: BLE001
                add_failed = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}"
                break
        click.echo(f"\nheld out {len(held)}; {len(kept)} remain -> "
                   f"{len(seen)} distinct embeddings")
        click.echo(f"add_vector: {added} accepted"
                   + (f", then {add_failed}" if add_failed else ", none failed"))

        searched, search_failed = 0, None
        for i, r in enumerate(rows):
            if i not in held:
                continue
            try:
                client.vector_search("Operator", "emb", list(embed(r["name"])), k=1)
                searched += 1
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:     # noqa: BLE001
                search_failed = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}"
                break
        click.echo(f"vector_search: {searched} of {len(held)} unseen operators queried"
                   + (f", then {search_failed}" if search_failed else ", none failed"))
        click.echo(holdout_tail(add_failed, search_failed, added, searched))


def holdout_tail(add_failed: str | None, search_failed: str | None,
                 added: int, searched: int) -> str:
    """The conclusion, read off the run rather than written into the module.

    This tail used to be a constant ending "and the search still panics". On
    `samyama` 1.7.1 nothing panics -- so the command printed "none failed" and
    then, four lines later, asserted the opposite, with the false half in the
    prose a reader quotes. A hardcoded conclusion is not a measurement; it is
    the previous measurement, surviving the thing it described.

    The add phase is checked before the search, and a panic is distinguished
    from any other failure. An add that panics `break`s with fewer vectors
    indexed, so the searches that follow may well survive -- branching on the
    search alone would report "the 0.6.1 blocker gone" about a run whose add
    phase panicked. And the blocker is specifically a `PanicException` from the
    HNSW index: calling any other exception by its name would be the same
    mislabelling in the other direction.

    `added` and `searched` are read *after* the failure branches, not before.
    A panic on the very first add leaves `added == 0`, and a vacuous-run guard
    placed first would answer "nothing to conclude, check the fleet was
    generated" -- burying the exact reproduction this command exists to
    produce. They are required arguments rather than defaulted, so a caller
    cannot skip the check by omission.
    """
    if add_failed:
        return (
            f"\n`add_vector` failed ({add_failed}), so the index is incomplete "
            "and the search\nphase above ran against fewer vectors than the "
            "experiment calls for. Whatever\nthe search reported, this run does "
            "not measure what the hold-out is for.\nA `PanicException` there is "
            "the 0.6.1 failure mode; anything else is a different\nproblem.\n")
    if search_failed and "PanicException" in search_failed:
        return (
            "\nThis is the measurement the argument rests on: exact duplicates "
            "removed, every\nadd accepted, and the search still panics -- so "
            "vectors that are merely *close*\ntrip the same assertion, and "
            "deduping is not a workaround.\n")
    if search_failed:
        return (
            f"\n`vector_search` failed, but not with a panic -- {search_failed}."
            "\n\nThe 0.6.1 blocker this experiment is about is a "
            "`PanicException` from the HNSW\nindex. Anything else is a different "
            "fault. Read the exception before concluding\nthe blocker is "
            "present.\n")
    if not added or not searched:
        # After the failure branches: nothing failed *and* nothing ran, which
        # is a fleet or hold-out problem rather than a result about the engine.
        return (
            f"\nNothing to conclude: {added} vectors were added and {searched} "
            f"hold-out searches ran,\nand neither raised. The experiment needs "
            f"both to be non-zero before its result\nmeans anything -- check "
            f"the fleet was generated and that `data/` holds operators.\n")
    return (
        "\nEvery unseen operator searched without panicking. "
        "docs/vector-search.md records\n0.6.1 panicking on the 9th of 40 here, "
        "so on this build that reproduction does\nnot reproduce -- which is a "
        "statement about the build you just ran, not about\nany version "
        "number. Note what it does not show: the engine *accepts* the "
        "workload,\nand nothing here says whether the nearest neighbour "
        "returned is the useful one.\nJudging the answers is #48.\n")


if __name__ == "__main__":
    main()
