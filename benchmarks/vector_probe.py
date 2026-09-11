"""Reproduce the findings in `docs/vector-search.md`, so a retest is a re-run.

The page records that `samyama` 0.6.1's HNSW index panics on vectors that are
identical or nearly so (#48). A defect record is only useful if someone can
re-run it against a bumped engine and see whether it still holds, so both
measurements behind that page live here rather than in prose.

    python -m benchmarks.vector_probe --repro        # the two-call panic
    python -m benchmarks.vector_probe --metrics      # every metric, same panic
    python -m benchmarks.vector_probe --collisions   # the embedding collisions

`--repro` and `--metrics` need nothing but the engine. `--collisions` needs
`data/`.

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
        click.echo("on samyama 0.6.1 this was: PanicException "
                   "(assertion failed: c.dist_to_ref <= 0.)")
        if r["outcome"] == "ok":
            click.echo("It did not happen here, so this run is evidence the "
                       "vector index works\non the installed build -- not "
                       "evidence of the 0.6.1 defect.")

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
        click.echo(holdout_tail(add_failed, search_failed))


def holdout_tail(add_failed: str | None, search_failed: str | None) -> str:
    """The conclusion, read off the run rather than written into the module.

    This tail used to be a constant ending "and the search still panics". On
    `samyama` 1.7.1 nothing panics -- so the command printed "none failed" and
    then, four lines later, asserted the opposite, with the false half in the
    prose a reader quotes. A hardcoded conclusion is not a measurement; it is
    the previous measurement, surviving the thing it described.
    """
    if search_failed:
        return (
            "\nThis is the measurement the argument rests on: exact duplicates "
            "removed, every\nadd accepted, and the search still fails "
            f"({search_failed}) -- so vectors that are\nmerely *close* trip the "
            "same assertion, and deduping is not a workaround.\n")
    if add_failed:
        return (
            "\nThe search was never reached: `add_vector` failed first "
            f"({add_failed}).\nThat is a different defect from the one "
            "docs/vector-search.md describes, which\nsurvived every add and "
            "failed on search. Re-read the page before citing this run.\n")
    return (
        "\nNothing failed on this build. Every add was accepted and every "
        "hold-out search\nanswered -- so the defect docs/vector-search.md "
        "records (a panic on vectors that\nare merely *close*) does not "
        "reproduce here, and that page is describing an\nolder engine. Check "
        "the installed version before citing either.\n")


if __name__ == "__main__":
    main()
