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


def run_repro(metric: str = "cosine") -> tuple[bool, str]:
    """Add one vector twice. Returns (panicked, detail)."""
    from samyama import SamyamaClient
    client = SamyamaClient.embedded()
    ids = two_nodes(client)
    v = normalised_random(8, seed=11)
    try:
        client.create_vector_index("T", "emb", dimensions=8, metric=metric)
        client.add_vector("T", "emb", ids[0], v)
        client.add_vector("T", "emb", ids[1], v)
        client.vector_search("T", "emb", v, k=2)
        return False, "no panic"
    except BaseException as exc:           # noqa: BLE001 - PanicException is not Exception
        return True, (str(exc).splitlines() or [type(exc).__name__])[0]


@click.command()
@click.option("--repro", is_flag=True, help="The two-call panic.")
@click.option("--metrics", is_flag=True, help="Every metric, same reproduction.")
@click.option("--collisions", is_flag=True, help="Embedding collisions on the operators.")
@click.option("--seed", default=20260814, show_default=True, type=int)
@click.option("--scale", default=0.3, show_default=True, type=float)
def main(repro, metrics, collisions, seed, scale):
    if not (repro or metrics or collisions):
        raise SystemExit("give --repro, --metrics or --collisions")

    if repro:
        panicked, detail = run_repro()
        click.echo(f"same normalised vector added twice -> "
                   f"{'PANIC' if panicked else 'OK'}: {detail}")
        click.echo("expected on samyama 0.6.1: PANIC "
                   "(assertion failed: c.dist_to_ref <= 0.)")

    if metrics:
        click.echo(f"\n{'metric':<18}{'result'}")
        for metric in ("cosine", "euclidean", "l2", "dot", "inner_product", "manhattan"):
            panicked, detail = run_repro(metric)
            click.echo(f"{metric:<18}{'PANIC' if panicked else 'OK'}  {detail[:48]}")
        click.echo("\nNote: `dot`, `inner_product` and `manhattan` are accepted by "
                   "create_vector_index\nwithout complaint and are not documented as "
                   "supported -- the metric is not validated.")

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

        names = [o["name"] for o in fleet.nodes["Operator"]]
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


if __name__ == "__main__":
    main()
