"""Build and load the Edge AI deployment graph into Samyama.

Usage:
    python -m etl.loader --url http://127.0.0.1:8080   # loads a graph you can query
    python -m etl.loader                               # embedded: TIMING ONLY
    python -m etl.loader --scale 4

**Without `--url` the graph is discarded when this process exits** (#4).
`SamyamaClient.embedded()` runs the engine in-process and in-memory, so a
following `python -m benchmarks.run_benchmark` -- also embedded by default --
finds nothing, with no error to explain why. The bare form is a
timing and verification exercise, not a way to prepare data, and it says so on
exit rather than leaving a newcomer to discover it.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import click

from etl import generate as gen
from etl.helpers import create_edges, create_nodes

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "edge_ai_kg.cypher"

# Every node label carries a unique `id`; indexing it is what makes the
# edge-creation MATCH ... WHERE lookups cheap.
NODE_LABELS = [
    "Vendor", "SoC", "Accelerator", "Board", "Runtime", "Operator", "Kernel",
    "Model", "ModelVariant", "Sensor", "SignalStage", "ClinicalTask",
    "BenchmarkTask", "Dataset", "Certification", "Deployment", "Site",
]


def connect(url: str | None):
    from samyama import SamyamaClient
    return SamyamaClient.connect(url) if url else SamyamaClient.embedded()


def apply_schema(client, graph: str) -> int:
    """Run the index statements from schema/edge_ai_kg.cypher."""
    applied = 0
    if not SCHEMA_PATH.exists():
        return 0
    for raw in SCHEMA_PATH.read_text(encoding="utf-8").splitlines():
        stmt = raw.split("//", 1)[0].strip().rstrip(";")
        if not stmt:
            continue
        try:
            client.query(stmt, graph)
            applied += 1
        except Exception as exc:  # index already exists, or unsupported syntax
            click.echo(f"  ! schema stmt skipped ({exc.__class__.__name__}): {stmt[:60]}",
                       err=True)
    return applied


def reset_graph(client, graph: str) -> None:
    try:
        client.query("MATCH (n) DETACH DELETE n", graph)
    except Exception:
        try:
            client.query("MATCH (n) DELETE n", graph)
        except Exception as exc:
            click.echo(f"  ! could not reset graph: {exc}", err=True)


def count_edges_by_type(client, graph: str, rel_types) -> dict[str, int]:
    """How many edges of each type the graph actually holds.

    Counts a property rather than the relationship variable: `count(r)` over a
    multi-variable MATCH does not aggregate on this engine (engine note 9).
    """
    counts = {}
    for rel in rel_types:
        result = client.query(
            f"MATCH (a)-[:{rel}]->(b) RETURN count(a.id) AS n", graph)
        counts[rel] = result.records[0][0] if result.records else 0
    return counts


def verify_edges(client, graph: str, edges) -> list[tuple[str, int, int]]:
    """Compare edges the loader intended to create against what the graph holds.

    `create_edges` reports the number of edges it *submitted*, not the number
    created. Endpoints are matched by id, and a whole batch shares one `MATCH`,
    so a single unresolvable id silently drops every edge in that batch -- with
    no error, and with the node counts still correct.

    Returns one `(rel_type, intended, actual)` row per type, worst shortfall
    first.
    """
    intended = Counter(e[2] for e in edges)
    actual = count_edges_by_type(client, graph, sorted(intended))
    rows = [(rel, intended[rel], actual[rel]) for rel in intended]
    rows.sort(key=lambda r: (r[2] - r[1], r[0]))
    return rows


@click.command()
@click.option("--url", default=None,
              help="Samyama server URL, e.g. http://127.0.0.1:8080. Omit for "
                   "embedded -- but note that an embedded load is DISCARDED when "
                   "this process exits (#4), so it is a timing exercise rather "
                   "than a way to prepare data.")
@click.option("--graph", default="default", show_default=True,
              help="Target graph / tenant. Accepted and IGNORED on the OSS build -- everything lands in 'default' whatever you pass (engine note 7). Kept because the engine takes the argument and a future build may honour it.")
@click.option("--seed", type=int, default=gen.DEFAULT_SEED, show_default=True)
@click.option("--scale", type=float, default=1.0, show_default=True,
              help="Fleet size multiplier. 1.0 ~ 24K nodes / 73K edges.")
@click.option("--limit", type=int, default=None,
              help="Cap nodes per label for a fast smoke load.")
@click.option("--regenerate/--use-cached", default=False,
              help="Regenerate the fleet instead of loading data/fleet/fleet.json.")
@click.option("--reset/--no-reset", default=True, show_default=True,
              help="Delete existing nodes in the target graph first. Against a "
                   "populated graph --no-reset mints every id twice; ids are not "
                   "unique-constrained so the write is accepted, but the "
                   "duplicates then multiply edges and the load does not "
                   "complete. NOTE: over --url on samyama 1.7.0 this is not a "
                   "real reset -- engine note 8, the property columns survive "
                   "DETACH DELETE, so a --layers real load onto a server that "
                   "held the full fleet puts generated cost-model values on "
                   "MLPerf nodes and --verify still passes. Start the server "
                   "from an empty data directory instead.")
@click.option("--layers", type=click.Choice(["all", "real", "synthetic"]),
              default="all", show_default=True,
              help="Load the real public-source subgraph, the generated fleet, or both.")
@click.option("--verify/--no-verify", default=True, show_default=True,
              help="After loading, count edges per type against what was intended.")
def main(url, graph, seed, scale, limit, regenerate, reset, layers, verify):
    started = time.time()

    def build(why: str, build_seed: int, build_scale: float):
        """Generate and cache a fleet, saying why and at which seed and scale.

        One helper for all three callers -- `--regenerate`, no cache, stale
        cache -- because they differ only in the reason and in *whose* seed
        and scale they use, and that difference is the part worth reading.
        """
        click.echo(f"[1/4] {why} (seed={build_seed}, scale={build_scale}) ...")
        built = gen.generate(seed=build_seed, scale=build_scale)
        gen.write(built)
        return built

    if regenerate:
        fleet = build("generating fleet", seed, scale)
    else:
        try:
            fleet = gen.load()
            click.echo(f"[1/4] loaded cached fleet (seed={fleet.seed}, scale={fleet.scale})")
        except FileNotFoundError:
            fleet = build("no cached fleet; generating", seed, scale)
        except gen.StaleFleetCache as exc:
            # Regenerated here, though `gen.load` refuses: `data/` is
            # gitignored, so this is the existing-checkout case -- `git pull`
            # brought a label the cache predates, and refusing would leave the
            # reader to run a command this can run itself. The alternative to
            # both is worse: a load that succeeds while every query over the
            # new label returns zero rows, which reads as "the answer is none".
            #
            # At the **cache's** seed and scale, not the CLI's. The cache was
            # written by some earlier run whose options are not these ones, and
            # rebuilding it at `--scale 1.0` because that is the default would
            # answer a stale-cache problem by silently swapping the graph.
            click.echo(f"[1/4] cached fleet predates {', '.join(exc.missing)}; "
                       f"rebuilding it as it was")
            fleet = build("regenerating", exc.seed, exc.scale)

    if layers == "synthetic":
        click.echo("      layers: synthetic only")
    else:
        from etl import onnx_catalog, real_layer
        if layers == "real":
            fleet = gen.Fleet(seed=fleet.seed, scale=fleet.scale)
        before = fleet.node_count
        real_layer.build_real(fleet, onnx_catalog.load_cached())
        click.echo(f"      layers: {layers} "
                   f"(+{fleet.node_count - before:,} real nodes from public sources)")

    nodes = fleet.nodes
    edges = fleet.edges
    if limit:
        kept: dict[str, set[str]] = {}
        trimmed = {}
        for label, rows in nodes.items():
            trimmed[label] = rows[:limit]
            kept[label] = {r["id"] for r in trimmed[label]}
        nodes = trimmed
        edges = [e for e in edges
                 if e[1] in kept.get(e[0], ()) and e[4] in kept.get(e[3], ())]
        click.echo(f"      --limit {limit}: {sum(len(v) for v in nodes.values()):,} nodes, "
                   f"{len(edges):,} edges")

    client = connect(url)
    where = url or "embedded"
    click.echo(f"[2/4] connected ({where}), graph={graph}")
    if reset:
        reset_graph(client, graph)
    applied = apply_schema(client, graph)
    click.echo(f"      schema: {applied} index statements applied")

    click.echo("[3/4] loading nodes ...")
    t0 = time.time()
    total_nodes = 0
    for label in NODE_LABELS:
        rows = nodes.get(label, [])
        if not rows:
            continue
        created = create_nodes(client, graph, label, rows)
        total_nodes += created
        click.echo(f"      {label:15s} {created:>7,}")
    node_secs = time.time() - t0

    click.echo("[4/4] loading edges ...")
    t0 = time.time()
    total_edges = create_edges(client, graph, edges)
    edge_secs = time.time() - t0

    elapsed = time.time() - started
    click.echo("")
    click.echo(f"  nodes {total_nodes:,} in {node_secs:.1f}s "
               f"({total_nodes / max(node_secs, 1e-6):,.0f}/s)")
    click.echo(f"  edges {total_edges:,} in {edge_secs:.1f}s "
               f"({total_edges / max(edge_secs, 1e-6):,.0f}/s)")
    click.echo(f"  total {elapsed:.1f}s")

    try:
        got = client.query("MATCH (n) RETURN count(n) AS n", graph).records[0][0]
        click.echo(f"  verified: {got:,} nodes in graph '{graph}'")
        if got != total_nodes:
            click.echo(f"  ! expected {total_nodes:,}", err=True)
            sys.exit(1)
    except Exception as exc:
        click.echo(f"  ! verification query failed: {exc}", err=True)
        sys.exit(1)

    if verify:
        try:
            rows = verify_edges(client, graph, edges)
        except Exception as exc:
            click.echo(f"  ! edge verification query failed: {exc}", err=True)
            sys.exit(1)
        missing = [r for r in rows if r[2] < r[1]]
        surplus = [r for r in rows if r[2] > r[1]]
        intended = sum(r[1] for r in rows)
        click.echo(f"  verified: {sum(r[2] for r in rows):,} of {intended:,} "
                   f"intended edges across {len(rows)} types")

        def table(heading: str, entries, note: str) -> None:
            click.echo("")
            click.echo(f"  ! {heading}", err=True)
            click.echo(f"    {'edge type':22} {'intended':>10} {'actual':>10} "
                       f"{'delta':>8}", err=True)
            for rel, want, have in entries:
                click.echo(f"    {rel:22} {want:>10,} {have:>10,} {have - want:>+8,}",
                           err=True)
            click.echo("")
            click.echo(f"    {note}", err=True)

        if missing:
            table("edges missing -- the load reported success and did not create these:",
                  missing,
                  "An endpoint id did not resolve. Edges are created in batches "
                  "sharing one MATCH,\n    so one unresolvable id drops its whole "
                  "batch. See issue #23.")
        if surplus:
            table("more edges than intended -- the graph holds edges this load did "
                  "not submit:",
                  surplus,
                  "Either the graph was not empty (--no-reset, a failed reset, or a "
                  "second load\n    on top of a first), or two nodes share an `id`: "
                  "endpoints are matched by\n    id and this engine parses no "
                  "uniqueness constraint (engine note 6), so a\n    duplicate id "
                  "binds twice and one submitted edge is created twice.")
        if missing or surplus:
            sys.exit(1)

    if not url:
        # #4: the load succeeded and is about to be thrown away. Said last, and
        # on stderr, because everything above it reads like a successful load --
        # the per-label counts, the timings, and `verified: N nodes` are all
        # true right up to the moment this process exits.
        for line in (
            "",
            "  ! this graph was NOT persisted.",
            "    Embedded mode is in-process and in-memory, so everything above is",
            "    gone when this command exits. A following",
            "    `python -m benchmarks.run_benchmark` -- also embedded by default --",
            "    will find an empty graph and not say why.",
            "",
            "    To keep it, run the server and point both commands at it:",
            "      ./target/release/samyama --http-port 8080",
            "      python -m etl.loader --url http://127.0.0.1:8080",
            "      python -m benchmarks.run_benchmark --url http://127.0.0.1:8080",
            "",
            "    Without --url this command is a timing and verification exercise (#4).",
        ):
            click.echo(line, err=True)


if __name__ == "__main__":
    main()
