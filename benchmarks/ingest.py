"""Measure ingest velocity -- nodes/s and edges/s -- as a repeatable artifact.

#10 asks for the third V. The only figures that existed were a sentence in
`docs/engine-notes.md`: "~35K nodes/s ... ~3.1K edges/s", an 11x gap. Measured
here, the node figure was stale and the gap is **~21x**, not 11x.

Usage:
    python -m benchmarks.ingest                      # one load, embedded
    python -m benchmarks.ingest --repeats 3          # medians
    python -m benchmarks.ingest --sweep-edge-batch   # the batch-size experiment
    python -m benchmarks.ingest --url http://127.0.0.1:8080 --scale 0.3

## What the gap actually is

Creating a node is a write. Creating an edge is **two lookups and a write**:
`etl/helpers.py:create_edges` emits one `MATCH` pattern per distinct endpoint in
the batch, constrains them all in one `WHERE`, then `CREATE`s the relationships.
So a 100-edge batch is a single statement with up to 200 patterns.

That statement is the whole cost, and it is **superlinear in the number of
patterns**. Measured at `--scale 1.0` on the embedded build:

| edges/batch | patterns/stmt | ms/stmt | edges/s |
|---:|---:|---:|---:|
| 40 | 30 | 13.3 | 3,010 |
| 50 | 37 | 15.9 | **3,140** |
| 60 | 44 | 19.1 | **3,146** |
| 75 | 54 | 26.0 | 2,886 |
| 100 | 71 | 38.5 | 2,594 |
| 200 | 136 | 128.6 | 1,553 |

4.5x the patterns costs 9.7x the time -- roughly `O(p^1.5)`, steepening toward
`O(p^2)` at the larger sizes. Below ~40 the per-statement overhead takes over
again, so the curve has a floor rather than trending to zero.

Node batching, by contrast, is **flat**: ~48K/s on 1.7.1 (52-53K when this was
written on 0.6.1) at every size from 100 to
2000, because a node `CREATE` looks nothing up. That contrast is the finding --
the gap is not "writes are slow", it is the endpoint lookup.

## Two things that do not help

- **Reordering edges.** The `Fleet`'s natural order already dedups endpoints
  well (56,823 patterns at batch 50, against a 152,606 ceiling). Sorting by
  source is a wash; sorting by target or relationship type is 38% *worse*,
  because it breaks up the runs of edges that share a source node.
- **Bigger batches.** Fewer statements, but each one superlinearly more
  expensive. Total patterns actually *fall* slightly as batches grow (51,112 at
  250 vs 56,823 at 50) while time doubles -- which is what rules out pattern
  count as the driver and points at patterns *per statement*.

Timings are machine- and build-specific, so nothing here is asserted in a test.
`tests/test_ingest_benchmark.py` checks the shape of the report, not its
numbers -- a performance figure pinned in CI fails for the hardware's reasons.

## Only the embedded path is trustworthy for the sweep

Every timed load calls `reset_graph` first, but engine note 8 records that
`DETACH DELETE` is **not** a true reset on this engine: the columnar property
store survives it. So over `--url`, successive configurations are not measuring
identical starting states, and a genuine reset means stopping the server and
deleting its data directory between runs.

Embedded has no such problem -- each `SamyamaClient.embedded()` is a fresh
in-memory graph -- which is why every figure quoted above and in
`docs/engine-notes.md` was taken embedded. Treat `--url --sweep-edge-batch`
output as indicative only.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import click
from click.core import ParameterSource

from etl.helpers import chunked, create_edges, create_nodes
from etl.loader import NODE_LABELS, apply_schema, reset_graph

GRAPH_DEFAULT = "default"


def connect(url: str | None):
    from samyama import SamyamaClient
    return SamyamaClient.connect(url) if url else SamyamaClient.embedded()


def build_fleet(seed: int, scale: float, layers: str):
    from etl import generate as gen
    from etl import onnx_catalog as oc
    ops = oc.load_cached()
    fleet = gen.Fleet(seed=seed, scale=scale) if layers == "real" else \
        gen.generate(seed=seed, scale=scale, operators=ops)
    if layers in ("all", "real"):
        from etl import real_layer
        real_layer.build_real(fleet, ops)
    return fleet


def patterns_per_statement(edges, batch: int) -> float:
    """Mean distinct endpoints per batch -- the `MATCH` patterns one statement carries.

    This is the number the cost tracks, which is why it is reported beside the
    timing rather than left to be inferred from the batch size.
    """
    total, statements = 0, 0
    for group in chunked(edges, batch):
        keys = set()
        for src_label, src_id, _rel, tgt_label, tgt_id, _props in group:
            keys.add((src_label, src_id))
            keys.add((tgt_label, tgt_id))
        total += len(keys)
        statements += 1
    return total / statements if statements else 0.0


def existing_indexes(client, graph: str, url: str | None):
    """Indexes on the target, or `[]` when an empty graph is guaranteed anyway.

    One round trip, and no silent fallback. An earlier version returned "can
    this build answer `SHOW INDEXES`?" and, on a build that refuses, let the run
    continue -- which handed a fully indexed load to the caller under a
    "NO INDEXES" banner and a ~1.0x ratio that read as "indexes do not matter".
    A build that cannot report its indexes is a build on which `--no-indexes`
    cannot be honest, so it is refused rather than assumed.

    Embedded is the one exception, and not by trust: `SamyamaClient.embedded()`
    constructs a fresh in-memory graph per process, so there is nothing for a
    prior load to have left behind.
    """
    try:
        return client.query("SHOW INDEXES", graph).records
    except Exception as exc:  # any refusal at all means "cannot check"
        if url is None:
            return []
        raise SystemExit(
            f"--no-indexes needs to verify the target has no indexes, and "
            f"{url} refused `SHOW INDEXES` ({type(exc).__name__}). `DETACH "
            f"DELETE` does not drop indexes and this repo has no DROP path, so "
            f"continuing would time an indexed load and label it unindexed. "
            f"Use an embedded target (omit --url), or start the server with an "
            f"empty data directory.") from exc


def time_one_load(client_factory, fleet, node_batch: int, edge_batch: int,
                  graph: str, patterns: float, skip_indexes: bool = False,
                  url: str | None = None) -> dict:
    """One clean load into a reset graph. Returns seconds and rates."""
    client = client_factory()
    # Reset first, exactly as `etl/loader.py` does. Embedded hides the need for
    # this -- each `SamyamaClient.embedded()` is a fresh in-memory graph -- but
    # over `--url` nothing clears prior state, so a second configuration would
    # load 76,303 edges into a graph that already holds them. Every endpoint
    # `MATCH` then scans a doubled index, and `--sweep-edge-batch` would report
    # a monotonic slowdown ordered by sweep position rather than by batch size:
    # indistinguishable from the superlinear result this module concludes.
    reset_graph(client, graph)
    # The loader's own routine rather than a copy of it. Applied before every
    # timed load and outside the timer: without the id indexes, edge creation is
    # much slower (#18), which would swamp everything measured here.
    #
    # `--no-indexes` skips it deliberately, which is the only way to reproduce
    # that claim rather than take it on trust. `docs/why-this-engine.md` states
    # the ratio; before this flag existed nothing in the repo produced it.
    if skip_indexes:
        # Skipping `apply_schema` is not the same as having no indexes.
        # `reset_graph` issues `DETACH DELETE`, which drops no index, and the
        # schema file has no DROP path -- so against a server that has ever had
        # a normal load, this would time a fully indexed load while printing
        # "NO INDEXES" and yield a ~1.0x ratio. Embedded is safe because each
        # `SamyamaClient.embedded()` is a fresh in-memory graph; anything else
        # is refused rather than measured wrongly.
        indexes = existing_indexes(client, graph, url)
        if indexes:
            raise SystemExit(
                f"--no-indexes was given but the target already has "
                f"{len(indexes)} index(es). `DETACH DELETE` does not drop them "
                f"and this repo has no DROP path, so the run would time an "
                f"indexed load and label it unindexed. Use an embedded target "
                f"(omit --url), or start the server with an empty data "
                f"directory.")
    else:
        apply_schema(client, graph)

    t0 = time.perf_counter()
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, graph, label, fleet.nodes[label], batch=node_batch)
    t1 = time.perf_counter()
    create_edges(client, graph, fleet.edges, batch=edge_batch)
    t2 = time.perf_counter()

    node_s, edge_s = t1 - t0, t2 - t1
    return {
        "nodes": fleet.node_count,
        "edges": fleet.edge_count,
        "node_seconds": round(node_s, 3),
        "edge_seconds": round(edge_s, 3),
        "nodes_per_s": round(fleet.node_count / node_s) if node_s else None,
        "edges_per_s": round(fleet.edge_count / edge_s) if edge_s else None,
        "node_batch": node_batch,
        "edge_batch": edge_batch,
        # Computed by the caller: it re-walks all 76K edges, and doing that
        # inside the reported entry meant recomputing it per repeat for a value
        # that depends only on (edges, batch).
        "patterns_per_statement": patterns,
    }


def median_of(runs: list[dict], key: str):
    values = [r[key] for r in runs if r[key] is not None]
    return round(statistics.median(values), 3) if values else None


@click.command()
@click.option("--url", default=None, help="Samyama server URL. Omit for embedded.")
@click.option("--graph", default=GRAPH_DEFAULT, show_default=True)
@click.option("--scale", default=1.0, show_default=True, type=float)
@click.option("--seed", default=20260814, show_default=True, type=int)
@click.option("--layers", type=click.Choice(["all", "generated", "real"]),
              default="all", show_default=True)
@click.option("--repeats", default=1, show_default=True, type=click.IntRange(1),
              help="Loads per configuration; the median is reported. With "
                   "--cold-start, runs after the first are the warm "
                   "comparison, so 5 is the useful value there.")
@click.option("--node-batch", default=250, show_default=True, type=int)
@click.option("--edge-batch", default=None, type=int,
              help="Defaults to etl.helpers.create_edges' own default.")
@click.option("--cold-start", is_flag=True,
              help="Time constructing an embedded client and answering one "
                   "trivial query, which is the in-process claim "
                   "`docs/why-this-engine.md` leads with. Loads nothing.")
@click.option("--no-indexes", is_flag=True,
              help="Load without `apply_schema`, to reproduce the index-"
                   "criticality figure `docs/why-this-engine.md` publishes. "
                   "Every endpoint lookup becomes a scan; this is a measurement, "
                   "not a mode anyone should load in.")
@click.option("--sweep-edge-batch", is_flag=True,
              help="Time a range of edge batch sizes instead of one load.")
@click.option("--json-out", type=click.Path(), default=None)
@click.pass_context
def main(ctx, url, graph, scale, seed, layers, repeats, node_batch, edge_batch,
         cold_start, no_indexes, sweep_edge_batch, json_out):
    import inspect

    if cold_start:
        # Every option --cold-start cannot act on, not just the two that used to
        # be named. It loads no fleet, so --scale/--seed/--layers describe a
        # graph never built, and --node-batch/--edge-batch/--no-indexes describe
        # a load never run. Accepting them printed a 2.5 ms figure that looked
        # like an answer to whatever was asked, which is worse than refusing.
        IGNORED = ("json_out", "sweep_edge_batch", "scale", "seed", "layers",
                   "no_indexes", "node_batch", "edge_batch")
        given = [p for p in IGNORED
                 if ctx.get_parameter_source(p) is not ParameterSource.DEFAULT]
        if given:
            raise SystemExit(
                "--cold-start times constructing a client and answering one "
                "trivial query. It loads nothing, so it cannot act on "
                + ", ".join("--" + p.replace("_", "-") for p in given)
                + ". Drop those flags, or drop --cold-start. An earlier version "
                "accepted them and exited 0, which a CI step would read as "
                "success.")
    if cold_start:
        # Constructing the client and answering one query, with nothing loaded.
        # Separately timed because the claim is about *starting*, and any load
        # would bury a millisecond figure under twenty-five seconds.
        samples = []
        # `repeats` as given. `max(repeats, 5)` silently overrode an explicit
        # `--repeats 1`, and the first run is the only cold one anyway.
        for _ in range(repeats):
            start = time.perf_counter()
            client = connect(url)
            client.query("RETURN 1", graph)
            samples.append((time.perf_counter() - start) * 1000)
        # The first run is the only cold one. Everything after it reuses an
        # already-imported extension in the same process, so a median over all
        # of them measures "construct another client", not "start". Reporting
        # only the median would have quietly replaced a 2.5 ms claim with 0.02.
        click.echo(f"cold start -> first query "
                   f"({'embedded' if url is None else url})")
        click.echo(f"  cold (first in this process)  {samples[0]:.2f} ms")
        if len(samples) > 1:
            warm = samples[1:]
            click.echo(f"  warm (same process, {len(warm)} runs)  "
                       f"median {statistics.median(warm):.2f} ms")
            click.echo("  The claim in docs/why-this-engine.md is the cold "
                       "figure. A fresh\n  process is the only way to measure "
                       "it -- re-run this command for another.")
        return
    if edge_batch is None:
        edge_batch = inspect.signature(create_edges).parameters["batch"].default

    try:
        fleet = build_fleet(seed, scale, layers)
    except FileNotFoundError:
        raise SystemExit("run `python -m etl.download_data` first") from None

    def factory():
        return connect(url)

    click.echo(f"fleet: {fleet.node_count:,} nodes, {fleet.edge_count:,} edges "
               f"(--scale {scale}, seed {seed}, layers {layers})")
    click.echo(f"target: {'embedded' if not url else url}, graph {graph!r}")
    click.echo("")

    batches = ([40, 50, 60, 75, 100, 200] if sweep_edge_batch else [edge_batch])
    results = []

    if no_indexes:
        # Checked before the banner: printing "NO INDEXES" and then refusing
        # reads as a contradiction. `time_one_load` re-checks per load, which is
        # where the guarantee has to hold; this is the fail-fast copy, so it
        # reuses one client and one round trip rather than opening a throwaway
        # connection and asking twice.
        if existing_indexes(factory(), graph, url):
            raise SystemExit(
                "--no-indexes was given but the target already has indexes. "
                "`DETACH DELETE` does not drop them and this repo has no DROP "
                "path, so the run would time an indexed load and label it "
                "unindexed. Use an embedded target (omit --url), or start the "
                "server with an empty data directory.")
        click.echo("  NO INDEXES: `apply_schema` skipped, so every endpoint "
                   "lookup is a scan.\n  Compare against a normal run -- that "
                   "ratio is the index-criticality figure.")
    if sweep_edge_batch:
        click.echo(f"{'edge batch':>11}{'median s':>10}{'edges/s':>10}"
                   f"{'pat/stmt':>10}{'ms/stmt':>9}")
    for batch in batches:
        patterns = round(patterns_per_statement(fleet.edges, batch), 1)
        runs = [time_one_load(factory, fleet, node_batch, batch, graph, patterns,
                              skip_indexes=no_indexes, url=url)
                for _ in range(repeats)]
        entry = dict(runs[0])
        entry["edge_seconds"] = median_of(runs, "edge_seconds")
        entry["node_seconds"] = median_of(runs, "node_seconds")
        entry["edges_per_s"] = (round(fleet.edge_count / entry["edge_seconds"])
                                if entry["edge_seconds"] else None)
        entry["nodes_per_s"] = (round(fleet.node_count / entry["node_seconds"])
                                if entry["node_seconds"] else None)
        entry["repeats"] = repeats
        # Recorded, because two JSON files that differ only by this flag differ
        # by a factor of six and nothing in the file said which was which.
        entry["no_indexes"] = no_indexes
        results.append(entry)

        if sweep_edge_batch:
            statements = -(-fleet.edge_count // batch)
            click.echo(f"{batch:>11}{entry['edge_seconds']:>10.2f}"
                       f"{entry['edges_per_s']:>10,}"
                       f"{entry['patterns_per_statement']:>10.0f}"
                       f"{1000 * entry['edge_seconds'] / statements:>9.2f}")
        else:
            click.echo(f"  nodes  {entry['node_seconds']:>7.2f}s  "
                       f"{entry['nodes_per_s']:>9,}/s   (batch {node_batch})")
            click.echo(f"  edges  {entry['edge_seconds']:>7.2f}s  "
                       f"{entry['edges_per_s']:>9,}/s   (batch {batch}, "
                       f"{entry['patterns_per_statement']:.0f} patterns/statement)")
            if entry["nodes_per_s"] and entry["edges_per_s"]:
                click.echo(f"  per-item gap: "
                           f"{entry['nodes_per_s'] / entry['edges_per_s']:.1f}x")

    if sweep_edge_batch and results:
        best = max(results, key=lambda r: r["edges_per_s"] or 0)
        click.echo("")
        click.echo(f"== fastest edge batch: {best['edge_batch']} "
                   f"({best['edges_per_s']:,} edges/s)")
        click.echo("== cost is superlinear in patterns per statement, so the "
                   "optimum is a floor, not a maximum")

    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        click.echo(f"== wrote {json_out}")


if __name__ == "__main__":
    main()
