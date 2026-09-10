"""Time snapshot import and export against a running server (#45).

`README.md` said a `.sgsnap` of the full graph imports in *"well under a
second"*. That is the most quotable performance claim in the repo and it was
prose -- nobody had timed it. This command is the measurement, so the figure in
the README can be replaced by one anybody can reproduce.

    python -m benchmarks.snapshot --url http://127.0.0.1:8080 --export kg.sgsnap
    python -m benchmarks.snapshot --url http://127.0.0.1:8080 --file kg.sgsnap --repeats 5
    python -m benchmarks.snapshot --url http://127.0.0.1:8080 --file kg.sgsnap --verify-queries

Snapshot is an **HTTP-only** feature: `SamyamaClient.embedded()` exposes no
snapshot method, so unlike the rest of `benchmarks/` this needs a server.

    docker run -d -p 8080:8080 ghcr.io/samyama-ai/samyama-graph:1

## What is being compared, and what is not

A `.sgsnap` is **serialised internal state**; `python -m etl.loader` is a
**build** -- it renders Cypher, parses it, mints ids and constructs indexes.
That is exactly why the snapshot is faster, and saying so is what stops the
comparison being a trick. The same distinction applies to Neo4j: `LOAD CSV` or
`neo4j-admin import` is a build, and the honest counterpart to a snapshot import
is restoring a Neo4j *backup*, which this repo has not measured (#47).

## Three things measured here that the flow depends on

**Import appends; it does not replace.** Importing into a graph that already
holds data leaves both. Measured on one container, twice, because the two cases
differ in a way that matters:

| | nodes | edges |
|---|---:|---:|
| after 1st import | 25,150 | 76,303 |
| **A:** 2nd import, nothing deleted | **50,300** | **152,606** |
| **B:** 2nd import, after `delete_graph` | 25,150 | **152,606** |

In **A** both double, and any sanity check catches it. **B** is the dangerous
one and the one this command has to survive: the nodes were deleted and put
back, so the count returns to 25,150 while the edges accumulate. A node-count
check passes and everything looks fine. See the `delete_graph` section below --
the two rows are the same experiment seen from either side.

The README's flow starts from a fresh server, so it is correct as written; it is
re-running it against a live one that silently doubles the graph. This command
refuses to time an import into a non-empty graph.

**`status()` under-reports after a delete, so emptiness is checked by query.**
After `delete_graph("default")` the server reported `nodes=0, edges=152594`
while `MATCH ()-[r]->() RETURN count(r)` returned `0`. The graph was genuinely
empty and the status counter was stale. Anything here that needs to know what
the graph holds asks the graph.

**`delete_graph` is not a reset, and re-importing resurrects the edges.** This
is engine note 8's shape ("`DETACH DELETE` is not a reset -- the columnar store
survives it") reaching `delete_graph` and reaching *edges*. Measured:

| step | by query | `status()` |
|---|---|---|
| fresh server | 0 nodes / 0 edges | 0 / 0 |
| after 1st import | 25,150 / 76,303 | 25,150 / 76,303 |
| after `delete_graph` | **0 / 0** | 0 / **76,303** |
| after 2nd import | 25,150 / **152,606** | 25,150 / 152,606 |

The graph queries as empty and then the deleted edges come back the moment nodes
carrying their ids exist again -- 76,303 old plus 76,303 new. **Nodes do not
double here**, unlike case A above, because they really were deleted; so a
node-count check passes and everything looks fine.

An earlier version of this command used `delete_graph` between `--repeats` and
duly reported five clean-looking timings over a graph reaching 381,515 edges.
So `--repeats` above 1 now **requires `--restart-cmd`**, and every run is
checked against what the first import produced rather than against zero.
"""
from __future__ import annotations

import statistics
import subprocess
import time
from pathlib import Path

import click
import requests

GRAPH = "default"
TIMEOUT = 300


def _client(url: str):
    from samyama import SamyamaClient

    return SamyamaClient.connect(url)


def graph_holds(client) -> tuple[int, int]:
    """Nodes and edges **as the graph reports them**, not as `status()` does.

    `status()` returned `edges=152594` on a graph whose own `count(r)` was `0`.
    A benchmark that trusted it would refuse to run against an empty server, or
    worse, report an import into a graph it believed was clean.
    """
    nodes = client.query("MATCH (n) RETURN count(n.id)", GRAPH).records[0][0]
    edges = client.query("MATCH ()-[r]->() RETURN count(r)", GRAPH).records[0][0]
    return nodes, edges


def time_import(url: str, path: Path) -> float:
    """Seconds for one `POST /api/snapshot/import`, upload included.

    The upload is part of it and is not separated out: over loopback a 1 MB
    multipart body is a small fraction of the total, and over a real network it
    is a cost the user actually pays. Splitting it would produce a number that
    is true of no one's run.
    """
    with open(path, "rb") as fh:
        body = fh.read()
    start = time.perf_counter()
    response = requests.post(f"{url}/api/snapshot/import",
                             files={"file": (path.name, body)}, timeout=TIMEOUT)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return elapsed


def time_export(url: str, path: Path) -> tuple[float, int]:
    start = time.perf_counter()
    response = requests.post(f"{url}/api/snapshot/export", timeout=TIMEOUT)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    path.write_bytes(response.content)
    return elapsed, len(response.content)


def _restart(command: str, url: str) -> None:
    """Run the user's restart command, then wait for the server to answer again.

    Waiting on `/api/status` rather than sleeping a fixed time: a container
    restart is fast but not instant, and a fixed sleep either wastes seconds per
    run or races on a slow machine, where the failure is a connection error in
    the middle of a timing loop.
    """
    subprocess.run(command, shell=True, check=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            requests.get(f"{url}/api/status", timeout=2).raise_for_status()
        except Exception:  # any failure here means "not up yet"
            time.sleep(0.25)
        else:
            return
    raise SystemExit(f"server did not come back within 60s of: {command}")


def _report(label: str, times: list[float]) -> None:
    click.echo(f"\n{label}")
    for i, t in enumerate(times, 1):
        click.echo(f"  run {i}: {t:6.3f} s")
    if len(times) > 1:
        click.echo(f"  median {statistics.median(times):.3f} s   "
                   f"min {min(times):.3f}   max {max(times):.3f}")


@click.command()
@click.option("--url", default="http://127.0.0.1:8080", show_default=True,
              help="A running server. Snapshot has no embedded API.")
@click.option("--file", "path", type=click.Path(exists=True, path_type=Path),
              help="Snapshot to import and time.")
@click.option("--export", "export_to", type=click.Path(path_type=Path),
              help="Export the server's current graph here, and time that.")
@click.option("--repeats", default=1, show_default=True, type=int,
              help="Import runs. Above 1 needs --restart-cmd: delete_graph does "
                   "not clear edges and re-importing resurrects them.")
@click.option("--restart-cmd", default=None,
              help="Shell command that restarts the server, run between repeats "
                   "(e.g. 'docker restart samyama'). The only reset that works.")
@click.option("--verify-queries", is_flag=True,
              help="Run all 16 catalog queries against the imported graph.")
def main(url, path, export_to, repeats, verify_queries, restart_cmd):
    if not path and not export_to:
        raise SystemExit("give --file to time an import, or --export to time an export")
    if path and export_to:
        # They are mutually exclusive by construction, not by preference:
        # --export needs a graph with something in it, and timing an import
        # needs an empty one. Together they always die at the second check,
        # after the export has already run. Refuse up front instead.
        raise SystemExit(
            "--export and --file cannot run in one invocation: --export needs a "
            "loaded graph and timing an import needs an empty one. Export first, "
            "restart the server, then time the import against the fresh one.")
    client = _client(url)
    click.echo(f"server {client.status().version} at {url}")

    if export_to:
        nodes, edges = graph_holds(client)
        if not nodes:
            raise SystemExit("the graph is empty, so there is nothing to export. "
                             "Load it first: python -m etl.loader --url " + url)
        click.echo(f"exporting {nodes:,} nodes / {edges:,} edges ...")
        elapsed, size = time_export(url, export_to)
        click.echo(f"  {elapsed:.3f} s   {size:,} bytes -> {export_to}")

    if not path:
        return

    if repeats > 1 and not restart_cmd:
        raise SystemExit(
            "--repeats above 1 needs --restart-cmd. `delete_graph` leaves the "
            "edges behind and the next import resurrects them, so runs 2..n "
            "would time an import into a graph that keeps growing -- measured, "
            "76,303 edges became 381,515 over five 'reset' runs while the node "
            "count stayed put and everything looked fine. Try:\n"
            "  --restart-cmd 'docker restart <container>'")

    times, expected = [], None
    for run in range(1, repeats + 1):
        nodes, edges = graph_holds(client)
        if nodes or edges:
            raise SystemExit(
                f"the graph already holds {nodes:,} nodes / {edges:,} edges. "
                f"Import APPENDS rather than replaces, so timing one here would "
                f"measure a merge into a graph larger than the snapshot. Restart "
                f"the server before run {run}.")
        times.append(time_import(url, path))
        nodes, edges = graph_holds(client)
        click.echo(f"  run {run}: {times[-1]:6.3f} s  -> {nodes:,} nodes / "
                   f"{edges:,} edges")

        # Against the first run, not against zero. A restart that silently fails
        # to clear the store leaves the *previous* import's edges in place, and
        # they reappear here rather than at the emptiness check above -- which is
        # exactly how the delete_graph version of this command produced five
        # plausible timings over a graph that had quadrupled.
        if expected is None:
            expected = (nodes, edges)
        elif (nodes, edges) != expected:
            raise SystemExit(
                f"run {run} left {nodes:,} nodes / {edges:,} edges where run 1 "
                f"left {expected[0]:,} / {expected[1]:,}. The restart did not "
                f"clear the store, so this run and any after it are not "
                f"comparable. Timings so far: "
                f"{', '.join(f'{t:.3f}s' for t in times)}")

        if run < repeats:
            _restart(restart_cmd, url)

    _report(f"import of {path.name} ({path.stat().st_size:,} bytes)", times)

    if verify_queries:
        from benchmarks.queries import BY_ID

        returned, empty, failed = [], [], []
        for qid, spec in BY_ID.items():
            try:
                got = client.query(spec["cypher"], GRAPH).records
            except Exception as exc:
                failed.append((qid, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"))
            else:
                (returned if got else empty).append(qid)
        click.echo(f"\ncatalog against the imported snapshot: "
                   f"{len(returned)} of {len(BY_ID)} return rows, "
                   f"{len(empty)} empty, {len(failed)} failed")
        if empty:
            click.echo(f"  empty : {empty}")
        for qid, err in failed:
            click.echo(f"  FAILED {qid}: {err}")


if __name__ == "__main__":
    main()
