"""Run the catalog against Neo4j and against this engine, and publish the losses (#47).

`benchmarks/run_benchmark.py` times the catalog on Samyama. A number with
nothing to compare it against is a number, not a claim. This runs the **same
Cypher text** on Neo4j 5 and on Samyama, on the same machine against the same
generated fleet, and prints every query including the ones we lose.

    docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \\
      -e NEO4J_AUTH=neo4j/$NEO4J_PASSWORD neo4j:5-community
    python -m benchmarks.compare_neo4j --neo4j http://127.0.0.1:7474

`--password` reads `NEO4J_PASSWORD`, so the one variable sets both sides.

## What makes this believable, and what does not

**Same query text, both engines.** All 16 catalog queries parse on Neo4j 5
unchanged -- checked before any of this was written. Nothing is translated, so
there is no translation to argue with.

**These are our query shapes, and it is worth knowing whether that helps us.**
`docs/engine-notes.md` records that the catalog is shaped around Samyama's
limits: projected through `WITH` before `RETURN` (note 3), single `ORDER BY` key
(note 3b), `count(DISTINCT x.id)` rather than `count(DISTINCT x)` (note 9),
`OPTIONAL MATCH ... count() = 0` instead of a negated pattern (note 5), and
conditional aggregation instead of a self-join (note 1). Neo4j needs none of
them, so the obvious worry is that we are making it do work it would not
otherwise do.

**Measured, and it is the other way round.** `--natural` times the hero query
both ways on Neo4j -- as we ship it, and with `NOT EXISTS { }` as a Neo4j author
would write it. Across the two published runs our shape ran in 6.7 / 6.2 ms
and the idiomatic one in 9.9 / 11.4 ms, same 15 rows: **our workarounds make
that query 1.48-1.84x faster on Neo4j**, not slower. An earlier draft of this
docstring asserted the opposite and was wrong; it was reasoning about what
workarounds usually cost rather than measuring this one. The figures live in
`docs/neo4j-comparison.md`; if this and that page disagree, the page is right.

That is one query and one reading, so it does not license "our shapes are
faster everywhere". What it does rule out is the specific objection that these
timings are a thumb on the scale against Neo4j, at least on the query the repo
is built around.

**Indexes on both sides.** `schema/edge_ai_kg.cypher`'s 22 indexes are created
on Neo4j too, in its own syntax. Without that the comparison measures index
availability rather than engines.

**Warmed properly, and medians.** `--warmup` runs of each query are discarded
on both engines before `--repeats` timed runs, and the median is reported.

Ten, not one, and that default was bought with a wrong answer. An earlier
version discarded a single run. Against a Neo4j that had been up an hour
serving queries it measured `EA01` at 6.0 ms; against a freshly started one,
12.1 ms -- and the headline "Neo4j is 2.6x faster on the hero query" became
1.6x on that difference alone. One warm-up does not warm a JVM, and the number
it produces depends on what the process happened to be doing beforehand.

**Load time is reported but is not the comparison.** Both figures exclude
index creation -- Neo4j builds its indexes after the nodes and subtracts that,
Samyama applies its schema before them and starts the clock after it. The two
paths still differ too much to be a like-for-like: Samyama takes batched `CREATE` over its own client,
Neo4j takes parameterised `UNWIND` over HTTP. Both are the normal way to fill
each engine, neither is that engine's fastest bulk path (`neo4j-admin import`
and `.sgsnap` respectively), and the figure is here for scale, not for scoring.

## What this does not measure

One machine, one dataset, one engine version each, single client, no
concurrency, no cold-cache case, no memory ceiling. `docs/volume.md` covers
growth on Samyama alone. Neo4j's operational maturity is not in scope and is
not in doubt -- `docs/why-this-engine.md` says so at length.
"""
from __future__ import annotations

import re
import time

import click

from benchmarks.natural_ea01 import (
    NATURAL_EA01,
    natural_ea01_matches_the_catalog,
)
from benchmarks.neo4j_client import (
    NEO4J_DB,
    SCHEMA_PATH,
    Neo4j,
    Neo4jLoadError,
    load_neo4j,
    neo4j_indexes,
)
from benchmarks.sweep import PARITY_BAND, compare_catalog, same_rows, timed

GRAPH = "default"


def load_samyama(client, fleet) -> float:
    """Fill Samyama, **with its schema applied**, exactly as `etl/loader.py` does.

    `apply_schema` is not optional and leaving it out was the mirror image of
    the Neo4j bug this module was already fixed for. Without the `id` indexes,
    every endpoint lookup in `create_edges` is a scan: edge loading measured
    157 s here against 24.5 s from `benchmarks/ingest.py`, which applies the
    schema -- 486 edges/s against 3,110.

    Worse than a slow load, it meant **the query comparison ran Samyama with no
    indexes at all while Neo4j had all 22**, which is not a comparison. Both
    engines now get their indexes before anything is timed.

    `apply_schema` runs **outside** the returned time, which is what
    `benchmarks/neo4j_client.load_neo4j` already assumed of this side when it
    subtracted its own index build. The clock used to start before it, so
    Samyama's figure carried its schema and Neo4j's did not -- the same
    asymmetry in the direction that flatters Neo4j. The indexes are still in
    place for every insert, so their maintenance cost stays in the figure; only
    the one-off `CREATE INDEX` statements are excluded, on both sides.
    """
    from etl.helpers import create_edges, create_nodes
    from etl.loader import NODE_LABELS, apply_schema

    apply_schema(client, GRAPH)
    start = time.perf_counter()
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    elapsed = time.perf_counter() - start
    # Counted, because the Neo4j side is. `load_neo4j` verifies its own load
    # and `--reuse-neo4j` verifies the server's, so a Samyama load that
    # silently dropped rows was the one half of the comparison nothing
    # checked -- and a query timed against a short graph is fast for the
    # wrong reason.
    expected = sum(len(rows) for rows in fleet.nodes.values())
    got = client.query("MATCH (n) RETURN count(n.id)", GRAPH).records
    held = got[0][0] if got else 0
    if held != expected:
        raise SystemExit(
            f"samyama loaded {held:,} nodes, expected {expected:,}. Every "
            f"timing below would be taken on a graph that is not the fleet.")
    return elapsed


def _verify_reused(neo, nodes: int, edges_expected: int) -> None:
    """`--reuse-neo4j`'s checks: the same graph, and all of its indexes usable.

    Exits on a mismatch. A `Neo4jLoadError` from the queries themselves is left
    to the caller, which turns it into an exit on both load paths alike.
    """
    held = neo.run("MATCH (n) RETURN count(n)")[0][0]
    edges = neo.run("MATCH ()-[r]->() RETURN count(r)")[0][0]
    if (held, edges) != (nodes, edges_expected):
        raise SystemExit(
            f"--reuse-neo4j was given, but Neo4j holds {held:,} nodes / "
            f"{edges:,} edges and this fleet is {nodes:,} / "
            f"{edges_expected:,}. Comparing two engines over different "
            f"graphs is not a comparison. Drop the flag to reload.")
    # Indexes too, not just counts. A reused database with the right rows and
    # no indexes is exactly the failure this module was corrected for in #47
    # -- an unindexed engine measured against an indexed one -- and a count
    # check cannot see it.
    #
    # Parsed with a pattern that either matches or is reported. `str.split`
    # raised IndexError on any statement not shaped exactly `FOR (n:L) ON
    # (n.p)` -- a LOOKUP index, say -- which is a crash where a message belongs.
    expected = set()
    for statement in neo4j_indexes(SCHEMA_PATH):
        shape = re.search(r"FOR \(n:(\w+)\) ON \(n\.(\w+)\)", statement)
        if not shape:
            raise SystemExit(
                f"cannot read the label and property out of {statement!r}. "
                f"`neo4j_indexes` changed shape; this check must be updated "
                f"with it rather than skipping the statement.")
        expected.add(f"{shape.group(1)}.{shape.group(2)}")
    # Present means label, property, `state` and `type` all match. A
    # POPULATING index is not usable yet, and `schema/edge_ai_kg.cypher`
    # creates plain RANGE indexes -- a UNIQUE or TEXT index on the same
    # `label.property` is a different structure with different lookup
    # behaviour. Counting either as the index we asked for would time an
    # unindexed or differently-indexed Neo4j while reporting all 22 verified.
    present = {f"{labels[0]}.{properties[0]}"
               for labels, properties, state, kind in neo.run(
                   "SHOW INDEXES YIELD labelsOrTypes, properties, state, type "
                   "RETURN labelsOrTypes, properties, state, type")
               if labels and properties and state == "ONLINE" and kind == "RANGE"}
    missing = expected - present
    if missing:
        raise SystemExit(
            f"--reuse-neo4j was given, but Neo4j is missing "
            f"{len(missing)} of the {len(expected)} schema indexes: "
            f"{sorted(missing)[:5]}. Measuring an unindexed Neo4j against "
            f"an indexed Samyama is the bug this module was fixed for. "
            f"Drop the flag to reload.")
    click.echo(f"  neo4j   : reused, {held:,} nodes / {edges:,} edges and "
               f"{len(expected)} indexes verified")


@click.command()
@click.option("--neo4j", "neo4j_url", default="http://127.0.0.1:7474", show_default=True)
@click.option("--user", default="neo4j", show_default=True)
@click.option("--password", envvar="NEO4J_PASSWORD", required=True,
              help="Neo4j password, or set NEO4J_PASSWORD -- the variable the "
                   "module docstring's docker line starts Neo4j with. No "
                   "default: a built-in one had to be published somewhere to "
                   "be usable, and was, while this help claimed to hide it.")
@click.option("--database", default=NEO4J_DB, show_default=True,
              help="Neo4j database to load and query. Host, user and password "
                   "were configurable and this was not.")
@click.option("--repeats", default=15, show_default=True, type=int,
              help="Timed runs per query; the median is reported. 15 because "
                   "that is what every published figure in this module's "
                   "docstring was measured at -- the default was 5, so the "
                   "documented command reproduced none of them.")
@click.option("--warmup", default=10, show_default=True, type=int,
              help="Runs discarded before timing. One is not enough: a "
                   "freshly-started Neo4j measured 12.1 ms on EA01 where one "
                   "that had been serving queries for an hour measured 6.0 ms, "
                   "and the headline ratio moved 2.6x -> 1.6x on that alone.")
@click.option("--scale", default=1.0, show_default=True, type=float)
@click.option("--seed", default=20260814, show_default=True, type=int)
@click.option("--natural", is_flag=True,
              help="Also time the hero query written the way a Neo4j author "
                   "would, to price the workarounds our shapes carry.")
@click.option("--force-wipe", is_flag=True,
              help="Delete everything in the Neo4j database before loading. "
                   "Required to re-run against a server that already holds "
                   "data; `load_neo4j` refuses otherwise. The refusal named "
                   "this flag before it existed, so the documented command "
                   "could not be run twice.")
@click.option("--reuse-neo4j", is_flag=True,
              help="Neo4j already holds this exact fleet; skip its load. The "
                   "embedded engine cannot be reused -- it is in-process and "
                   "in-memory, so it is loaded every run regardless.")
def main(neo4j_url, user, password, database, repeats, warmup, scale, seed,
         natural, reuse_neo4j, force_wipe):
    # Before any connection: an argument contradiction should not require a
    # running Neo4j to discover, and the refusal below is about the flags
    # rather than about the server.
    if reuse_neo4j and force_wipe:
        raise SystemExit(
            "--reuse-neo4j and --force-wipe contradict each other: one keeps "
            "the database as it is, the other deletes it. Taken together "
            "--reuse-neo4j wins silently and nothing is wiped, which is the "
            "opposite of what asking for both suggests.")

    from samyama import SamyamaClient

    from benchmarks.queries import BY_ID
    from etl import generate as gen
    from etl import onnx_catalog as oc
    from etl import real_layer

    neo = Neo4j(neo4j_url, user, password, database)
    # Reachable and authenticated, first, and fatal if not. The version probe
    # below used to be the first contact, and it caught everything -- so a
    # wrong --password or a stopped container printed `unknown (...)`, then
    # generated a whole fleet and loaded Samyama before failing, or worse,
    # reached the sweep and printed a table of ERR rows. `Neo4j.run` raises
    # `Neo4jLoadError` for exactly these (HTTP 401/403, connection refused)
    # with a message saying which flag to check.
    try:
        neo.run("RETURN 1")
    except Neo4jLoadError as exc:
        raise SystemExit(str(exc)) from exc
    # Only a *query* error is tolerated here -- the server is known to be up,
    # and a build that refuses `dbms.components` is still usable. An empty
    # answer is guarded too: it would otherwise end the run on a bare
    # IndexError, over a line that only labels the output.
    try:
        probe = neo.run("CALL dbms.components() YIELD versions RETURN versions[0]")
        version = probe[0][0] if probe and probe[0] else "unknown"
    except RuntimeError as exc:
        version = f"unknown ({type(exc).__name__})"
    click.echo(f"neo4j    : {version}")
    sam = SamyamaClient.embedded()
    click.echo(f"samyama  : {sam.status().version} (embedded)")

    operators = oc.load_cached()          # once: both calls want the same set
    fleet = gen.generate(seed=seed, scale=scale, operators=operators)
    real_layer.build_real(fleet, operators)
    nodes = sum(len(v) for v in fleet.nodes.values())
    click.echo(f"fleet    : {nodes:,} nodes / {len(fleet.edges):,} edges "
               f"(seed {seed}, scale {scale})")

    click.echo("\nloading (reported for scale, not as the comparison -- see "
               "the module docstring)")
    click.echo(f"  samyama : {load_samyama(sam, fleet):7.1f} s")
    try:
        if reuse_neo4j:
            _verify_reused(neo, nodes, len(fleet.edges))
        else:
            click.echo(f"  neo4j   : "
                       f"{load_neo4j(neo, fleet, force_wipe=force_wipe):7.1f} s")
    except Neo4jLoadError as exc:
        # The module raises one type rather than exiting, so a caller can
        # handle a bad fleet. This is the caller that cannot -- on either
        # path: the `--reuse-neo4j` counts used to run unguarded, so a server
        # that went away mid-run ended in a traceback.
        raise SystemExit(str(exc)) from exc

    # A whole-catalog pass before anything is timed, because per-query warm-up
    # is not enough on a JVM. Measured: in one run with 10 per-query warm-ups,
    # `EA01` came back at 11.6 ms in the sweep and 6.2 ms when `--natural`
    # re-ran it at the end -- 1.9x apart, same query, same process, differing
    # only in how much work the JVM had done by then. Per-query warm-up warms
    # that query's plan and page cache; it does not warm the process.
    click.echo(f"\nwarming both engines over the whole catalog "
               f"({warmup} passes) ...")
    for pass_number in range(warmup):
        for spec in BY_ID.values():
            for run in (lambda c: sam.query(c, GRAPH).records, neo.run):
                try:
                    run(spec["cypher"])
                except Neo4jLoadError as exc:
                    # Not swallowed with the rest: this is the class the client
                    # raises for a server that is unreachable or refusing auth,
                    # and warming on through it would spend `warmup` passes
                    # failing silently before the sweep reported every Neo4j
                    # query as an error.
                    raise SystemExit(
                        f"neo4j stopped answering during warm-up: {exc}") from exc
                except Exception:  # a query one engine refuses still warms the other
                    pass
        # A default run is `warmup` passes over the catalog on two engines, then
        # `warmup` more per query before the first timed one -- some hundreds of
        # executions with nothing printed. A dot per pass is enough to tell a
        # slow run from a hung one.
        click.echo(".", nl=False)
    click.echo(f" {warmup * len(BY_ID) * 2:,} executions")

    click.echo(f"\nquery latency, median of {repeats} after a {warmup}-pass "
               f"catalog warm-up and {warmup} further per-query runs\n")
    click.echo(f"{'query':<8}{'samyama':>12}{'neo4j':>12}{'ratio':>10}   rows")
    result = compare_catalog(lambda c: sam.query(c, GRAPH).records, neo.run,
                             BY_ID, repeats, warmup, echo=click.echo)
    if any(result["not_asked"].values()):
        # A sweep that silently skipped a query would drop exactly the ones
        # that error -- the ones a reader most wants -- so the run says so.
        click.echo(f"\nNOT ASKED: {result['not_asked']}. The table above is "
                   f"not the whole catalog.")
    wins, losses = result["wins"], result["losses"]
    parity, errors, content = result["parity"], result["errors"], result["content"]

    click.echo(f"\nsamyama faster on {len(wins)}: {wins}")
    click.echo(f"neo4j faster on {len(losses)}: {losses}"
               + ("   <- published, per #47" if losses else ""))
    click.echo(f"too close to call (within {PARITY_BAND:.0%}) on {len(parity)}: "
               f"{parity}")
    if result["counts"]:
        click.echo(f"\n*** ROW COUNTS DIFFER on {result['counts']} -- the two "
                   f"engines are not answering\n    the same question here, so "
                   f"the timings for these are not comparable. ***")
    if content:
        click.echo(f"\nsame row count, different rows: {content}")
        # Printed for the queries actually in the bucket, not as a fixed
        # paragraph. It used to name `EA10` unconditionally, and to explain
        # every entry as a tie -- so a genuinely wrong answer landing here
        # would have been captioned as harmless.
        ties = [qid for qid in content if "LIMIT" in BY_ID[qid]["cypher"]]
        if ties:
            click.echo(f"  {ties} sort on a column with ties and then take a "
                       f"LIMIT, so each\n  engine returns a different "
                       f"arbitrary N of many equal-ranked rows -- the\n  same "
                       f"answer, printed differently.")
        rest = [qid for qid in content if qid not in ties]
        if rest:
            click.echo(f"  *** {rest} differ with no LIMIT to explain it. That "
                       f"is two engines\n      giving different answers to the "
                       f"same question -- read the rows before\n      reading "
                       f"the timings. ***")
    if errors:
        # Its own bucket in `compare_catalog`, disjoint from `unstable`: an
        # engine refusing a query and an engine answering inconsistently are
        # different findings.
        click.echo(f"errored on one side: {errors}")
    if result["unstable"]:
        click.echo(f"unstable across repeats on {len(result['unstable'])}: "
                   f"{result['unstable']} -- each ran every time and answered "
                   f"differently between runs, so there is no single result to "
                   f"compare. Ties under `ORDER BY ... LIMIT` are the usual "
                   f"cause.")
    if result["empty"]:
        click.echo(f"both engines returned no rows on {len(result['empty'])}: "
                   f"{result['empty']} -- timed, but no verdict: a ratio over "
                   f"two empty answers says nothing about either")
    if result["unresolved"]:
        click.echo(f"below timer resolution on {len(result['unresolved'])}: "
                   f"{result['unresolved']} -- raise --repeats or --scale; "
                   f"these are excluded from the win/loss counts rather than "
                   f"counted as infinite speedups")

    if natural:
        click.echo("\n--- what our workarounds cost, on the hero query ---")
        click.echo("`EA01` as it ships is shaped around engine notes 3 and 5. "
                   "Neo4j needs neither.\nSame question, both spellings, both on "
                   "Neo4j:")
        drift = natural_ea01_matches_the_catalog(BY_ID)
        if drift:
            click.echo("  NATURAL_EA01 no longer asks EA01's question, so the "
                       "two are not\n  comparable: " + "; ".join(drift))
            return
        shaped_ms, shaped_n, shaped_rows = timed(neo.run, BY_ID["EA01"]["cypher"],
                                                 repeats, warmup)
        nat_ms, nat_n, nat_rows = timed(neo.run, NATURAL_EA01, repeats, warmup)
        click.echo(f"  neo4j, our shape      {shaped_ms:7.1f} ms   {shaped_n} rows")
        click.echo(f"  neo4j, natural shape  {nat_ms:7.1f} ms   {nat_n} rows")
        # `timed` signals both of its non-results by returning a *string*
        # where the row count goes: `ERR(...)` for a failure and `UNSTABLE ...`
        # for repeats that disagreed. `compare_catalog` tests for that with
        # `isinstance(..., str)`; this path did not, and each string slipped
        # through differently. `ERR` is truthy, so the empty-rows branch missed
        # it and `isnan` caught it one branch later. `UNSTABLE` carries a real
        # median, so `isnan` missed it too -- and when both spellings were
        # unstable their empty row lists compared equal, and a ratio was
        # printed over two measurements that had already declared themselves
        # unusable. One check for both, first, before anything reads the
        # numbers.
        if isinstance(shaped_n, str) or isinstance(nat_n, str):
            click.echo(f"  no usable measurement -- shaped: {shaped_n}, "
                       f"natural: {nat_n}. No ratio to print.")
        elif not shaped_n or not nat_n:
            click.echo(f"  one spelling returned no rows (shaped {shaped_n}, "
                       f"natural {nat_n}); a ratio over an empty result says "
                       f"nothing about either.")
        elif not same_rows(shaped_rows, nat_rows):
            click.echo("  WARNING: the two spellings return different rows, so "
                       "this is not a\n           like-for-like comparison and the "
                       "ratio means nothing.")
        elif shaped_ms and nat_ms:
            # Guard the denominator that is actually divided, and let the word
            # follow the number. Both were wrong: the guard tested `nat_ms`
            # while the division was by `shaped_ms`, and "FASTER" was printed
            # unconditionally, so a result showing our shape slower would have
            # been announced as a win.
            ratio = nat_ms / shaped_ms
            if ratio == 1:
                click.echo(f"  same {nat_n} rows; the two spellings measured "
                           f"the same on Neo4j")
            else:
                verdict = "FASTER" if ratio > 1 else "SLOWER"
                click.echo(f"  same {nat_n} rows; our shape is "
                           f"{ratio if ratio > 1 else 1 / ratio:.2f}x {verdict} on "
                           f"Neo4j than the idiomatic one")
        else:
            click.echo("  one of the two measured 0 ms; no ratio worth printing")


if __name__ == "__main__":
    main()
