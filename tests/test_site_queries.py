"""`EA20` against ground truth: both columns, per site, recomputed in Python.

Split from `tests/test_site_spine.py`, which crossed the 500 lines at which
the review harness stops reading a file -- and a file nobody reads is a file
nobody reviews. The seam is the one the module already had: that file is the
generator's invariants, which need no engine, and this one is the query's
answer, which needs a loaded graph.

The helpers are imported rather than copied. `RECALLED_BOARD` and `COL` are
both *read out of the catalog Cypher*, so a second copy here would be a second
source for one fact -- exactly what those two exist to avoid.
"""
from __future__ import annotations

import collections

import pytest

from benchmarks.queries import BY_ID
from tests.test_site_spine import (
    COL,
    GRAPH,
    NODE_LABELS,
    RECALLED_BOARD,
    create_edges,
    create_nodes,
    deployed_at,
    gen,
    loaded,
    on_board,
    operators,
)

__all__ = ["loaded", "operators"]      # re-exported fixtures, used by the tests



def test_ea20_counts_match_ground_truth(loaded):
    """Both columns, per site, recomputed from the fleet.

    Checked as a mapping rather than a list: the row *order* is the query's
    claim about ranking and is asserted separately below, and comparing both
    at once would make a wrong count look like a wrong sort.
    """
    client, fleet = loaded
    records = client.query(BY_ID["EA20"]["cypher"].strip(), GRAPH).records

    placements, boards = deployed_at(fleet), on_board(fleet)
    # One board per deployment, checked here rather than assumed: the ground
    # truth below counts a deployment as "on the recalled board" from this
    # mapping, and a deployment with two `ON_BOARD` edges would keep only the
    # last while `EA20` counted rows for both.
    board_edges = collections.Counter(
        src for _sl, src, rel, _tl, _tgt, _p in fleet.edges if rel == "ON_BOARD")
    doubled = {did: k for did, k in board_edges.items() if k > 1}
    assert not doubled, f"deployments on more than one board: {sorted(doubled)[:3]}"
    by_site = {s["id"]: s for s in fleet.nodes["Site"]}
    here: dict[str, int] = collections.Counter()
    recalled: dict[str, int] = collections.Counter()
    for did, site_id in placements.items():
        here[site_id] += 1
        if boards.get(did) == RECALLED_BOARD:
            recalled[site_id] += 1

    truth = {(by_site[sid]["campus"], by_site[sid]["name"]):
             (recalled[sid], here[sid]) for sid in here}
    # Keyed by `(campus, name)` on both sides, so a duplicated pair would
    # collapse two rows into one in *both* dicts and the comparison would
    # still pass -- losing a site each way and saying nothing. The distinctness
    # itself is `test_site_names_stay_distinct_where_the_suffix_list_wraps`;
    # what is checked here is that this comparison saw every row it was given.
    assert len(truth) == len(here), (
        f"two sites share a (campus, name) in the fixture: {len(here)} sites "
        f"collapsed to {len(truth)} keys. This comparison would merge them "
        f"and pass while under-reporting both.")
    got = {(row[COL["campus"]], row[COL["site"]]):
           (row[COL["on_recalled_board"]], row[COL["deployments_here"]])
           for row in records}
    assert len(got) == len(records), (
        f"EA20 returned {len(records)} rows that collapse to {len(got)} "
        f"(campus, name) keys; the duplicates would be compared as one.")
    assert got == truth, (
        f"EA20 disagrees with the fleet. query={got}\nfleet={truth}")


def test_ea20_ranks_the_worst_hit_site_first(loaded):
    """The ordering is the answer to "where do I start", so it is asserted.

    Engine note 3 says an `ORDER BY` on a RETURN-introduced alias is silently
    dropped -- a run that ignored the sort would return these same rows in
    load order and look fine.
    """
    client, _fleet = loaded
    records = client.query(BY_ID["EA20"]["cypher"].strip(), GRAPH).records
    counts = [row[COL["on_recalled_board"]] for row in records]
    assert counts == sorted(counts, reverse=True), (
        f"EA20 came back in {counts}, not descending; the site with the most "
        f"recalled boards is the one an ops team pages first")


def test_ea20_separates_a_site_wide_failure_from_one_device(loaded):
    """The question in the title, answered on the fixture rather than argued.

    At this fixture's scale every site holds at least one of the recalled
    boards -- 4 sites and 36 boards leave nowhere for a clean miss -- so what
    is asserted here is what this graph can actually show: a site where *some*
    but not all deployments are affected, and no site claiming more hits than
    it has deployments. The unaffected case needs the shipped graph and is
    checked by the `--full-scale` test below.
    """
    client, _fleet = loaded
    records = client.query(BY_ID["EA20"]["cypher"].strip(), GRAPH).records
    fractions = {(row[COL["campus"]], row[COL["site"]]):
                 (row[COL["on_recalled_board"]], row[COL["deployments_here"]])
                 for row in records}
    assert fractions, "EA20 returned nothing against a loaded fixture"
    impossible = {k: v for k, v in fractions.items() if v[0] > v[1]}
    assert not impossible, (
        f"sites reporting more recalled boards than deployments: {impossible}. "
        f"`on_recalled_board` counts rows and `deployments_here` counts "
        f"distinct ids, so this is what a multiplied join looks like.")
    assert any(0 < hit < total for hit, total in fractions.values()), (
        f"no partially affected site in {fractions}; without one, "
        f"'site-wide or one device' is not a distinction this row can make")


def test_ea20_shows_both_affected_and_untouched_sites_at_full_scale(
        request, engine_factory, operators):
    """The distinction the issue asks for, on the graph the docs describe.

    Gated on `--full-scale` because it builds the shipped fleet (seed 20260814,
    `--scale 1.0`) rather than this module's small one. That flag landed with
    #108 and had no caller; this is one, which is also the answer to "why is it
    registered if nothing reads it".

    Measured on that graph: of 12 sites, **6 held none of `board:00003` and 6
    held some but not all** -- so "replace one unit here" and "leave that site
    alone" are both readable off the same column.

    The ratio is deliberately not asserted. It moves with the upstream ONNX
    catalogue as well as with the seed -- the catalogue decides kernel
    coverage, which decides board fit, which decides placement -- and the same
    seed gives 6 untouched on a 205-operator catalogue against 5 on the
    379-operator one. What is asserted is that **both kinds exist**, which is
    the distinction the question turns on and the only part that is ours.
    """
    if not request.config.getoption("--full-scale"):
        pytest.skip("needs --full-scale: builds the shipped fleet at scale 1.0")

    from etl import real_layer

    client = engine_factory()
    full = gen.generate(seed=gen.DEFAULT_SEED, scale=1.0, operators=operators)
    real_layer.build_real(full, operators)
    for label in NODE_LABELS:
        if full.nodes.get(label):
            create_nodes(client, GRAPH, label, full.nodes[label])
    create_edges(client, GRAPH, full.edges)

    records = client.query(BY_ID["EA20"]["cypher"].strip(), GRAPH).records
    counts = [(row[COL["on_recalled_board"]], row[COL["deployments_here"]])
              for row in records]
    assert any(hit == 0 for hit, _total in counts), (
        f"every site at scale 1.0 holds a recalled board: {counts}. The "
        f"'is it site-wide' question needs an untouched site to contrast with.")
    assert any(0 < hit < total for hit, total in counts), (
        f"no partially affected site at scale 1.0: {counts}")
