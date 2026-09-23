"""The `Site` spine (#34): the generator's invariants, and `EA20` against truth.

`EA20` answers "is this a site-wide problem or one device", which is a
*comparison* of two counts on the same row -- how many deployments a site holds
against how many of them are on the recalled board. A query that returns
plausible rows is not evidence either count is right, so both are recomputed
here from the `Fleet` in Python and compared row by row, the way
`tests/test_correctness.py` checks the rest of the catalog.

The generator invariants are checked separately and without an engine, because
they are what the query rests on: one site per deployment, sites that exist at
every shipped scale, and more than one campus to group by. A deployment in two
places, or none, turns `deployments_here` from a count into a sum over an
unknown multiplicity and nothing in the query would say so.
"""
from __future__ import annotations

import collections

import pytest

from benchmarks.queries import BY_ID
from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
SCALE = 0.3
SEED = 4242

# The board `EA20` asks about, read from the query rather than restated: a test
# that hardcoded its own id would keep passing after the catalog moved on.
RECALLED_BOARD = BY_ID["EA20"]["cypher"].split('b.id = "')[1].split('"')[0]


@pytest.fixture(scope="module")
def operators():
    try:
        return oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


@pytest.fixture(scope="module")
def fleet(operators):
    """Generated layer only -- the real layer has no sites, by design."""
    return gen.generate(seed=SEED, scale=SCALE, operators=operators)


@pytest.fixture(scope="module")
def loaded(engine_factory, operators, fleet):
    """That same fleet in an embedded engine, plus the real layer.

    Both layers, because `EA20` sweeps every `Deployment` in the graph and the
    real layer contributes 73 of them at `--scale 1.0`. If those were reachable
    from a `Site` the ground truth below would be wrong; the point of loading
    them is that they are not.
    """
    from etl import real_layer

    client = engine_factory()
    real_layer.build_real(fleet, operators)
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client, fleet


def deployed_at(fleet) -> dict[str, str]:
    return {src: tgt for _sl, src, rel, _tl, tgt, _p in fleet.edges
            if rel == "DEPLOYED_AT"}


def on_board(fleet) -> dict[str, str]:
    return {src: tgt for _sl, src, rel, _tl, tgt, _p in fleet.edges
            if rel == "ON_BOARD"}


# --------------------------------------------------------------------------
# Generator invariants. No engine needed.


def test_every_generated_deployment_sits_at_exactly_one_site(fleet):
    """One place per deployment, which is what makes `deployments_here` a count."""
    placements = collections.Counter(
        src for _sl, src, rel, _tl, _tgt, _p in fleet.edges if rel == "DEPLOYED_AT")
    deployments = {d["id"] for d in fleet.nodes["Deployment"]}
    homeless = sorted(deployments - set(placements))
    assert not homeless, (
        f"{len(homeless)} deployments have no site ({homeless[:3]}). `EA20` "
        f"counts deployments *per site*, so these would vanish from every "
        f"row rather than showing up as an unplaced total.")
    multi = {did: k for did, k in placements.items() if k > 1}
    assert not multi, (
        f"deployments in more than one place: {sorted(multi)[:3]}. "
        f"`count(DISTINCT d.id)` would then be a set union across sites and "
        f"the per-site counts would sum to more than the fleet.")


def test_every_deployed_at_edge_points_at_a_real_site(fleet):
    site_ids = {s["id"] for s in fleet.nodes["Site"]}
    targets = set(deployed_at(fleet).values())
    assert targets <= site_ids, f"DEPLOYED_AT points at non-sites: {sorted(targets - site_ids)[:3]}"


def test_there_is_more_than_one_campus_to_group_by(fleet):
    """Below two campuses, "site-wide or one device" cannot be asked at all.

    At `--scale 0.3` the generator rounds to 4 sites; the check is on the
    campuses those sites carry, not on the site count, because the grouping is
    what the question needs.
    """
    campuses = {s["campus"] for s in fleet.nodes["Site"]}
    assert len(campuses) >= 2, (
        f"only {sorted(campuses)} at scale {SCALE}; `EA20` groups by campus, "
        f"so one campus makes every row site-wide by construction")


def test_sites_are_deterministic_from_the_seed(operators):
    """Same seed, same placement -- the graph is reproducible or it is nothing."""
    again = gen.generate(seed=SEED, scale=SCALE, operators=operators)
    once = gen.generate(seed=SEED, scale=SCALE, operators=operators)
    assert deployed_at(again) == deployed_at(once)
    assert [s["id"] for s in again.nodes["Site"]] == [s["id"] for s in once.nodes["Site"]]


def test_the_real_layer_gets_no_sites(operators, fleet):
    """Real MLPerf submissions have no known location, so they are given none.

    Inventing one would put a synthetic property on a node stamped
    `provenance: "real"`, which is the one thing this repo's provenance rule
    exists to prevent.
    """
    from etl import real_layer

    both = gen.generate(seed=SEED, scale=SCALE, operators=operators)
    before = set(deployed_at(both))
    real_layer.build_real(both, operators)
    after = deployed_at(both)
    real_ids = {d["id"] for d in both.nodes["Deployment"] if d.get("provenance") == "real"}
    assert real_ids, "no real deployments in the fixture; this test is not measuring anything"
    assert set(after) == before, (
        f"the real layer placed {sorted(set(after) - before)[:3]} at a site")
    assert not (real_ids & set(after)), "a real deployment was given a synthetic site"


def test_ea20_stays_inside_its_limit_at_shipped_scales(fleet):
    """`LIMIT 12` must not truncate, or the tail is an arbitrary pick among ties.

    `EA20` orders by `on_recalled_board`, where ties are common -- most sites
    hold none of the recalled board at all. Truncating a tie is how `EA02` and
    `EA11` came to be withdrawn from the Neo4j comparison as unstable, so the
    limit is checked against the site count rather than assumed generous.
    """
    limit = int(BY_ID["EA20"]["cypher"].rsplit("LIMIT", 1)[1])
    full = gen.generate(seed=gen.DEFAULT_SEED, scale=1.0, operators=None)
    assert len(full.nodes["Site"]) <= limit, (
        f"scale 1.0 generates {len(full.nodes['Site'])} sites and EA20 keeps "
        f"{limit}; the rows past the limit would be an arbitrary choice among "
        f"equal `on_recalled_board` values")


# --------------------------------------------------------------------------
# EA20 against ground truth computed in Python.


def test_ea20_counts_match_ground_truth(loaded):
    """Both columns, per site, recomputed from the fleet.

    Checked as a mapping rather than a list: the row *order* is the query's
    claim about ranking and is asserted separately below, and comparing both
    at once would make a wrong count look like a wrong sort.
    """
    client, fleet = loaded
    records = client.query(BY_ID["EA20"]["cypher"].strip(), GRAPH).records

    placements, boards = deployed_at(fleet), on_board(fleet)
    by_site = {s["id"]: s for s in fleet.nodes["Site"]}
    here: dict[str, int] = collections.Counter()
    recalled: dict[str, int] = collections.Counter()
    for did, site_id in placements.items():
        here[site_id] += 1
        if boards.get(did) == RECALLED_BOARD:
            recalled[site_id] += 1

    truth = {(by_site[sid]["campus"], by_site[sid]["name"]):
             (recalled[sid], here[sid]) for sid in here}
    got = {(row[0], row[1]): (row[3], row[4]) for row in records}
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
    counts = [row[3] for row in records]
    assert counts == sorted(counts, reverse=True), (
        f"EA20 came back in {counts}, not descending; the site with the most "
        f"recalled boards is the one an ops team pages first")


def test_ea20_separates_a_site_wide_failure_from_one_device(loaded):
    """The question in the title, answered on the fixture rather than argued.

    A site whose recalled count equals its deployment count is site-wide; one
    with a single hit is one device. Both shapes have to be *distinguishable*
    in the result, or the query answers nothing the issue asked for.
    """
    client, _fleet = loaded
    records = client.query(BY_ID["EA20"]["cypher"].strip(), GRAPH).records
    fractions = {(row[0], row[1]): (row[3], row[4]) for row in records}
    assert any(hit == 0 for hit, _total in fractions.values()), (
        "no unaffected site in the fixture, so the query cannot be shown to "
        "discriminate; widen the fleet or pick another board")
    assert any(0 < hit < total for hit, total in fractions.values()), (
        f"no partially affected site in {fractions}; without one, "
        f"'site-wide or one device' is not a distinction this row can make")
