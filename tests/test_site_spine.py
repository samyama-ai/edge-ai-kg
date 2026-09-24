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
import pathlib
import re

import pytest

from benchmarks.queries import BY_ID
from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = "default"
SCALE = 0.3
SEED = 4242

def _recalled_board() -> str:
    """The board id `EA20` asks about, read from the query rather than restated.

    A hardcoded copy here would keep passing after the catalog moved on, which
    is the failure this avoids. The pattern is anchored on the whole predicate
    and fails loudly with the line it could not read, rather than raising
    `IndexError` from a bare `split` three frames away from the cause.
    """
    match = re.search(r'b\.id\s*=\s*"([^"]+)"', BY_ID["EA20"]["cypher"])
    assert match, (
        "EA20 no longer filters on a literal `b.id = \"...\"`. This module "
        "recomputes its ground truth from that id, so update this reader "
        "together with the query."
    )
    return match.group(1)


RECALLED_BOARD = _recalled_board()


def _columns() -> dict[str, int]:
    """`EA20`'s RETURN names, mapped to their positions in each row.

    Rows come back as bare tuples, so every reader here was an index: `row[3]`
    for `on_recalled_board` and `row[4]` for `deployments_here`. Those two are
    both integers and adjacent, so swapping the RETURN clause would leave every
    assertion in this module passing while comparing the wrong column against
    the wrong truth -- silently, because the ground truth would be rebuilt the
    same wrong way. Reading the clause once turns that into an immediate,
    loud failure.
    """
    match = re.search(r"^RETURN (.+)$", BY_ID["EA20"]["cypher"], re.MULTILINE)
    assert match, (
        "EA20 has no single-line RETURN clause; this module maps its columns "
        "by name, so update this reader together with the query.")
    names = [name.strip() for name in match.group(1).split(",")]
    return {name: i for i, name in enumerate(names)}


COL = _columns()


@pytest.fixture(scope="module")
def operators():
    try:
        return oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


@pytest.fixture(scope="module")
def fleet(operators):
    """Generated layer only -- the real layer has no sites, by design.

    Nothing below may mutate this. `build_real` appends to the `Fleet` it is
    given, so a fixture that called it on this object would leave the
    invariant tests reading a two-layer graph -- and passing or failing by
    which test ran first. `loaded` builds its own fleet for that reason.
    """
    return gen.generate(seed=SEED, scale=SCALE, operators=operators)


@pytest.fixture(scope="module")
def loaded(engine_factory, operators):
    """An equivalent fleet in an embedded engine, plus the real layer.

    Its **own** `Fleet`, not the `fleet` fixture: `build_real` mutates what it
    is handed, and sharing one object made the invariant tests above depend on
    whether this had run yet. Same seed and scale, so it is the same generated
    graph.

    Both layers, because `EA20` sweeps every `Deployment` in the graph and the
    real layer contributes 73 of them at `--scale 1.0`. If those were reachable
    from a `Site` the ground truth below would be wrong; the point of loading
    them is that they are not.
    """
    from etl import real_layer

    client = engine_factory()
    fleet = gen.generate(seed=SEED, scale=SCALE, operators=operators)
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


def test_site_names_stay_distinct_where_the_suffix_list_wraps():
    """Above `--scale 1.04` the suffix list runs out, and `EA20` groups by name.

    `SITE_SUFFIXES` holds 12 names, so the 13th site is where the wrap
    begins.
    Without the wrap number it is a second `Ward 3` in the same campus as the
    first -- and `EA20` groups by `(campus, name)`, so the two merge into one
    row and each under-reports the other's deployments.

    Checked over a range rather than at one scale, because the collision moves
    with the campus cycle: every site must be distinct by id, and distinct by
    name *within* its campus.
    """
    from etl import sites as sites_mod

    for count in (13, 24, 25, 40):
        built = sites_mod.build_sites(count, lambda kind, i: f"{kind}:{i:05d}")
        assert len({s["id"] for s in built}) == len(built), (
            f"{count} sites do not have distinct ids")
        by_campus = collections.defaultdict(list)
        for site in built:
            by_campus[site["campus"]].append(site["name"])
        clashes = {campus: names for campus, names in by_campus.items()
                   if len(set(names)) != len(names)}
        assert not clashes, (
            f"at {count} sites, these campuses carry a repeated name: "
            f"{ {c: sorted(n) for c, n in clashes.items()} }. `EA20` groups by "
            f"(campus, name), so the rows merge and both are under-reported.")


def test_a_tiny_scale_still_has_two_places_to_compare():
    """Below about `--scale 0.125` the rounding gave one site in one campus.

    One place makes "is this site-wide or one device" unanswerable by
    construction -- every row is site-wide when there is nowhere else -- so
    `MIN_SITES` floors it. The floor is a property of `build_sites`, checked
    here at the scales that used to collapse rather than at the shipped one.
    """
    from etl import sites as sites_mod

    for scaled_count in (0, 1):
        built = sites_mod.build_sites(scaled_count, lambda kind, i: f"{kind}:{i:05d}")
        assert len(built) >= sites_mod.MIN_SITES, (
            f"a scaled count of {scaled_count} gave {len(built)} sites")
        assert len({s["campus"] for s in built}) >= 2, (
            f"a scaled count of {scaled_count} gave one campus "
            f"({[s['campus'] for s in built]}); the campus list must cycle "
            f"before the floor is reached, or the floor buys nothing")


def test_placing_sites_did_not_move_any_deployment_metric(operators):
    """The `Site` draw must not consume from the generator's shared stream.

    A `rng.choice(sites)` in the deployment loop shifts every later draw, so
    a location silently changes the cost model: at seed 20260814 that spelling
    gives `deploy:00001` latency 78.036 / power 3591.96 against the 87.684 /
    3990.6 below. `etl/generate.py`'s contract is that a seed reproduces the
    graph byte-for-byte, and `docs/data-provenance.md`'s figures rest on it.

    The values below are that contract, recorded from `main` before `Site`
    existed. If this fails, either a draw was added to the shared stream --
    which is a bug -- or the cost model changed on purpose, in which case these
    move together with the published figures.
    """
    full = gen.generate(seed=gen.DEFAULT_SEED, scale=1.0, operators=operators)
    first = {row["id"]: row for row in full.nodes["Deployment"][:2]}
    assert (first["deploy:00000"]["latency_ms"],
            first["deploy:00000"]["power_mw"]) == (309.291, 769.61)
    assert (first["deploy:00001"]["latency_ms"],
            first["deploy:00001"]["power_mw"]) == (87.684, 3990.6), (
        "deployment metrics moved for an unchanged seed. If a new field draws "
        "from the shared `rng`, give it its own `random.Random` as the site "
        "placement has.")


def test_sites_are_deterministic_from_the_seed(operators):
    """Same seed, same placement -- the graph is reproducible or it is nothing."""
    again = gen.generate(seed=SEED, scale=SCALE, operators=operators)
    once = gen.generate(seed=SEED, scale=SCALE, operators=operators)
    assert deployed_at(again) == deployed_at(once)
    assert [s["id"] for s in again.nodes["Site"]] == [s["id"] for s in once.nodes["Site"]]


def test_the_real_layer_gets_no_sites(operators):
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


# Every fleet scale the repo documents (`README.md`, `docs/volume.md`,
# `CLAUDE.md`), and whether `EA20`'s `LIMIT` covers it. Measured, not assumed:
# sites scale linearly with the fleet, so the limit that fits the shipped graph
# stops fitting above it.
DOCUMENTED_SCALES = {0.15: 2, 0.3: 4, 1.0: 12, 5.0: 60, 10.0: 120}


def test_ea20_keeps_every_site_up_to_the_shipped_scale_and_says_where_it_stops(
        operators):
    """`LIMIT 12` covers the shipped graph, and truncates above it -- both pinned.

    `EA20` orders by `on_recalled_board`, where ties are common: most sites
    hold none of the recalled board at all. Truncating a tie is how `EA02` and
    `EA11` came to be withdrawn from the Neo4j comparison as unstable, so the
    limit is measured against the site count rather than assumed generous.

    The earlier version of this test checked `--scale 1.0` alone while its name
    claimed "shipped scales", which hid the boundary. Measured across every
    scale the docs use: 2 sites at 0.15, 4 at 0.3, 12 at 1.0 -- all inside the
    limit -- then **60 at 5.0 and 120 at 10.0, which the limit cuts**. That is
    a real bound on the query rather than a bug in the fixture, and
    `benchmarks/queries.py` says so where the limit is written; this test fails
    if either half of that claim stops being true.
    """
    limit = int(BY_ID["EA20"]["cypher"].rsplit("LIMIT", 1)[1])

    # The note is read, not assumed. This test told the reader that
    # `benchmarks/queries.py` documents the boundary while the entry carried
    # no comment at all -- a test asserting a claim the code does not make,
    # which is worse than either alone because the cross-reference looks
    # checked. Source text rather than the dict: a `#` comment is not data.
    catalog_source = (ROOT / "benchmarks" / "queries.py").read_text(encoding="utf-8")
    start = catalog_source.index('"id": "EA20"')
    entry = catalog_source[start:catalog_source.index("\n    },", start)]
    for expected in ("truncates above it", str(max(DOCUMENTED_SCALES)),
                     str(DOCUMENTED_SCALES[max(DOCUMENTED_SCALES)])):
        assert expected in entry, (
            f"`benchmarks/queries.py`'s EA20 entry does not mention "
            f"{expected!r}. This test points a reader there for the "
            f"truncation boundary, so the note has to exist, and has to move "
            f"when the limit or the site density does.")

    # The cached operators this module already has -- `operators=None` made the
    # generator re-read them from disk, a second source for the same data
    # inside one test run.
    for scale, expected in sorted(DOCUMENTED_SCALES.items()):
        built = gen.generate(seed=gen.DEFAULT_SEED, scale=scale,
                             operators=operators)
        count = len(built.nodes["Site"])
        assert count == expected, (
            f"--scale {scale} now generates {count} sites, not {expected}. "
            f"Site density changed, so re-measure this table and the "
            f"truncation note in `benchmarks/queries.py` with it.")
        if scale <= 1.0:
            assert count <= limit, (
                f"--scale {scale} generates {count} sites and EA20 keeps "
                f"{limit}; the rows past the limit would be an arbitrary "
                f"choice among equal `on_recalled_board` values")
        else:
            assert count > limit, (
                f"--scale {scale} now fits inside EA20's LIMIT {limit}. If the "
                f"limit was raised, say so in `benchmarks/queries.py` and move "
                f"this boundary rather than leaving the comment stale.")


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

    Measured on that graph: of 12 sites, **6 hold none of `board:00003` and 6
    hold some but not all** -- so "replace one unit here" and "leave that site
    alone" are both readable off the same column. The numbers are not asserted
    exactly, because they move with the seed; what is asserted is that both
    kinds exist.
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
