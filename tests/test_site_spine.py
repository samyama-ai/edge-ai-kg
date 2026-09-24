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


def _generator_from(ref: str):
    """`etl/generate.py` as of `ref`, importable beside the current one.

    Skips rather than fails when git or the ref is unavailable: the point is
    to compare this branch against the code before `Site` existed, and a
    shallow clone or a detached export simply cannot do that.
    """
    import importlib.util
    import subprocess
    import sys

    try:
        source = subprocess.run(
            ["git", "show", f"{ref}:etl/generate.py"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"cannot read etl/generate.py at {ref}: {exc}")

    path = ROOT / ".pytest_cache" / f"generate_at_{ref.replace('/', '_')}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    name = f"generate_at_{ref.replace('/', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the module defines dataclasses, and
    # `@dataclass` resolves `cls.__module__` through `sys.modules`.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_placing_sites_did_not_move_any_deployment_metric(operators):
    """The `Site` draw must not consume from the generator's shared stream.

    A `rng.choice(sites)` in the deployment loop shifts every later draw, so a
    location silently changes the cost model. `etl/generate.py`'s contract is
    that a seed reproduces the graph byte-for-byte, and
    `docs/data-provenance.md`'s figures rest on it.

    **Compared against the generator before `Site` existed, not against pinned
    numbers.** This test used to assert `deploy:00000` reads 309.291 / 769.61,
    recorded from `main`. Those values are a function of the upstream ONNX
    catalogue, which is not pinned: the catalogue moved to 379 operators, the
    figures became 59.141 / 6.58, and the test failed on a fresh checkout while
    every cached checkout stayed green -- a pin that rots on someone else's
    release schedule, hidden by our own stale `data/`.

    Both runs here are given the *same* operator list, so the catalogue cancels
    out and what is left is the question actually worth asking: does adding
    `Site` move any deployment metric? Measured over all 1,440, not two.
    """
    before = _generator_from("origin/main")
    old = {row["id"]: (row["latency_ms"], row["power_mw"])
           for row in before.generate(seed=before.DEFAULT_SEED, scale=1.0,
                                      operators=operators).nodes["Deployment"]}
    new = {row["id"]: (row["latency_ms"], row["power_mw"])
           for row in gen.generate(seed=gen.DEFAULT_SEED, scale=1.0,
                                   operators=operators).nodes["Deployment"]}

    assert len(old) > 1000, (
        f"the pre-Site generator produced {len(old)} deployments; this "
        f"comparison needs the full fleet to be worth anything")
    assert set(old) == set(new), (
        f"the deployment set itself changed: "
        f"{sorted(set(new) ^ set(old))[:3]} differ between the two generators")
    moved = {did: (old[did], new[did]) for did in old if old[did] != new[did]}
    assert not moved, (
        f"{len(moved)} of {len(old)} deployment metrics moved when `Site` was "
        f"added, e.g. {list(moved.items())[:2]}. Site placement is drawing "
        f"from the shared `rng`, so every draw after it shifts: give the new "
        f"field its own `random.Random`, as `site_rng` has."
    )


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
# `CLAUDE.md`), and the sites each one generates. Measured, not assumed --
# and catalogue-independent: site count follows `--scale`, not the operator
# list, so unlike the cost-model figures these do not rot when upstream
# publishes. Checked by generating with a halved catalogue: 12 sites either
# way.
DOCUMENTED_SCALES = {0.15: 2, 0.3: 4, 1.0: 12, 5.0: 60, 10.0: 120}


def test_ea20_has_no_limit_so_no_affected_site_can_fall_off(operators):
    """A recall answer that omits a site is wrong, not abbreviated.

    `EA20` carried `LIMIT 12`, which covered the shipped 12 sites and cut
    everything above. At `--scale 5.0` there are 60 sites and the 12th
    `on_recalled_board` value is shared by 9 of them, so sites *running the
    recalled board* fell past the cut -- and which ones fell could differ
    between engines, the same tie instability that had `EA02` and `EA11`
    withdrawn from the Neo4j comparison. A second `ORDER BY` key cannot break
    that tie (engine note 3b: only the first key is honoured).

    So the limit is gone rather than raised: a number large enough today is
    the same defect waiting for a bigger fleet. The untouched sites stay in
    the result too -- filtering to `on_recalled_board > 0` would drop the
    "leave that site alone" half of the question `docs/location-scope.md`
    argues this query answers.

    What this pins: the query has no `LIMIT`, the reason is written where the
    query is, and the site counts the reason quotes are still true.
    """
    cypher = BY_ID["EA20"]["cypher"]
    assert "LIMIT" not in cypher.upper(), (
        f"EA20 has a LIMIT again:\n{cypher}\nAt --scale 5.0 the 12th "
        f"`on_recalled_board` value is shared by 9 sites, so a limit drops "
        f"affected sites and which ones is engine-dependent. If a cap is "
        f"genuinely needed, it has to come with a tiebreaker the engine "
        f"honours -- engine note 3b says a second ORDER BY key is not one.")

    # The reason is read, not assumed. An earlier version of this test told
    # the reader that `benchmarks/queries.py` documents the boundary while the
    # entry carried no comment at all -- a test asserting a claim the code does
    # not make. Source text, because a `#` comment is not data.
    catalog_source = (ROOT / "benchmarks" / "queries.py").read_text(encoding="utf-8")
    start = catalog_source.index('"id": "EA20"')
    entry = catalog_source[start:catalog_source.index("\n    },", start)]
    for expected in ("No `LIMIT`", str(max(DOCUMENTED_SCALES)),
                     str(DOCUMENTED_SCALES[max(DOCUMENTED_SCALES)])):
        assert expected in entry, (
            f"`benchmarks/queries.py`'s EA20 entry does not mention "
            f"{expected!r}. This test points a reader there for why the query "
            f"is unbounded, so the note has to exist and has to move when the "
            f"site density does.")

    # The cached operators this module already has -- `operators=None` made the
    # generator re-read them from disk, a second source for the same data
    # inside one test run.
    for scale, expected in sorted(DOCUMENTED_SCALES.items()):
        built = gen.generate(seed=gen.DEFAULT_SEED, scale=scale,
                             operators=operators)
        count = len(built.nodes["Site"])
        assert count == expected, (
            f"--scale {scale} now generates {count} sites, not {expected}. "
            f"Site density changed, so re-measure this table and the note in "
            f"`benchmarks/queries.py` with it.")
