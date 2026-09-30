"""`EA17` and `EA21` bound their `NEXT_STAGE` walks, and the bound has to stay safe.

From `samyama` 1.8.0 the planner refuses an unbounded variable-length pattern
producing over a million paths (engine note 14). The stage graph is tiny --
16 stages -- but it contains cycles, so *paths* explode while *depth* stays
small. `MAX_STAGE_HOPS` is the upper bound that makes the pattern legal.

A bound is only honest while the data stays shorter than it. `EA17` reports
how deep a blast radius reaches, so a chain longer than the bound would be
truncated and the answer would look complete. That is what #110 objected to
when it argued against a cap, and it is why the bound is checked here against
the fleet rather than asserted in a comment:

- the deepest chain reachable from any sensor must stay clear of the bound;
- every walk the repo ships -- the catalog, the demo and the MCP server --
  must carry an upper bound, so a new leg cannot be added unbounded and
  quietly re-break the 1.8 planner;
- and bounding must change no answer, which is asked of the engine rather
  than argued: each query is run both ways and the rows compared.
"""
from __future__ import annotations

import ast
import collections
import pathlib
import re

import pytest

from benchmarks.catalog.subjects import MAX_STAGE_HOPS
from benchmarks.queries import BY_ID
from tests.test_certification_alerts import (
    loaded_fleet,  # noqa: F401 -- shared fixture
)

GRAPH = "default"
HEADROOM = 2

# A relationship pattern carrying a variable-length `*`, whatever follows it.
# Whether it is *bounded* is `_is_unbounded`'s decision, not this pattern's.
VARIABLE_LENGTH = re.compile(r"\[[^\]]*?\*([^\]]*)\]")


def _is_unbounded(hop_range: str) -> bool:
    """Does what follows a `*` leave the walk with no upper hop limit?

    `*` and `*..` are unbounded. `*3`, `*0..3` and `*..3` are not. A `{...}`
    counts as a number, because that is how this repo templates the bound:
    `benchmarks/catalog/alerting.py` ships `*0..{hops}` and substitutes
    `MAX_STAGE_HOPS` at import, so reading the *source* rather than the
    assembled query means seeing the placeholder. That trade is worth making
    -- the assembled query cannot show a walk built inside a function, which
    is where `mcp_server/server.py` keeps all of its.

    The cost of it, stated rather than hidden: a genuine property map on a
    variable-length pattern (`[:R* {k: 1}]`) would be read as a hop count and
    called bounded. No query in this repo has one.
    """
    upper = hop_range.split("..")[-1] if ".." in hop_range else hop_range
    return upper.strip() == ""

# The files outside `benchmarks/catalog/` that ship Cypher a user runs.
SHIPPED_MODULES = ("demo/demo.py", "mcp_server/server.py")
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _cyphers_in_source(relative_path: str) -> dict[str, str]:
    """Every Cypher-looking string literal in a source file, by line number.

    Parsed rather than grepped, so a `*0..` inside a comment or a prose
    docstring is not reported as a query -- and so an f-string counts. That
    second half is why this reads the AST instead of module attributes: the
    MCP server builds each query as an f-string *inside* its tool function,
    where nothing module-level holds it, and a scan of module constants found
    exactly none of its seven queries while reporting success.
    """
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(part.value for part in node.values
                           if isinstance(part, ast.Constant)
                           and isinstance(part.value, str))
        else:
            continue
        if "MATCH" in text and "RETURN" in text:
            found[f"{relative_path}:{node.lineno}"] = text
    return found


@pytest.fixture(scope="module")
def stage_depths(loaded_fleet):  # noqa: F811 -- the imported fixture
    """Deepest `NEXT_STAGE` chain reachable from each sensor, by BFS.

    Depth, not path count: the graph has cycles, so counting paths would
    diverge while the question here -- how far downstream an answer has to
    reach -- is a shortest-path property.
    """
    _client, fleet = loaded_fleet
    successors = collections.defaultdict(list)
    feeds = collections.defaultdict(set)
    for _sl, src, rel, _tl, tgt, _p in fleet.edges:
        if rel == "NEXT_STAGE":
            successors[src].append(tgt)
        elif rel == "FEEDS":
            feeds[src].add(tgt)

    def depth_from(starts):
        seen, frontier, hops = set(starts), list(starts), 0
        while frontier:
            nxt = [n for f in frontier for n in successors[f] if n not in seen]
            seen.update(nxt)
            frontier = nxt
            if nxt:
                hops += 1
        return hops

    return {row["id"]: depth_from(feeds[row["id"]])
            for row in fleet.nodes.get("Sensor", ())}


def test_the_fleet_stays_well_inside_the_bound(stage_depths):
    """The measurement the bound rests on, re-run rather than remembered.

    Measured on the fleet this module loads (`--scale 0.3`): 4 hops deepest,
    median 4, across 14 sensors. The shipped `--scale 1.0` fleet reaches 5,
    and `benchmarks/catalog/subjects.py` records both beside the constant --
    they are different graphs, and quoting one for the other is how this
    docstring was wrong before. Neither number is what this test trusts: the
    BFS re-runs.

    If a generator change grows a chain to within `HEADROOM` of the bound,
    this fails while the answers are still correct -- which is the point.
    Waiting until a chain exceeds the bound would mean finding out from a
    silently short blast radius.
    """
    assert stage_depths, (
        "the fleet has no sensors, so there is no chain to measure and this "
        "test would otherwise fail inside max() with a bare ValueError. "
        "Check the fixture loaded what it meant to.")
    deepest = max(stage_depths.values())
    assert deepest + HEADROOM <= MAX_STAGE_HOPS, (
        f"the deepest chain reachable from a sensor is {deepest} hops and the "
        f"walks are bounded at {MAX_STAGE_HOPS}. Raise MAX_STAGE_HOPS and "
        f"re-check that EA17 still completes on the engine the floor allows, "
        f"or the blast radius will be truncated without saying so.")


def test_every_walk_carries_the_bound():
    """No variable-length pattern anywhere may go back in without an upper bound.

    What counts as "anywhere" is the point. The catalog is not the only place
    this repo ships Cypher: `demo/demo.py` and `mcp_server/server.py` both
    hold queries a user runs, and neither is in `BY_ID`. Scanning only the
    catalog would have let the note-14 failure ship in the two files most
    likely to be run by someone who is not us -- which is what happened to
    `demo/demo.py`'s `TASKS_OVER_BUDGET`. Both currently pass: the MCP
    server's `device_path` walks `*0..3`, its own fixed bound.

    The check is "a variable-length pattern with no upper limit", not a
    spelling of the hop range: `*]` and `*..]` and `*0..]` are unbounded,
    `*3]` and `*0..3]` and `*..3]` are not, and `*0..{hops}]` -- the form the
    catalog ships before substitution -- is bounded, because the placeholder
    is the bound. `_is_unbounded` is where that lives. `EA07` walks the same
    relationship with its own fixed `*0..3`, a tighter bound chosen for what
    it answers rather than for the planner, which is why the check is "has an
    upper bound" and not "uses `MAX_STAGE_HOPS`".
    """
    cyphers = {qid: entry["cypher"] for qid, entry in BY_ID.items()}
    for relative_path in SHIPPED_MODULES:
        found = _cyphers_in_source(relative_path)
        assert found, (
            f"no Cypher was found in {relative_path}, so this guard is not "
            f"covering it. Either the queries moved, or they are now built in "
            f"a way `_cyphers_in_source` cannot see -- which is the failure "
            f"this assertion exists to make loud rather than silent.")
        cyphers.update(found)

    unbounded = {}
    for name, cypher in cyphers.items():
        for match in VARIABLE_LENGTH.finditer(cypher):
            if _is_unbounded(match.group(1)):
                unbounded[name] = match.group(0)
    assert not unbounded, (
        f"{unbounded} walk a variable-length relationship with no upper hop "
        f"bound. From samyama 1.8.0 the planner refuses that outright "
        f"(engine note 14).")


def _bounded_walks() -> dict[str, str]:
    """Every shipped query that carries `MAX_STAGE_HOPS`, by name."""
    from demo import demo

    walks = {qid: BY_ID[qid]["cypher"] for qid in ("EA17", "EA21")}
    walks["demo.TASKS_OVER_BUDGET"] = demo.TASKS_OVER_BUDGET
    return walks


@pytest.mark.parametrize("name", sorted(_bounded_walks()))
def test_bounding_changes_no_answer(loaded_fleet, name):  # noqa: F811 -- shared fixture
    """The claim the whole bound rests on, run rather than remembered.

    Each query is asked twice against the same graph -- once as it ships,
    once with the bound removed -- and the results compared. If a chain in
    this fleet were longer than `MAX_STAGE_HOPS`, the bounded answer would be
    a strict subset and this fails.

    Compared as sets, not sequences, deliberately: `EA21` has tied rows and no
    tiebreaker (note 3b forbids a second `ORDER BY` key), and running the
    *unbounded* query three times in a row returns those ties in different
    orders. Comparing sequences would fail on that instability and blame the
    bound for it.

    `demo/demo.py`'s `TASKS_OVER_BUDGET` is in the same parametrisation
    rather than a test of its own: it is the same question about the same
    constant, and it earns its place because beat 7 reads its counts out in a
    sentence a human is sent, so a truncated answer would be narrated as a
    fact.
    """
    client, _fleet = loaded_fleet
    bounded = _bounded_walks()[name]
    unbounded = bounded.replace(f"*0..{MAX_STAGE_HOPS}]", "*0..]")
    assert unbounded != bounded, f"{name} no longer carries the bound this test measures"

    got_bounded = sorted(map(str, client.query(bounded, GRAPH).records))
    got_unbounded = sorted(map(str, client.query(unbounded, GRAPH).records))

    assert got_bounded == got_unbounded, (
        f"{name} answers differently once bounded at {MAX_STAGE_HOPS} hops:\n"
        f"  bounded  : {got_bounded}\n"
        f"  unbounded: {got_unbounded}\n"
        f"A chain in this fleet is longer than the bound, so the bounded "
        f"answer is truncated while looking complete.")
