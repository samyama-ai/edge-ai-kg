"""The MCP tools re-state catalog queries; nothing stopped them drifting (#6).

Six of the seven tools in `mcp_server/server.py` are parameterised variants of
`benchmarks/queries.py` entries. `CLAUDE.md` calls the duplication deliberate --
the tools take typed parameters and interpolate them -- but the catalog is
otherwise the single source of truth, so the MCP server is the one consumer that
can silently fall out of date.

**The risk is correctness, not tidiness.** The catalog queries are shaped the
way they are to dodge documented engine bugs (`docs/engine-notes.md`), several
of which return wrong rows rather than erroring. `quantization_unlock` already
carries a comment saying it must not be rewritten as a self-join; nothing
enforced that.

## Why not compare the queries whole

Because they are legitimately different. Measured across the six pairs, the
tools project different columns, use different aliases, and add optional
predicates -- `boards_for_task` filters `v.precision`, its twin `EA03` filters
`d.fits`. Demanding equality would fail on day one and be deleted.

## What is compared instead

1. **The `MATCH` patterns, which must be identical.** That is where a self-join
   would appear, and it is the specific drift #6 names. All six pairs match
   today, with parameter interpolation normalised away.
2. **The engine-note rules**, applied to the tools the way
   `tests/test_correctness.py` applies them to the catalog: project through
   `WITH` before `RETURN`, one `ORDER BY` key, no `RETURN DISTINCT`, and never
   aggregate a bare node variable.

Static: it parses `mcp_server/server.py` and the catalog, so it needs no engine,
no `data/`, and no MCP client.
"""
import pathlib
import re

import pytest

from benchmarks.queries import BY_ID

ROOT = pathlib.Path(__file__).resolve().parent.parent
MCP = ROOT / "mcp_server" / "server.py"

# Tool -> its catalog twin.
TWINS = {
    "fallback_audit": "EA01",
    "boards_for_task": "EA03",
    "quantization_unlock": "EA04",
    "operator_coverage": "EA05",
    "kernel_blast_radius": "EA06",
    "device_path": "EA07",
}

# Tools that deliberately have no twin. `run_cypher` executes whatever it is
# given, so there is nothing to compare it against.
NO_TWIN = {"run_cypher"}

TOOL_DEF = re.compile(r"^@mcp\.tool\(\)[^\n]*\n(?:async\s+)?def\s+(\w+)", re.MULTILINE)


def mcp_source() -> str:
    return MCP.read_text(encoding="utf-8")


def tool_names() -> set[str]:
    return set(TOOL_DEF.findall(mcp_source()))


def tool_cypher(name: str) -> str:
    """The Cypher a tool sends, with `{...}` interpolations blanked to `?`.

    Blanked rather than removed: the parameter is what makes these variants
    rather than copies, and the comparison below is about everything else.
    """
    match = re.search(rf"def {name}\(.*?_rows\(f?\"\"\"(.*?)\"\"\"",
                      mcp_source(), re.DOTALL)
    assert match, f"no Cypher found in MCP tool {name!r}"
    return re.sub(r"\{[^}]*\}", "?", match.group(1))


def match_patterns(cypher: str) -> list[str]:
    return [re.sub(r"\s+", " ", line.strip())
            for line in cypher.splitlines()
            if re.match(r"^\s*(OPTIONAL\s+MATCH|MATCH)\b", line, re.IGNORECASE)]


def test_every_tool_is_either_twinned_or_declared_twinless():
    """The mapping is complete, so a new tool cannot skip these checks.

    Without this, adding a seventh query-bearing tool would add an unchecked
    copy of catalog Cypher -- which is exactly how #6 arose.
    """
    known = set(TWINS) | NO_TWIN
    tools = tool_names()
    assert tools, "no @mcp.tool() functions found; has the decorator changed?"
    unlisted = tools - known
    assert not unlisted, (
        f"MCP tools not listed in TWINS or NO_TWIN: {sorted(unlisted)}. Either "
        f"name its catalog twin, or add it to NO_TWIN with a reason."
    )
    gone = known - tools
    assert not gone, f"TWINS/NO_TWIN name tools that no longer exist: {sorted(gone)}"


@pytest.mark.parametrize("tool,qid", sorted(TWINS.items()))
def test_the_match_patterns_still_agree_with_the_catalog(tool, qid):
    """The traversal shape, which is where an engine-note violation would show.

    Everything else about these queries is allowed to differ -- columns,
    aliases, predicates. The patterns are not: `EA04` is written as a single
    linear pattern plus conditional aggregation precisely because the
    comma-separated self-join returns a cartesian product at scale (note 1), and
    a hand-maintained copy is where that shaping gets lost.
    """
    mine = match_patterns(tool_cypher(tool))
    theirs = match_patterns(BY_ID[qid]["cypher"])
    assert mine, f"{tool} has no MATCH clause"
    assert mine == theirs, (
        f"{tool}'s traversal no longer matches {qid}'s.\n"
        f"  mcp: {mine}\n"
        f"  cat: {theirs}\n"
        f"The catalog shape dodges documented engine bugs (docs/engine-notes.md). "
        f"If {qid} changed, change {tool} with it; if {tool} needs a different "
        f"traversal, it is no longer a variant of {qid} and TWINS should say so."
    )


@pytest.mark.parametrize("tool", sorted(TWINS))
def test_each_tool_projects_through_with_before_returning(tool):
    """Engine note 3: `ORDER BY` on a `RETURN`-introduced alias is ignored.

    With `LIMIT`, that is an arbitrary N rows dressed up as a top-N -- wrong
    rows, silently. Every catalog query projects through `WITH` first; the
    tools must too.
    """
    cypher = tool_cypher(tool)
    if not re.search(r"^\s*ORDER BY", cypher, re.MULTILINE | re.IGNORECASE):
        pytest.skip(f"{tool} has no ORDER BY")
    assert re.search(r"^\s*WITH\b", cypher, re.MULTILINE | re.IGNORECASE), (
        f"{tool} sorts without projecting through WITH first, so the sort is "
        f"silently dropped (engine note 3)"
    )


@pytest.mark.parametrize("tool", sorted(TWINS))
def test_each_tool_sorts_on_a_single_key(tool):
    """Engine note 3b: only the first `ORDER BY` key is honoured."""
    cypher = tool_cypher(tool)
    match = re.search(r"ORDER BY\s+(.+?)(?:\s+LIMIT|\s*$)", cypher,
                      re.DOTALL | re.IGNORECASE)
    if not match:
        pytest.skip(f"{tool} has no ORDER BY")
    keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
    assert len(keys) == 1, (
        f"{tool} sorts on {len(keys)} keys ({keys}); only the first is honoured, "
        f"so the rest are a silent no-op (engine note 3b)"
    )


@pytest.mark.parametrize("tool", sorted(TWINS))
def test_no_tool_uses_return_distinct_or_aggregates_a_bare_node(tool):
    """Engine notes 2 and 9, both of which return wrong rows rather than erroring.

    `RETURN DISTINCT` is a no-op; `count(DISTINCT op)` over a multi-variable
    `MATCH` returns N rows of 1 where `count(DISTINCT op.id)` is correct.
    """
    cypher = tool_cypher(tool)
    assert not re.search(r"RETURN\s+DISTINCT", cypher, re.IGNORECASE), (
        f"{tool} uses RETURN DISTINCT, which is a silent no-op (engine note 2)"
    )
    # Only the DISTINCT form. `count(k)` without DISTINCT is the anti-join
    # idiom the notes *prescribe* -- "OPTIONAL MATCH ... WITH ... count(k) AS n
    # ... WHERE n = 0" -- and EA01 uses it, so flagging it would fail the
    # catalog's own shape. Note 9 is about `count(DISTINCT op)` specifically.
    bare = re.findall(r"\b(?:count|collect)\s*\(\s*DISTINCT\s+([a-z_]\w*)\s*\)",
                      cypher, re.IGNORECASE)
    assert not bare, (
        f"{tool} aggregates bare node variable(s) {bare} under DISTINCT; over a "
        f"multi-variable MATCH that returns N rows of 1 (engine note 9). "
        f"Aggregate a property, e.g. count(DISTINCT {bare[0]}.id)."
    )
