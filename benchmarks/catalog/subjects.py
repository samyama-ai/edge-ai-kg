"""The subjects the catalog's parameterised queries are asked about.

Separate from the entry modules because two of them share these values and
the package's `__init__` re-exports them: a constant defined beside one query
and imported by another is how `EA17_SUBJECT` ends up meaning two things.

The escaping helpers live here for the same reason. `EA21` is the only entry
that builds a list literal today, but the refusal rules below are about what
is safe to interpolate into Cypher at all, not about one query.
"""
from __future__ import annotations

from etl.helpers import cypher_literal

# `EA17`'s subject, in one place. The query names it ten times -- twice in each
# of five legs -- and a retarget that rewrote only some of them would leave one
# leg answering about a different sensor than the rest, which reads as a real
# finding rather than an editing mistake.
EA17_SUBJECT = "sensor:00000"

# `EA21`'s alerting set: the sensors paging right now. Written once and
# interpolated, for the same reason as `EA17_SUBJECT` -- the query names it
# twice, and a set rewritten in one place and not the other would rank one
# population against a different one, which reads as a finding rather than an
# edit. `retargeted_ea21`, in this package's `__init__` and re-exported
# from `benchmarks.queries`, is how a caller asks about a different set.
EA21_ALERTS = ("sensor:00000", "sensor:00003", "sensor:00007")

# The upper hop bound on every `NEXT_STAGE` walk in `EA17` and `EA21`.
#
# It is what makes the pattern *legal* on an engine newer than 1.7.1, which is
# not the same as making both queries run there: `EA21` does, `EA17` still
# times out on 1.9.0 at every bound deep enough to answer completely (engine
# note 14 has the table). The ceiling stays at `<1.8` because of that.
#
# From `samyama` 1.8.0 the planner refuses an unbounded variable-length
# pattern that produces over a million paths (engine note 14). `EA17` and
# `EA21` walk a stage graph that contains cycles, so the *path* count explodes
# even though the graph is tiny -- 16 stages, and a sensor reaches all of
# them.
#
# 8 is measured, not guessed. Two fleets, because they are not the same
# graph and the numbers differ:
#
#   - the **shipped** fleet (`data/fleet/fleet.json`, seed 20260814,
#     `--scale 1.0`): deepest chain from any sensor **5 hops**, median 4,
#     distribution 1:1, 3:4, 4:8, 5:1 across 14 sensors;
#   - the fleet `tests/test_bounded_walks.py` loads (same seed, `--scale 0.3`,
#     shared with `tests/test_certification_alerts.py`): deepest **4**,
#     median 4, distribution 3:6, 4:8.
#
# Both sit well inside 8. Neither figure is what the guard trusts: the test
# re-runs the BFS against whatever fleet it loads and fails if that fleet
# grows a chain within two hops of the bound, so these numbers are the record
# of when the constant was chosen, not the check. The claim that bounding
# changes no answer is separately measured -- `test_bounding_changes_no_answer`
# runs `EA17` and `EA21` bounded and unbounded and compares the rows, as sets
# rather than sequences because `EA21`'s ties have no tiebreaker.
#
# It is a real limit, not decoration: a graph whose chains passed 8 hops would
# have its blast radius silently truncated, which is why #110 argued against a
# cap. The guard is what makes the cap safe to carry.
MAX_STAGE_HOPS = 8


def alert_list(alert_ids) -> str:
    """The Cypher list literal for an alert set, escaped.

    Every id goes through `etl.helpers.cypher_literal`, which strips the
    quotes and backslashes that would end the literal early. Without it this
    function built Cypher out of whatever it was handed, and the docstring
    `retargeted_ea21` promises an MCP tool will hand it a live alert set --
    so the input is external by design. Measured before the fix:
    `['sensor:x"] OR true //']` produced `["sensor:x"] OR true //"]`, which
    closes the list, disjoins a true predicate and comments out the rest of
    the line; the query then returned every sensor in the graph.

    A bare `str` is refused rather than accepted. Python iterates it by
    character, so `"sensor:00000"` silently became a twelve-element list of
    single letters that matched nothing -- an empty answer with no error,
    which is the shape this catalog treats as a finding.
    """
    if isinstance(alert_ids, (str, bytes)):
        raise TypeError(
            f"alert ids must be a sequence of ids, not {type(alert_ids).__name__}: "
            f"a bare string iterates by character and would ask about "
            f"{len(alert_ids)} one-letter ids, returning nothing and raising "
            f"nothing.")
    return "[" + ", ".join(_alert_literal(a) for a in alert_ids) + "]"


def _alert_literal(alert_id) -> str:
    """One id as a Cypher literal, refused if escaping would change it.

    `cypher_literal` strips the characters that would end a literal early,
    which makes injection impossible but introduces a quieter problem:
    `sensor:x"` and `sensor:x` both render as `"sensor:x"`, so a caller asking
    about one would be answered about the other. Silently substituting a
    *different sensor* into an alerting query is worse than refusing, and the
    ids this repo generates (`sensor:00000`) never contain those characters --
    so an id that changes under escaping is a sign something is wrong upstream
    rather than an id to normalise.
    """
    text = str(alert_id)
    literal = cypher_literal(text)
    if literal != f'"{text}"':
        raise ValueError(
            f"alert id {alert_id!r} cannot be asked about safely: escaping it "
            f"gives {literal}, which names a different sensor. Quotes, "
            f"backslashes and newlines are stripped to keep the literal from "
            f"ending early, so an id containing them would silently become "
            f"another id.")
    return literal
