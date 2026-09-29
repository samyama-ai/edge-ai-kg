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
