"""`EA01` in the spelling a Neo4j author would use, and the check that the two
still ask the same question.

Split out of `benchmarks/compare_neo4j.py`, which crossed the 500-line limit the
review harness skips a file at -- and a skipped file is an unreviewed one. This
is the `--natural` half: the hand-written hero query, and the guard that refuses
to print a ratio when the two spellings have drifted apart. Nothing here times
anything; `compare_neo4j` owns the measurement.
"""
from __future__ import annotations

import re

# `EA01` as someone would write it for Neo4j, with none of the workarounds
# `docs/engine-notes.md` forces on us: a negated pattern instead of
# `OPTIONAL MATCH ... count() = 0` (note 5), and no projection through `WITH`
# before `RETURN` (note 3). Used by `--natural` to price the shaping, because a
# comparison run only on our shapes cannot say whether they cost Neo4j anything.
NATURAL_EA01 = """
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.id = "model:00000"
  AND NOT EXISTS {
    MATCH (k:Kernel)-[:IMPLEMENTS]->(op)
    MATCH (k)-[:RUNS_ON]->(a:Accelerator)
    WHERE a.id = "accel:00001"
  }
RETURN op.name AS operator, op.category AS category, op.since_version AS opset
ORDER BY category
"""


def natural_ea01_matches_the_catalog(catalog) -> list[str]:
    """Why the two spellings can be compared at all: same subjects, same columns.

    `NATURAL_EA01` restates `EA01`'s hardcoded subject ids, so a retarget of the
    catalog entry would leave the two asking about different things while still
    returning the same column names -- and the ratio would look fine. Both sets
    are extracted and compared, in both directions: a subject added to either
    spelling is drift as much as one removed. Returns the mismatches rather
    than asserting, so the caller can decline to print a ratio instead of dying
    mid-run.
    """
    shipped = catalog["EA01"]["cypher"]
    # Read the ids out of both and compare the sets. The previous form asked
    # `if subject in shipped and subject not in NATURAL_EA01` over the two
    # hardcoded literals -- so retargeting EA01 to `model:00042` made the first
    # half false, reported nothing, and let the comparison price two different
    # questions against each other. The one drift this guard exists to catch
    # was the one it could not see.
    SUBJECT = re.compile(r'"(?:model|accel):\d+"')
    in_shipped = set(SUBJECT.findall(shipped))
    in_natural = set(SUBJECT.findall(NATURAL_EA01))
    problems = [f"{s} is in EA01 but not in NATURAL_EA01"
                for s in sorted(in_shipped - in_natural)]
    problems += [f"{s} is in NATURAL_EA01 but not in EA01"
                 for s in sorted(in_natural - in_shipped)]
    if not in_shipped:
        problems.append(
            "EA01 names no `model:` or `accel:` subject at all, so there is "
            "nothing to match NATURAL_EA01 against -- this guard has stopped "
            "guarding. Either EA01 is now parameterised or the id format "
            "changed; update the pattern rather than deleting the check.")
    # The columns the caller receives, in order -- read off the final `RETURN`,
    # not off every `AS` in the query. EA01 projects through two `WITH`s
    # (engine note 3), so an `AS`-wide scan also picks up its intermediate
    # `count(k) AS kernels`, which is not a column either spelling returns:
    # that comparison reports drift on a pair that matches.
    #
    # Order matters. The two return the same three names, and a spelling that
    # returned them in a different order would return different rows while
    # every set comparison stayed quiet.
    if returned_columns(shipped) != returned_columns(NATURAL_EA01):
        problems.append(
            f"columns differ: EA01 returns {returned_columns(shipped)}, "
            f"NATURAL_EA01 returns {returned_columns(NATURAL_EA01)}")
    return problems


def returned_columns(cypher: str) -> list[str]:
    """The final `RETURN` clause's output names, in order.

    `RETURN a, b AS c` -> `["a", "c"]`. Deliberately last-`RETURN`-wins: a
    `CALL { ... RETURN x }` subquery's inner `RETURN` is not the result shape.
    """
    # Not anchored to the start of a line. The catalog writes one clause per
    # line, so `^\s*RETURN` worked on every query here and returned the whole
    # statement as a single column name for a one-line query -- silently, since
    # the caller only compares two of these against each other.
    body = re.split(r"(?i)\bRETURN\s+", cypher)[-1]
    body = re.split(r"(?i)\b(?:ORDER\s+BY|LIMIT|SKIP|UNION)\b", body)[0]
    out = []
    for cell in body.split(","):
        cell = cell.strip()
        if not cell:
            continue
        alias = re.search(r"\bAS\s+(\w+)\s*$", cell, re.IGNORECASE)
        out.append(alias.group(1) if alias else cell)
    return out
