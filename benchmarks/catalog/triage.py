"""`EA19`-`EA21`: who needs to be told, where it is, and which alert is first.

The three questions asked once a failure is known: which certifications it
implicates, which sites run the affected hardware, and which of several
simultaneous alerts is upstream of the others.
"""
from __future__ import annotations

from benchmarks.catalog.subjects import (
    EA17_SUBJECT,
    EA21_ALERTS,
    MAX_STAGE_HOPS,
    alert_list,
)

TRIAGE: list[dict] = [
    {
        "id": "EA19",
        "title": "COMPLIANCE: this sensor fails -- which certifications does that touch?",
        "question": (f"Sensor `{EA17_SUBJECT}` fails. Which certifications are "
                     "implicated, through the clinical tasks that require it?"),
        "why_graph": (
            "The difference between an ops ticket and a reportable event. "
            "`Certification` and `Sensor` share no edge: they meet only through "
            "`ClinicalTask`, which `REQUIRES_SENSOR` on one side and is "
            "`GOVERNED_BY` on the other. A table of sensors cannot answer it "
            "without knowing to join through tasks, and for a medical or "
            "industrial device that join decides who has to be told, and how "
            "fast."),
        # Deliberately one linear pattern, no `OPTIONAL MATCH` and no
        # re-binding. `EA17` needs note 1's trailing-rebind shape because an
        # anti-join has no other spelling on this build; this question does
        # not, so it does not carry the risk. That is worth stating rather
        # than leaving the next reader to wonder why two neighbouring alerting
        # queries look so different.
        #
        # Grouped by the three `Certification` properties rather than by the
        # node: note 9 says `count(DISTINCT cert)` over a multi-variable MATCH
        # returns rows of 1, so the grouping key has to be properties.
        #
        # The `DISTINCT` inside `count(DISTINCT t.id)` is **not** load-bearing
        # on today's data and was checked rather than assumed: the pattern
        # yields one row per (task, certification) pair, so `count(t.id)`
        # returns the same number, and removing it fails nothing. It stays as
        # insurance against a second `REQUIRES_SENSOR` edge between the same
        # pair, which the loader does not currently produce -- said plainly,
        # because a comment claiming a guard is load-bearing when it is not is
        # how the next person leaves a real one out.
        #
        # An unknown sensor id yields no rows rather than an error, the same
        # shape `EA01` and `EA17` have.
        #
        # `ORDER BY tasks_affected DESC LIMIT 20` has no tiebreaker, and note
        # 3b forbids a second key. It does not matter here: the generator emits
        # six certifications at every scale, so a sensor implicates at most six
        # rows and `LIMIT 20` never truncates -- ties change the order, never
        # the set. `tests/test_certification_alerts.py` compares sorted rows and
        # pins the six, so a catalog that outgrows the limit fails there first.
        #
        # The subject is `EA17_SUBJECT`, the same constant `EA17` uses and
        # interpolated the same way -- these two queries are asked about the
        # same failing sensor in the same breath ("what stops" then "what does
        # that implicate"), so two literals that could drift apart would make
        # the pair answer about different sensors while reading as one story.
        # Neither has a parameterised tool in `mcp_server/server.py`, and that
        # is a gap rather than an oversight to hide: the MCP surface covers
        # none of the alerting queries yet, and exposing them is #49's scope.
        "cypher": """
MATCH (s:Sensor)<-[:REQUIRES_SENSOR]-(t:ClinicalTask)-[:GOVERNED_BY]->(cert:Certification)
WHERE s.id = {subject}
WITH cert.name AS certification, cert.body AS body, cert.class AS cert_class,
     count(DISTINCT t.id) AS tasks_affected
RETURN certification, body, cert_class, tasks_affected
ORDER BY tasks_affected DESC
LIMIT 20
""".replace("{subject}", f'"{EA17_SUBJECT}"'),
    },
    # `EA20`, not `EA17`: `EA17`, `EA18` and `EA19` are the alerting stack
    # (#35, #37, #40), which reached `main` with #113. This spine reserved the
    # next free id rather than reusing a live one, and the two landed without
    # colliding -- which is what the gap was for.
    {
        "id": "EA20",
        "title": "Site-wide or one device: where a recalled board is installed",
        "question": ("A board model is recalled. Which sites run it, how many "
                     "of each site's deployments does it account for, and is "
                     "that site-wide or one device?"),
        "why_graph": ("Two facts about the same site have to arrive in one "
                      "row: how many deployments it holds, and how many of "
                      "those are on the recalled board. The graph walks "
                      "Site<-Deployment->Board once and aggregates both; the "
                      "relational form is a join plus a correlated subquery "
                      "per site, and neither is a traversal."),
        # `board:00003` is the recall subject, hardcoded like `EA01`'s model
        # and accelerator: the catalog asks one concrete question rather than
        # taking parameters, and `mcp_server/server.py` is where a caller
        # supplies their own. `tests/test_site_spine.py` reads this id back out
        # of the Cypher rather than restating it, so moving it here moves the
        # ground truth with it.
        #
        # **No `LIMIT`, deliberately.** Every other catalog entry caps its
        # rows because it asks a top-N question; this one asks "where is the
        # recalled board", and a recall answer that omits a site is wrong
        # rather than abbreviated. With `LIMIT 12` it did omit sites, and not
        # marginally: measured at `--scale 5.0`, 60 sites, 38 of them running
        # the recalled board, and **26 of those 38 fell past the cut**. The
        # 12th `on_recalled_board` value is 2 and nine sites share it, so
        # *which* of them survived could differ between engines -- the same tie
        # instability that had `EA02` and `EA11` withdrawn from the Neo4j
        # comparison. A second `ORDER BY` key cannot break the tie either
        # (engine note 3b: only the first key is honoured).
        #
        # The cost is row count at large scales -- 60 rows at `--scale 5.0`,
        # 120 at 10.0, both scales `docs/volume.md` uses -- and that is the
        # right trade for a question whose value is completeness. At the
        # shipped 12 sites the output is unchanged.
        #
        # The untouched sites are part of the answer, which is why this is not
        # filtered to `on_recalled_board > 0` either: "leave that site alone"
        # and "replace one unit here" are the two halves of "site-wide or one
        # device", and `docs/location-scope.md` rests on both being readable
        # off the same rows.
        "cypher": """
MATCH (s:Site)<-[:DEPLOYED_AT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WITH s.campus AS campus, s.name AS site, s.kind AS kind,
     count(DISTINCT d.id) AS deployments_here,
     sum(CASE WHEN b.id = "board:00003" THEN 1 ELSE 0 END) AS on_recalled_board
RETURN campus, site, kind, on_recalled_board, deployments_here
ORDER BY on_recalled_board DESC
""",
    },
    {
        "id": "EA21",
        "title": "ROOT CAUSE: twenty alerts, one fault -- which are upstream?",
        "question": ("These sensors are all alerting at once. Which of them "
                     "can reach the others through the pipeline, and which "
                     "are downstream of something else that is alerting?"),
        "why_graph": (
            "The failure that makes alerting hated is twenty pages at 3am for "
            "one fault. Ordering them needs a model of what feeds what, which "
            "a time-series alerting system does not have -- it holds "
            "thresholds and history, not dependencies -- and which in SQL is "
            "a recursive CTE over an edge table. Here it is one reachability "
            "pattern. "
            "**The failure model is the one `EA17` uses**, and the ranking "
            "only means anything under it: a *sensor* degrades, and what it "
            "feeds is degraded with it. So if `s` fails, every stage "
            "downstream of where `s` joins carries its bad data -- including "
            "the stages where another alerting sensor's own data joins, whose "
            "pipeline output is then bad too. `s` reaching `o` means \"o's "
            "alert is explained by s's failure\", which is why the count "
            "orders them. The inverse model -- a *stage* fails and sensors "
            "alert -- gives the opposite order, and this query does not "
            "answer it; there the useful answer is the stages common to every "
            "alerting sensor, an intersection rather than a per-sensor count. "
            "**It gives a partial order, and often no root at all.** "
            "`NEXT_STAGE` is the union of every sensor's chain over one "
            "shared pool of stages, so it contains cycles, and two alerts in "
            "a cycle have no upstream-of between them. Whether any alert is "
            "upstream of the rest is a property of the fleet, not of this "
            "query: on the generated graph it turns on the chain sampling, "
            "which moves with the upstream ONNX operator catalogue -- at 205 "
            "operators `sensor:00000` reaches the other two and neither "
            "reaches it, at 379 all three reach each other. So the catalog "
            "set is an example of the *shape*, not a worked root-cause. "
            "`reaches` is what carries the answer: each id appearing in the "
            "other's list means neither is upstream, and an alert named by "
            "nobody is a root."),
        # `OPTIONAL MATCH`, so an alert with nothing downstream still gets a
        # row. Not a detail: the case that proves this ranks by dependency
        # rather than by degree is the one where the alerts are independent
        # and every count is 0. An inner `MATCH` drops those rows, and the
        # output then reads "no answer" instead of "no root".
        #
        # `s` is re-bound in **leading** position, which is not note 1's shape
        # -- that is a *trailing* bound variable in a second `MATCH`, where the
        # join is not enforced. Measured rather than argued: in
        # `tests/test_root_cause.py` a sensor whose chain touches nothing else
        # scores 0 on a fixture where an unenforced join would score 3.
        #
        # **This counts reachability, and the stage graph has cycles**, so
        # `downstream_alerts` is "alerts I can reach", never "alerts strictly
        # beneath me". Two ways that happens:
        #
        #   - `*0..` includes the zero-length walk, so two sensors feeding the
        #     *same* stage each count the other;
        #   - `etl/generate.py` samples each sensor's chain from one shared
        #     pool of 16 stages, so the union really does contain cycles --
        #     `tests/test_blast_radius_semantics.py` names one
        #     (`stage:00012 -> 00009 -> 00010 -> 00012`).
        #
        # Whether the shipped fleet has a root at all is **not stable**: the
        # chain sampling moves with the upstream ONNX operator catalogue. At
        # 205 operators `sensor:00000` reaches the other two and neither
        # reaches it; at 379 (a fresh download, 2026-09-24) all three reach
        # each other and there is no root. `tests/test_root_cause.py` pins
        # the behaviour on fixtures it builds itself for that reason, and its
        # full-scale test asserts only what holds on any fleet: the engine
        # agrees with a Python BFS, and every mutual pair is visible in
        # `reaches`.
        #
        # `reaches` exists to make that visible rather than leave it implied:
        # if `b` is in `a`'s list and `a` is in `b`'s, neither is upstream of
        # the other. Two shapes produce that, and the output cannot tell them
        # apart: a genuine cycle, and two sensors feeding the *same* entry
        # stage (the `*0..` zero-length walk makes each reach the other).
        # Both mean the same thing to a caller -- no order between these two
        # -- which is why one column serves both;
        # `test_a_shared_entry_stage_reads_as_mutual_too` pins the second.
        # `tests/test_root_cause.py::test_two_alerts_in_a_cycle_each_reach_the_other`
        # pins it on a two-stage cycle built for the purpose.
        #
        # No `IS NOT NULL` guard: `o.id IN {alerts}` already excludes a
        # `Sensor` carrying no `id`, which is measured rather than reasoned
        # about -- a fixture holding one gives the same three rows with the
        # guard and without it, so the guard was inert and note 8b's `<>`
        # behaviour never reached it.
        #
        # **Cost.** `*0..` over a cyclic graph enumerates paths before
        # `count(DISTINCT)` reduces them, so the work is bounded by paths and
        # not by stages. Cheap on this fleet -- 16 stages, 40 `NEXT_STAGE`
        # edges, 0.1 ms warm at `--scale 1.0` -- and that is a statement about
        # the fixture, not about the shape: a denser real pipeline could grow
        # this sharply, and the fix there is a bound on the walk, which costs
        # the deep chains (`EA07` makes that trade, in
        # `benchmarks/catalog/core.py`). Measure
        # before assuming it still holds on real topology.
        #
        # One `ORDER BY` key (note 3b) and no tiebreaker: ties are alerts
        # reaching the same number of others, and their order among
        # themselves carries no meaning. `LIMIT 20` truncates to the top 20
        # by design -- the catalog's set is three ids, but the 3am case this
        # is written for is twenty or more, and the rows past the twentieth
        # are the ones nobody pages on.
        "cypher": ("""
MATCH (s:Sensor)
WHERE s.id IN {alerts}
OPTIONAL MATCH (s)-[:FEEDS]->(:SignalStage)-[:NEXT_STAGE*0..{hops}]->(x:SignalStage)<-[:FEEDS]-(o:Sensor)
WHERE o.id IN {alerts} AND o.id <> s.id
WITH s.id AS alert, count(DISTINCT o.id) AS downstream_alerts,
     collect(DISTINCT o.id) AS reaches
RETURN alert, downstream_alerts, reaches
ORDER BY downstream_alerts DESC
LIMIT 20
""".replace("{alerts}", alert_list(EA21_ALERTS)).replace("{hops}", str(MAX_STAGE_HOPS))),
    },
]
