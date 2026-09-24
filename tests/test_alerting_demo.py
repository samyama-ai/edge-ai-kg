"""The demo's alert beat says a sentence, and a sentence nobody checked is prose (#42).

`demo/demo.py`'s seventh beat ends by printing the message an on-call engineer
would be sent. #42's whole argument is that the message is a **query result**
rather than a template someone maintains, so the thing worth testing is exactly
that: every number in the sentence has to be a number one of the three queries
returned, and the wording has to change when the answers do.

Two halves, for two different failure modes:

- **`alert_sentence` driven with rows, no engine.** Templates fail on the
  branch nobody exercises -- the fleet with no breach, the task with no second
  source, the sensor that is not in the graph. Those cases are hand-built here
  because the shipped fleet does not produce all of them.
- **The beat run against a loaded graph.** The unit tests cannot catch a
  sentence that is internally consistent and quotes the wrong query, so the
  numbers are re-read out of the rows and compared, and `TASKS_OVER_BUDGET` is
  checked against ground truth computed in Python from the `Fleet`.

The second half needs `data/`; the first needs nothing.
"""
from __future__ import annotations

import collections

import pytest

from benchmarks.queries import BY_ID, EA17_SUBJECT
from demo.demo import TASKS_OVER_BUDGET, alert_sentence
from etl.helpers import create_edges, create_nodes

# `operators` is imported rather than copied: it is a fixture whose skip fires
# in **setup** when `data/` is absent, and a second copy here could drift from
# the one the blast-radius tests use -- two modules then disagreeing about what
# "the graph" is. Importing it is what makes pytest see it in this module, so
# it is used despite reading as unused (`F401`), and a test taking it as an
# argument reads to ruff as a redefinition (`F811`). Three suppressions, one
# reason; the alternative is the copy this avoids.
from tests.test_blast_radius import (  # noqa: F401
    SCALE,
    SEED,
    build_and_load,
    operators,
)

GRAPH = "default"

# One `EA17` answer, in the shape the query returns: kind, affected,
# only_via_me, nearest. Written out rather than fetched so the branches below
# can vary one number at a time.
RADIUS = [
    ("SignalStage", 15, 0, 1),
    ("Model", 30, 0, 2),
    ("Deployment", 360, 0, 4),
    ("Board", 60, 0, 5),
    ("ClinicalTask", 4, 0, 1),
]
CERTS = [
    ("FDA 510(k) Class II", "FDA", "II", 2),
    ("ISO 13485", "ISO", "QMS", 1),
]


def test_the_sentence_quotes_the_numbers_it_was_given():
    """The breach case: every figure in the message comes from a row."""
    text = alert_sentence("sensor:00000", RADIUS, [(1, 1)], CERTS)
    assert "15 signal stages" in text
    assert "30 models" in text
    assert "360 deployments" in text
    assert "60 boards" in text
    assert "4 clinical tasks" in text
    assert "1 of them is already over the latency budget, on 1 deployment" in text
    assert "FDA 510(k) Class II, ISO 13485" in text


def test_no_breach_reads_as_no_breach():
    """The branch a template gets wrong, and the one the shipped fleet takes.

    `EA18` returns six breaches at `--scale 0.5` and none at `--scale 1.0`, so
    a sentence that implied a breach would be wrong on the fleet this repo
    ships while looking right on the demo's default.
    """
    text = alert_sentence("sensor:00000", RADIUS, [], CERTS)
    assert "None of them is over its latency budget." in text
    assert "already over" not in text


def test_a_task_with_no_other_source_is_called_out():
    """`only_via_me` is the column that separates "degraded" from "stopped"."""
    radius = [row if row[0] != "ClinicalTask" else ("ClinicalTask", 4, 2, 1)
              for row in RADIUS]
    text = alert_sentence("sensor:00000", radius, [], CERTS)
    assert "2 of those have no other source" in text
    assert "none of them exclusively" not in text


def test_an_unknown_sensor_says_so_instead_of_reporting_zeros():
    """`EA17` returns nothing for an id that is not in the graph.

    A row of zeros and no rows at all mean different things on a pager: one
    says the sensor feeds nothing, the other says nobody knows this sensor.
    """
    text = alert_sentence("sensor:not-in-this-graph", [], [], [])
    assert "nothing downstream was found" in text
    assert "check the id" in text


def test_singulars_are_not_printed_as_plurals():
    """It is a sentence a human is sent, so "1 boards" is a defect."""
    radius = [("SignalStage", 1, 0, 1), ("Model", 1, 0, 2),
              ("Deployment", 1, 0, 4), ("Board", 1, 0, 5),
              ("ClinicalTask", 1, 0, 1)]
    text = alert_sentence("sensor:00000", radius, [(1, 1)], CERTS[:1])
    assert "1 signal stage," in text and "1 board." in text
    assert "1 clinical task depends on it" in text
    assert "1 certification is implicated" in text
    assert "stages" not in text and "boards" not in text


def test_no_certification_is_a_sentence_too():
    """An unregulated sensor is a different alert, not a missing clause."""
    text = alert_sentence("sensor:00000", RADIUS, [], [])
    assert "No certification is implicated." in text


def _rows(client, cypher):
    return [tuple(row) for row in client.query(cypher.strip(), GRAPH).records]


def test_the_beat_runs_and_its_sentence_matches_the_rows(
        engine_factory, operators):  # noqa: F811
    """The three queries the beat runs, against a real graph, end to end.

    The unit tests above cannot catch a sentence that is internally consistent
    and reads the wrong column, so the figures are re-derived from the rows
    here and compared against the text.
    """
    client, _fleet = build_and_load(engine_factory, operators, seed=SEED, scale=SCALE)
    radius = _rows(client, BY_ID["EA17"]["cypher"])
    budget = _rows(client, TASKS_OVER_BUDGET)
    certs = _rows(client, BY_ID["EA19"]["cypher"])
    text = alert_sentence(EA17_SUBJECT, radius, budget, certs)

    assert radius, "EA17 returned nothing for the catalog subject"
    by_kind = {row[0]: row for row in radius}
    for kind, noun in (("SignalStage", "signal stage"), ("Model", "model"),
                       ("Deployment", "deployment"), ("Board", "board"),
                       ("ClinicalTask", "clinical task")):
        assert f"{by_kind[kind][1]:,} {noun}" in text, (
            f"the alert does not quote EA17's {kind} count "
            f"({by_kind[kind][1]:,}): {text}")

    # `only_via_me`, which is the clause a reader acts on: "none of them
    # exclusively" and "N have no other source" are different alerts.
    exclusive = by_kind["ClinicalTask"][2]
    if exclusive == 0:
        assert "none of them exclusively" in text
    else:
        assert f"{exclusive:,} of those" in text and "no other source" in text

    for certification, *_rest in certs:
        assert certification in text, f"{certification} is missing from the alert"

    # Both halves, so reading the wrong column fails whichever way the fleet
    # happens to fall. `EA18` returns six breaches at --scale 0.5 and none at
    # --scale 1.0, so which branch a run takes is not a given.
    if budget:
        tasks_over, deployments_over = budget[0]
        assert f"{tasks_over:,} of them" in text
        assert f"on {deployments_over:,} deployment" in text
    else:
        assert "None of them is over its latency budget." in text


def budget_truth(fleet, sensor_id) -> list:
    """The scoped budget answer, walked from the `Fleet`'s own edges.

    Deliberately a different traversal from the query's: a forward walk to the
    models this sensor feeds, intersected with the tasks that require it. An
    earlier version walked `REQUIRES_SENSOR -> SOLVES` only, and so agreed with
    a query that counted deployments the sensor never feeds.
    """
    requires, solves, feeds = (collections.defaultdict(set) for _ in range(3))
    nexts, precedes, variant_of, of_variant = (collections.defaultdict(set)
                                               for _ in range(4))
    for _sl, src, rel, _tl, tgt, _p in fleet.edges:
        if rel == "REQUIRES_SENSOR":
            requires[tgt].add(src)          # sensor -> tasks
        elif rel == "SOLVES":
            solves[src].add(tgt)            # model -> tasks
        elif rel == "FEEDS":
            feeds[src].add(tgt)             # sensor -> entry stages
        elif rel == "NEXT_STAGE":
            nexts[src].add(tgt)
        elif rel == "PRECEDES":
            precedes[src].add(tgt)          # stage -> models
        elif rel == "VARIANT_OF":
            variant_of[tgt].add(src)        # model -> variants
        elif rel == "OF_VARIANT":
            of_variant[tgt].add(src)        # variant -> deployments

    reached, queue = set(feeds[sensor_id]), list(feeds[sensor_id])
    while queue:
        for stage in nexts[queue.pop()]:
            if stage not in reached:
                reached.add(stage)
                queue.append(stage)
    downstream_models = {m for stage in reached for m in precedes[stage]}
    its_tasks = requires[sensor_id]

    budgets = {row["id"]: row["latency_budget_ms"]
               for row in fleet.nodes.get("ClinicalTask", ())}
    # `.get("latency_ms")`, not `[...]`: the real layer's MLPerf deployments
    # carry `throughput_inf_s` and no latency at all. A missing property cannot
    # exceed a budget, and the Cypher agrees -- `d.latency_ms > ...` is not
    # true for a null. Indexing would raise here and hide that agreement.
    latency = {row["id"]: row.get("latency_ms")
               for row in fleet.nodes.get("Deployment", ())}

    over_tasks, over_deployments = set(), set()
    for model in downstream_models:
        for task in solves[model] & its_tasks:
            for variant in variant_of[model]:
                for deployment in of_variant[variant]:
                    measured = latency.get(deployment)
                    if measured is not None and measured > budgets[task]:
                        over_tasks.add(task)
                        over_deployments.add(deployment)
    return [(len(over_tasks), len(over_deployments))] if over_tasks else []


def test_tasks_over_budget_matches_ground_truth(
        engine_factory, operators):  # noqa: F811
    """`TASKS_OVER_BUDGET` against the same count computed from the `Fleet`.

    The query is the beat's one piece of bespoke Cypher -- it is not in the
    catalog, so `tests/test_correctness.py` does not sweep it -- which makes
    this the only thing standing between the sentence and a plausible wrong
    number. Walked over the edges here, not re-run as a variant of the same
    Cypher, and over **the same path the query takes**: an earlier version of
    both walked `REQUIRES_SENSOR -> SOLVES` only, so the truth agreed with a
    query that counted deployments the sensor never feeds.
    """
    client, fleet = build_and_load(engine_factory, operators, seed=SEED, scale=SCALE)

    got = _rows(client, TASKS_OVER_BUDGET)
    want = budget_truth(fleet, EA17_SUBJECT)
    assert got == want, (
        f"the scoped budget query returned {got}; the fleet's own edges give "
        f"{want}. One of the two is reading the graph differently.")


def test_the_comma_join_holds_at_full_cardinality(request, engine_factory,
                                                  operators):  # noqa: F811
    """`TASKS_OVER_BUDGET`'s second pattern, at `--scale 1.0` with real row counts.

    CLAUDE.md is explicit that a re-bind "isn't enforced" and that "at scale,
    even the comma-separated single-`MATCH` form breaks". This query has one:
    `(m)<-[:VARIANT_OF]-...` re-binds `m` after the comma. It is in **leading**
    position rather than trailing, which is not the forbidden shape, but a
    two-deployment fixture is exactly the cardinality at which the difference
    does not show, so it is measured here instead.

    A single linear pattern cannot express this question: the chain has to
    branch at `m`, to the task on one side and the deployment on the other, and
    the only way to avoid the comma is to re-bind `m` in **trailing** position,
    which is the shape CLAUDE.md forbids outright. So the comma stays and the
    join is checked.

    The shipped fleet has no breach at `--scale 1.0`, so an honest check has to
    make one: every task budget is dropped to 1 ms before loading, which puts
    most deployments over. That is the row count an unenforced join would
    inflate -- it would pair every downstream model with every deployment.

    **Measured 2026-09-24 at scale 1.0 with budgets forced to 1 ms**: all 14
    sensors, up to 396 deployments over budget for one sensor and 3,235 across
    the fleet, **0 disagreements** with the Python walk.

    Opt-in via `--full-scale`, like `test_ea17_matches_ground_truth_at_full_scale`:
    the load costs about three minutes. Skipped from the body, not a fixture,
    because `conftest.py` converts setup-phase skips under `--no-skips` and
    "you did not ask for the slow check" is not a broken environment.
    """
    if not request.config.getoption("--full-scale"):
        pytest.skip("needs --full-scale (about three minutes to build the graph)")

    from etl import generate as gen
    from etl.helpers import create_edges as add_edges
    from etl.helpers import create_nodes as add_nodes
    from etl.loader import NODE_LABELS

    fleet = gen.generate(seed=gen.DEFAULT_SEED, scale=1.0, operators=operators)
    for row in fleet.nodes["ClinicalTask"]:
        row["latency_budget_ms"] = 1
    client = engine_factory()
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            add_nodes(client, GRAPH, label, fleet.nodes[label])
    add_edges(client, GRAPH, fleet.edges)

    disagreed, busiest = [], 0
    for row in fleet.nodes["Sensor"]:
        sensor = row["id"]
        want = budget_truth(fleet, sensor)
        got = _rows(client, TASKS_OVER_BUDGET.replace(EA17_SUBJECT, sensor))
        busiest = max(busiest, want[0][1] if want else 0)
        if got != want:
            disagreed.append((sensor, got, want))

    assert busiest > 100, (
        f"the injected breach only reached {busiest} deployments for the "
        f"busiest sensor; this check is meant to run at a cardinality where an "
        f"unenforced join would show, so the injection is not doing its job")
    assert not disagreed, (
        f"{len(disagreed)} sensors disagree with the fleet's own edges at "
        f"scale 1.0: {disagreed[:3]}. The comma-joined second pattern is not "
        f"enforcing its join at this cardinality.")


def test_only_deployments_this_sensor_feeds_are_counted(engine_factory):
    """The shape the query must not have, on a graph where it is visible.

    `sensor:A` feeds `model:A`; `sensor:B` feeds `model:B`; both models solve
    the one task, and that task requires `sensor:A`. Both deployments are over
    budget. Only `deploy:A` is downstream of `sensor:A`, so only it belongs in
    `sensor:A`'s alert -- but a query joining through `REQUIRES_SENSOR ->
    SOLVES` alone counts both, and blames this sensor's fault for a breach on
    hardware its signal never reaches.

    This also measures the leading re-bind of `m` in the second pattern: an
    unenforced join would pair every model with every deployment and count two
    here as well.
    """
    client = engine_factory()
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": "sensor:A", "modality": "ecg"},
                  {"id": "sensor:B", "modality": "ecg"}])
    create_nodes(client, GRAPH, "SignalStage",
                 [{"id": "stage:A", "kind": "filter"},
                  {"id": "stage:B", "kind": "filter"}])
    create_nodes(client, GRAPH, "Model",
                 [{"id": "model:A", "name": "A"}, {"id": "model:B", "name": "B"}])
    create_nodes(client, GRAPH, "ModelVariant",
                 [{"id": "var:A"}, {"id": "var:B"}])
    create_nodes(client, GRAPH, "Deployment",
                 [{"id": "deploy:A", "latency_ms": 900.0},
                  {"id": "deploy:B", "latency_ms": 900.0}])
    create_nodes(client, GRAPH, "ClinicalTask",
                 [{"id": "task:T", "latency_budget_ms": 100}])
    create_edges(client, GRAPH, [
        ("Sensor", "sensor:A", "FEEDS", "SignalStage", "stage:A", None),
        ("Sensor", "sensor:B", "FEEDS", "SignalStage", "stage:B", None),
        ("SignalStage", "stage:A", "PRECEDES", "Model", "model:A", None),
        ("SignalStage", "stage:B", "PRECEDES", "Model", "model:B", None),
        ("Model", "model:A", "SOLVES", "ClinicalTask", "task:T", None),
        ("Model", "model:B", "SOLVES", "ClinicalTask", "task:T", None),
        ("ClinicalTask", "task:T", "REQUIRES_SENSOR", "Sensor", "sensor:A", None),
        ("ModelVariant", "var:A", "VARIANT_OF", "Model", "model:A", None),
        ("ModelVariant", "var:B", "VARIANT_OF", "Model", "model:B", None),
        ("Deployment", "deploy:A", "OF_VARIANT", "ModelVariant", "var:A", None),
        ("Deployment", "deploy:B", "OF_VARIANT", "ModelVariant", "var:B", None),
    ])

    retargeted = TASKS_OVER_BUDGET.replace(EA17_SUBJECT, "sensor:A")
    assert _rows(client, retargeted) == [(1, 1)], (
        "the scoped budget query should see one task and the one deployment "
        "sensor:A actually feeds; two deployments means it is joining through "
        "the task alone, or that the second pattern's join is not enforced")


def test_a_task_that_does_not_require_this_sensor_is_not_counted(engine_factory):
    """The `-[:REQUIRES_SENSOR]->(s)` that closes the loop, pinned directly.

    The other direction of the same pairing. `sensor:A` feeds `model:X`, and
    `model:X` solves a task that requires only `sensor:B` -- so the deployment
    is downstream of `sensor:A`, is over budget, and still must not appear in
    `sensor:A`'s alert: the sentence counts tasks that *depend on this
    sensor*, and this one does not.

    Without the loop back to `(s)`, the query would count it, and the fleet
    tests would not notice: stages are sampled from a shared pool, so whether
    such a model exists at `SCALE = 0.3` is luck.
    """
    client = engine_factory()
    create_nodes(client, GRAPH, "Sensor",
                 [{"id": "sensor:A", "modality": "ecg"},
                  {"id": "sensor:B", "modality": "ecg"}])
    create_nodes(client, GRAPH, "SignalStage", [{"id": "stage:A", "kind": "filter"}])
    create_nodes(client, GRAPH, "Model", [{"id": "model:X", "name": "X"}])
    create_nodes(client, GRAPH, "ModelVariant", [{"id": "var:X"}])
    create_nodes(client, GRAPH, "Deployment",
                 [{"id": "deploy:X", "latency_ms": 900.0}])
    create_nodes(client, GRAPH, "ClinicalTask",
                 [{"id": "task:B", "latency_budget_ms": 100}])
    create_edges(client, GRAPH, [
        ("Sensor", "sensor:A", "FEEDS", "SignalStage", "stage:A", None),
        ("SignalStage", "stage:A", "PRECEDES", "Model", "model:X", None),
        ("Model", "model:X", "SOLVES", "ClinicalTask", "task:B", None),
        ("ClinicalTask", "task:B", "REQUIRES_SENSOR", "Sensor", "sensor:B", None),
        ("ModelVariant", "var:X", "VARIANT_OF", "Model", "model:X", None),
        ("Deployment", "deploy:X", "OF_VARIANT", "ModelVariant", "var:X", None),
    ])

    assert _rows(client, TASKS_OVER_BUDGET.replace(EA17_SUBJECT, "sensor:A")) == [], (
        "sensor:A feeds this deployment, but the breached task requires "
        "sensor:B. Counting it would put a task in the alert that the line "
        "above -- 'N clinical tasks depend on it' -- does not count.")


def test_the_pages_that_describe_the_beat_are_not_ahead_of_the_code():
    """Two pages claim beat 7 exists. Neither claim is checked by anything else.

    `docs/alerting-scope.md` records #42 as delivered and `demo/README.md`
    says the story runs in seven beats -- the same class of cross-reference
    that this repo has twice found pointing at a guard that was not there.
    The beat is a `console.rule` in `demo/demo.py`, so it can be read.

    Deliberately in this module rather than `tests/test_alerting_scope.py`:
    the claim is about the demo, and that file is being edited on other
    branches.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    source = (root / "demo" / "demo.py").read_text(encoding="utf-8")
    rules = re.findall(r'console\.rule\(\s*"\[bold\](\d+)\.', source)
    assert rules, "no numbered beats found in demo/demo.py; update this test"
    beats = len(rules)
    assert [int(n) for n in rules] == list(range(1, beats + 1)), (
        f"the beats in demo/demo.py are numbered {rules}; they should run "
        f"1..{beats} with no gaps or repeats")

    scope = (root / "docs" / "alerting-scope.md").read_text(encoding="utf-8")
    claimed = re.search(r"delivered as beat (\d+) of `demo\.demo`", scope)
    assert claimed, (
        "docs/alerting-scope.md no longer says which beat delivers #42; that "
        "sentence is what this test pins, so update both together.")
    assert int(claimed.group(1)) <= beats, (
        f"the page says #42 is delivered as beat {claimed.group(1)}, and "
        f"demo/demo.py has {beats} beats.")

    readme = (root / "demo" / "README.md").read_text(encoding="utf-8")
    words = {6: "six", 7: "seven", 8: "eight", 9: "nine"}
    assert f"in {words.get(beats, beats)} beats" in readme, (
        f"demo/README.md does not say the story runs in "
        f"{words.get(beats, beats)} beats; demo/demo.py has {beats}.")
