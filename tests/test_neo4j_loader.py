"""The Neo4j loader's guarantees: same graph in, or no timing out (#47).

`benchmarks/neo4j_client.py` fills a Neo4j with the same `Fleet` the Samyama
side gets, and translates `schema/edge_ai_kg.cypher` so both ends have the same
22 indexes. Everything here is about whether that is true, because a comparison
given two different graphs measures nothing -- and the ways it can go wrong are
all quiet:

- **edge properties dropped**, so Neo4j got a smaller graph to load and a
  different one to query;
- **a partial load**, which `UNWIND ... MATCH ... CREATE` performs without
  complaint and which reports a *faster* time than a complete one;
- **nulls kept on nodes but stripped on edges**, one loader disagreeing with
  itself about what a property is;
- **a destructive wipe running before validation**, which emptied the target
  and then refused to load.

That last one was wrong twice in review -- the identifier check was moved above
the node `CREATE` loop and was still below the `DETACH DELETE`. Ordering is the
guarantee, so it is asserted directly rather than inferred from behaviour.

The comparison harness that uses this loader, and the timings themselves, are
separate: see `benchmarks/compare_neo4j.py` and its tests.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap
import types

import pytest
from click.testing import CliRunner

from benchmarks import compare_neo4j, neo4j_client


def test_the_neo4j_loader_writes_edge_properties():
    """Both engines must be given the same graph, not just the same queries.

    `etl/helpers.py` writes edge properties on the Samyama side; this loader
    dropped them, so Neo4j got 310 `USES_OPERATOR` edges without their `count`
    at scale 1.0. No catalog query binds an edge variable today, so no
    published number moved -- the first one that did would have had two
    answers and no way to tell which was right.
    """
    from benchmarks import neo4j_client

    sent = []

    class Fake:
        """Answers the two counts the loader checks: nodes held, edges created.

        `count(r)` must report the number of edges in the fleet or the
        partial-load guard fires -- which is the guard working, but it is not
        what this test is about.
        """

        def run(self, statement, parameters=None):
            sent.append((statement, parameters))
            if "count(r)" in statement:
                return [[len(fleet.edges)]]
            if "count(n)" in statement or "RETURN count" in statement:
                return [[0]]
            return []

    fleet = types.SimpleNamespace(
        nodes={"Model": [{"id": "model:1"}], "Operator": [{"id": "op:1"}]},
        edges=[("Model", "model:1", "USES_OPERATOR", "Operator", "op:1",
                {"count": 3, "_internal": "x", "empty": None})])

    neo4j_client.load_neo4j(Fake(), fleet)
    edge = [(s, p) for s, p in sent if "CREATE (a)-[" in s]
    assert edge, f"no edge CREATE was issued: {[s for s, _ in sent]}"
    statement, params = edge[0]
    assert "SET r = row.props" in statement, (
        f"edge properties are not written: {statement!r}"
    )
    assert params["rows"][0]["props"] == {"count": 3}, (
        f"`_`-prefixed internals and nulls must be stripped, as "
        f"`etl/helpers.py` does: got {params['rows'][0]['props']!r}"
    )


def test_nothing_is_sent_before_the_identifiers_are_checked():
    """The refusal has to come before the first interpolation, not after it.

    The check sat below the node-`CREATE` loop, which had already interpolated
    and sent every label -- so the one statement it existed to gate ran first,
    and the refusal arrived too late to prevent anything. Ordering is the whole
    guarantee here, and nothing else in the suite can see it.
    """
    from benchmarks import neo4j_client

    sent = []

    class Fake:
        def run(self, statement, parameters=None):
            sent.append(statement)
            return [[0]]

    fleet = types.SimpleNamespace(
        nodes={"Model": [{"id": "m:1"}], "Bad Label": [{"id": "b:1"}]},
        edges=[("Model", "m:1", "USES", "Model", "m:1", {})])

    with pytest.raises(neo4j_client.Neo4jLoadError, match="not plain identifiers"):
        neo4j_client.load_neo4j(Fake(), fleet, force_wipe=True)
    # Every statement, not just the ones containing "CREATE". Filtering on
    # CREATE is how this test missed that `DETACH DELETE` ran first: the wipe
    # emptied the target database and the refusal arrived afterwards, so a
    # fleet with an unusable label destroyed the data and loaded nothing. The
    # guarantee is that *nothing* is sent, so that is what is asserted.
    assert sent == [], (
        f"the identifier check must run before anything reaches the server -- "
        f"the count query and the wipe included. These were sent first: {sent}"
    )


def test_node_rows_and_edge_props_agree_about_nulls():
    """One loader, two halves, one rule.

    `SET n = row` with a null is a no-op in Neo4j, so this changes nothing the
    server stores -- it stops the two halves disagreeing about what a property
    is, which is the disagreement under which the edge half silently dropped
    every property it had.
    """
    from benchmarks import neo4j_client

    sent = []

    class Fake:
        def run(self, statement, parameters=None):
            sent.append((statement, parameters))
            return [[len(fleet.edges)]] if "count(r)" in statement else [[0]]

    fleet = types.SimpleNamespace(
        nodes={"Model": [{"id": "m:1", "nothing": None, "_hidden": 1}]},
        edges=[("Model", "m:1", "USES", "Model", "m:1",
                {"count": 2, "nothing": None, "_hidden": 1})])
    neo4j_client.load_neo4j(Fake(), fleet)

    node = next(p for s, p in sent if "CREATE (n:" in s)
    edge = next(p for s, p in sent if "CREATE (a)-[" in s)
    assert node["rows"][0] == {"id": "m:1"}, node["rows"][0]
    assert edge["rows"][0]["props"] == {"count": 2}, edge["rows"][0]["props"]


def test_the_wipe_the_refusal_demands_is_actually_reachable():
    """`--force-wipe` has to exist, and has to reach `load_neo4j`.

    `load_neo4j` refuses a non-empty database with "Pass --force-wipe if that
    is what you want" -- and for one commit that flag did not exist. `main`
    declared no such option and called `load_neo4j(neo, fleet)` with the
    default, so running the documented command a second time told you to pass
    a flag click would then reject as "no such option". The only ways through
    were `--reuse-neo4j`, which demands exact counts and all 22 indexes, or
    wiping by hand. The batched-delete block was unreachable.

    Three things, because each failed independently: the option exists, the
    refusal names an option that exists, and `main` forwards it.

    The third is read off the **syntax tree** of `main`'s callback: find the
    calls to `load_neo4j` and check one passes the flag, by keyword or in the
    `force_wipe` position. An earlier version grepped the source for
    `"force_wipe=force_wipe"`, which asserts a spelling -- a correct refactor
    to a positional argument or a renamed local fails it, and a comment
    mentioning the string passes it. Running `main` would be better still, but
    it reaches the loader only with a live Neo4j, so the tree is what is
    available without one.
    """
    declared = {opt for param in compare_neo4j.main.params for opt in param.opts}
    assert "--force-wipe" in declared, (
        f"`load_neo4j` refuses and tells the user to pass --force-wipe; the "
        f"command accepts {sorted(o for o in declared if o.startswith('--'))}"
    )

    # Flags named in the refusal *message*, not anywhere in the function. The
    # whole-source scan also read comments, so a note about a removed flag
    # failed a test about what users are told to type.
    refusal = inspect.getsource(neo4j_client.load_neo4j)
    told_to_pass = re.findall(r"Pass (--[a-z][a-z0-9-]+)", refusal)
    assert told_to_pass, (
        "no `Pass --flag` in load_neo4j's refusals; if the wording changed, "
        "point this at the new one rather than dropping the check"
    )
    for flag in told_to_pass:
        assert flag in declared, (
            f"`load_neo4j`'s refusal tells the user to pass `{flag}`, which "
            f"`compare_neo4j.main` does not accept. A message naming an option "
            f"that does not exist is a dead end."
        )

    position = list(
        inspect.signature(neo4j_client.load_neo4j).parameters).index("force_wipe")
    tree = ast.parse(textwrap.dedent(inspect.getsource(compare_neo4j.main.callback)))
    forwarded = [
        call for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", getattr(call.func, "attr", None)) == "load_neo4j"
        and (any(kw.arg == "force_wipe" for kw in call.keywords)
             or len(call.args) > position)
    ]
    assert forwarded, (
        "`main` calls `load_neo4j` without passing `force_wipe`, by keyword or "
        "in its position. However the flag is spelled at the call site, the "
        "wipe has to be reachable -- it was not, for one commit, and the "
        "refusal above told users to pass a flag that did nothing."
    )


def test_reuse_and_force_wipe_together_are_refused():
    """They contradict, and the silent winner is the non-destructive one.

    Which sounds harmless until you notice the user asked for a wipe and did
    not get one, then read timings from a database holding someone else's
    fleet.
    """
    # A password, so the refusal under test is the one that fires. Without
    # it click stops first on the missing option -- still a non-zero exit, so
    # the first assertion would pass for the wrong reason.
    result = CliRunner().invoke(compare_neo4j.main, ["--reuse-neo4j", "--force-wipe"],
                                env={"NEO4J_PASSWORD": "unused"})
    assert result.exit_code != 0, "both flags together must not be accepted"
    # Both flag names, rather than a phrase from the sentence. Coupling to
    # "contradict each other" would fail on a reworded message that still
    # refuses correctly, which is a test failing for the wrong reason.
    said = str(result.output) + str(result.exception)
    assert "--reuse-neo4j" in said and "--force-wipe" in said, (
        f"the refusal must name both flags so the user knows which to drop; "
        f"got {said!r}"
    )


def test_the_password_has_no_built_in_default():
    """It comes from `--password` or `NEO4J_PASSWORD`, and nowhere else.

    A built-in default had to be published to be usable -- it was, in the
    module docstring's docker line -- while `--help` claimed to hide it. Now
    the docker line and the flag read the same variable and neither holds the
    value.
    """
    result = CliRunner().invoke(compare_neo4j.main, [], env={"NEO4J_PASSWORD": None})
    assert result.exit_code == 2, result.output
    assert "--password" in result.output, result.output
    assert "benchmarkpw" not in (compare_neo4j.__doc__ or ""), (
        "the module docstring still publishes a password"
    )
