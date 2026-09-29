"""Assembling the catalog from four modules can lose a query quietly.

`benchmarks/queries.py` was one 916-line list, so a duplicate id was visible
and the order was whatever the file said. It is now four modules stitched
together in `benchmarks/catalog/__init__.py`, which makes two new failures
possible and neither of them raises on its own:

- a dict comprehension keeps the **last** entry for a repeated id, so two
  modules both defining `EA20` would leave the catalog one query short while
  every count still looked plausible;
- nothing sorts `QUERIES`, so a module imported in the wrong position
  reorders what `run_benchmark` sweeps and what `demo.questions` walks.

Both are asserted at assembly, and driven here with hand-built lists rather
than by importing the real catalog -- the point is what happens when the
catalog is *wrong*, which the real one is not.

`retargeted_ea21` is tested here too. It moved into the package with the
split because a caller with a live alert set is production, not a test
helper, and it arrived with no tests of its own.
"""
from __future__ import annotations

import pytest

from benchmarks.catalog import _assembled, retargeted_ea21
from benchmarks.catalog.subjects import EA21_ALERTS, alert_list
from benchmarks.queries import BY_ID, QUERIES


def _q(qid: str) -> dict:
    return {"id": qid, "title": qid, "question": qid, "why_graph": qid, "cypher": qid}


def test_the_shipped_catalog_assembles():
    """The real one, so the guards below are not only exercised on fakes."""
    assert len(BY_ID) == len(QUERIES) == 21
    assert list(BY_ID) == sorted(BY_ID), "ids are assembled in catalog order"


def test_a_duplicate_id_is_refused_rather_than_silently_dropped():
    """The failure the `EA20` reservation was written to avoid.

    Two queries sharing an id is not a crash without this: `BY_ID` keeps the
    second, the catalog runs, and one question is simply never asked again.
    """
    with pytest.raises(ValueError, match=r"EA07.*more than once"):
        _assembled([_q("EA01"), _q("EA07"), _q("EA07"), _q("EA09")])


def test_the_duplicate_message_says_how_many_queries_would_vanish():
    """A count, because "a duplicate" and "three queries gone" read differently."""
    with pytest.raises(ValueError, match="1 quer"):
        _assembled([_q("EA01"), _q("EA01"), _q("EA02")])


def test_a_module_imported_out_of_position_is_refused():
    """Nothing sorts `QUERIES`; the module order *is* the catalog order."""
    with pytest.raises(ValueError, match="out of order"):
        _assembled([_q("EA01"), _q("EA13"), _q("EA07")])


def test_retargeting_rewrites_both_occurrences_of_the_alert_set():
    """One rewritten leg would rank one population against another."""
    original = BY_ID["EA21"]["cypher"]
    wanted = ["sensor:00042", "sensor:00043"]
    out = retargeted_ea21(wanted)

    assert out.count(alert_list(wanted)) == 2
    assert alert_list(EA21_ALERTS) not in out, "the catalog's set must be gone"
    assert out != original


def test_retargeting_refuses_an_id_that_escaping_would_change():
    """`sensor:x"` and `sensor:x` must not become the same query.

    Answering about a different sensor is worse than failing, for a query
    someone is paged on.
    """
    with pytest.raises(ValueError):
        retargeted_ea21(['sensor:x") OR true //'])


def test_retargeting_refuses_a_bare_string():
    """A `str` iterates as characters, which would ask about 12 one-letter ids."""
    with pytest.raises(TypeError):
        retargeted_ea21("sensor:00042")


def test_the_package_builds_BY_ID_through_the_guard(monkeypatch):
    """The guard has to be on the path the import actually takes.

    Testing `_assembled` directly proves the function works and nothing about
    whether anything calls it: reverting `BY_ID = _assembled(QUERIES)` to a
    plain dict comprehension left every test above passing. This reloads the
    package with a duplicate planted in one of the four modules, which is how
    a real collision would arrive -- two modules edited by two PRs.
    """
    import importlib

    import benchmarks.catalog as catalog
    import benchmarks.catalog.core as core

    monkeypatch.setattr(core, "CORE", [*core.CORE, _q(core.CORE[0]["id"])])
    with pytest.raises(ValueError, match="more than once"):
        importlib.reload(catalog)
    importlib.reload(core)
    importlib.reload(catalog)          # leave the module as we found it
    assert len(catalog.BY_ID) == 21
