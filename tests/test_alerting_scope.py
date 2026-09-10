"""The alerting-scope decision, pinned so it cannot rot silently (#33, #34, #38, #39).

`docs/alerting-scope.md` records a decision: take every alerting question the
existing schema can already answer, decline every one that needs a label this
repo would have to invent. A decision document is only worth writing if it stays
true, and this one makes two kinds of checkable claim.

**Claims about what is absent.** "Nothing carries a place", "no ownership
spine", "no alert state". If someone adds a `Site` label, the document becomes
wrong quietly -- nothing else in the suite would notice, because adding a label
breaks no existing test. These fail instead, and the message points at the
decision rather than at the schema, because the right response is to revisit the
decision or revert the label, not to update a number.

**Claims about what is present.** The five takeable questions were called
takeable because the edges and properties they need are already loaded. If
`ClinicalTask.latency_budget_ms` is renamed, `#37` stops being buildable on
today's schema and the table in that document is no longer a measurement.

Both directions matter. A decision page that only asserted absences would go
stale the moment the takeable half moved.
"""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "alerting-scope.md"

# Words that would indicate a physical-location property, matched as whole
# tokens rather than substrings. Substring matching had `city` inside
# `capacity`, `site` inside `website`, `rack` inside `track`, `place` inside
# `placement` and `zone` inside `ozone` -- so a `flash_capacity_kb` or a
# `Vendor.website` would fail this suite with a message telling the developer to
# revisit a decision about physical location, which is worse than not checking.
#
# `country` is deliberately absent: `Vendor.country` exists, means where a
# vendor is headquartered, and the document says so. Listing it here would need
# an exception that reads like a loophole.
LOCATION_WORDS = frozenset((
    "site", "sites", "zone", "zones", "room", "rooms", "building", "buildings",
    "floor", "floors", "location", "locations", "latitude", "longitude", "lat",
    "lon", "geo", "address", "region", "city", "place", "rack", "ward",
    "premises", "campus", "facility",
))

DECLINED_LABELS = {
    # #34 -- physical location. Labels as well as properties: a `Location` node
    # carrying only `{id, name}` trips no property hint, so guarding one and not
    # the other left the decision reversible in silence.
    "Site": "#34 — physical location",
    "Zone": "#34 — physical location",
    "Location": "#34 — physical location",
    "Room": "#34 — physical location",
    "Building": "#34 — physical location",
    "Facility": "#34 — physical location",
    "Campus": "#34 — physical location",
    "Ward": "#34 — physical location",
    "Team": "#39 — ownership",
    "Contact": "#39 — ownership",
    "Owner": "#39 — ownership",
    "Person": "#39 — ownership",
    "Asset": "#39 — ownership",
    "Organisation": "#39 — ownership",
    "Organization": "#39 — ownership",
    "Alert": "#38 — alerting state",
    "Rule": "#38 — alerting state",
    "Threshold": "#38 — alerting state",
}

# The same decisions expressed as edges. A relationship between two already
# permitted labels needs no new label, so `DECLINED_LABELS` cannot see it --
# an `OWNS` edge from `Vendor` to `Board` would make `DATASET_CARD.md`'s "no
# team, contact or `OWNS` edge" false with the suite green.
DECLINED_EDGES = {
    "DEPLOYED_AT": "#34 — physical location",
    "LOCATED_AT": "#34 — physical location",
    "INSTALLED_AT": "#34 — physical location",
    "OWNS": "#39 — ownership",
    "OWNED_BY": "#39 — ownership",
    "RESPONSIBLE_FOR": "#39 — ownership",
    "ALERTS": "#38 — alerting state",
    "TRIGGERS": "#38 — alerting state",
}


def tokens(name: str) -> set[str]:
    """`flash_capacity_kb` -> {flash, capacity, kb}. Splits on `_`, `-` and case."""
    import re

    return {t.lower() for t in re.split(r"[_\-\s]+|(?<=[a-z])(?=[A-Z])", name) if t}


@pytest.fixture(scope="module")
def fleet():
    from etl import generate as gen
    from etl import onnx_catalog as oc
    from etl import real_layer

    # All three cached sources, not just the ONNX catalogue: `build_real` calls
    # `ort.load_cached()` and `tiny.load_cached()` itself, so a partial
    # `etl.download_data` gave a raw traceback instead of the intended skip.
    try:
        ops = oc.load_cached()
        built = gen.generate(seed=20260814, scale=1.0, operators=ops)
        real_layer.build_real(built, ops)
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    return built


def properties_of(fleet, label: str) -> set[str]:
    return {key for row in fleet.nodes.get(label, []) for key in row}


def test_the_decision_document_exists():
    assert DOC.exists(), (
        f"{DOC.relative_to(ROOT)} is gone. The tests below pin claims it makes; "
        f"without it they are asserting nothing anyone can read."
    )


def test_no_label_this_repo_decided_not_to_invent_has_appeared(fleet):
    """#34, #38 and #39 were declined. A label appearing means that was reversed."""
    present = {label for label, rows in fleet.nodes.items() if rows}
    added = {label: why for label, why in DECLINED_LABELS.items() if label in present}
    assert not added, (
        f"labels this repo decided not to add are now in the graph: {added}. "
        f"That is not a test failure so much as a decision reversal -- update "
        f"docs/alerting-scope.md and DATASET_CARD.md, or drop the label."
    )


def test_no_edge_type_this_repo_decided_not_to_add_has_appeared(fleet):
    """Labels are not the only way to reverse these decisions.

    An `OWNS` edge between `Vendor` and `Board`, or a `DEPLOYED_AT` between
    `Deployment` and something existing, needs no new label -- so the label
    guard cannot see it, and `DATASET_CARD.md`'s "no team, contact or `OWNS`
    edge" would be false with the suite green.
    """
    present = {rel for _sl, _s, rel, _tl, _t, _p in fleet.edges}
    added = {rel: why for rel, why in DECLINED_EDGES.items() if rel in present}
    assert not added, (
        f"edge types this repo decided not to add are now in the graph: {added}. "
        f"Update docs/alerting-scope.md and DATASET_CARD.md, or drop the edge."
    )


def test_nothing_carries_a_physical_location(fleet):
    """The claim `#34` rests on, and the one most likely to go quietly stale."""
    found = [f"{label}.{prop}"
             for label, rows in fleet.nodes.items() if rows
             for prop in sorted(properties_of(fleet, label))
             if tokens(prop) & LOCATION_WORDS]
    assert not found, (
        f"location-like properties appeared: {found}. docs/alerting-scope.md "
        f"says nothing carries a place, and DATASET_CARD.md repeats it. Revisit "
        f"the decision rather than editing the sentence."
    )


def test_vendor_country_is_the_only_near_miss_and_is_empty_where_it_is_real(fleet):
    """Named in the document so nobody mistakes it for a deployment location.

    Two claims, because the earlier version of this test made neither. It was
    called `..._is_still_the_only_near_miss` and never checked uniqueness, and
    its value assertion passed if a single vendor out of fifteen carried one.
    """
    assert "country" in properties_of(fleet, "Vendor"), (
        "`Vendor.country` is gone. docs/alerting-scope.md calls it out as the "
        "one near-miss a reader might take for a location; if it no longer "
        "exists, that paragraph is answering a question nobody has."
    )

    # Uniqueness: no *other* label may carry a country-like property, or the
    # document's "the one near-miss" is wrong.
    others = [f"{label}.{prop}"
              for label, rows in fleet.nodes.items() if rows and label != "Vendor"
              for prop in sorted(properties_of(fleet, label))
              if prop.lower() in ("country", "nation", "territory")]
    assert not others, (
        f"docs/alerting-scope.md calls `Vendor.country` *the* near-miss; "
        f"{others} now qualify too."
    )

    # And the split the document publishes: populated on the synthetic vendors,
    # empty on every real one, because no upstream supplies it.
    by_provenance = {}
    for row in fleet.nodes["Vendor"]:
        by_provenance.setdefault(row.get("provenance"), set()).add(row.get("country"))
    assert by_provenance.get("real") == {""}, (
        f"real-layer vendors now carry a country: {by_provenance.get('real')}. "
        f"docs/alerting-scope.md says the one location-like property is empty "
        f"for every row that is real."
    )
    assert by_provenance.get("synthetic", set()) - {""}, (
        "synthetic vendors carry no country at all, so the document's example "
        "values are describing nothing"
    )


def test_the_takeable_questions_are_still_buildable_on_this_schema(fleet):
    """The other half of the decision: they were taken *because* nothing is missing."""
    edges = {rel for _sl, _s, rel, _tl, _t, _p in fleet.edges}
    labels = {label for label, rows in fleet.nodes.items() if rows}

    missing = []
    for issue, needed in (
        ("#35 blast radius", {"FEEDS", "NEXT_STAGE", "PRECEDES", "ON_BOARD"}),
        ("#36 root cause versus symptom", {"NEXT_STAGE"}),
        ("#40 certification context", {"GOVERNED_BY", "REQUIRES_SENSOR"}),
    ):
        absent = needed - edges
        if absent:
            missing.append(f"{issue}: edge types {sorted(absent)}")
    if "Certification" not in labels:
        missing.append("#40 certification context: label Certification")

    for issue, label, prop in (
        ("#37 silent degradation", "ClinicalTask", "latency_budget_ms"),
        ("#37 silent degradation", "Deployment", "latency_ms"),
        ("#37 silent degradation", "Deployment", "fallback_op_count"),
    ):
        if prop not in properties_of(fleet, label):
            missing.append(f"{issue}: {label}.{prop}")

    # The join the document lists as verified. Without it #37's properties exist
    # and cannot be brought together, so the table would stay green while
    # ceasing to be a measurement.
    missing.extend(f"#37 silent degradation: Deployment->ClinicalTask needs {edge}"
                   for edge in ("OF_VARIANT", "VARIANT_OF", "SOLVES")
                   if edge not in edges)

    assert not missing, (
        "docs/alerting-scope.md takes these on because today's schema already "
        "holds what they need. It no longer does:\n  " + "\n  ".join(missing)
    )


def test_the_operator_name_clash_still_exists(fleet):
    """#39's trap: an organisational `Operator` would collide with the ONNX one."""
    operators = fleet.nodes.get("Operator", [])
    assert len(operators) > 100, (
        f"`Operator` holds {len(operators)} nodes. docs/alerting-scope.md warns "
        f"that the name is taken by ONNX operators, which is why an ownership "
        f"spine cannot reuse it. If that changed, the warning needs rewriting."
    )
    assert "since_version" in properties_of(fleet, "Operator"), (
        "`Operator` no longer looks like an ONNX operator, so the name-clash "
        "warning in docs/alerting-scope.md may no longer apply."
    )
