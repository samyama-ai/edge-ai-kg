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

import collections
import pathlib
import re

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
    "lon", "geo", "gps", "address", "region", "district", "city", "place",
    "rack", "ward", "premise", "premises", "campus", "facility",
    # `docs/alerting-scope.md` and `DATASET_CARD.md` both say "nothing carries a
    # site, zone, room or coordinate" -- so `coordinate` has to be here, and a
    # `Deployment.gps_coordinates` or `Board.postcode` would otherwise make the
    # central claim false with the suite green.
    "coordinate", "coordinates", "postcode", "postal",
))

# `Site` is deliberately absent from this dict: #34 was declined and then
# **taken**, and `docs/location-scope.md` records the reversal with the case
# against. The rest of the #34 family stays declined, so the reversal is one
# label wide -- a `Zone` or a `Building` appearing is still a decision nobody
# wrote down. `test_the_site_spine_is_exactly_what_was_agreed` below pins what
# `Site` is allowed to be.
DECLINED_LABELS = {
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
# `DEPLOYED_AT` is likewise permitted now, and only from `Deployment`, which
# the pinning test below checks. `LOCATED_AT` and `INSTALLED_AT` stay declined:
# they would place a *type* -- a board or a sensor -- which is the shape
# `docs/location-scope.md` argues against.
DECLINED_EDGES = {
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


def test_the_delivery_claims_match_the_catalog():
    """Every ``delivered as `EAnn` `` on the page, checked against `BY_ID`.

    The previous version keyed on one exact phrase -- "not** delivered on
    `main`" -- so any other wording of "pending" walked past it, and the page
    could claim a delivery the catalog did not have. This derives both
    directions from the catalog instead:

    - every id the page says is delivered must be in `BY_ID`;
    - the range the page states ("the catalog is `EA01`-`EAnn`") must be the
      catalog's actual last id;
    - and an issue whose query exists while the page carries no "delivered as"
      claim for it fails. That is the absence of a claim, not the presence of
      the word "pending" -- any other wording fails identically, because what
      is compared is the set of claimed ids against the catalog.
    """
    from benchmarks.queries import BY_ID

    text = DOC.read_text(encoding="utf-8")

    claimed = set(re.findall(r"delivered as `(EA\d\d)`", text))
    assert claimed, (
        "no `delivered as `EAnn`` claim on the page at all. The verdict table "
        "is what this test reads; if its shape changed, update this test.")
    missing = sorted(qid for qid in claimed if qid not in BY_ID)
    assert not missing, (
        f"docs/alerting-scope.md says {missing} are delivered, and the catalog "
        f"does not hold them. Either the queries were reverted or the page is "
        f"claiming work that has not landed.")

    stated = re.search(r"the catalog is\s+`EA01`-`(EA\d\d)`", text, re.IGNORECASE)
    assert stated, (
        "the page no longer states the catalog range as `EA01`-`EAnn`; that "
        "sentence is what this test pins, so update both together.")
    # By number, not lexicographically: `max()` over strings ranks "EA9"
    # above "EA10", and the catalog will pass EA99 eventually.
    last = max(BY_ID, key=lambda qid: int(qid[2:]))
    assert stated.group(1) == last, (
        f"the page says the catalog is `EA01`-`{stated.group(1)}`; the catalog "
        f"ends at `{last}`. A reader trusts that range to know what exists.")

    # Derived from the verdict table rather than a hardcoded triple, and
    # scoped to it: the prose names `EA01` and `EA07` as examples, which are
    # not this page's to deliver. A table row is `| #nn | ... | verdict |`, so
    # a row marked **take** that names a query must also say it was delivered
    # -- the drift being a shipped query whose row still reads as work not yet
    # done. Listing the ids here by hand would mean editing this test every
    # time the alerting theme grows.
    rows = [line for line in text.splitlines() if line.startswith("| #")]
    assert rows, (
        "no verdict table rows found; the table is what this test reads, so "
        "if its shape changed, update this test with it.")
    in_table = {qid for row in rows for qid in re.findall(r"`(EA\d\d)`", row)}
    assert in_table, (
        "the verdict table names no catalog query at all; if the delivery "
        "claims moved out of it, this test is pinning nothing.")
    unclaimed = sorted(qid for qid in in_table if qid in BY_ID and qid not in claimed)
    assert not unclaimed, (
        f"{unclaimed} are named in the verdict table and exist in the catalog, "
        f"but carry no `delivered as` claim. Either add the claim, or say why "
        f"the query exists while the verdict reads otherwise.")


def test_no_label_this_repo_decided_not_to_invent_has_appeared(fleet):
    """#38 and #39 stand declined; #34 was reversed for `Site` alone.

    A label from any of the three families appearing here means a decision
    moved without anyone writing it down -- which is the thing this file
    exists to make impossible, not the thing it forbids.
    """
    # Every declared label, not only the populated ones. A `Site` gated behind a
    # layer or a scale threshold would yield zero rows at this fixture's seed
    # and scale, reversing the #34 decision in the schema while this reported
    # nothing.
    present = set(fleet.nodes)
    added = {label: why for label, why in DECLINED_LABELS.items() if label in present}
    assert not added, (
        f"labels this repo decided not to add are now in the graph: {added}. "
        f"That is not a test failure so much as a decision reversal -- record "
        f"it the way docs/location-scope.md records #34's, update "
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


def test_only_the_site_label_carries_a_physical_location(fleet):
    """#34 was reversed for `Site` alone; everything else still carries no place.

    The original form of this test asserted that *nothing* carries a location,
    and it is what caught the reversal when `Site` landed. Deleting it would
    have removed the only guard over the whole family, so it is narrowed
    instead: `Site` may carry a place, and a `Board.postcode` or a
    `Deployment.gps_coordinates` still fails here.
    """
    found = [f"{label}.{prop}"
             for label, rows in fleet.nodes.items() if rows and label != "Site"
             for prop in sorted(properties_of(fleet, label))
             if tokens(prop) & LOCATION_WORDS]
    # Edges carry properties too -- the 6th tuple element -- and a
    # `(:Deployment)-[:ON_BOARD {site: "plant-2"}]->(:Board)` would put a place
    # in the graph without adding a property to any node. Scanning only nodes
    # left that hole open.
    found += sorted({f"{rel}.{prop}"
                     for _sl, _s, rel, _tl, _t, props in fleet.edges
                     for prop in (props or {})
                     if tokens(prop) & LOCATION_WORDS})
    assert not found, (
        f"location-like properties appeared outside `Site`: {found}. "
        f"docs/location-scope.md reverses #34 for `Site` and for nothing else, "
        f"and DATASET_CARD.md repeats that. Revisit the decision rather than "
        f"editing the sentence."
    )


def test_the_site_spine_is_exactly_what_was_agreed(fleet):
    """The reversal, bounded: one label, one edge, and only from `Deployment`.

    `docs/location-scope.md` argues for placing an *instance* and against
    placing a *type*. Without this, `Board -[:DEPLOYED_AT]-> Site` would satisfy
    every other test on this page while being the shape that page rejects.
    """
    # `.get(...)`, not `in`: the key survives an empty row list, so
    # `"Site" in fleet.nodes` is true for a generator that declares the label
    # and emits nothing -- the reversal reverted in everything but name.
    assert fleet.nodes.get("Site"), (
        "`Site` holds no rows, but docs/location-scope.md says #34 was taken. "
        "Either the reversal was reverted -- in which case restore `Site` to "
        "DECLINED_LABELS above and say so in alerting-scope.md -- or the "
        "generator stopped emitting it."
    )
    sources = {src_label for src_label, _s, rel, _tl, _t, _p in fleet.edges
               if rel == "DEPLOYED_AT"}
    assert sources == {"Deployment"}, (
        f"DEPLOYED_AT runs from {sorted(sources)}. docs/location-scope.md "
        f"places one installed instance, and a Board or a Sensor is a type -- "
        f"the objection that decision turns on."
    )
    targets = {tgt_label for _sl, _s, rel, tgt_label, _t, _p in fleet.edges
               if rel == "DEPLOYED_AT"}
    assert targets == {"Site"}, f"DEPLOYED_AT points at {sorted(targets)}, not Site"

    # Cardinality in both directions. `docs/location-scope.md` rests on "one
    # installed instance, in exactly one place": two sites turn EA20's per-site
    # count into a set union, and *no* site drops a deployment out of the
    # answer entirely. A `Counter` over the edges sees only the first of those
    # -- a deployment with no `DEPLOYED_AT` edge contributes no key at all --
    # so the generated deployments are enumerated and checked against it.
    placements = collections.Counter(
        src for _sl, src, rel, _tl, _t, _p in fleet.edges if rel == "DEPLOYED_AT")
    multi = {did: k for did, k in placements.items() if k > 1}
    assert not multi, (
        f"deployments placed more than once: {sorted(multi)[:3]}. "
        f"docs/location-scope.md commits to one site per deployment.")
    # Real deployments are MLPerf submissions with no known location and are
    # given none on purpose, so the floor is the generated half.
    generated = {d["id"] for d in fleet.nodes.get("Deployment", [])
                 if d.get("provenance") != "real"}
    assert generated, "no generated deployments; this check is measuring nothing"
    unplaced = sorted(generated - set(placements))
    assert not unplaced, (
        f"{len(unplaced)} generated deployments sit at no site "
        f"({unplaced[:3]}). EA20 counts `deployments_here` per site, so an "
        f"unplaced deployment is missing from the answer rather than reported "
        f"as homeless -- and docs/location-scope.md commits to exactly one "
        f"place per installed instance.")


def test_the_site_label_carries_only_what_the_decision_allows(fleet):
    """The reversal is bounded by *shape* too, not only by label name.

    #34 was taken as "where a deployment sits", explicitly not as an asset
    register: `docs/location-scope.md` says there is no move history, no
    commissioning date, no asset tag and no person. Those would arrive as
    properties on `Site` rather than as a new label, so `DECLINED_LABELS`
    above cannot see them coming.
    """
    allowed = {"id", "name", "kind", "campus", "region", "provenance", "source"}
    present = properties_of(fleet, "Site")
    extra = sorted(present - allowed)
    assert not extra, (
        f"`Site` grew {extra}. docs/location-scope.md commits to a place and "
        f"nothing else -- a commissioning date or an owner is the CMDB "
        f"duplication that decision accepts as its cost. Widen the decision "
        f"first, then this list."
    )
    assert "id" in present and "campus" in present, (
        f"`Site` lost {sorted(allowed - present)}; EA20 groups by campus")


def test_vendor_country_is_the_only_near_miss_and_is_empty_where_it_is_real(fleet):
    """Named in the document so nobody mistakes it for a deployment location.

    Two claims, and the name promises both: that `country` is the *only*
    property a reader could mistake for a location, and that it is empty on
    every real-layer vendor. A uniqueness check that never enumerates the
    other labels, or a value check satisfied by one vendor out of fifteen,
    would leave the document's "one near-miss" claim unenforced.
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


def test_every_quote_from_the_decline_is_verbatim():
    """`location-scope.md` argues against a page it quotes; the quotes must hold.

    The reversal's whole method is "here is what the decline said, and here is
    what changed". A quote that has drifted from its source -- even by a
    dropped `**` -- makes the argument look like it is answering something
    nobody wrote, and this is the one kind of error a reader cannot catch
    without opening both files.

    Only block-quoted, italicised strings are checked, which is how that page
    marks a quotation from another document. Whitespace and the `> ` prefix
    are normalised, because line wrapping is not part of the quote.

    Every `*"` opening in a block must yield a quote, and that guard is
    load-bearing rather than tidy. `*"..."*` closes on the first `"` followed
    by `*`, so a quotation holding an *italicised* inner quote ends early --
    and the truncated text is a **prefix** of the source, which passes the
    verbatim check below while leaving the rest of the sentence unchecked.
    Measured: quoting `the question that prompted this -- *"a company keeps a
    sensor in some place"* -- cannot be expressed` captures 75 of its 100
    characters and the tail goes unverified. Counting the openings turns that
    silence into a failure.
    """
    import re

    reversal = (ROOT / "docs" / "location-scope.md").read_text(encoding="utf-8")
    decline = (ROOT / "docs" / "alerting-scope.md").read_text(encoding="utf-8")

    def flatten(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"^> ?", "", text, flags=re.MULTILINE)).strip()

    source = flatten(decline)
    quotes, openings = [], 0
    for block in re.findall(r"(?:^> .*\n)+", reversal, re.MULTILINE):
        flat = flatten(block)
        openings += flat.count('*"')
        quotes += re.findall(r'\*"(.+?)"\*', flat, re.DOTALL)
    assert len(quotes) == openings, (
        f"{openings} quotation openings in docs/location-scope.md and "
        f"{len(quotes)} quotes parsed from them. A quote holding an "
        f"italicised inner quote closes early, and the truncated prefix still "
        f"matches the source -- so the rest of that sentence would go "
        f"unchecked. Re-mark the inner quotation, or quote it in two pieces.")
    assert quotes, (
        "no quotations found in docs/location-scope.md. Either the page stopped "
        "quoting the decline -- in which case its argument needs re-reading -- "
        "or the `> *\"...\"*` convention changed and this test with it.")
    missing = [q for q in quotes if q not in source]
    assert not missing, (
        f"docs/location-scope.md quotes {missing} as coming from "
        f"docs/alerting-scope.md, and that text is not in it. Copy the "
        f"sentence again rather than paraphrasing it: the reversal's argument "
        f"rests on answering what the decline actually said.")
