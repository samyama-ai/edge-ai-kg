"""Where a generated deployment physically sits (#34).

Its own module rather than another block in `etl/generate.py`, which is already
605 lines -- long enough that one more block makes it harder to read, not just
longer. The `Site` spine is also the one part of the generator that exists
because of a *decision* -- `docs/location-scope.md` reverses the decline
recorded in `docs/alerting-scope.md` -- so it is worth reading on its own.

Everything here is synthetic and generated-layer only. The real layer's MLPerf
Tiny deployments are never placed: nobody published where those submissions ran,
and inventing a location for a node stamped `provenance: "real"` is the exact
confusion this repo's provenance rule exists to prevent.
"""
from __future__ import annotations

from collections.abc import Callable

# Fictional facility names, for the same reason the vendors are fictional: a
# real hospital name would read as a claim about a real installation.
CAMPUSES = [
    ("Meridian General", "hospital", "IN"),
    ("Saltmarsh Clinic", "clinic", "UK"),
    ("Ferrous Line 4", "factory", "DE"),
    ("Anseri Home Care", "home", "US"),
]

# Two levels in one label, not two labels. A `Site` row is the smallest place a
# deployment is installed -- a ward, a line, a bay -- and `campus` groups the
# sites that share a building. A `Campus` label would add a node whose only
# edges point down at sites, identity nothing else references, and one more hop
# on every "is this site-wide?" query; a property carries the grouping at no
# cost.
#
# `zone` is deliberately *not* an edge property on `DEPLOYED_AT`. It reads
# correctly on the embedded build (measured), but the catalog is swept over
# HTTP by `benchmarks/run_benchmark.py` and no engine note covers edge
# properties on the 1.7.0 server -- so a catalog query resting on one would be
# a shape nothing in this repo has measured. The site *is* the zone here.
# Per kind, because a name has to be one a reader would find in that kind of
# place: a factory has no `Theatre 1` and a home-care round has no `Bay 2`.
# Cycling one flat list across the campuses produced exactly those, and a name
# that does not fit its site is a small thing that makes every figure beside it
# read as careless.
#
# Three each, so every kind wraps at the same site count and the wrap rule
# below has one number rather than four.
SITE_SUFFIXES_BY_KIND = {
    "hospital": ["Ward 3", "Ward 7", "Theatre 1"],
    "clinic": ["Clinic Floor 1", "Clinic Floor 2", "Day Unit"],
    "factory": ["Line A", "Line B", "Bay 2"],
    "home": ["Round North", "Round South", "Transport"],
}

# The flat count the caller scales against: how many sites fit before any kind
# has to wrap. Four campuses times three names each.
SITE_SUFFIXES = [name for names in SITE_SUFFIXES_BY_KIND.values()
                 for name in names]


# Never fewer than this, whatever the scale. At `--scale 0.05` the rounding
# gave one site in one campus, and "is this site-wide or one device" cannot be
# asked of a graph with a single place -- every row is site-wide by
# construction. Two sites land in two campuses, because the campus cycles.
MIN_SITES = 2


def build_sites(scaled_count: int,
                rid: Callable[[str, int], str]) -> list[dict]:
    """`scaled_count` sites (at least `MIN_SITES`), cycling the campuses.

    Scaled like everything else: 12 sites over 4 campuses at `--scale 1.0`, and
    4 sites over 4 campuses at 0.3 -- the campus list cycles every four sites,
    so four sites is four campuses, not two.

    **Names are unique within a campus**, which the caller's queries depend on.
    `EA20` groups by `(campus, name)` for readability, so two sites sharing a
    campus *and* a name would merge into one row and silently under-report both.
    Once a kind has more sites than it has names -- more than
    `len(SITE_SUFFIXES_BY_KIND[kind])`, three today -- the list wraps and the
    wrap number is what keeps them apart: `Ward 3`, then `Ward 3 (2)`. Stated
    against the name lists rather than a `--scale` threshold, because the scale
    at which that happens is the caller's rounding, not this module's.

    Names come from the campus's **kind**, so a factory gets a line or a bay
    and a hospital a ward or a theatre.

    `rid` is the caller's id minter, so site ids read like every other id here
    rather than this module inventing a second convention.
    """
    sites = []
    for i in range(max(MIN_SITES, scaled_count)):
        campus, kind, region = CAMPUSES[i % len(CAMPUSES)]
        # Position within this campus, not overall: the campuses cycle, so
        # every fourth site shares a campus and it is that sequence which must
        # not repeat a name.
        nth_here = i // len(CAMPUSES)
        names = SITE_SUFFIXES_BY_KIND[kind]
        name = names[nth_here % len(names)]
        wrap = nth_here // len(names)
        if wrap:
            name = f"{name} ({wrap + 1})"
        sites.append({
            "id": rid("site", i),
            "name": name,
            "kind": kind,
            "campus": campus,
            "region": region,
        })
    return sites
