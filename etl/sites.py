"""Where a generated deployment physically sits (#34).

Its own module rather than another block in `etl/generate.py`, which is already
605 lines and past the size at which the review harness stops reading a file.
The `Site` spine is also the one part of the generator that exists because of a
*decision* -- `docs/location-scope.md` reverses the decline recorded in
`docs/alerting-scope.md` -- so it is worth being able to read on its own.

Everything here is synthetic and generated-layer only. The real layer's MLPerf
Tiny deployments are never placed: nobody published where those submissions ran,
and inventing a location for a node stamped `provenance: "real"` is the exact
confusion this repo's provenance rule exists to prevent.
"""
from __future__ import annotations

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
SITE_SUFFIXES = ["Ward 3", "Ward 7", "Theatre 1", "Bay 2", "Line A", "Line B",
                 "Round North", "Round South", "Clinic Floor 1",
                 "Clinic Floor 2", "Store", "Transport"]


def build_sites(scaled_count: int, rid) -> list[dict]:
    """`scaled_count` sites, cycling the campuses so more than one always exists.

    Scaled like everything else, so a `--scale 0.3` graph still has more than
    one campus to group by: 12 sites over 4 campuses at 1.0, 4 over 2 at 0.3.
    Fewer than two campuses would make "site-wide or one device" unanswerable at
    small scale, which is the question the label exists for -- and
    `tests/test_site_spine.py` fails if it ever drops to one.

    `rid` is the caller's id minter, so site ids read like every other id here
    rather than this module inventing a second convention.
    """
    sites = []
    for i in range(scaled_count):
        campus, kind, region = CAMPUSES[i % len(CAMPUSES)]
        sites.append({
            "id": rid("site", i),
            "name": SITE_SUFFIXES[i % len(SITE_SUFFIXES)],
            "kind": kind,
            "campus": campus,
            "region": region,
        })
    return sites
