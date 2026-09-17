"""The embedded engine must be new enough that engine notes 10 and 11 do not apply.

`pyproject.toml` asks for `samyama>=1.7.1`, but a floor in the metadata is not a
check on what is actually imported: an editable install made before the floor
moved, a stale virtualenv or a resolver pinned elsewhere all leave an older
extension on `sys.path` while the metadata reads correctly.

That matters more here than a version guard usually does, because of *how* the
old build was wrong. Note 10 raises `Variable not found` -- loud, and no test
could miss it. **Note 11 does not raise.** On 0.6.x `sum(CASE ... THEN 1 ELSE 0
END)` returns a float, so `EA04`'s `WHERE int8_hits > 0` compares int to float,
silently matches nothing, and the filter is dropped: the query returns confident
extra rows rather than an error. A downgrade would therefore reintroduce a wrong
answer, not a failure, and `test_ea04_quantization_unlock_is_not_a_cartesian_product`
would be the only thing standing between that and a published number.

So this asserts the floor twice over -- the declared dependency and the running
engine -- and, rather than trusting the version string alone, re-runs note 11's
minimal reproduction. A build that reports 1.7.1 but still drops the predicate
fails here.

The engine is acquired through a **fixture**, so an absent or broken `samyama`
skips in the setup phase. The repo-root `conftest.py` converts only setup-phase
skips to failures under `--no-skips`; skipping from the test body would let a CI
job with no engine stay green against a module whose whole promise is that a
downgrade fails loudly.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

try:                                   # tomllib is 3.11+; pyproject allows 3.10
    import tomllib
except ModuleNotFoundError:            # pragma: no cover - depends on interpreter
    # `tomli` is a declared dev dependency under 3.11, so this import is the
    # fallback rather than a hope. The third branch that set `tomllib = None`
    # is gone with the `skipif` it fed: there is no supported interpreter where
    # neither reader exists, and a sentinel for an unreachable case is a branch
    # nobody can test.
    import tomli as tomllib

MIN_ENGINE = (1, 7, 1)
ROOT = Path(__file__).resolve().parents[1]
GRAPH = "default"


@dataclasses.dataclass(frozen=True)
class Version:
    """A version and whether it is a pre-release, which the floor cares about.

    `1.7.1rc1` parses to the same numbers as `1.7.1` and is *not* the release
    that fixed notes 10 and 11, so the two must not compare equal. Ordering puts
    a pre-release below its own release and above the previous one.

    A **post**-release is the opposite case and was getting the opposite of the
    right answer: `1.7.1.post1` is *newer* than `1.7.1`, so treating every
    non-numeric suffix as a pre-release rejected a build that satisfies the
    floor. Only the pre-release spellings demote.
    """
    parts: tuple[int, ...]
    prerelease: bool

    def satisfies(self, floor: tuple[int, ...]) -> bool:
        if self.parts != floor:
            return self.parts > floor
        return not self.prerelease


# The spellings PEP 440 gives a pre-release. Anything else trailing a number --
# `.post1`, `+local` -- is at or above the release, never below it.
_PRERELEASE = re.compile(r"(a|b|c|rc|alpha|beta|pre|preview|dev)\d*$", re.IGNORECASE)


def _parse(version: str) -> Version:
    """`1.7.1` -> (1, 7, 1); `1.7.1rc1` -> the same numbers, flagged pre-release.

    `1.7.1.post1` and `1.7.1+local` parse to (1, 7, 1) and are **not** flagged:
    a post-release is newer than its release, so demoting it would reject a
    build that satisfies the floor.

    PEP 440 treats `-` and `_` as separators equivalent to `.`, so `1.7.1-rc1`
    and `1.7.1_rc1` are the same version as `1.7.1.rc1`. Splitting on `.` alone
    left the whole `1-rc1` chunk to the digit scan, which read `1`, found the
    rest non-numeric only after a digit, and passed `-rc1` to a pattern
    anchored at the start -- so both spellings came back **not** pre-release
    and satisfied the floor. A release candidate of the fix is not the fix.
    """
    version = version.replace("-", ".").replace("_", ".")
    parts: list[int] = []
    prerelease = False
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            # A segment with no leading digits: `post1`, `rc1`, `dev0`. Only the
            # pre-release spellings change the ordering.
            prerelease = bool(_PRERELEASE.match(chunk))
            break
        if len(digits) != len(chunk):
            prerelease = bool(_PRERELEASE.match(chunk[len(digits):]))
        parts.append(int(digits))
        if prerelease:
            break
    return Version(tuple(parts), prerelease)


@pytest.fixture(scope="module")
def engine():
    """Skips in *setup* so `--no-skips` can see a missing engine."""
    try:
        from samyama import SamyamaClient

        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")
    # Every other embedded module resets first; this one must too, and must also
    # clean up. Without a reset it is not idempotent -- a second run in one
    # process sees eight :VGrp nodes -- the fixture creates four -- and `hits`
    # becomes 4, failing its own
    # assertion with a message that reads as a note-11 regression.
    _reset(client)
    yield client
    _reset(client)


def _reset(client):
    """Raises rather than swallowing: a failed reset would surface later as a
    bogus note-11 regression, and "the aggregate is a float again" is the last
    thing anyone should be told when the real fault was a delete that did not run.

    Two `SamyamaClient.embedded()` instances in one process are independent --
    a node created through one is invisible to the other, and one client's
    `DETACH DELETE` leaves the other's graph intact -- so this cannot wipe
    state another module relies on. That was asserted in this comment and
    nowhere else; it is now
    `test_two_embedded_clients_do_not_share_a_graph` above. It remains a
    property of the build rather than a guarantee, which is why every module
    here resets before use rather than trusting it.
    """
    client.query("MATCH (n) DETACH DELETE n", GRAPH)


def test_two_embedded_clients_do_not_share_a_graph(engine):
    """`_reset` assumes this, and it was only ever stated in a comment.

    That module-scoped fixture runs `MATCH (n) DETACH DELETE n` on an embedded
    client. If two `SamyamaClient.embedded()` instances shared one in-memory
    graph, the reset would wipe whatever another module had built -- and the
    damage would surface somewhere else entirely, as a query returning rows a
    fixture never created.

    The second client is taken **without** a reset: with one, this could not
    fail, because the reset would empty a shared graph on the way out and the
    count would read zero either way. Only the second, though -- the first
    comes from the `engine` fixture, so a missing extension skips in setup and
    `--no-skips` can still convert it. Building both inline put this one test
    outside the guarantee the module docstring claims for all of them.

    `count(n)` rather than `count(n.id)`: on a genuinely shared graph a node
    with no `id` would be invisible to the property count, and this is asking
    whether *anything* is there.
    """
    from samyama import SamyamaClient  # the fixture already proved this imports

    first = engine
    first.query('CREATE (:Indep {id:"a"})', GRAPH)
    try:
        # Only the *second* client is built inline, and that is the whole
        # reason this test avoided the fixture: it must not be reset, or a
        # shared graph would be emptied on the way out and the count below
        # would read zero either way. The first client can come from `engine`,
        # which skips in setup -- so this test no longer escapes `--no-skips`,
        # which it did while constructing both itself.
        second = SamyamaClient.embedded()
        held = second.query("MATCH (n) RETURN count(n)", GRAPH).records
        assert (held[0][0] if held else 0) == 0, (
            f"a second embedded client sees {held} from the first, so `_reset` "
            f"here would wipe another module's graph"
        )
        second.query("MATCH (n) DETACH DELETE n", GRAPH)
        survived = first.query("MATCH (n:Indep) RETURN count(n)", GRAPH).records
        assert survived and survived[0][0] == 1, (
            "the second client's DETACH DELETE removed the first client's node "
            "-- the independence `_reset`'s docstring claims does not hold"
        )
    finally:
        # `finally`, or a failing assertion above leaves `:Indep` behind for
        # every later test in this module -- and the fixture resets between
        # modules, not between tests.
        first.query("MATCH (n) DETACH DELETE n", GRAPH)


def test_parse_reads_versions_the_way_the_comparison_needs():
    """Fault injection for the comparison itself, which is the load-bearing part."""
    assert _parse("1.7.1").parts == (1, 7, 1)
    assert _parse("0.6.1").parts == (0, 6, 1)
    assert _parse("1.7.1").satisfies(MIN_ENGINE)
    assert not _parse("0.6.1").satisfies(MIN_ENGINE)
    assert _parse("1.10.0").parts > _parse("1.7.1").parts, "string compare would differ"
    assert _parse("1.8.0").satisfies(MIN_ENGINE)
    # A pre-release of the fix is not the fix.
    assert _parse("1.7.1rc1").prerelease
    assert not _parse("1.7.1rc1").satisfies(MIN_ENGINE)
    assert not _parse("1.7.1b2").satisfies(MIN_ENGINE)
    assert not _parse("1.7.1.dev0").satisfies(MIN_ENGINE)
    # PEP 440 says `-` and `_` separate the same way `.` does, so these are the
    # same version as `1.7.1rc1` and must not satisfy the floor either. Both
    # read as final before the normalisation.
    assert not _parse("1.7.1-rc1").satisfies(MIN_ENGINE)
    assert not _parse("1.7.1_rc1").satisfies(MIN_ENGINE)
    assert not _parse("1.7.1-alpha").satisfies(MIN_ENGINE)
    # `-1` is a *post* release in PEP 440, not a pre-release, and is ahead.
    assert _parse("1.7.1-1").satisfies(MIN_ENGINE)
    assert _parse("1.8.0rc1").satisfies(MIN_ENGINE), "later release, still ahead"
    # A POST-release of the fix *is* the fix, and then some.
    assert not _parse("1.7.1.post1").prerelease
    assert _parse("1.7.1.post1").satisfies(MIN_ENGINE)
    assert _parse("1.7.1+local").satisfies(MIN_ENGINE)


def _floor_of(spec: str) -> str | None:
    """The lower bound a PEP 508 requirement declares, or None if it declares none.

    Hand-parsed rather than `spec.split(">=")[1]`, which was wrong in two ways
    that would have failed a perfectly good `pyproject.toml`: a compound
    specifier (`samyama>=1.7.1,<2`) yielded the floor `"1.7.1,<2"`, and an
    environment marker (`samyama>=1.7.1 ; python_version>='3.10'`) yielded
    `"1.7.1 ; python_version"`. Both then parsed as pre-releases and failed.

    `>=` is not the only operator that sets a floor. `~=1.7.1` means
    `>=1.7.1, ==1.7.*`, and `>1.7.0` is a floor too. Recognising only `>=` would
    report "no floor declared" for a file that pins perfectly well -- so all
    are read, and the strictly-greater case is returned as-is because the
    caller only asks whether the bound is at least MIN_ENGINE.

    **`==` and `===` set a floor too**, and the earlier version missed them:
    `samyama==1.7.1` satisfies any floor at or below 1.7.1 trivially, and yet
    it was reported as "expected a `samyama>=X` floor" -- failing the file that
    pins hardest. `==1.7.*` yields `1.7`, which is the weakest thing that
    wildcard guarantees.

    A spec with only an upper bound (`samyama<2`) still returns None, which is
    correct: it declares no floor.
    """
    body = spec.split(";", 1)[0]                       # drop any marker
    # `===` before `==` before `=`; longest operator first, or `===1.7.1` reads
    # as `==` with a floor of `=1.7.1`.
    # `)` is excluded too: `samyama (>=1.7.1)` is legal PEP 508, and without it
    # the floor came back as `"1.7.1)"`. `_parse` stops at the first
    # non-digit, so the answer was right by accident -- which is worse than
    # wrong, because nothing would have surfaced it.
    match = re.search(r"(===|==|>=|~=|>)\s*([^,\s\])]+)", body)
    if not match:
        return None
    return match.group(2).removesuffix(".*")


def _name_of(spec: str) -> str:
    """`samyama[cli]>=1.7.1 ; ...` -> `samyama`. Extras are not part of the name."""
    body = spec.split(";", 1)[0].strip()
    return re.split(r"[\[<>=!~ ]", body, maxsplit=1)[0].strip().lower()


def test_requirement_parsing_survives_the_shapes_a_pyproject_may_use():
    """Fault injection: the floor check must not fail on a valid declaration.

    Every shape below is legal PEP 508 and none should be read as older than
    the floor. A test that goes red when someone adds an upper bound is a test
    people delete.
    """
    for spec in ("samyama>=1.7.1",
                 "samyama>=1.7.1,<2",
                 "samyama[cli]>=1.7.1",
                 "samyama >= 1.7.1",
                 "samyama~=1.7.1",
                 "samyama>1.7.1",
                 "samyama>=1.7.1 ; python_version>='3.10'",
                 # An exact pin is a floor. This used to assert the opposite --
                 # "a pin declares no lower bound here" -- which failed the
                 # `pyproject.toml` that pins hardest, with a message telling
                 # the author to add the floor they already had.
                 "samyama==1.7.1",
                 "samyama===1.7.1",
                 "samyama<2,==1.7.1",
                 # Legal PEP 508, and the floor read as `"1.7.1)"` until the
                 # closing paren joined the excluded characters.
                 "samyama (>=1.7.1)",
                 "samyama (>=1.7.1) ; python_version>='3.10'"):
        assert _name_of(spec) == "samyama", spec
        floor = _floor_of(spec)
        # Bound first: `_parse(None)` is a TypeError, which would report a
        # missing floor as a crash in the parser. Every spec above has one, so
        # this cannot fire today -- it fails legibly when someone adds a case
        # that does not.
        assert floor is not None, f"no floor read from {spec!r}"
        assert _parse(floor).satisfies(MIN_ENGINE), spec

    # `==1.7.*` guarantees only 1.7, which is below MIN_ENGINE -- read, and
    # correctly judged insufficient rather than missing.
    assert _floor_of("samyama==1.7.*") == "1.7"
    assert not _parse("1.7").satisfies(MIN_ENGINE)

    # No floor at all: a bare name, and an upper bound on its own.
    assert _floor_of("samyama") is None
    assert _floor_of("samyama<2") is None, "an upper bound is not a floor"
    assert not _parse(_floor_of("samyama>=0.6.0,<2")).satisfies(MIN_ENGINE)
    assert not _parse(_floor_of("samyama~=0.6.0")).satisfies(MIN_ENGINE)


def test_pyproject_floor_is_at_least_the_version_that_fixed_notes_10_and_11():
    with open(ROOT / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)
    project = pyproject.get("project", {})
    deps = project.get("dependencies")
    assert deps is not None, (
        f"pyproject.toml declares no `project.dependencies`. If samyama moved "
        f"to `optional-dependencies` this check has to follow it -- a bare "
        f"KeyError here would read as a broken test rather than as an "
        f"unpinned engine. Keys present: {sorted(project)}"
    )

    spec = next((d for d in deps if _name_of(d) == "samyama"), None)
    assert spec is not None, f"samyama is not a declared dependency: {deps}"
    floor = _floor_of(spec)
    assert floor is not None, (
        f"expected a `samyama>=X` floor, got {spec!r}. Without one, pip may "
        f"resolve 0.6.x, where engine note 11 silently drops EA04's WHERE."
    )
    assert _parse(floor).satisfies(MIN_ENGINE), (
        f"{spec!r} admits builds older than {'.'.join(map(str, MIN_ENGINE))}, "
        "where engine note 11 silently drops EA04's WHERE"
    )


def test_the_engine_actually_imported_is_new_enough(engine):
    version = engine.status().version
    assert _parse(version).satisfies(MIN_ENGINE), (
        f"the embedded engine reports {version}; engine notes 10 and 11 apply "
        f"below {'.'.join(map(str, MIN_ENGINE))}. Re-run `pip install -e '.[dev]'` "
        "-- an editable install does not rebuild the extension when the floor moves."
    )


def test_note_11_does_not_reproduce_on_the_running_engine(engine):
    """The behaviour, not the version string -- note 11 is silent when it is back.

    Cleans up in `finally`. The module fixture resets at teardown, so today the
    four `:VGrp` nodes harm nothing -- but this is the last test in the file by
    position only, and a test added after it would inherit them. Being last is
    not a property worth depending on.
    """
    try:
        engine.query(
            'CREATE (:VGrp {id: "v1", k: "A"}), (:VGrp {id: "v2", k: "A"}), '
            '(:VGrp {id: "v3", k: "B"}), (:VGrp {id: "v4", k: "B"})',
            GRAPH,
        )
        cypher = (
            'MATCH (g:VGrp) '
            'WITH g.k AS k, sum(CASE WHEN g.k = "A" THEN 1 ELSE 0 END) AS hits '
            'WHERE hits > 0 '
            'RETURN k, hits ORDER BY k'
        )
        rows = engine.query(cypher, GRAPH).records
        assert [tuple(r) for r in rows] == [("A", 2)], (
            f"`WHERE hits > 0` did not filter group B out: {rows}. That is engine "
            "note 11 -- the aggregate is a float again and the predicate is dropped."
        )
        # `("A", 2.0) == ("A", 2)` in Python, so the assertion above passes on a
        # build that returns a float *and* filters correctly. The dropped WHERE is
        # the harm, but the float type is the mechanism, and this module's argument
        # is that behaviour beats a version string -- so pin the mechanism too.
        # `not isinstance(..., bool)`: `True` is an `int` in Python, and an engine
        # returning a boolean for a `sum()` would pass an isinstance check while
        # being exactly the type confusion note 11 is about.
        assert isinstance(rows[0][1], int) and not isinstance(rows[0][1], bool), (
            f"the aggregate came back as {type(rows[0][1]).__name__}, not int. That "
            "is note 11's mechanism; the predicate happens to have been applied "
            "here, but EA04's `WHERE int8_hits > 0` is one literal away from silence."
        )
    finally:
        _reset(engine)
