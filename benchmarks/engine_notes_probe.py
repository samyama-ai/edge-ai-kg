"""Re-run the probeable engine notes against whatever engine is installed.

Eleven probes: notes 1-6, 8 and 9, plus 3b, 4b and 8b -- behaviours the notes
file lists under 3, 4 and 8 that carry CLAUDE.md rules of their own. **Note 7
has no probe**: "the `--graph` argument is
ignored" is a property of the OSS HTTP path, and there is no tenant boundary to
ignore on an embedded build, so there is nothing here to measure. It stands
un-re-measured, and the output says so.

`docs/engine-notes.md` records eleven engine behaviours, nine of which the
loader and the query catalog work around. Those nine were measured on the
**1.7.0 HTTP server** in August 2026 and written up in prose. Prose does not
re-measure itself: #56 pinned the embedded build to `samyama` 1.7.1 and the PR
initially asserted "notes 1-9 are unaffected -- they still reproduce", which was
never checked and turned out to be false for every one of them.

This module is the check that assertion should have been. Each probe is the
note's own minimal reproduction, and reports STILL REPRODUCES or FIXED against
the running engine rather than against a remembered result.

    python -m benchmarks.engine_notes_probe            # small fixtures
    python -m benchmarks.engine_notes_probe --scale 300  # note 1 at real cardinality

## Two limits worth stating before anyone acts on the output

**This runs embedded.** Notes 1-9 were measured against the HTTP server, so a
FIXED here says the *embedded 1.7.1* build no longer shows the behaviour. It
does not say the 1.7.0 server was fixed -- that is a different binary, and #56
is the standing reminder that assuming two builds agree is how this repo got
into trouble in the first place.

**Note 1 needs `--scale`.** Its own text warns that "a passing 6-node
reproduction proves nothing", because the join bug appears once cardinalities
are real. So without `--scale` note 1 is not run at all and is reported
INCONCLUSIVE. With it, the CLI refuses anything under `MIN_SCALE` (10), where
the wrong and right answers are too close to tell apart; `note_1` itself also
returns UNSOUND below 2, but only as a guard for callers that skip the CLI.
"""
from __future__ import annotations

import click

from benchmarks.engine_notes_cases import PROBES
from benchmarks.engine_notes_harness import FixtureNotBuilt, ResetFailed, verdict_for

# Below this, note 1's cartesian product is not distinguishable from its
# correct answer: at n models the wrong answer is n*n rows and the right one
# is n, which coincide at 1 and are adjacent at 2.
MIN_SCALE = 10


def _engine():
    from samyama import SamyamaClient

    return SamyamaClient.embedded()


@click.command()
@click.option("--scale", default=0, type=click.IntRange(min=0),
              help="Models on one board for note 1. Its own text says a small "
                   "reproduction proves nothing, so note 1 is INCONCLUSIVE "
                   "without this, and anything under 10 is refused: at --scale "
                   "1 the cartesian product and the correct answer are the "
                   "same one row, so it would report FIXED for a broken join.")
def main(scale):
    if scale and scale < MIN_SCALE:
        raise SystemExit(
            f"--scale {scale} cannot settle note 1: at that size the cartesian "
            f"product and the correct answer are {scale * scale} and {scale} "
            f"rows, which are the same number at 1 and adjacent at 2. Use at "
            f"least {MIN_SCALE}; the note's own write-up uses 300.")

    client = _engine()
    # Reported, not relied on: `status()` is not a documented contract and an
    # engine that changes the field name should not stop the probe from running.
    # The version is a label on the output, and losing the label is not a reason
    # to lose the measurement.
    try:
        version = client.status().version
    except Exception as exc:
        version = f"unknown ({type(exc).__name__} reading client.status())"
    click.echo(f"embedded engine: {version}\n")
    fixed, reproduces, inconclusive, unsound, errored = [], [], [], [], []
    for number, title, probe in PROBES:
        if number == 1 and not scale:
            # Skipped, not run and discarded. Building the fixture only to throw
            # the verdict away wastes the work and, worse, leaves a graph behind
            # that the next probe's reset is the only thing standing between.
            verdict, bucket, detail = "INCONCLUSIVE", inconclusive, "not run -- needs --scale"
        else:
            try:
                # `scale`, not `scale or 2`. Note 1 is the only probe that
                # reads it and it is skipped above when `scale` is 0, so the
                # fallback could never be reached -- and had it been, it would
                # have run note 1 at a size its own text calls proof of nothing.
                ok, detail = probe(client, scale)
            except (ResetFailed, FixtureNotBuilt) as exc:
                verdict, bucket, detail = "UNSOUND", unsound, str(exc)
            except Exception as exc:
                # Per-probe, so one unexpected failure does not abort the run
                # and hide the rest. A probe that cannot run is a result too
                # -- it is just not a result about the engine note, and it
                # gets its own bucket rather than borrowing UNSOUND's.
                verdict, bucket = "ERROR", errored
                detail = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}"
            else:
                verdict = verdict_for(ok)
                bucket = {"FIXED": fixed, "STILL REPRODUCES": reproduces,
                          "UNSOUND": unsound}[verdict]
        bucket.append(number)
        click.echo(f"note {number}  {verdict}")
        click.echo(f"          {title}")
        click.echo(f"          {detail}\n")

    # The denominator is what was actually measured. Counting an INCONCLUSIVE
    # note 1 among the probed would let "7 of 8" stand for a run in which the
    # single most important note was never really tested.
    measured = len(fixed) + len(reproduces)
    click.echo(f"{len(fixed)} of {measured} measured probes do not reproduce on {version}.")
    if reproduces:
        click.echo(f"still reproducing: {reproduces}")
    if unsound:
        click.echo(f"UNSOUND, the run could not tell: {unsound} -- not evidence "
                   "either way,\nand excluded from the count above.")
    if errored:
        click.echo(f"ERROR, the probe itself failed: {errored} -- excluded from the "
                   "count above;\nthe detail line under each says what raised.")
    if inconclusive:
        click.echo(f"NOT MEASURED: {inconclusive} -- note 1 needs --scale, because its "
                   "own\ntext says a small reproduction proves nothing. Re-run with "
                   "--scale 300.")
    click.echo("note 7 is not probed at all: it is a property of the OSS server's "
               "HTTP path,\nand there is no tenant boundary to ignore embedded.")
    click.echo("\nA FIXED here is about the EMBEDDED build. The notes were measured "
               "against the\n1.7.0 HTTP server, which is a different binary -- see "
               "#56 for what assuming\notherwise costs.")


if __name__ == "__main__":
    main()
