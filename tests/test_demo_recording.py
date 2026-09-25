"""The recorded demo shows figures, and figures go stale (#30).

`demo/edgeai-questions.gif` is the first thing a visitor sees and it shows real
query output -- 16 questions with their answers. Those answers are numbers. When
the generator changes or upstream publishes, the recording keeps showing the old
ones and nothing says so.

It is stale right now. The recording is from **2026-08-14** and shows:

    Graph already loaded: 25,145 nodes      (the graph now holds 25,150)
    real  onnxruntime  734                  (ONNX Runtime now registers 738)

Both moved for legitimate reasons -- 25,145 was corrected in #17, and 734 became
738 when upstream published -- which is exactly why a recording needs a
freshness signal rather than a promise to remember.

## How this is checked

The figures are read out of `demo/edgeai-questions.cast`, the asciinema
recording the GIF is rendered from. It is line-delimited JSON holding the real
terminal output, so the numbers a viewer sees can be extracted rather than
transcribed.

Two things are asserted:

1. **The README declares when the recording was made**, so a reader can judge
   its age without running anything. That is #30's second acceptance option and
   it holds regardless of drift.
2. **The figures match a fresh build** -- marked `xfail(strict=False)` because
   they do not today. Re-recording makes these XPASS, which is the signal to
   drop the mark, the same way `tests/test_correctness.py` handles engine note
   10. A hard failure would make `pytest` red on a checkout whose only fault is
   that nobody has re-recorded, and a suite that is red by default is one people
   stop reading.

Needs `data/` for the comparison; the README check needs nothing.
"""
import datetime
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CAST = ROOT / "demo" / "edgeai-questions.cast"
README = ROOT / "README.md"
DEMO_README = ROOT / "demo" / "README.md"

STALE_REASON = (
    "the recording predates the current build; re-record with "
    "scripts/record_gif.sh, then remove this mark (#30)"
)


def cast_text() -> str:
    """The terminal output a viewer sees, with escape sequences removed."""
    chunks = []
    for line in CAST.read_text(encoding="utf-8").splitlines()[1:]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if len(event) >= 3 and event[1] == "o":
            chunks.append(event[2])
    text = "".join(chunks)
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    return re.sub(r"\x1b\][^\x07]*\x07", "", text)


def cast_recorded_at() -> datetime.datetime:
    """The `timestamp` in the cast header -- when the recording was actually made."""
    header = json.loads(CAST.read_text(encoding="utf-8").splitlines()[0])
    assert "timestamp" in header, "the asciinema header carries no timestamp"
    return datetime.datetime.fromtimestamp(header["timestamp"], datetime.timezone.utc)


def figure(pattern: str) -> int:
    match = re.search(pattern, cast_text())
    assert match, (
        f"no figure matching {pattern!r} in the recording. If the demo's output "
        f"was reworded, update this pattern -- do not delete the check."
    )
    return int(match.group(1).replace(",", ""))


@pytest.fixture(scope="module")
def fresh():
    try:
        from etl import generate as gen
        from etl import onnx_catalog as oc
        from etl import real_layer
        ops = oc.load_cached()
        fleet = gen.generate(seed=gen.DEFAULT_SEED, scale=1.0, operators=ops)
        generated_implements = sum(
            1 for _s, _si, rel, _t, _ti, _p in fleet.edges if rel == "IMPLEMENTS")
        real_layer.build_real(fleet, ops)
        real_implements = sum(
            1 for _s, _si, rel, _t, _ti, _p in fleet.edges
            if rel == "IMPLEMENTS") - generated_implements
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    return {"nodes": fleet.node_count, "real_implements": real_implements}


def test_the_readme_says_when_the_recording_was_made():
    """#30's second acceptance option, and the one that holds even when stale.

    A reader cannot tell a three-week-old recording from a three-month-old one
    by looking at it. The date has to be written next to the image.
    """
    recorded = cast_recorded_at().strftime("%Y-%m-%d")
    text = README.read_text(encoding="utf-8")
    assert recorded in text, (
        f"README.md does not state when the demo was recorded. The cast header "
        f"says {recorded}; put that next to the image so a reader can judge its "
        f"age. Re-recording means updating both."
    )


def test_the_declared_date_matches_the_recording():
    """The declaration cannot drift from the thing it describes.

    Without this, re-recording and forgetting to update the README leaves a
    date that is confidently wrong -- worse than none.
    """
    text = README.read_text(encoding="utf-8")
    claimed = re.search(r"[Rr]ecorded\s+(\d{4}-\d{2}-\d{2})", text)
    assert claimed, (
        "README.md has no `recorded YYYY-MM-DD` next to the demo image"
    )
    actual = cast_recorded_at().strftime("%Y-%m-%d")
    assert claimed.group(1) == actual, (
        f"README says the demo was recorded {claimed.group(1)}, but "
        f"demo/edgeai-questions.cast was recorded {actual}"
    )


@pytest.mark.xfail(reason=STALE_REASON, strict=False)
def test_the_recording_shows_the_current_node_count(fresh):
    """`Graph already loaded: N nodes`, against a fresh build.

    XPASS here means someone re-recorded and the mark should come off.
    """
    shown = figure(r"Graph already loaded: ([\d,]+) nodes")
    assert shown == fresh["nodes"], (
        f"the recording shows {shown:,} nodes; the graph now holds "
        f"{fresh['nodes']:,}. Re-record with scripts/record_gif.sh."
    )


@pytest.mark.xfail(reason=STALE_REASON, strict=False)
def test_the_recording_shows_the_current_real_kernel_count(fresh):
    """EA16's `real onnxruntime N`, which moves when upstream publishes."""
    shown = figure(r"real\s+onnxruntime\s+([\d,]+)")
    assert shown == fresh["real_implements"], (
        f"the recording shows {shown:,} real ONNX Runtime kernels; the build "
        f"now has {fresh['real_implements']:,}. Re-record with "
        f"scripts/record_gif.sh."
    )


# Queries added to the catalog *after* the recording was made. A recording
# cannot show a query that did not exist when it was recorded, and re-recording
# needs `asciinema` and `agg` (#30 owns that).
#
# This is not a general excuse list. Each entry is a query the recording
# predates, and the README caption is worded to match -- it counts rather than
# saying "all". The exact numbers are derived from this set and the catalog
# size, and are asserted by `test_the_readme_caption_does_not_overclaim`, which
# is why they are not repeated here: quoting them went stale the moment `EA18`
# landed and the caption moved from "16 of the 17" to "16 of the 18".
ADDED_AFTER_THE_RECORDING = {"EA17",   # #35, added 2026-09-09
                             "EA18",   # #37, added 2026-09-15
                             "EA19",   # #40, added 2026-09-17
                             "EA20",   # #34, added 2026-09-23
                             "EA21"}   # #36, added 2026-09-23


def test_the_recording_still_covers_every_catalog_query():
    """Structure, not figures -- so this one is asserted rather than excused.

    A recording that silently dropped a query it *did* record would keep the
    caption while making it false, and no figure comparison would notice.
    """
    from benchmarks.queries import BY_ID
    text = cast_text()
    missing = [qid for qid in BY_ID
               if qid not in text and qid not in ADDED_AFTER_THE_RECORDING]
    assert not missing, (
        f"the recording does not show {missing}, and they are not listed as "
        f"post-dating it. Either re-record with scripts/record_gif.sh, or add "
        f"them to ADDED_AFTER_THE_RECORDING with the issue that introduced them."
    )


def test_every_query_said_to_postdate_the_recording_really_is_absent():
    """`ADDED_AFTER_THE_RECORDING` cannot outlive a re-record.

    Otherwise the set only ever grows: someone re-records, the entry stays, and
    the next query that quietly vanishes from the recording is excused by it.
    """
    from benchmarks.queries import BY_ID
    text = cast_text()
    stale = sorted(q for q in ADDED_AFTER_THE_RECORDING if q in text)
    assert not stale, (
        f"{stale} are in the recording, so they no longer post-date it. Remove "
        f"them from ADDED_AFTER_THE_RECORDING and update the README caption."
    )
    unknown = sorted(q for q in ADDED_AFTER_THE_RECORDING if q not in BY_ID)
    assert not unknown, f"{unknown} are not catalog queries at all"


def test_the_readme_caption_does_not_overclaim():
    """The caption must not say "all" while a query is missing from the recording.

    The caption is the claim a reader actually sees; the recording is the
    evidence. This is the pair that goes wrong silently -- a query is added, the
    caption keeps saying "all", and nothing on the page is false-looking.
    """
    from benchmarks.queries import BY_ID
    # The caption line itself, not the whole file. A substring search anywhere
    # in the README could be satisfied by an unrelated sentence -- the snapshot
    # section carries its own "16 of the 17", about a different claim -- so this
    # would pass while the caption under the GIF said something else entirely.
    lines = [line for line in README.read_text(encoding="utf-8").splitlines()
             if "[catalog queries](benchmarks/queries.py)" in line]
    assert len(lines) == 1, (
        f"expected exactly one line linking the catalog from the GIF caption, "
        f"found {len(lines)}. This test reads that line; if the caption moved, "
        f"point it at the new one."
    )
    caption = lines[0]
    shown = len(BY_ID) - len(ADDED_AFTER_THE_RECORDING)
    if ADDED_AFTER_THE_RECORDING:
        assert f"{shown} of the {len(BY_ID)} [catalog queries]" in caption, (
            f"the recording shows {shown} of {len(BY_ID)} catalog queries, so the "
            f"README caption must say so. Expected "
            f"'{shown} of the {len(BY_ID)} [catalog queries]'."
        )
    else:
        # The symmetric half. Without it, a re-record empties the set and the
        # caption keeps under-claiming forever -- the same silent drift in the
        # other direction, and nobody notices an understatement.
        assert f"All {len(BY_ID)} [catalog queries]" in caption, (
            f"nothing post-dates the recording any more, so the caption should "
            f"be back to 'All {len(BY_ID)} [catalog queries]'. It reads as though "
            f"the recording is still short."
        )


def test_the_demo_readme_alt_text_states_the_recorded_count():
    """`demo/README.md`'s alt text carries a count, so it drifts like a caption.

    It claims to be checked -- the page says this test checks it -- and until
    now nothing did, which is the worse half of the pair: a reader is told the
    number is guarded, and a re-record or a new query silently makes it false.
    Alt text is also the copy a screen reader announces, so it is not decorative
    prose that can be left to rot.
    """
    from benchmarks.queries import BY_ID
    lines = [line for line in DEMO_README.read_text(encoding="utf-8").splitlines()
             if line.startswith("![") and "edgeai-questions.gif" in line]
    assert len(lines) == 1, (
        f"expected exactly one image line for the recording in demo/README.md, "
        f"found {len(lines)}. This test reads that line; if it moved, point "
        f"this at the new one.")
    shown = len(BY_ID) - len(ADDED_AFTER_THE_RECORDING)
    assert f"{shown} questions answered" in lines[0], (
        f"the recording shows {shown} of {len(BY_ID)} catalog queries, so the "
        f"alt text must say '{shown} questions answered'. It reads:\n"
        f"  {lines[0]}")
