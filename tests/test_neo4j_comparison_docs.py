"""The two pages that quote the Neo4j comparison must quote the same run (#47).

`docs/why-this-engine.md` summarises `docs/neo4j-comparison.md`. The summary
kept an older run's numbers after the comparison page was re-measured: a
different tally, a different `EA01` ratio, and two "Neo4j wins" the page had
already withdrawn as unstable. Nothing compared the two, so review caught it.
These read both files and fail on the drift. No engine needed.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from benchmarks.sweep import PARITY_BAND

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
TALLY = re.compile(r"\*\*Samyama (\d+), Neo4j (\d+), unresolved (\d+), "
                   r"no verdict (\d+)\.\*\*")


def _read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _hero_section() -> str:
    """The summary's hero-query section, by its two headings.

    A heading edited on its own is a broken test, not a broken page, so it
    fails with the reason rather than a bare ValueError from `index`.
    """
    summary = _read("why-this-engine.md")
    start, end = "### The hero query", "### Operations and maturity"
    for heading in (start, end):
        if heading not in summary:
            pytest.fail(f"why-this-engine.md no longer has a {heading!r} heading; "
                        f"this test locates the section by it, so update both.")
    return summary[summary.index(start):summary.index(end)]


def test_both_pages_state_the_same_tally():
    comparison = TALLY.findall(_read("neo4j-comparison.md"))
    summary = TALLY.findall(_read("why-this-engine.md"))
    assert len(comparison) == 1, f"expected one tally on the comparison page: {comparison}"
    assert summary == comparison, (
        f"why-this-engine.md says {summary}, neo4j-comparison.md says "
        f"{comparison}. Copy the summary from the comparison page."
    )


def test_the_hero_query_figures_match():
    """`EA01`'s row on the comparison page, quoted in the summary."""
    row = re.search(r"^\| \*\*`EA01`\*\* \|[^|]*\| ([\d.]+) / ([\d.]+) ms \| "
                    r"\*\*([\d.]+) / ([\d.]+) ms\*\* \| \*\*neo4j ([\d.]+-[\d.]+)x\*\* \|$",
                    _read("neo4j-comparison.md"), re.MULTILINE)
    assert row, "the EA01 row of 'Where Neo4j wins' changed shape; update this test"
    sam1, sam2, neo1, neo2, ratio = row.groups()
    section = _hero_section()
    for figure in (sam1, sam2, neo1, neo2, f"{ratio}x"):
        assert figure in section, (
            f"{figure} from neo4j-comparison.md's EA01 row is not in "
            f"why-this-engine.md's hero-query section"
        )


def test_the_summary_does_not_credit_withdrawn_wins():
    """`EA02` and `EA11` are no-verdict on the comparison page, not Neo4j wins."""
    assert re.search(r"`EA02`, `EA08`, `EA11` \|.*no verdict",
                     _read("neo4j-comparison.md")), (
        "the comparison page no longer lists EA02/EA08/EA11 as no-verdict; "
        "update this test with it"
    )
    section = _hero_section()
    for qid in ("EA02", "EA11"):
        credited = re.search(rf"(takes|wins?|goes to|to Neo4j)\b[^.]*`{qid}`"
                             rf"|`{qid}`[^.]*\b(goes to|to) Neo4j", section)
        assert not credited, (
            f"why-this-engine.md still credits {qid} to Neo4j: {credited[0]!r}")


# `| `EA14` | 0.4 / 0.4 ms | 8.3 / 7.8 ms | 18.99x / 17.37x | samyama |`
ROW = re.compile(r"^\| `(EA\d\d)` \| ([\d.]+) / ([\d.]+) ms \| ([\d.]+) / ([\d.]+) ms "
                 r"\| ([\d.]+)x / ([\d.]+)x \| (.+?) \|$", re.MULTILINE)
NO_VERDICT = re.compile(r"^\| (`EA\d\d`(?:, `EA\d\d`)*) \|.*\| no verdict", re.MULTILINE)
# Imported, not restated: a widened band in the sweep has to move the verdicts
# this re-derives, or the page and the tool disagree about what parity is.
BAND = 1 + PARITY_BAND


def _timed_rows():
    rows = ROW.findall(_read("neo4j-comparison.md"))
    assert len(rows) >= 10, f"the results table changed shape: {len(rows)} rows parsed"
    return rows


# The four things a verdict cell can mean. Ordered most-specific-first, because
# the cells are prose: EA04 reads "unresolved -- samyama, then parity", and a
# scan in any other order calls it a win.
VERDICTS = ("unresolved", "parity", "neo4j", "samyama")


def _verdict_key(qid: str, cell: str) -> str:
    """The one reading of a verdict cell, shared by every test below.

    One helper rather than three inline scans: the tally test and the
    ratio test disagreeing about what `EA04`'s cell means is a way for both
    to pass while the page is wrong, and that is the failure this file exists
    to prevent.
    """
    key = next((k for k in VERDICTS if k in cell.lower()), None)
    assert key, (f"{qid}'s verdict {cell!r} is none of {list(VERDICTS)}; "
                 f"the tally line cannot count it, so name it here first")
    return key


def _expected_verdict(r1: float, r2: float) -> str:
    def one(r):
        return "samyama" if r > BAND else "neo4j" if r < 1 / BAND else "parity"
    a, b = one(r1), one(r2)
    return a if a == b else "unresolved"


def test_each_verdict_follows_from_its_two_ratios():
    """A verdict edited on its own, with the ratios left alone, is caught here.

    Compared as the cell's *key*, not as a substring of its prose: "samyama"
    is a substring of "unresolved -- samyama, then parity", so a `in` test
    passed a row whose verdict had been changed to the wrong one.
    """
    for qid, *_ms, r1, r2, verdict in _timed_rows():
        want = _expected_verdict(float(r1), float(r2))
        assert _verdict_key(qid, verdict) == want, (
            f"{qid}: ratios {r1}x / {r2}x give {want!r}, the table says {verdict!r}")


def test_each_ratio_is_consistent_with_its_milliseconds():
    """Displayed ms are rounded to 0.1 and ratios to 0.01; allow exactly that."""
    for qid, s1, s2, n1, n2, r1, r2, _v in _timed_rows():
        for sam, neo, ratio in ((s1, n1, r1), (s2, n2, r2)):
            sam, neo, ratio = float(sam), float(neo), float(ratio)
            lo = (neo - 0.05) / (sam + 0.05)
            hi = (neo + 0.05) / max(sam - 0.05, 1e-9)
            assert lo - 0.005 <= ratio <= hi + 0.005, (
                f"{qid}: {neo} / {sam} ms cannot give {ratio}x (range {lo:.2f}-{hi:.2f})")


def test_the_tally_is_the_table_counted():
    """The bold tally line, re-derived from the verdict column.

    `parity` has no column in the tally line. No row is in parity today, and
    one that appeared would have to be given a place there rather than
    silently dropped -- so it fails here by name instead of raising
    `StopIteration` out of the generator.
    """
    page = _read("neo4j-comparison.md")
    counted = dict.fromkeys(VERDICTS, 0)
    for qid, *_cells, verdict in _timed_rows():
        counted[_verdict_key(qid, verdict)] += 1
    assert not counted["parity"], (
        f"{counted['parity']} row(s) are in parity, and the tally line has no "
        f"parity column. Add one there and here.")
    no_verdict = sum(len(m.split(",")) for m in NO_VERDICT.findall(page))
    (tally,) = TALLY.findall(page)
    assert tuple(map(int, tally)) == (counted["samyama"], counted["neo4j"],
                                      counted["unresolved"], no_verdict), (
        f"the tally line says {tally}; the table counts to {counted}, "
        f"no verdict {no_verdict}")


def test_the_summarys_count_of_cheap_wins_is_the_table_counted():
    """"Six of our eight wins ... under three milliseconds", re-derived."""
    words = {w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split())}

    def count(word):
        """The prose spells these out, but a digit is not a reason to fail."""
        if word.isdigit():
            return int(word)
        assert word.lower() in words, (
            f"{word!r} in the cheap-wins sentence is not a number this test "
            f"can read; spell it out or use a digit")
        return words[word.lower()]
    claim = re.search(r"([\w\d]+) of our ([\w\d]+) wins are queries we answer in under "
                      r"(\w+)\s+milliseconds", _hero_section())
    assert claim, "the cheap-wins sentence changed shape; update this test"
    cheap, wins, limit = (count(w) for w in claim.groups())
    table_wins = [r for r in _timed_rows() if _verdict_key(r[0], r[-1]) == "samyama"]
    assert wins == len(table_wins), f"summary says {wins} wins, table has {len(table_wins)}"
    under = [r[0] for r in table_wins if max(float(r[1]), float(r[2])) < limit]
    assert cheap == len(under), f"summary says {cheap} under {limit} ms; table: {under}"
