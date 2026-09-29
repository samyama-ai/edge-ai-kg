"""Every upstream this KG reads is pinned, and no cache outlives its pin.

`etl/onnx_catalog.py` was pinned first (#120) and carries its own tests. The
other two were left tracking `main`, and on 2026-09-25 that cost a red CI on
`main`: ONNX Runtime merged kernel registrations, 738 became 743, the graph
gained 8 nodes, and four documents publishing 25,162 were describing a graph
nobody could rebuild. The suite passed on a cached `data/` and failed on a
fresh one, which is the shape of failure CI exists to catch.

So this file asserts the property rather than the incident: **each source
module names a revision, and builds from it**. A URL tracking a branch makes
every published figure a function of when someone last downloaded.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

from etl import mlperf_tiny as mt
from etl import onnx_catalog as oc
from etl import ort_kernels as ok

PINNED = [
    pytest.param(oc, "ONNX_REF", oc.OPERATORS_URL, id="onnx-catalogue"),
    pytest.param(ok, "ORT_REF", ok.KERNELS_URL, id="onnxruntime-kernels"),
    pytest.param(mt, "TINY_REF", mt.SUMMARY_URL, id="mlperf-tiny"),
]


@pytest.mark.parametrize(("module", "ref_name", "url"), PINNED)
def test_the_source_url_names_a_revision_not_a_branch(module, ref_name, url):
    """A 40-character sha in the URL, not `main`.

    Checked as a property of the URL rather than by reading the constant,
    because the constant existing proves nothing if the URL does not use it --
    which is exactly how `ORT_REF` could have been added and still fetched
    from `main`.
    """
    ref = getattr(module, ref_name)
    assert re.fullmatch(r"[0-9a-f]{40}", ref), (
        f"{ref_name} is {ref!r}; a pin is a full commit sha, so that moving it "
        f"is a reviewed change rather than whatever upstream merged today")
    assert ref in url, f"{ref_name} exists but {url} does not use it"
    assert "/main/" not in url, (
        f"{url} still tracks a branch, so the graph changes shape whenever "
        f"upstream merges and every published figure stops matching")


class _Recorded:
    """A `requests.get` stand-in that records its calls and answers offline."""

    text = "fresh-content"

    def __init__(self):
        self.urls: list[str] = []

    def __call__(self, url, timeout=0):
        self.urls.append(url)
        return self

    def raise_for_status(self):
        pass


@pytest.mark.parametrize(
    ("module", "dir_name", "raw_name", "parsed_name", "url_name"),
    [
        (ok, "ORT_DIR", "OperatorKernels.md", "kernels.json", "KERNELS_URL"),
        (mt, "TINY_DIR", "summary.csv", "results.json", "SUMMARY_URL"),
    ],
    ids=["onnxruntime-kernels", "mlperf-tiny"],
)
@pytest.mark.parametrize(
    ("recorded", "refetch", "why"),
    [
        (None, True, "no recorded source means unknown provenance; re-fetch is the safe answer"),
        ("PINNED", False, "the cache already holds what the pin names"),
        ("UNPINNED", True, "a cache written before the pin records the old branch URL"),
    ],
)
def test_a_cache_from_another_revision_does_not_outlive_the_pin(
        tmp_path, monkeypatch, module, dir_name, raw_name, parsed_name,
        url_name, recorded, refetch, why):
    """The pin has to govern what a build *reads*, not only what it would fetch.

    `download()` returning any existing file is what lets a pinned source and
    an unpinned build coexist. The failure is silent -- an older revision
    parses perfectly well -- so each case is measured rather than argued.
    """
    url = getattr(module, url_name)
    (tmp_path / raw_name).write_text("cached-content", encoding="utf-8")
    if recorded is not None:
        source = url if recorded == "PINNED" else url.replace(
            getattr(module, {"ORT_DIR": "ORT_REF", "TINY_DIR": "TINY_REF"}[dir_name]), "main")
        (tmp_path / parsed_name).write_text(
            json.dumps({"source": source}), encoding="utf-8")

    monkeypatch.setattr(module, dir_name, tmp_path)
    get = _Recorded()
    monkeypatch.setattr("requests.get", get)

    module.download()

    assert bool(get.urls) is refetch, why
    assert (tmp_path / raw_name).read_text(encoding="utf-8") == (
        "fresh-content" if refetch else "cached-content")
    if refetch:
        assert get.urls == [url], "a re-fetch must use the pinned URL"


def test_the_engine_declares_a_ceiling_as_well_as_a_floor():
    """The engine is an upstream too, and it was the last one left unbounded.

    `samyama>=1.7.1` let CI install whatever PyPI had that morning while every
    developer kept 1.7.1. On 2026-09-29 that was 1.9.0, whose planner refuses
    an unbounded variable-length pattern producing over a million paths:

        Query error: [Samyama.ClientError.Statement.PlanningFailed] Planning
        error: variable-length pattern produced more than 1000000 paths

    `EA17` and `EA21` both walk `NEXT_STAGE*0..` unbounded over a stage graph
    with cycles, so they do not run at all. Measured: 11 tests fail on 1.9.0
    and none on 1.7.1 -- a green machine and a red CI, from the same commit.

    The ceiling is a stopgap for reproducibility, not a fix; bounding those
    walks changes what they answer. This test exists so removing it is a
    decision someone makes on purpose, with the two queries in hand.
    """
    import tomllib

    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as handle:
        deps = tomllib.load(handle)["project"]["dependencies"]
    spec = next((d for d in deps if d.replace(" ", "").startswith("samyama")), None)

    assert spec, "samyama is no longer a declared dependency"
    assert "<" in spec, (
        f"{spec!r} has no upper bound, so a fresh install takes whatever PyPI "
        f"serves today. That is how CI came to test a different engine than "
        f"anyone had locally. If the walks in EA17/EA21 are now bounded, raise "
        f"the ceiling deliberately and say so here.")
