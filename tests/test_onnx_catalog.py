"""The ONNX catalog is the only real data in this KG -- pin its parsing."""
import json

import pytest

from etl import onnx_catalog as oc


@pytest.fixture(scope="module")
def ops():
    try:
        return oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


def test_catalog_is_populated(ops):
    assert len(ops) > 150, "ONNX has ~200 operators; parsing likely broke"


def test_core_operators_present(ops):
    names = {o.name for o in ops}
    for expected in ("Conv", "MatMul", "LSTM", "Softmax", "QuantizeLinear", "Gemm"):
        assert expected in names, f"{expected} missing from parsed catalog"


def test_ids_unique(ops):
    ids = [o.id for o in ops]
    assert len(ids) == len(set(ids))


def test_categories_are_sane(ops):
    by_name = {o.name: o for o in ops}
    assert by_name["Conv"].category == "convolution"
    assert by_name["MatMul"].category == "matmul"
    assert by_name["LSTM"].category == "recurrent"
    assert by_name["Softmax"].category == "activation"
    assert by_name["QuantizeLinear"].category == "quantization"


def test_control_flow_flagged(ops):
    by_name = {o.name: o for o in ops}
    assert by_name["Loop"].is_control_flow is True
    assert by_name["Conv"].is_control_flow is False


def test_since_version_is_the_latest(ops):
    # Conv has been revised many times; the parser should keep the highest opset.
    conv = next(o for o in ops if o.name == "Conv")
    assert conv.since_version >= 11
    assert conv.version_count >= 2


def test_categorize_falls_back_to_tensor():
    assert oc.categorize("SomeOperatorThatDoesNotExist") == "tensor"


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


def _cache(tmp_path, source: str | None):
    """A cached `Operators.md`, with or without a recorded source."""
    (tmp_path / "Operators.md").write_text("cached-content", encoding="utf-8")
    if source is not None:
        (tmp_path / "operators.json").write_text(
            json.dumps({"source": source}), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("source", "refetch", "why"),
    [
        (None, True, ("unknown provenance is not evidence of a match: a raw "
                      "file with no recorded source could be any revision, "
                      "and one fetch is cheaper than trusting it")),
        ("PINNED", False, "the cache already holds what the pin names"),
        ("https://raw.githubusercontent.com/onnx/onnx/main/docs/Operators.md", True,
         "a cache fetched before the pin holds a revision ONNX_REF no longer names"),
    ],
)
def test_a_cache_from_another_revision_does_not_outlive_the_pin(
        tmp_path, monkeypatch, source, refetch, why):
    """`ONNX_REF` must govern what a build reads, not only what it would fetch.

    `download()` returning any existing cache is what let a pinned source and
    an unpinned build coexist: every published figure would still be a
    function of when someone last downloaded, which is the drift this pin
    exists to stop. Measured per case rather than reasoned about, because the
    failure is silent -- the wrong catalogue parses perfectly well.
    """
    monkeypatch.setattr(oc, "ONNX_DIR", _cache(
        tmp_path, oc.OPERATORS_URL if source == "PINNED" else source))
    get = _Recorded()
    monkeypatch.setattr("requests.get", get)

    oc.download()

    assert bool(get.urls) is refetch, why
    assert (tmp_path / "Operators.md").read_text(encoding="utf-8") == (
        "fresh-content" if refetch else "cached-content")
    if refetch:
        assert get.urls == [oc.OPERATORS_URL], "a re-fetch must use the pinned URL"
