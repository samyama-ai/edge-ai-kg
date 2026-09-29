"""Fetch and parse the real ONNX operator catalog.

This is the one *real* public source in this KG. Everything else (boards, SoCs,
accelerators, kernels, models, deployments) is synthetic -- see `etl/generate.py`
and `docs/data-provenance.md`.

Source : https://github.com/onnx/onnx  -- docs/Operators.md
License: Apache-2.0
Parsed : the per-domain operator index tables at the top of the file, which give
         operator name, domain, and the opset versions the operator was revised in.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ONNX_DIR = DATA_DIR / "onnx"
# Pinned to a commit, not to `main`. Upstream edits `docs/Operators.md`
# continuously -- 20 commits touched it in the three weeks to 2026-09-16 -- and
# `etl/generate.py` builds `Kernel` rows from the parsed result, so the draw
# shifts the shared random stream: a catalogue that moves reshapes the fleet at
# an unchanged seed. Tracking `main` therefore made every published figure in
# this repo a function of *when* someone last downloaded, which is what
# `docs/build-manifest.json` had to fingerprint around.
#
# This is a pin, not a vendoring: the file still comes from upstream at build
# time, and which upstream state we track is a reviewed decision rather than an
# accident of timing.
#
# `adda7bc7e8` is the revision every figure currently published in this repo was
# measured on -- chosen for that reason rather than for being newest, so pinning
# cost no re-baselining. To move it: change `ONNX_REF`, run
# `python -m etl.download_data --force`, then `python -m etl.manifest --check`,
# and update the published figures it names. `main` HEAD on 2026-09-16 parsed to
# the same 205 operators but differed in 17 operator records, which is exactly
# the silent kind of move this pin exists to stop.
ONNX_REF = "adda7bc7e805f376c5c64ade2614133445ee8298"
OPERATORS_URL = (
    f"https://raw.githubusercontent.com/onnx/onnx/{ONNX_REF}/docs/Operators.md"
)

# Operator name -> coarse category. Used to shape which accelerators plausibly
# implement which operator, and to make the graph readable in a demo.
# Anything unmatched falls into "tensor".
#
# Three entries wrap one regex across several lines. The extra parentheses
# around those are load-bearing, not style: inside a collection literal an
# implicit concatenation is indistinguishable from a missing comma between
# two entries, which is what ISC004 exists to catch. Parenthesised, the
# intent is explicit, no suppression is needed, and a real missing comma
# anywhere in this table is still reported.
_CATEGORY_RULES: list[tuple[str, str]] = [
    (r"^(Conv|ConvTranspose|ConvInteger|QLinearConv|DeformConv|Col2Im)", "convolution"),
    (r"^(Gemm|MatMul|MatMulInteger|QLinearMatMul|Einsum)", "matmul"),
    (r"(Pool|Upsample|Resize|GridSample|AffineGrid)", "spatial"),
    (r"^(LSTM|GRU|RNN|Scan|Loop)", "recurrent"),
    (r"^(Attention|RotaryEmbedding|MultiHeadAttention)", "attention"),
    (r"(Normalization|LpNormalization|MeanVarianceNormalization)", "normalization"),
    ((r"^(Relu|LeakyRelu|PRelu|Elu|Selu|Celu|Gelu|Sigmoid|Tanh|HardSigmoid|HardSwish|"
      r"Softmax|LogSoftmax|Hardmax|Softplus|Softsign|Mish|ThresholdedRelu|Swish|Shrink)"), "activation"),
    ((r"^(Add|Sub|Mul|Div|Pow|Mod|Neg|Abs|Sqrt|Exp|Log|Sum|Mean|Max|Min|Clip|Sign|"
      r"Reciprocal|Ceil|Floor|Round|Sin|Cos|Tan|Asin|Acos|Atan|Sinh|Cosh|Tanh|"
      r"Asinh|Acosh|Atanh|Erf|Bitwise|And|Or|Xor|Not|Equal|Greater|Less)"), "elementwise"),
    (r"^Reduce", "reduction"),
    (r"^(QuantizeLinear|DequantizeLinear|DynamicQuantizeLinear|QLinear|Cast|CastLike|BitCast)", "quantization"),
    ((r"^(Reshape|Transpose|Concat|Split|Slice|Gather|Scatter|Squeeze|Unsqueeze|Pad|Tile|"
      r"Flatten|Expand|SpaceToDepth|DepthToSpace|Identity|Shape|Size|Compress|"
      r"ReverseSequence|OneHot|Range|Trilu|TopK|Sort|NonZero|Where)"), "shape"),
    (r"(RandomNormal|RandomUniform|Multinomial|Bernoulli|Dropout)", "stochastic"),
    (r"(STFT|DFT|MelWeightMatrix|BlackmanWindow|HammingWindow|HannWindow)", "signal"),
]

# Operators that a small always-on edge accelerator realistically cannot run.
# Kept explicit (not derived) so the CPU-fallback story in the demo is auditable.
CONTROL_FLOW_OPS = {"Loop", "Scan", "If", "SequenceMap"}


@dataclass(frozen=True)
class Operator:
    id: str
    name: str
    domain: str
    since_version: int
    version_count: int
    category: str
    is_control_flow: bool


# The prefix rules above are mostly right (Tanh -> activation, MaxPool ->
# spatial), but a handful of longer names get swallowed by a shorter prefix
# (Multinomial by ^Mul, Expand by ^Exp). Exact-name overrides win.
_CATEGORY_OVERRIDES: dict[str, str] = {
    "Multinomial": "stochastic",
    "Expand": "shape",
    "MaxUnpool": "spatial",
    "NegativeLogLikelihoodLoss": "loss",
    "SoftmaxCrossEntropyLoss": "loss",
}


def categorize(name: str) -> str:
    if name in _CATEGORY_OVERRIDES:
        return _CATEGORY_OVERRIDES[name]
    for pattern, cat in _CATEGORY_RULES:
        if re.search(pattern, name):
            return cat
    return "tensor"


def parse_operators(markdown: str) -> list[Operator]:
    """Parse the per-domain operator index tables from ONNX Operators.md."""
    ops: dict[str, Operator] = {}
    domain = "ai.onnx"
    row_re = re.compile(r'^\|<a href="#([A-Za-z0-9_.]+)">([^<]+)</a>\|(.*)\|$')

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("### ") and "<a name=" not in stripped:
            domain = stripped[4:].replace("(default)", "").strip()
            continue
        m = row_re.match(stripped)
        if not m:
            continue
        name = m.group(2).strip()
        versions = [int(v) for v in re.findall(r">(\d+)</a>", m.group(3))]
        if not versions:
            continue
        op = Operator(
            id=f"op:{domain}:{name}".lower(),
            name=name,
            domain=domain,
            since_version=max(versions),
            version_count=len(versions),
            category=categorize(name),
            is_control_flow=name in CONTROL_FLOW_OPS,
        )
        # First domain table wins; later duplicates are the detailed sections.
        ops.setdefault(op.id, op)
    return list(ops.values())


def cached_source() -> str | None:
    """The URL the cached catalogue was fetched from, or `None` if unknown.

    `build()` records it in `operators.json`. A checkout made before the pin
    holds `.../onnx/main/docs/Operators.md`, which is how a cache can hold a
    revision the pin no longer names.
    """
    path = ONNX_DIR / "operators.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("source")
    except (json.JSONDecodeError, OSError):
        return None


def download(force: bool = False) -> Path:
    """Fetch Operators.md into data/onnx/, unless the cache already matches the pin.

    A cache keeps whatever revision it was fetched from, so returning it
    unconditionally would let an old copy outlive a moved `ONNX_REF` -- the
    pin would be true of the source and false of every build. When the
    recorded source is not this `OPERATORS_URL`, the file is re-fetched
    rather than reused, which is what makes "a re-fetch reproduces what is
    pinned here" a fact rather than an instruction to remember `--force`.

    An unknown source -- no `operators.json` yet, or an unreadable one -- is
    treated as stale too, which is a change: it used to be reused. The raw
    file alone carries no provenance, so "it is probably the pinned one" was
    the only reason to keep it, and that is the assumption every pin in this
    repo exists to remove. It costs one fetch rather than one per call:
    `build()` writes the source back immediately after, so the next call
    reads a cache it can vouch for. The two cases it catches are an
    interrupted build and a checkout cached before `ONNX_REF` existed.

    `etl.ort_kernels` and `etl.mlperf_tiny` do exactly the same, and the
    three agreeing is the point -- a reader who learns the rule once should
    not find one source quietly kinder than the others.
    """
    import requests

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = ONNX_DIR / "Operators.md"
    source = cached_source()
    stale_pin = source != OPERATORS_URL
    if raw_path.exists() and not force and not stale_pin:
        return raw_path
    resp = requests.get(OPERATORS_URL, timeout=120)
    resp.raise_for_status()
    raw_path.write_text(resp.text, encoding="utf-8")
    return raw_path


def build(force: bool = False) -> list[Operator]:
    """Download (if needed), parse, and cache the operator catalog as JSON."""
    raw_path = download(force=force)
    ops = parse_operators(raw_path.read_text(encoding="utf-8"))
    out = ONNX_DIR / "operators.json"
    out.write_text(json.dumps({
        "source": OPERATORS_URL,
        "license": "Apache-2.0",
        "operator_count": len(ops),
        "operators": [asdict(o) for o in ops],
    }, indent=1), encoding="utf-8")
    return ops


def load_cached() -> list[Operator]:
    """Load the parsed catalog written by build()."""
    path = ONNX_DIR / "operators.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m etl.download_data` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Operator(**o) for o in payload["operators"]]


if __name__ == "__main__":
    catalog = build()
    print(f"[onnx] parsed {len(catalog)} operators -> {ONNX_DIR / 'operators.json'}")
