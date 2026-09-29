"""The generated graph's container, its id scheme, and its cache on disk.

Separated from `etl/generate.py` so that file stays under the 500 lines the
review harness reads. The division is deliberate: nothing here draws a random
number. `Fleet` collects rows, `write`/`load` move them to and from
`data/fleet/fleet.json`, and `StaleFleetCache` is what a reader gets when that
file predates a label the loader now expects -- all of it independent of how
the rows were produced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FLEET_PATH = DATA_DIR / "fleet" / "fleet.json"


# What `generate()` puts in the fleet, and therefore what a current
# `fleet.json` must contain. Not `etl.loader.NODE_LABELS`: that set also holds
# `BenchmarkTask`, which only `etl/real_layer.py` builds, so checking against
# it would reject every cache ever written.
#
# `tests/test_generate.py` compares this against a freshly generated fleet, so
# it cannot drift from the generator it describes.
GENERATED_LABELS = ("Accelerator", "Board", "Certification", "ClinicalTask",
                    "Dataset", "Deployment", "Kernel", "Model", "ModelVariant",
                    "Operator", "Runtime", "Sensor", "SignalStage", "Site",
                    "SoC", "Vendor")


class StaleFleetCache(RuntimeError):
    """`data/fleet/fleet.json` predates a label `etl.loader` now expects.

    Its own type rather than a bare `RuntimeError`: a caller that regenerates
    instead of failing -- `etl.loader` does -- needs to tell this apart from
    any other read failure.

    It carries the **cache's own** `seed` and `scale`, not the caller's. A
    regeneration that used the CLI defaults would hand back a different fleet
    from the one the reader had cached: a `--scale 0.3` cache would silently
    become a scale-1.0 one, and the run after it would be measuring a
    different graph than the run before.
    """

    def __init__(self, message: str, *, missing: tuple[str, ...],
                 seed: int, scale: float):
        super().__init__(message)
        self.missing = missing
        self.seed = seed
        self.scale = scale

def _rid(prefix: str, n: int) -> str:
    return f"{prefix}:{n:05d}"


@dataclass
class Fleet:
    """The complete generated graph, as plain dicts ready for the loader."""
    seed: int
    scale: float
    nodes: dict[str, list[dict]] = field(default_factory=dict)
    edges: list[tuple] = field(default_factory=list)

    def add_nodes(self, label: str, rows: list[dict],
                  provenance: str = "synthetic", source: str = "generated") -> None:
        """Store a copy of each row, stripped of internal `_`-prefixed keys and
        stamped with provenance.

        Copying matters: generators keep working with the original dicts after
        handing them over (e.g. attaching a `_chain` for later wiring), and
        without the copy those internals leak into the graph as node properties.

        Every node carries `provenance` ("real" | "synthetic") and `source`, so
        no query or viewer can confuse a measured value with a generated one.
        """
        self.nodes.setdefault(label, []).extend(
            {**{k: v for k, v in row.items() if not k.startswith("_")},
             "provenance": provenance, "source": source}
            for row in rows
        )

    def add_edge(self, src_label, src_id, rel, tgt_label, tgt_id, props=None) -> None:
        self.edges.append((src_label, src_id, rel, tgt_label, tgt_id, props))

    @property
    def node_count(self) -> int:
        return sum(len(v) for v in self.nodes.values())

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def write(fleet: Fleet, path: Path = FLEET_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "seed": fleet.seed, "scale": fleet.scale,
        "node_count": fleet.node_count, "edge_count": fleet.edge_count,
        "nodes": fleet.nodes, "edges": fleet.edges,
    }), encoding="utf-8")
    return path


def load(path: Path = FLEET_PATH) -> Fleet:
    """The cached fleet, refused if it predates a label the loader now expects.

    `data/` is gitignored, so a fresh clone regenerates and is always current.
    An *existing* checkout is the problem: `git pull` brings a new label --
    `Site` was the first -- while `fleet.json` keeps the shape it was written
    with, and nothing notices. The catalog query over the new label then
    returns zero rows, which reads exactly like "the answer is none" rather
    than "your cache is old"; `--verify` does not catch it either, because it
    compares the load against this same cache.

    A missing label is therefore refused here rather than loaded. What to do
    about it belongs to the caller, not to this function: `etl.loader`
    regenerates from the cache's own seed and scale, which the exception
    carries, while a caller reading the fleet directly gets the command in the
    message. Extra labels are ignored: a cache written by a *newer* checkout
    is not this function's business.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m etl.download_data` first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fleet = Fleet(seed=payload["seed"], scale=payload["scale"])
    fleet.nodes = payload["nodes"]
    fleet.edges = [tuple(e) for e in payload["edges"]]

    missing = [label for label in GENERATED_LABELS if label not in fleet.nodes]
    if missing:
        raise StaleFleetCache(
            f"{path} was written before {', '.join(missing)} existed, so every "
            f"query over {'it' if len(missing) == 1 else 'them'} would return "
            f"zero rows and look like a real answer. Re-run "
            f"`python -m etl.download_data` to rebuild it.",
            missing=tuple(missing), seed=fleet.seed, scale=fleet.scale)
    return fleet
