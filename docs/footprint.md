# Footprint, measured

Closes #46, under the tracking issue #43.

An edge-AI knowledge graph makes a claim by existing: that a graph is usable
somewhere near the device. That claim is about **memory and binary size**, and
neither was measured anywhere in this repo.

Both are measured here, including the number that does not flatter us.

## Method

Resident set size read from `/proc/self/status` (`VmRSS`) at four points in one
process, so each step's cost is the delta rather than a guess. Embedded build,
`--scale 1.0`, both layers, `samyama` 0.6.1 on Linux.

Reproduce:

```bash
python - <<'PY'
def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
print("baseline", rss_mb())
from samyama import SamyamaClient
from etl import generate as gen, onnx_catalog as oc, real_layer
from etl.helpers import create_nodes, create_edges
from etl.loader import NODE_LABELS, apply_schema
print("after import", rss_mb())
ops = oc.load_cached()
fleet = gen.generate(seed=20260814, scale=1.0, operators=ops)
real_layer.build_real(fleet, ops)
print("after Fleet", rss_mb())
client = SamyamaClient.embedded(); apply_schema(client, "default")
for label in NODE_LABELS:
    if fleet.nodes.get(label):
        create_nodes(client, "default", label, fleet.nodes[label])
create_edges(client, "default", fleet.edges)
print("after load", rss_mb())
PY
```

## Result

25,150 nodes and 76,303 edges:

| stage | RSS | delta |
|---|---:|---:|
| bare Python | 11.8 MB | — |
| after importing `samyama` + `etl` | 22.0 MB | +10.2 MB |
| after building the `Fleet` in Python | 48.6 MB | +26.6 MB |
| after loading into the engine | **247.7 MB** | **+199.1 MB** |

| | |
|---|---:|
| **Graph resident in the engine** | **~199 MB** |
| Whole process, peak | 247.7 MB |
| `samyama` shared object | 22.9 MB |
| `samyama` package on disk | 23.5 MB |

Dropping the Python-side `Fleet` and collecting returns the process to 228.6 MB
with the graph still queryable (`count(n.id)` = 25,150), which is what isolates
the ~199 MB as the engine's rather than Python's.

## Against Neo4j — a win

Neo4j's own guidance starts at **2 GB heap plus page cache**. A 248 MB process
holding the whole graph is roughly an order of magnitude smaller, and it needs
no server, no JVM and no install beyond `pip`.

That is a real difference and it is the one worth quoting.

## Against the devices in the graph — a loss, and a large one

The tempting line is *"the graph describes devices with less RAM than a Neo4j
heap"*. It is true. It is also true of **this** engine, and the page is worthless
if it says only the first half.

Measured across the 120 generated `Board` nodes:

| | ram_kb |
|---|---:|
| min | 64 (64 KB) |
| median | 1,024 (1 MB) |
| max | 16,384 (**16 MB**) |

The graph needs ~199 MB resident plus a 23 MB binary. The largest board in it
has 16 MB of RAM.

> **Boards in this graph that could hold this graph: 0 of 120.**

So "runs at the edge" must mean *near* the device — a gateway, a workstation, a
build machine deciding what to deploy — and not *on* the microcontroller. That
is a coherent and useful position, and it is not the position the phrase "edge"
tends to suggest. Nothing in this repo should imply otherwise.

## What that leaves

The defensible claim is **in-process and small enough to sit alongside something
else**: 248 MB beside a build, a CI job, an MCP server or a developer's editor,
where a 2 GB JVM would not fit comfortably. Not: resident on the target.

## Caveats

- One machine, one build (`samyama` 0.6.1, Linux, embedded). Not a fleet of
  measurements, and RSS includes allocator slack that a different allocator
  would report differently.
- ~8 KB of resident memory per node is not obviously good in absolute terms; it
  is untested whether that ratio holds at 10x, which is #11.
- No comparison was *run* against Neo4j — the 2 GB figure is their published
  guidance, not something measured here. #47 is the head-to-head.
