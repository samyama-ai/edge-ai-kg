# Edge AI KG -- schema

17 node labels, 23 edge types. At `--scale 1.0`, seed `20260814`, the generated
layer is **24,127 nodes, 75,265 edges**.

**Every count on this page is that generated layer unless it carries
`(+M real)`** (#14). The two totals a reader is likely to want:

| | nodes | edges |
|---|---:|---:|
| generated layer -- what this page's bare counts sum to | 24,127 | 75,265 |
| both layers -- what `python -m etl.loader` actually loads | 25,162 | 77,743 |
| the real layer adds | +1,035 | +2,478 |

The README quotes both-layer figures, so the two pages disagreed by 1,035 nodes
with nothing saying why. They are the same graph counted to different edges; the
`(+M real)` column below is the reconciliation.

**The both-layer row is a snapshot and nothing pins it.** The generated column
is ours and deterministic from the seed, so it is asserted by
`tests/test_schema_docs.py`. The `+1,035 / +2,478` comes from ONNX Runtime and
MLPerf, which publish on their own schedule -- 734 kernel registrations became
738 during one week of this backlog. **Other pages may therefore quote an
earlier snapshot than this one**; where they disagree, the figures here were
measured most recently, and #17 is the work that binds every page's published
counts to a rebuild so the lag is caught rather than discovered.

See [`data-provenance.md`](data-provenance.md) for what is real and what is
synthetic, and [`engine-notes.md`](engine-notes.md) for the v1.7.0 behaviours
the queries work around.

The graph answers one shape of question: **can this model run on this silicon,
and what does it cost me when it can't?**

## Node labels

| Label | Count | Key fields |
|---|---:|---|
| `Kernel` | 21,844 (+738 real) | id, name, efficiency, is_fallback |
| `Deployment` | 1,440 (+73 real) | id, latency_ms, power_mw, energy_mj, memory_kb, fallback_op_count, fallback_fraction, accelerator_kind, fits |
| `ModelVariant` | 240 | id, name, precision, size_kb, accuracy, format |
| `Operator` | 205 (+171 real) | id, name, domain, since_version, version_count, category, is_control_flow |
| `Board` | 120 (+14 real) | id, name, form_factor, price_usd, power_budget_mw, ram_kb, flash_kb, year, battery_powered |
| `Accelerator` | 85 (+6 real) | id, name, kind, gops_int8, sram_kb, clock_mhz, opset_ceiling, energy_factor, is_cpu_fallback |
| `Model` | 60 (+4 real) | id, name, family, task, params_k, macs_m |
| `SoC` | 40 (+12 real) | id, name, process_nm, cpu_arch, cpu_mhz, cores |
| `ClinicalTask` | 18 | id, name, category, latency_budget_ms, min_sensitivity |
| `SignalStage` | 16 | id, name, kind, window_ms, cost_kmacs |
| `Sensor` | 14 | id, name, modality, sample_rate_hz, channels, adc_bits |
| `Dataset` | 12 | id, name, source, subjects, hours, license |
| `Vendor` | 8 (+7 real) | id, name, country |
| `Runtime` | 7 (+6 real) | id, name, version, format |
| `Site` | 12 | id, name, kind, campus, region |
| `Certification` | 6 | id, name, body, class |
| `BenchmarkTask` | 0 (+4 real) | id, name, code, dataset, metric, quality_target |

Every node carries a unique `id`; `schema/edge_ai_kg.cypher` indexes it per
label. Uniqueness is a loader invariant -- this engine does not parse
`CREATE CONSTRAINT` -- and it holds per load, not per graph. Loading twice into
the same graph with `--no-reset` mints every id again; nothing rejects the
write, and the duplicate ids then multiply edges rather than merely doubling
nodes. `tests/test_id_uniqueness.py` asserts the invariant.

A count written `N (+M real)` is N from the generated layer plus M more once the
real layer is loaded. **N is what the generated-layer total counts; N+M is what
the both-layer total counts.** The seven labels with no `(+M real)` --
`ModelVariant`, `Sensor`, `SignalStage`, `ClinicalTask`, `Dataset`,
`Certification` and `Site` -- are generated only, so their two counts are the
same number.

The `(+M real)` figures are a snapshot of the current upstream dumps, not an
invariant: ONNX Runtime's kernel registrations went 734 to 738 during one week
of this backlog. `tests/test_schema_docs.py` pins the generated column, which is
ours, and deliberately not the real one.

`BenchmarkTask` is the one label the generator does not produce. Its four nodes
-- Anomaly Detection, Image Classification, Keyword Spotting, Visual Wake Words
-- come from MLPerf Tiny v1.2 and are stamped `provenance: "real"`, so a
generated-only load holds none of them and they fall outside the 24,127 total
above. That is why the label was absent from this table until now; the counts
here describe the generated layer, which is itself worth stating more plainly
(see #14).

## Edge types

| Edge | From -> To | Count | Meaning |
|---|---|---:|---|
| `IMPLEMENTS` | Kernel -> Operator | 21,844 (+738 real) | this kernel implements this operator |
| `RUNS_ON` | Kernel -> Accelerator | 21,844 (+738 real) | on this compute unit |
| `PROVIDED_BY` | Kernel -> Runtime | 21,844 (+738 real) | shipped by this runtime |
| `OF_VARIANT` | Deployment -> ModelVariant | 1,440 | what was deployed |
| `ON_BOARD` | Deployment -> Board | 1,440 (+73 real) | where |
| `VIA_RUNTIME` | Deployment -> Runtime | 1,440 (+60 real) | through which runtime |
| `USES_ACCELERATOR` | Deployment -> Accelerator | 1,440 (+11 real) | on which compute unit |
| `DEPLOYED_AT` | Deployment -> Site | 1,440 | where it physically sits |
| `USES_OPERATOR` | Model -> Operator `{count}` | 1,069 | model's operator surface |
| `TARGETS` | Runtime -> Accelerator | 426 (+3 real) | runtime can target this unit |
| `VARIANT_OF` | ModelVariant -> Model | 240 | fp32 / fp16 / int8 / int4 |
| `MADE_BY` | Board or SoC -> Vendor | 160 (+26 real) | supply chain |
| `HAS_SOC` | Board -> SoC | 120 (+14 real) | board's chip |
| `CERTIFIED_FOR` | Board -> Certification | 104 | regulatory posture |
| `HAS_ACCELERATOR` | SoC -> Accelerator | 85 | chip's compute units |
| `TRAINED_ON` | Model -> Dataset | 82 | provenance |
| `SOLVES` | Model -> ClinicalTask; Model -> BenchmarkTask | 60 (+4 real) | clinical purpose; MLPerf Tiny task |
| `PRECEDES` | SignalStage -> Model | 60 | pipeline feeds model |
| `REQUIRES_SENSOR` | ClinicalTask -> Sensor | 51 | required modality |
| `NEXT_STAGE` | SignalStage -> SignalStage | 40 | DSP chain |
| `GOVERNED_BY` | ClinicalTask -> Certification | 22 | regulatory requirement |
| `FEEDS` | Sensor -> SignalStage | 14 | front of the pipeline |
| `MEASURES` | Deployment -> Model | 0 (+73 real) | a measured MLPerf Tiny submission against its reference model |

The `N` column is the generated layer and sums to the **75,265** stated above --
verified against the `Fleet` and pinned by `tests/test_schema_docs.py`. Eleven
of the 23 types also gain real edges, written `(+M real)`; those add 2,478 more,
which is the whole of the real layer (see [`data-provenance.md`](data-provenance.md)).

**The `(+M real)` figures are a snapshot, not an invariant.** They move whenever
ONNX Runtime or MLPerf publish -- 734 kernel registrations became 738 during one
week -- so no test pins them; a test that fails for upstream's reasons is one
people learn to ignore. The generated column is ours and is pinned.

`MEASURES` is real-only for the same reason `BenchmarkTask` is -- the generator
does not emit it -- and carries one edge per real MLPerf Tiny submission.
`SOLVES` gains its second target, `Model -> BenchmarkTask`, only when the real
layer is loaded.

**Why this table once looked 160 short (#16).** `MADE_BY` covers two source
labels, and the row used to write them `Board\|SoC`. Markdown needs the pipe
escaped inside a cell, but anything splitting the row on every `|` -- a reader
counting columns, or a script -- gets an extra column, reads `160` as the
endpoints and `supply chain` as the count, and so scores `MADE_BY` as zero.
73,825 - 160 = 73,665, which is exactly the sum the issue reported. The table
was never wrong; it was unparseable. It now reads `Board or SoC`, and
`tests/test_schema_docs.py` rejects an escaped pipe anywhere in the table so the
column stays addable.

## The two spines

**Hardware spine** -- what silicon can do:

```
Vendor <- SoC <- Board
           |
           +-> Accelerator <- RUNS_ON - Kernel - IMPLEMENTS -> Operator
                    ^                      |
                    |                 PROVIDED_BY
                 TARGETS                   v
                    +-------------------- Runtime
```

**Clinical spine** -- what the device has to do:

```
Sensor -> SignalStage -> ... -> SignalStage -> Model -> ClinicalTask
                                                |            |
                                          ModelVariant   Certification
                                                |
                                           Deployment -> Board
```

The two spines meet at `Operator` (a model's operator surface vs. a kernel
library's coverage) and at `Deployment` (a measured landing of one variant on
one board). Those two joins are where every interesting query lives.

## Accelerator archetypes

`kind` drives which operator categories a unit can run and its opset ceiling --
this is what creates the coverage gaps the queries hunt for.

**These five archetypes describe the generated layer only.** Counts are
`--scale 1.0`, seed `20260814`, and sum to the 85 in the node table above.

| Kind | Count | Opset ceiling | Covers | int8 GOPS | Energy factor |
|---|---:|---:|---|---|---:|
| `MCU-CPU` | 40 | 99 | every category (universal fallback) | 0.5-3 | 1.00 |
| `DSP` | 9 | 17 | activation, convolution, elementwise, matmul, reduction, shape, signal, spatial | 8-40 | 0.42 |
| `NPU-Lite` | 12 | 13 | activation, convolution, elementwise, matmul, normalization, quantization, spatial | 30-120 | 0.16 |
| `NPU-Pro` | 9 | 19 | activation, attention, convolution, elementwise, matmul, normalization, quantization, reduction, shape, spatial | 150-900 | 0.11 |
| `GPU-Embedded` | 15 | 21 | activation, attention, convolution, elementwise, matmul, normalization, quantization, recurrent, reduction, shape, spatial, tensor | 400-2400 | 0.30 |
| | **85** | | | | |

Rows are in `ACCEL_ARCHETYPES` order and each `Covers` cell is spelled out in
full. It used to read incrementally -- `NPU-Pro` as "+ reduction, attention,
shape" over the row above -- and that notation was wrong: the sets are **not** a
subset chain. `DSP` carries `signal`, which nothing below it has, and
`NPU-Lite` drops `reduction` and `shape` that `DSP` holds. Four of the five rows
understated their categories. `tests/test_accelerator_kinds.py` now pins this
column against `ACCEL_ARCHETYPES`.

`Opset ceiling` is not a count and does not sum to anything -- summing it gives
169, which is how #26 came to compare it against 85. See
[What `opset_ceiling` means](#what-opset_ceiling-means) below; `99` is a
sentinel for *no ceiling*, not ninety-nine units.

### Kinds outside the archetype table

Loading the real layer as well takes `Accelerator` from **85 to 91**, and the
six extra are *not* more of the five kinds above. They carry four kinds that
have no archetype:

| Kind | Count | Source |
|---|---:|---|
| `NPU` | 3 | MLPerf Tiny submitter hardware |
| `CPU` | 1 | ONNX Runtime CPU execution provider |
| `GPU-CUDA` | 1 | ONNX Runtime CUDA execution provider |
| `GPU-DirectML` | 1 | ONNX Runtime DirectML execution provider |

So **`kind` is an open vocabulary, not a closed set of five.**
`etl/real_layer.py` names these from `ORT_DEVICES` and the MLPerf accelerator
strings -- and `ORT_DEVICES` also holds `GPU-ROCm` and `GPU-TensorRT`, which
appear as soon as a dump mentions them. Having no archetype, none of them
carries an `opset_ceiling`, a category list, a `gops_int8` or an
`energy_factor`.

**This changed catalog answers, and #69 fixed it.** `EA11` asks which models
fall back to the CPU. It used to filter `WHERE a.kind <> "MCU-CPU"`, so ONNX
Runtime's CPU execution provider -- spelled `CPU` -- did not match, and its
kernels counted as *acceleration*. Measured at `--scale 1.0`:

| models EA11 reports as CPU-only | generated only | both layers |
|---|---:|---:|
| old rule, `kind <> "MCU-CPU"` | 60 | **12** |
| new rule, `is_cpu_fallback = 0` | 60 | **20** |

`Accelerator.is_cpu_fallback` is now set where each accelerator is created --
`1` for `MCU-CPU` and for ONNX Runtime's CPU provider, `0` otherwise -- so the
classification lives with the definition instead of being inferred from a string
in a query. The generated layer is unchanged, because `MCU-CPU` is its only CPU;
the real layer recovers 8 models the spelling had hidden. Not all 60: CUDA and
DirectML are genuine accelerators, so operators they implement are correctly not
CPU-only.

`EA05` and `EA08` still group by `kind` and gain four extra rows, which is a
presentation difference rather than a wrong answer.

`tests/test_accelerator_kinds.py` pins the vocabulary: that the generated layer
uses exactly the archetype kinds, that the real layer uses none of them, that
kinds without an archetype carry no archetype properties, that **every CPU is
marked as a fallback target and no accelerator is**, and that the catalog no
longer infers the distinction from a `kind` spelling.

### Which count is on which page

| Page | Figure | Layer |
|---|---:|---|
| `docs/schema.md` node table | 85 | generated only |
| `DATASET_CARD.md` | 91 | both layers |
| `README.md` | 91 | both layers |

All three are correct; none of them said so.

### What `opset_ceiling` means

It is a **hard cut on kernel existence**, not a hint. `etl/generate.py` creates a
`Kernel` for an `(operator, accelerator, runtime)` triple only when all of these
hold, and the ceiling is one of them:

```python
if o.category not in a["_cats"]:            continue   # wrong category
if o.since_version > a["opset_ceiling"]:    continue   # above the ceiling
if o.is_control_flow and a["kind"] != "MCU-CPU": continue
if a["kind"] != "MCU-CPU" and rng.random() < 0.18:     continue   # vendor gap
```

So the rule the graph obeys is:

> **No `Kernel` exists on an `Accelerator` for an `Operator` whose
> `since_version` is above that accelerator's `opset_ceiling`.**

**Read `since_version` carefully -- it is the operator's *latest revision*, not
the opset it was introduced at.** `etl/onnx_catalog.py` sets it to
`max(versions)`, so `Pad`, introduced at opset 1 and revised ten times through
25, carries 25 and is excluded from every band below that. An `NPU-Lite` at
ceiling 13 is therefore not "too old for new operators" -- it is missing kernels
for plenty of ancient ones that upstream has since revised. That is the
realistic case the dataset exists to show, and it is easy to read backwards.

Measured on a full load: **0 of 21,844** generated kernel placements break it,
and the bound is tight rather than loose -- in every band the highest operator
opset actually used *equals* the ceiling:

| Ceiling | Highest `since_version` used | Kernels |
|---:|---:|---:|
| 13 | **13** | 780 |
| 17 | **17** | 704 |
| 19 | **19** | 1,273 |
| 21 | **21** | 2,892 |
| 99 | 28 | 16,195 |

`99` on `MCU-CPU` is a sentinel for *no ceiling*: ONNX's highest `since_version`
in this catalog is 28, so nothing is excluded by it. That is what makes the
MCU-CPU the universal fallback, and it is why the hero question always has an
answer -- something can always run, just slowly.

**The real layer has no ceiling at all.** Real accelerators -- currently the
MLPerf Tiny NPUs and ONNX Runtime's execution providers -- carry no
`opset_ceiling`, because their kernel registrations are read from upstream
rather than derived from a rule. Their placements are outside this invariant:
not violating it, not covered by it, and silently excluded by any query
filtering on `opset_ceiling`.

How many such accelerators there are is not fixed: `add_ort_layer` derives them
from the device ids present in the ONNX Runtime dump and `add_mlperf_layer` from
the distinct accelerator strings in the MLPerf results, so refreshing either
changes the count.

`tests/test_opset_ceiling.py` asserts the *rule* and the provenance split: no
kernel above its band's ceiling after a load, that every band both covers and
excludes operators (so the rule is not holding vacuously), and that real
accelerators carry no ceiling.

The **figures** above are not pinned. They are a full `--scale 1.0` load and the
test runs at `0.3`; more importantly they move whenever the generator, the seed
or the upstream dumps change. `tests/test_generate.py::test_kernels_respect_opset_ceiling`
asserts the same rule directly on the `Fleet` without an engine.

Every SoC has an `MCU-CPU`, so *something* can always run -- the question the
graph answers is never "can it run" but "is it ever accelerated, and what does
the fallback cost".
