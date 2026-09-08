# What this graph can and cannot do for alerting

Closes #41, under the alerting tracking issue #33.

The failure this page exists to prevent is a reader concluding the graph is an
alerting product. It is not, and saying so plainly is what makes the second half
credible.

Both halves below are checked against the schema rather than asserted.

---

## The graph is the wrong tool for most of alerting

**Measured by absence.** Searching the schema, the loader, the generator, the
catalog and the MCP server for the concepts alerting is built from:

| Concept | Occurrences in the schema and code |
|---|---:|
| `timestamp` | **0** |
| `reading` | **0** |
| `alert` | **0** |
| `severity` | **0** |
| `acknowledg…` | **0** |
| `location` / `site` / `latitude` | **0** |
| `owner` | **0** |

Nothing in this repo stores a value that changes over time. `Sensor` carries
`modality`, `sample_rate_hz`, `channels` and `adc_bits` — the *capability* to
produce readings, never a reading. `Deployment` carries measured
`latency_ms`, `power_mw` and `energy_mj`, but as a single characterisation of a
deployment, not a series.

So the graph cannot, and should not be extended to:

- **poll sensors** or ingest a stream
- **store readings** — there is no time dimension at all
- **evaluate thresholds at rate** — a graph query per reading is the wrong shape
  and this engine loads edges at ~3.1K/s ([`engine-notes.md`](engine-notes.md)),
  which is an ingest rate for a dataset, not for telemetry
- **deliver messages**, or offer retry and delivery guarantees
- **hold alert state** — firing, acknowledged, resolved, escalated — because
  there is nothing to hold it on and no time to hold it against

All of that is time-series and messaging infrastructure. Prometheus, InfluxDB,
Kafka, or the platform you already run.

**There is also no `Alert`, `Rule` or `Threshold` node.** #38 asks whether to
model one; this page describes today, where the answer is no.

---

## The graph is the right tool for what depends on what

The dependency chain is real and already loaded:

```
Sensor -FEEDS-> SignalStage -NEXT_STAGE-> … -PRECEDES-> Model
                                                          |
                                                       SOLVES
                                                          v
                                                    ClinicalTask -GOVERNED_BY-> Certification
                                                          |
                                                   REQUIRES_SENSOR
```

### Blast radius — answerable today

*One sensor fails. What stops working?* This runs against the shipped graph
with no new modelling:

```cypher
MATCH (s:Sensor)-[:FEEDS]->(st:SignalStage)-[:NEXT_STAGE*0..3]->(last:SignalStage)
      -[:PRECEDES]->(m:Model)-[:SOLVES]->(t:ClinicalTask)
WHERE s.name = "ECG 3-lead"
WITH t.name AS task, t.latency_budget_ms AS budget_ms, count(m.id) AS models
RETURN task, budget_ms, models
ORDER BY models DESC
LIMIT 5
```

**Measured** (`--scale 0.3`, seed 4242):

```
['Wheeze detection', 1500, 1]
['Cough event classification', 1000, 1]
```

The variable-length `NEXT_STAGE*0..3` is the part a relational schema makes
painful: the DSP chain is of unknown depth, and the answer must not depend on
knowing it in advance.

### Compliance implication — answerable today

*The same failure — is it also a regulatory event?*

```cypher
MATCH (s:Sensor)<-[:REQUIRES_SENSOR]-(t:ClinicalTask)-[:GOVERNED_BY]->(cert:Certification)
WHERE s.name = "ECG 3-lead"
WITH cert.name AS certification, cert.body AS body, count(t.id) AS tasks
RETURN certification, body, tasks
ORDER BY tasks DESC
LIMIT 5
```

**Measured:**

```
['IEC 62304 Class A', 'IEC', 2]
['ISO 13485', 'ISO', 1]
['EU MDR Class IIa', 'EU', 1]
```

This is the join a time-series database cannot do at all: it has the values and
none of the structure.

### Silent degradation — the shape the graph is uniquely good at

The catalog's hero question is *which operators have no kernel on this
accelerator and therefore fall back to the CPU*. Nothing about that shows up in
a threshold: no value changed, no reading went out of range. The device is
simply slower than its budget, and only the structure reveals why. `EA01`,
`EA02` and `EA11` are that question.

#37 is the general form of this and is open.

---

## What is missing before this is an alerting story

Stated so nobody assumes otherwise:

| Gap | Issue |
|---|---|
| Sensors have no location, so "a sensor in some place" cannot be asked | #34 |
| No blast-radius query in the catalog — the query above is in this page, not in `benchmarks/queries.py` | #35 |
| No ranking of simultaneous alerts by what is upstream | #36 |
| Silent degradation is not expressed as an alerting question | #37 |
| No `Alert` / `Rule` / `Threshold` spine, and no decision not to have one | #38 |
| No organisation spine, so "who owns this" has no answer | #39 |
| Certification-driven alerting is not worked through | #40 |
| No worked end-to-end demo | #42 |

**The honest summary:** the graph answers *what depends on what, and what does
that imply* — including at unknown depth and across the clinical/regulatory
join. It does not detect anything, store anything that changes, or tell anyone.
It is the map, not the monitor.

A system that alerts would use both: the time-series stack to notice, this graph
to work out who and what is affected.

---

## A note on the numbers above

They come from the generated fleet, which is deliberately fictional — vendor and
board names are invented so no figure reads as a claim about a real product
(`CLAUDE.md`). "ECG 3-lead feeds wheeze detection" is an artefact of random
pairing in `etl/generate.py`, not a clinical claim.

**What is being demonstrated is the traversal, not the pairing.**
