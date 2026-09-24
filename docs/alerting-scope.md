# What this graph takes on when a sensor misbehaves, and what it does not

Answers the scoping question in #33, and the three decisions it defers to:
#34 (location), #38 (alert/rule/threshold state), #39 (ownership).

**Read [`alerting.md`](alerting.md) first.** It answers #41 under the same
tracking issue and establishes the half that matters most: the graph is *not* an
alerting product, measured by absence — no `timestamp`, no threshold, no rule
anywhere in the schema. This page does not re-argue that. It takes it as settled
and decides which of the remaining questions this repo builds.

#33 asks it plainly: *a company puts a sensor somewhere. How do they get an
alert when something is wrong, and can the graph help at all?* It also says
**nothing is built until the decision exists**, which is why this page comes
before the queries.

## The rule

> **Take every question the existing schema can already answer. Decline every
> one that needs a new label this repo would have to invent.**

Not a preference — a consequence of what the data is. Every node here traces to
ONNX, ONNX Runtime, MLPerf Tiny, or a generator whose vendor and board names are
*deliberately fictional* so no number can be read as a claim about a real
product (`DATASET_CARD.md`). A `Site` or a `Team` has no upstream to derive
from. It would be invented wholesale, be 100% synthetic, and look exactly as
authoritative as the measured half.

The rule turns out to split the family cleanly, which is the reason to trust it
rather than the reason it was chosen. Measured on the shipped graph:

| # | question | needs a new label? | verdict |
|---|---|---|---|
| #35 | blast radius: what stops with this sensor | no | **take** — delivered as `EA17` (#96, reaching `main` with #113) |
| #36 | root cause versus symptom | no | **take** — delivered as `EA21` |
| #37 | silent degradation against a latency budget | no | **take** — delivered as `EA18` |
| #40 | which certifications a failure implicates | no | **take** — delivered as `EA19` |
| #42 | a worked demo beat | no | **take** |
| #34 | where the sensor physically is | `Site` | **decline** |
| #39 | who owns the affected asset | `Team` | **decline** |
| #38 | alert / rule / threshold state | `Alert` | **decline** |

## Why the five are takeable, checked rather than assumed

Every edge and property they need is already loaded:

```
#35  FEEDS, NEXT_STAGE, PRECEDES, ON_BOARD ............... all present
#36  NEXT_STAGE gives the upstream ordering .............. present
#37  ClinicalTask.latency_budget_ms ...................... present
     Deployment.latency_ms, Deployment.fallback_op_count .. present
     Deployment -> ModelVariant -> Model -> ClinicalTask ... joinable
#40  Certification + GOVERNED_BY .......................... present
```

`#37` is the one worth pointing at. It is the failure **no threshold on the
sensor would ever catch** — the sensor is fine, the model behind it now runs N
operators on the CPU and misses its task's latency budget — and the graph can
already see it, because the fallback count and the budget are two hops apart.
That is the alerting theme and the hero question meeting in one row.

`#35`, `#37`, `#40` and `#36` are delivered — as `EA17`, `EA18`, `EA19` and
`EA21`. The page and the queries land together, so wherever this is read they
are in the catalog beside it. The catalog is `EA01`-`EA21`, with one gap:
**`EA20` is not in it.** That id is claimed by #115's `Site` query, which is a
separate PR; two queries sharing an id would silently overwrite one of them in
`BY_ID`, so the gap is deliberate. If #115 is closed without merging — its own
body says it should be, if the decision to decline #34 stands — then `EA20` is
free again, and the next query to be added takes it rather than leaving a hole
here. Either way this sentence is what tells a reader the gap is not an
accident.

That is not a promise this page makes about itself.
`tests/test_alerting_scope.py::test_the_delivery_claims_match_the_catalog`
reads every "delivered as `EAnn`" **anywhere on this page** — the verdict table
is where they happen to be written, but the check is not scoped to it — and
fails three ways: if the catalog does not hold a claimed id, if the stated
`EA01`-`EAnn` range is not the catalog's actual last id, and if `EA17`, `EA18`
or `EA19` is in the catalog with **no delivery claim written for it**. That
last one is about the claim being absent, not about the word "pending": a page
that described them in any other words would fail it just the same.

`EA17` is the one catalog query the **1.7.0 server** cannot answer *at all*:
that build does not traverse variable-length relationships and rejects
`size(r)` on one (engine note 12). `EA07` uses a variable-length walk too, but
a bounded one with no `size(r)`, so the server runs it and returns only the
zero-length match -- wrong rows rather than an error, which note 12 calls the
more dangerous half. Note 12 is careful about what that does and does not show —
1.7.0 *embedded* was never measured, so this may be a fix that landed between
1.7.0 and 1.7.1 rather than anything HTTP-specific. What is measured is that
it runs on embedded 1.7.1 and fails on the 1.7.0 server.
`tests/test_alerting_scope.py::test_the_delivery_claims_match_the_catalog` —
the same check as above — keeps this honest in both directions: it fails if the
page calls `#35` pending while `EA17` is in the catalog, and if `EA17` is
reverted while the page still says it is delivered.

## Why the three are declined

### #34 — location is deployment-time state, not catalog state

`Sensor` carries `modality`, `sample_rate_hz`, `channels`, `adc_bits`. `Board`
carries a form factor and a price. **Nothing anywhere carries a place**, and the
one near-miss is worth naming so nobody mistakes it for one: `Vendor.country` is
where a *vendor* is headquartered, not where anything is installed.

It is also thinner than it looks. Only the 8 **synthetic** vendors carry a value
(`DE`, `IN`, `JP`, `UK`, `US`); all 7 real-layer vendors — Qualcomm,
STMicroelectronics, Bosch and the rest — carry `""`, because no upstream source
supplies it. So the one property that resembles a location is empty for every
row that is real.

Three reasons not to add it:

1. **No upstream supplies it.** ONNX, ONNX Runtime and MLPerf Tiny describe
   operators, kernels and submissions. None knows where a device sits. A `Site`
   spine would be invented entirely, and this repo's standard is that generated
   figures must not be readable as claims about real products.
2. **It changes without the catalog changing.** A board moves between wards; its
   kernels do not. Mixing a fact that changes on a maintenance schedule into a
   catalog derived from published releases means the two go stale on different
   clocks, and only one of them can be re-fetched.
3. **It belongs to the system that already owns it.** Sites, assets and their
   moves are a CMDB or asset-register concern. A graph that duplicates them
   holds a second, worse copy.

**What that costs, stated plainly:** *"which sensors are on this floor"*,
*"is this a site-wide failure or one device"*, and the "which sites" half of
#33's *who is affected* are unanswerable here and will stay so. `#40` still
answers the regulatory half.

**If the decision is revisited**, the join point is `Deployment`, not `Sensor`
and not `Board`. A deployment is already "this variant, on this board, with
these measurements" — the closest thing here to an installed instance — and a
board can be redeployed while a deployment cannot. `Sensor -[:DEPLOYED_AT]->`
would be wrong for the same reason: sensors are a catalog of types here, not
serial numbers.

### #39 — ownership, and a name clash worth recording

Same reasoning: `Team` or `Contact` has no upstream. There is also a concrete
trap #39 flags and this page confirms: **`Operator` is already taken.** It means
an ONNX operator, 376 of them. An organisational "operator" would collide with
the single most-referenced label in the schema, so anyone revisiting this must
pick another word before writing a line of Cypher.

### #38 — alert state is not a graph problem

#33 draws this line itself and it is the right one: reading a sensor,
thresholding a value, sending a message is a time-series pipeline and a
notification service. Storing `Alert` / `Rule` / `Threshold` rows in a graph
adds mutable operational state to a catalog whose value is that it is derived
and reproducible from published sources.

The graph's contribution is **the sentence that goes in the alert**, not the
alert. `DATASET_CARD.md` records that alerting state is out of scope, which #38
asks for explicitly.

## What this decision does not claim

- **Not that location is unimportant.** It is the first thing an operations team
  asks. It is answered somewhere else.
- **Not that the declined three are wrong forever.** Each names its join point
  and its blocker above, so revisiting is a decision rather than a rediscovery.
- **Not measured: whether anyone wants the five.** This records what the graph
  *can* answer and what it would have to invent. Demand is not evidence this
  repo holds.
