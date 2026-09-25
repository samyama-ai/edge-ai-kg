# Where a deployment sits, and why that is now in the graph

Answers #34. **Reverses the decline recorded in
[`alerting-scope.md`](alerting-scope.md)** — that decline was written on
2026-09-10; this reversal is dated **2026-09-23**.

> **This is a proposal, not a settled call.** `alerting-scope.md` declined #34
> with three reasons, and this page does not pretend they evaporated. Only
> **one is answered** below; the other **two are accepted as real costs**. The
> decision is the repo owners' to make. **If the decline stands, the PR
> carrying this page should be closed rather than merged**, which also
> removes the "superseded" banner this PR adds to `alerting-scope.md` and
> leaves that page reading exactly as it does today. The code is then what the
> "yes" branch would have cost, made concrete — which is easier to judge than
> the argument alone.

## What was decided before, and what changed

`alerting-scope.md`'s rule was: *take every question the existing schema can
already answer, decline every one that needs a new label this repo would have to
invent.* `Site` needs invention, so #34 was declined — while the same page
recorded what it cost:

> *"which sensors are on this floor"*, *"is this a site-wide failure or one
> device"*, and the "which sites" half of #33's *who is affected* are
> unanswerable here and will stay so.

#34 is the issue asking for exactly those. So the question is not whether the
rule was applied correctly — it was — but whether the cost it names is one this
repo should keep paying. This page says no, for one reason: **the alerting
theme this repo did take is incomplete without a place.** `EA17` (what stops)
and `EA19` (which certifications are implicated) answer the other halves, and
neither can say *where to send someone* — the first thing an operations team
asks once it knows what broke.

Both of those are now **on `main`**, delivered by #113. They were merged into
a base branch whose own PR had already closed, so for two days they existed
without reaching `main`; that is resolved, and the argument above no longer
leans on anything pending.

`EA20` itself depends on nothing beyond `main` in any case — it walks `Site`,
`Deployment` and `Board` — so the code stands on its own either way.

## The three reasons against, taken one at a time

**1. "No upstream supplies it, so a `Site` spine would be invented wholesale."**
True, and unchanged. What is answerable is the risk behind it — that generated
figures read as claims about real products:

- `Site` is **generated-layer only**. Every row is stamped
  `provenance: "synthetic"`, like every other generated node.
- **No real node is ever placed.** The real layer's 73 MLPerf Tiny deployments
  get no `DEPLOYED_AT` edge, because nobody published where those submissions
  ran. `tests/test_site_spine.py::test_the_real_layer_gets_no_sites` fails if
  one is ever attached.
- Campus names are fictional in the same way the vendors are — *Meridian
  General*, *Ferrous Line 4* — so no row can be read as a real installation.

**2. "It changes without the catalog changing — a board moves between wards,
its kernels do not."** This is the strongest objection and it is **accepted, not
answered**. The placement here is a demonstration of the *shape* of the
question, not a register anyone should keep current. Two consequences, both
deliberate:

- Placement is derived from the seed like everything else in
  `etl/generate.py`, so it is reproducible and moves only when the fleet is
  regenerated. It does not drift on a maintenance schedule because nothing here
  tracks a real maintenance schedule.
- The graph should not become the system of record for where devices are. If
  this repo ever imports a real asset register, the import — not the generator —
  is what supplies the edge.

**3. "It belongs to the system that already owns it (a CMDB)."** Also accepted.
The claim made here is narrower than a CMDB's: *this deployment is at this
site*, and nothing else. There is no move history, no commissioning date, no
asset tag, no person. If those start appearing, this decision should be
revisited again, because at that point the graph is duplicating an asset
register rather than demonstrating a question.

## The shape, and why it is the smallest one that answers #34

`alerting-scope.md` named the join point in advance, and this follows it:

> *"**If the decision is revisited**, the join point is `Deployment`, not
> `Sensor` and not `Board`."*

`Deployment` it is. A deployment is one installed instance — this variant, on
this board, with these measurements — while a `Board` is a product type that
could be installed in twenty places at once, and `Sensor` here is a catalogue of
types rather than serial numbers.

| choice | taken | why not the alternative |
|---|---|---|
| join point | `Deployment -[:DEPLOYED_AT]-> Site` | `Board` is a type, not an instance; `Sensor` is a catalogue entry |
| grouping | `Site.campus`, a property | a `Campus` label adds a node nothing else references and one more hop on every "is this site-wide?" query |
| zone | the `Site` *is* the zone | a `Zone` label would be a node per ward with one edge; a `zone` property on the edge reads correctly embedded but is unmeasured on the 1.7.0 server, and the catalog is swept over HTTP |
| cardinality | exactly one site per deployment | anything else turns `count(DISTINCT d.id)` per site into a set union, and the per-site counts would no longer sum to the fleet |

At `--scale 1.0` that is **12 sites across 4 campuses, and 1,440 `DEPLOYED_AT`
edges** — one per generated deployment, which
`tests/test_site_spine.py::test_every_generated_deployment_sits_at_exactly_one_site`
pins.

## What it buys, measured

`EA20` answers the second of the three questions the decline gave up:

> *A board model is recalled. Which sites run it, how many of each site's
> deployments does it account for, and is that site-wide or one device?*

Two counts arrive on one row — the site's deployments, and how many are on the
recalled board — so the difference between "replace one unit in Ward 3" and
"this whole floor is affected" is readable rather than inferred. Both columns
are recomputed from the `Fleet` in Python and compared row by row in
`tests/test_site_queries.py::test_ea20_counts_match_ground_truth`; the
generator invariants it rests on are in `tests/test_site_spine.py`.

**What that demonstrates depends on which run you do, and the difference is
worth stating on a page arguing to be judged on what was measured.** The
default `pytest` run uses a small fixture — 4 sites, 36 boards — where every
site holds at least one of the recalled board, so it can show a *partially*
affected site and nothing else. The contrast case, a site holding **none** of
them, needs the shipped graph: it is
`test_ea20_shows_both_affected_and_untouched_sites_at_full_scale`, which is
**skipped unless `pytest --full-scale` is given**. Measured on that run, on a
205-operator catalogue: of 12 sites, 6 held none of `board:00003` and 6 held
some but not all. So "site-wide or one device" is demonstrated at `--scale
1.0` and only half-demonstrated by a default run.

**That 6/6 split moves with the upstream ONNX catalogue, not only with the
seed.** The catalogue decides how many kernels each accelerator registers,
which decides which boards a variant fits, which decides where deployments
land. The 6-and-6 above is the 205-operator catalogue; on a 379-operator one
it is 5 untouched of 12. The test asserts only that **both
kinds exist**, never the ratio, because a number that moves on someone else's
release schedule is not a claim this repo can keep.

## What this still does not claim

- **Not that the other two declines moved.** #38 (alert state) and #39
  (ownership) stand exactly as `alerting-scope.md` records them. `Site` is one
  label, and the guard in `tests/test_alerting_scope.py` now permits that one
  and still fails on `Team`, `Alert`, `Zone`, `Building`, `OWNS` and the rest.
- **Not a location for anything real.** No real node carries a place, and a test
  fails if one ever does.
- **Not "which sensors are on this floor" yet.** Sensors are types here, so a
  sensor is placed only through the deployments that use its pipeline. The
  question #34's title asks is answerable one hop further out than this PR goes,
  and saying so is better than implying it landed.
- **Not measured: whether an operations team wants this shape.** Same limit
  `alerting-scope.md` states for the other five — demand is not evidence this
  repo holds.
