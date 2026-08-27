---
name: Finding
about: A defect, gap or drift in this repo, backed by a measurement and closed by acceptance criteria
title: ""
---

<!--
House style, in one line: state what is wrong, show the number that proves it,
and say what "fixed" looks like. Delete any section that genuinely does not
apply -- but do not delete "Done when".
-->

<!-- What is wrong, in one or two plain sentences. No adjectives. -->


## What was measured

<!--
The evidence. Concrete: file:line references, counts, timings, a table of two
figures that disagree. If two things should match and do not, put them side by
side:

| Where | Value |
|---|---:|
| `README.md` headline | 25,145 |
| `docs/schema.md` node table | 24,115 |

If nothing was measured because nothing measures it, say so -- "nothing times
it" is itself the finding.
-->


## Why nothing catches it

<!--
Optional, but the most useful section in this repo. The failures here tend not
to raise: a count drifts and the suite stays green, an edge is never created and
the load reports success, a query gets slower and no threshold fires.

Say what the symptom would be, or say plainly that there is none.
-->


## Done when

<!--
Acceptance criteria. The house rule: a reader should be able to pick this up
without asking what finished looks like.

Write it as the state of the repo when this is closed, not as a list of tasks.
Prefer criteria a test or a command can settle over criteria a reviewer has to
judge. If the honest answer is "a decision is recorded, either way", that is a
legitimate acceptance criterion -- say which decision and where it gets written.
-->
