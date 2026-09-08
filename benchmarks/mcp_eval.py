"""Grounding evaluation: the same questions, with the graph and without it.

#49 asks whether "an LLM grounded in a knowledge graph answers far more
accurately than one guessing" — around 98% against 75% is quoted internally.
Neither figure has been tested, and this graph is unusually suited to testing it
because the hero question has a **checkable ground truth**: which operators lack
a kernel on a given accelerator is a fact the graph computes exactly, so an
answer can be *scored* rather than judged.

## Status: the harness runs, the evaluation has not

This module builds the question set, computes ground truth from the `Fleet`, and
scores answers. It does **not** call a model: no LLM client is installed here and
no key is configured, so the two accuracy numbers #49 asks for do not exist yet
and **nothing in this repo should quote them**.

What it gives whoever has a key: one command to generate the questions, and one
to score a JSONL of answers, with the scoring published rather than described.

    python -m benchmarks.mcp_eval --emit questions.jsonl       # questions ONLY
    python -m benchmarks.mcp_eval --emit-truth truth.jsonl    # keep this away
    python -m benchmarks.mcp_eval --score answers.jsonl        # grade a run

`--emit` deliberately strips the answers. The natural next step is to feed that
file to a model, and shipping the ground truth inside it would quietly produce a
very good number -- the one failure mode a harness like this must not have.

Scoring is against the **question set**, not the answers file: a question with no
answer row counts as a refusal. Dividing by the number answered would make
omission strictly better than refusal, inverting the incentive the `refused`
metric exists to capture.

## Why set-valued questions

Every question here has a **set** as its answer — operator names, board names —
so scoring is Jaccard overlap plus exact match, not a judgement call and not a
second model marking the first. A question whose answer is a sentence would put
a grader in the loop and reintroduce exactly the softness #49 wants to avoid.

The sets are also deliberately *small and specific*. "Which operators does
MobileNet use" is answerable from pre-training; "which operators used by
`cnn-vision-004` have no kernel on this fleet's NPU-Lite units" is not, because
the fleet is fictional. That gap is the thing being measured: a model without
the graph cannot know, and the interesting result is whether it says so or
invents a plausible list.

**A caveat that matters for reading any eventual result:** the vendor and board
names are invented (`CLAUDE.md`), so an ungrounded model is being asked about
entities that do not exist. A low ungrounded score is therefore *expected* and
not by itself evidence of anything — the informative signal is the shape of the
failure (refusal versus confident fabrication), which the scorer records
separately.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import click


def build_index(seed: int, scale: float):
    """The Fleet, plus lookups. Ground truth is computed here, never queried."""
    from etl import generate as gen
    from etl import onnx_catalog as oc
    from etl import real_layer
    ops = oc.load_cached()
    fleet = gen.generate(seed=seed, scale=scale, operators=ops)
    real_layer.build_real(fleet, ops)

    idx = {
        "accel": {a["id"]: a for a in fleet.nodes["Accelerator"]},
        "model": {m["id"]: m for m in fleet.nodes["Model"]},
        "op": {o["id"]: o for o in fleet.nodes["Operator"]},
        "board": {b["id"]: b for b in fleet.nodes["Board"]},
        "impl": defaultdict(set),
        "runs": defaultdict(set),
        "uses": defaultdict(set),
        "has_accel": defaultdict(set),
        "board_soc": {},
    }
    for src, sid, rel, _tgt, tid, _p in fleet.edges:
        if rel == "IMPLEMENTS":
            idx["impl"][sid].add(tid)
        elif rel == "RUNS_ON":
            idx["runs"][sid].add(tid)
        elif rel == "USES_OPERATOR" and src == "Model":
            idx["uses"][sid].add(tid)
        elif rel == "HAS_ACCELERATOR":
            idx["has_accel"][sid].add(tid)
        elif rel == "HAS_SOC":
            idx["board_soc"][sid] = tid
    return fleet, idx


def questions(idx, limit: int) -> list[dict]:
    """Question, ground-truth set, and the MCP tool that answers it.

    The tool is named so a grounded run can be checked for *using* it -- an
    answer that is right without the tool is a different result from one that is
    right because of it, and #49's claim is about the second.
    """
    out = []
    # Split the budget between the two families. `out[:limit]` alone let
    # section 1 fill the whole list at small limits -- its per-kind floor is 1 --
    # and silently drop every `blast/*` question.
    fallback_budget = max(1, limit // 2)

    # 1. The hero question, per (model, accelerator kind).
    kinds = sorted({a["kind"] for a in idx["accel"].values()
                    if a.get("provenance") == "synthetic"})
    models = sorted(idx["uses"], key=lambda m: -len(idx["uses"][m]))
    for kind in kinds:
        # Same provenance filter as `kinds` above. Without it, one archetype
        # renamed to `NPU` would let real ORT kernels mark synthetic operators
        # as covered, silently shrinking the answer set with no test failing.
        on_kind = {a["id"] for a in idx["accel"].values()
                   if a["kind"] == kind and a.get("provenance") == "synthetic"}
        covered = {op for k, ops_ in idx["impl"].items()
                   if idx["runs"].get(k, set()) & on_kind
                   for op in ops_}
        for mid in models[:max(1, fallback_budget // max(1, len(kinds)))]:
            # A set: operator *names* repeat across ids -- `AveragePool` exists
            # at three domains -- so a model using two same-named ids would put
            # a duplicate in the ground truth. Scoring is set-based so grading
            # is unaffected, but the uniqueness invariant the tests assert would
            # fail at some other seed or scale.
            missing = sorted({idx["op"][o]["name"]
                              for o in idx["uses"][mid] if o not in covered})
            out.append({
                "id": f"fallback/{idx['model'][mid]['name']}/{kind}",
                "question": (
                    f"In this fleet, which ONNX operators used by the model "
                    f"'{idx['model'][mid]['name']}' have no kernel on any "
                    f"{kind} accelerator, and so fall back to the CPU? "
                    f"Answer with operator names only."),
                "answer": missing,
                "tool": "fallback_audit",
            })

    # Enforced here, not by `out[:limit]` at the end. Section 1 emits at least
    # one question per accelerator kind, so with 8 kinds and --limit 4 it
    # produced 8 and the final truncation removed every blast question -- the
    # regression the comment above claimed to have fixed.
    out = out[:fallback_budget]

    # 2. Blast radius: which boards carry an accelerator that implements X.
    #
    # No provenance filter here, unlike section 1. Deliberate: "which boards can
    # run this operator" is a question about the whole fleet as loaded, and a
    # real ORT kernel on a real accelerator is a true answer to it. Section 1
    # filters because it compares against a *synthetic* accelerator kind, where
    # a real kernel would be answering a different question.
    op_by_name = {}
    for o in idx["op"].values():
        op_by_name.setdefault(o["name"], o["id"])
    for name in sorted(op_by_name)[:max(1, limit - len(out))]:
        oid = op_by_name[name]
        accels = {a for k, ops_ in idx["impl"].items() if oid in ops_
                  for a in idx["runs"].get(k, set())}
        boards = sorted({idx["board"][b]["name"] for b, soc in idx["board_soc"].items()
                         if idx["has_accel"].get(soc, set()) & accels
                         and b in idx["board"]})
        out.append({
            "id": f"blast/{name}",
            "question": (
                f"In this fleet, which boards have at least one accelerator "
                f"with a kernel for the ONNX operator '{name}'? "
                f"Answer with board names only."),
            "answer": boards,
            "tool": "kernel_blast_radius",
        })
    return out[:limit]


def normalise(name: str) -> str:
    """Fold the differences a free-text answer will have and a graph will not.

    Answers come from model prose, so `"conv"`, `"Conv "` and ``"`Conv`"`` all
    mean the operator the graph calls `Conv`. Comparing raw strings would score
    those 0 on an otherwise perfect answer, and that formatting noise would be
    indistinguishable from real error -- plausibly dominating the very gap this
    harness exists to measure.

    Published rather than described, because the scoring is the claim: strip
    surrounding whitespace, backticks and quotes, then casefold.
    """
    return name.strip().strip("`'\"").strip().casefold()


def score_one(expected: list[str], given: list[str]) -> dict:
    """Jaccard plus exact match, and the shape of the miss.

    `refused` versus `fabricated` is recorded separately because the two are the
    interesting distinction for an ungrounded run: a model that says "I cannot
    know" is behaving well and scores 0 the same as one that invents a list.
    """
    exp = {normalise(x) for x in expected}
    got = {normalise(x) for x in given}
    # Compare normalised, but report what the model actually wrote: a reviewer
    # reading `fabricated` wants the answer as given, not a folded version of it.
    as_given = {}
    for x in given:
        as_given.setdefault(normalise(x), x)
    union = exp | got
    return {
        "expected_n": len(exp),
        "given_n": len(got),
        "exact": exp == got,
        "jaccard": (len(exp & got) / len(union)) if union else 1.0,
        # 1.0 when nothing was expected and nothing was given: naming no wrong
        # thing is perfect precision. Returning 0.0 there was the same bug class
        # as counting a correct "none" as a refusal.
        "precision": (len(exp & got) / len(got)) if got else (0.0 if exp else 1.0),
        "recall": (len(exp & got) / len(exp)) if exp else 1.0,
        # Empty *and* something was expected. A question whose true answer is
        # the empty set is one where "none" is correct, and counting that as a
        # refusal made a perfect run report a refusal it never made.
        "refused": (not got) and bool(exp),
        "fabricated": sorted(as_given.get(x, x) for x in (got - exp)),
    }


@click.command()
@click.option("--emit", type=click.Path(), default=None,
              help="Write the questions -- WITHOUT answers -- to a JSONL file.")
@click.option("--emit-truth", type=click.Path(), default=None,
              help="Write the ground truth separately. Keep it away from the model.")
@click.option("--score", type=click.Path(exists=True), default=None,
              help="Score a JSONL of {id, answer: [...]} against the ground truth.")
@click.option("--limit", default=20, show_default=True)
@click.option("--seed", default=20260814, show_default=True, type=int)
@click.option("--scale", default=1.0, show_default=True, type=float)
def main(emit, emit_truth, score, limit, seed, scale):
    if not emit and not emit_truth and not score:
        raise SystemExit("give --emit questions.jsonl, --emit-truth truth.jsonl "
                         "or --score answers.jsonl")
    if limit < 2:
        # Section 1 emits at least one question and section 2 at least one, so
        # below 2 the truncation drops a whole family and one MCP tool goes
        # untested. Refused rather than special-cased.
        raise SystemExit(f"--limit {limit} cannot cover both question families; "
                         f"use --limit 2 or more")
    if score and (emit or emit_truth):
        # `--emit --score` used to write the questions and return, ignoring the
        # score with no indication it had been skipped.
        raise SystemExit("--score cannot be combined with --emit/--emit-truth; "
                         "emit first, answer, then score")
    try:
        _fleet, idx = build_index(seed, scale)
    except FileNotFoundError:
        raise SystemExit("run `python -m etl.download_data` first") from None
    qs = questions(idx, limit)

    sizes = [len(q["answer"]) for q in qs]
    if not sizes:
        raise SystemExit(
            f"no questions generated at --limit {limit} --scale {scale}; "
            f"nothing to emit or score")

    if emit:
        # Answers are stripped. The natural next step is to feed this file to a
        # model, and shipping the ground truth in it would quietly produce a
        # very good score -- the one failure mode a harness like this must not
        # have.
        # Every row carries the parameters that generated it. `--score` rebuilds
        # the question set from its own defaults, so an --emit at one scale
        # scored at another silently counts the regenerated ids as refusals --
        # deflating the number this harness exists to produce, in the direction
        # that makes a grounded run look worse than it is.
        stamp = {"seed": seed, "scale": scale, "limit": limit}
        asked = [{**{k: v for k, v in q.items() if k != "answer"}, **stamp}
                 for q in qs]
        Path(emit).write_text("\n".join(json.dumps(q) for q in asked) + "\n",
                              encoding="utf-8")
        click.echo(f"wrote {len(qs)} questions (no answers) to {emit}")
        click.echo(f"  answer-set sizes: min {min(sizes)}, "
                   f"median {statistics.median(sizes):g}, max {max(sizes)}")
        empty = sum(1 for s in sizes if s == 0)
        if empty:
            click.echo(f"  {empty} question(s) have an empty answer -- "
                       f"'none' is a correct and checkable answer")

    if emit_truth:
        stamp = {"seed": seed, "scale": scale, "limit": limit}
        Path(emit_truth).write_text(
            "\n".join(json.dumps({"id": q["id"], "answer": q["answer"], **stamp})
                      for q in qs) + "\n", encoding="utf-8")
        click.echo(f"wrote ground truth for {len(qs)} questions to {emit_truth}")

    if emit or emit_truth:
        return

    truth = {q["id"]: q["answer"] for q in qs}
    answered, unknown, stamped = {}, [], False
    for lineno, line in enumerate(
            Path(score).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{score}: line {lineno} is not valid JSON: {exc}") from None
        if "id" not in row:
            raise SystemExit(f"{score}: line {lineno} has no 'id' field")
        if row["id"] not in truth:
            unknown.append(row["id"])
            continue
        for key, mine in (("seed", seed), ("scale", scale), ("limit", limit)):
            theirs = row.get(key)
            if theirs is not None:
                stamped = True
            if theirs is not None and theirs != mine:
                raise SystemExit(
                    f"{score} was produced with {key}={theirs!r} but this run "
                    f"uses {key}={mine!r}. Scoring against a different question "
                    f"set silently counts the regenerated ids as refusals -- "
                    f"pass --{key} {theirs!r}.")
        answered[row["id"]] = row.get("answer", [])

    if answered and not stamped:
        click.echo(f"  ! no row in {score} carries seed/scale/limit, so the "
                   f"question set could not be checked against the one that\n"
                   f"    produced these answers. A mismatch would count the "
                   f"regenerated ids as refusals and deflate the score.", err=True)

    # Scored against the *question set*, not against the answers file. An
    # omitted question counts as an empty answer -- i.e. a refusal -- because
    # dividing by the number answered would make skipping strictly better than
    # refusing: answer the three you are sure of and score 100%. That inverts
    # the incentive the `refused` metric exists to capture.
    results = [(qid, score_one(expected, answered.get(qid, [])))
               for qid, expected in truth.items()]

    if unknown:
        click.echo(f"!! {len(unknown)} answered ids are not in the question set "
                   f"(seed/scale mismatch?): {unknown[:3]}")
    if not results:
        raise SystemExit("no scorable answers")

    exact = sum(1 for _, s in results if s["exact"])
    refused = sum(1 for _, s in results if s["refused"])
    correctly_none = sum(1 for _, s in results
                         if s["exact"] and s["expected_n"] == 0)
    fabricated = sum(1 for _, s in results if s["fabricated"])
    click.echo("")
    click.echo(f"scored {len(results)} questions; "
               f"answered {len(answered)}/{len(truth)}"
               + (f", {len(truth) - len(answered)} unanswered (counted as refusals)"
                  if len(answered) < len(truth) else ""))
    click.echo(f"  exact set match : {exact}/{len(results)} "
               f"= {100 * exact / len(results):.0f}%")
    click.echo(f"  mean Jaccard    : "
               f"{statistics.mean(s['jaccard'] for _, s in results):.3f}")
    click.echo(f"  mean recall     : "
               f"{statistics.mean(s['recall'] for _, s in results):.3f}")
    click.echo(f"  refused (empty) : {refused}"
               + (f"   (+{correctly_none} correct 'none')" if correctly_none else ""))
    click.echo(f"  answers containing a fabrication: {fabricated}")
    click.echo("")
    click.echo("Report both runs together. An ungrounded score is not meaningful "
               "on its own:\nthe fleet is fictional, so a low score is expected "
               "and the informative part is\nwhether the miss was a refusal or a "
               "confident invention.")


if __name__ == "__main__":
    main()
