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

    python -m benchmarks.mcp_eval --emit questions.jsonl     # questions + truth
    python -m benchmarks.mcp_eval --score answers.jsonl      # grade a run

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


def questions(fleet, idx, limit: int) -> list[dict]:
    """Question, ground-truth set, and the MCP tool that answers it.

    The tool is named so a grounded run can be checked for *using* it -- an
    answer that is right without the tool is a different result from one that is
    right because of it, and #49's claim is about the second.
    """
    out = []

    # 1. The hero question, per (model, accelerator kind).
    kinds = sorted({a["kind"] for a in idx["accel"].values()
                    if a.get("provenance") == "synthetic"})
    models = sorted(idx["uses"], key=lambda m: -len(idx["uses"][m]))
    for kind in kinds:
        on_kind = {a["id"] for a in idx["accel"].values() if a["kind"] == kind}
        covered = {op for k, ops_ in idx["impl"].items()
                   if idx["runs"].get(k, set()) & on_kind
                   for op in ops_}
        for mid in models[:max(1, limit // (len(kinds) * 2))]:
            missing = sorted(idx["op"][o]["name"]
                             for o in idx["uses"][mid] if o not in covered)
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

    # 2. Blast radius: which boards carry an accelerator that implements X.
    op_by_name = {}
    for o in idx["op"].values():
        op_by_name.setdefault(o["name"], o["id"])
    for name in sorted(op_by_name)[:limit // 4 or 1]:
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


def score_one(expected: list[str], given: list[str]) -> dict:
    """Jaccard plus exact match, and the shape of the miss.

    `refused` versus `fabricated` is recorded separately because the two are the
    interesting distinction for an ungrounded run: a model that says "I cannot
    know" is behaving well and scores 0 the same as one that invents a list.
    """
    exp, got = set(expected), set(given)
    union = exp | got
    return {
        "expected_n": len(exp),
        "given_n": len(got),
        "exact": exp == got,
        "jaccard": (len(exp & got) / len(union)) if union else 1.0,
        "precision": (len(exp & got) / len(got)) if got else 0.0,
        "recall": (len(exp & got) / len(exp)) if exp else 1.0,
        # Empty *and* something was expected. A question whose true answer is
        # the empty set is one where "none" is correct, and counting that as a
        # refusal made a perfect run report a refusal it never made.
        "refused": (not got) and bool(exp),
        "fabricated": sorted(got - exp),
    }


@click.command()
@click.option("--emit", type=click.Path(), default=None,
              help="Write the question set with ground truth to a JSONL file.")
@click.option("--score", type=click.Path(exists=True), default=None,
              help="Score a JSONL of {id, answer: [...]} against the ground truth.")
@click.option("--limit", default=20, show_default=True)
@click.option("--seed", default=20260814, show_default=True, type=int)
@click.option("--scale", default=1.0, show_default=True, type=float)
def main(emit, score, limit, seed, scale):
    if not emit and not score:
        raise SystemExit("give --emit questions.jsonl or --score answers.jsonl")
    try:
        fleet, idx = build_index(seed, scale)
    except FileNotFoundError:
        raise SystemExit("run `python -m etl.download_data` first") from None
    qs = questions(fleet, idx, limit)

    if emit:
        Path(emit).write_text("\n".join(json.dumps(q) for q in qs) + "\n",
                              encoding="utf-8")
        click.echo(f"wrote {len(qs)} questions with ground truth to {emit}")
        sizes = [len(q["answer"]) for q in qs]
        click.echo(f"  answer-set sizes: min {min(sizes)}, "
                   f"median {int(statistics.median(sizes))}, max {max(sizes)}")
        empty = sum(1 for s in sizes if s == 0)
        if empty:
            click.echo(f"  {empty} question(s) have an empty answer -- "
                       f"'none' is a correct and checkable answer")
        return

    truth = {q["id"]: q["answer"] for q in qs}
    results, unknown = [], []
    for line in Path(score).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] not in truth:
            unknown.append(row["id"])
            continue
        results.append((row["id"], score_one(truth[row["id"]], row.get("answer", []))))

    if unknown:
        click.echo(f"!! {len(unknown)} answered ids are not in the question set "
                   f"(seed/scale mismatch?): {unknown[:3]}")
    if not results:
        raise SystemExit("no scorable answers")

    exact = sum(1 for _, s in results if s["exact"])
    refused = sum(1 for _, s in results if s["refused"])
    correctly_none = sum(1 for _, s in results
                         if s["exact"] and s["expected_n"] == 0)
    fabricated = sum(1 for _, s in results if s["fabricated"] and not s["exact"])
    click.echo("")
    click.echo(f"scored {len(results)} answers")
    click.echo(f"  exact set match : {exact}/{len(results)} "
               f"= {100 * exact / len(results):.0f}%")
    click.echo(f"  mean Jaccard    : "
               f"{statistics.mean(s['jaccard'] for _, s in results):.3f}")
    click.echo(f"  mean recall     : "
               f"{statistics.mean(s['recall'] for _, s in results):.3f}")
    click.echo(f"  refused (empty) : {refused}"
               + (f"   (+{correctly_none} correct 'none')" if correctly_none else ""))
    click.echo(f"  fabricated names: {fabricated}")
    click.echo("")
    click.echo("Report both runs together. An ungrounded score is not meaningful "
               "on its own:\nthe fleet is fictional, so a low score is expected "
               "and the informative part is\nwhether the miss was a refusal or a "
               "confident invention.")


if __name__ == "__main__":
    main()
