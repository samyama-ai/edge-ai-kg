"""The grounding harness scores correctly, and admits it has not been run.

#49 asks for an accuracy comparison with and without the graph. `benchmarks/
mcp_eval.py` builds the question set and the scorer; it cannot call a model
here, so the two numbers do not exist yet.

That makes the scorer the load-bearing part -- it is what any eventual claim
will rest on -- so it is tested on hand-computable cases rather than trusted.
No engine and no `data/` needed for the scoring tests.
"""
import pytest

from benchmarks import mcp_eval


def test_a_perfect_answer_scores_perfectly():
    s = mcp_eval.score_one(["Conv", "Relu"], ["Relu", "Conv"])
    assert s["exact"] is True
    assert s["jaccard"] == 1.0
    assert s["recall"] == 1.0
    assert s["fabricated"] == []
    assert s["refused"] is False


def test_order_does_not_matter_but_duplicates_do_not_help():
    assert mcp_eval.score_one(["A", "B"], ["B", "A", "A"])["exact"] is True


def test_a_refusal_and_a_fabrication_score_the_same_but_report_differently():
    """The distinction #49's result will turn on.

    An ungrounded model asked about a fictional fleet *should* say it cannot
    know. Scoring that identically to a confident invention -- both are 0 --
    would hide the only interesting signal, so the two are recorded separately.
    """
    refusal = mcp_eval.score_one(["Conv", "Relu"], [])
    invention = mcp_eval.score_one(["Conv", "Relu"], ["Gemm", "Softmax"])

    assert refusal["jaccard"] == 0.0
    assert invention["jaccard"] == 0.0          # same score
    assert refusal["refused"] is True
    assert refusal["fabricated"] == []
    assert invention["refused"] is False
    assert invention["fabricated"] == ["Gemm", "Softmax"]


def test_a_correct_none_is_not_counted_as_a_refusal():
    """Some questions have an empty answer, and 'none' is then correct.

    Counting that as a refusal made a *perfect* run report a refusal it never
    made -- the harness printed `refused: 1` while scoring 8/8 exact.
    """
    s = mcp_eval.score_one([], [])
    assert s["exact"] is True
    assert s["jaccard"] == 1.0
    assert s["refused"] is False, (
        "an empty answer to a question whose true answer is empty is correct, "
        "not a refusal"
    )
    assert mcp_eval.score_one([], ["Conv"])["refused"] is False


def test_partial_credit_is_between_zero_and_one():
    s = mcp_eval.score_one(["A", "B", "C", "D"], ["A", "B", "X"])
    assert s["exact"] is False
    assert s["recall"] == 0.5                    # 2 of 4
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["jaccard"] == pytest.approx(2 / 5)  # |{A,B}| / |{A,B,C,D,X}|
    assert s["fabricated"] == ["X"]


@pytest.fixture(scope="module")
def built():
    try:
        return mcp_eval.build_index(seed=4242, scale=0.15)
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


def test_the_questions_carry_ground_truth_and_name_a_tool(built):
    """Every question must be answerable *and* checkable.

    A question with no named tool cannot distinguish "right because grounded"
    from "right anyway", which is the whole comparison #49 asks for.
    """
    fleet, idx = built
    qs = mcp_eval.questions(fleet, idx, limit=10)
    assert qs, "no questions generated"
    for q in qs:
        assert q["question"].strip().endswith("only."), q["id"]
        assert isinstance(q["answer"], list)
        assert q["tool"], f"{q['id']} names no MCP tool"
        assert len(set(q["answer"])) == len(q["answer"]), (
            f"{q['id']} has a duplicate in its ground truth, which would make "
            f"an exact-set comparison unreachable"
        )


def test_the_ground_truth_is_not_trivially_empty_or_total(built):
    """A question everyone gets right measures nothing.

    Guards the two degenerate shapes: every answer empty (so refusing scores
    100%) and every answer the whole universe.
    """
    fleet, idx = built
    qs = mcp_eval.questions(fleet, idx, limit=10)
    sizes = [len(q["answer"]) for q in qs]
    assert any(n > 0 for n in sizes), (
        "every question has an empty answer, so a model that refuses everything "
        "scores 100%"
    )
    operators = len(idx["op"])
    assert all(n < operators for n in sizes), (
        "a question's answer is the entire operator catalog, so naming "
        "everything scores 100%"
    )
