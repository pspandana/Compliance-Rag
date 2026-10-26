from compliance_rag.evaluate import rank_docs, recall_at_k, reciprocal_rank


def test_rank_docs_keeps_first_occurrence():
    assert rank_docs(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_at_k():
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, ["a", "b"], 1) == 0
    assert recall_at_k(ranked, ["a", "b"], 2) == 0.5
    assert recall_at_k(ranked, ["a", "b"], 4) == 1


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0
