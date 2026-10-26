from compliance_rag.retrieval import reciprocal_rank_fusion, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Right to Erasure?") == ["right", "erasure"]


def test_reciprocal_rank_fusion_rewards_agreement():
    a, b, c = ({"chunk_id": x} for x in "abc")
    # "b" is second in both lists, so it beats "a" and "c", which are each first in only one.
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert [chunk["chunk_id"] for chunk, _ in fused] == ["b", "a", "c"]
