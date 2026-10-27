import numpy as np

from compliance_rag.retrieval import (
    FilterableIndex,
    matches,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Right to Erasure?") == ["right", "erasure"]


def test_reciprocal_rank_fusion_rewards_agreement():
    a, b, c = ({"chunk_id": x} for x in "abc")
    # "b" is second in both lists, so it beats "a" and "c", which are each first in only one.
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert [chunk["chunk_id"] for chunk, _ in fused] == ["b", "a", "c"]


def test_matches_supports_single_values_and_lists():
    chunk = {"kind": "article", "jurisdiction": "UK"}
    assert matches(chunk, None)
    assert matches(chunk, {"kind": "article"})
    assert matches(chunk, {"jurisdiction": ["EU", "UK"]})
    assert not matches(chunk, {"kind": "article", "jurisdiction": "EU"})


def test_filter_is_applied_before_ranking():
    index = FilterableIndex()
    index.chunks = [
        {"chunk_id": "eu-best", "jurisdiction": "EU"},
        {"chunk_id": "uk-ok", "jurisdiction": "UK"},
        {"chunk_id": "uk-low", "jurisdiction": "UK"},
    ]

    scores = np.array([0.9, 0.5, 0.1])
    hits = index._top(scores, k=5, where={"jurisdiction": "UK"})
    # The EU chunk scores highest overall but must not appear; k is capped at the 2 UK chunks.
    assert [c["chunk_id"] for c, _ in hits] == ["uk-ok", "uk-low"]
