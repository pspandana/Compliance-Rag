"""Retrievers over the chunk table. Each returns (chunk, score) pairs ranked best-first.

  DenseRetriever   embedding similarity (search by meaning)
  BM25Retriever    keyword matching
  HybridRetriever  dense + BM25 merged with reciprocal rank fusion
  RerankRetriever  re-scores another retriever's shortlist with a cross-encoder
"""

import re

import numpy as np
import pyarrow.parquet as pq

from compliance_rag.ingest import PROCESSED_DIR

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on queries (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Common words that match almost every chunk and only add noise to keyword scores.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or",
    "that", "the", "their", "them", "they", "this", "to", "was", "what", "when", "where",
    "which", "who", "will", "with", "without", "you", "your",
}  # fmt: skip


def load_chunks(path=PROCESSED_DIR / "chunks.parquet") -> list[dict]:
    return pq.read_table(path).to_pylist()


def passage_text(chunk: dict) -> str:
    # The title ("Article 17 - Right to erasure ...") carries a lot of the meaning.
    return f"{chunk['title']}\n{chunk['text']}"


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class DenseRetriever:
    def __init__(self, chunks: list[dict], model_name: str = DENSE_MODEL):
        # Imported here so that tests and light scripts don't pay for loading torch.
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [passage_text(c) for c in chunks], normalize_embeddings=True, batch_size=32
        )

    def search(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        q = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(passage_text(c)) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[tuple[dict, float]]:
    """Merge ranked chunk lists: each list adds 1 / (k + rank) to a chunk's score.

    Only ranks are used, so retrievers whose raw scores are on different scales
    (cosine similarity vs BM25) can be combined without normalising them.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            by_id[chunk["chunk_id"]] = chunk
            scores[chunk["chunk_id"]] = scores.get(chunk["chunk_id"], 0.0) + 1 / (k + rank)
    order = sorted(scores, key=scores.get, reverse=True)
    return [(by_id[cid], scores[cid]) for cid in order]


class HybridRetriever:
    def __init__(self, retrievers: list, pool: int = 50):
        self.retrievers = retrievers
        self.pool = pool

    def search(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        rankings = [[c for c, _ in r.search(query, self.pool)] for r in self.retrievers]
        return reciprocal_rank_fusion(rankings)[:k]


class RerankRetriever:
    def __init__(self, first_stage, model_name: str = RERANK_MODEL, pool: int = 30):
        from sentence_transformers import CrossEncoder

        self.first_stage = first_stage
        self.model = CrossEncoder(model_name)
        self.pool = pool

    def search(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        candidates = [c for c, _ in self.first_stage.search(query, self.pool)]
        scores = self.model.predict([(query, passage_text(c)) for c in candidates])
        top = np.argsort(-scores)[:k]
        return [(candidates[i], float(scores[i])) for i in top]


RETRIEVERS = ("dense", "bm25", "hybrid", "rerank")


def build_retrievers(names: list[str]) -> dict:
    """Build the named retrievers, sharing the models between them."""
    chunks = load_chunks()
    needed = set(names)
    if needed & {"hybrid", "rerank"}:
        needed |= {"dense", "bm25"}

    built = {}
    if "dense" in needed:
        built["dense"] = DenseRetriever(chunks)
    if "bm25" in needed:
        built["bm25"] = BM25Retriever(chunks)
    if needed & {"hybrid", "rerank"}:
        built["hybrid"] = HybridRetriever([built["dense"], built["bm25"]])
    if "rerank" in needed:
        built["rerank"] = RerankRetriever(built["hybrid"])
    return {name: built[name] for name in names}
