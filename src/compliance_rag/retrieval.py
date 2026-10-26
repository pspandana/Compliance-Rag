"""Retrievers over the chunk table. Each returns chunks ranked best-first."""

import numpy as np
import pyarrow.parquet as pq

from compliance_rag.ingest import PROCESSED_DIR

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on queries (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks(path=PROCESSED_DIR / "chunks.parquet") -> list[dict]:
    return pq.read_table(path).to_pylist()


def passage_text(chunk: dict) -> str:
    # The title ("Article 17 - Right to erasure ...") carries a lot of the meaning.
    return f"{chunk['title']}\n{chunk['text']}"


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
