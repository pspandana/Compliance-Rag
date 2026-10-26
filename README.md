# compliance-rag

Self-correcting RAG over regulatory documents (starting with the GDPR), with hybrid
retrieval, cross-encoder reranking and Bayesian answer confidence.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The project is pinned to Python 3.12.

```powershell
uv sync
```

## Data

1. Open the GDPR on EUR-Lex in a browser:
   https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679
2. Save the page (Ctrl+S, "Webpage, HTML only") as `data/raw/gdpr.html`.
3. Run ingestion:

```powershell
uv run ingest
```

This writes `data/processed/documents.parquet` (one row per article / recital) and
`data/processed/chunks.parquet`.

## Tests

```powershell
uv run pytest
```

## Roadmap

1. Corpus and ingestion
2. Golden set (50 questions) and dense-retrieval baseline: Recall@k, MRR
3. Hybrid retrieval (BM25 + dense, RRF) and cross-encoder reranking
4. Generator and fact-checker agents with Beta-posterior confidence
5. GitHub Actions: tests and retrieval eval on every push
6. Optional: PySpark ingestion, Azure OpenAI, Terraform
