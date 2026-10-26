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

## Retrieval eval

`eval/gdpr_golden.jsonl` holds 50 questions labelled with the doc_ids that answer them.

```powershell
uv run evaluate
```

Compares four retrievers (dense, BM25, hybrid RRF, hybrid + cross-encoder rerank) on
Recall@k and MRR, and writes per-question results to `data/eval/<retriever>.json`.
Add `--show-misses` to list the questions each retriever ranks wrong.

| retriever | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| dense (bge-small-en-v1.5) | 0.36 | 0.64 | 0.74 | 0.84 | 0.545 |
| bm25 | 0.26 | 0.49 | 0.61 | 0.70 | 0.445 |
| hybrid (RRF) | 0.37 | 0.69 | 0.77 | 0.81 | 0.577 |
| hybrid + rerank (ms-marco-MiniLM-L-6-v2) | 0.47 | 0.80 | 0.84 | 0.86 | 0.670 |

## Answering questions

Needs `.env` (copy `.env.example`, add `OPENAI_API_KEY`). Default model: `gpt-5.4-mini`.

```powershell
uv run ask "Can I ask a company to delete my data?"
uv run ask --golden 50     # answer the golden set; writes data/eval/answers.json
```

Pipeline: article-only hybrid + rerank retrieval (pool of 8) → **Grader** keeps the
relevant chunks (≤5) → **Generator** lists claims with chunk citations → **Verifier**
labels each claim → Beta posterior over verifier, grader-agreement and calibrated
retrieval evidence → route: `auto` / `review` (human) / `abstain`. One retry with a
bigger pool (15) if the grader rejects everything or a claim is unsupported.

```powershell
uv run ask --golden 50 --unanswerable   # evaluate, with an LLM judge (gpt-5.4) for correctness
uv run calibrate                        # refit reranker score -> probability (after changing retrieval)
uv run gate                             # regression gate (retrieval only, offline)
uv run gate --answers                   # ... plus the full answer evaluation
uv run gate --update                    # accept current numbers as the new baseline
```

Baseline (saved in `eval/baseline.json`; answer metrics vary a few points run to run):

| Retrieval (articles only, rerank) | R@1 0.81 | R@3 0.88 | MRR 0.878 |
|---|---|---|---|
| **Answers (50 questions)** | cites labelled doc 94% | claims supported 99% | judged correct 80% |
| **Unanswerable (10)** | abstained 10/10 | auto-answered 0 | |

Known limits: confidence separates answerable from unanswerable questions well but
does not predict correctness (AUC 0.41 on 50 questions); the golden set favours
articles, which flatters the article filter. See `docs/GUIDE.md` for details.

## Tests

```powershell
uv run pytest
```

## Roadmap

1. Corpus and ingestion
2. Golden set (50 questions) and dense-retrieval baseline: Recall@k, MRR
3. Hybrid retrieval (BM25 + dense, RRF) and cross-encoder reranking
4. Generator and fact-checker agents with Beta-posterior confidence
5. Grader, abstain / human review, calibrated confidence, LLM judge, regression gate, GitHub Actions workflow (written, not yet run on GitHub)
6. Not started: PySpark ingestion, multiple documents and versions, Docker, Terraform, Azure AI Foundry, Confluence
