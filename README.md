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
3. Optional second version, the UK GDPR (revised text, Open Government Licence): save
   https://www.legislation.gov.uk/eur/2016/679/data.htm as `data/raw/uk_gdpr.html`.
4. Run ingestion:

```powershell
uv run ingest                              # plain Python
uv run --group spark spark-ingest          # same output built as PySpark stages (needs Java 17+)
```

This writes `data/processed/documents.parquet` (one row per article / recital) and
`data/processed/chunks.parquet`, tagged with `jurisdiction`, `effective_date` and
`version`. The PySpark version also cleans, deduplicates and adds `same_text_as` (which
documents of the other jurisdiction have identical text); its output is checked against
the plain-Python pipeline on every run. `data/processed/` is committed so CI can run the
evals without the raw pages.

## Retrieval eval

`eval/gdpr_golden.jsonl` holds 50 EU questions labelled with the doc_ids that answer them;
`eval/jurisdiction_golden.jsonl` holds 20 more (10 UK-only, 10 EU-vs-UK comparisons).
Each question is searched within the jurisdiction(s) it names (`scope.py`); a question that
names none is assumed to be about the EU.

```powershell
uv run evaluate                        # the 50 EU questions
uv run evaluate --set jurisdiction     # the 20 jurisdiction questions
uv run evaluate --kind article         # metadata filter: articles only
```

Compares four retrievers (dense, BM25, hybrid RRF, hybrid + cross-encoder rerank) on
Recall@k and MRR, and writes per-question results to `data/eval/<retriever>.json`.
Add `--show-misses` to list the questions each retriever ranks wrong.

Original EU-only corpus (articles and recitals), when each stage was added:

| retriever | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| dense (bge-small-en-v1.5) | 0.36 | 0.64 | 0.74 | 0.84 | 0.545 |
| bm25 | 0.26 | 0.49 | 0.61 | 0.70 | 0.445 |
| hybrid (RRF) | 0.37 | 0.69 | 0.77 | 0.81 | 0.577 |
| hybrid + rerank (ms-marco-MiniLM-L-6-v2) | 0.47 | 0.80 | 0.84 | 0.86 | 0.670 |

Current corpus (EU + UK), articles only: dense 0.56 / 0.85 / MRR 0.731, hybrid 0.61 /
0.82 / 0.728, hybrid + rerank 0.81 / 0.88 / 0.878 (R@1 / R@3 / MRR, 50 EU questions).

## Answering questions

Needs `.env` (copy `.env.example`, add `OPENAI_API_KEY`). Default model: `gpt-5.4-mini`.

```powershell
uv run ask "Can I ask a company to delete my data?"
uv run ask "How does the age of consent differ between the EU and the UK?"
uv run ask --golden 50     # answer the golden set; writes data/eval/answers.json
```

Pipeline: article-only hybrid + rerank retrieval (pool of 8) → **Grader** keeps the
relevant chunks (≤5) → **Generator** lists claims with chunk citations → **Verifier**
labels each claim → Beta posterior over verifier, grader-agreement and calibrated
retrieval evidence → route: `auto` / `review` (human) / `abstain`. One retry with a
bigger pool (15) if the grader rejects everything or a claim is unsupported.

```powershell
uv run ask --golden 50 --jurisdiction --unanswerable   # evaluate, with an LLM judge (gpt-5.4) for correctness
uv run calibrate                        # refit reranker score -> probability (after changing retrieval)
uv run gate                             # regression gate (retrieval only, offline)
uv run gate --answers                   # ... plus the full answer evaluation
uv run gate --update                    # accept current numbers as the new baseline
```

Baseline (saved in `eval/baseline.json`; answer metrics vary a few points run to run):

| | Retrieval R@1 / R@3 / MRR | Cites labelled doc(s) | Claims supported | Judged correct |
|---|---|---|---|---|
| **50 EU questions** | 0.81 / 0.88 / 0.878 | 94% | 97% | 76% |
| **20 jurisdiction questions** | 0.525 / 0.925 / 0.804 | 95% (all of them) | 98% | 90% |
| **12 unanswerable** | – | – | – | abstained 10, review 2, auto-answered 0 |

Known limits: confidence separates answerable from unanswerable questions well but
does not predict correctness (AUC 0.41 on 50 questions); the golden set favours
articles, which flatters the article filter. See `docs/GUIDE.md` for details.

## Tuning

Chunk size and the generator prompt were tuned on 35 of the 70 answerable questions and
confirmed on the other 35, with decision rules fixed in advance and a noise measurement
(the same prompt run twice).

```powershell
uv run tune-chunks                                              # search only, no API calls
uv run python -m compliance_rag.tune_prompts v0 v0 v1 v2        # ~20 min per variant, calls the API
```

Outcome: **no change adopted.** 800-character chunks scored slightly better than 1200
(+0.017 tune MRR, +0.015 holdout) but below the 0.02 bar; chunks above ~2,000 characters
clearly hurt. The "complete" prompt variants did not beat run-to-run noise: identical runs
flipped the verdict on 7 of 70 questions. Seventy questions cannot detect small gains.
Details in `docs/GUIDE.md` (Phase 8).

## Explaining the project

`docs/EXPLAINING.md` has short answers on why each piece (and each tool not yet built) is
used, likely questions, and which claims not to make. `docs/INTERVIEW_QA.md` goes deeper:
33 interview-style questions on the challenges and decisions, in simple English.

## Tests

```powershell
uv run pytest
```

## Roadmap

1. Corpus and ingestion
2. Golden set (50 questions) and dense-retrieval baseline: Recall@k, MRR
3. Hybrid retrieval (BM25 + dense, RRF) and cross-encoder reranking
4. Generator and fact-checker agents with Beta-posterior confidence
5. Grader, abstain / human review, calibrated confidence, LLM judge, regression gate, GitHub Actions workflow (lint, tests and retrieval gate pass on GitHub; the manual answer-gate job has not been run)
6. Second version of the law (UK GDPR): jurisdiction / effective date / version metadata, metadata-filtered and jurisdiction-aware retrieval and answers
7. PySpark ingestion (clean, tag, dedupe, link identical text across jurisdictions, chunk), output checked against the plain-Python pipeline
8. Chunk-size and prompt tuning with a holdout half and a noise measurement (no change adopted)
9. Not started: Docker, Terraform, Azure AI Foundry, Confluence; GitHub run of CI; no corpus beyond two related documents
