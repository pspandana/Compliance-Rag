"""Score a retriever against the golden set: Recall@k and MRR at the document level.

Retrievers return chunks; these are collapsed to a ranked list of doc_ids (first
occurrence wins) before scoring, since the golden set is labelled by doc_id.
"""

import argparse
import json
from pathlib import Path

from compliance_rag.ingest import PROJECT_ROOT
from compliance_rag.retrieval import DenseRetriever, load_chunks

GOLDEN_PATH = PROJECT_ROOT / "eval" / "gdpr_golden.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval"
KS = (1, 3, 5, 10)


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_docs(chunk_doc_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(chunk_doc_ids))


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1 / i
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-chunks", type=int, default=50, help="chunks retrieved per question")
    args = parser.parse_args()

    golden = load_golden()
    retriever = DenseRetriever(load_chunks())

    rows = []
    for q in golden:
        hits = retriever.search(q["question"], k=args.top_chunks)
        ranked = rank_docs([chunk["doc_id"] for chunk, _ in hits])
        row = {**q, "ranked": ranked[:10], "rr": reciprocal_rank(ranked, q["relevant"])}
        row.update({f"recall@{k}": recall_at_k(ranked, q["relevant"], k) for k in KS})
        rows.append(row)

    n = len(rows)
    print(f"dense retrieval, {n} questions")
    for k in KS:
        print(f"  Recall@{k:<2} {sum(r[f'recall@{k}'] for r in rows) / n:.3f}")
    print(f"  MRR       {sum(r['rr'] for r in rows) / n:.3f}")

    misses = [r for r in rows if r["rr"] < 1]
    print(f"\nTop result wrong for {len(misses)} questions:")
    for r in misses:
        print(f"  {r['id']} expected {r['relevant']}, got {r['ranked'][:3]}  {r['question']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "dense.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPer-question results: {out}")


if __name__ == "__main__":
    main()
