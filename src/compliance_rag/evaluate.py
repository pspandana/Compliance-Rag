"""Score retrievers against the golden set: Recall@k and MRR at the document level.

Retrievers return chunks; these are collapsed to a ranked list of doc_ids (first
occurrence wins) before scoring, since the golden set is labelled by doc_id.
"""

import argparse
import json
from pathlib import Path

from compliance_rag.ingest import PROJECT_ROOT
from compliance_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    load_chunks,
)

GOLDEN_PATH = PROJECT_ROOT / "eval" / "gdpr_golden.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval"
KS = (1, 3, 5, 10)
RETRIEVERS = ("dense", "bm25", "hybrid", "rerank")


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


def build_retrievers(names: list[str]) -> dict:
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


def score(retriever, golden: list[dict], top_chunks: int) -> list[dict]:
    rows = []
    for q in golden:
        hits = retriever.search(q["question"], k=top_chunks)
        ranked = rank_docs([chunk["doc_id"] for chunk, _ in hits])
        row = {**q, "ranked": ranked[:10], "rr": reciprocal_rank(ranked, q["relevant"])}
        row.update({f"recall@{k}": recall_at_k(ranked, q["relevant"], k) for k in KS})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=list(RETRIEVERS))
    parser.add_argument("--top-chunks", type=int, default=50, help="chunks retrieved per question")
    parser.add_argument("--show-misses", action="store_true", help="list questions where the top result is wrong")
    args = parser.parse_args()

    golden = load_golden()
    retrievers = build_retrievers(args.retrievers)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    header = f"{'retriever':<10}" + "".join(f"{f'R@{k}':>8}" for k in KS) + f"{'MRR':>8}"
    lines = [header]
    for name, retriever in retrievers.items():
        rows = score(retriever, golden, args.top_chunks)
        n = len(rows)
        lines.append(
            f"{name:<10}"
            + "".join(f"{sum(r[f'recall@{k}'] for r in rows) / n:>8.3f}" for k in KS)
            + f"{sum(r['rr'] for r in rows) / n:>8.3f}"
        )
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

        if args.show_misses:
            misses = [r for r in rows if r["rr"] < 1]
            print(f"\n{name}: top result wrong for {len(misses)} questions")
            for r in misses:
                print(f"  {r['id']} expected {r['relevant']}, got {r['ranked'][:3]}  {r['question']}")

    print(f"\n{len(golden)} questions\n" + "\n".join(lines))
    print(f"\nPer-question results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
