"""Score retrievers against the golden set: Recall@k and MRR at the document level.

Retrievers return chunks; these are collapsed to a ranked list of doc_ids (first
occurrence wins) before scoring, since the golden set is labelled by doc_id.
"""

import argparse
import json
from pathlib import Path

from compliance_rag.ingest import PROJECT_ROOT
from compliance_rag.retrieval import RETRIEVERS, build_retrievers
from compliance_rag.scope import scoped_search

GOLDEN_PATH = PROJECT_ROOT / "eval" / "gdpr_golden.jsonl"
JURISDICTION_PATH = PROJECT_ROOT / "eval" / "jurisdiction_golden.jsonl"
UNANSWERABLE_PATH = PROJECT_ROOT / "eval" / "gdpr_unanswerable.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval"
KS = (1, 3, 5, 10)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return _load_jsonl(path)


def load_unanswerable(path: Path = UNANSWERABLE_PATH) -> list[dict]:
    """Questions the GDPR cannot answer; the system should abstain or escalate."""
    return _load_jsonl(path)


def tuning_split() -> tuple[list[dict], list[dict]]:
    """All answerable questions (50 EU + 20 jurisdiction), split by alternating position.

    Tune on the first half, confirm on the second, so a setting is not chosen on the same
    questions it is then reported on.
    """
    questions = load_golden(GOLDEN_PATH) + load_golden(JURISDICTION_PATH)
    return questions[0::2], questions[1::2]


def rank_docs(chunk_doc_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(chunk_doc_ids))


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1 / i
    return 0.0


def score(retriever, golden: list[dict], top_chunks: int, kind: str | None = None) -> list[dict]:
    """Search each question within its detected jurisdiction(s) (see scope.py) and score the ranking."""
    rows = []
    for q in golden:
        hits, _ = scoped_search(retriever, q["question"], top_chunks, kind)
        ranked = rank_docs([chunk["doc_id"] for chunk, _ in hits])
        row = {**q, "ranked": ranked[:10], "rr": reciprocal_rank(ranked, q["relevant"])}
        row.update({f"recall@{k}": recall_at_k(ranked, q["relevant"], k) for k in KS})
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    summary = {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n for k in KS}
    summary["mrr"] = sum(r["rr"] for r in rows) / n
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=list(RETRIEVERS))
    parser.add_argument("--kind", choices=["article", "recital"], help="metadata filter: only search this kind")
    parser.add_argument("--set", choices=["golden", "jurisdiction"], default="golden",
                        help="golden: 50 EU questions; jurisdiction: UK-only and EU-vs-UK questions")
    parser.add_argument("--top-chunks", type=int, default=50, help="chunks retrieved per question")
    parser.add_argument("--show-misses", action="store_true", help="list questions where the top result is wrong")
    args = parser.parse_args()

    golden = load_golden(JURISDICTION_PATH if args.set == "jurisdiction" else GOLDEN_PATH)
    retrievers = build_retrievers(args.retrievers)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    header = f"{'retriever':<10}" + "".join(f"{f'R@{k}':>8}" for k in KS) + f"{'MRR':>8}"
    lines = [header]
    for name, retriever in retrievers.items():
        rows = score(retriever, golden, args.top_chunks, args.kind)
        s = summarize(rows)
        lines.append(f"{name:<10}" + "".join(f"{s[f'recall@{k}']:>8.3f}" for k in KS) + f"{s['mrr']:>8.3f}")
        suffix = (f"_{args.kind}" if args.kind else "") + ("_jurisdiction" if args.set == "jurisdiction" else "")
        (RESULTS_DIR / f"{name}{suffix}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

        if args.show_misses:
            misses = [r for r in rows if r["rr"] < 1]
            print(f"\n{name}: top result wrong for {len(misses)} questions")
            for r in misses:
                print(f"  {r['id']} expected {r['relevant']}, got {r['ranked'][:3]}  {r['question']}")

    where = f", only {args.kind}s" if args.kind else ""
    print(f"\n{len(golden)} questions{where}\n" + "\n".join(lines))
    print(f"\nPer-question results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
