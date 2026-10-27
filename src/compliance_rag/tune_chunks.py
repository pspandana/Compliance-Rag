"""Compare chunk sizes by retrieval quality (no API calls).

For each maximum chunk size the saved documents are re-chunked in memory, a fresh index is
built, and the questions are scored on a tune half and a holdout half (see evaluate.tuning_split).
Decision rule, fixed before running: adopt a size other than the current one only if it improves
tune MRR by at least 0.02 and does not lower holdout MRR by more than 0.01.

Note the embedding model reads at most ~512 tokens (about 2,000 characters), so larger chunks are
truncated when embedded; the comparison includes that effect.
"""

import argparse
from dataclasses import asdict, fields

import pyarrow.parquet as pq

from compliance_rag.evaluate import score, summarize, tuning_split
from compliance_rag.ingest import PROCESSED_DIR, Document, chunk_document
from compliance_rag.retrieval import build_retrievers

SIZES = (400, 600, 800, 1200, 1800, 2400, 4000, 8000)
CURRENT = 1200
MIN_TUNE_GAIN, MAX_HOLDOUT_LOSS = 0.02, 0.01


def load_documents_table() -> list[Document]:
    names = {f.name for f in fields(Document)}
    rows = pq.read_table(PROCESSED_DIR / "documents.parquet").to_pylist()
    return [Document(**{k: v for k, v in row.items() if k in names}) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = parser.parse_args()

    docs = load_documents_table()
    tune, holdout = tuning_split()
    print(f"{len(docs)} documents; {len(tune)} tune / {len(holdout)} holdout questions\n")
    print(f"{'max chars':>9}{'chunks':>8}{'avg len':>9} | {'tune R@1':>9}{'R@3':>6}{'MRR':>7} | "
          f"{'hold R@1':>9}{'R@3':>6}{'MRR':>7} | {'all MRR':>8}")

    results = {}
    for size in args.sizes:
        chunks = [asdict(c) for d in docs for c in chunk_document(d, size)]
        retriever = build_retrievers(["rerank"], chunks=chunks)["rerank"]
        t = summarize(score(retriever, tune, 50, "article"))
        h = summarize(score(retriever, holdout, 50, "article"))
        everything = (t["mrr"] * len(tune) + h["mrr"] * len(holdout)) / (len(tune) + len(holdout))
        results[size] = (t, h)
        avg = sum(len(c["text"]) for c in chunks) / len(chunks)
        print(f"{size:>9}{len(chunks):>8}{avg:>9.0f} | {t['recall@1']:>9.3f}{t['recall@3']:>6.2f}{t['mrr']:>7.3f} | "
              f"{h['recall@1']:>9.3f}{h['recall@3']:>6.2f}{h['mrr']:>7.3f} | {everything:>8.3f}")

    if CURRENT in results:
        base_t, base_h = results[CURRENT]
        best = max(results, key=lambda s: results[s][0]["mrr"])
        gain = results[best][0]["mrr"] - base_t["mrr"]
        loss = base_h["mrr"] - results[best][1]["mrr"]
        adopt = best != CURRENT and gain >= MIN_TUNE_GAIN and loss <= MAX_HOLDOUT_LOSS
        print(f"\nBest on tune: {best} chars (tune MRR {gain:+.3f} vs {CURRENT}, holdout MRR {-loss:+.3f})")
        print(f"Decision rule: {'ADOPT ' + str(best) if adopt else 'keep ' + str(CURRENT)}")


if __name__ == "__main__":
    main()
