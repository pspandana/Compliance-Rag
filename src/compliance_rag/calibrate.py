"""Calibrate reranker scores to probabilities, for the retrieval part of the confidence score.

The cross-encoder returns raw logits (about -5 for a good match, -11 for an off-topic
question). We fit p = sigmoid(a * score + b) on the golden set: label 1 when the best
labelled document is in the top 3 results, 0 for questions the GDPR cannot answer.
"""

import argparse
import json

from compliance_rag.confidence import CALIBRATION_PATH, fit_platt, retrieval_probability
from compliance_rag.evaluate import (
    JURISDICTION_PATH,
    load_golden,
    load_unanswerable,
    rank_docs,
)
from compliance_rag.retrieval import build_retrievers
from compliance_rag.scope import scoped_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=["article", "recital"], help="only search this kind of document")
    args = parser.parse_args()

    retriever = build_retrievers(["rerank"])["rerank"]
    scores, labels = [], []
    for q in load_golden() + load_golden(JURISDICTION_PATH):
        hits, _ = scoped_search(retriever, q["question"], 50, args.kind)
        top3 = rank_docs([c["doc_id"] for c, _ in hits])[:3]
        scores.append(hits[0][1])
        labels.append(int(bool(set(top3) & set(q["relevant"]))))
    for q in load_unanswerable():
        hits, _ = scoped_search(retriever, q["question"], 50, args.kind)
        scores.append(hits[0][1])
        labels.append(0)

    a, b = fit_platt(scores, labels)
    print(f"{len(scores)} questions ({sum(labels)} with a labelled document in the top 3)")
    print(f"p = sigmoid({a:.3f} * score + {b:.3f})")
    for s in (-12, -10, -8, -6, -4, -2, 0):
        print(f"  score {s:>4}: p = {retrieval_probability(s, a, b):.2f}")
    CALIBRATION_PATH.write_text(json.dumps({"a": a, "b": b, "kind": args.kind}, indent=2), encoding="utf-8")
    print(f"Wrote {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
