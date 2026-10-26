"""Answer a GDPR question with citations, a fact-check and a confidence score.

  1. Retrieve the top chunks (hybrid search + rerank).
  2. Generator writes an answer and lists its claims, each citing chunk ids.
  3. Fact-checker labels every claim supported / unsupported by its cited chunks.
  4. Confidence is the Beta posterior over those labels (see confidence.py).
  5. Self-correction: if any claim is unsupported, retry once with more chunks and
     the checker's feedback, and keep whichever attempt is more confident.

Usage:
  uv run ask "Can I ask a company to delete my data?"
  uv run ask --golden 10      # run the first 10 golden-set questions and summarise
"""

import argparse
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from compliance_rag.confidence import Confidence, beta_confidence
from compliance_rag.ingest import PROJECT_ROOT
from compliance_rag.llm import get_chat_model
from compliance_rag.retrieval import build_retrievers

RESULTS_DIR = PROJECT_ROOT / "data" / "eval"


class Claim(BaseModel):
    text: str = Field(description="One factual statement made in the answer")
    citations: list[str] = Field(description="ids of the passages that support this statement")


class Draft(BaseModel):
    answer: str = Field(description="Plain-English answer to the question, 2-6 sentences")
    claims: list[Claim] = Field(description="Every factual statement in the answer, one per item")


class Verdict(BaseModel):
    claim_number: int
    supported: bool
    reason: str = Field(description="One short sentence explaining the verdict")


class FactCheck(BaseModel):
    verdicts: list[Verdict]


GENERATOR_PROMPT = """You answer questions about the GDPR using ONLY the passages below.
Each passage starts with its id in square brackets.

- Write a short, plain-English answer that a non-lawyer can understand.
- Then list every factual statement in your answer as a separate claim, citing the
  ids of the passages that support it.
- Do not use outside knowledge. If the passages do not answer the question, say so
  and return no claims.
{feedback}
Passages:
{context}

Question: {question}"""

FEEDBACK = """
A fact-checker rejected these claims from your previous attempt. Fix or drop them:
{rejected}
"""

CHECKER_PROMPT = """You are a strict fact-checker for answers about the GDPR.
For each numbered claim, decide whether the passages it cites support it.
A claim is supported only if the cited passages state it or directly imply it.
Anything added, exaggerated, or not found in the cited passages is unsupported.

Passages:
{context}

Claims:
{claims}"""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] {c['title']}\n{c['text']}" for c in chunks)


@dataclass
class Attempt:
    draft: Draft
    chunks: list[dict]
    verdicts: list[Verdict]
    confidence: Confidence


class AnswerPipeline:
    def __init__(self, k: int = 5, retry_k: int = 10):
        self.retriever = build_retrievers(["rerank"])["rerank"]
        llm = get_chat_model()
        self.generator = llm.with_structured_output(Draft)
        self.checker = llm.with_structured_output(FactCheck)
        self.k = k
        self.retry_k = retry_k

    def attempt(self, question: str, k: int, feedback: str = "") -> Attempt:
        chunks = [c for c, _ in self.retriever.search(question, k)]
        context = format_context(chunks)
        draft = self.generator.invoke(
            GENERATOR_PROMPT.format(context=context, question=question, feedback=feedback)
        )
        verdicts = self.check(draft, chunks, context)
        supported = sum(v.supported for v in verdicts)
        return Attempt(draft, chunks, verdicts, beta_confidence(supported, len(verdicts) - supported))

    def check(self, draft: Draft, chunks: list[dict], context: str) -> list[Verdict]:
        if not draft.claims:
            return []
        known = {c["chunk_id"] for c in chunks}
        claims = "\n".join(
            f"{i}. {claim.text} [cites: {', '.join(claim.citations) or 'nothing'}]"
            for i, claim in enumerate(draft.claims, start=1)
        )
        by_number = {v.claim_number: v for v in self.checker.invoke(CHECKER_PROMPT.format(context=context, claims=claims)).verdicts}

        verdicts = []
        for i, claim in enumerate(draft.claims, start=1):
            verdict = by_number.get(i, Verdict(claim_number=i, supported=False, reason="Not checked."))
            # A claim citing nothing, or citing a passage it was never shown, can't be trusted.
            if not claim.citations or not set(claim.citations) <= known:
                verdict = Verdict(claim_number=i, supported=False, reason="Cites a passage that was not provided.")
            verdicts.append(verdict)
        return verdicts

    def ask(self, question: str) -> list[Attempt]:
        attempts = [self.attempt(question, self.k)]
        rejected = [
            f"- {claim.text} ({v.reason})"
            for claim, v in zip(attempts[0].draft.claims, attempts[0].verdicts)
            if not v.supported
        ]
        if rejected:
            feedback = FEEDBACK.format(rejected="\n".join(rejected))
            attempts.append(self.attempt(question, self.retry_k, feedback))
        return attempts


def best(attempts: list[Attempt]) -> Attempt:
    return max(attempts, key=lambda a: a.confidence.mean)


def print_result(question: str, attempts: list[Attempt]) -> None:
    final = best(attempts)
    conf = final.confidence
    titles = {c["chunk_id"]: c["title"] for c in final.chunks}

    print(f"\nQuestion: {question}\n")
    print(f"Answer: {final.draft.answer}\n")
    print("Claims:")
    for claim, v in zip(final.draft.claims, final.verdicts):
        mark = "[supported]  " if v.supported else "[UNSUPPORTED]"
        print(f"  {mark} {claim.text}  {claim.citations}")
        if not v.supported:
            print(f"                 why: {v.reason}")
    print(
        f"\nConfidence: {conf.label} - {conf.mean:.2f} (90% range {conf.low:.2f}-{conf.high:.2f}), "
        f"{conf.supported} of {conf.supported + conf.unsupported} claims supported"
    )
    cited = dict.fromkeys(cid for claim in final.draft.claims for cid in claim.citations if cid in titles)
    print("Sources: " + "; ".join(sorted({titles[cid] for cid in cited})))
    print(f"Attempts: {len(attempts)}" + (" (retried after the fact-check)" if len(attempts) > 1 else ""))


def run_golden(pipeline: AnswerPipeline, n: int) -> None:
    from compliance_rag.evaluate import load_golden

    rows = []
    for q in load_golden()[:n]:
        attempts = pipeline.ask(q["question"])
        final = best(attempts)
        cited_docs = {cid.rsplit("-c", 1)[0] for claim in final.draft.claims for cid in claim.citations}
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "relevant": q["relevant"],
            "answer": final.draft.answer,
            "cited_docs": sorted(cited_docs),
            "cites_relevant": bool(cited_docs & set(q["relevant"])),
            "supported": final.confidence.supported,
            "unsupported": final.confidence.unsupported,
            "confidence": final.confidence.mean,
            "attempts": len(attempts),
        })
        r = rows[-1]
        print(f"{r['id']}  conf {r['confidence']:.2f}  cites right article: {'yes' if r['cites_relevant'] else 'NO '}  {r['question']}")

    total = len(rows)
    claims = sum(r["supported"] + r["unsupported"] for r in rows)
    print(f"\n{total} questions")
    print(f"  cites the right article   {sum(r['cites_relevant'] for r in rows) / total:.0%}")
    print(f"  claims supported          {sum(r['supported'] for r in rows) / max(claims, 1):.0%} of {claims}")
    print(f"  average confidence        {sum(r['confidence'] for r in rows) / total:.2f}")
    print(f"  needed a retry            {sum(r['attempts'] > 1 for r in rows)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "answers.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPer-question results: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--golden", type=int, metavar="N", help="answer the first N golden-set questions")
    args = parser.parse_args()
    if not args.question and not args.golden:
        parser.error("give a question or --golden N")

    pipeline = AnswerPipeline()
    if args.golden:
        run_golden(pipeline, args.golden)
    else:
        print_result(args.question, pipeline.ask(args.question))


if __name__ == "__main__":
    main()
