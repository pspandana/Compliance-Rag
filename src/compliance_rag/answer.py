"""Answer a GDPR question with a grader, generator and verifier, then decide what to do with it.

  1. Retrieve a pool of chunks (metadata filter + hybrid search + rerank).
  2. Grader keeps only the chunks that help answer the question.
  3. Generator writes an answer and lists its claims, each citing chunk ids.
  4. Verifier labels every claim supported / unsupported by its cited chunks.
  5. Confidence is a Beta posterior over verifier, grader and retrieval evidence.
  6. Route: "auto" answer, "review" by a human, or "abstain" when there is no evidence.
  Self-correction: if the grader rejects everything, or any claim is unsupported, retry
  once with a bigger pool (and the verifier's feedback) and keep the better attempt.

Usage:
  uv run ask "Can I ask a company to delete my data?"
  uv run ask --golden 50        # answer the golden set and score it
  uv run ask --unanswerable     # questions the GDPR cannot answer (should abstain / escalate)
"""

import argparse
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from compliance_rag.confidence import (
    Confidence,
    beta_confidence,
    load_calibration,
    retrieval_probability,
    route,
)
from compliance_rag.evaluate import load_golden, load_unanswerable
from compliance_rag.grader import Grader
from compliance_rag.ingest import PROJECT_ROOT
from compliance_rag.llm import get_chat_model
from compliance_rag.retrieval import build_retrievers

RESULTS_DIR = PROJECT_ROOT / "data" / "eval"
POOL = 8  # chunks retrieved before grading
RETRY_POOL = 15
MAX_CONTEXT = 5  # chunks the generator sees at most
NO_ANSWER = "I can't answer this from the GDPR text I have."


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
    pool: list[dict]  # everything retrieved, best first
    kept: list[dict]  # what the grader let through (the generator's context)
    draft: Draft
    verdicts: list[Verdict]
    confidence: Confidence
    route: str
    retrieval_p: float

    @property
    def answer(self) -> str:
        return self.draft.answer if self.route != "abstain" else NO_ANSWER


class AnswerPipeline:
    def __init__(self, kind: str | None = "article", use_grader: bool = True):
        self.retriever = build_retrievers(["rerank"], kind=kind)["rerank"]
        llm = get_chat_model()
        self.grader = Grader(llm) if use_grader else None
        self.generator = llm.with_structured_output(Draft)
        self.checker = llm.with_structured_output(FactCheck)
        self.calibration = load_calibration()

    def attempt(self, question: str, pool_size: int, feedback: str = "") -> Attempt:
        hits = self.retriever.search(question, pool_size)
        pool = [c for c, _ in hits]
        if self.grader:
            grades = self.grader.grade(question, pool)
        else:
            grades = {c["chunk_id"]: True for c in pool}
        kept = [c for c in pool if grades[c["chunk_id"]]][:MAX_CONTEXT]
        retrieval_p = retrieval_probability(hits[0][1], *self.calibration)

        # Grader agreement is judged on the three chunks the retriever ranked highest.
        agree = sum(grades[c["chunk_id"]] for c in pool[:3])
        evidence = {"grader_yes": agree, "grader_no": len(pool[:3]) - agree, "retrieval_p": retrieval_p}

        if not kept:
            conf = beta_confidence(0, 0, **evidence)
            return Attempt(pool, kept, Draft(answer="", claims=[]), [], conf, "abstain", retrieval_p)

        context = format_context(kept)
        draft = self.generator.invoke(
            GENERATOR_PROMPT.format(context=context, question=question, feedback=feedback)
        )
        verdicts = self.check(draft, kept, context)
        supported = sum(v.supported for v in verdicts)
        conf = beta_confidence(supported, len(verdicts) - supported, **evidence)
        return Attempt(pool, kept, draft, verdicts, conf, route(conf, len(draft.claims), len(kept)), retrieval_p)

    def check(self, draft: Draft, chunks: list[dict], context: str) -> list[Verdict]:
        if not draft.claims:
            return []
        known = {c["chunk_id"] for c in chunks}
        claims = "\n".join(
            f"{i}. {claim.text} [cites: {', '.join(claim.citations) or 'nothing'}]"
            for i, claim in enumerate(draft.claims, start=1)
        )
        result = self.checker.invoke(CHECKER_PROMPT.format(context=context, claims=claims))
        by_number = {v.claim_number: v for v in result.verdicts}

        verdicts = []
        for i, claim in enumerate(draft.claims, start=1):
            verdict = by_number.get(i, Verdict(claim_number=i, supported=False, reason="Not checked."))
            # A claim citing nothing, or citing a passage it was never shown, can't be trusted.
            if not claim.citations or not set(claim.citations) <= known:
                verdict = Verdict(claim_number=i, supported=False, reason="Cites a passage that was not provided.")
            verdicts.append(verdict)
        return verdicts

    def ask(self, question: str) -> list[Attempt]:
        first = self.attempt(question, POOL)
        attempts = [first]
        rejected = [
            f"- {claim.text} ({v.reason})"
            for claim, v in zip(first.draft.claims, first.verdicts, strict=True)
            if not v.supported
        ]
        if not first.kept or rejected:
            feedback = FEEDBACK.format(rejected="\n".join(rejected)) if rejected else ""
            attempts.append(self.attempt(question, RETRY_POOL, feedback))
        return attempts


def best(attempts: list[Attempt]) -> Attempt:
    return max(attempts, key=lambda a: (a.route != "abstain", a.confidence.mean))


def print_result(question: str, attempts: list[Attempt]) -> None:
    final = best(attempts)
    conf = final.confidence
    print(f"\nQuestion: {question}\n")
    print(f"Answer: {final.answer}\n")
    for claim, v in zip(final.draft.claims, final.verdicts, strict=True):
        mark = "[supported]  " if v.supported else "[UNSUPPORTED]"
        print(f"  {mark} {claim.text}  {claim.citations}")
        if not v.supported:
            print(f"                 why: {v.reason}")
    print(
        f"\nDecision: {final.route.upper()}"
        f"  (confidence {conf.mean:.2f}, 90% range {conf.low:.2f}-{conf.high:.2f};"
        f" {conf.supported} of {conf.supported + conf.unsupported} claims supported)"
    )
    print(f"Grader kept {len(final.kept)} of {len(final.pool)} retrieved chunks; retrieval p = {final.retrieval_p:.2f}")
    titles = sorted({c["title"] for c in final.kept})
    print("Sources: " + ("; ".join(titles) if titles else "none"))
    print(f"Attempts: {len(attempts)}")


def irrelevant_share(chunks: list[dict], relevant: list[str]) -> float | None:
    """Share of chunks that belong to a document the golden set does not label as relevant."""
    if not chunks:
        return None
    return sum(c["doc_id"] not in relevant for c in chunks) / len(chunks)


def mean(values) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def run_golden(pipeline: AnswerPipeline, judge, n: int) -> dict:
    rows = []
    for q in load_golden()[:n]:
        attempts = pipeline.ask(q["question"])
        final = best(attempts)
        first = attempts[0]
        cited_docs = {cid.rsplit("-c", 1)[0] for claim in final.draft.claims for cid in claim.citations}
        verdict = judge.correctness(q["question"], final.answer, q["relevant"]) if judge and final.kept else None
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "relevant": q["relevant"],
            "answer": final.answer,
            "route": final.route,
            "cites_relevant": bool(cited_docs & set(q["relevant"])),
            "pool_has_relevant": any(c["doc_id"] in q["relevant"] for c in first.pool),
            "kept_has_relevant": any(c["doc_id"] in q["relevant"] for c in first.kept),
            "irrelevant_before": irrelevant_share(first.pool[:5], q["relevant"]),
            "irrelevant_after": irrelevant_share(first.kept, q["relevant"]),
            "supported": final.confidence.supported,
            "unsupported": final.confidence.unsupported,
            "confidence": final.confidence.mean,
            "attempts": len(attempts),
            "correctness": verdict.verdict if verdict else "incorrect",
            "judge_reason": verdict.reason if verdict else "Abstained on an answerable question.",
        })
        r = rows[-1]
        print(f"{r['id']}  {r['route']:<7} conf {r['confidence']:.2f}  {r['correctness']:<9} {r['question']}")

    summary = summarize_golden(rows)
    print(f"\n{len(rows)} answerable questions")
    print(f"  cites the labelled article   {summary['cites_relevant']:.0%}")
    print(f"  claims supported             {summary['supported_rate']:.0%}")
    print(f"  judged correct               {summary['correct']:.0%}  (partial {summary['partial']:.0%})")
    print(f"  routing                      auto {summary['auto']:.0%}, review {summary['review']:.0%}, abstain {summary['abstain']:.0%}")
    print(f"  correct when auto-answered   {summary['correct_when_auto']:.0%}")
    print(f"  correct when sent to review  {summary['correct_when_review']:.0%}")
    print(f"  irrelevant context           {summary['irrelevant_before']:.0%} before grader -> {summary['irrelevant_after']:.0%} after")
    print(f"  grader kept a labelled doc   {summary['grader_recall']:.0%} of the questions where the pool had one")
    print(f"  needed a retry               {sum(r['attempts'] > 1 for r in rows)}")
    save("answers.json", rows)
    return summary


def summarize_golden(rows: list[dict]) -> dict:
    n = len(rows)
    claims = sum(r["supported"] + r["unsupported"] for r in rows)

    def share(rs, key, value):
        return sum(r[key] == value for r in rs) / len(rs) if rs else float("nan")

    auto = [r for r in rows if r["route"] == "auto"]
    review = [r for r in rows if r["route"] == "review"]
    in_pool = [r for r in rows if r["pool_has_relevant"]]
    return {
        "cites_relevant": sum(r["cites_relevant"] for r in rows) / n,
        "supported_rate": sum(r["supported"] for r in rows) / max(claims, 1),
        "correct": share(rows, "correctness", "correct"),
        "partial": share(rows, "correctness", "partial"),
        "auto": len(auto) / n,
        "review": len(review) / n,
        "abstain": share(rows, "route", "abstain"),
        "correct_when_auto": share(auto, "correctness", "correct"),
        "correct_when_review": share(review, "correctness", "correct"),
        "irrelevant_before": mean(r["irrelevant_before"] for r in rows),
        "irrelevant_after": mean(r["irrelevant_after"] for r in rows),
        "grader_recall": sum(r["kept_has_relevant"] for r in in_pool) / max(len(in_pool), 1),
    }


def run_unanswerable(pipeline: AnswerPipeline, judge) -> dict:
    rows = []
    for q in load_unanswerable():
        final = best(pipeline.ask(q["question"]))
        verdict = judge.abstention(q["question"], final.answer) if judge else None
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "route": final.route,
            "answer": final.answer,
            "confidence": final.confidence.mean,
            "handled": verdict.verdict if verdict else "n/a",
        })
        r = rows[-1]
        print(f"{r['id']}  {r['route']:<7} conf {r['confidence']:.2f}  {r['handled']:<9} {r['question']}")

    n = len(rows)
    summary = {
        "abstained": sum(r["route"] == "abstain" for r in rows) / n,
        "wrongly_auto": sum(r["route"] == "auto" for r in rows),
        "handled_correctly": sum(r["handled"] == "correct" for r in rows) / n,
    }
    print(f"\n{n} unanswerable questions")
    print(f"  abstained                    {summary['abstained']:.0%}")
    print(f"  answered automatically       {summary['wrongly_auto']} (should be 0)")
    print(f"  response judged appropriate  {summary['handled_correctly']:.0%}")
    save("unanswerable.json", rows)
    return summary


def save(name: str, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / name
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPer-question results: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--golden", type=int, metavar="N", help="answer the first N golden-set questions")
    parser.add_argument("--unanswerable", action="store_true", help="run the questions the GDPR cannot answer")
    parser.add_argument("--kind", choices=["article", "recital", "any"], default="article",
                        help="metadata filter for retrieval (default: article)")
    parser.add_argument("--no-grader", action="store_true", help="skip the grader (ablation)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM correctness judge")
    args = parser.parse_args()
    if not (args.question or args.golden or args.unanswerable):
        parser.error("give a question, --golden N or --unanswerable")

    pipeline = AnswerPipeline(kind=None if args.kind == "any" else args.kind, use_grader=not args.no_grader)
    if args.question:
        print_result(args.question, pipeline.ask(args.question))
        return

    judge = None
    if not args.no_judge:
        from compliance_rag.judge import Judge

        judge = Judge()
    if args.golden:
        run_golden(pipeline, judge, args.golden)
    if args.unanswerable:
        run_unanswerable(pipeline, judge)


if __name__ == "__main__":
    main()
