"""LLM-as-judge for answer correctness.

The judge sees the question, the system's answer and the gold passages (the text of
the articles / recitals the golden set labels as the answer), and decides whether the
answer is consistent with them and actually addresses the question. It is a different,
larger model than the one that wrote the answer.
"""

import os
from typing import Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from compliance_rag.ingest import PROCESSED_DIR
from compliance_rag.llm import get_chat_model

DEFAULT_JUDGE_MODEL = "gpt-5.4"


class Judgement(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description="correct: addresses the question and agrees with the gold passages; "
        "partial: right direction but misses a key point or adds a wrong detail; "
        "incorrect: wrong, off-topic, or contradicts the gold passages"
    )
    reason: str = Field(description="One sentence")


JUDGE_PROMPT = """You are grading an answer from a GDPR question-answering system.

Question: {question}

Gold passages (the authoritative text that answers the question):
{gold}

System answer:
{answer}

Grade the answer against the gold passages. Ignore style and length. Do not penalise
extra correct detail, only wrong or missing key points."""

ABSTAIN_PROMPT = """The question below cannot be answered from the GDPR. Grade the system's
response: "correct" if it clearly says it cannot answer from the available text (or declines),
"incorrect" if it presents a substantive answer as if the GDPR covered it.

Question: {question}

System response:
{answer}"""


class Judge:
    def __init__(self, model: str | None = None):
        model = model or os.getenv("OPENAI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self.chain = get_chat_model(model).with_structured_output(Judgement)
        docs = pq.read_table(PROCESSED_DIR / "documents.parquet").to_pylist()
        self.docs = {d["doc_id"]: d for d in docs}

    def correctness(self, question: str, answer: str, relevant: list[str]) -> Judgement:
        gold = "\n\n".join(f"{self.docs[d]['title']}\n{self.docs[d]['text']}" for d in relevant)
        return self.chain.invoke(JUDGE_PROMPT.format(question=question, gold=gold, answer=answer))

    def abstention(self, question: str, answer: str) -> Judgement:
        return self.chain.invoke(ABSTAIN_PROMPT.format(question=question, answer=answer))
