"""Grader agent: decide which retrieved chunks are actually relevant to the question.

The reranker orders chunks but never says "none of these help". The grader reads the
question and every candidate and keeps only the chunks that help answer it.
"""

from pydantic import BaseModel, Field


class Grade(BaseModel):
    chunk_id: str
    relevant: bool = Field(description="True only if the passage helps answer the question")


class Grades(BaseModel):
    grades: list[Grade]


GRADER_PROMPT = """You grade search results for a GDPR question-answering system.
For each passage, decide whether it contains information that helps answer the
question. A passage on the same general topic that does not help answer this
specific question is NOT relevant. If the GDPR cannot answer the question at all,
mark every passage not relevant.

Question: {question}

Passages:
{context}"""


class Grader:
    def __init__(self, llm):
        self.chain = llm.with_structured_output(Grades)

    def grade(self, question: str, chunks: list[dict]) -> dict[str, bool]:
        """chunk_id -> relevant. Chunks the grader fails to mention count as not relevant."""
        context = "\n\n".join(f"[{c['chunk_id']}] {c['title']}\n{c['text']}" for c in chunks)
        result = self.chain.invoke(GRADER_PROMPT.format(question=question, context=context))
        verdicts = {g.chunk_id: g.relevant for g in result.grades}
        return {c["chunk_id"]: verdicts.get(c["chunk_id"], False) for c in chunks}
