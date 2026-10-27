"""Decide which jurisdiction(s) a question is about, and search only those.

The corpus holds two versions of the same regulation: the EU GDPR and the UK GDPR.
Their texts differ in places (fines, the age of consent, who regulates), so mixing
them silently would give wrong answers. The rules are deliberately simple:

  mentions the UK and the EU, or asks to compare/differ    -> both
  mentions the UK only                                      -> UK
  mentions the EU only                                      -> EU
  mentions neither                                          -> EU (assumed; the answer says so)
"""

import re
from dataclasses import dataclass
from math import ceil

UK_TERMS = re.compile(
    r"\b(uk|u\.k\.|united kingdom|britain|british|england|ico|information commissioner?|"
    r"secretary of state|data protection act 2018|dpa 2018)\b",
    re.IGNORECASE,
)
EU_TERMS = re.compile(r"\b(eu|e\.u\.|european union|european|edpb|member states?)\b", re.IGNORECASE)
COMPARE_TERMS = re.compile(r"\b(differ\w*|compar\w*|versus|vs\.?|contrast|both)\b", re.IGNORECASE)


@dataclass
class Scope:
    jurisdictions: list[str]  # subset of ["EU", "UK"]
    assumed: bool  # True when the question named no jurisdiction and we defaulted to the EU

    @property
    def label(self) -> str:
        names = " and ".join(f"{j} GDPR" for j in self.jurisdictions)
        return f"{names} (assumed: the question names no jurisdiction)" if self.assumed else names


def detect_scope(question: str) -> Scope:
    uk, eu = bool(UK_TERMS.search(question)), bool(EU_TERMS.search(question))
    compare = bool(COMPARE_TERMS.search(question))
    if (uk and eu) or (compare and (uk or eu)):
        return Scope(["EU", "UK"], assumed=False)
    if uk:
        return Scope(["UK"], assumed=False)
    return Scope(["EU"], assumed=not eu)


def scoped_search(retriever, question: str, k: int, kind: str | None = "article") -> tuple[list, Scope]:
    """Search only the jurisdiction(s) the question is about (and, optionally, one kind of document).

    For a comparison, each jurisdiction gets half of the k slots, so neither can crowd the other out.
    """
    scope = detect_scope(question)
    base = {"kind": kind} if kind else {}
    if len(scope.jurisdictions) == 1:
        return retriever.search(question, k, {**base, "jurisdiction": scope.jurisdictions[0]}), scope

    hits = []
    for jurisdiction in scope.jurisdictions:
        hits += retriever.search(question, ceil(k / 2), {**base, "jurisdiction": jurisdiction})
    return sorted(hits, key=lambda hit: hit[1], reverse=True), scope
