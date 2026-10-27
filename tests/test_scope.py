import json

import pytest

from compliance_rag.evaluate import GOLDEN_PATH, JURISDICTION_PATH, load_golden
from compliance_rag.scope import detect_scope


@pytest.mark.parametrize(
    ("question", "jurisdictions", "assumed"),
    [
        ("Can I ask a company to delete my data?", ["EU"], True),
        ("Does a company outside the EU need a representative in the Union?", ["EU"], False),
        ("What is the maximum fine in the UK?", ["UK"], False),
        ("Does the ICO have to be told about a breach?", ["UK"], False),
        ("How does the child consent age differ between the EU and the UK?", ["EU", "UK"], False),
        ("How do the rules differ in the United Kingdom?", ["EU", "UK"], False),
        # "between" alone is not a comparison without a named jurisdiction
        ("How can a group transfer data between its own companies?", ["EU"], True),
    ],
)
def test_detect_scope(question, jurisdictions, assumed):
    scope = detect_scope(question)
    assert (scope.jurisdictions, scope.assumed) == (jurisdictions, assumed)


def test_every_golden_question_is_scoped_as_labelled():
    for q in load_golden(JURISDICTION_PATH):
        assert detect_scope(q["question"]).jurisdictions == q["scope"], q["id"]
    for q in load_golden(GOLDEN_PATH):
        assert detect_scope(q["question"]).jurisdictions == ["EU"], q["id"]


def test_golden_files_are_wellformed():
    ids = [q["id"] for path in (GOLDEN_PATH, JURISDICTION_PATH) for q in load_golden(path)]
    assert len(ids) == len(set(ids))
    assert all(json.dumps(q) for q in load_golden(JURISDICTION_PATH))
