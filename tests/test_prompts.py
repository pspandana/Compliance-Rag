import pytest

from compliance_rag.answer import GENERATOR_VARIANTS, Draft, DraftComplete

PLACEHOLDERS = ("{scope}", "{feedback}", "{context}", "{question}")


@pytest.mark.parametrize("name", list(GENERATOR_VARIANTS))
def test_every_prompt_variant_has_the_placeholders_it_is_formatted_with(name):
    prompt, schema = GENERATOR_VARIANTS[name]
    assert all(p in prompt for p in PLACEHOLDERS)
    assert issubclass(schema, Draft)
    prompt.format(scope="EU GDPR", feedback="", context="[x] text", question="Q?")


def test_complete_variants_differ_from_the_default_only_where_intended():
    default, default_schema = GENERATOR_VARIANTS["v0"]
    complete, complete_schema = GENERATOR_VARIANTS["v1"]
    restrained, _ = GENERATOR_VARIANTS["v2"]

    assert default_schema is Draft and complete_schema is DraftComplete
    assert "Be complete" in complete and "Be complete" not in default
    assert "State only what the passages say" in restrained and "State only" not in complete
