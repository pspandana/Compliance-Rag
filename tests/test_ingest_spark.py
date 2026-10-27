import pytest

from compliance_rag.ingest import Document

pytest.importorskip("pyspark", reason="needs the spark dependency group")

from compliance_rag import ingest_spark


@pytest.fixture(scope="module")
def spark():
    try:
        session = ingest_spark.get_spark()
    except SystemExit as exc:  # no Java on this machine
        pytest.skip(str(exc))
    yield session
    session.stop()


def doc(doc_id, text, jurisdiction="EU", date="2018-05-25"):
    return Document(
        doc_id, "GDPR", "article", 1, "Article 1", text,
        jurisdiction=jurisdiction, effective_date=date, version="v1",
    )  # fmt: skip


def test_dedupes_within_a_jurisdiction_and_links_across_jurisdictions(spark):
    docs = [
        doc("gdpr-art1", "1. Same  text.\n2. More."),
        doc("gdpr-art2", "1. same text.\n2. more."),  # same after whitespace/case normalisation
        doc("ukgdpr-art1", "1. Same text.\n2. More.", jurisdiction="UK", date="2020-12-31"),
        doc("ukgdpr-art2", "1. Different."),
    ]
    documents, chunks, stats = ingest_spark.transform(spark, docs)

    assert stats["duplicates_removed"] == ["gdpr-art2"]
    assert [d["doc_id"] for d in documents] == ["gdpr-art1", "ukgdpr-art1", "ukgdpr-art2"]
    links = {d["doc_id"]: d["same_text_as"] for d in documents}
    assert links == {"gdpr-art1": "ukgdpr-art1", "ukgdpr-art1": "gdpr-art1", "ukgdpr-art2": ""}
    assert {c["doc_id"] for c in chunks} == {"gdpr-art1", "ukgdpr-art1", "ukgdpr-art2"}


def test_matches_the_plain_python_pipeline_when_nothing_is_removed(spark):
    docs = [doc("gdpr-art1", "1. " + "x" * 700 + "\n2. " + "y" * 700), doc("gdpr-art2", "1. Short.")]
    _, chunks, stats = ingest_spark.transform(spark, docs)

    assert not stats["duplicates_removed"]
    assert ingest_spark.matches_plain_python(docs, chunks, 1200) == 0


def test_rejects_records_with_invalid_metadata(spark):
    with pytest.raises(SystemExit):
        ingest_spark.transform(spark, [doc("gdpr-art1", "1. Text.", date="not-a-date")])
