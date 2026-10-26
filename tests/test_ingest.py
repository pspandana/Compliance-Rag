from compliance_rag.ingest import Document, chunk_document, parse_gdpr_html

# Trimmed-down copy of the EUR-Lex markup structure.
SAMPLE_HTML = """
<html><body>
<div class="eli-subdivision" id="rct_65">
  <table><tr><td><p class="oj-normal">(65)</p></td>
  <td><p class="oj-normal">A data subject should have the right to have personal data rectified.</p></td></tr></table>
</div>
<div class="eli-subdivision" id="art_17">
  <p class="oj-ti-art">Article 17</p>
  <div class="eli-title" id="art_17.tit_1"><p class="oj-sti-art">Right to erasure (&#8216;right to be forgotten&#8217;)</p></div>
  <div id="017.001"><p class="oj-normal">1. The data subject shall have the right to obtain erasure.</p></div>
  <div id="017.002"><p class="oj-normal">2. Where the controller has made the personal data public, it shall take reasonable steps.</p></div>
  <table><tr><td><p class="oj-normal">(a)</p></td>
  <td><p class="oj-normal">published in the <span class="oj-italic">Official Journal</span>.</p></td></tr></table>
</div>
</body></html>
"""


def test_parse_gdpr_html():
    docs = {d.doc_id: d for d in parse_gdpr_html(SAMPLE_HTML)}
    assert set(docs) == {"gdpr-rct65", "gdpr-art17"}

    recital = docs["gdpr-rct65"]
    assert recital.kind == "recital"
    assert recital.text.startswith("A data subject")

    article = docs["gdpr-art17"]
    assert article.title == "Article 17 - Right to erasure (‘right to be forgotten’)"
    assert "Article 17" not in article.text
    assert article.text.startswith("1. The data subject")
    # Inline markup stays in its paragraph; point labels stay with their text.
    assert article.text.endswith("\n(a) published in the Official Journal.")


def test_chunk_document_keeps_paragraphs_whole():
    paras = [f"{i}. " + "x" * 500 for i in range(1, 5)]
    doc = Document("gdpr-art1", "GDPR", "article", 1, "Article 1", "\n".join(paras))

    chunks = chunk_document(doc, max_chars=1200)

    assert [c.chunk_id for c in chunks] == ["gdpr-art1-c0", "gdpr-art1-c1"]
    assert chunks[0].text == "\n".join(paras[:2])
    assert all(c.doc_id == "gdpr-art1" for c in chunks)
