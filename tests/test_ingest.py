from compliance_rag.ingest import (
    Document,
    chunk_document,
    parse_gdpr_html,
    parse_uk_gdpr_html,
)

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


# Trimmed-down copy of the legislation.gov.uk markup: amendment links, change brackets,
# an extent label, a repealed paragraph, the amendment notes, and a fully repealed article.
UK_SAMPLE_HTML = """
<html><body>
<a class="LegAnchorID" id="section-Article-8"></a>
<h3 class="LegClearFix LegP1Container"><span class="LegDS LegP1No">Article 8<span class="LegExtentRestriction">U.K.</span></span><span class="LegDS LegP1GroupTitle">Child's consent</span></h3>
<p class="LegClearFix LegP2Container"><span class="LegDS LegLHS LegP2No">1.</span><span class="LegDS LegRHS LegP2Text">the child is at least <span class="LegChangeDelimiter">[</span><a class="LegCommentaryLink">F57</a><span class="LegSubstitution">13 years old</span><span class="LegChangeDelimiter">]</span>.</span></p>
<p class="LegRHS LegP2Text"><a class="LegCommentaryLink">F59</a>...</p>
<div class="LegAnnotations">Textual Amendments F57 Words substituted</div>
<a class="LegAnchorID" id="section-Article-9"></a>
<h3 class="LegClearFix LegP1Container"><span class="LegDS LegP1No">Article 9</span><span class="LegDS LegP1GroupTitle">Repealed</span></h3>
<p class="LegRHS LegP1Text"><a class="LegCommentaryLink">F60</a> . . . . .</p>
</body></html>
"""


def test_parse_uk_gdpr_html_strips_amendment_markup_and_skips_repealed():
    docs = parse_uk_gdpr_html(UK_SAMPLE_HTML)

    assert [d.doc_id for d in docs] == ["ukgdpr-art8"]
    doc = docs[0]
    assert doc.title == "Article 8 - Child's consent"
    assert doc.text == "1. the child is at least 13 years old."
    assert (doc.source, doc.jurisdiction, doc.kind) == ("UK GDPR", "UK", "article")


def test_chunks_carry_document_metadata():
    doc = parse_uk_gdpr_html(UK_SAMPLE_HTML)[0]
    chunk = chunk_document(doc)[0]
    assert (chunk.jurisdiction, chunk.effective_date) == ("UK", doc.effective_date)
