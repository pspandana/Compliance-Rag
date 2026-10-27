"""Load regulatory source documents, clean them, and split them into chunks.

Output (in data/processed/):
  documents.parquet  one row per article / recital
  chunks.parquet     one row per chunk, with a stable chunk_id like "gdpr-art17-c1"

The golden set labels questions by doc_id (e.g. "gdpr-art17"), not chunk_id,
so changing chunk sizes never invalidates the labels.
"""

import argparse
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GDPR_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"
UK_GDPR_URL = "https://www.legislation.gov.uk/eur/2016/679/data.htm"

# Document-level metadata, copied onto every document and chunk of that source.
EU_METADATA = {
    "jurisdiction": "EU",
    "effective_date": "2018-05-25",  # applies from this date (Article 99)
    "version": "Regulation (EU) 2016/679, OJ L 119, 4.5.2016",
}
UK_EFFECTIVE_DATE = "2020-12-31"  # end of the Brexit transition period (IP completion day)


@dataclass
class Document:
    doc_id: str  # e.g. "gdpr-art17", "gdpr-rct65", "ukgdpr-art17"
    source: str  # "GDPR" or "UK GDPR"
    kind: str  # "article" or "recital"
    number: int
    title: str
    text: str
    jurisdiction: str = "EU"  # "EU" or "UK"
    effective_date: str = ""  # ISO date the text applies from
    version: str = ""  # which published version this text is


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    kind: str  # "article" or "recital", copied from the document
    title: str
    text: str
    jurisdiction: str = "EU"
    effective_date: str = ""
    version: str = ""


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# Point labels such as "(a)", "(65)", "(iv)" that EUR-Lex puts in their own table cell.
LABEL_RE = re.compile(r"^\([0-9a-z]+\)$")


def block_text(el) -> str:
    """Text of an element, one line per <p>.

    Inline markup (italics etc.) is joined into its paragraph rather than split onto
    its own line, and a point label is merged with the text in the next cell.
    """
    lines: list[str] = []
    label = ""
    for p in el.find_all("p"):
        line = " ".join(p.get_text().replace("\xa0", " ").split())
        if not line:
            continue
        if LABEL_RE.match(line):
            label = f"{label} {line}".strip()
            continue
        lines.append(f"{label} {line}" if label else line)
        label = ""
    return "\n".join(lines)


def parse_gdpr_html(html: str) -> list[Document]:
    """Parse the EUR-Lex HTML of Regulation (EU) 2016/679.

    EUR-Lex wraps each recital in an element with id="rct_N" and each article in
    id="art_N"; the article heading is the paragraph whose class ends in "sti-art".
    """
    soup = BeautifulSoup(html, "html.parser")
    docs: list[Document] = []

    for el in soup.find_all(id=re.compile(r"^rct_\d+$")):
        number = int(el["id"].split("_")[1])
        # Recitals are rendered as a two-cell table: "(N)" | text.
        text = re.sub(rf"^\({number}\)\s*", "", block_text(el))
        docs.append(
            Document(f"gdpr-rct{number}", "GDPR", "recital", number, f"Recital {number}", text, **EU_METADATA)
        )

    for el in soup.find_all(id=re.compile(r"^art_\d+$")):
        number = int(el["id"].split("_")[1])
        heading = el.find("p", class_=re.compile(r"sti-art$"))
        title = f"Article {number}"
        if heading:
            title += f" - {clean_text(heading.get_text(' '))}"
        # Drop the "Article N" label and heading; they live in `title`.
        for p in el.find_all("p", class_=re.compile(r"(ti|sti)-art$")):
            p.decompose()
        text = block_text(el)
        docs.append(Document(f"gdpr-art{number}", "GDPR", "article", number, title, text, **EU_METADATA))

    if not docs:
        raise ValueError(
            "No recitals or articles found. Make sure the file is the full HTML page "
            f"saved from {GDPR_URL}"
        )
    return docs


def paragraph_text(p) -> str:
    """Text of one legislation.gov.uk paragraph: "<number> <text>", with amendment markup removed."""
    for junk in p.find_all(class_=["LegCommentaryLink", "LegChangeDelimiter", "LegExtentRestriction"]):
        junk.decompose()
    parts = [p.find(class_="LegLHS"), p.find(class_="LegRHS")]
    parts = [x.get_text("") for x in parts if x is not None] or [p.get_text("")]
    text = " ".join(" ".join(parts).replace("\xa0", " ").split())
    return text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")


def parse_uk_gdpr_html(html: str, version: str = "UK GDPR, latest revised text") -> list[Document]:
    """Parse the UK GDPR from legislation.gov.uk (revised text, whole document).

    Each article is an empty anchor `section-Article-<n>` followed by a heading and one
    paragraph element per sub-paragraph, up to the next anchor or the amendment notes.
    Footnote markers ("F57"), change brackets and "U.K." extent labels are stripped;
    repealed paragraphs, which are left as "...", are dropped, and so are articles that
    are repealed entirely (the UK moved those provisions to other Acts).
    """
    soup = BeautifulSoup(html, "html.parser")

    docs: list[Document] = []
    for anchor in soup.find_all("a", id=re.compile(r"^section-Article-")):
        label = anchor["id"].removeprefix("section-Article-").rstrip(".")  # "17", "45A"
        title = f"Article {label}"
        paragraphs: list[str] = []
        for el in anchor.next_siblings:
            if not getattr(el, "name", None):
                continue
            classes = el.get("class") or []
            if "LegAnchorID" in classes or "LegAnnotations" in classes:
                break
            if el.name in ("h3", "h4"):
                group_title = el.find(class_=re.compile(r"^LegP1GroupTitle"))
                if group_title:
                    title += f" - {paragraph_text(group_title)}"
            elif el.name == "p":
                text = paragraph_text(el)
                if text.strip(". "):
                    paragraphs.append(text)
        if not paragraphs:
            continue  # omitted / fully repealed article
        number = int(re.match(r"\d+", label).group())
        docs.append(
            Document(
                f"ukgdpr-art{label}", "UK GDPR", "article", number, title, "\n".join(paragraphs),
                jurisdiction="UK", effective_date=UK_EFFECTIVE_DATE, version=version,
            )
        )

    if not docs:
        raise ValueError(f"No articles found. Make sure the file is the whole-document page {UK_GDPR_URL}")
    return docs


def chunk_document(doc: Document, max_chars: int = 1200) -> list[Chunk]:
    """Pack whole paragraphs into chunks of at most max_chars.

    A single paragraph longer than max_chars becomes its own chunk rather than
    being cut mid-sentence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n+", doc.text) if p.strip()]
    groups: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 1 > max_chars:
            groups.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        groups.append(current)

    return [
        Chunk(
            f"{doc.doc_id}-c{i}", doc.doc_id, doc.source, doc.kind, doc.title, text,
            doc.jurisdiction, doc.effective_date, doc.version,
        )
        for i, text in enumerate(groups)
    ]


def load_documents(gdpr: Path, uk_gdpr: Path) -> list[Document]:
    """Parse every available raw source (the EU GDPR is required, the UK GDPR optional)."""
    if not gdpr.exists():
        raise SystemExit(f"Missing {gdpr}\nOpen {GDPR_URL} in a browser and save the page there.")

    docs = parse_gdpr_html(gdpr.read_text(encoding="utf-8"))
    if uk_gdpr.exists():
        retrieved = datetime.fromtimestamp(uk_gdpr.stat().st_mtime).date().isoformat()
        version = f"UK GDPR, latest revised text from legislation.gov.uk (retrieved {retrieved})"
        docs += parse_uk_gdpr_html(uk_gdpr.read_text(encoding="utf-8"), version)
    else:
        print(f"Note: {uk_gdpr} not found; skipping the UK GDPR ({UK_GDPR_URL})")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gdpr", type=Path, default=RAW_DIR / "gdpr.html")
    parser.add_argument("--uk-gdpr", type=Path, default=RAW_DIR / "uk_gdpr.html")
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()

    docs = load_documents(args.gdpr, args.uk_gdpr)
    chunks = [c for d in docs for c in chunk_document(d, args.max_chars)]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([asdict(d) for d in docs]), PROCESSED_DIR / "documents.parquet")
    pq.write_table(pa.Table.from_pylist([asdict(c) for c in chunks]), PROCESSED_DIR / "chunks.parquet")

    for source in dict.fromkeys(d.source for d in docs):
        n_art = sum(d.source == source and d.kind == "article" for d in docs)
        n_rct = sum(d.source == source and d.kind == "recital" for d in docs)
        print(f"{source}: {n_art} articles, {n_rct} recitals")
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    print(f"Wrote {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
