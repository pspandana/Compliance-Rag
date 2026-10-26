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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GDPR_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"


@dataclass
class Document:
    doc_id: str  # e.g. "gdpr-art17", "gdpr-rct65"
    source: str  # e.g. "GDPR"
    kind: str  # "article" or "recital"
    number: int
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    kind: str  # "article" or "recital", copied from the document
    title: str
    text: str


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
        docs.append(Document(f"gdpr-rct{number}", "GDPR", "recital", number, f"Recital {number}", text))

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
        docs.append(Document(f"gdpr-art{number}", "GDPR", "article", number, title, text))

    if not docs:
        raise ValueError(
            "No recitals or articles found. Make sure the file is the full HTML page "
            f"saved from {GDPR_URL}"
        )
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
        Chunk(f"{doc.doc_id}-c{i}", doc.doc_id, doc.source, doc.kind, doc.title, text)
        for i, text in enumerate(groups)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gdpr", type=Path, default=RAW_DIR / "gdpr.html")
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()

    if not args.gdpr.exists():
        raise SystemExit(f"Missing {args.gdpr}\nOpen {GDPR_URL} in a browser and save the page there.")

    docs = parse_gdpr_html(args.gdpr.read_text(encoding="utf-8"))
    chunks = [c for d in docs for c in chunk_document(d, args.max_chars)]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([asdict(d) for d in docs]), PROCESSED_DIR / "documents.parquet")
    pq.write_table(pa.Table.from_pylist([asdict(c) for c in chunks]), PROCESSED_DIR / "chunks.parquet")

    n_art = sum(d.kind == "article" for d in docs)
    n_rct = sum(d.kind == "recital" for d in docs)
    print(f"{len(docs)} documents ({n_art} articles, {n_rct} recitals) -> {len(chunks)} chunks")
    print(f"Wrote {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
