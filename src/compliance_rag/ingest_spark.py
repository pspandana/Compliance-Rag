"""PySpark ingestion: clean, deduplicate, link versions, chunk and tag the regulatory documents.

Same inputs and outputs as `ingest` (data/processed/documents.parquet and chunks.parquet),
built as Spark DataFrame stages:

  1. parse     the raw HTML pages become Document records (BeautifulSoup, on the driver)
  2. clean     normalise whitespace in text and titles
  3. tag       validate the metadata (jurisdiction, effective date, version) on every record
  4. dedupe    drop records whose normalised text repeats within the same jurisdiction
  5. link      record which other-jurisdiction documents have word-for-word the same text
               (`same_text_as`, e.g. an EU and a UK article that did not change)
  6. chunk     split each document into chunks (the same packing rules as `ingest`)

Spark runs in local mode. The data is small, so this shows the pipeline, not scale. Results
are checked against the plain-Python pipeline: for the same input they must be identical.

Needs Java 17+ and the `spark` dependency group:  uv run --group spark spark-ingest
"""

import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from compliance_rag.ingest import (
    PROCESSED_DIR,
    RAW_DIR,
    Document,
    chunk_document,
    load_documents,
)

VALID_JURISDICTIONS = ("EU", "UK")
DOC_FIELDS = [f.name for f in fields(Document)]


def ensure_java() -> None:
    """Make sure Spark can find Java: JAVA_HOME, PATH, or a JDK unzipped under %LOCALAPPDATA%."""
    if os.environ.get("JAVA_HOME") or shutil.which("java"):
        return
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "compliance-rag" / "jdk17"
    jdks = sorted(local.glob("jdk-*"))
    if not jdks:
        raise SystemExit("Java 17+ not found. Install a JDK, or set JAVA_HOME, then run again.")
    os.environ["JAVA_HOME"] = str(jdks[-1])
    os.environ["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ["PATH"]


def get_spark():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise SystemExit("PySpark is not installed. Run:  uv run --group spark spark-ingest") from None

    ensure_java()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)  # workers use this virtual environment
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("compliance-rag-ingest")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.ansi.enabled", "false")  # malformed dates become NULL, which we then catch
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def chunk_row(row: dict, max_chars: int) -> list[dict]:
    """Chunk one document row. Module-level so Spark workers can import it."""
    doc = Document(**{name: row[name] for name in DOC_FIELDS})
    return [
        {**asdict(chunk), "same_text_as": row["same_text_as"], "_ord": row["_ord"]}
        for chunk in chunk_document(doc, max_chars)
    ]


def transform(spark, docs: list[Document], max_chars: int = 1200) -> tuple[list[dict], list[dict], dict]:
    """Run the Spark stages. Returns (documents, chunks, stats) in the original document order."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    schema = StructType(
        [StructField(f.name, IntegerType() if f.type is int else StringType(), False) for f in fields(Document)]
        + [StructField("_ord", IntegerType(), False)]
    )
    rows = [{**asdict(d), "_ord": i} for i, d in enumerate(docs)]
    raw = spark.createDataFrame(rows, schema=schema)

    # 2. clean (paragraph breaks, "\n", are kept: the chunker packs whole paragraphs)
    cleaned = raw.withColumn(
        "text", F.trim(F.regexp_replace(F.regexp_replace("text", " ", " "), "[ \t]+", " "))
    ).withColumn("title", F.trim(F.regexp_replace("title", r"\s+", " ")))

    # 3. tag: every record must carry valid metadata
    invalid = cleaned.filter(
        ~F.col("jurisdiction").isin(*VALID_JURISDICTIONS)
        | F.to_date("effective_date", "yyyy-MM-dd").isNull()
        | (F.length("version") == 0)
        | (F.length("text") == 0)
    )
    if invalid.count():
        bad = [r["doc_id"] for r in invalid.select("doc_id").limit(5).collect()]
        raise SystemExit(f"Records with missing or invalid metadata or text, e.g. {bad}")

    # 4. dedupe: same normalised text in the same jurisdiction, keep the first doc_id
    hashed = cleaned.withColumn("text_hash", F.sha2(F.lower(F.regexp_replace("text", r"\s+", " ")), 256))
    ranked = hashed.withColumn(
        "_rn", F.row_number().over(Window.partitionBy("jurisdiction", "text_hash").orderBy("doc_id"))
    )
    removed = [r["doc_id"] for r in ranked.filter("_rn > 1").select("doc_id").collect()]
    deduped = ranked.filter("_rn = 1").drop("_rn")

    # 5. link: documents whose text is identical in the other jurisdiction
    other = deduped.select(
        F.col("text_hash").alias("_h"), F.col("jurisdiction").alias("_j"), F.col("doc_id").alias("_other")
    )
    linked = (
        deduped.join(other, (deduped.text_hash == other._h) & (deduped.jurisdiction != other._j), "left")
        .groupBy(*deduped.columns)
        .agg(F.concat_ws(",", F.sort_array(F.collect_set("_other"))).alias("same_text_as"))
    ).cache()

    documents = sorted((r.asDict() for r in linked.collect()), key=lambda d: d["_ord"])

    # 6. chunk, in parallel across documents
    chunks = linked.rdd.flatMap(lambda r: chunk_row(r.asDict(), max_chars)).collect()
    chunks.sort(key=lambda c: (c["_ord"], int(c["chunk_id"].rsplit("-c", 1)[1])))

    stats = {
        "documents_in": len(docs),
        "duplicates_removed": removed,
        "documents_out": len(documents),
        "identical_in_other_jurisdiction": sum(bool(d["same_text_as"]) for d in documents),
        "chunks": len(chunks),
    }
    for record in (*documents, *chunks):
        record.pop("_ord")
    return documents, chunks, stats


def matches_plain_python(docs: list[Document], chunks: list[dict], max_chars: int) -> int:
    """Number of chunks that differ from the plain-Python pipeline (0 means identical)."""
    reference = [asdict(c) for d in docs for c in chunk_document(d, max_chars)]
    ours = [{k: v for k, v in c.items() if k != "same_text_as"} for c in chunks]
    return sum(a != b for a, b in zip(reference, ours, strict=False)) + abs(len(reference) - len(ours))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gdpr", type=Path, default=RAW_DIR / "gdpr.html")
    parser.add_argument("--uk-gdpr", type=Path, default=RAW_DIR / "uk_gdpr.html")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    docs = load_documents(args.gdpr, args.uk_gdpr)
    spark = get_spark()
    start = time.time()
    try:
        documents, chunks, stats = transform(spark, docs, args.max_chars)
    finally:
        spark.stop()

    args.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(documents), args.out / "documents.parquet")
    pq.write_table(pa.Table.from_pylist(chunks), args.out / "chunks.parquet")

    print(f"{stats['documents_in']} documents in -> {stats['documents_out']} out, "
          f"{len(stats['duplicates_removed'])} duplicates removed {stats['duplicates_removed'][:5] or ''}")
    print(f"{stats['identical_in_other_jurisdiction']} documents have word-for-word identical text in the other jurisdiction")
    print(f"{stats['chunks']} chunks, Spark stages took {time.time() - start:.0f}s")
    differing = matches_plain_python(docs, chunks, args.max_chars) if not stats["duplicates_removed"] else None
    if differing is None:
        print("Parity check skipped (duplicates were removed, so the outputs differ by design)")
    else:
        print("Parity with the plain-Python pipeline: " + ("identical" if differing == 0 else f"{differing} chunks differ"))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
