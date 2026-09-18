"""
Phase 1 — PDF Ingestion Pipeline

Reads insurance policy PDFs from raw_policies/, performs layout-aware
parsing, serializes tables to Markdown, applies sliding-window chunking
to prose, generates embeddings, and batch-inserts into policy_chunks.

Usage: python 02_ingest_policies.py
"""

import asyncio
import os
import re
import glob
from typing import Optional

from dotenv import load_dotenv
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg

load_dotenv()

from init_db import init_db 

# ── Imports from Phase 1 ───────────────────────────────────────────
# from _shared import PG_DSN  # if you extract config

# ── Configuration ──────────────────────────────────────────────────

RAW_POLICIES_DIR = os.getenv("RAW_POLICIES_DIR", "raw_policies")
CHUNK_WORDS = 120
CHUNK_OVERLAP = 40
EMBED_BATCH_SIZE = 32
PG_DSN = os.getenv("PG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/risk_db")
LEGACY_SESSION_ID = "legacy"

# ── Clause Header Extraction ───────────────────────────────────────

CLAUSE_PATTERN = re.compile(
    r'('
    r'Section\s+(?:\d+[\.\d]*[a-z]?\(?[a-z]?\)?|[IVXLCDM]+)(?:\s*[-–]\s*[A-Z][A-Z\s&/,()-]+)?'
    r'|Article\s+(?:\d+|[IVXLCDM]+)'
    r'|Clause\s+\d+[\.\d]*'
    r'|Exclusions?'
    r')',
    re.IGNORECASE,
)


def extract_clause_id(text: str) -> Optional[str]:
    """Extract clause/section ID from the first 200 chars of a chunk.

    Why 200 chars? Insurance clause headers always appear at the start
    of a section. Scanning the full chunk is noisy — clause-like patterns
    ('Section 3.2' in body text) cause false positives.
    """
    header = text[:200]
    match = CLAUSE_PATTERN.search(header)
    return match.group(0) if match else None



def serialize_table_to_markdown(table: list[list[str]]) -> str:
    if not table or not table[0]:
        return ""

    table = [[cell or "" for cell in row] for row in table]
    width = len(table[0])
    table = [row[:width] + [""] * max(0, width - len(row)) for row in table]

    result = "| " + " | ".join(table[0]) + " |\n"
    result += "|" + "---|" * len(table[0]) + "\n"

    for row in table[1:]:
        result += "| " + " | ".join(row) + " |\n"

    return result
    



def extract_from_page(page, page_number: int) -> list[dict]:
    results = []

    tables = page.extract_tables()

    for table in tables:
        content = serialize_table_to_markdown(table)

        if content:
            results.append({
                "content": content, 
                "page_number": page_number,
                "is_table": True,
                "clause_id": f"Page {page_number} Table", 
                "source_hint": f"Table on page {page_number}",
            })

    text = page.extract_text()

    if not text:
        return results

    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP

    if len(words) <= CHUNK_WORDS:
        results.append({
            "content": " ".join(words),
            "page_number": page_number,
            "is_table": False,
            "clause_id": extract_clause_id(" ".join(words)) or f"Page {page_number}",
            "source_hint":f"Text on page {page_number}",
        })

    else:
        for start in range(0, len(words), step):
            chunk_words = words[start:start+CHUNK_WORDS]
            if not chunk_words:
                break
            chunk = " ".join(chunk_words)
        
            results.append({
                "content": chunk,
                "page_number": page_number,
                "is_table": False,
                "clause_id": extract_clause_id(chunk) or f"Page {page_number}",
                "source_hint":f"Text on page {page_number}",
            })
            if start + CHUNK_WORDS >= len(words):
                break

    return results




# ── Known Boilerplate: Embedding Generation ─────────────────────────

_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embeddings(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


# ── Known Boilerplate: Database Batch Insert ────────────────────────

def ensure_session_schema() -> None:
    """Add upload scoping to databases created before the upload workflow."""
    conn = psycopg.connect(PG_DSN, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE policy_chunks
                ADD COLUMN IF NOT EXISTS session_id TEXT
            """)
            cur.execute("""
                UPDATE policy_chunks
                SET session_id = %s
                WHERE session_id IS NULL
            """, (LEGACY_SESSION_ID,))
            cur.execute("""
                ALTER TABLE policy_chunks
                ALTER COLUMN session_id SET DEFAULT 'legacy'
            """)
            cur.execute("""
                UPDATE policy_chunks
                SET session_id = 'legacy'
                WHERE session_id IS NULL
            """)
            cur.execute("""
                ALTER TABLE policy_chunks
                ALTER COLUMN session_id SET NOT NULL
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_policy_chunks_session
                ON policy_chunks (session_id)
            """)
    finally:
        conn.close()


async def batch_insert_chunks(chunks: list[dict]) -> None:
    """Bulk insert parsed chunks into policy_chunks."""
    conn = psycopg.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO policy_chunks
                    (session_id, document_id, policy_type, jurisdiction, effective_year,
                     page_number, clause_id, is_table, content, embedding)
                VALUES
                    (%(session_id)s, %(document_id)s, %(policy_type)s, %(jurisdiction)s, %(effective_year)s,
                     %(page_number)s, %(clause_id)s, %(is_table)s, %(content)s, %(embedding)s)
                """,
                chunks,
            )
        conn.commit()
    finally:
        conn.close()


# ── Known Boilerplate: PDF Discovery ────────────────────────────────

def discover_pdfs(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "*.pdf")))



async def ingest_single_pdf(
    filepath: str,
    session_id: str = LEGACY_SESSION_ID,
    policy_type: str = "Unknown",
    document_title: Optional[str] = None,
) -> dict:
    ensure_session_schema()
    with pdfplumber.open(filepath) as pdf:
        page_count = len(pdf.pages)
        chunks = []

        for page_number, page in enumerate(pdf.pages, start=1):
            page_chunks = extract_from_page(page, page_number)
            chunks.extend(page_chunks)

    document_id = os.path.basename(filepath)

    for chunk in chunks:
        chunk["session_id"] = session_id
        chunk["document_id"] = document_id
        chunk["policy_type"] = policy_type
        chunk["jurisdiction"] = "Unknown"
        chunk["effective_year"] = 2024

    contents = [chunk["content"] for chunk in chunks]
    embeddings = generate_embeddings(contents)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    await batch_insert_chunks(chunks)

    tables = sum(1 for chunk in chunks if chunk["is_table"])

    return {
        "file": filepath,
        "document_id": document_id,
        "title": document_title or document_id,
        "pages": page_count,
        "chunks": len(chunks),
        "tables":  tables,
    }



async def main():
    pdf_files = discover_pdfs(RAW_POLICIES_DIR)

    init_db()

    for filepath in pdf_files:
        summary = await ingest_single_pdf(filepath)
        print(
            f"Processed {summary['file']} | "
            f"Chunks: {summary['chunks']} | "
            f"Tables: {summary['tables']}"
        )



if __name__ == "__main__":
    asyncio.run(main())
