"""
Phase 1 — Database Schema & Index Initialization

Creates the policy_chunks table with:
- Metadata columns for predicate pushdown (jurisdiction, year, policy_type)
- pgvector HNSW index for dense similarity search
- tsvector GIN index for sparse full-text search
- Generated content_tsv column for full-text indexing

Usage: python 01_init_db.py
"""

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

# ── Connection Configuration ──────────────────────────────────────────

PG_DSN = os.getenv("PG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/risk_db")


def init_db() -> None:
    """Create extensions, table, and indexes if they don't exist."""
    conn = psycopg.connect(PG_DSN, autocommit=True)

    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Enable full-text search extension (usually already available)
        cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

        # Drop existing table for clean initialization (dev only — use migrations in prod)
        cur.execute("DROP TABLE IF EXISTS policy_chunks")

        # Create the policy_chunks table
        cur.execute("""
            CREATE TABLE policy_chunks (
                id             BIGSERIAL PRIMARY KEY,
                document_id    TEXT NOT NULL,
                policy_type    TEXT NOT NULL,
                jurisdiction   TEXT NOT NULL,
                effective_year INT NOT NULL,
                page_number    INT NOT NULL,
                clause_id      TEXT,
                is_table       BOOLEAN NOT NULL DEFAULT FALSE,
                content        TEXT NOT NULL,
                content_tsv    tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', content)
                ) STORED,
                embedding      vector(384)
            )
        """)

        # pgvector HNSW index on embedding (cosine distance)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_policy_chunks_embedding
            ON policy_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200)
        """)

        # GIN index on tsvector for sparse full-text search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_policy_chunks_tsv
            ON policy_chunks
            USING GIN (content_tsv)
        """)

        # Composite index for metadata-filtered queries (Phase 2 predicate pushdown)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_policy_chunks_metadata
            ON policy_chunks (jurisdiction, effective_year, policy_type)
        """)

    conn.close()
    print("✓ Database initialized — policy_chunks table with HNSW + GIN indexes ready")


if __name__ == "__main__":
    init_db()
