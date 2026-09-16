"""
Phase 2 — Metadata-Filtered Hybrid Retrieval (with Connection Pooling)

Combines:
- Dense vector search (pgvector) with metadata predicate pushdown
- Sparse full-text search (tsvector) with same metadata filters
- Reciprocal Rank Fusion (RRF, k=60) to merge both rankings

All searches enforce jurisdiction/year/policy_type at the DB level.
Uses AsyncConnectionPool — no per-request connection overhead,
no event loop blocking, no connection exhaustion at scale.

Usage: python hybrid_retrieval.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional

import numpy as np
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from ingest_policies import generate_embeddings

PG_DSN = os.getenv("PG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/risk_db")

# ── Configuration ──────────────────────────
RRF_K = 60
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
POOL_MIN = 2
POOL_MAX = 10

# Module-level pool (singleton per process)
_pool: Optional[AsyncConnectionPool] = None


async def get_pool() -> AsyncConnectionPool:
    """Lazily initialize and return the async connection pool."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=PG_DSN,
            min_size=POOL_MIN,
            max_size=POOL_MAX,
            open=True,       # Open connections immediately
            timeout=30.0,    # Wait up to 30s for a connection
        )
        await _pool.wait()  # Block until pool is ready
    return _pool


async def close_pool() -> None:
    """Shutdown the pool gracefully."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ── RRF Function ───────────────────────

def reciprocal_rank_fusion(results_a: list[dict], results_b: list[dict], k: int = RRF_K) -> list[dict]:
    rrf_scores: dict[int, float] = {}
    docs_lookup: dict[int, dict] = {}

    for rank, item in enumerate(results_a, start=1):
        d_id = item["id"]
        rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (k + rank))
        docs_lookup[d_id] = item

    for rank, item in enumerate(results_b, start=1):
        d_id = item["id"]
        rrf_scores[d_id] = rrf_scores.get(d_id, 0.0) + (1.0 / (k + rank))
        docs_lookup[d_id] = item

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    fused = []
    for d_id in sorted_ids:
        entry = dict(docs_lookup[d_id])
        entry["rrf_score"] = rrf_scores[d_id]
        fused.append(entry)
    return fused


# ── Filtered Dense Search ──────────────

async def async_filtered_dense_search(
    query_vector: np.ndarray,
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    top_k: int = DENSE_TOP_K,
) -> list[dict]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, document_id, page_number, clause_id, is_table, content,
                       jurisdiction,
                       (embedding <=> %s::vector(384)) AS distance
                FROM policy_chunks
                WHERE jurisdiction = %s AND effective_year = %s AND policy_type = %s
                ORDER BY distance ASC
                LIMIT %s;
            """, (
                query_vector.tolist(),
                jurisdiction,
                effective_year,
                policy_type,
                top_k,
            ))
            rows = await cur.fetchall()
        return [
            {
                "id": r["id"],
                "document_id": r["document_id"],
                "page_number": r["page_number"],
                "clause_id": r["clause_id"],
                "is_table": r["is_table"],
                "content": r["content"],
                "jurisdiction": r["jurisdiction"],
                "score": float(r["distance"]),
            }
            for r in rows
        ]


# ── Filtered Sparse Search ─────────────────────

async def async_filtered_sparse_search(
    query: str,
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    top_k: int = SPARSE_TOP_K,
) -> list[dict]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, document_id, page_number, clause_id, is_table, content,
                       jurisdiction,
                       ts_rank(content_tsv, plainto_tsquery('english', %s)) AS rank
                FROM policy_chunks
                WHERE jurisdiction = %s AND effective_year = %s AND policy_type = %s
                  AND content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s;
            """, (
                query,
                jurisdiction,
                effective_year,
                policy_type,
                query,
                top_k,
            ))
            rows = await cur.fetchall()
        return [
            {
                "id": r["id"],
                "document_id": r["document_id"],
                "page_number": r["page_number"],
                "clause_id": r["clause_id"],
                "is_table": r["is_table"],
                "content": r["content"],
                "jurisdiction": r["jurisdiction"],
                "score": float(r["rank"]),
            }
            for r in rows
        ]


# ── Orchestration ──────────────────────

async def retrieve_filtered(
    query: str,
    query_vector: np.ndarray,
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    top_k: int = DENSE_TOP_K,
) -> list[dict]:
    dense_results, sparse_results = await asyncio.gather(
        async_filtered_dense_search(query_vector, jurisdiction, effective_year, policy_type, top_k),
        async_filtered_sparse_search(query, jurisdiction, effective_year, policy_type, top_k),
    )
    merged_results = reciprocal_rank_fusion(dense_results, sparse_results)
    return merged_results[:top_k]


# ── Main Entry Point ───────────────────

async def main():
    test_query = "What is the maximum payout for a cyber liability claim?"
    jurisdiction = "Unknown"
    effective_year = 2024
    policy_type = "Unknown"

    print(f"Query: {test_query}")
    print(f"Filters: jurisdiction={jurisdiction}, year={effective_year}, type={policy_type}")
    print()

    query_vector = generate_embeddings([test_query])[0]

    results = await retrieve_filtered(
        query=test_query,
        query_vector=query_vector,
        jurisdiction=jurisdiction,
        effective_year=effective_year,
        policy_type=policy_type,
        top_k=DENSE_TOP_K,
    )

    assert all(r["jurisdiction"] == jurisdiction for r in results), \
        f"Jurisdiction leak! Expected {jurisdiction}, got {[r['jurisdiction'] for r in results]}"

    print(f"Retrieved {len(results)} results — jurisdiction check passed ✓")
    for r in results:
        print(f"  ID={r['id']} | Page={r['page_number']} | Dist={r['score']:.4f} | RRF={r.get('rrf_score', 0):.6f}")
        preview = r['content'][:100].replace('\n', ' ')
        print(f"    {preview}...")

    await close_pool()
    print("\n✓ Pool closed — clean shutdown")


if __name__ == "__main__":
    asyncio.run(main())
