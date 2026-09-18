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
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional

from dotenv import load_dotenv
import numpy as np
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from ingest_policies import generate_embeddings

load_dotenv()

PG_DSN = os.getenv("PG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/risk_db")

# ── Configuration ──────────────────────────
RRF_K = 60
DENSE_TOP_K = 50
SPARSE_TOP_K = 50
NEIGHBOR_WINDOW = 1
LEGACY_SESSION_ID = "legacy"
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
            timeout=30.0,    # Wait up to 30s for a connection
            open=False,
        )
        await _pool.open()  # Open connections
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


QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "to",
    "under", "what", "when", "where", "which", "who", "why", "with",
}


def build_soft_tsquery(query: str) -> str:
    """Build an OR tsquery so long questions can match partial evidence chunks."""
    terms = []
    for term in re.findall(r"[A-Za-z][A-Za-z0-9]+", query.lower()):
        if term in QUERY_STOPWORDS or len(term) < 3:
            continue
        if term not in terms:
            terms.append(term)

    if not terms:
        return ""

    return " | ".join(terms)


# ── Filtered Dense Search ──────────────

async def async_filtered_dense_search(
    query_vector: np.ndarray,
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    session_id: str = LEGACY_SESSION_ID,
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
                                WHERE session_id = %s
                                    AND jurisdiction = %s AND effective_year = %s
                                    AND (%s = 'All' OR policy_type = %s)
                ORDER BY distance ASC
                LIMIT %s;
            """, (
                query_vector.tolist(),
                session_id,
                jurisdiction,
                effective_year,
                policy_type,
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
    session_id: str = LEGACY_SESSION_ID,
    top_k: int = SPARSE_TOP_K,
) -> list[dict]:
    pool = await get_pool()
    soft_tsquery = build_soft_tsquery(query)

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                WITH queries AS (
                    SELECT
                        plainto_tsquery('english', %(query)s) AS strict_query,
                        CASE
                            WHEN %(soft_tsquery)s = '' THEN NULL::tsquery
                            ELSE to_tsquery('english', %(soft_tsquery)s)
                        END AS soft_query
                )
                SELECT id, document_id, page_number, clause_id, is_table, content,
                       jurisdiction,
                       (
                           ts_rank(content_tsv, strict_query) * 2.0
                           + COALESCE(ts_rank(content_tsv, soft_query), 0.0)
                       ) AS rank
                FROM policy_chunks, queries
                                WHERE session_id = %(session_id)s
                                    AND jurisdiction = %(jurisdiction)s
                  AND effective_year = %(effective_year)s
                                    AND (%(policy_type)s = 'All' OR policy_type = %(policy_type)s)
                  AND (
                      content_tsv @@ strict_query
                      OR (soft_query IS NOT NULL AND content_tsv @@ soft_query)
                  )
                ORDER BY rank DESC
                LIMIT %(top_k)s;
            """, {
                "query": query,
                "soft_tsquery": soft_tsquery,
                "jurisdiction": jurisdiction,
                "effective_year": effective_year,
                "policy_type": policy_type,
                "session_id": session_id,
                "top_k": top_k,
            })
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


async def fetch_neighbor_chunks(
    candidates: list[dict],
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    session_id: str = LEGACY_SESSION_ID,
    window: int = NEIGHBOR_WINDOW,
) -> list[dict]:
    if not candidates or window < 1:
        return []

    candidate_ids = {candidate["id"] for candidate in candidates}
    neighbor_ids = sorted({
        candidate["id"] + offset
        for candidate in candidates
        for offset in range(-window, window + 1)
        if offset != 0
    })

    if not neighbor_ids:
        return []

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT id, document_id, page_number, clause_id, is_table, content,
                       jurisdiction
                FROM policy_chunks
                                WHERE id = ANY(%s)
                                    AND session_id = %s
                  AND jurisdiction = %s
                  AND effective_year = %s
                  AND (%s = 'All' OR policy_type = %s)
                ORDER BY id ASC;
            """, (
                neighbor_ids,
                session_id,
                jurisdiction,
                effective_year,
                policy_type,
                policy_type,
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
            "score": 0.0,
            "rrf_score": 0.0,
            "retrieval_source": "neighbor",
        }
        for r in rows
        if r["id"] not in candidate_ids
    ]


def dedupe_by_id(candidates: list[dict]) -> list[dict]:
    seen = set()
    deduped = []

    for candidate in candidates:
        candidate_id = candidate["id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        deduped.append(candidate)

    return deduped


# ── Orchestration ──────────────────────

async def retrieve_filtered(
    query: str,
    query_vector: np.ndarray,
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    session_id: str = LEGACY_SESSION_ID,
    top_k: int = DENSE_TOP_K,
) -> list[dict]:
    search_k = max(top_k, DENSE_TOP_K, SPARSE_TOP_K)
    dense_results, sparse_results = await asyncio.gather(
        async_filtered_dense_search(query_vector, jurisdiction, effective_year, policy_type, session_id, search_k),
        async_filtered_sparse_search(query, jurisdiction, effective_year, policy_type, session_id, search_k),
    )
    merged_results = reciprocal_rank_fusion(dense_results, sparse_results)
    focused_results = merged_results[:top_k]
    neighbor_results = await fetch_neighbor_chunks(
        candidates=focused_results,
        jurisdiction=jurisdiction,
        effective_year=effective_year,
        policy_type=policy_type,
        session_id=session_id,
    )
    return dedupe_by_id(focused_results + neighbor_results)


# ── Main Entry Point ───────────────────

async def main():
    test_query = (
        "What is the maximum liability for buildings, plants and machinery, "
        "furniture, fixtures and fittings under Section I Material Damage?"
    )
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
