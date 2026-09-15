"""
Phase 2 — Metadata-Filtered Hybrid Retrieval

Combines:
- Dense vector search (pgvector) with metadata predicate pushdown
- Sparse full-text search (tsvector) with same metadata filters
- Reciprocal Rank Fusion (RRF, k=60) to merge both rankings

All searches enforce jurisdiction/year/policy_type at the DB level —
never post-filter in Python.

Usage: python hybrid_retrieval.py
"""

import asyncio
import os
import math
from typing import Optional

import numpy as np
import psycopg

PG_DSN = os.getenv("PG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/risk_db")

# ── RRF Configuration ──────────────────────────────────────────
RRF_K = 60          # Reciprocal rank constant
DENSE_TOP_K = 10    # Top-K per search
SPARSE_TOP_K = 10   # Top-K per search


# ── Known: RRF Algorithm ────────────────────────────────────────
# Copy from previous project — this is well-tested.
# RRF scores: for a document at rank r in a result list,
# score = 1 / (r + K). Lower rank (higher r) → lower contribution.
# Documents appearing in BOTH dense and sparse lists score highest.


# ██ YOUR CODE HERE ██ — Task 1: RRF Function (approx 20 lines)
# Write reciprocal_rank_fusion(results_a: list[dict], results_b: list[dict], k: int = RRF_K) -> list[dict].
#
# Each result dict has an "id" field (database chunk id).
#
# Algorithm:
#   1. Create a set of all unique chunk IDs from both result lists.
#   2. For each unique ID, compute RRF score:
#        - Find its rank in results_a (1-indexed). If not present, skip its dense contribution.
#        - Find its rank in results_b (1-indexed). If not present, skip its sparse contribution.
#        - score = 1/(rank_a + k) + 1/(rank_b + k)
#        - Only include if score > 0 (i.e., it appeared in at least one list).
#   3. Sort by score descending.
#   4. Return merged list (include original dicts from whichever list had them,
#      or merge fields from both).
#
# Important: rank is 1-indexed. First result has rank 1, not 0.


def reciprocal_rank_fusion(results_a: list[dict], results_b: list[dict], k: int = RRF_K) -> list[dict]:
    # ██ YOUR CODE HERE ██
    pass


# ██ YOUR CODE HERE ██ — Task 2: Filtered Dense Search (approx 25 lines)
# Write async_filtered_dense_search(query_vector: np.ndarray, jurisdiction: str,
#     effective_year: int, policy_type: str, top_k: int = DENSE_TOP_K) -> list[dict].
#
# SQL:
#   SELECT id, document_id, page_number, clause_id, is_table, content,
#          (embedding <=> %s::vector(384)) AS distance
#   FROM policy_chunks
#   WHERE jurisdiction = %s AND effective_year = %s AND policy_type = %s
#   ORDER BY distance ASC
#   LIMIT %s;
#
# Notes:
#   - %s::vector(384) casts the parameter to vector type. psycopg handles
#     the list[float] → vector adaptation automatically.
#   - ORDER BY distance BEFORE LIMIT — the DB engine filters metadata,
#     then sorts by vector distance. This is PREDICATE PUSHDOWN.
#   - Return list of dicts with at least: id, document_id, page_number,
#     clause_id, is_table, content, score (the distance value).


async def async_filtered_dense_search(
    query_vector: np.ndarray,
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    top_k: int = DENSE_TOP_K,
) -> list[dict]:
    # ██ YOUR CODE HERE ██
    pass


# ██ YOUR CODE HERE ██ — Task 3: Filtered Sparse Search (approx 25 lines)
# Write async_filtered_sparse_search(query: str, jurisdiction: str,
#     effective_year: int, policy_type: str, top_k: int = SPARSE_TOP_K) -> list[dict].
#
# SQL:
#   SELECT id, document_id, page_number, clause_id, is_table, content,
#          ts_rank(content_tsv, plainto_tsquery('english', %s)) AS rank
#   FROM policy_chunks
#   WHERE jurisdiction = %s AND effective_year = %s AND policy_type = %s
#     AND content_tsv @@ plainto_tsquery('english', %s)
#   ORDER BY rank DESC
#   LIMIT %s;
#
# Notes:
#   - plainto_tsquery converts a plain text query to a tsquery (AND semantics).
#   - content_tsv @@ plainto_tsquery(...) — the GIN index lookup.
#   - ts_rank gives relevance score (higher = better match).
#   - The WHERE clause includes the tsvector match — sparse search returns
#     ONLY keyword-matching chunks (within metadata scope).
#   - Return same dict structure as dense search, with "score" = rank value.


async def async_filtered_sparse_search(
    query: str,
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    top_k: int = SPARSE_TOP_K,
) -> list[dict]:
    # ██ YOUR CODE HERE ██
    pass


# ██ YOUR CODE HERE ██ — Task 4: Orchestration (approx 15 lines)
# Write async retrieve_filtered(query: str, query_vector: np.ndarray,
#     jurisdiction: str, effective_year: int, policy_type: str,
#     top_k: int = DENSE_TOP_K) -> list[dict].
#
# Logic:
#   1. Run dense and sparse searches in PARALLEL (asyncio.gather)
#   2. Apply RRF to merge the two ranked lists
#   3. Return top_k merged results
#
# Why parallel? Dense and sparse are independent — no data dependency.
# asyncio.gather fires both at the same time, waits for both.


async def retrieve_filtered(
    query: str,
    query_vector: np.ndarray,
    jurisdiction: str,
    effective_year: int,
    policy_type: str,
    top_k: int = DENSE_TOP_K,
) -> list[dict]:
    # ██ YOUR CODE HERE ██
    pass


# ██ YOUR CODE HERE ██ — Task 5: Main Entry Point (approx 15 lines)
# Write async main() that:
#   1. Generates a query vector for a test query (use generate_embeddings from ingest_policies)
#   2. Calls retrieve_filtered with test parameters
#   3. Asserts that ALL results match the requested jurisdiction (jurisdiction leak test)
#   4. Prints results


async def main():
    # ██ YOUR CODE HERE ██
    pass


if __name__ == "__main__":
    asyncio.run(main())
