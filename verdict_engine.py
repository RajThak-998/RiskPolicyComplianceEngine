"""
Phase 3 — Cross-Encoder Reranker & Deterministic Citation Schema

Takes RRF top-15 candidates, reranks with cross-encoder, sends to an LLM
that is forced to output structured citations via Pydantic, then verifies
every citation against actual chunk content (anti-hallucination).

Usage: python verdict_engine.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sentence_transformers import CrossEncoder 
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

from hybrid_retrieval import retrieve_filtered, reciprocal_rank_fusion
from ingest_policies import generate_embeddings

from typing import Literal, Optional
import re

PG_DSN = os.getenv("PG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/risk_db")

# ── Model Loading ──────────────────────────

REASONER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL = "all-MiniLM-L6-v2"


class Citation(BaseModel):
    document_id: str
    page_number: int
    clause_id: str
    exact_quote: str

    @field_validator("exact_quote")
    @classmethod
    def validate_exact_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("exact_quoute must not be empty")
        return value


class AuditVerdict(BaseModel):
    status: Literal["COVERED", "EXCLUDED", "CONDITIONAL", "OUT_OF_SCOPE"]
    financial_limit: Optional[str] = None
    applicable_deductible: Optional[str] = None
    resoning: str
    citations: list[Citation]

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, value: list[Citation]) -> list[Citation]:
        if len(value)<1:
            raise ValueError("At least one citation is required")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: Literal["COVERED", "EXCLUDED", "CONDITIONAL", "OUT_OF_SCOPE"]) -> str:
        valid_statuses = {"COVERED", "EXCLUDED", "CONDITIONAL", "OUT_OF_SCOPE"}
        if value not in valid_statuses:
            raise ValueError(f"Invalid Status {value}")

        return value

    @field_validator("financial_limit")
    @classmethod
    def validate_financial_limit(cls, value: Optional[str]) -> Optional[str]:
        if value in None:
            return value

        pattern = r"^\$?[\d,]+(\.\d{2})?$"

        if not re.fullmatch(pattern, value):
            raise ValueError(
                "financial_limit must be a valid monetary amount"
                "such as '$5,000,000'"
            )

        return value


# =====================================================================
# TASK 2: Cross-Encoder Reranking (Write this yourself)
# =====================================================================
# Write rerank_candidates(candidates: list[dict], query: str,
#     top_n: int = 3) -> list[dict].
#
# Algorithm:
#   1. Load the cross-encoder model (REASONER_MODEL) — cache it globally.
#   Use a module-level singleton pattern like _model: Optional[CrossEncoder] = None
#   2. For each candidate, build a pair: (query_text, candidate["content"])
#   3. model.predict(pairs) returns relevance scores (higher = better)
#   4. Attach score to each candidate dict
#   5. Sort by score descending, return top_n
#
# Important: Cross-Encoder inference is CPU-bound and SLOW.
# Wrap it in asyncio.to_thread() or run_in_executor() to avoid blocking
# the event loop. See how batch_insert_chunks avoided blocking — same idea.
# Edge case: candidates list could be empty. Return [].

_model: Optional[CrossEncoder] = None

def _get_model() -> CrossEncoder:
    """Load and Cache the cross-encoder model."""
    global _model

    if _model is None:
        _model = CrossEncoder(REASONER_MODEL)

    return _model


async def rerank_candidates(candidates: list[dict], query: str, top_n: int = 3) -> list[dict]:
    """
    Rerank candidate documents using a cross-encoder.
    Higher cross-encoder scores indicate greater relevance.
    """
    if not candidates:
        return []

    pairs = [(query, candidate["content"]) for candidate in candidates]

    model = _get_model()

    scores = await asyncio.to_thread(model.predict, pairs)

    scored_candidates = []

    for candidate, score in zip(candidates, scores):
        candidate_with_score = {
            **candidate,
            "score": float(score)
        }

        scored_candidates.append(candidate_with_score)

    scored_candidates.sort(
        key= lambda candidate: candidate["score"],
        reverse=True
    )

    return scored_candidates[:top_n]



def build_grounded_prompt(query: str, chunks: list[dict]) -> str:
    source_blocks = []

    for chunk in chunks:
        source_block = (
            f"[Source: {chunk['document_id']} | "
            f"Page: {chunk['page_number']} | "
            f"Clause: {chunk['clause_id']}]\n"
            f"{chunk['content']}"
        )

        source_blocks.append(source_block)

    source_blocks_joined = "\n\n".join(source_blocks)

    prompt = f"""You are a policy compliance auditor. Answer strictly from the provided sources only. If the answer 
    cannot be found in the sources, respond with status 'OUT_OF_SCOPE'.
    
    SOURCES: {source_blocks_joined}
    QUERY: {query}

    Respond with a JSON object matching this exact schema:
    {{
        "status": "COVERED" | "EXCLUDED" | "CONDITIONAL" | "OUT_OF_SCOPE",
        "financial_limit": "<string or null>",
        "applicable_deductible": "<string or null>",
        "reasoning": "<cite which source and clause supports your conclusion>",
        "citations": [
            {{"document_id": "...", "page_number": <int>, "clause_id": "...", "exact_quote": "..."}}
        ]
    }}"""
    return prompt

# =====================================================================
# TASK 4: LLM Streaming Function (NEW CONCEPT — Explain yourself)
# =====================================================================
# Write async stream_llm_verdict(prompt: str) -> AuditVerdict.
#
# This function:
#   1. Calls Groq API with streaming (see roadmap section 3.3)
#   2. Accumulates tokens into full_response_parts
#   3. Yields each token as SSE data (if called from an SSE context)
#   4. After stream ends: strips markdown code fences if present,
#      parses JSON, validates with AuditVerdict.model_validate_json()
#
# Groq pattern (from roadmap):
#   from groq import AsyncGroq
#   client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
#   response = await client.chat.completions.create(
#       model="llama-3.1-8b-instant",
#       messages=[{"role": "user", "content": prompt}],
#       response_format={"type": "json_object"},
#       temperature=0.0,
#       stream=True,
#   )
#   async for chunk in stream:
#       delta = chunk.choices[0].delta.content or ""
#       if delta:
#           full_response_parts.append(delta)
#
# Key insight: Streaming gives you fast Time-To-First-Token (TTFT).
# The FULL response is only available after stream ends. You then
# parse + validate the accumulated JSON and return the AuditVerdict.
#
# If Groq API key is not set,
# raise RuntimeError with a clear message.


async def stream_llm_verdict(prompt: str) -> AuditVerdict:
    # ██ YOUR CODE HERE ██
    pass


# =====================================================================
# TASK 5: Citation Verification (Write this yourself — ~10 lines)
# =====================================================================
# Write verify_citations(verdict: AuditVerdict, source_chunks: list[dict]) -> list[str].
#
# Returns list of grounding failure messages. Empty list = fully grounded.
#
# For each citation in verdict.citations:
#   1. Find the source chunk by document_id
#   2. Check if citation.exact_quote is a substring of chunk.content
#   3. If NOT found → add failure message:
#        f"Hallucinated quote in {doc_id} p.{page}: '{quote[:50]}...'"
#
# Edge case: citation.document_id not in source_chunks → failure.


def verify_citations(verdict: "AuditVerdict", source_chunks: list[dict]) -> list[str]:
    # ██ YOUR CODE HERE ██
    pass


# =====================================================================
# TASK 6: Full Pipeline Orchestration (Write this yourself)
# =====================================================================
# Write async run_verdict_pipeline(query, jurisdiction, effective_year,
#     policy_type="Unknown", top_k=15) -> dict.
#
# Pipeline:
#   1. Generate query vector (use generate_embeddings)
#   2. Run retrieve_filtered (Phase 2 — dense + sparse + RRF)
#      → top_k=15 candidates
#   3. Cross-encoder rerank top candidates (top_n=3)
#      → reranked candidates
#   4. Build grounded prompt from reranked candidates
#   5. stream_llm_verdict(prompt) → AuditVerdict
#   6. verify_citations(verdict, reranked_candidates)
#   7. Return dict:
#        {
#          "verdict": AuditVerdict,
#          "grounding_failures": list[str],
#          "candidates_used": list[dict],
#          "timestamp": datetime.now().isoformat(),
#        }
#
# Run steps 1-2 in parallel? No — 1 depends on query, 2 depends on vector.
# Steps 1 and 2 are sequential (vector → search). Step 3 depends on 2.
# Steps 4 depends on 3. Steps 5-6 depend on 4. All sequential.


async def run_verdict_pipeline(
    query: str,
    jurisdiction: str,
    effective_year: int,
    policy_type: str = "Unknown",
    top_k: int = 15,
) -> dict:
    # ██ YOUR CODE HERE ██
    pass


# =====================================================================
# Main Entry Point
# =====================================================================

async def main():
    test_query = "What is the cyber liability coverage limit?"
    print(f"Query: {test_query}")
    print()

    result = await run_verdict_pipeline(
        query=test_query,
        jurisdiction="Unknown",
        effective_year=2024,
        policy_type="Unknown",
    )

    verdict = result["verdict"]
    failures = result["grounding_failures"]

    print(f"Status: {verdict.status}")
    print(f"Financial Limit: {verdict.financial_limit}")
    print(f"Reasoning: {verdict.reasoning[:200]}...")
    print(f"Citations: {len(verdict.citations)}")
    for c in verdict.citations:
        print(f"  Doc: {c.document_id} | Page: {c.page_number} | Clause: {c.clause_id}")
        print(f"  Quote: {c.exact_quote[:80]}...")
    print()

    if failures:
        print(f"⚠ Grounding failures ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
    else:
        print("✓ All citations verified — fully grounded")


if __name__ == "__main__":
    asyncio.run(main())
