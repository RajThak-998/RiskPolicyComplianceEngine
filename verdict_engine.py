"""
Phase 3 — Cross-Encoder Reranker & Deterministic Citation Schema

Takes RRF top-15 candidates, reranks with cross-encoder, sends to an LLM
that is forced to output structured citations via Pydantic, then verifies
every citation against actual chunk content (anti-hallucination).

Usage: python verdict_engine.py
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import CrossEncoder 
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from hybrid_retrieval import close_pool, retrieve_filtered
from ingest_policies import generate_embeddings

from typing import Literal
import re
from groq import AsyncGroq

PG_DSN = os.getenv("PG_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/risk_db")

# ── Model Loading ──────────────────────────

REASONER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_DEBUG = os.getenv("AUDIT_DEBUG_RETRIEVAL", "1") == "1"


class Citation(BaseModel):
    document_id: str
    page_number: int
    clause_id: str
    exact_quote: str

    @field_validator("exact_quote")
    @classmethod
    def validate_exact_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("exact_quote must not be empty")
        return value


class AuditVerdict(BaseModel):
    status: Literal["COVERED", "EXCLUDED", "CONDITIONAL", "OUT_OF_SCOPE"]
    financial_limit: Optional[str] = None
    applicable_deductible: Optional[str] = None
    reasoning: str
    citations: list[Citation]

    @model_validator(mode="after")
    def validate_grounded_citations(self) -> "AuditVerdict":
        if self.status != "OUT_OF_SCOPE" and len(self.citations) < 1:
            raise ValueError("At least one citation is required for grounded verdicts")
        if self.status == "OUT_OF_SCOPE" and self.citations:
            raise ValueError("OUT_OF_SCOPE verdicts must not include citations")
        return self

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
        if value is None:
            return value

        if not value.strip():
            raise ValueError("financial_limit must not be empty")

        return value




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
        clause_id = normalized_clause_id(chunk)
        source_block = (
            f"[Source: {chunk['document_id']} | "
            f"Page: {chunk['page_number']} | "
            f"Clause: {clause_id}]\n"
            f"{chunk['content']}"
        )

        source_blocks.append(source_block)

    source_blocks_joined = "\n\n".join(source_blocks)

    prompt = f"""You are a policy compliance auditor. Answer strictly from the provided sources only. If the answer 
    cannot be found in the sources, respond with status 'OUT_OF_SCOPE'.
    If the query asks for a policy term, limit, liability cap, deductible, or exclusion and the sources answer it, use status 'COVERED'.
    For COVERED, EXCLUDED, or CONDITIONAL verdicts, you must include at least one citation.
    For OUT_OF_SCOPE verdicts, set financial_limit and applicable_deductible to null and citations to [].
    Every citation must copy document_id, page_number, and clause_id exactly from a source header.
    Every exact_quote must be copied verbatim from that same source block.
    
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


class LLMValidationError(ValueError):
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


async def stream_llm_json(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("API KEY environment variable is not set.")

    client = AsyncGroq(api_key=api_key)

    stream = await client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format= {"type": "json_object"},
        temperature=0.0,
        stream=True,
    )

    full_response_parts = []

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            full_response_parts.append(delta)

    full_response = "".join(full_response_parts).strip()
    full_response = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        full_response,
        flags=re.IGNORECASE,
    ).strip()

    return full_response


async def stream_llm_verdict(prompt: str) -> AuditVerdict:
    full_response = await stream_llm_json(prompt)
    try:
        return AuditVerdict.model_validate_json(full_response)
    except ValidationError as e:
        raise LLMValidationError(str(e), full_response) from e


async def stream_llm_verdict_with_retry(prompt: str) -> AuditVerdict:
    try:
        return await stream_llm_verdict(prompt)
    except LLMValidationError as first_error:
        retry_prompt = f"""{prompt}

Your previous JSON response failed validation:
{first_error}

Previous invalid JSON:
{first_error.raw_response}

Return corrected JSON only. If the answer is grounded in the sources, include at least one valid citation with an exact_quote copied verbatim from the cited source block. If it is not grounded, return OUT_OF_SCOPE with citations as [].
"""
        return await stream_llm_verdict(retry_prompt)



def normalized_clause_id(chunk: dict) -> str:
    return chunk.get("clause_id") or f"Page {chunk['page_number']}"


def print_candidate_debug(label: str, candidates: list[dict], limit: int = 12) -> None:
    if not RETRIEVAL_DEBUG:
        return

    print(f"\n{label} ({len(candidates)} candidates)")
    for rank, candidate in enumerate(candidates[:limit], start=1):
        preview = re.sub(r"\s+", " ", candidate["content"]).strip()[:180]
        score = candidate.get("score", 0.0)
        rrf_score = candidate.get("rrf_score", 0.0)
        source = candidate.get("retrieval_source", "rrf")
        print(
            f"  {rank:02d}. ID={candidate['id']} "
            f"Doc={candidate['document_id']} "
            f"Page={candidate['page_number']} "
            f"Clause={normalized_clause_id(candidate)} "
            f"Source={source} "
            f"Score={score:.4f} "
            f"RRF={rrf_score:.6f}"
        )
        print(f"      {preview}...")


def verify_citations(verdict: "AuditVerdict", source_chunks: list[dict]) -> list[str]:
    failures = []

    chunks_by_citation = {
        (chunk["document_id"], chunk["page_number"], normalized_clause_id(chunk)): chunk
        for chunk in source_chunks
    }

    for citation in verdict.citations:
        key = (citation.document_id, citation.page_number, citation.clause_id)
        chunk = chunks_by_citation.get(key)

        if chunk is None:
            failures.append(
                f"Source not found: {citation.document_id} "
                f"p.{citation.page_number} clause {citation.clause_id}"
            )
            continue

        if citation.exact_quote not in chunk["content"]:
            failures.append(
                f"Hallucinated quote in "
                f"{citation.document_id} p.{citation.page_number}: "
                f"'{citation.exact_quote[:50]}...'"
            )

    return failures



async def run_verdict_pipeline(
    query: str,
    jurisdiction: str,
    effective_year: int,
    policy_type: str = "Unknown",
    top_k: int = 50,
) -> dict:

    query_vector = generate_embeddings([query])[0]

    candidates = await retrieve_filtered(
       query=query,
       query_vector=query_vector,
       jurisdiction=jurisdiction,
       effective_year=effective_year,
       policy_type=policy_type,
       top_k=top_k,
   )

    print_candidate_debug("Retrieved candidates before cross-encoder", candidates)

    reranked_candidates = await rerank_candidates(
       candidates=candidates,
       query=query,
       top_n=5
    )

    print_candidate_debug("Reranked candidates sent to LLM", reranked_candidates, limit=5)

    prompt = build_grounded_prompt(
        query=query,
        chunks=reranked_candidates,
    )

    try:
        verdict = await stream_llm_verdict_with_retry(prompt)
    except Exception as e:
        print(f"⚠ LLM call failed: {e}")
        print("  Returning partial result with no verdict")
        return {
            "verdict": None,
            "grounding_failures": [f"LLM error: {str(e)}"],
            "candidates_used": reranked_candidates,
            "timestamp": datetime.now().isoformat(),
        }

    grounding_failures = verify_citations(
        verdict=verdict,
        source_chunks=reranked_candidates,
    )

    return {
        "verdict": verdict,
        "grounding_failures": grounding_failures,
        "candidates_used": reranked_candidates,
        "timestamp": datetime.now().isoformat(),
    }



async def main():
    test_query = (
        "What is the maximum liability for buildings, plants and machinery, "
        "furniture, fixtures and fittings under Section I Material Damage?"
    )
    print(f"Query: {test_query}")
    print()

    try:
        result = await run_verdict_pipeline(
            query=test_query,
            jurisdiction="Unknown",
            effective_year=2024,
            policy_type="Unknown",
        )
    finally:
        await close_pool()

    verdict = result["verdict"]
    failures = result["grounding_failures"]

    if verdict is None:
        print("No verdict generated (see errors above)")
    else:
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
