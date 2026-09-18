"""
Phase 4 - Redis Semantic Cache & Streaming Gateway

FastAPI wrapper around the Phase 3 verdict engine.

Usage:
    uvicorn 05_gateway:app --reload --host 127.0.0.1 --port 8000
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import AsyncGroq
from pydantic import BaseModel, Field, ValidationError, field_validator
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from hybrid_retrieval import close_pool, retrieve_filtered
from ingest_policies import generate_embeddings
from verdict_engine import (
    AuditVerdict,
    build_grounded_prompt,
    rerank_candidates,
    verify_citations,
)
from ingest_policies import ensure_session_schema, ingest_single_pdf

load_dotenv()


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_INDEX = os.getenv("REDIS_CACHE_INDEX", "idx:risk_cache")
REDIS_PREFIX = os.getenv("REDIS_CACHE_PREFIX", "cache:policy:")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
CACHE_DISTANCE_THRESHOLD = float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.30"))
EMBED_DIM = 384
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "50"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
LEGACY_SESSION_ID = "legacy"
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "risk-policy-uploads")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


app = FastAPI(
    title="Risk Policy Compliance Engine",
    version="0.4.0",
)

redis_client: Redis = Redis.from_url(REDIS_URL, decode_responses=False)
sessions: dict[str, dict] = {}


class AuditRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    jurisdiction: str = Field(default="Unknown", min_length=2, max_length=32)
    effective_year: int = Field(default=2024, ge=2000, le=2035)
    policy_type: str = Field(default="Unknown", min_length=2, max_length=80)
    stream: bool = Field(default=True)
    session_id: str = Field(default=LEGACY_SESSION_ID, min_length=3, max_length=80)

    @field_validator("jurisdiction")
    @classmethod
    def validate_jurisdiction(cls, value: str) -> str:
        if value == "Unknown":
            return value
        if not re.fullmatch(r"[A-Z]{2}(?:-[A-Z]{2,4})?", value):
            raise ValueError("jurisdiction must be 'Unknown' or a code like 'US-NY'")
        return value


class AuditResponse(BaseModel):
    cached: bool
    cache_distance: Optional[float] = None
    verdict: AuditVerdict
    grounding_failures: list[str]
    candidates_used: list[dict]
    timestamp: str


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _latency_headers(start_ms: float, cached: bool) -> dict[str, str]:
    return {
        "X-Latency-Ms": f"{_now_ms() - start_ms:.2f}",
        "X-Cache": "HIT" if cached else "MISS",
    }


def _vector_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _cache_key(req: AuditRequest) -> str:
    material = "|".join([
        req.query.strip().lower(),
        req.jurisdiction,
        str(req.effective_year),
        req.policy_type,
    ])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"{REDIS_PREFIX}{digest}"


def _escape_tag(value: str) -> str:
    return re.sub(r"([\\{}|,<>\\[\\]\"':;!@#$%^&*()+=~\\s])", r"\\\1", str(value))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def ensure_redis_index() -> None:
    try:
        await redis_client.execute_command("FT.INFO", REDIS_INDEX)
        try:
            await redis_client.execute_command(
                "FT.ALTER", REDIS_INDEX, "SCHEMA", "ADD", "session_id", "TAG"
            )
        except ResponseError as alter_error:
            alter_message = str(alter_error).lower()
            if "already exists" not in alter_message and "duplicate field" not in alter_message:
                raise
        return
    except ResponseError as exc:
        if "unknown index name" not in str(exc).lower():
            raise

    await redis_client.execute_command(
        "FT.CREATE", REDIS_INDEX,
        "ON", "HASH",
        "PREFIX", "1", REDIS_PREFIX,
        "SCHEMA",
        "query", "TEXT",
        "jurisdiction", "TAG",
        "effective_year", "TAG",
        "policy_type", "TAG",
        "session_id", "TAG",
        "verdict_json", "TEXT",
        "created_at", "NUMERIC",
        "prompt_vector", "VECTOR", "HNSW", "6",
        "TYPE", "FLOAT32",
        "DIM", str(EMBED_DIM),
        "DISTANCE_METRIC", "COSINE",
    )


async def check_semantic_cache(
    req: AuditRequest,
    query_vector: np.ndarray,
) -> tuple[Optional[AuditVerdict], Optional[float]]:
    await ensure_redis_index()

    redis_query = (
        f"(@jurisdiction:{{{_escape_tag(req.jurisdiction)}}} "
        f"@effective_year:{{{req.effective_year}}} "
        f"@session_id:{{{_escape_tag(req.session_id)}}}"
        + (f" @policy_type:{{{_escape_tag(req.policy_type)}}}" if req.policy_type != "All" else "")
        + ")"
        f"=>[KNN 1 @prompt_vector $vec AS distance]"
    )

    result = await redis_client.execute_command(
        "FT.SEARCH", REDIS_INDEX, redis_query,
        "PARAMS", "2", "vec", _vector_bytes(query_vector),
        "SORTBY", "distance",
        "RETURN", "2", "verdict_json", "distance",
        "DIALECT", "2",
    )

    if isinstance(result, dict):
        total_results = result.get(
            b"total_results",
            result.get("total_results", 0),
        )
        if not total_results:
            return None, None

        results = result.get(b"results", result.get("results", []))
        first_result = results[0]
        fields = first_result.get(
            b"extra_attributes",
            first_result.get("extra_attributes", {}),
        )
        values = {
            (key.decode("utf-8") if isinstance(key, bytes) else key): value
            for key, value in fields.items()
        }
    else:
        if not result or result[0] == 0:
            return None, None

        fields = result[2]
        values = {
            fields[i].decode("utf-8"): fields[i + 1]
            for i in range(0, len(fields), 2)
        }

    distance = float(values["distance"])

    if distance > CACHE_DISTANCE_THRESHOLD:
        return None, distance

    verdict_json = values["verdict_json"]
    if isinstance(verdict_json, bytes):
        verdict_json = verdict_json.decode("utf-8")
    return AuditVerdict.model_validate_json(verdict_json), distance


async def persist_policy_cache(
    req: AuditRequest,
    verdict: AuditVerdict,
    query_vector: np.ndarray,
) -> None:
    await ensure_redis_index()

    key = _cache_key(req)
    await redis_client.hset(key, mapping={
        "query": req.query,
        "jurisdiction": req.jurisdiction,
        "effective_year": str(req.effective_year),
        "policy_type": req.policy_type,
        "session_id": req.session_id,
        "verdict_json": verdict.model_dump_json(),
        "created_at": str(time.time()),
        "prompt_vector": _vector_bytes(query_vector),
    })
    await redis_client.expire(key, CACHE_TTL_SECONDS)


async def prepare_rag_context(
    req: AuditRequest,
    query_vector: np.ndarray,
) -> tuple[list[dict], str]:
    candidates = await retrieve_filtered(
        query=req.query,
        query_vector=query_vector,
        jurisdiction=req.jurisdiction,
        effective_year=req.effective_year,
        policy_type=req.policy_type,
        session_id=req.session_id,
        top_k=RETRIEVAL_TOP_K,
    )
    reranked_candidates = await rerank_candidates(
        candidates=candidates,
        query=req.query,
        top_n=RERANK_TOP_N,
    )
    prompt = build_grounded_prompt(req.query, reranked_candidates)
    return reranked_candidates, prompt


async def call_llm_verdict(prompt: str) -> AuditVerdict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    client = AsyncGroq(api_key=api_key)
    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        stream=True,
    )

    parts = []
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            parts.append(delta)

    return AuditVerdict.model_validate_json("".join(parts).strip())


async def stream_llm_verdict(
    prompt: str,
    request_id: str,
    request: Request,
) -> AsyncGenerator[tuple[str, Optional[AuditVerdict]], None]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    client = AsyncGroq(api_key=api_key)
    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        stream=True,
    )

    parts = []
    async for chunk in stream:
        if await request.is_disconnected():
            break

        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue

        parts.append(delta)
        yield _sse("token", {
            "token": delta,
            "source": "rag",
            "request_id": request_id,
        }), None

    if parts:
        verdict = AuditVerdict.model_validate_json("".join(parts).strip())
        yield "", verdict


def response_payload(
    verdict: AuditVerdict,
    grounding_failures: list[str],
    candidates: list[dict],
    cached: bool,
    cache_distance: Optional[float],
) -> dict:
    return AuditResponse(
        cached=cached,
        cache_distance=cache_distance,
        verdict=verdict,
        grounding_failures=grounding_failures,
        candidates_used=candidates,
        timestamp=datetime.now().isoformat(),
    ).model_dump(mode="json")


@app.on_event("startup")
async def startup() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    await asyncio.to_thread(ensure_session_schema)
    await ensure_redis_index()


@app.on_event("shutdown")
async def shutdown() -> None:
    await redis_client.aclose()
    await close_pool()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict:
    await redis_client.ping()
    return {"status": "ready", "redis_index": REDIS_INDEX}


@app.post("/v1/sessions")
async def create_session() -> dict:
    session_id = uuid.uuid4().hex
    sessions[session_id] = {
        "session_id": session_id,
        "status": "ready",
        "documents": [],
    }
    return sessions[session_id]


async def ingest_upload(
    session_id: str,
    document_id: str,
    filepath: str,
    title: str,
    policy_type: str,
) -> None:
    document = next(
        item for item in sessions[session_id]["documents"]
        if item["document_id"] == document_id
    )
    try:
        document["status"] = "processing"
        sessions[session_id]["status"] = "processing"
        result = await ingest_single_pdf(
            filepath=filepath,
            session_id=session_id,
            policy_type=policy_type,
            document_title=title,
        )
        document.update({
            "status": "ready",
            "chunks": result["chunks"],
            "tables": result["tables"],
            "pages": result["pages"],
        })
        sessions[session_id]["status"] = "ready"
    except Exception as exc:
        document.update({"status": "failed", "error": str(exc)})
        sessions[session_id]["status"] = "failed"
    finally:
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass


@app.post("/v1/sessions/{session_id}/documents", status_code=202)
async def upload_document(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(..., min_length=1, max_length=160),
    policy_type: str = Form("Unknown", min_length=2, max_length=80),
) -> dict:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the upload size limit")
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")

    document_id = f"{uuid.uuid4().hex}_{os.path.basename(file.filename or 'policy.pdf')}"
    filepath = os.path.join(UPLOAD_DIR, document_id)
    with open(filepath, "wb") as output:
        output.write(payload)

    document = {
        "document_id": document_id,
        "filename": file.filename or "policy.pdf",
        "title": title,
        "policy_type": policy_type,
        "status": "queued",
    }
    session["documents"].append(document)
    session["status"] = "processing"
    session["chat_ready"] = False
    background_tasks.add_task(
        ingest_upload, session_id, document_id, filepath, title, policy_type
    )
    return {"session_id": session_id, "document": document}


@app.get("/v1/sessions/{session_id}/status")
async def session_status(session_id: str) -> dict:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/v1/audit/stream")
async def audit_stream(req: AuditRequest, request: Request) -> Response:
    if req.session_id != LEGACY_SESSION_ID:
        session = sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["status"] != "ready":
            raise HTTPException(status_code=409, detail="Uploaded documents are not ready")
        req.policy_type = "All"
    start_ms = _now_ms()
    request_id = hashlib.sha256(f"{time.time()}:{req.query}".encode()).hexdigest()[:12]
    query_vector = generate_embeddings([req.query])[0]

    cached_verdict, cache_distance = await check_semantic_cache(req, query_vector)
    if cached_verdict is not None:
        payload = response_payload(
            verdict=cached_verdict,
            grounding_failures=[],
            candidates=[],
            cached=True,
            cache_distance=cache_distance,
        )

        if not req.stream:
            return JSONResponse(
                payload,
                headers=_latency_headers(start_ms, cached=True),
            )

        async def cached_events() -> AsyncGenerator[str, None]:
            yield _sse("verdict", payload)

        return StreamingResponse(
            cached_events(),
            media_type="text/event-stream",
            headers=_latency_headers(start_ms, cached=True),
        )

    candidates, prompt = await prepare_rag_context(req, query_vector)

    if not req.stream:
        verdict = await call_llm_verdict(prompt)
        grounding_failures = verify_citations(verdict, candidates)
        if not grounding_failures:
            await persist_policy_cache(req, verdict, query_vector)

        return JSONResponse(
            response_payload(
                verdict=verdict,
                grounding_failures=grounding_failures,
                candidates=candidates,
                cached=False,
                cache_distance=cache_distance,
            ),
            headers=_latency_headers(start_ms, cached=False),
        )

    async def rag_events() -> AsyncGenerator[str, None]:
        yield _sse("meta", {
            "cached": False,
            "cache_distance": cache_distance,
            "request_id": request_id,
            "candidates": [
                {
                    "id": c["id"],
                    "document_id": c["document_id"],
                    "page_number": c["page_number"],
                    "clause_id": c.get("clause_id"),
                    "score": c.get("score"),
                }
                for c in candidates
            ],
        })

        try:
            async for event, verdict in stream_llm_verdict(prompt, request_id, request):
                if await request.is_disconnected():
                    break
                if event:
                    yield event
                if verdict is None:
                    continue

                grounding_failures = verify_citations(verdict, candidates)
                if not grounding_failures:
                    await persist_policy_cache(req, verdict, query_vector)

                yield _sse("verdict", response_payload(
                    verdict=verdict,
                    grounding_failures=grounding_failures,
                    candidates=candidates,
                    cached=False,
                    cache_distance=cache_distance,
                ))
        except (ValidationError, RuntimeError) as exc:
            yield _sse("error", {
                "request_id": request_id,
                "message": str(exc),
            })

    return StreamingResponse(
        rag_events(),
        media_type="text/event-stream",
        headers=_latency_headers(start_ms, cached=False),
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
