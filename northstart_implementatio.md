# Enterprise Risk & Policy Compliance Engine — Development Roadmap

> **How to use this doc:** Before starting any phase, read the 🆕 section first — those are the things you haven't done before. The 🔁 sections are reference-only; look at the previous project file, adapt, move on. The agent updates `[ ]` checkboxes as you build.

---

## Project North Star

A production RAG engine that reads unstructured insurance policy PDFs, preserves their table structure, answers compliance questions with grounded citations and structured verdicts, and serves everything through a semantic-cached FastAPI gateway.

**What's new over the previous project (the non-obvious additions):**
| New Thing | Why It's Non-Trivial |
|---|---|
| `pdfplumber` layout-aware parsing | PDFs have no semantic structure; tables collapse into garbage without bounding-box detection |
| SQL predicate pushdown with metadata | The WHERE clause must run *before* the vector sort — wrong ordering leaks cross-jurisdiction data |
| `is_table` structural routing | Tables and prose need different chunking strategies; mixing them degrades retrieval |
| Pydantic output schemas (Citation, AuditVerdict) | Forces structured extraction from LLM — prevents hallucinated clause numbers |
| Groq/Ollama LLM integration | Real inference, not simulated word-streaming |
| Disconnect safety (`request.is_disconnected()`) | Without this, abandoned SSE streams hold DB connections open indefinitely |
| Faithfulness audit | Automated grounding verification — checks exact quote presence, not just "does it answer" |

---

## Phase Tracker

| Phase | Title | Status |
|---|---|---|
| 1 | Multi-Tenant Ingestion & Layout-Aware Parser | `[ ]` |
| 2 | Scoped Hybrid Retrieval with Metadata Filters | `[ ]` |
| 3 | Cross-Encoder Reranker & Pydantic Citation Contracts | `[ ]` |
| 4 | Redis Semantic Cache & Streaming Gateway | `[ ]` |
| 5 | Evaluation, Benchmarking & Portfolio Defense | `[ ]` |

---

## Phase 1 — Multi-Tenant Ingestion & Layout-Aware Parser

**Goal:** Ingest multi-page insurance PDFs without losing table structure. Store chunks with enterprise metadata that enables tenant isolation in Phase 2.

### 🔁 Recycle from Previous Project
- **PostgreSQL + pgvector setup** → adapt `14_init_db.py` directly
  - Same: `CREATE EXTENSION vector`, HNSW index (`m=16, ef_construction=64`), GIN index on tsvector
  - Different: new columns listed below
- **SentenceTransformer embedding** → same model (`all-MiniLM-L6-v2`), same `.encode(normalize_embeddings=True)` call
- **Sliding window chunking** → adapt `chunk_text_sliding_window()` from `15_ingest_documents.py`
  - Tune: 60–80 words per chunk, 15-word overlap (wider than before to capture clause context)
- **Batch insert with psycopg** → same pattern as `15_ingest_documents.py`

### 🆕 Learn & Build

#### 1.1 New Schema — `policy_chunks` Table

```sql
CREATE TABLE policy_chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   TEXT NOT NULL,          -- e.g. "policy_acme_2024.pdf"
    policy_type   TEXT NOT NULL,          -- "Cyber", "Commercial Property", "GL"
    jurisdiction  TEXT NOT NULL,          -- "US-NY", "UK", "EU-DE"
    effective_year INT NOT NULL,          -- 2024
    page_number   INT NOT NULL,
    clause_id     TEXT,                   -- "Section 4.1(b)" — extracted from PDF headers
    is_table      BOOLEAN NOT NULL DEFAULT FALSE,
    content       TEXT NOT NULL,
    content_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding     vector(384)
);
```

> **Why these metadata columns matter:** In Phase 2, SQL filters on `jurisdiction` and `effective_year` run *inside the same query* as the vector sort. This is what prevents a UK policy chunk from appearing in a US-NY compliance answer — not application logic, but database enforcement.

#### 1.2 pdfplumber — Dual Extraction Strategy

**New library. Install:** `pip install pdfplumber`

The core insight: PDFs store content as positioned rectangles, not a reading-order stream. `pdfplumber` exposes these bounding boxes.

```python
import pdfplumber

with pdfplumber.open("policy.pdf") as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        # --- Path A: Tables (bounding boxes detected automatically) ---
        tables = page.extract_tables()
        for table in tables:
            # table is List[List[str]] — rows × columns
            # Serialize to Markdown to preserve column relationships
            md = serialize_table_to_markdown(table)
            # Flag is_table=True, wider chunk (don't split tables)

        # --- Path B: Text (everything outside table bounding boxes) ---
        text = page.extract_text()
        # Apply sliding window chunking here
        # Flag is_table=False
```

**Table serialization — do this, not raw `.join()`:**
```python
def serialize_table_to_markdown(table: list[list[str]]) -> str:
    """Preserves column relationships for retrieval.
    Raw join would produce: 'Cyber Liability $5M $100K' — uninterpretable.
    Markdown produces: | Coverage Type | Limit | Deductible |
    """
    if not table or not table[0]:
        return ""
    header = table[0]
    rows = table[1:]
    lines = ["| " + " | ".join(str(c or "") for c in header) + " |"]
    lines.append("|" + "---|" * len(header))
    for row in rows:
        lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
    return "\n".join(lines)
```

> **Why Markdown, not JSON?** The embedding model tokenizes text, not JSON keys. Markdown `|` separators keep column relationships in-sequence so the semantic embedding captures "Cyber Liability → $5M limit" as a unit.

#### 1.3 Clause Header Extraction

```python
import re

CLAUSE_PATTERN = re.compile(
    r'(Section\s+\d+[\.\d]*[a-z]?\(?[a-z]?\)?|Article\s+\d+|Clause\s+\d+[\.\d]*)',
    re.IGNORECASE
)

def extract_clause_id(text: str) -> str | None:
    match = CLAUSE_PATTERN.search(text)
    return match.group(0) if match else None
```

> **Watch out:** Call this on the first 200 chars of each chunk (headers appear early). Calling it on full chunks is noisy.

### Files to Create

- `[ ]` `01_init_db.py` — schema + indexes
- `[ ]` `02_ingest_policies.py` — pdfplumber dual extraction + batch insert
- `[ ]` `sample_policies/` — at least 2 sample PDFs (different jurisdictions)

### ✅ Phase 1 Verification Checkpoint

```sql
-- Run after ingestion. Expect: multiple rows per document_id, both True/False for is_table
SELECT document_id, jurisdiction, policy_type,
       COUNT(*) AS total_chunks,
       SUM(CASE WHEN is_table THEN 1 ELSE 0 END) AS table_chunks
FROM policy_chunks
GROUP BY document_id, jurisdiction, policy_type;
```

Then manually inspect a table chunk:
```sql
SELECT content FROM policy_chunks WHERE is_table = TRUE LIMIT 3;
```
The output should show `|` separated Markdown, not a collapsed string.

---

## Phase 2 — Scoped Hybrid Retrieval with SQL Predicate Pushdown

**Goal:** Dense + sparse search that cannot return results outside the requested tenant's jurisdiction and year range. Filter enforcement happens in the database, not in Python.

### 🔁 Recycle from Previous Project
- `asyncio.gather()` parallel search pattern → identical
- RRF algorithm → copy verbatim from `17_production_rag_gateway.py` `reciprocal_rank_fusion()`
- `psycopg.AsyncConnection` cursor pattern → same as `async_dense_search()` / `async_sparse_search()`

### 🆕 Learn & Build

#### 2.1 Why Predicate Pushdown, Not Post-Filter

**Wrong approach (post-filter):**
```python
results = await dense_search(query_vector, top_k=100)
filtered = [r for r in results if r["jurisdiction"] == "US-NY"]  # ← data leak risk
```

The vector index returns 100 results — but if only 3 are US-NY, you've read 97 unauthorized rows into Python memory. At scale this also wastes index scans.

**Correct approach (predicate pushdown):**
```sql
-- The WHERE clause runs BEFORE the ORDER BY distance — DB engine enforces isolation
SELECT id, document_id, page_number, clause_id, is_table, content,
       (embedding <=> %s::vector) AS distance
FROM policy_chunks
WHERE jurisdiction = %s AND effective_year >= %s   -- ← enforced at DB level
ORDER BY distance ASC
LIMIT 10;
```

> **Implementation note:** pgvector's HNSW index still does approximate search — it searches the index first, then filters. For strict isolation guarantees on very large corpora, add `SET enable_indexscan = off` to force a sequential scan with exact filtering. For this project at demo scale, HNSW + WHERE is fine.

#### 2.2 Filtered Search Function Signatures

```python
async def async_filtered_dense_search(
    query_vector: np.ndarray,
    jurisdiction: str,
    effective_year: int,
    top_k: int = 10,
) -> list[dict]: ...

async def async_filtered_sparse_search(
    query: str,
    jurisdiction: str,
    effective_year: int,
    top_k: int = 10,
) -> list[dict]: ...
```

Both functions take the same filter params. This matters for the RRF fusion — both result sets are scoped to the same tenant, so RRF scores are comparing apples to apples.

#### 2.3 Metadata Passthrough in RRF

The RRF function from the previous project works identically — but the output dict now needs to carry the new metadata fields for Phase 3 citations:

```python
# Add to the result dict in your search functions:
{
    "id": r[0],
    "document_id": r[1],
    "page_number": r[2],
    "clause_id": r[3],
    "is_table": r[4],
    "content": r[5],
    "score": float(r[6]),
}
```

### Files to Create

- `[ ]` `03_hybrid_retrieval.py` — filtered dense + sparse + RRF (standalone, testable)

### ✅ Phase 2 Verification Checkpoint

```python
# Test: query for cyber liability with US-NY filter
# Your sample data should have a UK policy — it must NOT appear in results
results = await retrieve_filtered(
    query="What is the maximum payout for a data breach?",
    jurisdiction="US-NY",
    effective_year=2023
)
assert all(r["jurisdiction"] == "US-NY" for r in results), "Jurisdiction leak detected!"
print(f"Retrieved {len(results)} chunks, all US-NY ✓")
```

---

## Phase 3 — Cross-Encoder Reranker & Pydantic Citation Contracts

**Goal:** Take the RRF top-15 candidates, rerank with a cross-encoder, then send to an LLM that is *forced* to output structured citations via Pydantic. No hallucinated clause numbers.

### 🔁 Recycle from Previous Project
- Cross-encoder loading and `.predict(pairs)` → identical to `rerank_candidates()` in `17_production_rag_gateway.py`
- `run_in_executor` for CPU-bound reranking → copy exactly

### 🆕 Learn & Build

#### 3.1 Pydantic Output Contracts

This is the critical anti-hallucination layer. Without it, LLMs invent clause numbers.

```python
from pydantic import BaseModel, field_validator
from typing import Optional, Literal

class Citation(BaseModel):
    document_id: str
    page_number: int
    clause_id: str                    # Must match a real clause_id from retrieved chunks
    exact_quote: str                  # Must be a substring of the actual chunk content

class AuditVerdict(BaseModel):
    status: Literal["COVERED", "EXCLUDED", "CONDITIONAL", "OUT_OF_SCOPE"]
    financial_limit: Optional[str] = None    # e.g. "$5,000,000"
    applicable_deductible: Optional[str] = None
    reasoning: str
    citations: list[Citation]

    @field_validator("citations")
    @classmethod
    def citations_not_empty(cls, v):
        if not v:
            raise ValueError("Response must include at least one citation")
        return v
```

> **The `exact_quote` field is the key:** When you parse the LLM response, verify `citation.exact_quote` is actually a substring of the chunk it claims to come from. This catches hallucinated quotes automatically.

#### 3.2 Grounded Prompt Construction

```python
def build_grounded_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for c in chunks:
        header = f"[Source: {c['document_id']} | Page: {c['page_number']} | Clause: {c.get('clause_id', 'N/A')}]"
        context_blocks.append(f"{header}\n{c['content']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""You are a policy compliance auditor. Answer strictly from the provided sources only.
If the answer cannot be found in the sources, respond with status "OUT_OF_SCOPE".

SOURCES:
{context_str}

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
```

#### 3.3 LLM Integration — Groq API (recommended) or Ollama (offline)

**Groq (fastest for dev, free tier available):**
```python
from groq import AsyncGroq

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

async def generate_verdict(prompt: str) -> AuditVerdict:
    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},  # Forces JSON output
        temperature=0.0,                           # Deterministic — critical for auditing
    )
    raw_json = response.choices[0].message.content
    return AuditVerdict.model_validate_json(raw_json)
```

**Ollama (no API key, runs locally):**
```python
import httpx

async def generate_verdict_ollama(prompt: str) -> AuditVerdict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False, "format": "json"},
            timeout=60.0,
        )
        return AuditVerdict.model_validate_json(resp.json()["response"])
```

> **temperature=0.0 is non-negotiable for compliance use cases.** Any temperature > 0 introduces randomness into financial figures and coverage decisions.

#### 3.4 Citation Verification (Anti-Hallucination Check)

```python
def verify_citations(verdict: AuditVerdict, source_chunks: list[dict]) -> list[str]:
    """Returns list of grounding failures. Empty list = fully grounded."""
    chunk_map = {c["document_id"]: c["content"] for c in source_chunks}
    failures = []
    for cit in verdict.citations:
        chunk_content = chunk_map.get(cit.document_id, "")
        if cit.exact_quote not in chunk_content:
            failures.append(
                f"Hallucinated quote in {cit.document_id} p.{cit.page_number}: '{cit.exact_quote[:50]}...'"
            )
    return failures
```

### Files to Create

- `[ ]` `04_reranker_and_verdict.py` — cross-encoder + Pydantic schemas + LLM call + citation verifier

### ✅ Phase 3 Verification Checkpoint

**Adversarial test — the answer must NOT be in the document:**
```python
# Inject a query about a coverage type your sample PDFs don't contain
verdict = await run_pipeline(
    query="What is the coverage limit for earthquake damage?",  # not in your docs
    jurisdiction="US-NY",
    effective_year=2023
)
assert verdict.status == "OUT_OF_SCOPE", f"Hallucination detected: {verdict.status}"
print("Adversarial test passed — model correctly reported OUT_OF_SCOPE ✓")
```

---

## Phase 4 — Redis Semantic Cache & Streaming Gateway

**Goal:** Wrap everything in FastAPI. Serve cached queries in <20ms. Stream cache-miss LLM tokens over SSE. Handle client disconnects cleanly.

### 🔁 Recycle from Previous Project
- Redis HNSW index creation → copy `_ensure_redis_index()` from `17_production_rag_gateway.py`
  - Change index name to `idx:risk_cache`, prefix to `cache:policy:`
- `check_semantic_cache()` → copy and adapt (add citation JSON to the stored hash)
- `persist_to_cache()` → adapt (serialize `AuditVerdict` to JSON string for storage)
- Circuit breakers, structured logging, health/ready endpoints → copy verbatim
- Middleware (request-id, latency) → copy verbatim
- `run_in_executor` for reranker → copy verbatim
- `asyncio.gather()` for parallel search → copy verbatim

### 🆕 Learn & Build

#### 4.1 Storing Citations in Redis Cache

The cache entry now stores more than just a text response — it stores the full structured verdict:

```python
async def persist_policy_cache(
    query: str, verdict: AuditVerdict, query_vector: np.ndarray
):
    vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    doc_id = hashlib.sha256(query.encode()).hexdigest()[:16]
    key = f"cache:policy:{doc_id}"

    pipe = redis_client.pipeline()
    pipe.hset(key, mapping={
        "query": query,
        "verdict_json": verdict.model_dump_json(),   # ← Pydantic → JSON string
        "prompt_vector": vector_bytes,
    })
    pipe.expire(key, CFG.cache_ttl_seconds)
    await pipe.execute()
```

On cache hit, deserialize back:
```python
cached_verdict = AuditVerdict.model_validate_json(best.verdict_json)
```

#### 4.2 SSE Streaming with Live LLM Tokens

On cache miss, instead of streaming from a pre-generated string, you stream real LLM tokens:

**Groq streaming:**
```python
async def sse_llm_stream(prompt: str, request_id: str) -> AsyncGenerator[str, None]:
    full_response_parts = []

    async with AsyncGroq() as client:
        stream = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.0,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response_parts.append(delta)
                event = json.dumps({"token": delta, "source": "rag", "request_id": request_id})
                yield f"data: {event}\n\n"

    # After stream ends: parse and cache the full response
    full_text = "".join(full_response_parts)
    # (cache write happens here, outside the generator, via background task)
```

#### 4.3 Disconnect Safety — The Pattern You Haven't Used Before

Without this, if a client closes the connection mid-stream, the server keeps calling the LLM and holding the DB connection:

```python
@app.post("/v1/audit/stream")
async def audit_stream(req: AuditRequest, request: Request):
    # ... setup ...

    async def generate():
        async for chunk in stream:
            # ← Check disconnect before yielding each token
            if await request.is_disconnected():
                logger.info("Client disconnected — cancelling stream",
                            extra={"request_id": request_id})
                break    # ← Exits the generator, triggers cleanup
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream", ...)
```

> **Why this matters at enterprise scale:** Without disconnect checks, a load test of 50 VUs with short timeouts creates 50 orphaned LLM calls and 50 held DB connections — a silent resource leak that shows up as latency spikes hours later.

#### 4.4 The New Request Model

```python
class AuditRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    jurisdiction: str = Field(..., pattern=r"^[A-Z]{2}-[A-Z]{2,4}$")  # "US-NY", "UK", "EU-DE"
    effective_year: int = Field(..., ge=2000, le=2030)
    stream: bool = Field(default=True)
```

> The `jurisdiction` regex pattern is important — it enforces the format that maps to your SQL filter column. A malformed jurisdiction silently returns empty results without it.

### Files to Create

- `[ ]` `05_gateway.py` — full FastAPI app (replaces / builds on `17_production_rag_gateway.py`)

### ✅ Phase 4 Verification Checkpoint

```bash
# Cold query — should stream token-by-token, take ~300-500ms TTFT
curl -N -X POST http://127.0.0.1:8000/v1/audit/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the cyber liability limit?", "jurisdiction": "US-NY", "effective_year": 2023, "stream": true}'

# Warm query — must return in <20ms (check X-Latency-Ms header)
curl -v -X POST http://127.0.0.1:8000/v1/audit/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Cyber liability coverage maximum?", "jurisdiction": "US-NY", "effective_year": 2023, "stream": false}' \
  2>&1 | grep "X-Latency-Ms"
```

---

## Phase 5 — Evaluation, Benchmarking & Portfolio Defense

**Goal:** Prove the system works under load and is factually grounded. Produce numbers for your portfolio.

### 🔁 Recycle from Previous Project
- `load_test.js` → adapt from existing file
  - Add `jurisdiction` and `effective_year` to the request payload
  - Keep 70/30 warm/cold split
  - Raise to 50 max VUs for this project
- `benchmark_ttft.sh` → adapt (update endpoint to `/v1/audit/stream`)

### 🆕 Learn & Build

#### 5.1 k6 Load Test Adaptations

```javascript
// In load_test.js — add the new required fields
const payload = JSON.stringify({
    query: selectedQuery,
    jurisdiction: "US-NY",
    effective_year: 2023,
    stream: false
});

// Add a threshold for cache-hit latency specifically
thresholds: {
    'http_req_duration{cache:hit}': ['p(95)<20'],   // Cache hits must be <20ms at p95
    'http_req_duration{cache:miss}': ['p(95)<800'],  // RAG pipeline under 800ms
    'http_req_failed': ['rate<0.01'],
}
```

To tag cache hits vs. misses, check the response body's `"cached": true/false` field in k6 and tag accordingly.

#### 5.2 Faithfulness Audit Script

This is fully new — there's nothing like this in the previous project.

```python
# 06_faithfulness_audit.py

GROUND_TRUTH = [
    {
        "query": "What is the cyber liability coverage limit?",
        "jurisdiction": "US-NY",
        "effective_year": 2023,
        "expected_financial_limit": "$5,000,000",
        "expected_status": "COVERED",
    },
    # ... 14 more questions covering every policy type in your sample data
]

async def run_faithfulness_audit():
    passed = 0
    for test in GROUND_TRUTH:
        verdict = await run_full_pipeline(**test)

        # Check 1: Correct coverage status
        status_ok = verdict.status == test["expected_status"]

        # Check 2: Financial limit extracted correctly
        limit_ok = test["expected_financial_limit"] in (verdict.financial_limit or "")

        # Check 3: All citations point to real chunks
        grounding_failures = verify_citations(verdict, retrieved_chunks)
        grounded_ok = len(grounding_failures) == 0

        if status_ok and limit_ok and grounded_ok:
            passed += 1
        else:
            print(f"FAILED: {test['query'][:50]}")
            print(f"  status={status_ok}, limit={limit_ok}, grounded={grounded_ok}")

    print(f"\nFaithfulness Score: {passed}/{len(GROUND_TRUTH)} ({passed/len(GROUND_TRUTH)*100:.0f}%)")
```

### Files to Create

- `[ ]` `load_test.js` — updated k6 script (adapt existing)
- `[ ]` `benchmark_ttft.sh` — updated curl script (adapt existing)
- `[ ]` `06_faithfulness_audit.py` — 15-question grounding benchmark
- `[ ]` `README.md` — architecture + benchmark table

### ✅ Phase 5 Verification Checkpoint

```
Target metrics (based on previous project as baseline):
  k6 p95 latency:     < 60ms  (cache-weighted avg across 70% warm / 30% cold)
  k6 error rate:      0.00%
  k6 throughput:      > 100 req/s
  TTFT warm:          < 20ms
  TTFT cold:          < 500ms (LLM dependent)
  Faithfulness score: 100% citations verifiable, 90%+ status correct
```

---

## Key Environment Variables

```bash
# Database
PG_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/risk_db
PG_POOL_MIN=2
PG_POOL_MAX=10

# Redis
REDIS_URL=redis://localhost:6379

# LLM (choose one)
GROQ_API_KEY=gsk_...           # Groq cloud (fast, free tier)
OLLAMA_HOST=http://localhost:11434  # Ollama local

# Models
EMBED_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LLM_PROVIDER=groq              # "groq" or "ollama"
LLM_MODEL=llama-3.1-8b-instant

# Tuning
CACHE_DISTANCE_THRESHOLD=0.30
CACHE_TTL_SECONDS=86400
DENSE_TOP_K=10
SPARSE_TOP_K=10
RRF_K=60
RERANK_CANDIDATES=15
RERANK_TOP_N=3
```

---

## New Dependencies (beyond what's already installed)

```bash
pip install pdfplumber groq httpx
# OR for Ollama: no extra pip install needed, just run ollama pull llama3.2:3b
```

---

## What the Agent Tracks Each Session

Before each session, I'll tell you:
1. **Current phase status** (which checkboxes are done)
2. **What we're building today** (specific file + function)
3. **The one non-obvious thing to watch for** in this session's new code
4. **The verification command** to run before we close

You never need to re-read the whole doc — just the current phase's `🆕` section.
