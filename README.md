# Risk Policy Compliance Engine — Policy Desk

> **Grounded risk audit intelligence.** Upload a PDF policy, watch the evidence pipeline index it in real-time, then interrogate it with fully cited, hallucination-checked answers — not guesses.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [How It Works — Pipeline Deep Dive](#how-it-works--pipeline-deep-dive)
   - [Phase 1 · Database Initialisation](#phase-1--database-initialisation)
   - [Phase 2 · PDF Ingestion](#phase-2--pdf-ingestion)
   - [Phase 3 · Hybrid Retrieval](#phase-3--hybrid-retrieval)
   - [Phase 4 · Cross-Encoder Reranking & Verdict Engine](#phase-4--cross-encoder-reranking--verdict-engine)
   - [Phase 5 · Semantic Cache & Streaming Gateway](#phase-5--semantic-cache--streaming-gateway)
6. [Frontend — Policy Desk UI](#frontend--policy-desk-ui)
7. [API Reference](#api-reference)
8. [Environment Variables](#environment-variables)
9. [Local Setup](#local-setup)
10. [Running the Application](#running-the-application)
11. [User Guide (Step-by-Step)](#user-guide-step-by-step)
12. [Query Tips for CGL Policies](#query-tips-for-cgl-policies)
13. [Known Limitations](#known-limitations)
14. [Design Decisions](#design-decisions)

---

## Overview

The **Risk Policy Compliance Engine** is a RAG (Retrieval-Augmented Generation) system purpose-built for auditing insurance policy documents. It lets a compliance analyst upload any PDF policy, automatically parse and embed every clause and table, and then ask natural-language questions that are answered with pinpoint citations from the source document.

Key guarantees the system makes:

| Guarantee | How it is enforced |
|---|---|
| **Answers are grounded** | Every non-`OUT_OF_SCOPE` response must include at least one `exact_quote` copied verbatim from the retrieved chunks |
| **Hallucinations are flagged** | `verify_citations()` checks every LLM-generated citation: document ID, page number, clause, and exact quote must all match the actual retrieved text |
| **Repeated queries are fast** | A semantic Redis vector cache serves identical (or near-identical) queries instantly instead of re-running the entire RAG pipeline |
| **Streaming keeps UX alive** | LLM responses are streamed token-by-token via Server-Sent Events so the UI is responsive even on long answers |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Browser (Policy Desk UI)                    │
│   Upload PDF ──► Pipeline status (polled every 1.2 s)            │
│   Ask query   ◄─► SSE stream (token-by-token or cache instant)   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP / SSE
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Gateway  (gateway.py)                 │
│                                                                  │
│  POST /v1/sessions               → create session                │
│  POST /v1/sessions/{id}/documents → upload + background ingest   │
│  GET  /v1/sessions/{id}/status   → poll ingestion progress       │
│  POST /v1/audit/stream           → RAG or cache-hit SSE stream   │
└────────┬──────────────────────────────────┬──────────────────────┘
         │                                  │
         ▼                                  ▼
┌────────────────┐                ┌──────────────────────────┐
│  Ingestion     │                │   Semantic Redis Cache    │
│  Pipeline      │                │   (RediSearch HNSW)       │
│  (background)  │                │   TTL: 24 h               │
│                │                └─────────────┬────────────┘
│  pdfplumber    │                              │ MISS
│  → chunks      │                              ▼
│  → embeddings  │         ┌───────────────────────────────────┐
│  → PostgreSQL  │         │        Hybrid Retrieval            │
└────────┬───────┘         │                                   │
         │                 │  Dense:  pgvector HNSW cosine     │
         ▼                 │  Sparse: PostgreSQL tsvector GIN  │
┌────────────────┐         │  Merge:  Reciprocal Rank Fusion   │
│  PostgreSQL    │◄────────│  Window: ±1 neighbour chunks      │
│  + pgvector   │         └─────────────┬─────────────────────┘
│                │                      │
│  policy_chunks │                      ▼
│  HNSW index    │         ┌───────────────────────────────────┐
│  GIN  index    │         │    Cross-Encoder Reranker          │
└────────────────┘         │    ms-marco-MiniLM-L-6-v2         │
                           │    Top-12 → LLM context            │
                           └─────────────┬─────────────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────────────┐
                           │    LLM  (via Groq API)             │
                           │    model: openai/gpt-oss-120b      │
                           │    response_format: json_object    │
                           │    temperature: 0.0                │
                           └─────────────┬─────────────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────────────┐
                           │    Citation Verifier               │
                           │    verify_citations()              │
                           │    Exact-quote substring match     │
                           │    Always caches result to Redis   │
                           └───────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **API Server** | FastAPI + Uvicorn |
| **Database** | PostgreSQL 15+ with **pgvector** extension |
| **Vector Index** | pgvector **HNSW** (cosine, m=16, ef_construction=200) |
| **Full-text Index** | PostgreSQL **tsvector GIN** (English stemming) |
| **Semantic Cache** | Redis Stack with **RediSearch** HNSW vector index |
| **PDF Parsing** | pdfplumber (layout-aware, table extraction) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalised) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Groq API — `openai/gpt-oss-120b` |
| **Frontend** | Vanilla HTML / CSS / JS (no framework), SSE streaming |
| **Connection Pool** | `psycopg_pool.AsyncConnectionPool` (min 2, max 10) |

---

## Project Structure

```
RiskPolicyComplianceEngine/
│
├── gateway.py              # FastAPI app — sessions, upload, audit, cache, streaming
├── hybrid_retrieval.py     # Dense + sparse retrieval, RRF fusion, neighbour chunks
├── ingest_policies.py      # PDF parsing, chunking, embedding, DB insert
├── verdict_engine.py       # Cross-encoder reranker, prompt builder, citation verifier
├── init_db.py              # One-time DB schema + index creation
│
├── static/
│   ├── index.html          # Single-page Policy Desk UI
│   ├── app.js              # SSE client, pipeline animation, hallucination badges
│   └── styles.css          # Design system (DM Sans + DM Mono + Syne)
│
├── raw_policies/           # Dev-time PDFs (not required at runtime)
│   ├── cgl_specimen_policy.pdf
│   └── property_risk_policy.pdf
│
├── .env                    # GROQ_API_KEY and optional overrides (never commit)
├── decisions.md            # Architecture Decision Records
├── flow.md                 # Sequence diagrams
└── README.md               # This file
```

---

## How It Works — Pipeline Deep Dive

### Phase 1 · Database Initialisation

**File:** `init_db.py`

Run once before the first ingest. Creates the `policy_chunks` table and all necessary indexes.

```sql
CREATE TABLE policy_chunks (
    id             BIGSERIAL PRIMARY KEY,
    session_id     TEXT NOT NULL DEFAULT 'legacy',  -- upload session isolation
    document_id    TEXT NOT NULL,                   -- source PDF filename
    policy_type    TEXT NOT NULL,
    jurisdiction   TEXT NOT NULL,
    effective_year INT  NOT NULL,
    page_number    INT  NOT NULL,
    clause_id      TEXT,                            -- extracted section header
    is_table       BOOLEAN NOT NULL DEFAULT FALSE,
    content        TEXT NOT NULL,
    content_tsv    tsvector GENERATED ALWAYS AS (   -- auto-maintained FTS column
                       to_tsvector('english', content)
                   ) STORED,
    embedding      vector(384)                      -- all-MiniLM-L6-v2 output
);
```

**Indexes created:**

| Index | Type | Purpose |
|---|---|---|
| `idx_policy_chunks_embedding` | HNSW cosine | Fast approximate nearest-neighbour vector search |
| `idx_policy_chunks_tsv` | GIN | Fast full-text search |
| `idx_policy_chunks_metadata` | B-tree composite | Predicate pushdown on jurisdiction / year / type |
| `idx_policy_chunks_session` | B-tree | Fast session-scoped queries |

---

### Phase 2 · PDF Ingestion

**File:** `ingest_policies.py`
**Triggered by:** `POST /v1/sessions/{session_id}/documents` → runs as a FastAPI `BackgroundTask`

Ingestion runs entirely in the background after the upload endpoint returns HTTP 202. The frontend polls `/v1/sessions/{id}/status` every 1.2 seconds to track progress.

**Steps:**

1. **Open PDF** with `pdfplumber` (layout-aware; handles multi-column text and embedded tables)
2. **Extract tables** per page → serialised to Markdown (`| col | col |`)
3. **Chunk prose** with a sliding window:
   - Window: **120 words**
   - Overlap: **40 words** (so clauses that span chunk boundaries are still findable)
4. **Extract clause ID** from the first 200 characters of each chunk using a regex that matches `Section X`, `Article X`, `Clause X`, and `Exclusion` headers
5. **Batch-embed** all chunks with `all-MiniLM-L6-v2` (batch size 32, L2-normalised)
6. **Bulk-insert** into `policy_chunks` with `psycopg.executemany`
7. **Update session status** to `ready` — the frontend poll detects this and triggers the ready signal

**Chunking configuration:**

| Parameter | Value | Meaning |
|---|---|---|
| `CHUNK_WORDS` | 120 | Words per chunk window |
| `CHUNK_OVERLAP` | 40 | Words shared between consecutive chunks |
| `EMBED_BATCH_SIZE` | 32 | Texts per embedding forward pass |

---

### Phase 3 · Hybrid Retrieval

**File:** `hybrid_retrieval.py`

Every query fires two searches in parallel (via `asyncio.gather`), then merges results.

#### Dense Search (pgvector)

```sql
SELECT id, content, ...,
       (embedding <=> $query_vector::vector(384)) AS distance
FROM policy_chunks
WHERE session_id    = $session_id
  AND jurisdiction  = $jurisdiction
  AND effective_year = $effective_year
  AND ($policy_type = 'All' OR policy_type = $policy_type)
ORDER BY distance ASC
LIMIT 50;
```

The HNSW index enables sub-millisecond approximate nearest-neighbour lookup. Metadata filters are applied at the index level (predicate pushdown) so only session-scoped chunks are ever evaluated.

#### Sparse Search (tsvector)

Builds a two-tier tsquery from the natural-language query:

- **Strict query** — `plainto_tsquery('english', query)`: requires all meaningful terms
- **Soft query** — OR-joined individual terms (stopwords and short words stripped): allows partial matches for partial recall

```sql
SELECT id, content, ...,
       ts_rank(content_tsv, strict_query) * 2.0
       + COALESCE(ts_rank(content_tsv, soft_query), 0.0) AS rank
FROM policy_chunks
WHERE session_id = $session_id
  AND (content_tsv @@ strict_query OR content_tsv @@ soft_query)
ORDER BY rank DESC
LIMIT 50;
```

Strict matches are weighted 2× because they indicate higher precision.

#### Reciprocal Rank Fusion (RRF)

Both ranked lists are merged with RRF (k = 60):

```
score(doc) = Σ  1 / (60 + rank_in_list)
```

RRF is rank-position-based rather than score-based, making it robust to the scale difference between cosine distance (0–2) and ts_rank (0–1).

#### Neighbour Window Expansion

After RRF, the top-50 results are expanded with their ±1 adjacent chunks from the same session and document. This recovers context cut off at chunk boundaries — a common issue when a clause header lands at the end of one chunk and the actual clause body starts in the next.

---

### Phase 4 · Cross-Encoder Reranking & Verdict Engine

**File:** `verdict_engine.py`

#### Cross-Encoder Reranking

The ~76 candidates from hybrid retrieval are reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike the bi-encoder used for embedding, a cross-encoder processes the **query and document together** as a single input, producing a much more accurate relevance score at the cost of latency (all pairs scored sequentially via `asyncio.to_thread`).

The top **12** candidates survive into the LLM prompt (configurable via `RERANK_TOP_N`).

#### Prompt Construction

Each of the top-12 chunks becomes a labelled source block fed to the LLM:

```
[Source: policy.pdf | Page: 3 | Clause: Section c - Liquor Liability]
Bodily injury or property damage for which any Insured may be held liable ...
```

The LLM is instructed to answer **only from these sources**, classify the answer as one of four statuses, and include verbatim `exact_quote` citations that can be verified.

#### Verdict Schema (Pydantic)

```python
class AuditVerdict(BaseModel):
    status: Literal["COVERED", "EXCLUDED", "CONDITIONAL", "OUT_OF_SCOPE"]
    financial_limit: Optional[str]
    applicable_deductible: Optional[str]
    reasoning: str
    citations: list[Citation]  # required (non-empty) for all statuses except OUT_OF_SCOPE
```

```python
class Citation(BaseModel):
    document_id: str
    page_number: int
    clause_id: str
    exact_quote: str  # must be a verbatim substring of the source chunk
```

#### Citation Verification (Anti-Hallucination)

After the LLM returns a verdict, `verify_citations()` performs a deterministic check:

1. Build a lookup map: `(document_id, page_number, clause_id) → chunk content`
2. For each citation the LLM produced:
   - Does the `(document_id, page_number, clause_id)` triple exist in the retrieved chunks? If not → **Source not found** failure
   - Is `exact_quote` a literal substring of the actual chunk content? If not → **Hallucinated quote** failure

All failures are included in the API response under `grounding_failures` and displayed as a red `⚠ HALLUCINATION DETECTED` banner in the UI. The verdict is **always cached** regardless of failures — grounding failures are informational, not blocking.

---

### Phase 5 · Semantic Cache & Streaming Gateway

**File:** `gateway.py`

#### Semantic Cache (Redis)

Before running the RAG pipeline, every query checks the Redis vector cache:

1. Embed the query with `all-MiniLM-L6-v2`
2. Run a KNN vector search on the RediSearch index (filtered by `session_id`, `jurisdiction`, `effective_year`, `policy_type`)
3. If the nearest cached vector has cosine distance ≤ **0.30** → return the cached verdict instantly (no LLM call, no retrieval)
4. On a miss → run the full RAG pipeline, then write the result to Redis with a **24-hour TTL**

The cache is per-session. Uploading a new document creates a fresh session, so it won't return cached answers from a different document.

#### Streaming (Server-Sent Events)

For cache misses, the LLM response is streamed token-by-token:

| Event | Payload | When sent |
|---|---|---|
| `meta` | candidate list, request ID, cache distance | Before LLM generation starts |
| `token` | `{ "token": "...", "source": "rag" }` | Each LLM output token as it arrives |
| `verdict` | Full `AuditResponse` JSON | After LLM completes + citations verified |
| `error` | Error message string | On any failure during generation |

For cache **hits**, a single `verdict` event is emitted immediately — no `token` events.

Response headers on every request:

| Header | Values |
|---|---|
| `X-Cache` | `HIT` or `MISS` |
| `X-Latency-Ms` | Total gateway latency in milliseconds |

---

## Frontend — Policy Desk UI

**Files:** `static/index.html`, `static/app.js`, `static/styles.css`

A single-page application built with vanilla HTML, CSS, and JavaScript — no build step, no framework, no node_modules.

### Step Bar

A persistent 3-step progress tracker at the top of the page:

```
① Upload Policy  ──────  ② Ingestion Pipeline  ──────  ③ Audit Ready
```

Steps highlight and check as the user moves through the workflow.

### Live Pipeline Panel

After upload, a sidebar panel shows animated real-time ingestion progress:

```
✓ Uploading file to server
⟳ Chunking & extracting tables    ← spinning
  Generating embeddings
  Indexing into PostgreSQL + Redis
  Verifying index & signalling ready
```

Steps are animated cosmetically (~3.5 s each) and finalised by the real server poll. The "Policy indexed!" overlay only appears **after** all steps visually complete, strictly sequenced via an `async/await` chain.

### Streaming Conversation

- **Cache MISS** — a streaming bubble appears with a blinking `▋` cursor; raw JSON tokens stream into a monospace preview box; when the `verdict` SSE event arrives the bubble morphs into the full structured verdict card
- **Cache HIT** — verdict renders instantly from the single `verdict` event; badge shows `⚡ Redis cache hit · dist 0.xxx`

### Hallucination Warning

If `grounding_failures` is non-empty, a red `⚠ HALLUCINATION DETECTED` banner appears inside the message card with a bulleted list of each failed citation. A coloured toast also fires in the bottom-right corner.

### Cache / Source Badge

| Badge | State |
|---|---|
| `Waiting for policy` | No document uploaded |
| `Indexing evidence…` | Background ingest running |
| `Ready to audit` | Document ready, no query yet |
| `⚡ Redis cache hit` | Response served from Redis |
| `⟳ Cache miss — live RAG` | Response generated fresh |

---

## API Reference

### `POST /v1/sessions`

Create a new isolated upload session.

**Response:**
```json
{
  "session_id": "a1bc78da080a4b6f...",
  "status": "ready",
  "documents": []
}
```

---

### `POST /v1/sessions/{session_id}/documents`

Upload a PDF. Returns HTTP **202 Accepted** immediately; ingestion runs in the background.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | PDF binary | ✅ | Max 25 MB |
| `title` | string | ✅ | 1–160 chars |
| `policy_type` | string | ❌ | Defaults to `Unknown` |

**Response (202):**
```json
{
  "session_id": "a1bc78da...",
  "document": {
    "document_id": "abc123_policy.pdf",
    "filename": "policy.pdf",
    "title": "CGL Specimen Policy",
    "policy_type": "Unknown",
    "status": "queued"
  }
}
```

---

### `GET /v1/sessions/{session_id}/status`

Poll ingestion progress.

**Response:**
```json
{
  "session_id": "a1bc78da...",
  "status": "ready",
  "documents": [
    {
      "document_id": "abc123_policy.pdf",
      "title": "CGL Specimen Policy",
      "status": "ready",
      "pages": 32,
      "chunks": 160,
      "tables": 3
    }
  ]
}
```

`status` progression: `queued` → `processing` → `ready` | `failed`

---

### `POST /v1/audit/stream`

Run a grounded audit query. Returns an SSE stream.

**Request body (JSON):**

```json
{
  "query": "What is the duty to defend under this policy?",
  "session_id": "a1bc78da...",
  "jurisdiction": "Unknown",
  "effective_year": 2024,
  "policy_type": "Unknown",
  "stream": true
}
```

| Field | Type | Default | Constraints |
|---|---|---|---|
| `query` | string | — | 5–2000 chars, required |
| `session_id` | string | `legacy` | Must match an active session |
| `jurisdiction` | string | `Unknown` | `Unknown` or ISO code e.g. `US-NY` |
| `effective_year` | int | 2024 | 2000–2035 |
| `policy_type` | string | `Unknown` | Freeform label |
| `stream` | bool | `true` | `false` returns a single JSON body |

**SSE Events:**

```
event: meta
data: {"cached": false, "cache_distance": null, "candidates": [...], "request_id": "abc123"}

event: token
data: {"token": "The", "source": "rag", "request_id": "abc123"}

event: verdict
data: {
  "cached": false,
  "cache_distance": null,
  "verdict": {
    "status": "COVERED",
    "financial_limit": null,
    "applicable_deductible": null,
    "reasoning": "The policy states...",
    "citations": [
      {
        "document_id": "policy.pdf",
        "page_number": 1,
        "clause_id": "SECTION I - COVERAGES COVERAGE A...",
        "exact_quote": "We will have the right and duty to defend..."
      }
    ]
  },
  "grounding_failures": [],
  "candidates_used": [...],
  "timestamp": "2026-09-18T19:00:00"
}

event: error
data: {"request_id": "abc123", "message": "The audit stream failed."}
```

---

### `GET /health`
```json
{ "status": "ok" }
```

### `GET /ready`
```json
{ "status": "ready", "redis_index": "idx:risk_cache" }
```

---

## Environment Variables

Create a `.env` file in the project root:

```dotenv
# ── Required ──────────────────────────────────────────────
GROQ_API_KEY=gsk_...

# ── Database ───────────────────────────────────────────────
PG_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/risk_db

# ── Redis ──────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379
REDIS_CACHE_INDEX=idx:risk_cache
REDIS_CACHE_PREFIX=cache:policy:
CACHE_TTL_SECONDS=86400              # 24 hours
CACHE_DISTANCE_THRESHOLD=0.30        # cosine distance cutoff for a cache hit

# ── LLM ────────────────────────────────────────────────────
LLM_MODEL=openai/gpt-oss-120b

# ── Retrieval tuning ───────────────────────────────────────
RETRIEVAL_TOP_K=50                   # chunks returned from hybrid retrieval
RERANK_TOP_N=12                      # top-N passed to LLM after cross-encoder reranking

# ── Upload ─────────────────────────────────────────────────
MAX_UPLOAD_BYTES=26214400            # 25 MB
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with the `pgvector` extension
- Redis Stack (includes RediSearch for vector index support)
- A [Groq API key](https://console.groq.com)

### 1 — PostgreSQL + pgvector

```bash
# Fedora / RHEL
sudo dnf install pgvector_15

# Or build from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install

# Create the database
psql -U postgres -c "CREATE DATABASE risk_db;"
psql -U postgres -d risk_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2 — Redis Stack

```bash
# Docker (easiest)
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```

### 3 — Python Environment

```bash
python3 -m venv env
source env/bin/activate

pip install fastapi "uvicorn[standard]" pydantic python-dotenv \
            psycopg psycopg-pool \
            pgvector pdfplumber \
            sentence-transformers \
            "redis[hiredis]" groq numpy
```

### 4 — Initialise the Database

```bash
python init_db.py
# ✓ Database initialized — policy_chunks table with HNSW + GIN indexes ready
```

### 5 — (Optional) Pre-ingest Dev Policies

Place PDFs in `raw_policies/` and run:

```bash
python ingest_policies.py
```

This ingests under the `legacy` session ID and is useful for testing retrieval without going through the upload flow.

---

## Running the Application

```bash
uvicorn gateway:app --reload --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

> **Note:** The `--reload` flag restarts the server automatically on code changes. Because sessions are held in memory, any restart clears active sessions — previously ingested chunks remain in PostgreSQL but must be re-uploaded through the UI.

---

## User Guide (Step-by-Step)

### Step 1 — Upload a Policy

1. Open **http://127.0.0.1:8000**
2. Drag a PDF onto the dropzone or click **Choose PDF**
3. The title is pre-filled from the filename; edit it and optionally set a policy type label
4. Click **Index policy →**

### Step 2 — Watch the Ingestion Pipeline

The sidebar shows a live animated panel:

```
✓ Uploading file to server
⟳ Chunking & extracting tables
  Generating embeddings
  Indexing into PostgreSQL + Redis
  Verifying index & signalling ready
```

When the backend completes all steps, a **"Policy indexed!"** modal appears.

### Step 3 — Ask Audit Questions

Click **Start auditing →** to dismiss the modal. The chat input unlocks.

1. Type a question and press **Send** or **Ctrl / Cmd + Enter**
2. Watch the badge in the conversation header:
   - `⟳ Cache miss — live RAG` → tokens stream in; a structured verdict card appears when complete
   - `⚡ Redis cache hit · dist 0.xxx` → answer appears instantly from cache
3. If any LLM citations fail verification, a red `⚠ HALLUCINATION DETECTED` banner appears in the response card

---

## Query Tips for CGL Policies

The cross-encoder reranker rewards **specific, focused questions** whose vocabulary directly matches the source clause. Broad multi-section queries cause it to surface mixed-relevance chunks instead of the core insuring agreement.

**Works well ✅**

```
Does this policy pay when the insured is legally liable for bodily injury or property damage?
What is the duty to defend under this commercial general liability policy?
Is bodily injury from intentional acts covered?
What is the liquor liability exclusion in this policy?
Is pollution liability excluded under Coverage A?
What is the general aggregate limit under Section III?
What is the per occurrence limit?
What is the Damage To Premises Rented To You limit?
What does personal and advertising injury liability cover under Coverage B?
Who qualifies as a named insured under Section II?
What are the conditions under Section IV for coverage to apply?
What supplementary payments does this policy make?
```

**Avoid ❌**

```
What does Coverage A cover and what are all its exclusions and conditions?
```

This asks about multiple sections simultaneously. Break it into separate focused questions for reliable results.

---

## Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| **Fixed-window chunking** | Clause headers sometimes split across chunk boundaries | Ask specific questions; ±1 neighbour window partially mitigates this |
| **In-memory sessions** | Server restart clears session map; uploaded documents stay in PostgreSQL but the session is lost | Re-upload the PDF after restarting the server |
| **Hardcoded jurisdiction / year in UI** | UI sends `Unknown` jurisdiction and `2024` effective year | Pass custom values directly via the REST API |
| **No authentication** | Any client can create sessions and query | Add API-key middleware for production deployments |
| **`OUT_OF_SCOPE` on broad queries** | Cross-encoder may not surface the insuring agreement for queries that also mention exclusions and conditions | Rephrase as a focused question (see Query Tips above) |

---

## Design Decisions

See [`decisions.md`](decisions.md) for full Architecture Decision Records. Key choices summarised:

| Decision | Rationale |
|---|---|
| **Hybrid retrieval (dense + sparse)** | Dense alone misses exact legal terminology; sparse alone misses semantic paraphrases. RRF fusion gets the best of both without tuning score scales. |
| **Cross-encoder reranker** | Bi-encoder embeddings are approximate. A cross-encoder reads query + document jointly and re-scores with much higher precision, reducing irrelevant context reaching the LLM. |
| **`RERANK_TOP_N = 12`** | Raising from the original 5 ensures broad multi-topic queries still surface the core insuring agreement, which the cross-encoder ranks lower due to mixed chunk boundaries. |
| **Semantic cache over exact-match** | The same question rephrased slightly should still be a cache hit. Cosine distance ≤ 0.30 captures paraphrases while rejecting genuinely different questions. |
| **Always cache regardless of grounding failures** | Blocking cache writes on hallucination warnings caused every query with any citation warning to re-run the full RAG pipeline indefinitely. Grounding failures are informational — caching and citation checking are independent concerns. |
| **`response_format: json_object`** | Forces the LLM to return strict JSON without markdown fences, making Pydantic validation reliable and token streaming clean. |
| **SSE over WebSockets** | SSE is unidirectional (server → client) and simpler to implement. The browser's native `fetch` + `ReadableStream` API handles it without a library. WebSockets would only add value if the client needed to interrupt a stream mid-flight. |
| **Vanilla JS frontend** | No build step, no node_modules, instant load. The project is an API-first backend — the UI is a demo surface, not the product. |

---

*Built with FastAPI · pgvector · Redis Stack · sentence-transformers · Groq*
