# System Execution & Data Lifecycle Flow

> Updated after each phase. The ASCII diagrams show the **current** state of the system.

---

## Phase 1 — Ingestion Flow (Current)

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Raw PDF Files  │────▶│  pdfplumber Parser    │────▶│  Dual Extraction     │
│  raw_policies/  │     │  (bounding boxes)    │     │  ┌────────────────┐  │
│  .pdf           │     │                      │     │  │ Tables: T|H|M  │  │
│                 │     │  extract_tables()    │     │  │ Text: raw str  │  │
└─────────────────┘     └──────────────────────┘     │  │ is_table flag  │  │
                                                    │  └────────────────┘  │
                                                    └─────────┬────────────┘
                                                              │
                                                    ┌─────────▼────────────┐
                                                    │  Markdown Serializer │
                                                    │  (table → MD table)  │
                                                    │  (text → plain)      │
                                                    └─────────┬────────────┘
                                                              │
                                                    ┌─────────▼────────────┐
                                                    │  Sliding Window      │
                                                    │  Chunker             │
                                                    │  (60-80w, 15w ov)   │
                                                    │  + clause_id extract │
                                                    └─────────┬────────────┘
                                                              │
                                                    ┌─────────▼────────────┐
                                                    │  Embedding Generator │
                                                    │  (all-MiniLM-L6-v2) │
                                                    │  normalize=True      │
                                                    └─────────┬────────────┘
                                                              │
                                                    ┌─────────▼────────────┐
                                                    │  PostgreSQL          │
                                                    │  policy_chunks table │
                                                    │  + pgvector HNSW idx │
                                                    │  + tsvector GIN idx  │
                                                    └──────────────────────┘
```

### Phase 1 Data Schema

```
policy_chunks:
  id            BIGSERIAL PK
  document_id   TEXT          -- "policy_acme_2024.pdf"
  policy_type   TEXT          -- "Cyber", "Commercial Property", "GL"
  jurisdiction  TEXT          -- "US-NY", "UK", "EU-DE"
  effective_year INT          -- 2024
  page_number   INT
  clause_id     TEXT          -- "Section 4.1(b)" — extracted
  is_table      BOOLEAN       -- True/False (structural routing)
  content       TEXT          -- Markdown table or plain text
  content_tsv   tsvector      -- GENERATED ALWAYS (full-text index)
  embedding     vector(384)   -- normalized MiniLM embedding
```

### Phase 1 Component Responsibilities

| Component | Responsibility |
|---|---|
| `pdfplumber` | Extract tables (bounding box detection) and text from each page |
| Markdown serializer | Preserve column relationships in table rows |
| Sliding window chunker | Segment prose into 60-80 word chunks with 15-word overlap |
| Clause extractor | Regex pattern on chunk header (first 200 chars) |
| Embedding generator | 384-dim normalized vectors via all-MiniLM-L6-v2 |
| PostgreSQL | Store chunks + metadata + vectors + tsvector; HNSW + GIN indexes |

---

## Phase 1 — COMPLETE ✅

- 1,139 chunks ingested across 2 PDFs
- 22 table chunks (Markdown format verified)
- 0 NULL embeddings
- HNSW + GIN indexes active

---

## Phase 1 Status: COMPLETE ✅

Skeleton files created:
- ✅ `decisions.md` — ADRs 001-005 recorded
- ✅ `flow.md` — ingestion flow diagram current
- ✅ `01_init_db.py` — schema + indexes (complete)
- ⏳ `02_ingest_policies.py` — Tasks 1-4 are user challenges (marked `██ YOUR CODE HERE ██`)

### Phase 1 Verification (after user completes 02_ingest_policies.py)

```sql
SELECT document_id, jurisdiction, policy_type,
       COUNT(*) AS total_chunks,
       SUM(CASE WHEN is_table THEN 1 ELSE 0 END) AS table_chunks
FROM policy_chunks
GROUP BY document_id, jurisdiction, policy_type;
```

Then inspect a table chunk:
```sql
SELECT content FROM policy_chunks WHERE is_table = TRUE LIMIT 3;
```
Output must show `|` separated Markdown, not collapsed text.

---

## Phase 2 — COMPLETE ✅

- RRF + dense + sparse hybrid search working
- Jurisdiction leak test passes (all results match filter)
- 10 results retrieved with proper ranking (Dist + RRF scores)
- Async connections throughout (no event loop blocking)

---

## Phase 2 — COMPLETE ✅

### Phase 2 Execution Flow

```
┌─────────────────┐
│  Incoming Query  │
│  (text + vector) │
└────────┬────────┘
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐
│  Filtered Dense │   │  Filtered Sparse     │
│  SEARCH         │   │  SEARCH              │
│                 │   │                      │
│ WHERE clause    │   │ WHERE clause         │
│ runs FIRST      │   │ runs FIRST           │
│ (jurisdiction,  │   │ (jurisdiction,       │
│  year, type)    │   │  year, type)         │
│                 │   │ AND tsvector match   │
│ THEN vector     │   │ THEN ts_rank         │
│ distance sort   │   │ sort                 │
│ LIMIT K         │   │ LIMIT K              │
└────────┬────────┘   └──────────┬───────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│  Reciprocal Rank Fusion (RRF, k=60)     │
│  Merge both ranked lists by:            │
│  score = 1/(rank_dense + k) +          │
│          1/(rank_sparse + k)           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Top-K Merged Results                   │
│  Each with: id, content, score,         │
│  document_id, page_number, clause_id    │
│  ALL guaranteed within jurisdiction     │
└─────────────────────────────────────────┘
```

### Phase 2 Key Principles

| Principle | Why |
|---|---|
| **Predicate pushdown** | WHERE runs BEFORE ORDER BY distance — DB enforces isolation |
| **Parallel dense+sparse** | asyncio.gather — no data dependency between them |
| **RRF k=60** | Balances rank position vs. list presence — tuned for top-10 lists |
| **Same metadata filter on both** | RRF compares apples to apples — both lists jurisdiction-scoped |

### Phase 2 Component Responsibilities

| Component | Responsibility |
|---|---|
| `async_filtered_dense_search` | pgvector with WHERE clause, distance-sorted |
| `async_filtered_sparse_search` | tsvector with WHERE clause, rank-sorted |
| `reciprocal_rank_fusion` | Merge two ranked lists into one |
| `retrieve_filtered` | Parallel execution + RRF orchestration |

---

## Phase 3 — Will Add: Cross-Encoder & Pydantic Citation Flow

_TBD after Phase 3 completion._

---

## Phase 3 — Will Add: Cross-Encoder & Pydantic Citation Flow

_TBD after Phase 3 completion._

---

## Phase 4 — Will Add: Redis Cache & SSE Gateway Flow

_TBD after Phase 4 completion._
