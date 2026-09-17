# Architectural Decision Records (ADRs)

> Every non-trivial engineering choice in this project gets an entry. Append, don't overwrite.

---

## [ADR-001] Markdown Table Serialization (not JSON or raw join)

- **Context & Problem:** Insurance PDFs contain multi-column tables (coverage type, limit, deductible). Raw text join destroys column relationships (`Cyber Liability $5M $100K` is uninterpretable). JSON preserves structure but embedding models tokenize raw text, not JSON keys.
- **Decision Taken:** Serialize tables to Markdown pipe format: `| Coverage | Limit | Deductible |`. Column adjacency is preserved in the token sequence, so the embedding model captures "Cyber Liability → $5M limit" as a semantic unit.
- **Alternatives Considered:**
  - Raw `.join(" ")` — destroys column structure. Rejected.
  - JSON `{"coverage": "Cyber Liability", "limit": "$5M"}` — embedding ignores key semantics. Rejected.
  - CSV — same adjacency problem as raw join, no visual delimiter. Rejected.
- **Trade-offs & Latency/Memory Impact:** Markdown adds ~3% token overhead per table chunk vs. raw join. Negligible memory cost. Preserves the single most important signal: column co-occurrence.

---

## [ADR-002] `is_table` Boolean Flag (not a separate table)

- **Context & Problem:** Table chunks and prose chunks need different chunking strategies (tables shouldn't be split across rows; prose benefits from sliding window). But Phase 2 requires a single vector index and single RRF merge across all chunks.
- **Decision Taken:** Single `policy_chunks` table with `is_table BOOLEAN` flag. Table chunks get `is_table=True` and store Markdown; prose chunks get `is_table=False` and store plain text.
- **Alternatives Considered:**
  - Separate `policy_table_chunks` and `policy_text_chunks` tables — requires UNION or two indexes in every Phase 2 query. Rejected for simplicity.
  - `chunk_type ENUM('table','prose','header')` — over-specced for Phase 1's two-type need. Boolean is sufficient.
- **Trade-offs & Latency/Memory Impact:** Zero-join filtering in Phase 2 (`WHERE is_table = TRUE AND ...`). Single HNSW index search. Boolean is 1 byte vs. ENUM's 4 bytes. Simpler schema now, still semantically clear.

---

## [ADR-003] Clause Header Extraction — First 200 Chars Only

- **Context & Problem:** Each chunk needs a `clause_id` (e.g., "Section 4.1(b)") for citation traceability. Clause headers appear at the start of sections, not randomly in body text.
- **Decision Taken:** Run `CLAUSE_PATTERN` regex only on the first 200 characters of each chunk. Body text often contains references like "per Section 3.2" that would create false positives.
- **Alternatives Considered:**
  - Full-text scan — too noisy. Rejected.
  - Dedicated header line detection (lines matching `^(Section|Article|Clause)\s+...`) — good but requires line-by-line parsing. The 200-char window is faster and covers the typical header position.
- **Trade-offs & Latency/Memory Impact:** Constant-time regex on fixed-length window. May miss clause headers that appear after a long preamble (>200 chars), but those are rare in insurance policy formatting.

---

## [ADR-004] Sliding Window: 70 Words / 15 Word Overlap

- **Context & Problem:** Prose chunks must be semantically self-contained for retrieval, but small enough to fit in an LLM context window.
- **Decision Taken:** 70-word target chunks with 15-word overlap. Overlap prevents clause boundaries from splitting a key concept across two chunks.
- **Alternatives Considered:**
  - Fixed sentence splitting — sentences vary wildly in length (one insurance sentence can be 80 words). Rejected.
  - 100-word chunks — too large for the cross-encoder context window in Phase 3. Rejected.
  - 0 overlap — boundary concepts get split and lose meaning. Rejected.
- **Trade-offs & Latency/Memory Impact:** 15-word overlap increases total chunk count by ~20% vs. 0 overlap, but prevents retrieval failures at clause boundaries. Net positive for recall at minimal storage cost.

---

## [ADR-006] Predicate Pushdown: WHERE Before ORDER BY Distance

- **Context & Problem:** Vector similarity search returns nearest neighbors globally. Filtering in application code (post-filter) reads unauthorized rows into memory — a data leak vector at enterprise scale.
- **Decision Taken:** Metadata filters (jurisdiction, year, policy_type) live in the SQL `WHERE` clause, which executes BEFORE `ORDER BY distance`. The database engine enforces isolation, not application logic.
- **Alternatives Considered:**
  - Post-filter in Python — data leak risk, wasted memory/compute. Rejected.
  - Separate tables per jurisdiction — schema complexity, cross-tenant queries break. Rejected.
  - Application-level middleware filtering — bypasses DB indexes, slow. Rejected.
- **Trade-offs & Latency/Memory Impact:** pgvector HNSW still does approximate search — it searches the index, then filters. For very large corpora requiring strict isolation, sequential scan with exact filtering is safer. At demo scale, HNSW + WHERE is the right trade-off: <5ms query time with guaranteed isolation.

---

## [ADR-007] Parallel Dense + Sparse via asyncio.gather

- **Context & Problem:** Dense (semantic) and sparse (keyword) searches are independent — neither needs results from the other to start.
- **Decision Taken:** Fire both searches concurrently with `asyncio.gather()`. Merge results only after both complete, via RRF.
- **Alternatives Considered:**
  - Sequential (dense first, then sparse) — adds ~100ms latency unnecessarily. Rejected.
  - Single combined query — psycopg doesn't support multi-index queries cleanly. Rejected.
- **Trade-offs & Latency/Memory Impact:** Total latency = max(dense_time, sparse_time) instead of sum. Memory holds two ranked lists briefly during RRF — negligible for top-K=10 each.

---

## [ADR-008] RRF k=60 for Rank Fusion

- **Context & Problem:** Dense and sparse rankings use different scoring units (distance vs. rank). A naive merge (average rank) over-rewards lists with fewer results.
- **Decision Taken:** Reciprocal Rank Fusion with k=60. Score = 1/(rank+k) per list. k=60 was benchmarked against k=20, k=100, k=200 — it provided the best balance between promoting cross-list matches and not over-flattening scores.
- **Alternatives Considered:**
  - Weighted average of ranks — sensitive to list length differences. Rejected.
  - k=20 — too aggressive, over-promotes low-rank cross-list matches. Rejected.
  - k=200 — too flat, high-ranked items barely differentiated. Rejected.
- **Trade-offs & Latency/Memory Impact:** Pure arithmetic — O(n) where n = unique IDs across both lists. No external dependencies. At top-K=10 from each list, n ≤ 20, trivially fast.

---

## [ADR-009] Async psycopg Throughout — No Sync Calls in Async Functions

- **Context & Problem:** Using `psycopg.connect()` (sync) inside `async def` functions blocks the event loop. In a gateway handling concurrent SSE streams, this would serialize all DB queries — defeating asyncio entirely.
- **Decision Taken:** Use `psycopg.AsyncConnection.connect()` for all DB access inside async functions. Connections are opened per-request and closed in `finally` blocks. For production, upgrade to `AsyncConnectionPool`.
- **Alternatives Considered:**
  - `asyncio.to_thread()` wrapper around sync psycopg — works but adds thread-pool overhead. Rejected.
  - Sync psycopg in async functions — blocks event loop, serializes queries. Rejected (this was the bug we caught).
- **Trade-offs & Latency/Memory Impact:** Slightly more connection overhead per request (vs. pool). At <100 concurrent requests, negligible. Pool will be added in Phase 4 for the gateway.

---

## [ADR-010] `dict_row` Factory for Named Column Access

- **Context & Problem:** psycopg default returns tuples (`r[0], r[1], ...`). With 7+ columns and multiple SELECT patterns, index tracking is error-prone (the KeyError we caught proves it).
- **Decision Taken:** `row_factory=dict_row` — access columns by name (`r["id"]`, `r["content"]`) everywhere.
- **Alternatives Considered:**
  - Named tuples — more verbose, no stdlib support in psycopg. Rejected.
  - Index constants (`COL_ID = 0, COL_CONTENT = 5`) — maintainable but ugly. Rejected.
- **Trade-offs & Latency/Memory Impact:** Negligible — dict_row is the standard psycopg approach. One-line setting change.

---

## [ADR-011] AsyncConnectionPool — Reuse, Don't Recreate

- **Context & Problem:** Each search function previously opened a brand-new `AsyncConnection`. At 200 concurrent requests × 2 searches = 400 simultaneous connections, PostgreSQL's `max_connections` (default 100) is exceeded and connections are rejected.
- **Decision Taken:** `psycopg_pool.AsyncConnectionPool` with `min_size=2`, `max_size=10`. Connections are reused across requests. Pool lazily initializes on first use, cleanly shuts down at exit.
- **Alternatives Considered:**
  - Per-request `AsyncConnection` — breaks at scale (this was the original bug). Rejected.
  - `asyncio.to_thread()` with sync connections — works but wastes threads. Rejected.
  - Hard-coded connection limit via `Semaphore` — doesn't solve DB-side exhaustion. Rejected.
- **Trade-offs & Latency/Memory Impact:** Pool adds ~5ms warmup on first use (negligible after that). Memory: 10 connections × ~256KB each = ~2.5MB. Net positive: 400-request concurrency handled by 10 physical connections.

---

## [ADR-012] Cross-Encoder Reranking Before LLM

- **Context & Problem:** RRF returns top-15 candidates by combined rank. Some may be topically related but factually irrelevant (e.g., discusses the topic but doesn't answer the specific question). Feeding these to the LLM wastes tokens and increases hallucination risk.
- **Decision Taken:** Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks top-15 → top-3 by relevance. Only top-3 reach the LLM.
- **Alternatives Considered:**
  - Skip reranking, feed top-15 to LLM — wastes context window, increases hallucination. Rejected.
  - Embedding-based reranking — same signal as Phase 2, no improvement. Rejected.
  - Cross-Encoder on full corpus — ~100x too slow. Rejected.
- **Trade-offs & Latency/Memory Impact:** Cross-Encoder is CPU-bound and adds ~200-500ms for top-15 pairs (vs. 5ms for embedding search). But it reduces LLM input from ~15K to ~3K tokens, saving LLM latency and cost. Net latency neutral, quality significantly improved.

---

## [ADR-013] LLM Output Enforcement via Pydantic + `response_format`

- **Context & Problem:** LLMs return freeform text by default. For compliance audit verdicts, we need structured fields (status, financial_limit, citations) — not prose. Hallucinated clause numbers or fabricated limits are legal liabilities.
- **Decision Taken:** Two-layer enforcement:
  1. `response_format={"type": "json_object"}` — Groq API forces JSON output
  2. `AuditVerdict.model_validate_json()` — Pydantic validates types, enforces non-empty citations, valid literals
- **Alternatives Considered:**
  - Parse JSON with `json.loads()` + manual validation — no type safety, misses edge cases. Rejected.
  - Fine-tune LLM to output schema — expensive, model-specific, still error-prone. Rejected.
- **Trade-offs & Latency/Memory Impact:** Negligible — Pydantic validation is O(n) fields. One failed parse → ValueError → can retry or return OUT_OF_SCOPE. Adds ~1ms per validation.





- **Context & Problem:** During rapid iteration in Phase 1-3, schema changes are frequent. Migrations add overhead.
- **Decision Taken:** `01_init_db.py` executes `DROP TABLE IF EXISTS policy_chunks` before CREATE. This is **development-only** — production will need a migration tool (Alembic or Flyway).
- **Alternatives Considered:**
  - idempotent CREATE with column checks — complex, error-prone at this stage. Rejected.
  - Always keep data, ALTER on change — risky with vector column changes. Rejected.
- **Trade-offs & Latency/Memory Impact:** Zero migration overhead during development. Loses data on re-init, which is fine since ingestion is deterministic and takes seconds.
