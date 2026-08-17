# Basic RAG Service

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector_HNSW-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Celery](https://img.shields.io/badge/Celery-Distributed_Tasks-37814A.svg?logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: BasedPyright](https://img.shields.io/badge/type%20check-basedpyright-informational.svg)](https://github.com/detachhead/basedpyright)

A production-grade, enterprise-ready Retrieval-Augmented Generation (RAG) backend engine designed around **transactional durability, asynchronous processing pipelines, and strict consistency invariants**.

Unlike naive RAG prototypes that couple HTTP lifecycles to LLM inference or perform fragile dual-writes to message brokers, this service implements a **Transactional Outbox Pattern** with **idempotent worker consumption**, **pgvector HNSW cosine similarity search**, **HMAC-signed cursor pagination**, and **resilient Server-Sent Events (SSE) token streaming with active heartbeat monitoring**.

---

## Table of Contents

- [System Architecture](#system-architecture)
  - [Core Distributed Invariants](#core-distributed-invariants)
  - [End-to-End Sequence Flow](#end-to-end-sequence-flow)
- [Technical Stack Matrix](#technical-stack-matrix)
- [Data Pipeline & Ingestion Lifecycle](#data-pipeline--ingestion-lifecycle)
  - [Document State Machine](#document-state-machine)
  - [Transactional Outbox Dispatch](#transactional-outbox-dispatch)
  - [Failure Classification & Retry Policies](#failure-classification--retry-policies)
- [Retrieval & Inference Engine](#retrieval--inference-engine)
  - [Vector Indexing Strategy](#vector-indexing-strategy)
  - [SSE Streaming & Backpressure Safeguards](#sse-streaming--backpressure-safeguards)
  - [Cryptographic Cursor Pagination (v2 API)](#cryptographic-cursor-pagination-v2-api)
- [API Surface Reference](#api-surface-reference)
- [Operational Runbook & SRE Triage](#operational-runbook--sre-triage)
- [Local Development & Engineering Workflow](#local-development--engineering-workflow)

---

## System Architecture

```mermaid
flowchart TB
    subgraph Ingestion ["Ingestion & Task Dispatch Pipeline"]
        Client["Client / SDK"] -->|"1. Multipart Upload"| API["FastAPI Application"]
        API -->|"2. Write File"| Storage["Local / Object Storage"]
        API -->|"3. Atomic Transaction"| DB[("PostgreSQL<br/>Document + Outbox Row")]
        
        Beat["Celery Beat<br/>Periodic Tick"] -->|"4. Trigger"| OutboxTask["outbox.publish<br/>Maintenance Queue"]
        OutboxTask -->|"5. Claim via SKIP LOCKED"| DB
        OutboxTask -->|"6. Dispatch Task"| Redis[("Redis Broker<br/>Priority Queues")]
        
        Redis -->|"7. Consume"| Worker["Celery Worker<br/>Documents Queue"]
        Worker -->|"8. SELECT FOR UPDATE"| DB
        Worker -->|"9. Generate Embeddings"| GeminiEmbed["Google Gemini<br/>gemini-embedding-001"]
        Worker -->|"10. Store 768d Chunks"| DB
    end

    subgraph RetrievalEngine ["Retrieval & Chat Engine"]
        Client -->|"11. Chat Request"| API
        API -->|"12. Embed Query"| GeminiEmbed
        API -->|"13. Vector Cosine Search"| DB
        API -->|"14. Assemble Context & Stream"| GeminiChat["Google Gemini<br/>gemini-2.5-flash"]
        API -->|"15. Stream SSE Events"| Client
    end
```

### Core Distributed Invariants

| Architectural Principle | Problem Solved | Implementation Detail |
| :--- | :--- | :--- |
| **Transactional Outbox** | Eliminates dual-write inconsistency between PostgreSQL commits and Redis broker publishes. | Document creation and `OutboxEvent` creation occur within the exact same ACID transaction. Broker dispatches are decoupled via `outbox.publish`. |
| **Row-Level Concurrency Locks** | Prevents concurrent duplicate worker processing and race conditions during re-ingestion. | Workers acquire exclusive locks using `SELECT ... FOR UPDATE` on `documents` rows during state transitions. |
| **Leased Batch Event Claiming** | Eliminates publisher race conditions and prevents multiple schedulers from double-publishing. | The outbox repository uses `SELECT ... FOR UPDATE SKIP LOCKED` combined with UUID claim tokens and time-bound leases (300s). |
| **Worker Queue Separation** | Prevents high-volume document ingestion from starving operational maintenance tasks. | Celery queues are isolated into `documents` (heavy CPU/IO worker pool) and `maintenance` (outbox dispatch & future reaper jobs). |
| **Cryptographic Pagination** | Prevents parameter tampering, deep-offset performance degradation, and drift during continuous ingestion. | Uses HMAC-SHA256 signed cursors encoding opaque timestamp and primary key boundary vectors with dual-key rotation support. |
| **Defensive SSE Streaming** | Prevents zombie connection leaks and hung client connections during upstream AI provider degradation. | Guarded by first-token timeouts (20s), idle heartbeat pings (15s), total stream cutoffs (120s), and buffer ceiling limits (64KB). |

---

### End-to-End Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Client Application
    participant API as FastAPI Gateway
    participant DB as PostgreSQL
    participant Beat as Celery Beat
    participant Worker as Celery Worker
    participant AI as Google Gemini API

    Note over User,DB: 1. Ingestion Phase (Synchronous HTTP Write Path)
    User->>API: POST /api/v1/knowledge-bases/{kb_id}/documents (File Upload)
    API->>API: Validate MIME, Calculate SHA-256 Hash
    API->>DB: BEGIN TX: Insert Document (PENDING) + Insert OutboxEvent
    DB-->>API: COMMIT TX
    API-->>User: 201 Created (Document ID, Status: PENDING)

    Note over Beat,Worker: 2. Outbox Relay Phase (Asynchronous Publisher Loop)
    Beat->>Worker: Trigger outbox.publish (Every 5s on maintenance queue)
    Worker->>DB: SELECT * FROM outbox_events WHERE published_at IS NULL FOR UPDATE SKIP LOCKED
    Worker->>Worker: Enqueue document.process to documents queue in Redis
    Worker->>DB: UPDATE outbox_events SET published_at = NOW() WHERE claim_token = :token

    Note over Worker,AI: 3. Ingestion & Vectorization Phase (Heavy Processing Worker)
    Worker->>DB: SELECT * FROM documents WHERE id = :id FOR UPDATE
    Worker->>DB: UPDATE documents SET status = 'PROCESSING'
    Worker->>Worker: Parse PDF/Text & Split Chunks (Recursive Text Splitter)
    Worker->>AI: Batch Generate Embeddings (768 Dimensions)
    AI-->>Worker: Return Vector Embeddings
    Worker->>DB: Bulk INSERT INTO document_chunks (HNSW Indexed)
    Worker->>DB: UPDATE documents SET status = 'READY'
```

---

## Technical Stack Matrix

| Layer | Technology | Version / Configuration | Design Rationale |
| :--- | :--- | :--- | :--- |
| **Runtime** | Python | `>= 3.14` | Modern type system features (`type` aliases, generic syntax `[T]`), high performance. |
| **Web Framework** | FastAPI + Uvicorn | `0.140+` | Asynchronous native request handling, automatic OpenAPI schema generation, clean dependency injection. |
| **Database & ORM** | PostgreSQL + SQLAlchemy | `2.0+` (via `psycopg3`) | Strict transactional semantics, connection pooling, typed query expressions, JSONB metadata filtering. |
| **Vector Engine** | pgvector | `0.5.0+` (HNSW Index) | Cosine similarity (`vector_cosine_ops`), avoiding multi-database operational overhead at current scale. |
| **Task Broker & Queue** | Celery + Redis | Celery `5.6+`, Redis `8.1+` | Distributed task execution, automatic retries with exponential backoff & jitter, multi-queue topology. |
| **LLM & Embeddings** | Google Gemini | `gemini-embedding-001` (768d)<br>`gemini-2.5-flash` | Enterprise-grade context windows, high token velocity, cost-efficient vector representation. |
| **Streaming Protocol** | Server-Sent Events | `sse-starlette` | Standard HTTP streaming for unidirectional tokens without WebSocket state overhead. |
| **Tooling & Linter** | uv, Ruff, BasedPyright | Latest | Ultra-fast virtual environment sync, strict static typing, zero-tolerance code style enforcement. |

---

## Data Pipeline & Ingestion Lifecycle

### Document State Machine

```mermaid
stateDiagram-v2
    direction LR
    [*] --> PENDING: Multipart Upload & Outbox Staged
    PENDING --> PROCESSING: Worker Claim (SELECT FOR UPDATE)
    PROCESSING --> READY: Embeddings Generated & Indexed
    PROCESSING --> FAILED: Permanent Error / Retries Exhausted
    FAILED --> PENDING: Operational Reprocess / Retry
    READY --> [*]
```

* **`PENDING`**: File persisted to disk/storage, record committed to DB, outbox event staged.
* **`PROCESSING`**: Document claimed by worker via `SELECT ... FOR UPDATE`. Deduplication verified, text extracted, chunked, and embeddings generated.
* **`READY`**: Vector chunks committed to `document_chunks` table with HNSW index. Document is live for semantic search.
* **`FAILED`**: Non-recoverable error encountered or transient retry ceiling reached. `last_error` and `retry_count` logged to database.

---

### Transactional Outbox Dispatch

To eliminate race conditions between multiple Celery beat executions or distributed publisher workers, outbox processing uses lease-based, non-blocking row claiming:

```sql
SELECT id, event_type, aggregate_id, payload
FROM outbox_events
WHERE published_at IS NULL
  AND (claimed_at IS NULL OR claimed_at < NOW() - INTERVAL '300 seconds')
ORDER BY created_at ASC, id ASC
LIMIT 100
FOR UPDATE SKIP LOCKED;
```

#### Claim & Fencing Token Protocol
1. The publisher generates an ephemeral `claim_token = uuid4()`.
2. Staged rows are claimed with `claimed_at = NOW()` and `claim_token = :token`.
3. Tasks are dispatched to Redis.
4. Rows are marked published only if `claim_token` still matches:
   ```sql
   UPDATE outbox_events
   SET published_at = NOW(), claimed_at = NULL, claim_token = NULL, last_error = NULL
   WHERE id = :event_id AND published_at IS NULL AND claim_token = :claim_token;
   ```

---

### Failure Classification & Retry Policies

Failures are strictly categorized into **Transient** (retryable) and **Permanent** (non-retryable):

```mermaid
flowchart TB
    Ex["Ingestion Exception"] -->|"Recoverable Failure"| TransBranch["Transient Exception"]
    Ex -->|"Fatal Failure"| PermBranch["Permanent Exception"]

    subgraph Transient ["Transient Category (Retryable)"]
        TransBranch --> T1["• Gemini API 429 Rate Limit<br/>• AI Provider 502 / 503 / 504<br/>• Database Lock / Deadlock<br/>• Temporary Network Timeout"]
        T1 --> RetryAction["Celery Exponential Backoff & Jitter<br/>Backoff: 2s ➔ 4s ➔ 8s | Max Retries: 3"]
    end

    subgraph Permanent ["Permanent Category (Non-Retryable)"]
        PermBranch --> P1["• Corrupted / Unreadable File<br/>• Empty Text Extraction<br/>• 401 / 403 Authentication Error<br/>• Vector Dimension Mismatch"]
        P1 --> FailAction["Transition Document to FAILED<br/>Persist last_error to DB | No Retry"]
    end
```

The Celery ingestion task uses a custom `DocumentProcessTask` base class that hooks into `on_failure` to automatically transition documents to `FAILED` and record the truncated error message in PostgreSQL once the retry budget is exhausted.

---

## Retrieval & Inference Engine

### Vector Indexing Strategy

Document chunks are mapped into a 768-dimensional space using `gemini-embedding-001` (configurable via `GEMINI_EMBEDDING_MODEL`). Retrieval runs through an **HNSW index** optimized for cosine distance:

```sql
CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

Search queries enforce status validation, ensuring **only chunks from `READY` documents in the target knowledge base** are evaluated:

```sql
SELECT dc.id, dc.content, dc.metadata, (1 - (dc.embedding <=> :query_vector)) AS similarity
FROM document_chunks dc
JOIN documents d ON d.id = dc.document_id
WHERE d.knowledge_base_id = :kb_id
  AND d.status = 'ready'
  AND (1 - (dc.embedding <=> :query_vector)) >= :threshold
ORDER BY dc.embedding <=> :query_vector
LIMIT :top_k;
```

---

### SSE Streaming & Backpressure Safeguards

The streaming chat endpoint (`/api/v1/conversations/{id}/messages/stream`) adheres to a structured event envelope:

```
event: metadata
data: {"conversation_id": "...", "message_id": "..."}

event: citations
data: [{"document_id": "...", "chunk_id": "...", "snippet": "...", "score": 0.89}]

event: token
data: {"content": "Retrieval"}

event: token
data: {"content": "-Augmented"}

event: complete
data: {"finish_reason": "stop", "total_tokens": 342}
```

#### Defensive Streaming Controls
* **First-Token Watchdog**: Terminates with `event: error` if the upstream LLM takes $>20\text{s}$ to yield initial output.
* **Idle-Stream Watchdog**: Terminates if chunks stall for $>30\text{s}$ mid-generation.
* **Stream Heartbeat Ping**: Transmits `: ping` comment every 15 seconds to prevent intermediate proxy/load-balancer connection drops.
* **Deferred Persistence**: Assistant responses are committed to PostgreSQL **only after complete, verified stream finalization**, preventing corrupted or partial answers from polluting conversation history.

---

### Cryptographic Cursor Pagination (v2 API)

The v2 endpoints implement opaque, stateless, signed cursor pagination. Cursors are generated using **HMAC-SHA256** digests:

$$\text{Cursor} = \text{Base64UrlEncode}\Big(\text{JSON}(\text{Payload}) \,\|\, \text{HMAC-SHA256}_{\text{key}}(\text{Payload})\Big)$$

* **Tamper Proof**: Clients cannot alter sort keys, resource scopes, or offsets.
* **Dual-Key Rotation**: Supports zero-downtime key rotation via `CURSOR_SIGNING_KEY` (primary signer) and `CURSOR_PREVIOUS_SIGNING_KEY` (fallback verifier).

---

## API Surface Reference

### Core Endpoints

| Method | Route | Description | Status Code | Deprecation |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Liveness and health check | `200 OK` | — |
| `POST` | `/api/v1/knowledge-bases/{kb_id}/documents` | Multipart document upload (Enqueue ingestion) | `201 Created` | — |
| `POST` | `/api/v1/knowledge-bases/{kb_id}/search` | Semantic vector search | `200 OK` | — |
| `POST` | `/api/v1/knowledge-bases/{kb_id}/conversations` | Initialize knowledge base conversation | `201 Created` | — |
| `GET` | `/api/v1/conversations/{id}` | Fetch conversation entity | `200 OK` | — |
| `DELETE`| `/api/v1/conversations/{id}` | Delete conversation and cascade messages | `204 No Content` | — |
| `POST` | `/api/v1/conversations/{id}/messages` | Synchronous context-grounded chat | `200 OK` | — |
| `POST` | `/api/v1/conversations/{id}/messages/stream`| Real-time SSE token streaming | `200 OK` | — |
| `GET` | `/api/v1/knowledge-bases/{kb_id}/conversations` | Offset pagination for conversations | `200 OK` | **Deprecated** (`use v2`) |
| `GET` | `/api/v1/conversations/{id}/messages` | Offset pagination for messages | `200 OK` | **Deprecated** (`use v2`) |
| `GET` | `/api/v2/knowledge-bases/{kb_id}/conversations` | Signed cursor conversation pagination | `200 OK` | — |
| `GET` | `/api/v2/conversations/{id}/messages` | Signed cursor message pagination | `200 OK` | — |

---

## Operational Runbook & SRE Triage

### 1. Triaging Unpublished Outbox Backlog
If documents are stuck in `PENDING`, verify whether the outbox queue is publishing:

```sql
SELECT 
    event_type,
    COUNT(*) AS total_backlog,
    MAX(attempt_count) AS max_attempts,
    MIN(created_at) AS oldest_pending_event
FROM outbox_events
WHERE published_at IS NULL
GROUP BY event_type;
```

**Remediation**:
1. Check Celery Beat logs: Ensure `celery_app beat` is active and scheduling `outbox.publish`.
2. Verify the `maintenance` queue worker:
   ```bash
   uv run celery -A app.worker.celery_app:celery_app inspect active --queues=maintenance
   ```

---

### 2. Identifying Zombie `PROCESSING` Documents
If workers crash mid-execution or hit OOM, documents may remain in `PROCESSING`:

```sql
SELECT id, filename, knowledge_base_id, retry_count, processing_started_at
FROM documents
WHERE status = 'processing'
  AND processing_started_at < NOW() - INTERVAL '15 minutes';
```

---

### 3. Reviewing Failed Ingestion Root Causes
To inspect permanent failures or exhausted retry states:

```sql
SELECT 
    id,
    filename,
    retry_count,
    last_error,
    updated_at
FROM documents
WHERE status = 'failed'
ORDER BY updated_at DESC
LIMIT 20;
```

---

## Local Development & Engineering Workflow

### Prerequisites
* **Python 3.14+**
* **Docker & Docker Compose** (for PostgreSQL + pgvector & Redis)
* **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
* **Google Gemini API Key**

---

### 1. Environment & Infrastructure Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/basic-rag.git
cd basic-rag

# 2. Initialize environment configuration
make env

# 3. Synchronize virtual environment dependencies with uv
make sync

# 4. Spin up PostgreSQL (pgvector) and Redis
make docker-up
```

Update `.env` with your Google Gemini credentials:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key
CURSOR_SIGNING_KEY=your-32-char-random-secret-key
```

---

### 2. Database Migrations

Apply all Alembic database migrations:

```bash
make migrate
```

---

### 3. Running Services

Open separate terminal windows or use a process manager:

```bash
# Terminal 1: FastAPI Gateway Server
make dev

# Terminal 2: Celery Worker (Documents & Maintenance Queues)
uv run celery -A app.worker.celery_app:celery_app worker \
  --queues=documents,maintenance \
  --concurrency=2 \
  --loglevel=INFO

# Terminal 3: Celery Beat Scheduler (Outbox Publisher)
uv run celery -A app.worker.celery_app:celery_app beat \
  --loglevel=INFO
```

The interactive API documentation is available at `http://127.0.0.1:8000/docs`.

---

### 4. Code Quality, Typing & Test Suite

The project enforces strict typing and linting standards:

```bash
# Format codebase with Ruff
make format

# Run full linting, format checks, and static typechecks (BasedPyright)
make check

# Execute complete Pytest test suite with coverage report
make test-cov

# Run full pre-commit verification pipeline
make verify
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.