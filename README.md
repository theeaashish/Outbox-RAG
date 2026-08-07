# Basic RAG

A Retrieval-Augmented Generation (RAG) service built with FastAPI, PostgreSQL, pgvector, LangChain, and Google Gemini.

---

## Overview

Basic RAG is a REST API service that enables document-grounded question answering over custom knowledge bases. It ingests PDF documents, splits text into chunks, generates vector embeddings using Google Gemini, indexes the vectors in PostgreSQL via `pgvector` with HNSW indexes, and retrieves relevant context chunks to answer user queries with synchronous or streaming Server-Sent Events (SSE) responses.

---

## Features

- **Document Processing & Ingestion**: Uploads PDF files, validates size and format, extracts text with `pypdf`, hashes file contents using SHA-256 to prevent duplicate uploads per knowledge base, and stores raw files in local storage.
- **Recursive Chunking & Embeddings**: Splits extracted text using LangChain `RecursiveCharacterTextSplitter` and generates 768-dimensional embeddings using Google Gemini models (`text-embedding-004` or `gemini-embedding-001`).
- **Vector Retrieval with HNSW**: Performs cosine distance similarity searches against PostgreSQL `pgvector` columns equipped with HNSW indexes (`ix_document_chunks_embedding_hnsw`), filtering by similarity threshold and document readiness status.
- **RAG Chat & SSE Streaming**: Provides synchronous chat endpoints and real-time streaming endpoints via `sse-starlette` emitting `metadata`, `citations`, `token`, and `complete` events.
- **Signed Cursor Pagination (v2 API)**: Offers HMAC-SHA256 signed keyset pagination (`CursorCodec`) for listing conversations and messages with key rotation support.
- **Automated Tooling**: Features database migrations via Alembic, linting and formatting via Ruff, static type checking via BasedPyright, and test suites via Pytest.

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Framework & Server** | FastAPI 0.140+, Uvicorn 0.51+, Pydantic Settings 2.14+, sse-starlette 3.4+ |
| **Database & ORM** | PostgreSQL 15+, pgvector 0.5+, SQLAlchemy 2.0+, Alembic 1.18+ |
| **AI & RAG Framework** | LangChain 1.3+, `langchain-google-genai` 4.3+, Google Gemini API |
| **Document Parsing** | `pypdf` 6.14+ |
| **Package Manager** | `uv` |
| **Testing & Linting** | Pytest 9.1+, pytest-cov 7.1+, Ruff 0.16+, BasedPyright 1.39+ |

---

## Project Structure

```
.
├── alembic/              # Alembic database migration scripts and environment
│   └── versions/         # Migration revision files
├── app/                  # Application source code
│   ├── api/              # API router definitions (v1 and v2)
│   ├── core/             # Core functionality (AI, document parsers, storage, config, pagination)
│   │   ├── ai/           # Chunking, context assembly, embeddings, LLM providers, prompts
│   │   ├── document/     # PDF parsers, validators, file hasher
│   │   ├── storage/      # Local file storage provider
│   │   ├── config.py     # Application configuration settings
│   │   ├── exceptions.py # Custom exception classes and HTTP handlers
│   │   └── pagination.py # Signed cursor codec implementation
│   ├── db/               # SQLAlchemy Base, session setup, and ORM models
│   │   └── models/       # Models: KnowledgeBase, Document, DocumentChunk, Conversation, Message
│   ├── dependencies/     # FastAPI Dependency Injection providers
│   ├── modules/          # Domain services (chat, conversations, document, retrieval)
│   ├── repositories/     # Repository implementations for database models
│   └── main.py           # FastAPI application entry point
├── docs/                 # Documentation files
├── tests/                # Test suite
├── uploads/              # Storage path for uploaded documents
├── .env.example          # Template environment variable file
├── Makefile              # Makefile automation commands
├── pyproject.toml        # Project dependencies and configuration settings
└── uv.lock               # uv lockfile
```

---

## Getting Started

### Prerequisites

- **Python**: `>= 3.14`
- **uv**: Python package installer and virtual environment manager
- **Docker**: Optional, for running PostgreSQL with `pgvector` locally

---

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd basic-rag
   ```

2. **Create local environment file**:
   ```bash
   make env
   ```

3. **Install dependencies**:
   ```bash
   make sync
   ```

4. **Start PostgreSQL database**:
   ```bash
   make docker-up
   ```

5. **Run database migrations**:
   ```bash
   make migrate
   ```

6. **Set Google API Key**:
   Configure your Gemini API key in `.env`:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

7. **Start the development server**:
   ```bash
   make dev
   ```
   The application runs at `http://127.0.0.1:8000`.

---

## Environment Variables

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `DATABASE_URL` | Yes | `postgresql+psycopg://postgres:postgres@localhost:5432/basic_rag` | PostgreSQL connection string |
| `GOOGLE_API_KEY` | Yes | `""` | Google Gemini API key |
| `GEMINI_CHAT_MODEL` | No | `gemini-2.5-flash` | Gemini LLM model identifier |
| `GEMINI_EMBEDDING_MODEL` | No | `text-embedding-004` | Gemini embedding model identifier |
| `APP_NAME` | No | `Basic RAG` | Application display name |
| `APP_ENV` | No | `development` | Environment status (`development`, `testing`, `staging`, `production`) |
| `APP_DEBUG` | No | `true` | Debug flag |
| `APP_HOST` | No | `127.0.0.1` | Application host interface |
| `APP_PORT` | No | `8000` | Application port |
| `UPLOAD_DIRECTORY` | No | `uploads` | Directory for uploaded documents |
| `MAX_UPLOAD_SIZE_MB` | No | `20` | Maximum file upload size in megabytes |
| `DEFAULT_TOP_K` | No | `5` | Default number of vector chunks to retrieve |
| `SIMILARITY_THRESHOLD` | No | `0.7` | Minimum similarity score threshold for retrieval |
| `LOG_LEVEL` | No | `INFO` | Application log level |
| `API_V1_PREFIX` | No | `/api/v1` | Prefix for v1 routes |
| `API_V2_PREFIX` | No | `/api/v2` | Prefix for v2 routes |
| `CHAT_HISTORY_MESSAGE_LIMIT` | No | `20` | Maximum historical messages included in chat context |
| `CHAT_MAX_MESSAGE_CHARACTERS` | No | `10000` | Maximum allowed user message characters |
| `CHAT_STREAM_FIRST_TOKEN_TIMEOUT_SECONDS` | No | `20` | Timeout waiting for first token in stream |
| `CHAT_STREAM_IDLE_TIMEOUT_SECONDS` | No | `30` | Idle timeout during stream generation |
| `CHAT_STREAM_TOTAL_TIMEOUT_SECONDS` | No | `120` | Total stream execution timeout |
| `CHAT_STREAM_MAX_BUFFERED_CHARACTERS` | No | `65536` | Maximum buffered stream output limit |
| `CHAT_STREAM_PING_INTERVAL_SECONDS` | No | `15` | SSE stream ping interval |
| `GEMINI_CHAT_MAX_OUTPUT_TOKENS` | No | `4096` | Gemini max output tokens setting |
| `CURSOR_SIGNING_KEY` | Yes | `development-only-change-me` | Secret key used to sign pagination cursors |
| `CURSOR_PREVIOUS_SIGNING_KEY` | No | `None` | Key used to support cursor key rotation |

---

## Available Scripts

Command targets available via `Makefile`:

| Command | Action |
| :--- | :--- |
| `make env` | Create `.env` from `.env.example` if it does not exist |
| `make sync` / `make install` | Synchronize dependencies using `uv sync` |
| `make dev` / `make run` | Start FastAPI development server with auto-reload |
| `make shell` | Start interactive Python REPL in project environment |
| `make format` | Format code using Ruff (`uv run ruff format .`) |
| `make format-check` | Check code formatting using Ruff |
| `make lint` | Run Ruff linter (`uv run ruff check .`) |
| `make lint-fix` | Fix auto-fixable Ruff linter issues |
| `make typecheck` | Run static type checks using BasedPyright |
| `make check` | Run `format-check`, `lint`, and `typecheck` |
| `make test` | Run pytest suite |
| `make test-unit` | Run unit tests under `tests/` |
| `make test-cov` | Run pytest with coverage report |
| `make verify` / `make ci` | Run full check and test suite |
| `make precommit` | Run pre-commit hooks on repository files |
| `make migrate` | Apply database migrations (`alembic upgrade head`) |
| `make makemigrations` | Create migration file (requires `m="description"`) |
| `make downgrade` | Rollback last database migration (`alembic downgrade -1`) |
| `make current` | Display current active database migration revision |
| `make history` | Display migration history |
| `make docker-up` | Start local PostgreSQL container with pgvector (`pgvector/pgvector:pg15`) |
| `make docker-down` | Stop local database container |
| `make docker-logs` | Stream database container logs |
| `make clean` | Clean bytecode and test cache artifacts |

---

## Usage

### Ingest a PDF Document

Upload a PDF document to a Knowledge Base UUID:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/knowledge-bases/123e4567-e89b-12d3-a456-426614174000/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf;type=application/pdf"
```

### Search Knowledge Base

Perform semantic search over processed document chunks:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/knowledge-bases/123e4567-e89b-12d3-a456-426614174000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the summary of the document?",
    "limit": 5,
    "threshold": 0.7
  }'
```

### Create a Conversation

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/knowledge-bases/123e4567-e89b-12d3-a456-426614174000/conversations"
```

### Send a Synchronous Chat Message

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/conversations/98765432-e89b-12d3-a456-426614174000/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What are the main topics in the document?"
  }'
```

### Stream Chat Response via SSE

```bash
curl -N -X POST "http://127.0.0.1:8000/api/v1/conversations/98765432-e89b-12d3-a456-426614174000/messages/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Explain the conclusion in detail."
  }'
```

---

## API Overview

### System Endpoints
- `GET /` - Root endpoint welcome message
- `GET /health` - Health check status

### API v1 (`/api/v1`)
- **Documents & Retrieval**
  - `POST /knowledge-bases/{knowledge_base_id}/documents` - Upload and process a PDF document
  - `POST /knowledge-bases/{knowledge_base_id}/search` - Execute vector similarity search
- **Conversations & Chat**
  - `POST /knowledge-bases/{knowledge_base_id}/conversations` - Create a conversation
  - `GET /knowledge-bases/{knowledge_base_id}/conversations` - List conversations (deprecated)
  - `GET /conversations/{conversation_id}` - Retrieve a conversation
  - `GET /conversations/{conversation_id}/messages` - List messages in a conversation (deprecated)
  - `DELETE /conversations/{conversation_id}` - Delete a conversation
  - `POST /conversations/{conversation_id}/messages` - Send a synchronous RAG chat message
  - `POST /conversations/{conversation_id}/messages/stream` - Stream RAG response via Server-Sent Events

### API v2 (`/api/v2`)
- **Signed Cursor Pagination**
  - `GET /knowledge-bases/{knowledge_base_id}/conversations` - List conversations with signed cursor pagination (`after`, `before`, `page_size`)
  - `GET /conversations/{conversation_id}/messages` - List messages with signed cursor pagination (`after`, `before`, `page_size`)

---

## Database

The application uses PostgreSQL with the `pgvector` extension.

### Entities
- `KnowledgeBase`: Container for documents and conversations (`name`, `description`).
- `Document`: Uploaded document record (`title`, `filename`, `mime_type`, `storage_path`, `sha256_hash`, `file_size`, `status`). Unique per knowledge base by hash (`uq_document_kb_sha256`).
- `DocumentChunk`: Text chunk from document (`content`, `chunk_index`, `char_start`, `char_end`, `metadata`, `embedding`). Contains HNSW cosine index `ix_document_chunks_embedding_hnsw` on `Vector(768)`.
- `Conversation`: Thread associated with a knowledge base (`knowledge_base_id`).
- `Message`: Chat message record (`role`, `content`, `conversation_id`, `created_at`). Indexed on (`conversation_id`, `created_at`).

---

## AI Features

- **LLM Integration**: Uses Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai` to generate answers.
- **Embeddings**: Uses Google Gemini (`text-embedding-004` or `gemini-embedding-001`) generating 768-dimensional embeddings.
- **Text Chunking**: Uses LangChain `RecursiveCharacterTextSplitter` with default chunk size of 1000 and overlap of 200.
- **Context Assembly**: Retrieves relevant vector chunks using cosine similarity search and formats them into RAG system prompts.
- **SSE Streaming**: Streams response deltas and delivers `metadata`, `citations`, `token`, and `complete` events.

---

## Deployment

The database infrastructure can be deployed using the PostgreSQL `pgvector/pgvector:pg15` container image.

To run migrations:
```bash
uv run alembic upgrade head
```

To run the application server:
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Contributing

1. Ensure code passes formatting, linter, and type checks:
   ```bash
   make check
   ```
2. Run unit and integration tests:
   ```bash
   make test
   ```
3. Run the full verification suite before submitting PRs:
   ```bash
   make verify
   ```
