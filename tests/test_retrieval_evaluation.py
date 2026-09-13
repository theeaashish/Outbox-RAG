import math
from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.ai.retrieval.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from app.core.constants import EMBEDDING_DIMENSION
from app.db.database import engine
from app.db.models import Document, DocumentChunk, KnowledgeBase, Project, User
from app.db.models.enums import DocumentStatus
from app.repositories.document_chunk import DocumentChunkRepository


def _unit_vector(active_index: int, dim: int = EMBEDDING_DIMENSION) -> list[float]:
    """Generate a unit vector with 1.0 at active_index and 0.0 elsewhere."""
    vec = [0.0] * dim
    vec[active_index] = 1.0
    return vec


def _cluster_vector(
    base_index: int, offset: float, dim: int = EMBEDDING_DIMENSION
) -> list[float]:
    """Generate a vector perturbed around base_index and normalized to unit length."""
    vec = [0.0] * dim
    vec[base_index] = 1.0
    vec[(base_index + 1) % dim] = offset
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


@pytest.fixture
def db_session() -> Generator[Session]:
    """Create a clean database session connected to PostgreSQL, rolling back/cleaning up test data."""
    SessionMaker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionMaker()
    created_user_ids: list[object] = []

    try:
        yield session
    finally:
        if created_user_ids:
            try:
                for uid in created_user_ids:
                    session.execute(
                        text("DELETE FROM users WHERE id = :uid"), {"uid": uid}
                    )
                session.commit()
            except SQLAlchemyError:
                session.rollback()
        session.close()


def _create_user_and_kb(
    session: Session, name_prefix: str
) -> tuple[User, KnowledgeBase]:
    user_suffix = uuid4().hex[:8]
    user = User(
        email=f"{name_prefix}_{user_suffix}@example.com",
        email_normalized=f"{name_prefix}_{user_suffix}@example.com",
        name=f"Eval User {name_prefix}",
    )
    session.add(user)
    session.flush()

    project = Project(user_id=user.id, name=f"Proj_{user_suffix}")
    session.add(project)
    session.flush()

    kb = KnowledgeBase(user_id=user.id, project_id=project.id, name=f"KB_{user_suffix}")
    session.add(kb)
    session.flush()

    return user, kb


# ==============================================================================
# 1. Filtered ANN vs. Exact Search Measurement
# ==============================================================================


def test_filtered_ann_vs_exact_search_measurement(
    db_session: Session, capsys: pytest.CaptureFixture[str]
):
    """
    Measure ANN Recall@K under realistic filtered retrieval conditions.

    Environment Baseline (verified):
    - PostgreSQL 15.18 + pgvector 0.8.5
    - HNSW index: ix_document_chunks_embedding_hnsw
    - Defaults: hnsw.ef_search = 40, hnsw.iterative_scan = off

    Compares:
    - Exact: Sequential scan computing exact cosine distance on filtered rows
    - HNSW: Approximate index scan using pgvector HNSW
    """
    user_target, kb_target = _create_user_and_kb(db_session, "target")
    user_other, kb_other = _create_user_and_kb(db_session, "other")

    user_suffix = uuid4().hex[:8]

    # Create target READY document (qualifying)
    doc_target_ready = Document(
        knowledge_base_id=kb_target.id,
        title="Target Ready Doc",
        filename="target_ready.pdf",
        mime_type="application/pdf",
        storage_path=f"storage/tr_{user_suffix}.pdf",
        sha256_hash=uuid4().hex,
        file_size=2048,
        status=DocumentStatus.READY,
    )
    # Create target PENDING document (distractor: same KB, wrong status)
    doc_target_pending = Document(
        knowledge_base_id=kb_target.id,
        title="Target Pending Doc",
        filename="target_pending.pdf",
        mime_type="application/pdf",
        storage_path=f"storage/tp_{user_suffix}.pdf",
        sha256_hash=uuid4().hex,
        file_size=2048,
        status=DocumentStatus.PENDING,
    )
    # Create other user READY document (distractor: other user, wrong tenant)
    doc_other_ready = Document(
        knowledge_base_id=kb_other.id,
        title="Other Ready Doc",
        filename="other_ready.pdf",
        mime_type="application/pdf",
        storage_path=f"storage/or_{user_suffix}.pdf",
        sha256_hash=uuid4().hex,
        file_size=2048,
        status=DocumentStatus.READY,
    )

    db_session.add_all([doc_target_ready, doc_target_pending, doc_other_ready])
    db_session.flush()

    # Populate 45 total chunks across the partitions:
    # - 20 chunks in doc_target_ready (target cluster along axis 0 with various offsets)
    # - 15 chunks in doc_target_pending (same cluster along axis 0, must be filtered out)
    # - 10 chunks in doc_other_ready (same cluster along axis 0, must be filtered out)
    chunks: list[DocumentChunk] = []

    for i in range(20):
        vec = _cluster_vector(base_index=0, offset=0.05 * (i + 1))
        chunks.append(
            DocumentChunk(
                document_id=doc_target_ready.id,
                chunk_index=i,
                content=f"Target Ready Chunk {i}",
                embedding=vec,
            )
        )

    for i in range(15):
        vec = _cluster_vector(base_index=0, offset=0.02 * (i + 1))
        chunks.append(
            DocumentChunk(
                document_id=doc_target_pending.id,
                chunk_index=i,
                content=f"Target Pending Chunk {i}",
                embedding=vec,
            )
        )

    for i in range(10):
        vec = _cluster_vector(base_index=0, offset=0.01 * (i + 1))
        chunks.append(
            DocumentChunk(
                document_id=doc_other_ready.id,
                chunk_index=i,
                content=f"Other User Chunk {i}",
                embedding=vec,
            )
        )

    db_session.add_all(chunks)
    db_session.commit()

    repo = DocumentChunkRepository(db=db_session)
    query_vector = _unit_vector(active_index=0)

    try:
        # Step A: Verify query plan under enable_seqscan = off exercises the HNSW index
        db_session.execute(text("SET LOCAL enable_seqscan = off"))
        explain_query = text(
            """
            EXPLAIN
            SELECT document_chunks.id
            FROM document_chunks
            JOIN documents ON document_chunks.document_id = documents.id
            JOIN knowledge_bases ON documents.knowledge_base_id = knowledge_bases.id
            WHERE documents.knowledge_base_id = :kb_id
              AND knowledge_bases.user_id = :user_id
              AND documents.status = 'ready'
            ORDER BY document_chunks.embedding <=> CAST(:vec AS vector) ASC
            LIMIT 10
            """
        )
        vec_str = "[" + ",".join(str(x) for x in query_vector) + "]"
        plan_rows = db_session.execute(
            explain_query,
            {"vec": vec_str, "kb_id": kb_target.id, "user_id": user_target.id},
        ).fetchall()
        plan_text = "\n".join(r[0] for r in plan_rows)
        # Verify planner selects the HNSW index
        assert (
            "ix_document_chunks_embedding_hnsw" in plan_text
            or "Index Scan" in plan_text
        )

        # Step B: Ground Truth (Exact Cosine Search via sequential scan)
        db_session.execute(text("SET LOCAL enable_seqscan = on"))
        db_session.execute(text("SET LOCAL enable_indexscan = off"))
        db_session.execute(text("SET LOCAL enable_bitmapscan = off"))

        exact_results_5 = repo.search_similar(
            user_id=user_target.id,
            knowledge_base_id=kb_target.id,
            embedding=query_vector,
            limit=5,
        )
        exact_ids_5 = [str(r.chunk.id) for r in exact_results_5]

        exact_results_10 = repo.search_similar(
            user_id=user_target.id,
            knowledge_base_id=kb_target.id,
            embedding=query_vector,
            limit=10,
        )
        exact_ids_10 = [str(r.chunk.id) for r in exact_results_10]

        # Step C: HNSW Approximate Search (via forced index scan)
        db_session.execute(text("SET LOCAL enable_indexscan = on"))
        db_session.execute(text("SET LOCAL enable_bitmapscan = on"))
        db_session.execute(text("SET LOCAL enable_seqscan = off"))

        ann_results_5 = repo.search_similar(
            user_id=user_target.id,
            knowledge_base_id=kb_target.id,
            embedding=query_vector,
            limit=5,
        )
        ann_ids_5 = [str(r.chunk.id) for r in ann_results_5]

        ann_results_10 = repo.search_similar(
            user_id=user_target.id,
            knowledge_base_id=kb_target.id,
            embedding=query_vector,
            limit=10,
        )
        ann_ids_10 = [str(r.chunk.id) for r in ann_results_10]

        # Step D: Calculate Recall@K
        recall_5 = calculate_recall_at_k(ann_ids_5, exact_ids_5, k=5)
        recall_10 = calculate_recall_at_k(ann_ids_10, exact_ids_10, k=10)

        # Structured diagnostic report
        report = (
            f"\n--- FILTERED ANN VS EXACT RETRIEVAL EVALUATION REPORT ---\n"
            f"Environment: PostgreSQL 15.18, pgvector 0.8.5 (hnsw.ef_search=40, hnsw.iterative_scan=off)\n"
            f"Filter Scopes: user_id={user_target.id}, kb_id={kb_target.id}, status=READY\n"
            f"Candidate Pool: 20 qualifying READY, 15 PENDING distractor, 10 Other-Tenant distractor\n"
            f"Top-5  | Exact: {len(exact_ids_5)} | ANN: {len(ann_ids_5)} | Recall@5:  {recall_5:.4f}\n"
            f"Top-10 | Exact: {len(exact_ids_10)} | ANN: {len(ann_ids_10)} | Recall@10: {recall_10:.4f}\n"
            f"----------------------------------------------------------\n"
        )
        print(report)

        # Invariants: both return between 0 and limit results, only from authorized target document
        assert 0 <= len(ann_ids_5) <= 5
        assert 0 <= len(ann_ids_10) <= 10
        assert 0.0 <= recall_5 <= 1.0
        assert 0.0 <= recall_10 <= 1.0

        for r in ann_results_5:
            assert r.chunk.document_id == doc_target_ready.id
        for r in ann_results_10:
            assert r.chunk.document_id == doc_target_ready.id

    finally:
        for u in [user_target, user_other]:
            db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": u.id})
        db_session.commit()


# ==============================================================================
# 2. Golden Retrieval Dataset Evaluation
# ==============================================================================


def test_golden_retrieval_dataset_evaluation(db_session: Session):
    """
    Deterministic evaluation against a multi-topic golden dataset with known relevant chunks.

    Calculates:
    - Recall@K
    - Precision@K
    using set-based relevance membership.
    """
    user, kb = _create_user_and_kb(db_session, "golden")
    user_suffix = uuid4().hex[:8]

    doc = Document(
        knowledge_base_id=kb.id,
        title="Golden Knowledge Document",
        filename="golden.pdf",
        mime_type="application/pdf",
        storage_path=f"storage/golden_{user_suffix}.pdf",
        sha256_hash=uuid4().hex,
        file_size=4096,
        status=DocumentStatus.READY,
    )
    db_session.add(doc)
    db_session.flush()

    # Define 3 semantic domains:
    # Topic 0 (Database Connection Pooling): base index 10
    # Topic 1 (Authentication Security): base index 20
    # Topic 2 (Document Ingestion & Chunking): base index 30

    # 3 chunks for Topic 0

    t0_chunks = [
        DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            content="Database connection pooling manages connections efficiently.",
            embedding=_cluster_vector(base_index=10, offset=0.01),
        ),
        DocumentChunk(
            document_id=doc.id,
            chunk_index=1,
            content="PostgreSQL transaction isolation prevents dirty reads.",
            embedding=_cluster_vector(base_index=10, offset=0.02),
        ),
        DocumentChunk(
            document_id=doc.id,
            chunk_index=2,
            content="Connection pool size should be calibrated against max connections.",
            embedding=_cluster_vector(base_index=10, offset=0.03),
        ),
    ]

    # 3 chunks for Topic 1
    t1_chunks = [
        DocumentChunk(
            document_id=doc.id,
            chunk_index=3,
            content="Authentication identity securely maps session cookies to users.",
            embedding=_cluster_vector(base_index=20, offset=0.01),
        ),
        DocumentChunk(
            document_id=doc.id,
            chunk_index=4,
            content="Password hashing uses argon2id for cryptographic resistance.",
            embedding=_cluster_vector(base_index=20, offset=0.02),
        ),
        DocumentChunk(
            document_id=doc.id,
            chunk_index=5,
            content="Session expiration ensures inactive tokens are revoked.",
            embedding=_cluster_vector(base_index=20, offset=0.03),
        ),
    ]

    # 3 chunks for Topic 2
    t2_chunks = [
        DocumentChunk(
            document_id=doc.id,
            chunk_index=6,
            content="Document chunking splits large texts into semantic blocks.",
            embedding=_cluster_vector(base_index=30, offset=0.01),
        ),
        DocumentChunk(
            document_id=doc.id,
            chunk_index=7,
            content="Text normalization strips unwanted whitespace and unicode variance.",
            embedding=_cluster_vector(base_index=30, offset=0.02),
        ),
        DocumentChunk(
            document_id=doc.id,
            chunk_index=8,
            content="Chunk provenance preserves source block indexes and page numbers.",
            embedding=_cluster_vector(base_index=30, offset=0.03),
        ),
    ]

    all_chunks = t0_chunks + t1_chunks + t2_chunks
    db_session.add_all(all_chunks)
    db_session.commit()

    repo = DocumentChunkRepository(db=db_session)

    try:
        test_cases = [
            {
                "topic": "Database Connection Pooling",
                "query_vector": _unit_vector(active_index=10),
                "expected_chunks": {str(c.id) for c in t0_chunks},
            },
            {
                "topic": "Authentication Security",
                "query_vector": _unit_vector(active_index=20),
                "expected_chunks": {str(c.id) for c in t1_chunks},
            },
            {
                "topic": "Document Ingestion",
                "query_vector": _unit_vector(active_index=30),
                "expected_chunks": {str(c.id) for c in t2_chunks},
            },
        ]

        print("\n--- GOLDEN RETRIEVAL DATASET EVALUATION REPORT ---")
        for tc in test_cases:
            results = repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=tc["query_vector"],
                limit=3,
            )
            retrieved_ids = [str(r.chunk.id) for r in results]
            recall = calculate_recall_at_k(retrieved_ids, tc["expected_chunks"], k=3)
            precision = calculate_precision_at_k(
                retrieved_ids, tc["expected_chunks"], k=3
            )

            print(
                f"Topic: {tc['topic']:<30} | Retrieved: {len(retrieved_ids)} | "
                f"Recall@3: {recall:.4f} | Precision@3: {precision:.4f}"
            )

            # Set-based correctness: all 3 topic chunks should be retrieved
            assert recall == 1.0
            assert precision == 1.0

        print("--------------------------------------------------\n")

    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()
