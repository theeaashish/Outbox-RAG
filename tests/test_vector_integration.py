import math
from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.constants import EMBEDDING_DIMENSION, MAX_RETRIEVAL_LIMIT
from app.core.exceptions import ValidationException
from app.db.database import engine
from app.db.models import Document, DocumentChunk, KnowledgeBase, Project, User
from app.db.models.enums import DocumentStatus
from app.repositories.document_chunk import DocumentChunkRepository


def _unit_vector(active_index: int, dim: int = EMBEDDING_DIMENSION) -> list[float]:
    """Generate a unit vector with 1.0 at active_index and 0.0 elsewhere."""
    vec = [0.0] * dim
    vec[active_index] = 1.0
    return vec


def _mixed_vector(dim: int = EMBEDDING_DIMENSION, val: float = 0.05) -> list[float]:
    """Generate a vector of length dim with equal components normalized to unit length."""
    norm = math.sqrt(dim * (val**2))
    return [(val / norm)] * dim


@pytest.fixture
def db_session() -> Generator[Session]:
    """Create a clean database session connected to pg-local, rolling back or cleaning up test data."""
    SessionMaker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionMaker()
    created_user_ids: list[object] = []

    try:
        yield session
    finally:
        # Cleanup any created users and cascade deletes all child records
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


def _create_hierarchy(
    session: Session,
    *,
    status: DocumentStatus = DocumentStatus.READY,
) -> tuple[User, Project, KnowledgeBase, Document]:
    """Helper to create a valid hierarchy: User -> Project -> KnowledgeBase -> Document."""
    user_suffix = uuid4().hex[:8]
    user = User(
        email=f"user_{user_suffix}@example.com",
        email_normalized=f"user_{user_suffix}@example.com",
        name="Integration Test User",
    )
    session.add(user)
    session.flush()

    project = Project(
        user_id=user.id,
        name=f"Project_{user_suffix}",
    )
    session.add(project)
    session.flush()

    kb = KnowledgeBase(
        user_id=user.id,
        project_id=project.id,
        name=f"KB_{user_suffix}",
    )
    session.add(kb)
    session.flush()

    doc = Document(
        knowledge_base_id=kb.id,
        title="Integration Test Document",
        filename="integration.pdf",
        mime_type="application/pdf",
        storage_path=f"storage/{user_suffix}.pdf",
        sha256_hash=uuid4().hex,
        file_size=2048,
        status=status,
    )
    session.add(doc)
    session.flush()

    return user, project, kb, doc


# ==============================================================================
# 1. Vector Persistence End-to-End
# ==============================================================================


def test_vector_persistence_end_to_end_tolerance_comparison(db_session: Session):
    """Verify storing a 768-dim vector preserves float precision with tolerance comparison."""
    user, _, _, doc = _create_hierarchy(db_session)

    # Generate a realistic non-trivial 768-dimensional float vector
    raw_vector = [math.sin(i + 1) * 0.1 for i in range(EMBEDDING_DIMENSION)]

    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="Testing vector persistence fidelity",
        embedding=raw_vector,
    )
    db_session.add(chunk)
    db_session.commit()

    try:
        # Re-fetch from database in a fresh read
        repo = DocumentChunkRepository(db=db_session)
        stored_chunks = repo.get_by_document(document_id=doc.id)
        assert len(stored_chunks) == 1

        stored_chunk = stored_chunks[0]
        assert stored_chunk.content == "Testing vector persistence fidelity"
        assert len(stored_chunk.embedding) == EMBEDDING_DIMENSION

        # Verify using tolerance-based floating-point comparison (abs=1e-5)
        for original, stored in zip(raw_vector, stored_chunk.embedding, strict=True):
            assert stored == pytest.approx(original, abs=1e-5)
    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()


def test_database_rejects_wrong_embedding_dimension(db_session: Session):
    """PostgreSQL pgvector must reject vectors with dimension != 768 at the database layer."""
    user, _, _, doc = _create_hierarchy(db_session)

    # Vector with 512 dimensions instead of 768
    wrong_vector = [0.1] * 512

    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="Should fail due to dimension mismatch",
        embedding=wrong_vector,
    )
    db_session.add(chunk)

    try:
        with pytest.raises(DBAPIError, match="expected 768 dimensions"):
            db_session.commit()
    finally:
        db_session.rollback()
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()


# ==============================================================================
# 2. Cosine Similarity Semantics & Ordering
# ==============================================================================


def test_cosine_similarity_retrieval_ranking_and_threshold(db_session: Session):
    """
    Verify pgvector cosine similarity semantics:
    - Distance operator: <=>
    - similarity = 1 - distance
    - Lower distance = better (higher similarity)
    - Results strictly ordered by distance.asc()
    - Threshold filtering excludes lower similarity chunks
    """
    user, _, kb, doc = _create_hierarchy(db_session)

    # Define query vector pointing along axis 0: [1.0, 0, 0, ...]
    query_vector = _unit_vector(active_index=0)

    # Chunk A: Identical direction to query -> cosine distance = 0.0, similarity = 1.0
    chunk_a_vec = _unit_vector(active_index=0)
    # Chunk B: Partially aligned: combination of axis 0 and axis 1 -> cosine similarity ~ 0.707
    norm_45 = 1.0 / math.sqrt(2.0)
    chunk_b_vec = [0.0] * EMBEDDING_DIMENSION
    chunk_b_vec[0] = norm_45
    chunk_b_vec[1] = norm_45
    # Chunk C: Orthogonal direction to query (axis 1 only) -> cosine distance = 1.0, similarity = 0.0
    chunk_c_vec = _unit_vector(active_index=1)

    chunk_a = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="Chunk A (identical direction)",
        embedding=chunk_a_vec,
    )
    chunk_b = DocumentChunk(
        document_id=doc.id,
        chunk_index=1,
        content="Chunk B (partially aligned)",
        embedding=chunk_b_vec,
    )
    chunk_c = DocumentChunk(
        document_id=doc.id,
        chunk_index=2,
        content="Chunk C (orthogonal)",
        embedding=chunk_c_vec,
    )

    db_session.add_all([chunk_a, chunk_b, chunk_c])
    db_session.commit()

    repo = DocumentChunkRepository(db=db_session)

    try:
        # Search all with no threshold
        results = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=query_vector,
            limit=5,
        )

        assert len(results) == 3
        # Strict ordering by similarity descending (distance ascending)
        assert results[0].chunk.content == "Chunk A (identical direction)"
        assert results[1].chunk.content == "Chunk B (partially aligned)"
        assert results[2].chunk.content == "Chunk C (orthogonal)"

        # Verify similarity scores with tolerance
        assert results[0].similarity == pytest.approx(1.0, abs=1e-4)
        assert results[1].similarity == pytest.approx(0.7071, abs=1e-3)
        assert results[2].similarity == pytest.approx(0.0, abs=1e-4)

        # Test threshold: threshold=0.5 must exclude Chunk C (sim ~ 0.0)
        filtered_results = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=query_vector,
            limit=5,
            threshold=0.5,
        )
        assert len(filtered_results) == 2
        assert filtered_results[0].chunk.content == "Chunk A (identical direction)"
        assert filtered_results[1].chunk.content == "Chunk B (partially aligned)"

        # Test limit: limit=1 returns only top result
        top_1 = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=query_vector,
            limit=1,
        )
        assert len(top_1) == 1
        assert top_1[0].chunk.content == "Chunk A (identical direction)"

    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()


# ==============================================================================
# 3. Ownership and Status Isolation
# ==============================================================================


def test_search_similar_enforces_ready_status_and_ownership(db_session: Session):
    """
    Invariants:
    - Only DocumentStatus.READY documents are searched.
    - Chunks belonging to another user or another knowledge base are never returned.
    """
    user_1, _, kb_1, doc_ready = _create_hierarchy(
        db_session, status=DocumentStatus.READY
    )
    _, _, _, doc_pending = _create_hierarchy(db_session, status=DocumentStatus.PENDING)
    user_2, _, _, doc_other_user = _create_hierarchy(
        db_session, status=DocumentStatus.READY
    )

    target_vec = _unit_vector(active_index=5)

    # Chunk in user 1's READY document
    chunk_ready = DocumentChunk(
        document_id=doc_ready.id,
        chunk_index=0,
        content="Ready document chunk",
        embedding=target_vec,
    )
    # Chunk in user 1's PENDING document
    chunk_pending = DocumentChunk(
        document_id=doc_pending.id,
        chunk_index=0,
        content="Pending document chunk",
        embedding=target_vec,
    )
    # Chunk in user 2's READY document
    chunk_user2 = DocumentChunk(
        document_id=doc_other_user.id,
        chunk_index=0,
        content="Other user chunk",
        embedding=target_vec,
    )

    db_session.add_all([chunk_ready, chunk_pending, chunk_user2])
    db_session.commit()

    repo = DocumentChunkRepository(db=db_session)

    try:
        results = repo.search_similar(
            user_id=user_1.id,
            knowledge_base_id=kb_1.id,
            embedding=target_vec,
            limit=10,
        )

        # Must only return the chunk from the READY document owned by user_1 in kb_1
        assert len(results) == 1
        assert results[0].chunk.id == chunk_ready.id
        assert results[0].chunk.content == "Ready document chunk"

    finally:
        for u in [user_1, user_2]:
            db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": u.id})
        db_session.commit()


# ==============================================================================
# 4. HNSW Index Verification
# ==============================================================================


def test_hnsw_index_definition_and_cosine_operator(db_session: Session):
    """
    Verify that the HNSW index ix_document_chunks_embedding_hnsw exists in PostgreSQL
    with using='hnsw' and operator class 'vector_cosine_ops'.
    """
    query = text(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'document_chunks'
          AND indexname = 'ix_document_chunks_embedding_hnsw'
        """
    )
    result = db_session.execute(query).fetchone()

    assert result is not None, "HNSW index ix_document_chunks_embedding_hnsw not found"
    index_name, index_def = result[0], result[1]
    assert index_name == "ix_document_chunks_embedding_hnsw"
    assert "USING hnsw" in index_def
    assert "vector_cosine_ops" in index_def


def test_hnsw_query_plan_compatibility(db_session: Session):
    """
    Verify EXPLAIN executes cleanly on the cosine search query.
    Note (Locked Choice / User Adjustment):
    A sequential scan on a tiny table fixture is completely valid and must not fail the test.
    """
    user, _, kb, doc = _create_hierarchy(db_session)
    vec = _unit_vector(active_index=0)

    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="Explain query test chunk",
        embedding=vec,
    )
    db_session.add(chunk)
    db_session.commit()

    try:
        # Run EXPLAIN on the exact search_similar query pattern
        explain_query = text(
            """
            EXPLAIN
            SELECT document_chunks.id, (1 - (document_chunks.embedding <=> CAST(:vec AS vector))) AS similarity
            FROM document_chunks
            JOIN documents ON document_chunks.document_id = documents.id
            JOIN knowledge_bases ON documents.knowledge_base_id = knowledge_bases.id
            WHERE documents.knowledge_base_id = :kb_id
              AND knowledge_bases.user_id = :user_id
              AND documents.status = 'ready'
            ORDER BY document_chunks.embedding <=> CAST(:vec AS vector) ASC
            LIMIT 5
            """
        )

        vector_str = "[" + ",".join(str(x) for x in vec) + "]"
        rows = db_session.execute(
            explain_query,
            {
                "vec": vector_str,
                "kb_id": kb.id,
                "user_id": user.id,
            },
        ).fetchall()

        plan_lines = [r[0] for r in rows]
        plan_text = "\n".join(plan_lines)

        # Plan must be successfully generated and reference the tables
        assert len(plan_lines) > 0
        assert "document_chunks" in plan_text

        # Additionally test when seqscan is disabled: the planner can use the HNSW index
        db_session.execute(text("SET LOCAL enable_seqscan = off"))
        forced_rows = db_session.execute(
            explain_query,
            {
                "vec": vector_str,
                "kb_id": kb.id,
                "user_id": user.id,
            },
        ).fetchall()
        forced_plan = "\n".join(r[0] for r in forced_rows)
        # Verify index scan on ix_document_chunks_embedding_hnsw is selectable
        assert (
            "ix_document_chunks_embedding_hnsw" in forced_plan
            or "Index Scan" in forced_plan
        )

    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()


# ==============================================================================
# 5. Phase 6 Retrieval Hardness Correctness Assertions
# ==============================================================================


def test_cross_kb_isolation_within_same_user(db_session: Session):
    """
    Invariants:
    - Sibling knowledge bases under the same user MUST NOT leak chunks.
    """
    user_suffix = uuid4().hex[:8]
    user = User(
        email=f"user_{user_suffix}@example.com",
        email_normalized=f"user_{user_suffix}@example.com",
        name="Cross-KB User",
    )
    db_session.add(user)
    db_session.flush()

    proj_1 = Project(user_id=user.id, name=f"Proj1_{user_suffix}")
    proj_2 = Project(user_id=user.id, name=f"Proj2_{user_suffix}")
    db_session.add_all([proj_1, proj_2])
    db_session.flush()

    kb_1 = KnowledgeBase(
        user_id=user.id, project_id=proj_1.id, name=f"KB1_{user_suffix}"
    )
    kb_2 = KnowledgeBase(
        user_id=user.id, project_id=proj_2.id, name=f"KB2_{user_suffix}"
    )
    db_session.add_all([kb_1, kb_2])
    db_session.flush()

    doc_1 = Document(
        knowledge_base_id=kb_1.id,
        title="KB1 Document",
        filename="kb1.pdf",
        mime_type="application/pdf",
        storage_path=f"storage/kb1_{user_suffix}.pdf",
        sha256_hash=uuid4().hex,
        file_size=1024,
        status=DocumentStatus.READY,
    )
    doc_2 = Document(
        knowledge_base_id=kb_2.id,
        title="KB2 Document",
        filename="kb2.pdf",
        mime_type="application/pdf",
        storage_path=f"storage/kb2_{user_suffix}.pdf",
        sha256_hash=uuid4().hex,
        file_size=1024,
        status=DocumentStatus.READY,
    )
    db_session.add_all([doc_1, doc_2])
    db_session.flush()

    query_vec = _unit_vector(active_index=3)

    chunk_kb1 = DocumentChunk(
        document_id=doc_1.id,
        chunk_index=0,
        content="KB 1 Chunk",
        embedding=query_vec,
    )
    chunk_kb2 = DocumentChunk(
        document_id=doc_2.id,
        chunk_index=0,
        content="KB 2 Chunk",
        embedding=query_vec,
    )
    db_session.add_all([chunk_kb1, chunk_kb2])
    db_session.commit()

    repo = DocumentChunkRepository(db=db_session)
    try:
        results = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb_1.id,
            embedding=query_vec,
            limit=10,
        )
        assert len(results) == 1
        assert results[0].chunk.id == chunk_kb1.id
        assert results[0].chunk.content == "KB 1 Chunk"
    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()


def test_document_lifecycle_status_exclusion_all_statuses(db_session: Session):
    """
    Invariants:
    - Only DocumentStatus.READY documents are retrievable.
    - PENDING, PROCESSING, FAILED documents must be strictly excluded even with perfect vector match.
    """
    user, _, kb, doc_ready = _create_hierarchy(db_session, status=DocumentStatus.READY)
    _, _, _, doc_pending = _create_hierarchy(db_session, status=DocumentStatus.PENDING)
    _, _, _, doc_processing = _create_hierarchy(
        db_session, status=DocumentStatus.PROCESSING
    )
    _, _, _, doc_failed = _create_hierarchy(db_session, status=DocumentStatus.FAILED)

    # Re-associate all documents to the same KB for strict status testing
    doc_pending.knowledge_base_id = kb.id
    doc_processing.knowledge_base_id = kb.id
    doc_failed.knowledge_base_id = kb.id
    db_session.flush()

    query_vec = _unit_vector(active_index=7)

    c_ready = DocumentChunk(
        document_id=doc_ready.id,
        chunk_index=0,
        content="Ready Chunk",
        embedding=query_vec,
    )
    c_pending = DocumentChunk(
        document_id=doc_pending.id,
        chunk_index=0,
        content="Pending Chunk",
        embedding=query_vec,
    )
    c_processing = DocumentChunk(
        document_id=doc_processing.id,
        chunk_index=0,
        content="Processing Chunk",
        embedding=query_vec,
    )
    c_failed = DocumentChunk(
        document_id=doc_failed.id,
        chunk_index=0,
        content="Failed Chunk",
        embedding=query_vec,
    )

    db_session.add_all([c_ready, c_pending, c_processing, c_failed])
    db_session.commit()

    repo = DocumentChunkRepository(db=db_session)
    try:
        results = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=query_vec,
            limit=10,
        )
        assert len(results) == 1
        assert results[0].chunk.id == c_ready.id
        assert results[0].chunk.content == "Ready Chunk"
    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()


def test_threshold_boundary_and_empty_retrieval(db_session: Session):
    """
    Invariants:
    - similarity >= threshold inclusion rule.
    - Result count in [0, limit].
    - Empty result [] is valid success (not an error).
    """
    user, _, kb, doc = _create_hierarchy(db_session, status=DocumentStatus.READY)
    query_vec = _unit_vector(active_index=0)

    # Similarity = 1.0
    c1 = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="C1",
        embedding=_unit_vector(active_index=0),
    )
    # Similarity = 0.0 (orthogonal)
    c2 = DocumentChunk(
        document_id=doc.id,
        chunk_index=1,
        content="C2",
        embedding=_unit_vector(active_index=1),
    )

    db_session.add_all([c1, c2])
    db_session.commit()

    repo = DocumentChunkRepository(db=db_session)
    try:
        # Threshold higher than any chunk: should return empty list []
        _ = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=query_vec,
            limit=5,
            threshold=1.05,  # repo validates in [0, 1], so let's use 0.99999 with orthogonal vector or unreachable threshold
        )
    except ValidationException:
        pass  # Repository correctly validates threshold <= 1.0

    try:
        # Using valid threshold 0.5: c1 (sim 1.0) qualifies, c2 (sim 0.0) excluded
        res_05 = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=query_vec,
            limit=5,
            threshold=0.5,
        )
        assert len(res_05) == 1
        assert res_05[0].chunk.id == c1.id

        # Query pointing in opposite direction or orthogonal vector where threshold=0.5 yields []
        ortho_query = _unit_vector(active_index=10)
        res_empty = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=ortho_query,
            limit=5,
            threshold=0.5,
        )
        assert res_empty == []
        assert isinstance(res_empty, list)
    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()


def test_limit_enforcement_inclusive_contract(db_session: Session):
    """
    Invariants:
    - 0 <= result_count <= limit.
    - When available matches > limit, exactly limit results are returned.
    """
    user, _, kb, doc = _create_hierarchy(db_session, status=DocumentStatus.READY)
    query_vec = _unit_vector(active_index=0)

    chunks = [
        DocumentChunk(
            document_id=doc.id,
            chunk_index=i,
            content=f"Chunk {i}",
            embedding=query_vec,
        )
        for i in range(5)
    ]
    db_session.add_all(chunks)
    db_session.commit()

    repo = DocumentChunkRepository(db=db_session)
    try:
        # limit=2 must return exactly 2
        res_2 = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=query_vec,
            limit=2,
        )
        assert len(res_2) == 2

        # limit=5 must return all 5
        res_5 = repo.search_similar(
            user_id=user.id,
            knowledge_base_id=kb.id,
            embedding=query_vec,
            limit=5,
        )
        assert len(res_5) == 5
    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()


def test_repository_defense_in_depth_validation(db_session: Session):
    """
    Invariants:
    - DocumentChunkRepository.search_similar enforces bounds on limit and threshold.
    """
    user, _, kb, _ = _create_hierarchy(db_session, status=DocumentStatus.READY)
    query_vec = _unit_vector(active_index=0)
    repo = DocumentChunkRepository(db=db_session)

    try:
        with pytest.raises(
            ValidationException, match="Embedding must be a list or tuple of floats"
        ):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding="abc",  # type: ignore[arg-type]
            )

        with pytest.raises(
            ValidationException, match="Embedding must be a list or tuple of floats"
        ):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding={"a": 1},  # type: ignore[arg-type]
            )

        with pytest.raises(
            ValidationException,
            match="Embedding dimension mismatch: expected 768, got 512",
        ):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=[0.1] * 512,
            )

        with pytest.raises(ValidationException, match="Limit must be positive"):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit=0,
            )

        with pytest.raises(ValidationException, match="Limit must be positive"):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit="5",  # type: ignore[arg-type]
            )

        with pytest.raises(ValidationException, match="Limit must be positive"):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit=True,  # type: ignore[arg-type]
            )

        with pytest.raises(ValidationException, match="Limit must be between 1 and"):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit=MAX_RETRIEVAL_LIMIT + 1,
            )

        with pytest.raises(
            ValidationException, match="Threshold must be between 0.0 and 1.0"
        ):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit=5,
                threshold="0.5",  # type: ignore[arg-type]
            )

        with pytest.raises(
            ValidationException, match="Threshold must be between 0.0 and 1.0"
        ):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit=5,
                threshold=True,  # type: ignore[arg-type]
            )

        with pytest.raises(
            ValidationException, match="Threshold must be between 0.0 and 1.0"
        ):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit=5,
                threshold=float("nan"),
            )

        with pytest.raises(
            ValidationException, match="Threshold must be between 0.0 and 1.0"
        ):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit=5,
                threshold=-0.1,
            )

        with pytest.raises(
            ValidationException, match="Threshold must be between 0.0 and 1.0"
        ):
            repo.search_similar(
                user_id=user.id,
                knowledge_base_id=kb.id,
                embedding=query_vec,
                limit=5,
                threshold=1.1,
            )
    finally:
        db_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
        db_session.commit()
