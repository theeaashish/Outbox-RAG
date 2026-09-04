"""add projects and project scope

Revision ID: c3d4e5f6a7b8
Revises: b3abf5835419
Create Date: 2026-09-04 17:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b3abf5835419"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create projects and backfill project ownership for existing rows."""
    op.create_table(
        "projects",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_projects_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_projects"),
        sa.UniqueConstraint("user_id", "name", name="uq_projects_user_name"),
    )
    op.create_index("ix_projects_user_id", "projects", ["user_id"], unique=False)

    op.add_column(
        "knowledge_bases",
        sa.Column("project_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("project_id", sa.Uuid(), nullable=True),
    )

    # Reuse each knowledge base UUID for its migrated project. This gives a
    # deterministic one-to-one mapping without requiring a database UUID
    # extension and keeps the backfill renderable in Alembic offline mode.
    # Disambiguate duplicate historical KnowledgeBase names deterministically
    # while guaranteeing that generated names never collide with another
    # generated name or any other original KnowledgeBase name for that user.
    op.execute(
        sa.text(
            """
            DO $$
            DECLARE
                kb RECORD;
                cand_name VARCHAR(255);
                n INT;
            BEGIN
                -- 1. Insert the deterministic primary KB for each (user_id, name).
                -- All (user_id, name) tuples here are guaranteed strictly distinct original names.
                INSERT INTO projects (id, user_id, name, description, created_at, updated_at)
                SELECT
                    id,
                    user_id,
                    name,
                    description,
                    created_at,
                    updated_at
                FROM (
                    SELECT
                        id,
                        user_id,
                        name,
                        description,
                        created_at,
                        updated_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY user_id, name
                            ORDER BY created_at ASC, id ASC
                        ) AS rn
                    FROM knowledge_bases
                ) ranked_kb
                WHERE rn = 1;

                -- 2. Process duplicate rows (rn > 1) in deterministic order.
                -- Find the lowest suffix (n >= 2) such that the generated name does not
                -- collide with ANY already-existing project name for that user (which includes
                -- all original names from step 1 as well as previously generated names).
                FOR kb IN
                    SELECT
                        id,
                        user_id,
                        name,
                        description,
                        created_at,
                        updated_at
                    FROM (
                        SELECT
                            id,
                            user_id,
                            name,
                            description,
                            created_at,
                            updated_at,
                            ROW_NUMBER() OVER (
                                PARTITION BY user_id, name
                                ORDER BY created_at ASC, id ASC
                            ) AS rn
                        FROM knowledge_bases
                    ) ranked_kb
                    WHERE rn > 1
                    ORDER BY user_id ASC, created_at ASC, id ASC
                LOOP
                    n := 2;
                    LOOP
                        cand_name := SUBSTR(kb.name, 1, 255 - LENGTH(' (' || n || ')')) || ' (' || n || ')';
                        IF NOT EXISTS (
                            SELECT 1
                            FROM projects
                            WHERE user_id = kb.user_id AND name = cand_name
                        ) THEN
                            EXIT;
                        END IF;
                        n := n + 1;
                    END LOOP;

                    INSERT INTO projects (id, user_id, name, description, created_at, updated_at)
                    VALUES (kb.id, kb.user_id, cand_name, kb.description, kb.created_at, kb.updated_at);
                END LOOP;
            END $$;
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE knowledge_bases
            SET project_id = id
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE conversations AS conversations
            SET project_id = knowledge_bases.project_id
            FROM knowledge_bases
            WHERE conversations.knowledge_base_id = knowledge_bases.id
            """
        )
    )

    op.alter_column("knowledge_bases", "project_id", nullable=False)
    op.alter_column("conversations", "project_id", nullable=False)

    op.create_foreign_key(
        "fk_knowledge_bases_project_id_projects",
        "knowledge_bases",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_conversations_project_id_projects",
        "conversations",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "uq_knowledge_bases_project_id",
        "knowledge_bases",
        ["project_id"],
    )
    op.create_index(
        "ix_knowledge_bases_project_id",
        "knowledge_bases",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_conversations_project_id",
        "conversations",
        ["project_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove project scope while retaining existing domain rows."""
    op.drop_index("ix_conversations_project_id", table_name="conversations")
    op.drop_index("ix_knowledge_bases_project_id", table_name="knowledge_bases")
    op.drop_constraint(
        "uq_knowledge_bases_project_id",
        "knowledge_bases",
        type_="unique",
    )
    op.drop_constraint(
        "fk_conversations_project_id_projects",
        "conversations",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_knowledge_bases_project_id_projects",
        "knowledge_bases",
        type_="foreignkey",
    )
    op.drop_column("conversations", "project_id")
    op.drop_column("knowledge_bases", "project_id")
    op.drop_index("ix_projects_user_id", table_name="projects")
    op.drop_table("projects")
