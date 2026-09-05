"""enforce composite ownership invariants

Revision ID: d5e6f7a8b9c0
Revises: c3d4e5f6a7b8
Create Date: 2026-09-05 18:15:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5e6f7a8b9c0"
down_revision: str | Sequence[str] | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # -------------------------------------------------------------------------
    # 1. Preflight Validation: Fail migration loudly if inconsistent rows exist
    # -------------------------------------------------------------------------
    op.execute(
        sa.text(
            """
            DO $$
            DECLARE
                mismatched_kb_users INT;
                mismatched_conv_users INT;
                mismatched_conv_projects INT;
            BEGIN
                -- Check a: KnowledgeBase.user_id != Project.user_id
                SELECT COUNT(*) INTO mismatched_kb_users
                FROM knowledge_bases kb
                JOIN projects p ON kb.project_id = p.id
                WHERE kb.user_id != p.user_id;

                IF mismatched_kb_users > 0 THEN
                    RAISE EXCEPTION 'Preflight check failed: % knowledge base(s) have user_id differing from their parent project.user_id', mismatched_kb_users;
                END IF;

                -- Check b: Conversation.user_id != Project.user_id
                SELECT COUNT(*) INTO mismatched_conv_users
                FROM conversations c
                JOIN projects p ON c.project_id = p.id
                WHERE c.user_id != p.user_id;

                IF mismatched_conv_users > 0 THEN
                    RAISE EXCEPTION 'Preflight check failed: % conversation(s) have user_id differing from their parent project.user_id', mismatched_conv_users;
                END IF;

                -- Check c: Conversation.project_id != KnowledgeBase.project_id
                SELECT COUNT(*) INTO mismatched_conv_projects
                FROM conversations c
                JOIN knowledge_bases kb ON c.knowledge_base_id = kb.id
                WHERE c.project_id != kb.project_id;

                IF mismatched_conv_projects > 0 THEN
                    RAISE EXCEPTION 'Preflight check failed: % conversation(s) have project_id differing from their knowledge_base.project_id', mismatched_conv_projects;
                END IF;
            END $$;
            """
        )
    )

    # -------------------------------------------------------------------------
    # 2. Supporting Unique Constraints for Composite Foreign Key Targets
    # -------------------------------------------------------------------------
    op.create_unique_constraint(
        "uq_projects_id_user_id",
        "projects",
        ["id", "user_id"],
    )
    op.create_unique_constraint(
        "uq_knowledge_bases_id_project_id",
        "knowledge_bases",
        ["id", "project_id"],
    )

    # -------------------------------------------------------------------------
    # 3. KnowledgeBase Composite Foreign Key: (project_id, user_id) -> projects(id, user_id)
    # -------------------------------------------------------------------------
    op.drop_constraint(
        "fk_knowledge_bases_project_id_projects",
        "knowledge_bases",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_knowledge_bases_project_id_user_id_projects",
        "knowledge_bases",
        "projects",
        ["project_id", "user_id"],
        ["id", "user_id"],
        ondelete="CASCADE",
    )

    # -------------------------------------------------------------------------
    # 4. Conversation Composite Foreign Keys:
    #    - (project_id, user_id) -> projects(id, user_id)
    #    - (knowledge_base_id, project_id) -> knowledge_bases(id, project_id)
    # -------------------------------------------------------------------------
    op.drop_constraint(
        "fk_conversations_project_id_projects",
        "conversations",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_conversations_project_id_user_id_projects",
        "conversations",
        "projects",
        ["project_id", "user_id"],
        ["id", "user_id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(
        "fk_conversations_knowledge_base_id_knowledge_bases",
        "conversations",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_conversations_kb_id_project_id_knowledge_bases",
        "conversations",
        "knowledge_bases",
        ["knowledge_base_id", "project_id"],
        ["id", "project_id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # -------------------------------------------------------------------------
    # 1. Revert Conversation composite foreign keys to original single-column FKs
    # -------------------------------------------------------------------------
    op.drop_constraint(
        "fk_conversations_kb_id_project_id_knowledge_bases",
        "conversations",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_conversations_knowledge_base_id_knowledge_bases",
        "conversations",
        "knowledge_bases",
        ["knowledge_base_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(
        "fk_conversations_project_id_user_id_projects",
        "conversations",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_conversations_project_id_projects",
        "conversations",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # -------------------------------------------------------------------------
    # 2. Revert KnowledgeBase composite foreign key to original single-column FK
    # -------------------------------------------------------------------------
    op.drop_constraint(
        "fk_knowledge_bases_project_id_user_id_projects",
        "knowledge_bases",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_knowledge_bases_project_id_projects",
        "knowledge_bases",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # -------------------------------------------------------------------------
    # 3. Drop supporting unique constraints
    # -------------------------------------------------------------------------
    op.drop_constraint(
        "uq_knowledge_bases_id_project_id",
        "knowledge_bases",
        type_="unique",
    )
    op.drop_constraint(
        "uq_projects_id_user_id",
        "projects",
        type_="unique",
    )
