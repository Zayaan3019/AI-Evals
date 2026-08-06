"""enable row level security

Revision ID: 1464f2fe634a
Revises: dd0263b3db6c
Create Date: 2026-08-03 08:36:25.231007+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = '1464f2fe634a'
down_revision: str | None = 'dd0263b3db6c'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Enable RLS and install per-tenant policies.

    The statement list lives in ``agentlens_platform.tenancy`` so the policy
    definition has exactly one source. A migration that restated the SQL would
    drift from the module that documents why each clause is there.
    """
    from agentlens_platform.tenancy import enable_rls_statements

    for statement in enable_rls_statements():
        op.execute(statement)


def downgrade() -> None:
    from agentlens_platform.tenancy import disable_rls_statements

    for statement in disable_rls_statements():
        op.execute(statement)
