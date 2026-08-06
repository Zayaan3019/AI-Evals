"""policies tolerate empty tenant setting

Revision ID: c23dc6fedf4b
Revises: 5179404ea0cd
Create Date: 2026-08-03 08:40:57.535931+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'c23dc6fedf4b'
down_revision: str | None = '5179404ea0cd'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Re-create every policy with a NULLIF guard.

    ``SET LOCAL`` resets a custom setting to the empty string at commit, not to
    NULL, so on a pooled connection every transaction after the first
    tenant-scoped one evaluated ``''::uuid`` and raised instead of returning
    zero rows.
    """
    from agentlens_platform.tenancy import enable_rls_statements

    for statement in enable_rls_statements():
        op.execute(statement)


def downgrade() -> None:
    # Deliberately not restored: the previous policy raised on any unscoped
    # query over a reused connection.
    pass
