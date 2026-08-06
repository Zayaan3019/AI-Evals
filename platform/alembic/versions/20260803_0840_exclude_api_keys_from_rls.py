"""exclude api keys from rls

Revision ID: 5179404ea0cd
Revises: 3b17964e8a95
Create Date: 2026-08-03 08:40:15.304157+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = '5179404ea0cd'
down_revision: str | None = '3b17964e8a95'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Remove the tenant policy from ``api_keys``.

    An earlier migration scoped it like every other tenant table. That is
    circular: ``api_keys`` is the table consulted *to determine* which tenant a
    request belongs to, so with a policy attached authentication needs a tenant
    context it can only get by authenticating. The practical effect was that
    every request failed to authenticate and no API key could be issued.

    The table stores SHA-256 hashes, is reachable only by exact-hash lookup, and
    still cascades from ``tenants``.
    """
    op.execute("DROP POLICY IF EXISTS api_keys_tenant_isolation ON api_keys")
    op.execute("ALTER TABLE api_keys NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE api_keys DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    # Deliberately not restored: the previous state made authentication
    # impossible, so re-creating it would only reproduce a known outage.
    pass
