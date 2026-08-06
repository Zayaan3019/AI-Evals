"""create non superuser app role

Revision ID: 3b17964e8a95
Revises: 1464f2fe634a
Create Date: 2026-08-03 08:38:21.121569+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = '3b17964e8a95'
down_revision: str | None = '1464f2fe634a'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the role the application connects as at run time.

    Row-level security is bypassed entirely by superusers and by any role with
    BYPASSRLS -- ``FORCE ROW LEVEL SECURITY`` does not change that, because
    FORCE only removes the *owner's* exemption. The docker-compose Postgres
    creates POSTGRES_USER as a superuser, so connecting the application as
    ``agentlens`` left every policy correct and completely inert. Verified
    directly: ``rolsuper=True, rolbypassrls=True``.

    So DDL and DML are split by role, which is the right production shape
    anyway: migrations run as the owner, the application runs as a role with
    only the data privileges it needs and no ability to escape a policy.
    """
    from agentlens_platform.tenancy import APP_ROLE

    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                -- Local-development password. Deployments must override it;
                -- it is not a secret and is not treated as one.
                CREATE ROLE {APP_ROLE} LOGIN PASSWORD 'agentlens_app'
                    NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
            ELSE
                ALTER ROLE {APP_ROLE} NOSUPERUSER NOBYPASSRLS;
            END IF;
        END
        $$;
        """
    )
    op.execute(f"GRANT CONNECT ON DATABASE agentlens TO {APP_ROLE}")
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {APP_ROLE}"
    )
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")
    # Tables created by later migrations must be reachable without another grant.
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {APP_ROLE}"
    )
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"GRANT USAGE, SELECT ON SEQUENCES TO {APP_ROLE}"
    )


def downgrade() -> None:
    from agentlens_platform.tenancy import APP_ROLE

    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM {APP_ROLE}"
    )
    op.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {APP_ROLE}")
    op.execute(f"REVOKE ALL ON SCHEMA public FROM {APP_ROLE}")
    op.execute(f"DROP ROLE IF EXISTS {APP_ROLE}")
