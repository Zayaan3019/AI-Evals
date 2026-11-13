"""Tenant isolation enforced by Postgres row-level security.

Application-level ``WHERE tenant_id = ...`` filters are necessary but not
sufficient. They are correct exactly as long as every query author remembers
them, forever, including in the ad-hoc script written at 2am during an incident.
The failure mode is silent and catastrophic: one missing predicate leaks another
customer's data, and nothing in the code looks wrong.

RLS moves the guarantee into the database, where it holds regardless of what the
query says. A query that forgets its tenant filter returns zero rows rather than
everyone's rows.

Three details make this actually work rather than merely look like it works:

``FORCE ROW LEVEL SECURITY``
    Postgres exempts the table *owner* from RLS by default. Without FORCE, every
    policy here would be silently inert while the tables still reported
    ``rowsecurity = true``.

A non-superuser application role
    FORCE is not enough on its own. Superusers -- and any role with BYPASSRLS --
    ignore policies entirely, and FORCE does not change that because it only
    removes the *owner's* exemption. The docker-compose Postgres creates its
    POSTGRES_USER as a superuser, so the first version of this module had
    correct policies that enforced nothing at all; ``rolsuper`` and
    ``rolbypassrls`` were both true. The application now connects as
    :data:`APP_ROLE`, and ``test_the_application_role_cannot_bypass_rls``
    asserts that it cannot escape.

``SET LOCAL``, not ``SET``
    ``SET LOCAL`` is scoped to the transaction, so the tenant context cannot
    survive on a pooled connection and leak into the next request that borrows
    it. With a connection pool, a plain ``SET`` is a cross-tenant data leak with
    extra steps.

ClickHouse enforces the same rule with a row policy on the span table: the
API reads as a restricted user whose policy is ``tenant_id =
getSetting('SQL_agentlens_tenant')``, set per query (db/clickhouse.py). An
earlier version of this note said ClickHouse had no equivalent and relied on
the predicate alone; it does, and the predicate is now the fast path rather
than the only guard.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .db.postgres import get_sessionmaker

#: The role the application connects as at run time. Deliberately NOT the role
#: that owns the tables and runs migrations: superusers and roles with
#: BYPASSRLS ignore every policy, and ``FORCE ROW LEVEL SECURITY`` does not
#: help -- FORCE only removes the *owner's* exemption. The compose Postgres
#: creates its POSTGRES_USER as a superuser, so running the app as that user
#: left the policies correct and completely unenforced.
APP_ROLE = "agentlens_app"

#: Session variable the policies read. A dotted name is required: Postgres only
#: permits custom settings inside a namespace.
TENANT_SETTING = "app.current_tenant"

#: Tables carrying a ``tenant_id`` and therefore needing a policy. Kept explicit
#: rather than derived from metadata so that adding a tenant-scoped table
#: without a policy is a visible omission in a migration diff.
#: ``api_keys`` is deliberately absent. It is the table consulted *to determine*
#: which tenant a request belongs to, so scoping it by tenant is circular: with
#: a policy attached, authentication would need a tenant context it can only
#: obtain by authenticating. It stores SHA-256 hashes rather than credentials,
#: it is reachable only by exact-hash lookup, and it still cascades from
#: ``tenants``. ``tenants`` itself has no ``tenant_id`` column and is likewise
#: unscoped.
TENANT_SCOPED_TABLES: tuple[str, ...] = (
    "datasets",
    "eval_cases",
    "evaluators",
    "eval_runs",
    "eval_results",
    "human_labels",
)


def enable_rls_statements() -> list[str]:
    """DDL enabling RLS and installing one policy per tenant-scoped table.

    The policy compares against ``current_setting(..., true)``, where ``true``
    means "missing is NULL rather than an error". ``NULLIF(..., '')`` is then
    load-bearing and not defensive padding: once a setting has been assigned in
    a session, ``SET LOCAL`` resets it to the **empty string** at commit, not to
    NULL. On a pooled connection that means every transaction after the first
    tenant-scoped one saw ``''``, and ``''::uuid`` raises
    ``InvalidTextRepresentationError`` instead of returning nothing. Guarding
    only for NULL turned an unscoped query from "zero rows" into a hard error
    on exactly the connections that had already served a request.

    With the guard, an unset tenant context yields **zero rows**. Failing closed
    is the correct default: an empty result is recoverable, a query that
    accidentally ran unfiltered is a breach.
    """
    statements: list[str] = []
    for table in TENANT_SCOPED_TABLES:
        statements.append(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # Without FORCE the owner bypasses every policy below.
        statements.append(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        statements.append(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}")
        statements.append(
            f"""
            CREATE POLICY {table}_tenant_isolation ON {table}
            USING (
                NULLIF(current_setting('{TENANT_SETTING}', true), '') IS NOT NULL
                AND tenant_id = NULLIF(current_setting('{TENANT_SETTING}', true), '')::uuid
            )
            WITH CHECK (
                NULLIF(current_setting('{TENANT_SETTING}', true), '') IS NOT NULL
                AND tenant_id = NULLIF(current_setting('{TENANT_SETTING}', true), '')::uuid
            )
            """
        )
    return statements


def disable_rls_statements() -> list[str]:
    statements: list[str] = []
    for table in TENANT_SCOPED_TABLES:
        statements.append(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}")
        statements.append(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        statements.append(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    return statements


async def set_tenant_context(session: AsyncSession, tenant_id: uuid.UUID | str) -> None:
    """Bind the current transaction to one tenant.

    Parameterised rather than interpolated. ``SET LOCAL`` does not accept bind
    parameters directly, so this uses ``set_config``, which does -- string
    interpolation of a value into DDL-adjacent SQL is exactly the hole the
    policies exist to close.

    The third argument ``true`` makes it transaction-local.
    """
    await session.execute(
        text(f"SELECT set_config('{TENANT_SETTING}', :tenant, true)"),
        {"tenant": str(tenant_id)},
    )


@asynccontextmanager
async def tenant_session(tenant_id: uuid.UUID | str) -> AsyncIterator[AsyncSession]:
    """A transaction scoped to one tenant, with RLS active.

    Every request-handling path should use this rather than ``session_scope``.
    The tenant context is established inside the transaction that will run the
    queries, so it cannot outlive it on a pooled connection.
    """
    maker = get_sessionmaker()
    async with maker() as session:
        try:
            await set_tenant_context(session, tenant_id)
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def current_tenant_setting(session: AsyncSession) -> str | None:
    """Read back the active tenant context. Used by tests and diagnostics."""
    result = await session.execute(
        text(f"SELECT current_setting('{TENANT_SETTING}', true)")
    )
    return result.scalar()
