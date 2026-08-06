"""Tenant isolation, verified against the real Postgres policies.

These are the tests that matter most in the whole suite. Every other failure
costs correctness; this one costs a customer's data. So the assertions are
deliberately adversarial: they issue queries with **no tenant predicate at all**
and require the database to return nothing, which is exactly what a forgotten
``WHERE`` clause looks like in production.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select, text

from agentlens_platform.db.models import Dataset, EvalCase, Tenant
from agentlens_platform.db.postgres import session_scope
from agentlens_platform.promotion import ExtractedCase, get_or_create_dataset, promote_cases
from agentlens_platform.tenancy import (
    APP_ROLE,
    TENANT_SCOPED_TABLES,
    current_tenant_setting,
    set_tenant_context,
    tenant_session,
)


async def _make_tenant_with_data(label: str, case_count: int = 3) -> tuple[uuid.UUID, str]:
    """Create a tenant plus a dataset of cases. Returns (tenant_id, slug)."""
    slug = f"t{uuid.uuid4().hex[:14]}"
    async with session_scope() as session:
        tenant = Tenant(slug=slug, name=label)
        session.add(tenant)
        await session.flush()
        tenant_id = tenant.id

    async with tenant_session(tenant_id) as session:
        dataset = await get_or_create_dataset(
            session, tenant_id=tenant_id, slug="d", name="D"
        )
        await promote_cases(
            session,
            [
                ExtractedCase(
                    external_id=f"{label}-case-{i}",
                    inputs={"question": f"{label} secret question {i}"},
                    expected_output=f"{label} secret answer {i}",
                )
                for i in range(case_count)
            ],
            tenant_id=tenant_id,
            dataset=dataset,
        )
    return tenant_id, slug


async def _cleanup(*tenant_ids: uuid.UUID) -> None:
    # Tenants themselves are not RLS-scoped (they have no tenant_id column),
    # so cleanup runs without a tenant context and cascades.
    async with session_scope() as session:
        for tenant_id in tenant_ids:
            obj = await session.get(Tenant, tenant_id)
            if obj is not None:
                await session.delete(obj)


async def test_rls_is_enabled_and_forced_on_every_scoped_table(pg_engine) -> None:
    """FORCE is the difference between real RLS and decorative RLS.

    Postgres exempts the table owner from policies unless FORCE is set, and the
    application usually connects as the owner. Without this assertion the whole
    isolation model could be inert while every table still reported
    ``rowsecurity = true``.
    """
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT relname, relrowsecurity, relforcerowsecurity "
                    "FROM pg_class WHERE relname = ANY(:names)"
                ),
                {"names": list(TENANT_SCOPED_TABLES)},
            )
        ).all()

    found = {name: (enabled, forced) for name, enabled, forced in rows}
    assert set(found) == set(TENANT_SCOPED_TABLES)
    for table, (enabled, forced) in found.items():
        assert enabled, f"{table} does not have RLS enabled"
        assert forced, f"{table} has RLS enabled but not FORCED -- the owner bypasses it"


async def test_every_scoped_table_has_a_policy(pg_engine) -> None:
    async with session_scope() as session:
        policies = set(
            (
                await session.execute(
                    text("SELECT tablename FROM pg_policies WHERE schemaname = 'public'")
                )
            )
            .scalars()
            .all()
        )
    assert set(TENANT_SCOPED_TABLES) <= policies


async def test_a_query_with_no_tenant_predicate_returns_only_this_tenants_rows(
    pg_engine,
) -> None:
    """The core guarantee, tested the way it actually fails in production.

    The SELECT below has no ``WHERE tenant_id`` at all -- it is precisely the
    forgotten-filter bug. The database must still return only one tenant's data.
    """
    alice_id, _ = await _make_tenant_with_data("alice", 3)
    bob_id, _ = await _make_tenant_with_data("bob", 5)
    try:
        async with tenant_session(alice_id) as session:
            cases = (await session.execute(select(EvalCase))).scalars().all()
            assert len(cases) == 3
            assert all(c.tenant_id == alice_id for c in cases)
            assert all("alice" in c.inputs["question"] for c in cases)

        async with tenant_session(bob_id) as session:
            cases = (await session.execute(select(EvalCase))).scalars().all()
            assert len(cases) == 5
            assert all("bob" in c.inputs["question"] for c in cases)
    finally:
        await _cleanup(alice_id, bob_id)


async def test_an_aggregate_cannot_count_another_tenants_rows(pg_engine) -> None:
    """Leaks through COUNT are still leaks -- they disclose volume."""
    alice_id, _ = await _make_tenant_with_data("alice", 3)
    bob_id, _ = await _make_tenant_with_data("bob", 7)
    try:
        async with tenant_session(alice_id) as session:
            total = (await session.execute(select(func.count()).select_from(EvalCase))).scalar()
            assert total == 3
    finally:
        await _cleanup(alice_id, bob_id)


async def test_fetching_another_tenants_row_by_primary_key_returns_nothing(
    pg_engine,
) -> None:
    """Knowing the UUID must not be enough. RLS applies to a PK lookup too."""
    alice_id, _ = await _make_tenant_with_data("alice", 2)
    bob_id, _ = await _make_tenant_with_data("bob", 2)
    try:
        async with tenant_session(bob_id) as session:
            bob_case = (await session.execute(select(EvalCase))).scalars().first()
            bob_case_id = bob_case.id

        async with tenant_session(alice_id) as session:
            stolen = await session.get(EvalCase, bob_case_id)
            assert stolen is None
    finally:
        await _cleanup(alice_id, bob_id)


async def test_writing_a_row_for_another_tenant_is_rejected(pg_engine) -> None:
    """WITH CHECK blocks writes, not just reads.

    A policy that only filtered SELECT would let one tenant plant rows in
    another's account.
    """
    alice_id, _ = await _make_tenant_with_data("alice", 1)
    bob_id, _ = await _make_tenant_with_data("bob", 1)
    try:
        with pytest.raises(Exception) as exc:
            async with tenant_session(alice_id) as session:
                session.add(
                    Dataset(tenant_id=bob_id, slug="planted", name="planted by alice")
                )
                await session.flush()
        assert "policy" in str(exc.value).lower() or "violates" in str(exc.value).lower()

        # Bob's data is untouched.
        async with tenant_session(bob_id) as session:
            datasets = (await session.execute(select(Dataset))).scalars().all()
            assert all(d.slug != "planted" for d in datasets)
    finally:
        await _cleanup(alice_id, bob_id)


async def test_an_unset_tenant_context_returns_nothing_rather_than_everything(
    pg_engine,
) -> None:
    """Fail closed. An unfiltered query must be useless, not omniscient."""
    alice_id, _ = await _make_tenant_with_data("alice", 3)
    try:
        # session_scope deliberately does NOT set a tenant context.
        async with session_scope() as session:
            assert await current_tenant_setting(session) in (None, "")
            cases = (await session.execute(select(EvalCase))).scalars().all()
            assert cases == []
    finally:
        await _cleanup(alice_id)


async def test_tenant_context_does_not_leak_across_pooled_connections(pg_engine) -> None:
    """SET LOCAL, not SET.

    With a connection pool, a session-scoped setting survives on the connection
    and the next borrower inherits it. That is a cross-tenant leak with extra
    steps, and it is invisible until it is a breach.
    """
    alice_id, _ = await _make_tenant_with_data("alice", 4)
    try:
        async with tenant_session(alice_id) as session:
            assert await current_tenant_setting(session) == str(alice_id)

        # A fresh transaction, very likely on the same pooled connection.
        async with session_scope() as session:
            leaked = await current_tenant_setting(session)
            assert leaked in (None, ""), f"tenant context leaked: {leaked!r}"
            assert (await session.execute(select(EvalCase))).scalars().all() == []
    finally:
        await _cleanup(alice_id)


async def test_context_is_restored_after_an_error(pg_engine) -> None:
    """A rolled-back transaction must not leave a tenant bound."""
    alice_id, _ = await _make_tenant_with_data("alice", 1)
    try:
        with pytest.raises(RuntimeError):
            async with tenant_session(alice_id) as session:
                await session.execute(select(EvalCase))
                raise RuntimeError("boom")

        async with session_scope() as session:
            assert await current_tenant_setting(session) in (None, "")
    finally:
        await _cleanup(alice_id)


async def test_set_tenant_context_is_parameterised_not_interpolated(pg_engine) -> None:
    """The tenant value must never be concatenated into SQL."""
    alice_id, _ = await _make_tenant_with_data("alice", 1)
    try:
        async with session_scope() as session:
            # A value that would break or subvert naive interpolation.
            hostile = "'; DROP TABLE eval_cases; --"
            await set_tenant_context(session, hostile)
            assert await current_tenant_setting(session) == hostile
            # The comparison to a uuid column fails for a non-uuid value, which
            # must surface as an error on the query -- not as data loss.
            with pytest.raises(Exception):
                await session.execute(select(EvalCase))
            await session.rollback()

        async with session_scope() as session:
            # The table is still there.
            assert (
                await session.execute(text("SELECT to_regclass('public.eval_cases')"))
            ).scalar() == "eval_cases"
    finally:
        await _cleanup(alice_id)


async def test_deleting_another_tenants_rows_affects_nothing(pg_engine) -> None:
    alice_id, _ = await _make_tenant_with_data("alice", 2)
    bob_id, _ = await _make_tenant_with_data("bob", 6)
    try:
        async with tenant_session(alice_id) as session:
            await session.execute(text("DELETE FROM eval_cases"))

        async with tenant_session(bob_id) as session:
            remaining = (await session.execute(select(EvalCase))).scalars().all()
            assert len(remaining) == 6
    finally:
        await _cleanup(alice_id, bob_id)


async def test_the_application_role_cannot_bypass_rls(pg_engine) -> None:
    """The check that the first version of this module silently failed.

    Policies were correct and completely inert because the app connected as the
    compose Postgres superuser, which ignores RLS regardless of FORCE. Verified
    at the time: rolsuper=True, rolbypassrls=True. This asserts the runtime role
    has neither attribute, so the policies above can actually apply.
    """
    async with session_scope() as session:
        role, superuser, bypass = (
            await session.execute(
                text(
                    "SELECT current_user, rolsuper, rolbypassrls "
                    "FROM pg_roles WHERE rolname = current_user"
                )
            )
        ).one()

    assert role == APP_ROLE, f"runtime connections must use {APP_ROLE}, got {role}"
    assert not superuser, "a superuser ignores every RLS policy"
    assert not bypass, "BYPASSRLS makes every policy inert"


async def test_the_app_role_cannot_change_its_own_policies(pg_engine) -> None:
    """Isolation that the isolated party can switch off is not isolation."""
    async with session_scope() as session:
        with pytest.raises(Exception):
            await session.execute(text("ALTER TABLE eval_cases DISABLE ROW LEVEL SECURITY"))
        await session.rollback()

    # Still enforced afterwards.
    alice_id, _ = await _make_tenant_with_data("alice", 2)
    try:
        async with session_scope() as session:
            assert (await session.execute(select(EvalCase))).scalars().all() == []
    finally:
        await _cleanup(alice_id)
