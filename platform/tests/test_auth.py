"""API key authentication against the real Postgres.

Uses real rows and real constraints: the uniqueness of ``key_hash``, the
``ON DELETE CASCADE`` from tenants to keys, and the tenant slug check
constraint are all enforced by the database, not by application code, so they
can only be tested against a database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from agentlens_platform.auth import (
    KEY_PREFIX,
    authenticate,
    generate_api_key,
    hash_api_key,
)
from agentlens_platform.db.models import ApiKey, Tenant
from agentlens_platform.db.postgres import get_sessionmaker, session_scope


async def _make_tenant(slug: str | None = None, **kwargs) -> tuple[str, str, uuid.UUID]:
    """Create a tenant plus one key. Returns ``(slug, plaintext, tenant_id)``."""
    slug = slug or f"t{uuid.uuid4().hex[:16]}"
    plaintext, key_hash, prefix = generate_api_key()
    async with session_scope() as session:
        tenant = Tenant(slug=slug, name=slug, is_active=kwargs.pop("tenant_active", True))
        session.add(tenant)
        await session.flush()
        session.add(
            ApiKey(tenant_id=tenant.id, name="k", key_hash=key_hash, prefix=prefix, **kwargs)
        )
        return slug, plaintext, tenant.id


async def _cleanup(tenant_id: uuid.UUID) -> None:
    async with session_scope() as session:
        obj = await session.get(Tenant, tenant_id)
        if obj is not None:
            await session.delete(obj)


def test_generated_keys_are_prefixed_and_high_entropy() -> None:
    plaintext, digest, prefix = generate_api_key()
    assert plaintext.startswith(KEY_PREFIX)
    assert prefix == plaintext[:11]
    assert digest == hash_api_key(plaintext)
    # token_urlsafe(32) is 256 bits, base64url-encoded to 43 characters.
    assert len(plaintext) - len(KEY_PREFIX) >= 43


def test_generated_keys_are_unique() -> None:
    keys = {generate_api_key()[0] for _ in range(1000)}
    assert len(keys) == 1000


async def test_a_valid_key_resolves_to_its_tenant(pg_engine) -> None:
    slug, plaintext, tenant_id = await _make_tenant()
    try:
        async with session_scope() as session:
            result = await authenticate(session, plaintext)
        assert result is not None
        assert result.tenant_slug == slug
    finally:
        await _cleanup(tenant_id)


async def test_plaintext_is_never_stored(pg_engine) -> None:
    """A database dump must not be a credential dump."""
    slug, plaintext, tenant_id = await _make_tenant()
    try:
        async with session_scope() as session:
            row = (
                await session.execute(select(ApiKey).where(ApiKey.key_hash == hash_api_key(plaintext)))
            ).scalar_one()
            assert plaintext not in (row.key_hash, row.prefix, row.name)
            # The stored prefix must be too short to be useful on its own.
            assert len(row.prefix) < len(plaintext) // 3
    finally:
        await _cleanup(tenant_id)


@pytest.mark.parametrize("bad", ["", None, "al_not-a-real-key", "totally wrong"])
async def test_unknown_keys_are_rejected(pg_engine, bad) -> None:
    async with session_scope() as session:
        assert await authenticate(session, bad) is None


async def test_revoked_key_is_rejected(pg_engine) -> None:
    slug, plaintext, tenant_id = await _make_tenant(is_active=False)
    try:
        async with session_scope() as session:
            assert await authenticate(session, plaintext) is None
    finally:
        await _cleanup(tenant_id)


async def test_expired_key_is_rejected(pg_engine) -> None:
    slug, plaintext, tenant_id = await _make_tenant(
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    try:
        async with session_scope() as session:
            assert await authenticate(session, plaintext) is None
    finally:
        await _cleanup(tenant_id)


async def test_unexpired_key_is_accepted(pg_engine) -> None:
    slug, plaintext, tenant_id = await _make_tenant(
        expires_at=datetime.now(timezone.utc) + timedelta(days=1)
    )
    try:
        async with session_scope() as session:
            assert await authenticate(session, plaintext) is not None
    finally:
        await _cleanup(tenant_id)


async def test_disabled_tenant_rejects_its_own_valid_keys(pg_engine) -> None:
    """Deactivating a tenant must cut off access without revoking each key."""
    slug, plaintext, tenant_id = await _make_tenant(tenant_active=False)
    try:
        async with session_scope() as session:
            assert await authenticate(session, plaintext) is None
    finally:
        await _cleanup(tenant_id)


async def test_deleting_a_tenant_cascades_to_its_keys(pg_engine) -> None:
    """Enforced by the foreign key, not by application code."""
    slug, plaintext, tenant_id = await _make_tenant()
    digest = hash_api_key(plaintext)
    await _cleanup(tenant_id)
    async with session_scope() as session:
        remaining = (
            await session.execute(select(ApiKey).where(ApiKey.key_hash == digest))
        ).scalar_one_or_none()
        assert remaining is None


async def test_tenant_slug_format_is_enforced_by_the_database(pg_engine) -> None:
    """A slug becomes ClickHouse's tenant_id; junk there is junk forever.

    Uses a raw session rather than ``session_scope``: that helper commits on a
    clean exit, and a session whose flush has already failed cannot commit.
    """
    async with get_sessionmaker()() as session:
        session.add(Tenant(slug="Not Valid Slug!", name="x"))
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()


async def test_duplicate_slugs_are_rejected(pg_engine) -> None:
    slug, _, tenant_id = await _make_tenant()
    try:
        async with get_sessionmaker()() as session:
            session.add(Tenant(slug=slug, name="duplicate"))
            with pytest.raises(IntegrityError):
                await session.flush()
            await session.rollback()
    finally:
        await _cleanup(tenant_id)
