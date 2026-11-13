"""Test fixtures.

Every test in this package runs against the real Postgres and the real
ClickHouse from ``docker-compose.yml``. Nothing is faked. The behaviours under
test -- TTL, sparse-index scans, check constraints, foreign keys, transactional
rollback, ``Map(String,String)`` coercion -- exist only in the real engines, so
a substitute would test the substitute.

Isolation comes from giving each test session its own scratch ClickHouse
database and its own tenant slug, not from mocking.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from sqlalchemy import text

from agentlens_platform.auth import generate_api_key
from agentlens_platform.config import Settings, get_settings
from agentlens_platform.db.clickhouse import (
    build_client,
    drop_row_policies,
    ensure_reader,
    ensure_schema,
)
from agentlens_platform.db.models import ApiKey, Tenant
from agentlens_platform.db.postgres import build_engine, dispose_engine, session_scope
from agentlens_platform.ingest.writer import SpanWriter

#: Raised instead of skipping when the stack is down. A suite that silently
#: skips on missing infrastructure reports green while testing nothing, which is
#: the failure mode this whole project exists to prevent.
_STACK_DOWN = (
    "{name} is not reachable at {url}. Start the stack with "
    "`docker compose up -d` from the repository root."
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest_asyncio.fixture(autouse=True)
async def _dispose_global_engine() -> AsyncIterator[None]:
    """Drop the process-wide engine after every test.

    ``db.postgres`` memoises one engine per process, but pytest-asyncio gives
    each test its own event loop, and an asyncpg pool is bound to the loop that
    created it. Reusing it across tests therefore fails with confusing
    AttributeErrors from inside the driver rather than an obvious message. This
    is a property of the async driver, not of the tests -- production has one
    loop for the process lifetime, which is exactly why the memoisation is
    correct there and wrong here.
    """
    yield
    await dispose_engine()


@pytest.fixture(scope="session")
def clickhouse_database() -> str:
    """A scratch database per session, dropped afterwards."""
    return f"agentlens_test_{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="session")
def ch_client(settings: Settings, clickhouse_database: str) -> Iterator[object]:
    try:
        admin = build_client(settings)
        admin.query("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            _STACK_DOWN.format(name="ClickHouse", url=settings.clickhouse_url)
        ) from exc

    admin.command(f"CREATE DATABASE IF NOT EXISTS {clickhouse_database}")
    client = build_client(settings, database=clickhouse_database)
    ensure_schema(client, clickhouse_database, settings.span_retention_days)
    ensure_reader(
        client,
        clickhouse_database,
        reader_user=settings.clickhouse_reader_user,
        reader_password=settings.clickhouse_reader_secret,
        admin_user=settings.clickhouse_user,
    )
    try:
        yield client
    finally:
        try:
            # Policies outlive a dropped table in ClickHouse's access storage.
            drop_row_policies(admin, clickhouse_database)
            admin.command(f"DROP DATABASE IF EXISTS {clickhouse_database}")
        finally:
            client.close()
            admin.close()


@pytest.fixture(scope="session")
def ch_reader(settings: Settings, clickhouse_database: str, ch_client) -> Iterator[object]:
    """The tenant-restricted reader the API uses, on the scratch database."""
    client = build_client(settings, database=clickhouse_database, reader=True)
    try:
        yield client
    finally:
        client.close()


@pytest_asyncio.fixture(scope="function")
async def pg_engine(settings: Settings) -> AsyncIterator[object]:
    """Verify Postgres is reachable and that migrations have been applied."""
    engine = build_engine(settings)
    try:
        async with engine.connect() as conn:
            applied = (
                await conn.execute(
                    text("SELECT count(*) FROM information_schema.tables "
                         "WHERE table_schema='public' AND table_name='tenants'")
                )
            ).scalar()
        if not applied:
            raise RuntimeError(
                "Postgres schema is missing. Run `alembic upgrade head` in platform/."
            )
    except RuntimeError:
        await engine.dispose()
        raise
    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        raise RuntimeError(
            _STACK_DOWN.format(name="Postgres", url=settings.postgres_dsn)
        ) from exc
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def tenant(pg_engine) -> AsyncIterator[tuple[str, str]]:
    """Create a throwaway tenant and key; return ``(slug, plaintext_key)``.

    Removed afterwards. The cascade on api_keys means deleting the tenant is
    sufficient -- which is itself worth exercising, since it is a real foreign
    key with a real ON DELETE CASCADE.
    """
    slug = f"t{uuid.uuid4().hex[:16]}"
    plaintext, key_hash, prefix = generate_api_key()
    async with session_scope() as session:
        row = Tenant(slug=slug, name=f"test tenant {slug}")
        session.add(row)
        await session.flush()
        session.add(ApiKey(tenant_id=row.id, name="test", key_hash=key_hash, prefix=prefix))
        tenant_id = row.id

    yield slug, plaintext

    async with session_scope() as session:
        obj = await session.get(Tenant, tenant_id)
        if obj is not None:
            await session.delete(obj)
    await dispose_engine()


@pytest_asyncio.fixture(scope="function")
async def writer(ch_client, clickhouse_database: str) -> AsyncIterator[SpanWriter]:
    instance = SpanWriter(
        ch_client,
        database=clickhouse_database,
        batch_size=500,
        flush_interval_seconds=0.2,
        max_buffered_spans=5_000,
    )
    await instance.start()
    try:
        yield instance
    finally:
        await instance.aclose()
