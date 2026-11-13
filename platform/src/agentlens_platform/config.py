"""Runtime configuration, resolved from the environment.

Every value has a working local default matching ``docker-compose.yml`` so the
stack runs with no setup, but nothing here is a secret in disguise: the
Postgres and ClickHouse passwords default to the local compose credentials and
are expected to be overridden in any real deployment.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTLENS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # -- Postgres: relational state, transactions, RLS --------------------
    postgres_dsn: str = Field(
        default="postgresql+asyncpg://agentlens:agentlens@localhost:55433/agentlens",
        description="SQLAlchemy async DSN. Must use the asyncpg driver.",
    )
    #: DSN the application uses at run time. Connects as a non-superuser role
    #: so row-level security actually applies -- a superuser bypasses every
    #: policy regardless of FORCE. ``postgres_dsn`` above remains the
    #: owner/migration connection and must not be used to serve requests.
    postgres_app_dsn: str = Field(
        default="postgresql+asyncpg://agentlens_app:agentlens_app@localhost:55433/agentlens",
        description="Runtime DSN. Must be a role with neither SUPERUSER nor BYPASSRLS.",
    )
    #: Pool sizing. The ingestion path does not touch Postgres per span (see
    #: ingest/writer.py), so this pool serves the API only.
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 10

    # -- ClickHouse: span storage -----------------------------------------
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 58123
    clickhouse_user: str = "agentlens"
    clickhouse_password: str = "agentlens"
    clickhouse_database: str = "agentlens"
    clickhouse_secure: bool = False
    #: The API reads spans as this user, which a ClickHouse row policy limits
    #: to one tenant per query (see db/clickhouse.py). The password defaults to
    #: one derived from clickhouse_password, so a deployment that only sets the
    #: admin password still gets a non-guessable reader password.
    clickhouse_reader_user: str = "agentlens_reader"
    clickhouse_reader_password: str | None = None

    # -- Redis: cache, rate limits, idempotency ---------------------------
    redis_url: str = "redis://localhost:56380/0"

    # -- Ingestion ---------------------------------------------------------
    otlp_grpc_host: str = "0.0.0.0"
    otlp_grpc_port: int = 4317
    #: Spans buffered before a forced flush. ClickHouse strongly prefers large
    #: infrequent inserts: each INSERT creates a part, and too many small parts
    #: triggers the "too many parts" backpressure error. Batching is not an
    #: optimisation here, it is a correctness requirement of the engine.
    ingest_batch_size: int = 5_000
    #: Maximum time a span waits in the buffer before being flushed anyway, so a
    #: low-traffic tenant still sees its spans promptly.
    ingest_flush_interval_seconds: float = 2.0
    #: Hard ceiling on the in-memory buffer. Past this, ingestion sheds load and
    #: counts it rather than growing without bound and being OOM-killed.
    ingest_max_buffered_spans: int = 200_000

    # -- Retention ---------------------------------------------------------
    #: ClickHouse TTL for raw spans, in days. Enforced by the table definition,
    #: so expiry is a background merge rather than a DELETE.
    span_retention_days: int = 30

    # -- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8100

    #: Slug of a tenant whose data may be READ without an API key, or None.
    #: Exists for a public read-only deployment: the alternative is shipping a key
    #: in the browser bundle or at the edge, and a key that everyone holds is
    #: not a credential. Anonymous access is read-only by construction -- it is
    #: honoured only by GET handlers, and OTLP ingestion never consults it.
    public_read_tenant: str | None = None

    @property
    def clickhouse_reader_secret(self) -> str:
        if self.clickhouse_reader_password:
            return self.clickhouse_reader_password
        import hashlib

        return hashlib.sha256(
            f"agentlens-reader:{self.clickhouse_password}".encode()
        ).hexdigest()

    @property
    def clickhouse_url(self) -> str:
        scheme = "https" if self.clickhouse_secure else "http"
        return f"{scheme}://{self.clickhouse_host}:{self.clickhouse_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
