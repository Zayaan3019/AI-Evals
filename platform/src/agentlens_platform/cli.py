"""Operational entrypoints.

Provisioning a tenant and issuing its first API key has to happen before any
span can be ingested, and it must not require a running API (which would itself
need a key). So it lives here as a command.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select

from .auth import generate_api_key
from .config import get_settings
from .db.models import ApiKey, Tenant
from .db.postgres import dispose_engine, session_scope


async def _create_tenant(slug: str, name: str, key_name: str) -> int:
    async with session_scope() as session:
        existing = (
            await session.execute(select(Tenant).where(Tenant.slug == slug))
        ).scalar_one_or_none()
        if existing is None:
            tenant = Tenant(slug=slug, name=name)
            session.add(tenant)
            await session.flush()
        else:
            tenant = existing
            print(f"tenant {slug!r} already exists, issuing an additional key", file=sys.stderr)

        plaintext, key_hash, prefix = generate_api_key()
        session.add(
            ApiKey(tenant_id=tenant.id, name=key_name, key_hash=key_hash, prefix=prefix)
        )
        # Printed once. There is no recovery path -- only the hash is stored.
        print(plaintext)
    await dispose_engine()
    return 0


async def _serve() -> int:
    import uvicorn

    settings = get_settings()
    config = uvicorn.Config(
        "agentlens_platform.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
    await uvicorn.Server(config).serve()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentlens-platform")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-tenant", help="Provision a tenant and print one API key")
    create.add_argument("slug")
    create.add_argument("--name", default=None)
    create.add_argument("--key-name", default="default")

    sub.add_parser("serve", help="Run the HTTP API and OTLP gRPC listener")

    args = parser.parse_args(argv)
    if args.command == "create-tenant":
        return asyncio.run(_create_tenant(args.slug, args.name or args.slug, args.key_name))
    return asyncio.run(_serve())


if __name__ == "__main__":
    raise SystemExit(main())
