"""Request dependencies: who is asking, and what they may do.

Two identities, deliberately separate:

* :data:`TenantDep` -- a caller holding a valid API key. Required for anything
  that changes state.
* :data:`ReaderDep` -- a caller allowed to *read* one tenant. A valid key
  qualifies. So does no key at all, but only when the deployment names a public
  read-only tenant (``AGENTLENS_PUBLIC_READ_TENANT``), and only for that tenant.

The split exists so "public read" cannot quietly become "public write". A
handler opts in to anonymous access by depending on ``ReaderDep``; a handler
that mutates keeps ``TenantDep`` and stays closed no matter how the deployment
is configured. OTLP ingestion authenticates on its own path and never consults
either.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import AuthenticatedTenant, authenticate
from ..config import get_settings
from ..db.models import Tenant
from ..db.postgres import get_session

#: ``api_key_id`` recorded for anonymous public reads, so they are identifiable
#: in logs and can never be mistaken for a real key.
PUBLIC_READ_KEY_ID = "public-read"


def _unauthorised() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or missing API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


async def current_tenant(
    session: Annotated[AsyncSession, Depends(get_session)],
    x_agentlens_api_key: Annotated[str | None, Header()] = None,
) -> AuthenticatedTenant:
    tenant = await authenticate(session, x_agentlens_api_key)
    if tenant is None:
        raise _unauthorised()
    return tenant


async def current_reader(
    session: Annotated[AsyncSession, Depends(get_session)],
    x_agentlens_api_key: Annotated[str | None, Header()] = None,
) -> AuthenticatedTenant:
    if x_agentlens_api_key:
        # A key was presented, so it must be valid. A wrong key is a 401 even
        # when a public tenant exists: silently downgrading a failed login to
        # anonymous would hide a misconfigured client behind plausible data
        # from the wrong tenant.
        tenant = await authenticate(session, x_agentlens_api_key)
        if tenant is None:
            raise _unauthorised()
        return tenant

    slug = get_settings().public_read_tenant
    if not slug:
        raise _unauthorised()

    row = (
        await session.execute(
            select(Tenant.id, Tenant.slug).where(Tenant.slug == slug, Tenant.is_active.is_(True))
        )
    ).first()
    if row is None:
        raise _unauthorised()
    return AuthenticatedTenant(
        tenant_id=str(row.id), tenant_slug=row.slug, api_key_id=PUBLIC_READ_KEY_ID
    )


TenantDep = Annotated[AuthenticatedTenant, Depends(current_tenant)]
ReaderDep = Annotated[AuthenticatedTenant, Depends(current_reader)]
