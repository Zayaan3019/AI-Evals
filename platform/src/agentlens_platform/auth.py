"""API key authentication and tenant resolution.

Keys are random 256-bit values, shown once at creation and stored only as a
SHA-256 hash. SHA-256 rather than a password KDF is the right choice *here*
specifically because the key is high-entropy and machine-generated: bcrypt's
work factor exists to slow brute force against low-entropy human passwords, and
against 256 bits of entropy there is nothing to slow down. It would only add
latency to the ingestion hot path, which authenticates every request.

Lookup is by hash, which is indexed and unique, so authentication costs one
indexed read. Comparison still uses ``compare_digest`` so the verification step
does not leak timing information.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db.models import ApiKey, Tenant

#: Distinguishes an agentlens key on sight, e.g. in a leaked log or a git diff.
KEY_PREFIX = "al_"
#: Characters of the plaintext retained for display. Enough to identify a key
#: in a list, far too few to reconstruct it.
DISPLAY_PREFIX_LENGTH = 11


@dataclass(frozen=True, slots=True)
class AuthenticatedTenant:
    """Result of a successful authentication."""

    tenant_id: str
    tenant_slug: str
    api_key_id: str


def generate_api_key() -> tuple[str, str, str]:
    """Return ``(plaintext, key_hash, display_prefix)``.

    The plaintext is returned to the caller once and never stored.
    """
    plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
    return plaintext, hash_api_key(plaintext), plaintext[:DISPLAY_PREFIX_LENGTH]


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


async def authenticate(session: AsyncSession, plaintext: str | None) -> AuthenticatedTenant | None:
    """Resolve a plaintext key to its tenant, or ``None``.

    Returns ``None`` for every failure mode -- unknown key, revoked key, expired
    key, disabled tenant -- so a caller cannot distinguish them and use the
    difference to enumerate valid keys.
    """
    if not plaintext:
        return None

    digest = hash_api_key(plaintext)
    result = await session.execute(
        select(ApiKey, Tenant).join(Tenant, ApiKey.tenant_id == Tenant.id).where(
            ApiKey.key_hash == digest
        )
    )
    row = result.first()
    if row is None:
        return None

    api_key, tenant = row
    # Redundant given the indexed equality lookup above, but it keeps the
    # verification constant-time with respect to the stored value.
    if not hmac.compare_digest(api_key.key_hash, digest):
        return None
    if not api_key.is_active or not tenant.is_active:
        return None
    if api_key.expires_at is not None and api_key.expires_at <= datetime.now(timezone.utc):
        return None

    return AuthenticatedTenant(
        tenant_id=str(tenant.id),
        tenant_slug=tenant.slug,
        api_key_id=str(api_key.id),
    )
