"""FastAPI authentication dependencies.

Authentication answers who is calling. Resource helpers answer whether that
person may touch a space, document, or job. A bearer token is never a grant
to every MongoDB document.
"""

from __future__ import annotations

import hmac
from typing import Annotated, Any

from bson import ObjectId
from fastapi import Header, HTTPException, Query

from services.auth.identity import AuthenticatedUser, IdentityMismatch, bind_user_id, same_user_id
from services.auth.node_token import AuthNotConfigured, verify_auth_token
from services.auth.space_access import AccessDenied


def raise_for_access(error: AccessDenied) -> None:
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


def require_matching_user(actor: AuthenticatedUser, supplied: Any) -> str:
    try:
        return bind_user_id(actor, supplied)
    except IdentityMismatch as error:
        raise HTTPException(status_code=403, detail="Forbidden") from error


def forwarded_user_token(actor: AuthenticatedUser, body_token: str | None) -> str:
    """Return the verified request token. A second token for someone else is rejected."""
    presented = (body_token or "").strip()
    if not presented:
        return actor.token
    if len(presented) == len(actor.token) and hmac.compare_digest(presented, actor.token):
        return actor.token
    try:
        other = verify_auth_token(presented)
    except AuthNotConfigured as error:
        raise HTTPException(status_code=401, detail="Unauthorized") from error
    if other is None or not same_user_id(other.user_id, actor.user_id):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return actor.token


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, rest = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return rest.strip()


async def require_user(
    authorization: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Query()] = None,
) -> AuthenticatedUser:
    header_token = _bearer_token(authorization)
    query_token = access_token.strip() if isinstance(access_token, str) else ""
    primary = header_token or query_token
    if not primary:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        if header_token and query_token:
            header_identity = verify_auth_token(header_token)
            query_identity = verify_auth_token(query_token)
            if (
                header_identity is None
                or query_identity is None
                or not same_user_id(header_identity.user_id, query_identity.user_id)
            ):
                raise HTTPException(status_code=401, detail="Unauthorized")
        identity = verify_auth_token(primary)
    except AuthNotConfigured as error:
        raise HTTPException(status_code=401, detail="Unauthorized") from error
    if identity is None or not ObjectId.is_valid(identity.user_id):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return AuthenticatedUser(
        user_id=identity.user_id,
        provider=identity.provider,
        expires_at=identity.expires_at,
        token=primary,
    )
