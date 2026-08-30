from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from .config import get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> str:
    """Accepts `X-API-Key: <key>` or `Authorization: Bearer <key>`."""
    s = get_settings()
    if not s.require_auth:
        return "anonymous"

    presented = x_api_key
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()

    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    for known in s.keys:
        if secrets.compare_digest(presented, known):
            return presented

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid API key")
