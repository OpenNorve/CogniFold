"""API key authentication for the Cognifold service layer."""

from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyValidator:
    """FastAPI dependency for API key validation.

    If ``valid_keys`` is ``None``, authentication is disabled (dev mode).
    """

    def __init__(self, valid_keys: set[str] | None = None) -> None:
        self._valid_keys = valid_keys

    async def __call__(self, api_key: str | None = Security(_api_key_header)) -> str | None:
        # Auth disabled
        if self._valid_keys is None:
            return None

        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key",
            )

        if api_key not in self._valid_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        return api_key
