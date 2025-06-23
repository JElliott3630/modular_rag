from __future__ import annotations
import logging
from functools import lru_cache
from typing import Callable
import httpx
import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from src import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AuthService:
    def __init__(self) -> None:
        self._jwks_client = PyJWKClient(self._jwks_url())

    def verify_jwt(self, token: str) -> dict:
        signing_key = self._jwks_client.get_signing_key_from_jwt(token).key
        try:
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=config.SUPABASE_URL,
                options={"verify_at_hash": False},
            )
            return claims
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    def fastapi_dependency(self) -> Callable[..., dict]:
        def _dep(token: str = Depends(self._bearer_header)) -> dict:  # type: ignore
            return self.verify_jwt(token)

        return _dep

    @staticmethod
    @lru_cache
    def _jwks_url() -> str:
        return f"{config.SUPABASE_URL}/auth/v1/keys"

    @staticmethod
    def _bearer_header(authorization: str | None = None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer")
        return authorization.split(" ", 1)[1]
