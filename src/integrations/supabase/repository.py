from __future__ import annotations
import logging
from typing import Any, Sequence
import httpx
from src import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SupabaseRepository:
    def __init__(self) -> None:
        self._url = f"{config.SUPABASE_URL}/rest/v1"
        self._key = config.SUPABASE_SERVICE_ROLE_KEY
        self._headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

    def get(self, table: str, filters: str = "", limit: int | None = None) -> list[dict]:
        logger.info("supabase get %s", table)
        params: dict[str, Any] = {"select": "*"}
        if filters:
            params.update(self._parse_filters(filters))
        if limit:
            params["limit"] = str(limit)
        r = httpx.get(f"{self._url}/{table}", headers=self._headers, params=params, timeout=8)
        r.raise_for_status()
        return r.json()

    def insert(self, table: str, rows: Sequence[dict]) -> None:
        logger.info("supabase insert %s rows into %s", len(rows), table)
        r = httpx.post(f"{self._url}/{table}", headers=self._headers, json=rows, timeout=8)
        r.raise_for_status()

    def update(self, table: str, patch: dict, eq: dict[str, str]) -> None:
        logger.info("supabase update %s", table)
        params = {f"eq.{k}": v for k, v in eq.items()}
        r = httpx.patch(
            f"{self._url}/{table}", headers=self._headers, params=params, json=patch, timeout=8
        )
        r.raise_for_status()

    @staticmethod
    def _parse_filters(expr: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for part in expr.split(","):
            k, v = part.split("=", 1)
            out[f"eq.{k.strip()}"] = v.strip()
        return out
