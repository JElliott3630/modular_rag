from __future__ import annotations
import hmac
import hashlib
import logging
import json
from typing import Iterable
import httpx
from src import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DropboxWebhookHandler:
    def __init__(self) -> None:
        self._secret = config.DROPBOX_APP_SECRET

    def verify(self, signature_header: str, body: bytes) -> None:
        sig = hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, signature_header):
            raise ValueError("invalid dropbox signature")

    def parse_accounts(self, body: bytes) -> list[str]:
        data = json.loads(body)
        return data.get("list_folder", {}).get("accounts", [])

    def fetch_delta(self, access_token: str, cursor: str | None = None) -> Iterable[dict]:
        url = "https://api.dropboxapi.com/2/files/list_folder/continue" if cursor else \
              "https://api.dropboxapi.com/2/files/list_folder"
        payload = {"cursor": cursor} if cursor else {"path": "", "recursive": True, "include_deleted": False}
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        while True:
            r = httpx.post(url, headers=headers, json=payload, timeout=15)
            r.raise_for_status()
            out = r.json()
            for entry in out["entries"]:
                yield entry
            if not out.get("has_more"):
                break
            payload = {"cursor": out["cursor"]}
            url = "https://api.dropboxapi.com/2/files/list_folder/continue"
