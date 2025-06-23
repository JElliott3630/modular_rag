from __future__ import annotations
from typing import Sequence
import logging
import asyncio
from openai import AsyncOpenAI
from src import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


class OpenAIEmbedding:
    def __init__(self, model: str | None = None, batch: int = 100) -> None:
        self._model = model or config.EMBED_MODEL
        self._batch = batch
        self._sem = asyncio.Semaphore(5)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        logger.info("embedding %s texts", len(texts))
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            async with self._sem:
                resp = await _client.embeddings.create(
                    model=self._model,
                    input=texts[i : i + self._batch],
                    encoding_format="float",
                )
            out.extend([d.embedding for d in resp.data])
        return out
