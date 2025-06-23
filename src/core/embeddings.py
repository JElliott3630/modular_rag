from __future__ import annotations
from typing import Sequence
import logging
import openai
from src import config

openai.api_key = config.OPENAI_API_KEY
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenAIEmbedding:
    def __init__(self, model: str | None = None, batch: int = 100) -> None:
        self._model = model or config.EMBED_MODEL
        self._batch = batch

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        logger.info("embedding %s texts", len(texts))
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            resp = openai.embeddings.create(
                model=self._model,
                input=texts[i : i + self._batch],
                encoding_format="float",
            )
            out.extend([d.embedding for d in resp.data])
        return out
