from __future__ import annotations
from typing import Sequence
import logging
from dataclasses import asdict
from core.schema import Chunk
from core.vector_stores.base import BaseVectorStore, EmbeddingModel
from core.strategies.expansion import PromptExpansion
from core.strategies.rerank import SbertRerank
from core.strategies.generation import OpenAICompletion

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RagOrchestrator:
    def __init__(
        self,
        store: BaseVectorStore,
        generation: OpenAICompletion,
        embedding: EmbeddingModel,
        expansion: PromptExpansion | None = None,
        rerank: SbertRerank | None = None,
    ) -> None:
        self._store = store
        self._gen = generation
        self._embed = embedding
        self._exp = expansion
        self._rerank = rerank

    def answer(
        self,
        query: str,
        user_id: str,
        k: int = 6,
        trace: bool = False,
    ):
        logger.info("orchestrating answer")
        queries: Sequence[str] = [query]
        if self._exp:
            queries = [query, *self._exp.run(query)]

        retrieved: list[Chunk] = []
        for q in queries:
            retrieved.extend(self._store.query(q, namespace=user_id, k=k))

        dedup = {c.id: c for c in retrieved}.values()
        candidates = list(dedup)

        if self._rerank:
            top = self._rerank.run(query, candidates, top_n=k)
        else:
            top = candidates[:k]

        context = [c.text for c in top]
        answer = self._gen.run(query, context)

        if trace:
            return {"answer": answer, "chunks": [asdict(c) for c in top]}
        return answer
