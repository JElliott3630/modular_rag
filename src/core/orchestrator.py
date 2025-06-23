from __future__ import annotations
import asyncio
import functools
import inspect
import logging
from dataclasses import asdict
from typing import Sequence

from core.schema import Chunk
from core.vector_stores.base import BaseVectorStore
from core.strategies.expansion import PromptExpansion
from core.strategies.generation import OpenAICompletion
from core.strategies.rerank import SbertRerank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RagOrchestrator:
    def __init__(
        self,
        store: BaseVectorStore,
        generation: OpenAICompletion,
        embedding,  # kept for future hooks
        expansion: PromptExpansion | None = None,
        rerank: SbertRerank | None = None,
    ) -> None:
        self._store = store
        self._gen = generation
        self._exp = expansion
        self._rerank = rerank or SbertRerank()

    # ── Async public API ───────────────────────────────────────────
    async def answer_async(
        self,
        query: str,
        user_id: str,
        k: int = 8,
        trace: bool = False,
    ):
        # 1. Expansion (await OpenAI)
        queries: list[str] = [query]
        if self._exp:
            queries += await self._exp.run(query)

        # 2. Parallel retrieval
        loop = asyncio.get_running_loop()
        tasks = [
            self._store.query_async(q, namespace=user_id, k=k)  # type: ignore[attr-defined]
            if hasattr(self._store, "query_async")
            else loop.run_in_executor(None, functools.partial(self._store.query, q, user_id, k))
            for q in queries
        ]
        results: Sequence[list[Chunk]] = await asyncio.gather(*tasks)
        dedup: dict[str, Chunk] = {c.id: c for sub in results for c in sub}

        # 3. Rerank (CPU) in background thread
        top = await loop.run_in_executor(
            None, functools.partial(self._rerank.run, query, list(dedup.values()), k)
        )

        # 4. Generation (await if coroutine; else off-thread)
        if inspect.iscoroutinefunction(self._gen.run):
            answer = await self._gen.run(query, [c.text for c in top])
        else:
            answer = await loop.run_in_executor(
                None, lambda: self._gen.run(query, [c.text for c in top])
            )

        return {"answer": answer, "chunks": [asdict(c) for c in top]} if trace else answer

    # ── Sync wrapper for legacy scripts / tests ────────────────────
    def answer(self, *args, **kwargs):
        return asyncio.run(self.answer_async(*args, **kwargs))
