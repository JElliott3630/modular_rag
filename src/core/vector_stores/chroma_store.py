from __future__ import annotations
from typing import Iterable, Sequence
import logging
import chromadb
import asyncio
from core.vector_stores.base import BaseVectorStore, EmbeddingModel
from core.schema import Chunk, DocumentBatch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, embedding: EmbeddingModel, path: str = ".chroma") -> None:
        super().__init__(embedding)
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection("rag")

    # ── Ingest (sync – runs offline) ───────────────────────────────
    def upsert(self, batch: DocumentBatch, namespace: str) -> None:
        ids = [c.id for c in batch.chunks]
        present = set(self._existing_ids(ids))
        new_chunks = [c for c in batch.chunks if c.id not in present]
        if not new_chunks:
            return

        embeds = asyncio.run(self._embedding.embed_texts([c.text for c in new_chunks]))
        metas = [
            {"source": c.source, "index": c.index, "namespace": namespace}
            for c in new_chunks
        ]
        self._collection.upsert(
            ids=[c.id for c in new_chunks],
            embeddings=embeds,
            metadatas=metas,
            documents=[c.text for c in new_chunks],
        )

    # ── Retrieval (async wrapper) ──────────────────────────────────
    def query(self, query_text: str, namespace: str, k: int = 6) -> list[Chunk]:
        emb = asyncio.run(self._embedding.embed_texts([query_text]))[0]
        res = self._collection.query(query_embeddings=[emb], n_results=k)
        out: list[Chunk] = []
        for i, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            out.append(
                Chunk(
                    id=cid,
                    text=res["documents"][0][i],
                    index=meta["index"],
                    source=meta["source"],
                )
            )
        return out

    async def query_async(self, query_text: str, namespace: str, k: int = 6) -> list[Chunk]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.query(query_text, namespace, k)
        )

    # ── Helpers ────────────────────────────────────────────────────
    def _existing_ids(self, ids: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(ids), 100):
            res = self._collection.get(ids=ids[i : i + 100], include=[])
            out.extend(res["ids"])
        return out

    def delete(self, ids: Iterable[str], namespace: str) -> None:
        self._collection.delete(ids=list(ids))
