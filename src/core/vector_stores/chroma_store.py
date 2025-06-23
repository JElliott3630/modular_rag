from __future__ import annotations
from typing import Iterable, Sequence
import logging
import chromadb
from core.vector_stores.base import BaseVectorStore, EmbeddingModel
from core.schema import Chunk, DocumentBatch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, embedding: EmbeddingModel, path: str = ".chroma") -> None:
        super().__init__(embedding)
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection("rag")

    # ──────────────────────────────────────────────────────────────────────────
    # ingest
    # ──────────────────────────────────────────────────────────────────────────
    def upsert(self, batch: DocumentBatch, namespace: str) -> None:
        logger.info("upserting %s chunks to %s", len(batch.chunks), namespace)

        ids          = [c.id for c in batch.chunks]
        present_ids  = set(self._existing_ids(ids))
        new_chunks   = [c for c in batch.chunks if c.id not in present_ids]

        if not new_chunks:
            logger.info("all chunks already exist, skip")
            return

        texts   = [c.text for c in new_chunks]
        embeds  = self._embedding.embed_texts(texts)
        metas   = [
            {"source": c.source, "index": c.index, "namespace": namespace}
            for c in new_chunks
        ]

        self._collection.upsert(
            ids=[c.id for c in new_chunks],
            embeddings=embeds,
            metadatas=metas,
            documents=texts,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # query
    # ──────────────────────────────────────────────────────────────────────────
    def query(self, query_text: str, namespace: str, k: int = 6) -> list[Chunk]:
        emb = self._embedding.embed_texts([query_text])[0]
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

    # ──────────────────────────────────────────────────────────────────────────
    # delete
    # ──────────────────────────────────────────────────────────────────────────
    def delete(self, ids: Iterable[str], namespace: str) -> None:
        self._collection.delete(ids=list(ids))

    # ──────────────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _existing_ids(self, ids: list[str]) -> list[str]:
        existing: list[str] = []
        for i in range(0, len(ids), 100):
            res = self._collection.get(ids=ids[i : i + 100], include=[])
            existing.extend(res["ids"])
        return existing
