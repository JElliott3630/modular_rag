from __future__ import annotations
from typing import Iterable, Sequence, Any
import logging
from pinecone import Pinecone, ServerlessSpec
from core.vector_stores.base import BaseVectorStore, EmbeddingModel
from core.schema import Chunk, DocumentBatch
from src import config  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PineconeVectorStore(BaseVectorStore):
    _MAX_BATCH = 80

    def __init__(self, embedding: EmbeddingModel) -> None:
        super().__init__(embedding)
        self._pc = Pinecone(api_key=config.PINECONE_API_KEY)  # pyright: ignore
        if config.PINECONE_INDEX not in self._pc.list_indexes().names():
            self._pc.create_index(
                name=config.PINECONE_INDEX,
                dimension=config.EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2"),
            )
        self._index: Any = self._pc.Index(config.PINECONE_INDEX)

    def upsert(self, batch: DocumentBatch, namespace: str) -> None:
        logger.info("upserting %s chunks to %s", len(batch.chunks), namespace)
        vectors = self._build_vectors(batch.chunks)
        if len(vectors) >= self._MAX_BATCH:
            self._batched_upsert(vectors, namespace)
        else:
            self._index.upsert(vectors=vectors, namespace=namespace)

    def query(self, query_text: str, namespace: str, k: int = 6) -> list[Chunk]:
        emb = self._embedding.embed_texts([query_text])[0]
        res = self._index.query(vector=emb, top_k=k, namespace=namespace, include_metadata=True)
        return [
            Chunk(
                id=str(m["id"]),
                text=m["metadata"]["text"],
                index=m["metadata"]["index"],
                source=m["metadata"]["source"],
            )
            for m in res["matches"]
        ]

    def delete(self, ids: Iterable[str], namespace: str) -> None:
        self._index.delete(ids=list(ids), namespace=namespace)

    def _build_vectors(self, chunks: Sequence[Chunk]) -> list[tuple[str, Sequence[float], dict]]:
        texts = [c.text for c in chunks]
        embeds = self._embedding.embed_texts(texts)
        return [
            (
                chunk.id,
                embeds[i],
                {"text": chunk.text, "index": chunk.index, "source": chunk.source},
            )
            for i, chunk in enumerate(chunks)
        ]

    def _batched_upsert(
        self,
        vectors: Sequence[tuple[str, Sequence[float], dict]],
        namespace: str,
    ) -> None:
        for i in range(0, len(vectors), self._MAX_BATCH):
            self._index.upsert(vectors=vectors[i : i + self._MAX_BATCH], namespace=namespace)
