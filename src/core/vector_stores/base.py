from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Protocol, Sequence
import logging
from core.schema import Chunk, DocumentBatch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbeddingModel(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class BaseVectorStore(ABC):
    def __init__(self, embedding: EmbeddingModel) -> None:
        self._embedding = embedding

    @abstractmethod
    def upsert(self, batch: DocumentBatch, namespace: str) -> None: ...

    @abstractmethod
    def query(self, query_text: str, namespace: str, k: int) -> list[Chunk]: ...

    @abstractmethod
    def delete(self, ids: Iterable[str], namespace: str) -> None: ...
