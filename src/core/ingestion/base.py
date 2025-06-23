from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import logging
import tiktoken
from core.schema import Chunk, DocumentBatch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True, slots=True)
class IngestParams:
    chunk_size: int = 500
    overlap: int = 75


class AbstractIngestor(ABC):
    params: IngestParams

    def __init__(self, params: IngestParams | None = None) -> None:
        self.params = params or IngestParams()

    def ingest_bytes(self, data: bytes, filename: str) -> DocumentBatch:
        logger.info("ingesting %s", filename)
        markdown = self._convert_to_markdown(data)
        chunks = list(self._chunk(markdown, filename))
        return DocumentBatch(source=filename, chunks=chunks)

    @abstractmethod
    def _convert_to_markdown(self, data: bytes) -> str: ...

    # ───────────────────────────────────────────────────────────────
    # deterministic id: <file_stem>_<md5(first-12) >
    # ───────────────────────────────────────────────────────────────
    def _chunk(self, text: str, source: str) -> Iterable[Chunk]:
        tokens = _TOKENIZER.encode(text)
        size, overlap = self.params.chunk_size, self.params.overlap
        for i in range(0, len(tokens), size - overlap):
            window = tokens[i : i + size]
            chunk_text = _TOKENIZER.decode(window)
            digest = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:12]
            yield Chunk(
                id=f"{Path(source).stem}_{digest}",
                text=chunk_text,
                index=i // (size - overlap),
                source=source,
            )
