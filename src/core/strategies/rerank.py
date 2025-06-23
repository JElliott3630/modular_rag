from __future__ import annotations
from typing import Sequence
import logging
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from core.schema import Chunk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SbertRerank:
    """
    Prefer a cross-encoder for precise reranking.
    If unavailable (no download, offline, etc.), fall back to
    embedding cosine similarity via SentenceTransformer util.dot_score.
    """

    _CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._mode = "none"

        # 1️⃣ try cross-encoder
        try:
            self._xe = CrossEncoder(self._CROSS_MODEL)
            self._mode = "cross"
            logger.info("loaded cross-encoder %s", self._CROSS_MODEL)
            return
        except Exception as exc:
            logger.warning("cross-encoder unavailable (%s)", exc)

        # 2️⃣ fall back to bi-encoder similarity
        try:
            self._be = SentenceTransformer(self._EMBED_MODEL)
            self._mode = "bi"
            logger.info("loaded bi-encoder %s for cosine rerank", self._EMBED_MODEL)
        except Exception as exc:
            logger.warning("bi-encoder unavailable (%s) – rerank disabled", exc)

    # ------------------------------------------------------------------
    def run(self, query: str, chunks: Sequence[Chunk], top_n: int = 6) -> list[Chunk]:
        if self._mode == "none" or len(chunks) <= 1:
            return list(chunks)[:top_n]

        if self._mode == "cross":
            pairs = [[query, c.text] for c in chunks]
            scores = self._xe.predict(pairs)
        else:  # bi-encoder
            q_emb = self._be.encode([query], convert_to_tensor=True)
            c_emb = self._be.encode([c.text for c in chunks], convert_to_tensor=True)
            scores = util.dot_score(q_emb, c_emb)[0].cpu().numpy()

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [c for c, _ in ranked]
