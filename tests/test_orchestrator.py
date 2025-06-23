from typing import Sequence
from core.schema import Chunk, DocumentBatch
from core.vector_stores.base import BaseVectorStore, EmbeddingModel
from core.orchestrator import RagOrchestrator

class DummyEmbed(EmbeddingModel):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0] * 3 for _ in texts]

class MemoryStore(BaseVectorStore):
    def __init__(self, embedding: EmbeddingModel):
        super().__init__(embedding)
        self._chunks: list[Chunk] = [
            Chunk(id="c", text="ctx", index=0, source="s")
        ]

    def upsert(self, batch: DocumentBatch, namespace: str): ...

    def query(self, query_text: str, namespace: str, k: int):
        return self._chunks[:k]

    def delete(self, ids, namespace: str): ...

class EchoGen:
    def run(self, query: str, context: Sequence[str]) -> str:
        return f"answer:{query}:{context[0]}"

def test_orchestrator_answer():
    emb = DummyEmbed()
    store = MemoryStore(emb)
    gen = EchoGen()
    rag = RagOrchestrator(store, gen, emb)
    out = rag.answer("q", user_id="u", k=1, trace=False)
    assert out.startswith("answer:q:")
