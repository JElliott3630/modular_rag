from typing import Sequence
from core.vector_stores.base import BaseVectorStore, EmbeddingModel
from core.schema import Chunk, DocumentBatch

class DummyEmbed(EmbeddingModel):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(len(t))] * 3 for t in texts]

class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, embedding: EmbeddingModel):
        super().__init__(embedding)
        self._db: dict[str, dict[str, Chunk]] = {}

    def upsert(self, batch: DocumentBatch, namespace: str) -> None:
        ns = self._db.setdefault(namespace, {})
        for c in batch:
            ns[c.id] = c

    def query(self, query_text: str, namespace: str, k: int):
        ns = self._db.get(namespace, {})
        return list(ns.values())[:k]

    def delete(self, ids, namespace: str):
        ns = self._db.get(namespace, {})
        for i in ids:
            ns.pop(i, None)

def test_upsert_query_delete():
    emb = DummyEmbed()
    store = InMemoryVectorStore(emb)
    c1 = Chunk(id="1", text="hello", index=0, source="s")
    batch = DocumentBatch(source="s", chunks=[c1])
    store.upsert(batch, "u")
    out = store.query("x", "u", 1)
    assert out[0].text == "hello"
    store.delete(["1"], "u")
    assert store.query("x", "u", 1) == []
