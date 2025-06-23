from core.ingestion.base import AbstractIngestor, IngestParams

class TextIngestor(AbstractIngestor):
    def _convert_to_markdown(self, data: bytes) -> str:
        return data.decode()

def test_chunking():
    txt = b"abcdefghijklmnopqrstuvwxyz"
    params = IngestParams(chunk_size=10, overlap=3)
    ingestor = TextIngestor(params)
    batch = ingestor.ingest_bytes(txt, "dummy.txt")
    assert len(batch) == 4
    ids = [c.id for c in batch]
    assert len(set(ids)) == 4
