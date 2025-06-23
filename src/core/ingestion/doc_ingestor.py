from io import BytesIO
from docx import Document
from core.ingestion.base import AbstractIngestor


class DocIngestor(AbstractIngestor):
    def _convert_to_markdown(self, data: bytes) -> str:
        doc = Document(BytesIO(data))
        parts: list[str] = []
        for p in doc.paragraphs:
            if p.text.strip():
                parts.append(p.text.strip())
        return "\n\n".join(parts)
