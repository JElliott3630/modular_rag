from io import BytesIO
import pdfplumber
from core.ingestion.base import AbstractIngestor


class PdfIngestor(AbstractIngestor):
    def _convert_to_markdown(self, data: bytes) -> str:
        contents: list[str] = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    contents.append(text.strip())
        return "\n\n".join(contents)
