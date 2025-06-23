from pathlib import Path
from typing import Mapping
from core.ingestion.doc_ingestor import DocIngestor
from core.ingestion.pdf_ingestor import PdfIngestor
from core.ingestion.xlsx_ingestor import XlsxIngestor
from core.ingestion.base import AbstractIngestor, IngestParams

_EXT_TO_CLASS: Mapping[str, type[AbstractIngestor]] = {
    ".docx": DocIngestor,
    ".doc": DocIngestor,
    ".pdf": PdfIngestor,
    ".xlsx": XlsxIngestor,
}


def get_ingestor(filename: str, params: IngestParams | None = None) -> AbstractIngestor:
    ext = Path(filename).suffix.lower()
    if ext not in _EXT_TO_CLASS:  # explicit to surface unsupported types loudly
        raise ValueError(f"no ingestor for extension {ext}")
    return _EXT_TO_CLASS[ext](params)
