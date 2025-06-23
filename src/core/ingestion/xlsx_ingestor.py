from io import BytesIO
import pandas as pd
from core.ingestion.base import AbstractIngestor


class XlsxIngestor(AbstractIngestor):
    def _convert_to_markdown(self, data: bytes) -> str:
        dfs = pd.read_excel(BytesIO(data), sheet_name=None, header=None)
        lines: list[str] = []
        for name, df in dfs.items():
            lines.append(f"# {name}")
            for row in df.itertuples(index=False):
                cells = [str(cell) for cell in row if str(cell).strip()]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)
