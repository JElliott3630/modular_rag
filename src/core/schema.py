from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Chunk:
    id: str
    text: str
    index: int
    source: str


@dataclass(slots=True)
class DocumentBatch:
    source: str
    chunks: list[Chunk]

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)
