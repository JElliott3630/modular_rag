#!/usr/bin/env python
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
from core.ingestion import get_ingestor
from core.embeddings import OpenAIEmbedding
from core.vector_stores.pinecone_store import PineconeVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()
    embedding = OpenAIEmbedding()
    store = PineconeVectorStore(embedding)

    for file_path in args.files:
        data = Path(file_path).read_bytes()
        ingestor = get_ingestor(file_path)
        batch = ingestor.ingest_bytes(data, Path(file_path).name)
        store.upsert(batch, namespace=args.namespace)
