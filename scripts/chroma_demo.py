from pathlib import Path
from core.ingestion import get_ingestor
from core.embeddings import OpenAIEmbedding
from core.vector_stores.chroma_store import ChromaVectorStore
from core.strategies.generation import OpenAICompletion
from core.strategies.rerank import SbertRerank
from core.orchestrator import RagOrchestrator

DOCS_DIR  = Path("documents")
NAMESPACE = "local-test"

QUESTIONS = [
    "What actions does Custom Care Pharmacy take when an LTC facility resident’s house-charge balance reaches 90 days past due?",
    "According to the OTC dosing table, what is the recommended single dose of ibuprofen for a 3-year-old child?",
    "List the three core values called out in the Operations Manager description.",
    "Which calendar year shows $14,737,436 in revenue for Custom Care Pharmacy, and what part of the business generated about $10 million of that total?",
    "What long-term gross-sales target does the Vision Script set for the company’s future?",
    "Name two vaccines Pennsylvania pharmacists may administer to children aged five years and older under the standing orders.",
    "Under the Value Drug Vendor Agreement, after how many days of non-payment may the wholesaler immediately terminate the contract for a payment default?",
    "For pre-exposure prophylaxis, how long before travel should a single dose of Typhim Vi be given, according to the standing order?",
]


def ingest_all(store):
    for fp in DOCS_DIR.iterdir():
        if not fp.is_file():
            continue
        data = fp.read_bytes()
        ingestor = get_ingestor(fp.name)
        batch = ingestor.ingest_bytes(data, fp.name)
        store.upsert(batch, NAMESPACE)
        print(f"ingested {len(batch)} chunks from {fp.name}")


def main():
    embed = OpenAIEmbedding()
    store = ChromaVectorStore(embed, path=".chroma-local")
    ingest_all(store)

    gen    = OpenAICompletion()
    rerank = SbertRerank()
    rag    = RagOrchestrator(store, gen, embed, rerank=rerank)

    for q in QUESTIONS:
        res = rag.answer(q, user_id=NAMESPACE, k=8, trace=True)
        print("\nQUESTION ▸", q)
        print("ANSWER   ▸", res["answer"].strip())
        print("CHUNKS   ▸", [f"{c['source']}#{c['index']}" for c in res["chunks"]])


if __name__ == "__main__":
    main()
