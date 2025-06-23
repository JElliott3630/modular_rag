from __future__ import annotations
import logging
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from core.embeddings import OpenAIEmbedding
from core.vector_stores.pinecone_store import PineconeVectorStore
from core.strategies.expansion import PromptExpansion
from core.strategies.rerank import SbertRerank
from core.strategies.generation import OpenAICompletion
from core.orchestrator import RagOrchestrator
from integrations.supabase.auth import AuthService

logging.basicConfig(level=logging.INFO)
auth_service = AuthService()

embedding = OpenAIEmbedding()
store = PineconeVectorStore(embedding)
expansion = PromptExpansion()
rerank = SbertRerank()
generation = OpenAICompletion()
rag = RagOrchestrator(store, generation, embedding, expansion, rerank)

app = FastAPI()

class AskRequest(BaseModel):
    query: str
    k: int | None = 6
    trace: bool | None = False

@app.post("/rag/ask")
def ask(
    body: AskRequest,
    claims: dict = Depends(auth_service.fastapi_dependency()),
):
    user_id = claims["sub"]
    answer = rag.answer(
        query=body.query,
        user_id=user_id,
        k=body.k or 6,
        trace=body.trace or False,
    )
    return JSONResponse(answer if isinstance(answer, dict) else {"answer": answer})

@app.get("/health")
def health():
    return {"status": "ok"}
