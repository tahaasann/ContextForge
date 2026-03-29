from fastapi import APIRouter, Request
from pydantic import BaseModel

from contextforge.rag.generator import generate_answer
from contextforge.rag.retriever import retrieve

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("/ask")
async def ask(request: Request, body: QueryRequest):
    client = request.app.state.qdrant

    chunks = await retrieve(
        query=body.question,
        client=client,
        top_k=body.top_k,
    )

    result = await generate_answer(
        query=body.question,
        chunks=chunks,
    )

    return result
