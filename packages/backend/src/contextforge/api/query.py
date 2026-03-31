from fastapi import APIRouter, Request
from pydantic import BaseModel

from contextforge.rag.graph import rag_graph

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("/ask")
async def ask(request: Request, body: QueryRequest):
    # Client'ları config üzerinden graph'a geçiriyoruz.
    # Bu LangGraph'ın "dependency injection" yöntemi.
    config = {
        "configurable": {
            "qdrant": request.app.state.qdrant,
            "openai": request.app.state.openai,
        }
    }

    # Graph'ı başlangıç state'iyle çalıştır.
    # Diğer key'ler node'lar tarafından doldurulacak.
    result = await rag_graph.ainvoke(
        {"question": body.question},
        config=config,
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "eval_score": result["eval_score"],
        "eval_reasoning": result["eval_reasoning"],
    }
