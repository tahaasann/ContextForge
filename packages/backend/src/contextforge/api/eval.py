from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from contextforge.evaluation.pipeline import run_evaluation

router = APIRouter(prefix="/eval", tags=["evaluation"])


class EvalRequest(BaseModel):
    n_questions: int = 10


@router.post("/run")
async def run_eval(request: Request, body: EvalRequest):
    qdrant = request.app.state.qdrant
    openai_client = request.app.state.openai

    try:
        result = await run_evaluation(
            qdrant=qdrant,
            openai_client=openai_client,
            n_questions=body.n_questions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "metrics": {
            "faithfulness": round(result.faithfulness, 3),
            "answer_relevancy": round(result.answer_relevancy, 3),
            "context_precision": round(result.context_precision, 3),
        },
        "n_questions": result.n_questions,
        "interpretation": _interpret(result),
        "samples": [
            {
                "question": s["user_input"],
                "answer": s["response"][:200],
                "n_contexts": len(s["retrieved_contexts"]),
            }
            for s in result.samples
        ]
    }


def _interpret(result) -> str:
    avg = (
        result.faithfulness
        + result.answer_relevancy
        + result.context_precision
    ) / 3

    if avg >= 0.8:
        return "Pipeline is performing well. Answers are grounded and relevant."
    elif avg >= 0.6:
        return "Pipeline is acceptable. Consider tuning chunk size or top_k."
    else:
        return "Pipeline needs attention. Check chunking strategy and retrieval quality."