import json
import random

from openai import AsyncOpenAI

from contextforge.core.config import settings

async def generate_test_questions(
    openai_client: AsyncOpenAI,
    chunks: list[str],
    n_questions: int = 10,
) -> list[dict]:
    """
    Verilen chunk listesinden n adet soru-cevap çifti üretir.
    RAGAS evaluation için ground truth dataset oluşturur.

    Her eleman şöyle görünür:
    {
        "question": "...",
        "ground_truth": "...",
        "context": "..."   -> sorunun kaynağı olan chunk
    }
    """

    # Rastgele chunk seç - tüm dökümanı temsil etsin
    selected = random.sample(chunks, min(n_questions, len(chunks)))

    qa_pairs = []

    for chunk in selected:
        prompt = f"""Given this text passage, generate ONE specific question that:
1. Can be answered directly from the passage
2. Requires understanding, not just keyword matching
3. Is in Turkish if the passage is in Turkish

Then provide the answer based solely on the passage.

Passage:
{chunk}

Respond ONLY with a JSON object:
{{"question": "...", "answer": "..."}}"""

        response = await openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        qa_pairs.append({
            "question": result["question"],
            "ground_truth": result["answer"],
            "context": chunk,
        })

    return qa_pairs