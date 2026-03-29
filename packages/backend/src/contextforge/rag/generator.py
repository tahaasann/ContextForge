from openai import AsyncOpenAI

from contextforge.core.config import settings
from contextforge.rag.retriever import RetrievedChunk

# System prompt: modelin rolünü ve kısıtlarını tanımlar.
# "Sadece verilen bağlamı kullan" — hallucination'ı engeller.
_SYSTEM_PROMPT = """You are a precise document assistant.
Answer questions using ONLY the provided context chunks.
If the answer is not in the context, say "I could not find this in the provided documents."
Always cite which file and page your answer comes from."""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    # Her chunk'ı kaynak bilgisiyle birlikte formatla.
    # Model bu formatı görünce kaynağa atıf yapmayı öğreniyor.
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk.filename}, page {chunk.page_hint}\n" f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> dict:
    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
        }

    context = _build_context(chunks)
    openai = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await openai.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0,  # RAG'da deterministik cevap istiyoruz — 0 en iyi
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    # Kaynakları deduplicate et — aynı sayfa birden fazla chunk'tan gelebilir
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk.filename, chunk.page_hint)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "filename": chunk.filename,
                    "page": chunk.page_hint,
                    "score": round(chunk.score, 4),
                }
            )

    return {"answer": answer, "sources": sources}
