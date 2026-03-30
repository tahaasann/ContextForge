from contextforge.rag.retriever import RetrievedChunk

_SYSTEM_PROMPT = """You are a precise document assistant.
Answer questions using ONLY the provided context chunks.
If the answer is not in the context, say "I could not find this in the provided documents."
Always cite which file and page your answer comes from."""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk.filename}, page {chunk.page_hint}\n" f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    openai_client,
) -> dict:
    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    context = _build_context(chunks)

    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

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
