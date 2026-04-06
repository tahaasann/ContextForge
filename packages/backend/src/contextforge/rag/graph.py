from typing_extensions import TypedDict

from contextforge.rag.retriever import RetrievedChunk

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from contextforge.rag.generator import generate_answer
from contextforge.rag.retriever import retrieve

from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig


class AgentState(TypedDict):
    # Kullanıcının sorusu — hiç değişmez, tüm node'lar okur
    question: str

    # RAG Agent'ın bulduğu chunk'lar
    retrieved_chunks: list[RetrievedChunk]

    # RAG Agent'ın ürettiği ham cevap
    answer: str

    # Kaynak listesi
    sources: list[dict]

    # Eval Agent'ın vereceği skor (0.0 - 1.0)
    # None = henüz değerlendirilmedi
    eval_score: float | None

    # Eval Agent'ın yorumu
    eval_reasoning: str


async def rag_node(state: AgentState, config: RunnableConfig) -> dict:
    # config["configurable"] içinde runtime'da geçilen
    # Qdrant ve OpenAI client'ları var.
    # Her node bu pattern'ı kullanarak dış bağımlılıklara erişir.
    configurable = config.get("configurable", {})
    qdrant: AsyncQdrantClient = configurable["qdrant"]
    openai: AsyncOpenAI = configurable["openai"]

    # Mevcut retriever ve generator'ımızı kullanıyoruz.
    # Graph bunları sıfırdan yazmıyor — mevcut katmanların üstüne oturuyor.
    chunks = await retrieve(
        query=state["question"],
        client=qdrant,
        openai_client=openai,
    )

    result = await generate_answer(
        query=state["question"],
        chunks=chunks,
        openai_client=openai,
    )

    # Node sadece değiştirdiği key'leri döndürür.
    # LangGraph bunu mevcut state ile merge eder.
    return {
        "retrieved_chunks": chunks,
        "answer": result["answer"],
        "sources": result["sources"],
    }


async def eval_node(state: AgentState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    openai: AsyncOpenAI = configurable["openai"]

    # Cevabın kalitesini değerlendirmek için LLM'e sor.
    # Bu "LLM as a judge" pattern — production'da yaygın kullanım.
    eval_prompt = f"""You are an evaluation expert. Score the following answer.

Question: {state["question"]}

Answer: {state["answer"]}

Retrieved context chunks: {len(state["retrieved_chunks"])} chunks found.

Evaluate on these criteria:
1. Is the answer grounded in the retrieved context? (not hallucinated)
2. Does it directly address the question?
3. Are sources cited?

Respond with ONLY a JSON object, nothing else:
{{"score": 0.0-1.0, "reasoning": "one sentence explanation"}}"""

    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    import json

    result = json.loads(response.choices[0].message.content)

    return {
        "eval_score": result["score"],
        "eval_reasoning": result["reasoning"],
    }


def build_rag_graph():
    graph = StateGraph(AgentState)

    graph.add_node("rag", rag_node)
    graph.add_node("eval", eval_node)

    # Yol: RAG biter → Eval başlar → Biter
    graph.add_edge(START, "rag")
    graph.add_edge("rag", "eval")
    graph.add_edge("eval", END)

    

    return graph.compile()


# Modül yüklenince bir kez compile edilir.
# Her request yeni bir state ile bu grafiği çalıştırır.
rag_graph = build_rag_graph()
