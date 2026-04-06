from dataclasses import dataclass

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from contextforge.core.config import settings
from contextforge.rag.generator import generate_answer
from contextforge.rag.retriever import retrieve
from contextforge.evaluation.synth import generate_test_questions
from contextforge.ingestion.embedder import DENSE_VECTOR_NAME


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    n_questions: int
    samples: list[dict]


async def run_evaluation(
    qdrant: AsyncQdrantClient,
    openai_client: AsyncOpenAI,
    n_questions: int = 10,
) -> EvalResult:
    # 1. Qdrant'tan rastgele chunk'lar çek — synth dataset için
    scroll_result = await qdrant.scroll(
        collection_name=settings.qdrant_collection,
        limit=50,              # 50 chunk'tan n_questions kadar seç
        with_payload=True,
        with_vectors=False,    # Vektörlere ihtiyacımız yok, sadece metin
    )

    chunks = [
        point.payload["text"]
        for point in scroll_result[0]
        if point.payload and "text" in point.payload
    ]

    if not chunks:
        raise ValueError("No documents found. Please ingest documents first.")

    # 2. Synth soru-cevap çiftleri üret
    qa_pairs = await generate_test_questions(
        openai_client=openai_client,
        chunks=chunks,
        n_questions=n_questions,
    )

    # 3. Her soru için pipeline'ı çalıştır
    samples = []
    for qa in qa_pairs:
        retrieved = await retrieve(
            query=qa["question"],
            client=qdrant,
            openai_client=openai_client,
        )

        result = await generate_answer(
            query=qa["question"],
            chunks=retrieved,
            openai_client=openai_client,
        )

        samples.append({
            "user_input": qa["question"],
            "response": result["answer"],
            "retrieved_contexts": [c.text for c in retrieved],
            "reference": qa["ground_truth"],
        })

    # 4. RAGAS ile değerlendir
    # RAGAS LangChain wrapper bekliyor — mevcut OpenAI client'ımızı wrap ediyoruz
    langchain_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    )
    langchain_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    )

    dataset = EvaluationDataset.from_list(samples)

    metrics = [
        Faithfulness(llm=langchain_llm),
        ResponseRelevancy(llm=langchain_llm, embeddings=langchain_embeddings),
        LLMContextPrecisionWithoutReference(llm=langchain_llm),
    ]

    eval_result = evaluate(dataset=dataset, metrics=metrics)
    scores = eval_result.to_pandas()

    # RAGAS versiyon farkları kolon adlarını değiştirebilir.
    # Güvenli okuma: kolon yoksa 0.0 döndür, log bas.
    def safe_mean(df, *possible_keys) -> float:
        for key in possible_keys:
            if key in df.columns:
                return float(df[key].mean())
        available = list(df.columns)
        print(f"[WARN] None of {possible_keys} found. Available: {available}")
        return 0.0

    return EvalResult(
        faithfulness=safe_mean(scores, "faithfulness"),
        answer_relevancy=safe_mean(
            scores,
            "answer_relevancy",       # RAGAS 0.4.x
            "response_relevancy",     # RAGAS 0.3.x
        ),
        context_precision=safe_mean(
            scores,
            "llm_context_precision_without_reference",  # 0.4.x
            "context_precision",                         # 0.3.x
        ),
        n_questions=len(samples),
        samples=samples,
    )