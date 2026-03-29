from dataclasses import dataclass

from fastembed.sparse.bm25 import Bm25
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models

from contextforge.core.config import settings
from contextforge.ingestion.embedder import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

# BM25 modeli modül yüklenince bir kez init edilir.
# Her sorguda yeniden yüklemek gereksiz maliyet.
_bm25 = Bm25("Qdrant/bm25")


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page_hint: int
    score: float  # RRF fusion skoru - ne kadar yüksek, o kadar alakalı


async def retrieve(
    query: str,
    client: AsyncQdrantClient,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    openai = AsyncOpenAI(api_key=settings.openai_api_key)

    # Dense Vector:
    embed_response = await openai.embeddings.create(
        model=settings.embedding_model,
        input=[query],
    )
    dense_vector = embed_response.data[0].embedding

    # Sparse vector: BM25 ile keyword vektörü üret
    sparse_result = list(_bm25.query_embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    # Hybrid search: iki vektörü Qdrant'a ver, RRF fusion dönsün.
    # prefetch: önce her kanaldan top_k*2 çek, sonra fuse et, sonra top_k al.
    # Neden top_k*2? Fusion sonrası sıralama değişebilir —
    # daha geniş havuzdan fuse etmek kaliteyi artırır.
    results = await client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=top_k * 2,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                filename=payload.get("filename", ""),
                page_hint=payload.get("page_hint", 0),
                score=point.score,
            )
        )

    return chunks
