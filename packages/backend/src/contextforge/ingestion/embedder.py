import uuid

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models

from contextforge.core.config import settings
from contextforge.ingestion.chunker import Chunk

# Collection'da iki vector alanı olacak.
# Bu isimler Qdrant'a "hangi alanın ne olduğunu" söyler.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1536  # text-embedding-3-small boyutu


async def get_qdrant_client() -> AsyncQdrantClient:
    # Şimdilik in-memory. Faz sonunda tek bu satır değişecek:
    # AsyncQdrantClient(url=settings.qdrant_url)
    return AsyncQdrantClient(":memory:")


async def ensure_collection(client: AsyncQdrantClient) -> None:
    # Collection yoksa oluştur, varsa dokunma.
    # "ensure" prefix'i bu pattern'ı ifade eder — idempotent operasyon.
    exists = await client.collection_exists(settings.qdrant_collection)
    if exists:
        return

    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,  # anlam benzerliği için
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # BM25'in IDF ağırlıklandırması
            )
        },
    )


async def embed_chunks(chunks: list[Chunk]) -> list[models.PointStruct]:
    openai = AsyncOpenAI(api_key=settings.openai_api_key)

    # Tüm chunk metinlerini bir API çağrısında embed et.
    # Her chunk için ayrı çağrı yapmak hem yavaş hem pahalı.
    texts = [chunk.text for chunk in chunks]

    response = await openai.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    dense_vectors = [item.embedding for item in response.data]

    # Sparse vector için FastEmbed — Qdrant client'a built-in geliyor.
    # ONNX Runtime ile CPU'da çalışır, API çağrısı gerekmez.
    from fastembed.sparse.bm25 import Bm25

    bm25 = Bm25("Qdrant/bm25")
    sparse_vectors = list(bm25.query_embed(texts))

    points = []
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload={
                    "text": chunk.text,
                    "filename": chunk.source_filename,
                    "page_hint": chunk.page_hint,
                    "chunk_index": chunk.chunk_index,
                },
            )
        )

    return points


async def ingest_chunks(chunks: list[Chunk], client: AsyncQdrantClient) -> int:
    await ensure_collection(client)
    points = await embed_chunks(chunks)

    # upsert: varsa güncelle, yoksa ekle — idempotent.
    # Aynı dokümanı iki kez yüklersen duplicate olmaz.
    await client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    return len(points)
