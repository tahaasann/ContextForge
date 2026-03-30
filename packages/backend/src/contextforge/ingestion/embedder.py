import hashlib

from fastembed.sparse.bm25 import Bm25
from qdrant_client import AsyncQdrantClient, models

from contextforge.core.config import settings
from contextforge.ingestion.chunker import Chunk

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1536

_bm25 = Bm25("Qdrant/bm25")


def _chunk_id(filename: str, chunk_index: int) -> str:
    # Deterministik ID: aynı dosyanın aynı chunk'ı → her zaman aynı UUID formatı
    # sha256'nın ilk 32 hex karakteri → UUID formatına çevir
    raw = f"{filename}::{chunk_index}"
    h = hashlib.sha256(raw.encode()).hexdigest()
    # UUID formatı: 8-4-4-4-12
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


async def ensure_collection(client: AsyncQdrantClient) -> None:
    exists = await client.collection_exists(settings.qdrant_collection)
    if exists:
        return

    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


async def embed_and_store(
    chunks: list[Chunk],
    client: AsyncQdrantClient,
    openai_client,  # dışarıdan inject — artık burada yaratmıyoruz
) -> int:
    await ensure_collection(client)

    texts = [chunk.text for chunk in chunks]

    # Dense embedding — OpenAI
    response = await openai_client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    dense_vectors = [item.embedding for item in response.data]

    # Sparse embedding — BM25
    sparse_results = list(_bm25.passage_embed(texts))

    points = []
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_results):
        points.append(
            models.PointStruct(
                id=_chunk_id(chunk.source_filename, chunk.chunk_index),
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

    await client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    return len(points)
