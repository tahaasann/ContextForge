from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from contextforge.ingestion.chunker import chunk_document
from contextforge.ingestion.embedder import embed_and_store
from contextforge.ingestion.parser import parse_pdf


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    client: AsyncQdrantClient,
    openai_client: AsyncOpenAI,
) -> dict:
    doc = parse_pdf(file_bytes, filename)
    chunks = chunk_document(doc)
    stored = await embed_and_store(chunks, client, openai_client)

    return {
        "filename": filename,
        "total_pages": doc.total_pages,
        "total_chunks": stored,
    }
