from qdrant_client import AsyncQdrantClient

from contextforge.ingestion.chunker import chunk_document
from contextforge.ingestion.embedder import ingest_chunks
from contextforge.ingestion.parser import parse_pdf


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    client: AsyncQdrantClient,
) -> dict:
    # Adım 1: Parse
    doc = parse_pdf(file_bytes, filename)

    # Adım 2: Chunk
    chunks = chunk_document(doc)

    # Adım 3: Embed + Store
    stored = await ingest_chunks(chunks, client)

    return {
        "filename": filename,
        "total_pages": doc.total_pages,
        "total_chunks": stored,
    }
