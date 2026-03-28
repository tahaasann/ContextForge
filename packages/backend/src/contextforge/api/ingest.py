from fastapi import APIRouter, HTTPException, Request, UploadFile

from contextforge.ingestion.embedder import get_qdrant_client
from contextforge.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Desteklenen dosya tipleri — şimdilik sadece PDF.
# Faz 2'de markdown ve txt ekleyeceğiz.
ALLOWED_TYPES = {"application/pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/upload")
async def upload_document(request: Request, file: UploadFile):
    # request.app.state.qdrant → lifespan'de oluşturduğumuz tek client
    # Her istek aynı client'ı, dolayısıyla aynı veriyi görüyor
    client = request.app.state.qdrant

    # Content-type kontrolü — tarayıcının gönderdiği header'a güven ama doğrula.
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Only PDF allowed.",
        )

    file_bytes = await file.read()

    # Boyut kontrolü read()'den sonra yapılır — stream tüketilmeli.
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is 20MB.",
        )

    result = await ingest_document(
        file_bytes=file_bytes,
        filename=file.filename,
        client=client,
    )

    return {"status": "success", **result}
