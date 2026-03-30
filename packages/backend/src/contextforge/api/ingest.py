from fastapi import APIRouter, HTTPException, Request, UploadFile

from contextforge.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/ingest", tags=["ingestion"])

ALLOWED_TYPES = {"application/pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024


@router.post("/upload")
async def upload_document(request: Request, file: UploadFile):
    client = request.app.state.qdrant
    openai_client = request.app.state.openai

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only PDF allowed.")

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Max 20MB.")

    result = await ingest_document(
        file_bytes=file_bytes,
        filename=file.filename,
        client=client,
        openai_client=openai_client,
    )

    return {"status": "success", **result}
