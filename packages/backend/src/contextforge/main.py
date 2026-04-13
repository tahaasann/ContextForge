from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from typing import TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from contextforge.core.config import settings
from contextforge.core.observability import create_instrumented_openai




class AppState(TypedDict):
    qdrant: AsyncQdrantClient
    openai: object   # LangfuseAsyncOpenAI — AsyncOpenAI ile aynı interface


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 ContextForge starting — env: {settings.app_env}")

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    # Tek değişiklik bu satır — gerisi tamamen aynı
    openai = create_instrumented_openai()

    app.state.qdrant = qdrant
    app.state.openai = openai

    yield

    await qdrant.close()
    print("🛑 ContextForge shutting down")


app = FastAPI(
    title="ContextForge API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from contextforge.api.eval import router as eval_router
from contextforge.api.ingest import router as ingest_router
from contextforge.api.query import router as query_router

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(eval_router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "env": settings.app_env,
        "model": settings.openai_model,
    }