from contextlib import asynccontextmanager
from typing import TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from contextforge.api.ingest import router as ingest_router
from contextforge.core.config import settings


# Uygulama state'inin tipini tanımlıyoruz.
# TypedDict: dict ama key'leri ve tipleri önceden bilinen.
# Bu sayede her yerden app.state.qdrant derken IDE bizi uyarıyor.
class AppState(TypedDict):
    qdrant: AsyncQdrantClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uygulama başlarken burası çalışır.
    # Veritabanı bağlantısı, model yükleme gibi işlemler buraya gelir.
    # Şimdilik sadece log basıyoruz.
    print(f"🚀 ContextForge starting — env: {settings.app_env}")

    qdrant = AsyncQdrantClient(":memory:")
    app.state.qdrant = qdrant

    yield
    # yield'den sonrası uygulama kapanırken çalışır.
    # Bağlantıları kapatmak, kaynakları serbest bırakmak buraya gelir.
    await qdrant.close()
    print("🛑 ContextForge shutting down")


app = FastAPI(
    title="ContextForge API",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(ingest_router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "env": settings.app_env,
        "model": settings.openai_model,
    }
