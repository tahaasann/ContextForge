from langfuse import get_client
from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

from contextforge.core.config import settings


def get_langfuse():
    return get_client()


def create_instrumented_openai() -> LangfuseAsyncOpenAI:
    # Key'ler os.environ'dan okunuyor — load_dotenv() main.py'da çağrılmalı
    return LangfuseAsyncOpenAI(api_key=settings.openai_api_key)