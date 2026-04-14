import asyncio
import json

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ContextForge backend'in adresi
# MCP server, backend ile HTTP üzerinden konuşuyor.
# İkisi ayrı process — birbirinden bağımsız çalışıyor.
BACKEND_URL = "http://localhost:8001"

app = Server("contextforge")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    MCP client "hangi tool'ların var?" diye sorduğunda bu çalışır.
    Tool description'ları kritik — AI bu metni okuyarak
    ne zaman hangi tool'u kullanacağına karar verir.
    """
    return [
        Tool(
            name="search_documents",
            description=(
                "Search the ContextForge knowledge base using semantic and keyword search. "
                "Returns relevant document chunks with page references. "
                "Use when the user wants to find specific information in uploaded documents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask_question",
            description=(
                "Ask a question and get a complete RAG-generated answer with source citations "
                "and a reliability evaluation score. "
                "Use when the user wants a synthesized answer, not just raw search results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of context chunks to use (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="get_pipeline_health",
            description=(
                "Check if the ContextForge backend is running and get current configuration. "
                "Use when you need to verify the system is operational."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    MCP client bir tool çalıştırmak istediğinde bu çalışır.
    Her tool backend'e HTTP isteği atıyor.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:

        if name == "search_documents":
            # Backend'in /query/ask endpoint'ini kullan ama
            # sadece chunk'ları döndür, cevap üretme
            response = await client.post(
                f"{BACKEND_URL}/query/ask",
                json={
                    "question": arguments["query"],
                    "top_k": arguments.get("top_k", 5),
                },
            )
            data = response.json()

            # Sadece sources ve answer'ın ilk kısmını döndür
            result = {
                "sources": data.get("sources", []),
                "preview": data.get("answer", "")[:300],
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "ask_question":
            response = await client.post(
                f"{BACKEND_URL}/query/ask",
                json={
                    "question": arguments["question"],
                    "top_k": arguments.get("top_k", 5),
                },
            )
            data = response.json()

            result = {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "eval_score": data.get("eval_score"),
                "eval_reasoning": data.get("eval_reasoning", ""),
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

        elif name == "get_pipeline_health":
            response = await client.get(f"{BACKEND_URL}/health")
            return [TextContent(type="text", text=json.dumps(response.json(), indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())