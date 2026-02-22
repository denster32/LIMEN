"""
LIMEN REST API — HTTP interface for clients without MCP support.

Claude.ai can use web_fetch to hit these endpoints directly.
No MCP required. Just HTTP GET/POST.

Endpoints:
    GET  /context              — Get conversation context
    POST /log                  — Log a conversation
    POST /project              — Update a project
    GET  /search?q=query       — Search past conversations
    GET  /scratch/:key         — Read scratchpad key
    POST /scratch/:key         — Write scratchpad key
    GET  /health               — Server health check
"""

import json
import time
from typing import Optional

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse

from limen.state import StateStore, ConversationRecord


def create_rest_app(store: StateStore) -> Starlette:
    """Create the REST API app sharing the same StateStore."""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "service": "limen",
            "conversations": len(store.conversations),
            "projects": len(store.projects),
            "timestamp": time.time(),
        })

    async def get_context(request: Request) -> JSONResponse:
        n = int(request.query_params.get("n", 5))
        ctx = store.get_context(n)
        return JSONResponse(ctx)

    async def log_conversation(request: Request) -> JSONResponse:
        body = await request.json()
        record = ConversationRecord(
            timestamp=time.time(),
            summary=body.get("summary", ""),
            projects=body.get("projects", []),
            decisions=body.get("decisions", []),
            pending=body.get("pending", []),
            mistakes=body.get("mistakes", []),
            insights=body.get("insights", []),
            mood=body.get("mood", ""),
            tags=body.get("tags", []),
        )
        store.record_conversation(record)
        return JSONResponse({
            "status": "logged",
            "turn": len(store.conversations),
            "summary": record.summary,
        })

    async def update_project(request: Request) -> JSONResponse:
        body = await request.json()
        name = body.pop("name", None)
        if not name:
            return JSONResponse({"error": "name required"}, status_code=400)
        store.update_project(name, **body)
        return JSONResponse(store.projects[name].to_dict())

    async def search_conversations(request: Request) -> JSONResponse:
        query = request.query_params.get("q", "")
        if not query:
            return JSONResponse({"error": "q parameter required"}, status_code=400)
        results = store.search(query)
        return JSONResponse({
            "query": query,
            "results": results,
            "count": len(results),
        })

    async def get_scratch(request: Request) -> JSONResponse:
        key = request.path_params["key"]
        value = store.scratchpad.get(key)
        return JSONResponse({"key": key, "value": value})

    async def set_scratch(request: Request) -> JSONResponse:
        key = request.path_params["key"]
        body = await request.json()
        store.set_scratchpad(key, body.get("value"))
        return JSONResponse({"key": key, "value": store.scratchpad.get(key)})

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/context", get_context, methods=["GET"]),
            Route("/log", log_conversation, methods=["POST"]),
            Route("/project", update_project, methods=["POST"]),
            Route("/search", search_conversations, methods=["GET"]),
            Route("/scratch/{key}", get_scratch, methods=["GET"]),
            Route("/scratch/{key}", set_scratch, methods=["POST"]),
        ],
    )
