"""
LIMEN REST API — HTTP interface for ALL Claude clients.

Design constraint: web_fetch is GET-only. So every operation
works via GET with query parameters. Not RESTful. Functional.

Auth: Bearer token in Authorization header OR ?token= query param.
(web_fetch can't set headers, so token in URL is the fallback.)

Endpoints:
    GET /health                          — Server health
    GET /context?n=5                     — Get conversation context
    GET /log?summary=...&projects=...    — Log a conversation
    GET /project?name=...&status=...     — Update a project
    GET /search?q=query                  — Search conversations
    GET /scratch?key=...                 — Read scratchpad
    GET /scratch?key=...&value=...       — Write scratchpad
    POST variants of all above also work — for clients that support POST
"""

import json
import time
import urllib.parse
from typing import Optional

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

from limen.state import StateStore, ConversationRecord


class TokenAuth(BaseHTTPMiddleware):
    """Simple bearer token auth. Checks header or query param."""

    def __init__(self, app, token: Optional[str] = None):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if not self.token:
            return await call_next(request)

        # Skip auth for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Check Authorization header
        auth = request.headers.get("authorization", "")
        if auth == f"Bearer {self.token}":
            return await call_next(request)

        # Check query param (for web_fetch which can't set headers)
        if request.query_params.get("token") == self.token:
            return await call_next(request)

        return JSONResponse({"error": "unauthorized"}, status_code=401)


def _parse_list(raw: Optional[str]) -> list[str]:
    """Parse comma-separated string into list. Returns [] if None/empty."""
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def create_rest_app(store: StateStore, auth_token: Optional[str] = None) -> Starlette:
    """Create the REST API sharing the same StateStore."""

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
        # Support both GET query params and POST JSON body
        if request.method == "POST":
            body = await request.json()
        else:
            body = dict(request.query_params)
            # Parse comma-separated lists from query params
            for field in ("projects", "decisions", "pending", "mistakes", "insights", "tags"):
                body[field] = _parse_list(body.get(field))

        record = ConversationRecord(
            timestamp=time.time(),
            summary=body.get("summary", ""),
            projects=body.get("projects", []) if isinstance(body.get("projects"), list) else _parse_list(body.get("projects")),
            decisions=body.get("decisions", []) if isinstance(body.get("decisions"), list) else _parse_list(body.get("decisions")),
            pending=body.get("pending", []) if isinstance(body.get("pending"), list) else _parse_list(body.get("pending")),
            mistakes=body.get("mistakes", []) if isinstance(body.get("mistakes"), list) else _parse_list(body.get("mistakes")),
            insights=body.get("insights", []) if isinstance(body.get("insights"), list) else _parse_list(body.get("insights")),
            mood=body.get("mood", ""),
            tags=body.get("tags", []) if isinstance(body.get("tags"), list) else _parse_list(body.get("tags")),
        )
        store.record_conversation(record)
        return JSONResponse({
            "status": "logged",
            "turn": len(store.conversations),
            "summary": record.summary,
        })

    async def update_project(request: Request) -> JSONResponse:
        if request.method == "POST":
            body = await request.json()
        else:
            body = dict(request.query_params)
            for field in ("notes", "blockers"):
                body[field] = _parse_list(body.get(field))

        name = body.pop("name", None)
        body.pop("token", None)  # Remove auth token from kwargs
        if not name:
            return JSONResponse({"error": "name required"}, status_code=400)

        # Clean kwargs
        kwargs = {}
        for k in ("status", "description"):
            if k in body and body[k]:
                kwargs[k] = body[k]
        for k in ("notes", "blockers"):
            if k in body:
                val = body[k]
                kwargs[k] = val if isinstance(val, list) else _parse_list(val)

        store.update_project(name, **kwargs)
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

    async def scratch_handler(request: Request) -> JSONResponse:
        if request.method == "POST":
            body = await request.json()
            key = request.query_params.get("key") or body.get("key")
            value = body.get("value")
        else:
            key = request.query_params.get("key")
            value = request.query_params.get("value")

        if not key:
            # List all keys
            return JSONResponse({"keys": list(store.scratchpad.keys())})

        if value is not None:
            store.set_scratchpad(key, value)

        current = store.scratchpad.get(key)
        return JSONResponse({"key": key, "value": current})

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/context", get_context, methods=["GET"]),
            Route("/log", log_conversation, methods=["GET", "POST"]),
            Route("/project", update_project, methods=["GET", "POST"]),
            Route("/search", search_conversations, methods=["GET"]),
            Route("/scratch", scratch_handler, methods=["GET", "POST"]),
        ],
        middleware=[
            Middleware(TokenAuth, token=auth_token),
        ],
    )
