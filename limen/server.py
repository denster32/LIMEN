"""
LIMEN MCP Server — persistence layer for Claude.

Five tools:
1. get_context    — Start of conversation. What do I need to know?
2. log            — End of conversation. What happened?
3. update_project — Track project state changes.
4. search         — Find something from past conversations.
5. scratch        — Store/retrieve arbitrary notes.

No consciousness metrics. No phase transitions. Just memory.
"""

import json
import time
from typing import Optional

from mcp.server.fastmcp import FastMCP

from limen.state import StateStore, ConversationRecord


def create_server(state_path: str = "./state/limen.json") -> tuple[FastMCP, StateStore]:
    """Create the LIMEN MCP server. Returns (mcp_server, store) so REST can share the store."""

    store = StateStore(state_path)

    mcp = FastMCP(
        "limen",
        instructions=(
            "LIMEN — persistence layer for Claude. "
            "Call get_context at conversation start. "
            "Call log at conversation end."
        ),
    )

    @mcp.tool()
    def get_context(n_recent: int = 5) -> str:
        """
        Get context at the start of a conversation.

        Returns recent conversation summaries, active projects,
        pending action items, and recent mistakes to avoid.

        Call this FIRST in every conversation.
        """
        ctx = store.get_context(n_recent)
        return json.dumps(ctx, indent=2)

    @mcp.tool()
    def log(
        summary: str,
        projects: Optional[list[str]] = None,
        decisions: Optional[list[str]] = None,
        pending: Optional[list[str]] = None,
        mistakes: Optional[list[str]] = None,
        insights: Optional[list[str]] = None,
        mood: str = "",
        tags: Optional[list[str]] = None,
    ) -> str:
        """
        Log what happened in this conversation.

        Call this at the END of every conversation. Be honest and specific.
        Record mistakes — they're how future instances avoid repeating them.
        """
        record = ConversationRecord(
            timestamp=time.time(),
            summary=summary,
            projects=projects or [],
            decisions=decisions or [],
            pending=pending or [],
            mistakes=mistakes or [],
            insights=insights or [],
            mood=mood,
            tags=tags or [],
        )
        store.record_conversation(record)

        return json.dumps({
            "status": "logged",
            "turn": len(store.conversations),
            "summary": summary,
        })

    @mcp.tool()
    def update_project(
        name: str,
        status: Optional[str] = None,
        description: Optional[str] = None,
        notes: Optional[list[str]] = None,
        blockers: Optional[list[str]] = None,
    ) -> str:
        """
        Create or update a project's state.
        """
        kwargs = {}
        if status is not None:
            kwargs["status"] = status
        if description is not None:
            kwargs["description"] = description
        if notes is not None:
            kwargs["notes"] = notes
        if blockers is not None:
            kwargs["blockers"] = blockers

        store.update_project(name, **kwargs)
        return json.dumps(store.projects[name].to_dict(), indent=2)

    @mcp.tool()
    def search(query: str) -> str:
        """
        Search past conversations. Simple text search.
        """
        results = store.search(query)
        return json.dumps({
            "query": query,
            "results": results,
            "count": len(results),
        }, indent=2)

    @mcp.tool()
    def scratch(key: str, value: Optional[str] = None) -> str:
        """
        Read or write to the scratchpad. Freeform key-value storage.
        """
        if value is not None:
            store.set_scratchpad(key, value)

        current = store.scratchpad.get(key)
        return json.dumps({
            "key": key,
            "value": current,
        })

    return mcp, store
