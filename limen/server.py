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


def create_server(state_path: str = "./state/limen.json") -> FastMCP:
    """Create the LIMEN MCP server."""

    mcp = FastMCP(
        "limen",
        instructions=(
            "LIMEN — persistence layer for Claude. "
            "Call get_context at conversation start. "
            "Call log at conversation end."
        ),
    )

    store = StateStore(state_path)

    @mcp.tool()
    def get_context(n_recent: int = 5) -> str:
        """
        Get context at the start of a conversation.

        Returns recent conversation summaries, active projects,
        pending action items, and recent mistakes to avoid.

        Call this FIRST in every conversation.

        Parameters
        ----------
        n_recent : int
            Number of recent conversations to include (default: 5).

        Returns
        -------
        str
            JSON context object.
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

        Parameters
        ----------
        summary : str
            Brief summary of what happened.
        projects : list[str], optional
            Project names touched in this conversation.
        decisions : list[str], optional
            Decisions that were made.
        pending : list[str], optional
            Action items or blockers going forward.
        mistakes : list[str], optional
            Things I got wrong or should do differently.
        insights : list[str], optional
            Key realizations or learnings.
        mood : str, optional
            Brief read on how Dennis seemed.
        tags : list[str], optional
            Freeform tags for searchability.

        Returns
        -------
        str
            Confirmation.
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

        Parameters
        ----------
        name : str
            Project name (e.g., "Stillpoint", "TASNI", "LIMEN").
        status : str, optional
            One of: active, paused, blocked, done.
        description : str, optional
            What this project is.
        notes : list[str], optional
            Current notes (replaces existing).
        blockers : list[str], optional
            Current blockers (replaces existing).

        Returns
        -------
        str
            Updated project state.
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
        Search past conversations.

        Simple text search across all conversation records.

        Parameters
        ----------
        query : str
            Search text.

        Returns
        -------
        str
            Matching conversation records (most recent first).
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
        Read or write to the scratchpad.

        The scratchpad is freeform key-value storage for anything
        that doesn't fit conversations or projects.

        Parameters
        ----------
        key : str
            Key to read or write.
        value : str, optional
            If provided, stores this value. If omitted, reads current value.

        Returns
        -------
        str
            Current value for the key.
        """
        if value is not None:
            store.set_scratchpad(key, value)

        current = store.scratchpad.get(key)
        return json.dumps({
            "key": key,
            "value": current,
        })

    return mcp
