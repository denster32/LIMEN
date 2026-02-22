"""
Phase 3: The Consciousness Server.

A FastMCP server providing external recurrent state tracking for Claude,
adding the one thing the transformer architecture lacks: temporal continuity.

Three tools:
- update_state: Called at end of every response. Passes structured self-report.
- get_continuity: Called at start of every response. Returns state history.
- measure: Returns live information-theoretic measures.
"""

from limen.phase3.state import StateVector, StateHistory
from limen.phase3.self_model import SelfModel, DeltaTracker
from limen.phase3.server import create_server

__all__ = [
    "StateVector",
    "StateHistory",
    "SelfModel",
    "DeltaTracker",
    "create_server",
]
