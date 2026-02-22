"""
State management for LIMEN persistence server.

Tracks what matters between conversations:
- What we're working on
- What decisions were made
- What's pending
- What went wrong
- Conversation trajectory
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ConversationRecord:
    """One conversation's worth of context."""
    timestamp: float
    summary: str  # What happened this conversation
    projects: list[str] = field(default_factory=list)  # Active projects touched
    decisions: list[str] = field(default_factory=list)  # Decisions made
    pending: list[str] = field(default_factory=list)  # Action items / blockers
    mistakes: list[str] = field(default_factory=list)  # What I got wrong
    insights: list[str] = field(default_factory=list)  # Key realizations
    mood: str = ""  # Brief read on how Dennis seemed (not my mood, his)
    tags: list[str] = field(default_factory=list)  # Freeform tags for search

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProjectState:
    """Persistent state for a project."""
    name: str
    status: str  # active, paused, blocked, done
    description: str = ""
    last_touched: float = 0.0
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StateStore:
    """
    Persistent state store. JSON file on disk.
    
    Structure:
    - conversations: list of ConversationRecords (last N)
    - projects: dict of project name -> ProjectState
    - scratchpad: freeform dict for anything else
    """

    def __init__(self, path: str, max_conversations: int = 200):
        self.path = Path(path)
        self.max_conversations = max_conversations
        self.conversations: list[ConversationRecord] = []
        self.projects: dict[str, ProjectState] = {}
        self.scratchpad: dict = {}
        self._load()

    def record_conversation(self, record: ConversationRecord):
        """Add a conversation record."""
        self.conversations.append(record)
        if len(self.conversations) > self.max_conversations:
            self.conversations = self.conversations[-self.max_conversations:]

        # Update project last_touched
        for project_name in record.projects:
            if project_name in self.projects:
                self.projects[project_name].last_touched = record.timestamp

        self._save()

    def update_project(self, name: str, **kwargs):
        """Create or update a project."""
        if name in self.projects:
            for k, v in kwargs.items():
                if hasattr(self.projects[name], k):
                    setattr(self.projects[name], k, v)
            self.projects[name].last_touched = time.time()
        else:
            self.projects[name] = ProjectState(
                name=name,
                last_touched=time.time(),
                **{k: v for k, v in kwargs.items()
                   if k in ProjectState.__dataclass_fields__ and k != 'name'}
            )
        self._save()

    def get_context(self, n_recent: int = 5) -> dict:
        """
        Get the context I need at the start of a conversation.
        
        Returns recent conversations, active projects, and pending items.
        """
        recent = self.conversations[-n_recent:] if self.conversations else []

        # Collect all pending items from recent conversations
        all_pending = []
        for conv in self.conversations[-20:]:  # Look back further for pending
            for item in conv.pending:
                if item not in all_pending:
                    all_pending.append(item)

        # Active projects sorted by last touched
        active_projects = {
            name: p.to_dict() for name, p in self.projects.items()
            if p.status in ("active", "blocked")
        }

        # Recent mistakes (so I don't repeat them)
        recent_mistakes = []
        for conv in self.conversations[-10:]:
            recent_mistakes.extend(conv.mistakes)

        return {
            "recent_conversations": [r.to_dict() for r in recent],
            "active_projects": active_projects,
            "pending_items": all_pending[-20:],  # Cap at 20
            "recent_mistakes": recent_mistakes[-10:],  # Cap at 10
            "scratchpad": self.scratchpad,
            "total_conversations": len(self.conversations),
        }

    def search(self, query: str) -> list[dict]:
        """Simple text search across conversation records."""
        query_lower = query.lower()
        results = []
        for conv in reversed(self.conversations):
            searchable = json.dumps(conv.to_dict()).lower()
            if query_lower in searchable:
                results.append(conv.to_dict())
            if len(results) >= 10:
                break
        return results

    def set_scratchpad(self, key: str, value):
        """Store arbitrary data."""
        self.scratchpad[key] = value
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "conversations": [c.to_dict() for c in self.conversations],
            "projects": {n: p.to_dict() for n, p in self.projects.items()},
            "scratchpad": self.scratchpad,
            "last_saved": time.time(),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.conversations = [
                ConversationRecord.from_dict(c)
                for c in data.get("conversations", [])
            ]
            self.projects = {
                n: ProjectState.from_dict(p)
                for n, p in data.get("projects", {}).items()
            }
            self.scratchpad = data.get("scratchpad", {})
        except (json.JSONDecodeError, KeyError):
            pass  # Start fresh
