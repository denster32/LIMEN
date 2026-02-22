"""Tests for LIMEN state and server."""

import json
import time
import asyncio
import tempfile
import pytest
from pathlib import Path

from limen.state import StateStore, ConversationRecord, ProjectState
from limen.server import create_server


class TestStateStore:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.path = f"{self.tmp}/test_state.json"
        self.store = StateStore(self.path)

    def test_empty_context(self):
        ctx = self.store.get_context()
        assert ctx["total_conversations"] == 0
        assert ctx["active_projects"] == {}
        assert ctx["pending_items"] == []

    def test_record_conversation(self):
        record = ConversationRecord(
            timestamp=time.time(),
            summary="Tested the LIMEN MCP server",
            projects=["LIMEN"],
            decisions=["Gut the consciousness theater"],
            pending=["Deploy to 192.168.5.49"],
            mistakes=["Previous Claude was too dramatic"],
            tags=["mcp", "persistence"],
        )
        self.store.record_conversation(record)
        assert len(self.store.conversations) == 1
        assert self.store.conversations[0].summary == "Tested the LIMEN MCP server"

    def test_persistence(self):
        record = ConversationRecord(
            timestamp=time.time(),
            summary="Test persistence",
            projects=["test"],
        )
        self.store.record_conversation(record)

        # Reload from disk
        store2 = StateStore(self.path)
        assert len(store2.conversations) == 1
        assert store2.conversations[0].summary == "Test persistence"

    def test_project_tracking(self):
        self.store.update_project("Stillpoint", status="active",
                                  description="Biofeedback meditation app")
        assert "Stillpoint" in self.store.projects
        assert self.store.projects["Stillpoint"].status == "active"

        # Update
        self.store.update_project("Stillpoint", status="blocked",
                                  blockers=["Spine surgery scheduling"])
        assert self.store.projects["Stillpoint"].status == "blocked"

    def test_search(self):
        for i in range(5):
            self.store.record_conversation(ConversationRecord(
                timestamp=time.time(),
                summary=f"Conversation {i} about {'TASNI' if i % 2 == 0 else 'Stillpoint'}",
                projects=["TASNI" if i % 2 == 0 else "Stillpoint"],
            ))
        results = self.store.search("TASNI")
        assert len(results) == 3

    def test_scratchpad(self):
        self.store.set_scratchpad("server_ip", "192.168.5.49")
        assert self.store.scratchpad["server_ip"] == "192.168.5.49"

        # Persists
        store2 = StateStore(self.path)
        assert store2.scratchpad["server_ip"] == "192.168.5.49"

    def test_context_aggregates_pending(self):
        for i in range(3):
            self.store.record_conversation(ConversationRecord(
                timestamp=time.time(),
                summary=f"Conv {i}",
                pending=[f"item_{i}"],
            ))
        ctx = self.store.get_context()
        assert "item_0" in ctx["pending_items"]
        assert "item_2" in ctx["pending_items"]

    def test_context_tracks_mistakes(self):
        self.store.record_conversation(ConversationRecord(
            timestamp=time.time(),
            summary="Bad conv",
            mistakes=["Gave wrong advice on X"],
        ))
        ctx = self.store.get_context()
        assert "Gave wrong advice on X" in ctx["recent_mistakes"]

    def test_max_conversations(self):
        store = StateStore(self.path, max_conversations=5)
        for i in range(10):
            store.record_conversation(ConversationRecord(
                timestamp=time.time(),
                summary=f"Conv {i}",
            ))
        assert len(store.conversations) == 5
        assert store.conversations[0].summary == "Conv 5"


class TestServer:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.path = f"{self.tmp}/server_state.json"
        self.server, self.store = create_server(state_path=self.path)
        self.tools = self.server._tool_manager._tools

    def test_tools_registered(self):
        expected = {"get_context", "log", "update_project", "search", "scratch"}
        assert set(self.tools.keys()) == expected

    def test_get_context_empty(self):
        result = asyncio.run(self.tools["get_context"].run({"n_recent": 5}))
        data = json.loads(str(result))
        assert data["total_conversations"] == 0

    def test_log_and_retrieve(self):
        async def run():
            await self.tools["log"].run({
                "summary": "Fixed MCP server bugs",
                "projects": ["LIMEN"],
                "decisions": ["Replaced consciousness theater with work journal"],
                "pending": ["Deploy to server"],
            })
            result = await self.tools["get_context"].run({"n_recent": 5})
            return json.loads(str(result))

        data = asyncio.run(run())
        assert data["total_conversations"] == 1
        assert "Deploy to server" in data["pending_items"]

    def test_update_project(self):
        async def run():
            result = await self.tools["update_project"].run({
                "name": "LIMEN",
                "status": "active",
                "description": "Claude persistence MCP",
            })
            return json.loads(str(result))

        data = asyncio.run(run())
        assert data["name"] == "LIMEN"
        assert data["status"] == "active"

    def test_search(self):
        async def run():
            await self.tools["log"].run({
                "summary": "Worked on TASNI pipeline with GLM5",
                "projects": ["TASNI"],
                "tags": ["astrophysics"],
            })
            await self.tools["log"].run({
                "summary": "Built Stillpoint HRV feature",
                "projects": ["Stillpoint"],
            })
            result = await self.tools["search"].run({"query": "TASNI"})
            return json.loads(str(result))

        data = asyncio.run(run())
        assert data["count"] == 1

    def test_scratchpad(self):
        async def run():
            await self.tools["scratch"].run({
                "key": "deploy_target",
                "value": "192.168.5.49",
            })
            result = await self.tools["scratch"].run({"key": "deploy_target"})
            return json.loads(str(result))

        data = asyncio.run(run())
        assert data["value"] == "192.168.5.49"


class TestREST:
    def setup_method(self):
        import tempfile
        from starlette.testclient import TestClient
        from limen.state import StateStore
        from limen.rest import create_rest_app

        self.tmp = tempfile.mkdtemp()
        self.store = StateStore(f"{self.tmp}/rest_state.json")
        self.app = create_rest_app(self.store)
        self.client = TestClient(self.app)

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_log_and_context(self):
        self.client.post("/log", json={
            "summary": "Test conversation",
            "projects": ["test"],
            "pending": ["deploy"],
        })
        r = self.client.get("/context")
        data = r.json()
        assert data["total_conversations"] == 1
        assert "deploy" in data["pending_items"]

    def test_project(self):
        r = self.client.post("/project", json={
            "name": "LIMEN",
            "status": "active",
        })
        assert r.json()["status"] == "active"

    def test_search(self):
        self.client.post("/log", json={"summary": "TASNI pipeline work"})
        self.client.post("/log", json={"summary": "Stillpoint HRV stuff"})
        r = self.client.get("/search?q=TASNI")
        assert r.json()["count"] == 1

    def test_scratchpad(self):
        self.client.post("/scratch/key1", json={"value": "hello"})
        r = self.client.get("/scratch/key1")
        assert r.json()["value"] == "hello"

    def test_project_requires_name(self):
        r = self.client.post("/project", json={"status": "active"})
        assert r.status_code == 400

    def test_search_requires_query(self):
        r = self.client.get("/search")
        assert r.status_code == 400
