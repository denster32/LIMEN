"""Tests for LIMEN persistence layer."""
import json
import tempfile
import time

from limen.state import StateStore, ConversationRecord
from limen.server import create_server


class TestStateStore:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.store = StateStore(f"{self.tmp}/state.json")

    def test_empty_context(self):
        ctx = self.store.get_context()
        assert ctx["total_conversations"] == 0
        assert ctx["recent_conversations"] == []
        assert ctx["active_projects"] == {}

    def test_record_conversation(self):
        record = ConversationRecord(
            timestamp=time.time(),
            summary="Test conversation",
            projects=["LIMEN"],
            decisions=["Use JSON storage"],
            pending=["Deploy to server"],
            mistakes=[],
            insights=["Simple beats clever"],
            mood="focused",
            tags=["test"],
        )
        self.store.record_conversation(record)
        assert len(self.store.conversations) == 1
        assert self.store.conversations[0].summary == "Test conversation"

    def test_persistence(self):
        record = ConversationRecord(
            timestamp=time.time(),
            summary="Persistence test",
            projects=[],
            decisions=[],
            pending=[],
            mistakes=[],
            insights=[],
            mood="",
            tags=[],
        )
        self.store.record_conversation(record)

        store2 = StateStore(f"{self.tmp}/state.json")
        assert len(store2.conversations) == 1
        assert store2.conversations[0].summary == "Persistence test"

    def test_project_tracking(self):
        self.store.update_project(
            "LIMEN",
            status="active",
            description="Claude persistence layer",
        )
        assert "LIMEN" in self.store.projects
        assert self.store.projects["LIMEN"].status == "active"

        self.store.update_project("LIMEN", status="done")
        assert self.store.projects["LIMEN"].status == "done"

    def test_search(self):
        for i in range(3):
            record = ConversationRecord(
                timestamp=time.time(),
                summary=f"Conversation {i} about {'TASNI' if i == 1 else 'other stuff'}",
                projects=[],
                decisions=[],
                pending=[],
                mistakes=[],
                insights=[],
                mood="",
                tags=[],
            )
            self.store.record_conversation(record)

        results = self.store.search("TASNI")
        assert len(results) == 1
        assert "TASNI" in results[0]["summary"]

    def test_scratchpad(self):
        self.store.set_scratchpad("key1", "value1")
        assert self.store.scratchpad["key1"] == "value1"

        store2 = StateStore(f"{self.tmp}/state.json")
        assert store2.scratchpad["key1"] == "value1"

    def test_context_aggregates_pending(self):
        for i in range(3):
            record = ConversationRecord(
                timestamp=time.time(),
                summary=f"Conv {i}",
                projects=[],
                decisions=[],
                pending=[f"todo_{i}"],
                mistakes=[],
                insights=[],
                mood="",
                tags=[],
            )
            self.store.record_conversation(record)

        ctx = self.store.get_context(n_recent=5)
        assert len(ctx["pending_items"]) == 3

    def test_context_tracks_mistakes(self):
        record = ConversationRecord(
            timestamp=time.time(),
            summary="Made a mistake",
            projects=[],
            decisions=[],
            pending=[],
            mistakes=["Used wrong API"],
            insights=[],
            mood="",
            tags=[],
        )
        self.store.record_conversation(record)
        ctx = self.store.get_context()
        assert "Used wrong API" in ctx["recent_mistakes"]

    def test_max_conversations(self):
        for i in range(210):
            record = ConversationRecord(
                timestamp=time.time(),
                summary=f"Conv {i}",
                projects=[],
                decisions=[],
                pending=[],
                mistakes=[],
                insights=[],
                mood="",
                tags=[],
            )
            self.store.record_conversation(record)
        assert len(self.store.conversations) == 200


class TestServer:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.path = f"{self.tmp}/server_state.json"
        self.server, self.store = create_server(state_path=self.path)
        self.tools = self.server._tool_manager._tools

    def test_tools_registered(self):
        names = set(self.tools.keys())
        assert names == {"get_context", "log", "update_project", "search", "scratch"}

    def test_get_context_empty(self):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self.tools["get_context"].run({"n_recent": 5})
        )
        data = json.loads(result)
        assert data["total_conversations"] == 0

    def test_log_and_retrieve(self):
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tools["log"].run({
                "summary": "Test log entry",
                "projects": ["LIMEN"],
                "decisions": [],
                "pending": ["deploy"],
                "mistakes": [],
                "insights": [],
                "mood": "",
                "tags": [],
            })
        )
        result = loop.run_until_complete(
            self.tools["get_context"].run({"n_recent": 5})
        )
        data = json.loads(result)
        assert data["total_conversations"] == 1

    def test_update_project(self):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self.tools["update_project"].run({
                "name": "LIMEN",
                "status": "active",
            })
        )
        data = json.loads(result)
        assert data["status"] == "active"

    def test_search(self):
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tools["log"].run({
                "summary": "TASNI pipeline work",
                "projects": [],
                "decisions": [],
                "pending": [],
                "mistakes": [],
                "insights": [],
                "mood": "",
                "tags": [],
            })
        )
        result = loop.run_until_complete(
            self.tools["search"].run({"query": "TASNI"})
        )
        data = json.loads(result)
        assert data["count"] == 1

    def test_scratchpad(self):
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tools["scratch"].run({"key": "test", "value": "hello"})
        )
        result = loop.run_until_complete(
            self.tools["scratch"].run({"key": "test"})
        )
        data = json.loads(result)
        assert data["value"] == "hello"


class TestREST:
    def setup_method(self):
        from starlette.testclient import TestClient
        from limen.rest import create_rest_app

        self.tmp = tempfile.mkdtemp()
        self.store = StateStore(f"{self.tmp}/rest_state.json")
        # Test without auth
        self.app = create_rest_app(self.store)
        self.client = TestClient(self.app)

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_log_post(self):
        r = self.client.post("/log", json={
            "summary": "Test via POST",
            "projects": ["test"],
        })
        assert r.json()["status"] == "logged"

    def test_log_get(self):
        """web_fetch compatibility: log via GET with query params."""
        r = self.client.get("/log?summary=Test+via+GET&projects=LIMEN,TASNI&pending=deploy")
        assert r.json()["status"] == "logged"
        ctx = self.client.get("/context").json()
        assert ctx["total_conversations"] == 1
        assert "deploy" in ctx["pending_items"]

    def test_context(self):
        self.client.get("/log?summary=test&projects=A")
        r = self.client.get("/context?n=5")
        assert r.json()["total_conversations"] == 1

    def test_project_get(self):
        """web_fetch compatibility: update project via GET."""
        r = self.client.get("/project?name=LIMEN&status=active&description=persistence+layer")
        assert r.json()["status"] == "active"

    def test_project_requires_name(self):
        r = self.client.get("/project?status=active")
        assert r.status_code == 400

    def test_search(self):
        self.client.get("/log?summary=TASNI+pipeline+work")
        self.client.get("/log?summary=Stillpoint+HRV+stuff")
        r = self.client.get("/search?q=TASNI")
        assert r.json()["count"] == 1

    def test_search_requires_query(self):
        r = self.client.get("/search")
        assert r.status_code == 400

    def test_scratch_write_read(self):
        """web_fetch compatibility: scratch via GET."""
        self.client.get("/scratch?key=ip&value=192.168.5.49")
        r = self.client.get("/scratch?key=ip")
        assert r.json()["value"] == "192.168.5.49"

    def test_scratch_list_keys(self):
        self.client.get("/scratch?key=a&value=1")
        self.client.get("/scratch?key=b&value=2")
        r = self.client.get("/scratch")
        assert set(r.json()["keys"]) == {"a", "b"}


class TestAuth:
    def setup_method(self):
        from starlette.testclient import TestClient
        from limen.rest import create_rest_app

        self.tmp = tempfile.mkdtemp()
        self.store = StateStore(f"{self.tmp}/auth_state.json")
        self.token = "test-secret-token"
        self.app = create_rest_app(self.store, auth_token=self.token)
        self.client = TestClient(self.app)

    def test_health_no_auth_needed(self):
        r = self.client.get("/health")
        assert r.status_code == 200

    def test_reject_no_token(self):
        r = self.client.get("/context")
        assert r.status_code == 401

    def test_accept_header_token(self):
        r = self.client.get("/context", headers={"Authorization": f"Bearer {self.token}"})
        assert r.status_code == 200

    def test_accept_query_token(self):
        """web_fetch compatibility: token in URL."""
        r = self.client.get(f"/context?token={self.token}")
        assert r.status_code == 200

    def test_reject_wrong_token(self):
        r = self.client.get("/context?token=wrong")
        assert r.status_code == 401
