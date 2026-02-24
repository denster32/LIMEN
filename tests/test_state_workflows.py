import json
import tempfile
import unittest
from pathlib import Path

from scripts.backup_state import _decrypt, _encrypt
from scripts.reconcile_journal import reconcile
from scripts.state_ops import compute_entry_hash
from scripts.validate_state import validate_state


class StateWorkflowTests(unittest.TestCase):
    def base_state(self):
        entry = {
            "timestamp": "2026-01-01T00:00:00Z",
            "summary": "seed",
            "projects": [],
            "decisions": [],
            "pending": [],
            "mistakes": [],
            "insights": [],
            "entry_id": "seed-1",
            "prev_hash": "GENESIS",
        }
        entry["entry_hash"] = compute_entry_hash(entry)
        return {
            "brief": "brief",
            "active": {"LIMEN": "build"},
            "pending": ["a"],
            "avoid": ["b"],
            "log": [entry],
            "scratch": {},
            "meta": {
                "version": 2,
                "total_conversations": 1,
                "last_saved": "2026-01-01T00:00:00Z",
                "human_id": "h",
                "agent_id": "a",
                "session_id": "s",
            },
        }

    def test_validate_hash_chain(self):
        state = self.base_state()
        validate_state(state)

    def test_reconcile_adds_event(self):
        state = self.base_state()
        updated = reconcile(
            state,
            [{"timestamp": "2026-01-01T01:00:00Z", "summary": "did work", "pending": ["next"]}],
        )
        self.assertEqual(len(updated["log"]), 2)
        self.assertIn("next", updated["pending"])
        validate_state(updated)

    def test_backup_roundtrip(self):
        raw = json.dumps(self.base_state()).encode("utf-8")
        blob = _encrypt(raw, "pw")
        out = _decrypt(blob, "pw")
        self.assertEqual(raw, out)


if __name__ == "__main__":
    unittest.main()
