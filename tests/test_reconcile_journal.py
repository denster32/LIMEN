import unittest

from scripts.reconcile_journal import apply_event


class ReconcileJournalTests(unittest.TestCase):
    def test_apply_events(self) -> None:
        state = {
            "brief": "b",
            "active": {},
            "pending": [],
            "avoid": [],
            "log": [],
            "scratch": {},
            "meta": {"session_id": "s", "total_conversations": 0},
        }
        apply_event(state, {"type": "set_active", "project": "LIMEN", "status": "shipping"})
        apply_event(state, {"type": "add_pending", "item": "deploy"})
        apply_event(state, {"type": "log", "timestamp": "2026-01-01T00:00:00Z", "summary": "done"})
        self.assertEqual(state["active"]["LIMEN"], "shipping")
        self.assertIn("deploy", state["pending"])
        self.assertEqual(len(state["log"]), 1)
        self.assertIn("entry_hash", state["log"][0])


if __name__ == "__main__":
    unittest.main()
