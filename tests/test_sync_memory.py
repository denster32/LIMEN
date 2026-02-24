import unittest

from scripts.state_integrity import with_integrity_fields
from scripts.sync_memory import render_memory
from scripts.validate_state import validate_state


class SyncMemoryTests(unittest.TestCase):
    def test_render_memory_contains_sections(self) -> None:
        state = {
            "brief": "short brief",
            "active": {"Project": "doing stuff"},
            "pending": ["item1"],
            "avoid": ["bad"],
            "log": [{"timestamp": "2026-01-01T12:00:00Z", "summary": "did work"}],
            "scratch": {},
            "meta": {
                "version": 2,
                "human_id": "human-a",
                "agent_id": "agent-a",
                "session_id": "session-a",
                "total_conversations": 1,
                "last_saved": "2026-01-01T12:00:00Z",
            },
        }

        complete_state = with_integrity_fields(state)
        validate_state(complete_state)
        output = render_memory(complete_state)

        self.assertIn("# Memory", output)
        self.assertIn("## Working On", output)
        self.assertIn("**Project** — doing stuff", output)
        self.assertIn("## Pending", output)
        self.assertIn("## Don't", output)
        self.assertIn("## Last Session", output)
        self.assertIn("Updated 2026-01-01", output)

    def test_cold_start_recall_fields_present(self) -> None:
        state = with_integrity_fields(
            {
                "brief": "Keep focus",
                "active": {"LIMEN": "ship deploy-ready tooling"},
                "pending": ["run checks", "sync memory"],
                "avoid": ["overengineering"],
                "log": [{"timestamp": "2026-02-22T23:57:18Z", "summary": "session summary"}],
                "scratch": {},
                "meta": {
                    "version": 2,
                    "human_id": "human",
                    "agent_id": "agent",
                    "session_id": "s-1",
                    "total_conversations": 1,
                    "last_saved": "2026-02-22T23:57:18Z",
                },
            }
        )

        output = render_memory(state)
        self.assertIn("Keep focus", output)
        self.assertIn("**LIMEN** — ship deploy-ready tooling", output)
        self.assertIn("- run checks", output)
        self.assertIn("- overengineering", output)


if __name__ == "__main__":
    unittest.main()
