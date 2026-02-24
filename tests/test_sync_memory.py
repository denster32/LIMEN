import unittest

from scripts.sync_memory import render_memory


class SyncMemoryTests(unittest.TestCase):
    def test_render_memory_contains_sections(self) -> None:
        state = {
            "brief": "short brief",
            "active": {"Project": "doing stuff"},
            "pending": ["item1"],
            "avoid": ["bad"],
            "log": [{"summary": "did work"}],
            "scratch": {},
            "meta": {
                "last_saved": "2026-01-01T12:00:00Z",
                "human_id": "h1",
                "agent_id": "a1",
                "session_id": "s1",
                "state_checksum": "abc123",
            },
        }

        output = render_memory(state)

        self.assertIn("# Memory", output)
        self.assertIn("## Working On", output)
        self.assertIn("**Project** — doing stuff", output)
        self.assertIn("## Pending", output)
        self.assertIn("## Don't", output)
        self.assertIn("## Last Session", output)
        self.assertIn("## Identity", output)
        self.assertIn("Updated 2026-01-01", output)


if __name__ == "__main__":
    unittest.main()
