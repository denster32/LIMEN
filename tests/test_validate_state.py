import unittest

from scripts.state_core import log_entry_hash, state_snapshot_checksum
from scripts.validate_state import validate_state


class ValidateStateTests(unittest.TestCase):
    def _sample_state(self) -> dict:
        state = {
            "brief": "brief",
            "active": {"LIMEN": "ok"},
            "pending": ["a"],
            "avoid": ["b"],
            "log": [{"timestamp": "2026-01-01T00:00:00Z", "summary": "start"}],
            "scratch": {},
            "meta": {
                "version": 3,
                "total_conversations": 1,
                "human_id": "h1",
                "agent_id": "a1",
                "session_id": "s1",
                "last_saved": "2026-01-01T00:00:00Z",
                "snapshot_checksum": "",
            },
        }
        prev = "GENESIS"
        for entry in state["log"]:
            entry["entry_hash"] = log_entry_hash(prev, entry)
            prev = entry["entry_hash"]
        state["meta"]["snapshot_checksum"] = state_snapshot_checksum(state)
        return state

    def test_valid_state(self) -> None:
        validate_state(self._sample_state())

    def test_tamper_fails(self) -> None:
        state = self._sample_state()
        state["log"][0]["summary"] = "tampered"
        with self.assertRaises(ValueError):
            validate_state(state)


if __name__ == "__main__":
    unittest.main()
