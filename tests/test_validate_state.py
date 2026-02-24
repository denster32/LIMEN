import json
import unittest

from scripts.migrate_state import migrate
from scripts.validate_state import validate_state


class ValidateStateTests(unittest.TestCase):
    def test_migrated_state_validates(self) -> None:
        state = {
            "brief": "b",
            "active": {"P": "S"},
            "pending": ["n"],
            "avoid": ["a"],
            "log": [{"timestamp": "2026-01-01T00:00:00Z", "summary": "x"}],
            "scratch": {},
            "meta": {"version": 2, "total_conversations": 1},
        }
        migrated = migrate(json.loads(json.dumps(state)))
        validate_state(migrated)


if __name__ == "__main__":
    unittest.main()
