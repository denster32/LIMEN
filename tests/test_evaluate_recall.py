import unittest

from scripts.evaluate_recall import evaluate


class EvaluateRecallTests(unittest.TestCase):
    def test_evaluate_high_recall(self) -> None:
        state = {
            "brief": "short brief",
            "active": {"Project": "doing stuff"},
            "pending": ["item1"],
            "avoid": ["bad"],
            "log": [{"summary": "did work"}],
            "scratch": {},
            "meta": {"last_saved": "2026-01-01T12:00:00Z"},
        }
        hits, total = evaluate(state)
        self.assertEqual(hits, total)


if __name__ == "__main__":
    unittest.main()
