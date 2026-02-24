import unittest

from scripts.evaluate_recall import recall_score


class RecallTests(unittest.TestCase):
    def test_recall_score_high(self) -> None:
        state = {
            "brief": "brief",
            "active": {"A": "B"},
            "pending": ["p"],
            "avoid": ["x"],
        }
        memory = "brief\nA\np\nx"
        self.assertGreaterEqual(recall_score(state, memory), 0.95)


if __name__ == "__main__":
    unittest.main()
