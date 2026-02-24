import json
import unittest

from scripts.export_state import encrypt
from scripts.import_state import decrypt


class BackupRoundtripTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        original = json.dumps({"hello": "world"}).encode("utf-8")
        payload = encrypt(original, "pw")
        restored = decrypt(payload, "pw")
        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
