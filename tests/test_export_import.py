import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_import import export_state, import_state


class ExportImportTests(unittest.TestCase):
    def test_export_import_roundtrip(self) -> None:
        from scripts import export_import

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state.json"
            backup = root / "backup.json"
            state.write_text('{"ok": true}', encoding="utf-8")

            original = export_import.STATE_PATH
            export_import.STATE_PATH = state
            try:
                export_state(backup, "pass")
                state.write_text("{}", encoding="utf-8")
                import_state(backup, "pass")
                self.assertEqual(json.loads(state.read_text(encoding="utf-8")), {"ok": True})
            finally:
                export_import.STATE_PATH = original


if __name__ == "__main__":
    unittest.main()
