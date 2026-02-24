.PHONY: validate sync-memory test check recall migrate-state journal-sync journal-append export-state import-state

validate:
	python3 scripts/validate_state.py

sync-memory:
	python3 scripts/sync_memory.py

recall:
	python3 scripts/evaluate_recall.py

migrate-state:
	python3 scripts/migrate_state.py

journal-sync:
	python3 scripts/local_journal.py sync

journal-append:
	python3 scripts/local_journal.py append --summary "manual journal entry"

export-state:
	python3 scripts/export_import.py export --output backups/limen.enc.json --passphrase "$${LIMEN_BACKUP_PASSPHRASE}"

import-state:
	python3 scripts/export_import.py import --input backups/limen.enc.json --passphrase "$${LIMEN_BACKUP_PASSPHRASE}"

test:
	python3 -m unittest discover -s tests -q

check: validate sync-memory recall test
	git diff --exit-code -- MEMORY.md
