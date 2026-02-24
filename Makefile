.PHONY: validate sync-memory test check reconcile-journal

validate:
	python3 -m scripts.validate_state

sync-memory:
	python3 -m scripts.sync_memory

reconcile-journal:
	python3 -m scripts.local_journal reconcile

test:
	python3 -m unittest discover -s tests -q

check: validate sync-memory test
	git diff --exit-code -- MEMORY.md
