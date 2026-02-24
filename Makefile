.PHONY: validate sync-memory test check

validate:
	python3 scripts/validate_state.py

sync-memory:
	python3 scripts/sync_memory.py

test:
	python3 -m unittest discover -s tests -q

check: validate sync-memory test
	git diff --exit-code -- MEMORY.md
