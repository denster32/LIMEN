.PHONY: validate sync-memory test recall check reconcile-journal

validate:
	python3 -m scripts.validate_state

sync-memory:
	python3 -m scripts.sync_memory

reconcile-journal:
	python3 -m scripts.reconcile_journal

test:
	python3 -m unittest discover -s tests -q

recall:
	python3 -m scripts.evaluate_recall

check: validate sync-memory test recall
	git diff --exit-code -- MEMORY.md
