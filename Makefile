.PHONY: validate sync-memory reconcile-journal eval-recall test check

validate:
	python3 -m scripts.validate_state

sync-memory:
	python3 -m scripts.sync_memory

reconcile-journal:
	python3 -m scripts.reconcile_journal

eval-recall:
	python3 -m scripts.evaluate_recall

test:
	python3 -m unittest discover -s tests -q

check: validate sync-memory eval-recall test
	git diff --exit-code -- MEMORY.md
