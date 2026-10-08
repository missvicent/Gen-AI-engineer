include ../../../../../_lesson.mk

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }
	cp "$(FROM)/src/assistant/deadline.py" src/assistant/deadline.py
	cp "$(FROM)/src/assistant/resilience.py" src/assistant/resilience.py
	cp "$(FROM)/src/assistant/idempotency.py" src/assistant/idempotency.py
	cp "$(FROM)/src/assistant/outbox.py" src/assistant/outbox.py
	cp "$(FROM)/src/assistant/approvals.py" src/assistant/approvals.py
