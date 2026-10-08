include ../../../../../_lesson.mk

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }
	cp "$(FROM)/src/assistant/guardrails.py" src/assistant/guardrails.py
	cp "$(FROM)/src/assistant/screening.py" src/assistant/screening.py
	cp "$(FROM)/src/assistant/guard.py" src/assistant/guard.py
