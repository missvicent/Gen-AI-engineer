include ../../../../../_lesson.mk

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }
	cp "$(FROM)/src/assistant/service.py" src/assistant/service.py
	cp "$(FROM)/src/assistant/api.py" src/assistant/api.py
	cp "$(FROM)/src/assistant/core.py" src/assistant/core.py
	cp "$(FROM)/src/assistant/composers.py" src/assistant/composers.py
	cp "$(FROM)/src/assistant/connectors.py" src/assistant/connectors.py
	cp "$(FROM)/src/assistant/output_gate.py" src/assistant/output_gate.py
	cp "$(FROM)/src/assistant/audit_log.py" src/assistant/audit_log.py
