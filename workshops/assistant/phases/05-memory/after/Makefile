include ../../../../../_lesson.mk

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }
	cp "$(FROM)/src/assistant/tools.py" src/assistant/tools.py
	cp "$(FROM)/src/assistant/agent.py" src/assistant/agent.py
