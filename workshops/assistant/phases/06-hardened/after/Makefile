include ../../../../../_lesson.mk

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }
	cp "$(FROM)/src/assistant/memory.py" src/assistant/memory.py
	cp "$(FROM)/src/assistant/tenancy.py" src/assistant/tenancy.py
	cp "$(FROM)/src/assistant/crew.py" src/assistant/crew.py
