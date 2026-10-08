include ../../../../../_lesson.mk

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }
	cp "$(FROM)/src/assistant/observe.py" src/assistant/observe.py
	cp "$(FROM)/src/assistant/cache.py" src/assistant/cache.py
	cp "$(FROM)/src/assistant/usage.py" src/assistant/usage.py
