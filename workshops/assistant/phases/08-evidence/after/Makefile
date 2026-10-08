include ../../../../../_lesson.mk

.PHONY: adopt-mine
adopt-mine:  ## copy the files you finished in the previous layer
	@test -n "$(FROM)" || { echo "usage: make adopt-mine FROM=../<previous>/before"; exit 1; }
	cp "$(FROM)/src/assistant/auth.py" src/assistant/auth.py
	cp "$(FROM)/src/assistant/oauth.py" src/assistant/oauth.py
