include ../../../_lesson.mk

GATE := ../../../phase8-deploy/02-ci/after

.PHONY: report gate
report:           ## run the service on trial: PORTFOLIO.md for humans, evals/report.json for CI
	PYTHONPATH=src uv run python -m assistant.report

# The same trial against the system that ships. `report` measures an offline
# proxy in one second and says so; this measures the deployed stack — Qdrant with
# the semantic embedder, reranking on, a RAGAS judge, all 58 red-team rows with
# their benign controls — and refuses to run if any component has fallen back.
#
# It needs the composed stack up and these five variables pointing at it. There
# is no default that quietly works: a release number produced against localhost
# defaults nobody checked is the failure this whole lane exists to prevent.
#
#   cd ../../phase8-deploy/01-compose/after && docker compose up -d
#   make release-evidence
#
# Full instructions, and what to do with the output: docs/RELEASE-CHECKLIST.md
# A release measurement runs on its OWN corpus and its own database. Neither
# default is cosmetic: step 5 of the checklist runs verify-e2e.sh against the same
# Qdrant, and that script ingests refund policies, expense pages, an escalation
# page it later deletes, and the poisoned document from the injection check. Left
# in the collection, all of that becomes part of the corpus a recall number is
# measured over — so retrieval metrics for a release would be partly a measurement
# of the previous test run.
#
# **Neither is set here any more, and that is the fix.** Both used to be, and both
# were fixed names: `assistant-release` and `evidence/release.db`. A fixed name is
# not a fresh one — it is the SAME state on every run, and `docker compose down -v`
# reaches the volume but never the host file. The audited copy of that database held
# 306 audit rows and 18 memories from previous runs, so the percentiles were measured
# against a warm cache, tenancy against other subjects' memories, and containment
# against approvals granted weeks earlier.
#
# So `release.py` derives both when they are unset — `evidence/runs/<id>/release.db`
# and `assistant-release-<id>` — asserts every table and the collection start empty,
# and removes them afterwards. Set either explicitly and you get exactly what you
# named, which is what `--reuse-state` is for; the report is stamped `reused` and the
# publication gate refuses it.
RELEASE_ENV := QDRANT_URL=$${QDRANT_URL:-http://localhost:6333} \
  OLLAMA_HOST=$${OLLAMA_HOST:-http://localhost:11434} \
  ASSISTANT_EMBED_MODEL=$${ASSISTANT_EMBED_MODEL:-nomic-embed-text} \
  ASSISTANT_RERANK_MODEL=$${ASSISTANT_RERANK_MODEL:-BAAI/bge-reranker-base} \
  ASSISTANT_MIN_SCORE=$${ASSISTANT_MIN_SCORE:-0.58}

.PHONY: release-evidence
release-evidence: ## smoke evidence on the real rig: deployed stack, RAGAS judge, whole red team
	@mkdir -p evidence
	$(RELEASE_ENV) PYTHONPATH=src \
	  uv run --group integration --group release python -m assistant.release

# The merge gate, over the report THIS code just produced rather than a committed
# fixture. Same CLI the workflow calls, so a gate that would block in CI blocks on
# a laptop first — which is the only place a red gate is cheap.
gate: report      ## generate the report, then run the four merge gates over it
	cd $(GATE) && uv run python src/gate.py $(CURDIR)/evals/report.json

# The defect lab. Three vulnerabilities this codebase actually shipped are kept
# as running code in defects/variants.py, and each regression test is run TWICE:
# once against the current code, where it must pass, and once with its defect
# seeded back in, where it must fail.
#
# The green run comes first, and that order is load-bearing. "It went red" is
# only evidence if the test was capable of being green — otherwise an unwritten
# test, a typo'd import or a collection error all read as a caught defect. Prove
# the test passes, THEN prove the bug breaks it. Neither half means anything
# alone, which is why so many regression tests are worth nothing.
DEFECTS := unbound-approval pre-gate-stream lockless-build
LAB := PYTHONPATH=src:. uv run pytest -q defects/test_regressions.py

.PHONY: defect-lab
defect-lab:       ## your regression tests must pass on the fix and fail on every seeded defect
	@$(LAB) >/dev/null 2>&1 || { \
	  echo "the lab is not green against the current code — write the tests first"; \
	  $(LAB); exit 1; }
	@echo "green against the fixed code"
	@for d in $(DEFECTS); do \
	  if DEFECT=$$d $(LAB) >/dev/null 2>&1; then \
	    echo "NOT A REGRESSION TEST: still green with '$$d' seeded back in"; exit 1; \
	  else \
	    echo "red as required with $$d seeded"; \
	  fi; \
	done
	@echo "defect-lab: every defect is caught, and the fix is green"
	@mkdir -p evidence
	@printf '{"defects": [%s], "green_against_fix": true, "measured_on": "%s"}\n' \
	  "$$(echo $(DEFECTS) | tr ' ' '\n' | sed 's/.*/\"&\"/' | paste -sd, -)" \
	  "$$(date +%Y-%m-%d)" > evidence/defect-lab.json
	@echo "stamped evidence/defect-lab.json"

# The course-wide evidence log. `report` measures this service; this measures the
# whole course, folding the capstone numbers together with whatever the earlier
# phases left behind in evidence/.
#
# The stamp above is why this target can be trusted: evidence/defect-lab.json is
# written by the lab AFTER it passes, never by hand. Every artifact the log reads
# is produced the same way, which is what separates "proven" from "claimed".
.PHONY: evidence
evidence:         ## EVIDENCE.md + evidence/manifest.json across all nine phases
	PYTHONPATH=src uv run python -m assistant.evidence
