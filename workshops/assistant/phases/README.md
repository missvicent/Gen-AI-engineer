# Workshop phases

Open one folder. Edit the files it names. Run `make test` there.

`workshops/assistant/generator` writes these folders. It is not where you work.
`workshops/assistant/after` is the finished assistant the compose file builds.
Do not rename it.

| Phase | You edit | First test |
|-------|----------|------------|
| [`02-rag`](02-rag/before/) | `rag.py` | `test_semantic_search` |
| [`03-evals`](03-evals/before/) | `evals.py` | `test_the_suite_scores_every_slice` |
| [`04-agent`](04-agent/before/) | `tools.py`, `agent.py` | `test_readonly_tool_runs_and_finishes` |
| [`05-memory`](05-memory/before/) | `memory.py`, `tenancy.py`, `crew.py` | `test_a_remembered_fact_comes_back_with_its_source` |
| [`06-hardened`](06-hardened/before/) | `guardrails.py`, `screening.py`, `guard.py` | `test_screen_blocks_injection` |
| [`07-mcp`](07-mcp/before/) | `mcp_client.py`, `planner.py` | `test_discovered_tools_are_added_by_discovery_not_hardcoding` |
| [`08-trace`](08-trace/before/) | `observe.py`, `cache.py`, `usage.py` | `test_wrapping_the_registry_traces_every_tool_without_editing_any_tool` |
| [`08-corpus`](08-corpus/before/) | `adapters.py` | `test_reingesting_the_same_source_updates_instead_of_accumulating` |
| [`08-reliability`](08-reliability/before/) | `deadline.py`, `resilience.py`, `idempotency.py`, `outbox.py`, `approvals.py` | `test_a_bug_is_not_retried_three_times` |
| [`08-service`](08-service/before/) | `service.py`, `api.py`, `core.py`, `composers.py`, `connectors.py`, `output_gate.py`, `audit_log.py` | `test_health_reports_the_offline_tier` |
| [`08-auth`](08-auth/before/) | `auth.py`, `oauth.py` | `test_without_a_key_source_the_zero_key_demo_path_stays_open` |
| [`08-evidence`](08-evidence/before/) | `report.py` | `test_build_portfolio_is_still_the_page` |

The defect lab is not another copy of the service. It runs in
[`../after`](../after/) — see [`09-defect-lab/WORKSHOP-DEFECT-LAB.md`](09-defect-lab/WORKSHOP-DEFECT-LAB.md).
