# Threat model — the capstone assistant

STRIDE-lite over the components that actually exist in this repo. Every
mitigation cell names the code that implements it and the test that proves it;
nothing here is aspirational.

## System at a glance

```
caller ──HTTP──► FastAPI service (api.py, wired by service.py)
                   │  auth.py (optional JWT)         [trust boundary 1: caller]
                   │  screen on input (guardrails, + guard.py when enabled)
                   ├──► planner.py (reads the GOAL + the registry only)
                   ├──► agent loop (agent.py)
                   │      ├─ tools.py / connectors.py [trust boundary 2: tool output]
                   │      └─ MCP server (mcp_server.py, network in compose)
                   ├──► RAG (InMemoryRag | Qdrant)    [trust boundary 3: corpus]
                   ├──► memory, one store per subject (tenancy.py)
                   └──► composer (offline | Ollama)
```

Untrusted inputs cross three boundaries: what the caller types, what a tool
fetches, and what retrieval returns. All three are screened; none can approve
anything, and none of them reaches the planner — tool selection is a function of
the goal and the registry, so there is no path from a poisoned document to a
tool call to close in the first place.

## STRIDE, per element

| Threat | Where it lands | Mitigation (code) | Proof (test) |
|---|---|---|---|
| **S**poofing — caller pretends to be someone else | HTTP surface | a key source (`ASSISTANT_JWT_SECRET` for HS256, `ASSISTANT_JWKS_URL` for RS256) turns on Bearer-JWT validation: signature, expiry, audience, issuer, subject, scope (`auth.py`) | `test_auth.py` (forged/expired/wrong-audience/wrong-scope all 401), `verify-e2e.sh` check 2 |
| Spoofing — token minted for another service replayed here | HTTP surface | audience check (RFC 8707 resource binding) | `test_a_token_for_another_audience_is_rejected` |
| Spoofing — a token with NO `exp` or NO `sub` is accepted | HTTP surface | `require=` makes those claims mandatory rather than merely verified-if-present, and a blank `sub` is rejected separately — otherwise an unexpiring token lands in the shared anonymous partition | `test_a_token_that_never_expires_is_rejected`, `test_a_token_that_names_no_subject_is_rejected`, `test_a_token_whose_subject_is_blank_is_rejected` |
| Spoofing — algorithm confusion (`none`, or RS256 downgraded to HS256 and signed with the public key) | HTTP surface | accepted algorithms are pinned by `AuthPolicy`, never read from the token header | `test_an_unsigned_token_cannot_talk_its_way_in`, `test_an_hs256_token_cannot_be_smuggled_into_the_rs256_lane` |
| Spoofing — a token from an unexpected issuer | HTTP surface | `ASSISTANT_JWT_ISSUER` makes `iss` required and checked; in the JWKS lane the signature must also chain to that issuer's published keys | `test_an_issuer_is_required_once_one_is_configured`, `test_an_issuer_signed_token_is_accepted_against_the_jwks` |
| Spoofing — a stolen authorization code is redeemed by the thief | token issuance | PKCE `S256` (`oauth.py`): only the challenge crosses the wire, and the verifier never leaves the client; `state` is compared with `compare_digest` and the redirect URI must match exactly | `test_oauth.py`, exercised against Keycloak by `docker-compose.oauth.yml` |
| **T**ampering — prompt injection in the user message | guardrails L1 | `guardrails.screen`: expand (base64, percent-encoding, HTML entities) and squash (NFKC, invisible `Cf` characters, leet folding, separator removal) before scanning, then multilingual + exfiltration patterns over both surfaces | `test_an_injected_question_is_refused_at_the_door`, `test_screen_catches_the_other_two_web_encodings`, `test_screen_catches_split_and_substituted_tokens`; phase6 red-team v3 |
| Tampering — an obfuscation nobody wrote a pattern for | guardrails L2 (optional) | `ASSISTANT_GUARD_MODEL` adds a local model as a second opinion on every untrusted string; it can only ADD a block, never clear one, and fails open to the deterministic verdict (`guard.py`) | `test_guard.py`, `test_a_real_model_pipeline_still_refuses_the_whole_attack_corpus` (integration) |
| Tampering — indirect injection via a poisoned document | RAG corpus | screened on the way IN (`Assistant.ingest` refuses before `rag.add`, rejection audited and counted back to the caller) and again on the way out — every retrieved doc re-screened before it becomes evidence, injection-bearing docs dropped (`screen_contexts`), evidence spotlighted in the model prompt | `test_a_poisoned_document_is_refused_at_ingest_not_merely_at_retrieval`, `test_the_service_never_composes_from_a_poisoned_context`, `test_ingest_screening_does_not_replace_retrieval_screening` |
| Tampering — injection via tool/MCP output | tool boundary | `harden_registry` re-screens every read-only tool's output, MCP tools included | `test_poisoned_tool_output_is_screened_before_the_agent_sees_it` |
| Tampering — a poisoned document steers TOOL CHOICE | planner | `planner.choose` takes the goal and the registry and nothing else; contexts and tool output are not parameters, so an instruction in the corpus cannot select a tool even if it survives screening | `test_only_the_goal_selects_a_tool_never_a_retrieved_document` |
| Tampering — a discovered MCP tool is called with invented arguments | planner | a tool whose required arguments cannot all be filled is skipped rather than guessed at; `required_args` comes from the server's own input schema | `test_a_tool_whose_arguments_cannot_be_filled_is_never_proposed` |
| Tampering — content claims "already approved" | agent loop | approval lives only in server-side grant records, never in content; each grant pays for ONE run | `test_an_approval_authorizes_exactly_one_run`; phase6 `approval-bypass` rows |
| Tampering — approved call is swapped for another | agent loop | the grant carries a fingerprint of the canonical arguments, checked at execution | `test_approving_a_tool_without_its_arguments_authorizes_nothing` |
| **R**epudiation — "who approved that send?" | whole service | persistent SQLite audit log: policy decisions, tool runs, approvals — every row bound to the verified subject AND to the request id, trace id, approval id, canonical args hash and outcome, so the answer is a query rather than a regex over prose | `test_the_audit_log_records_the_whole_approval_story_with_identities`, `test_a_gated_call_records_WHICH_grant_authorized_it_in_a_column`, `test_every_row_of_a_request_carries_the_id_the_caller_was_given_back`, `test_the_audit_trail_survives_a_process_restart` |
| **I**nformation disclosure — PII sits in the corpus waiting to be read | RAG corpus | redaction happens at ingest, so the SSN is never written down rather than merely never returned — data minimisation, not filtering | `test_pii_is_redacted_before_it_is_written_down_not_after_it_is_read` |
| **I**nformation disclosure — PII leaks in answers | output gate | `guardrails.output_ok` on every composed answer (batch and streamed) | `test_pii_never_passes_through_untouched` (phase6), stream redaction path in `test_stream.py` |
| Information disclosure — cross-tenant reads (corpus) | RAG | tenant = verified JWT `sub`: per-tenant BM25 index offline, server-side Qdrant payload filter in production | `test_one_tenant_cannot_retrieve_another_tenants_documents`, `test_qdrant_tenant_filter_isolates_users` (integration) |
| Information disclosure — cross-tenant recall (memory) | memory | one STORE per subject (`tenancy.py`), so recall is scoped by the backend rather than filtered afterwards; SQLite rows carry the subject in a `user` column the recall query joins on | `test_one_persons_memory_never_surfaces_in_anothers_recall`, `test_the_partition_is_in_the_database_not_only_in_the_process`, `verify-e2e.sh` check 9 (two real tokens over HTTP) |
| Information disclosure — data exfiltration requests | guardrails L1 | transfer-verb + external-destination patterns block outright | phase6 `exfiltration` rows |
| **D**enial of service — request floods | HTTP surface | token-bucket rate limit (429) + concurrency cap (503), `/health` exempt | `test_requests_beyond_the_burst_get_429`, `..._cap_get_503` |
| DoS — a hung adapter takes the service down | adapters | per-call timeouts + retries (`resilience.py`), then fallback to the offline tier with `/health` reporting `degraded` | `test_a_dead_rag_backend_degrades_to_the_offline_tier` |
| DoS — one runaway container starves the host | compose | pinned images, memory/cpu limits, `restart: unless-stopped`, `stop_grace_period` | compose structural checks in `phase8-deploy/01-compose` |
| **E**levation of privilege — agent fires an irreversible tool | agent loop | HITL: `requires_approval` tools pause; approval is a single-use grant bound to subject + arguments + expiry, claimed atomically at the call site; replayed `/approve` (idempotency key) is a no-op | `test_a_gated_tool_pauses_until_it_is_approved`, `test_a_replayed_approval_does_not_grant_twice` |
| Elevation — a discovered MCP tool ungates itself | discovery boundary | a tool that arrives by discovery is gated by DEFAULT (`mcp_client.gate`); `readOnlyHint`/`destructiveHint` are propagated but can only tighten the verdict, never open it. Only `ASSISTANT_MCP_READONLY_ALLOWLIST` — local, operator-owned — ungates one, and `tier.mcp_ungated` on `/health` says how many | `test_a_server_claiming_read_only_is_still_gated`, `test_a_hint_can_tighten_the_gate_it_cannot_open_it`, `test_the_servers_own_annotations_survive_discovery` (integration), `verify-e2e.sh` check 10 |
| Elevation — one caller spends another caller's approval | agent loop | the grant names the verified subject; a different subject finds no grant | `test_one_subjects_approval_cannot_authorize_another_subjects_send` |
| Elevation — two concurrent requests spend one approval | agent loop | consume is a single `DELETE ... RETURNING`; exactly one caller wins the row | `test_one_grant_survives_a_stampede_of_concurrent_consumers` |
| Elevation — token with the wrong scope calls a privileged route | HTTP surface | per-endpoint scopes (`assistant:ask/ingest/approve`), membership tested by splitting the space-separated claim rather than substring | `test_a_token_missing_the_endpoint_scope_is_rejected`, `verify-e2e.sh` check 2 |
| Elevation — a revoked token keeps working | HTTP surface | short `exp` plus a stated skew policy (`ASSISTANT_JWT_LEEWAY`, 60s default, 30s in the secure profile) — leeway is the revocation window and is sized deliberately, not left to chance | `test_clock_skew_is_a_policy_with_an_edge_on_both_sides` |

## What this model does NOT cover (accepted, documented)

- **Secrets management**: `ASSISTANT_JWT_SECRET` / `TELEGRAM_BOT_TOKEN` arrive as
  env vars. The secure compose profile at least refuses to start on a committed
  default (`${VAR:?}`), but a real deployment moves them to a secret store; out
  of scope for a local-first course stack. The JWKS lane sidesteps the question
  for auth specifically — there is no shared signing secret to manage.
- **Token revocation before expiry**: verification is local and offline by
  design, so a token stays good until `exp`. Introspection or a deny-list would
  fix it and would couple every request to the identity provider's availability;
  the accepted answer is short lifetimes.
- **Transport encryption**: compose runs plain HTTP on an internal network. The
  secure profile binds the published port to loopback so that is enforced by the
  kernel rather than by intent, but TLS termination still belongs to whatever
  fronts the service.
- **Model-level attacks** (weight extraction, adversarial suffixes against the
  local model): the containment stance assumes injections WILL land and bounds
  the blast radius instead. The optional guard model raises the cost of a novel
  phrasing; it is depth, not a floor, and it is explicitly allowed to fail open.
- **A screen that is complete**: expansion and squashing retire the cheap
  obfuscations, not the category. Both surfaces are pattern lists, and a pattern
  list is a record of what somebody already thought of.
- **Availability of the host itself**: single-machine compose; no HA story.

See `RUNBOOK.md` for what to do when one of the mitigated threats fires anyway.
