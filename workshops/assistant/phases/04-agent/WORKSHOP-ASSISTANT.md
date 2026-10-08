# Workshop · Personal assistant  (ends Phase 4)

**Effort.** ~2 h of focused build time · +30 min for the integration tier · ~3 h realistic first pass.

*An author's estimate, bounded by measured volume — deliverables, TODO groups, tests, brief length — and not by learner telemetry, which this course does not collect. Treat it as relative sizing, not a stopwatch.*

Give the assistant an agent loop and real tools: read email, read news, send a
Telegram message, schedule an event. Everything is a tool; sends/schedules are
gated behind human approval.

Three passes. **Minimum** is the walking skeleton — the smallest thing that is
really an agent, and a place to stop that is not quitting. **Full** is the
version you would show someone. **Stretch** is for when this came easily.

## Minimum
- [ ] Read + summarize email (read-only tool, no gate)
- [ ] `send_telegram` and `schedule_event` work AND pause for approval first

## Full
- [ ] Fetch + summarize a news page
- [ ] Every connector is a proper tool: docstring, type hints, validation, error-as-data
- [ ] Hard step cap + timeout in code; a local model handles triage and summarizing,
      escalating only when it has to
- [ ] Credentials come from env vars, never from code

## Stretch
- [ ] A "morning brief" chaining inbox summary + top news into one message you approve, then send
- [ ] Cost per run logged with the Phase-1 meter, so you can see what the assistant costs per day
- [ ] A second read-only connector of your own choosing

`docker compose up` is **not** a deliverable here. Containerizing the stack is
Workshop 8's job, and asking for it in Phase 4 is how a workshop that should take
an evening turns into one nobody finishes. Env-var credentials are the part that
matters now, because Phase 6 attacks this agent and Phase 8 deploys it.

Implement `tools.py` + `agent.py`. Tests: `tests/test_agent.py`.
