# 4.1 ReAct from scratch

**Goal.** Build the agent loop yourself — decide, act, observe, repeat — with the
leash (a hard step cap and a wall-clock deadline) living in code, not in a prompt.
Once you've written it, no framework loop is magic again.
**Prerequisite.** 1.1 Universal client (the `decide` brain you'll inject in prod is
your `complete()` call).
**Effort.** ~30 min to green on the fast tests · no integration tier · ~50 min realistic first pass.

## Do this

```bash
make setup && make test     # 3 failing tests — read them, they are the spec
$EDITOR src/agent.py        # fill the one TODO: run_agent, the loop with hard caps
make check                  # green: ruff + pyright + pytest, all offline
```

## What the first failure means

`test_agent_calls_a_tool_then_finishes` fails because `run_agent` isn't built yet.
It scripts the model brain: step one asks for the `calc` tool, step two returns a
final answer. Your loop has to call `decide`, run the requested tool, append the
`(decision, result)` pair to state, and hand the answer back when `is_final` is set.

## Done when

- [ ] `make check` is green (lint, types, fast tests).
- [ ] A brain that never finishes still terminates: `max_steps_exceeded` appears in
      the output instead of an infinite loop (the step-cap test pins this).
- [ ] An unknown tool or a tool exception becomes an observation the brain can
      recover from, not a crash.

## Stuck?

1. The tests never touch a real model — `decide` is just a function you call with
   `(goal, state)`. Treat its `Decision` as data and branch on `is_final`.
2. Wrap the loop body in `for step in range(max_steps)` and check
   `time.monotonic()` against a deadline each iteration; catch exceptions from
   `tools[decision.tool]` and store the error string as the observation. The
   provided `calculator` already shows the tool side: it walks an AST allowlist
   instead of `eval`, because model-supplied text is untrusted input.

No integration lane: the brain is injected and scripted, so the loop's logic proves
out offline — wire `decide` to your Phase-1 client when you want a live run.
