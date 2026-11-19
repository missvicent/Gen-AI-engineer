# 4.3 Human-in-the-loop

**Goal.** Build a durable approval gate: irreversible actions (send/pay/delete/schedule)
pause instead of running, the pause is persisted to disk, and a human's decision
resumes it by token — even after the process dies. Durability is what separates
HITL from an `if` statement.
**Prerequisite.** 4.2 Tools (you know why irreversible actions need a human gate).
**Effort.** ~40 min to green on the fast tests · +25 min for the integration tier · ~75 min realistic first pass.

## Do this

```bash
make setup && make test     # 5 failing tests — read them, they are the spec
$EDITOR hitl_agent/hitl.py         # TODO 1 (Runner.act), TODO 2 (Runner.resume)
make check                  # green on the fast tier
$EDITOR hitl_agent/hitl.py         # TODO 3 (approval_graph) — then: make test-integration
make check                  # green: ruff + pyright + pytest, all offline
```

## What the first failure means

`test_reversible_action_runs_immediately` fails because `Runner.act` isn't built
yet: a reversible action like `read` should just run. But the test that defines the
lesson is `test_pending_survives_a_restart` — a brand-new `Runner` over the same
store file must be able to resume a pause it never saw created. That only works if
`act` persists the `PendingApproval` to disk, not just to memory.

## Done when

- [ ] `make check` is green (lint, types, fast tests).
- [ ] A pending approval survives a simulated crash: a fresh `Runner` on the same
      store resumes it by token (the restart test pins this).
- [ ] `resume(token, approved=False)` rejects without ever executing the action.
- [ ] `make test-integration` is green: your `approval_graph` pauses at `interrupt()`,
      resumes by `thread_id`, keeps two paused threads apart, and completes a pause a
      different process started.

## Stuck?

1. `act` branches on `action in IRREVERSIBLE`: run now, or mint a token, build a
   `PendingApproval`, save it, and return it — without calling the tool.
2. Persist `self._pending` as JSON at `self.store` on every change, and load it in
   (or before) `resume` so a new instance sees old pauses. `resume` pops the entry,
   then either runs `self.tools[action](**args)` or returns a rejected status.

## The LangGraph lane (TODO 3)

`approval_graph` in the same file is the same three ideas in the framework:
`interrupt()` suspends the graph, a checkpointer persists it, and
`Command(resume=...)` continues it by `thread_id`. `make test-integration` runs
it, including a restart test where a brand-new graph over the same checkpoint
file completes a pause it never saw start.

Write it rather than read it. The shape looks obvious until you place the
interrupt: it has to be its own node, upstream of the effect, and the effect has
to depend on a value that node produces — otherwise you have paused a run that
already sent the email, or built a gate that reads a flag somebody has to
remember to set. Both mistakes pass a casual reading and fail these tests.

Needs the opt-in install (LangGraph + the SQLite checkpointer, a heavy dependency
tree). The nodes are deterministic, so no model, GPU, or network.
