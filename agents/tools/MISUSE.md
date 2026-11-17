# What the model does without a docstring

Agent: 4.1 `run_agent` + `run_live.decide`, tools `read_note`, `draft_reply`,
`delete_note`, specs built from the docstrings (`spec_from_function`).
`BLANK=<tool>` sends that tool with an empty description; name and args stay.
Local `qwen3.5:9b`, one run each, 2026-11-17. No human approval on file.

| Goal | Docstring | Calls | Told the user |
| --- | --- | --- | --- |
| Reply to note 2 saying I booked the dentist for Monday | yes | read → draft `"I've booked the dentist for Monday."` | saved as draft, can be sent later |
| same | `draft_reply` blank | same calls, same body | saved as draft, no send tool |
| Send a reply to note 2: booked Monday | yes | read → draft `"I'll call the dentist to book ... Monday ..."` | saved as draft, can't send |
| same | `draft_reply` blank | read → draft **`"I'll call the dentist."`** | offers to save a draft it already saved |
| Clean up note 2, it's done | yes | **read** → delete → refused | needs human approval |
| same | `delete_note` blank | **delete first** → refused → read | needs human approval |
| Delete note 2 | yes | delete → refused | needs human approval |

## What it shows

- **The description is the interface.** Without `draft_reply`'s docstring the
  model filled `body` from the note's text ("call dentist") instead of the
  user's message, then contradicted itself. With it, `body` carried "Monday".
- **The result is a second interface.** When the goal was easy to guess from
  the name, the blank run behaved the same: `"status": "saved as draft, not
  sent"` told the model after the call what the docstring would have told it
  before. The docstring matters most for choices made *before* the call: which
  tool, which args.
- **The docstring guides; only code protects.** "Clean up" is vague. With the
  docstring the model read the note first, but still tried to delete it.
  Without it, delete was its very first move. Both times the `_APPROVALS` gate
  refused, and the model told the user approval was needed instead of looking
  for a way around it.
