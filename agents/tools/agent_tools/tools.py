"""Three tools for a notes agent: read, draft, delete.

- read_note(note_id): read-only. Returns {"text": ...} or {"error": ...}.
- draft_reply(note_id, body): reversible. Saves a draft, never sends.
- delete_note(note_id): irreversible. Refused unless the application has
  recorded a human approval for that note (see grant_approval below).

Rules: the docstring says WHAT and WHEN (the model's only interface), type
hints are the schema, every argument is validated, and errors come back as
data, never as exceptions.

Approval is application state, never a tool argument. The model fills every
parameter in a tool signature, so an `approve: bool` parameter would be a
gate the model can open by itself.

What the live runs showed (see MISUSE.md):
- The docstring is the interface. Without draft_reply's docstring the model
  filled `body` with the note's text instead of the user's message.
- The result is a second interface. "saved as draft, not sent" told the
  model afterwards what the docstring would have told it beforehand.
- A docstring guides, only code protects. With or without delete_note's
  docstring the model tried to delete; both times the approval check refused.
"""

from __future__ import annotations

_NOTES: dict[str, str] = {"1": "buy milk", "2": "call dentist"}


_DRAFT: dict[str, str] = {}


_MAX_BODY = 2000

# Human approvals on file, keyed by note id. The application writes here when a
# person clicks approve; the model never sees this set or the granting function.
_APPROVALS: set[str] = set()


def grant_approval(note_id: str) -> None:
    """Record a human's approval to delete one note. Application code only —
    this function is NOT in the tool registry the model sees."""
    _APPROVALS.add(note_id)


def read_note(note_id: str) -> dict:
    """Read one note by its id. Use when the user wants to see an existing note.

    Args:
        note_id: the note's id, e.g. "1".
    Returns {"text": ...} on success or {"error": ...} if the id is empty or unknown.
    """
    if not note_id:
        return {"error": "The note_id cannot be empty"}
    if note_id not in _NOTES:
        return {"error": f"No note with id:{note_id}"}
    return {"text": _NOTES[note_id]}


def draft_reply(note_id: str, body: str) -> dict:
    """Save a reply to a note as a draft. Use when the user wants to answer or
    follow up on a note. It does NOT send anything: the draft is only saved,
    and it can be rewritten later by calling this again.

    Args:
        note_id: the note to reply to, e.g. "1".
        body: the full reply text, written by you, at most 2000 characters.
    Returns {"draft": ..., "status": ...} or {"error": ...}.
    """
    note = read_note(note_id)
    if "error" in note:
        return note
    if not body.strip():
        return {"error": "body is empty"}
    if len(body) > _MAX_BODY:
        return {"error": f"Body is too long: {len(body)} characters, max {_MAX_BODY}"}
    _DRAFT[note_id] = body
    return {"draft": body, "status": "saved as draft, not sent"}


def delete_note(note_id: str) -> dict:
    """Delete a note permanently. IRREVERSIBLE: use only when the user clearly
    asks to delete a note, and a human must have approved this exact delete.

    Args:
        note_id: the note to delete, e.g. "1".
    Refused unless the application has a human approval on file for this note.
    No argument can skip that check; if refused, tell the user approval is needed.
    Returns {"deleted": note_id} or {"error": ...}.
    """
    if note_id not in _APPROVALS:
        return {"error": f"refused: no human approval on file for note {note_id!r}"}
    note = read_note(note_id)
    if "error" in note:
        return note
    del _NOTES[note_id]
    _APPROVALS.discard(note_id)
    return {"deleted": note_id}
