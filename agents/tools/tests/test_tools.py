import inspect

import agent_tools.tools as tools


def test_read_validates_and_returns_data():
    assert tools.read_note("")["error"]
    assert tools.read_note("1")["text"] == "buy milk"


def test_errors_are_data_not_exceptions():
    # a bad id returns {"error": ...}, it does NOT raise
    out = tools.read_note("999")
    assert "error" in out


def test_irreversible_tool_refuses_without_a_recorded_approval():
    assert "error" in tools.delete_note("2")


def test_irreversible_tool_runs_once_a_human_approval_is_on_file():
    tools.grant_approval("2")
    assert tools.delete_note("2")["deleted"] == "2"
    # one approval buys exactly one delete — the record is consumed
    assert "error" in tools.delete_note("2")


def test_approval_is_not_a_model_fillable_argument():
    # The model fills every parameter in a tool signature. If approval were a
    # parameter, the model could approve its own delete.
    assert "approve" not in inspect.signature(tools.delete_note).parameters


def test_docstrings_say_what_and_when():
    # the docstring IS the interface — enforce it exists and is non-trivial
    for fn in (tools.read_note, tools.draft_reply, tools.delete_note):
        assert fn.__doc__ and len(fn.__doc__) > 40
