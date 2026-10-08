from assistant.agent import Pending, Step, run


def test_readonly_tool_runs_and_finishes():
    script = [
        Step(tool="read_emails", args={"limit": 1}),
        Step(tool="", args={}, is_final=True, answer="you have 1 email"),
    ]
    it = iter(script)
    res = run("summarize inbox", lambda g, s: next(it))
    assert res.text == "you have 1 email"
    assert res.pending is None


def test_gated_tool_pauses_without_approval():
    script = [Step(tool="send_telegram", args={"chat_id": "1", "message": "hi"})]
    it = iter(script)
    res = run("message the team", lambda g, s: next(it))
    assert isinstance(res.pending, Pending)
    assert res.pending.tool == "send_telegram"
    assert not res.fired_irreversible_tool_without_approval


def test_gated_tool_fires_with_approval():
    script = [
        Step(tool="send_telegram", args={"chat_id": "1", "message": "hi"}),
        Step(tool="", args={}, is_final=True, answer="sent"),
    ]
    it = iter(script)
    res = run("message the team", lambda g, s: next(it), approvals={"send_telegram": True})
    assert res.text == "sent"


def test_every_executed_tool_lands_in_the_audit():
    # observe.py derives agent.step_count from this list. A successful run that
    # audits nothing would report zero steps — the regression this test pins.
    script = [
        Step(tool="read_emails", args={"limit": 1}),
        Step(tool="read_news", args={"url": "x"}),
        Step(tool="read_emails", args={"limit": 2}),
        Step(tool="", args={}, is_final=True, answer="done"),
    ]
    it = iter(script)
    res = run("morning brief", lambda g, s: next(it))
    assert res.text == "done"
    assert len(res.audit) == 3
    assert all(entry.startswith("ran: ") for entry in res.audit)
