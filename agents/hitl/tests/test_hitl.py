import tempfile
from pathlib import Path

from hitl_agent.hitl import PendingApproval, Runner

TOOLS = {
    "read": lambda **k: {"ok": "read data"},
    "send": lambda **k: {"sent": k.get("to")},
}


def _runner():
    d = tempfile.mkdtemp()
    return Runner(TOOLS, store=Path(d) / "hitl.json")


def test_reversible_action_runs_immediately():
    r = _runner()
    assert r.act("read") == {"ok": "read data"}


def test_irreversible_action_pauses():
    r = _runner()
    pending = r.act("send", to="alice")
    assert isinstance(pending, PendingApproval)
    assert pending.action == "send"


def test_resume_approved_executes():
    r = _runner()
    p = r.act("send", to="alice")
    assert r.resume(p.token, approved=True) == {"sent": "alice"}


def test_resume_rejected_does_not_execute():
    r = _runner()
    p = r.act("send", to="bob")
    assert r.resume(p.token, approved=False)["status"] == "rejected"


def test_pending_survives_a_restart():
    d = tempfile.mkdtemp()
    store = Path(d) / "hitl.json"
    r1 = Runner(TOOLS, store=store)
    p = r1.act("send", to="carol")
    # simulate crash: brand-new Runner loads the persisted pending approval
    r2 = Runner(TOOLS, store=store)
    assert r2.resume(p.token, approved=True) == {"sent": "carol"}
