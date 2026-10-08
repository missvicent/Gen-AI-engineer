"""The planner reads the registry, not a hardcoded list.

The bug these tests exist to prevent is subtle and was shipped: Workshop 7
discovered tools from an MCP server and merged them into the registry, and the
planner — which knew two tool names by heart — could never pick one. Every test
here is really the same question asked from a different angle: can a tool that
did not exist when this file was written be chosen, called, and refused?
"""


def test_the_whole_service_answers_using_a_discovered_tool():
    """End to end through /ask: nothing in core.py or service.py knows this tool's
    name, and the answer still contains what it returned."""
    from assistant.mcp_client import extend_assistant
    from assistant.service import build_assistant
    from assistant.settings import Settings

    spec = {
        "name": "lookup_fact",
        "description": "Look up a company fact by topic. Use for policy questions.",
        "required_args": ("topic",),
        "read_only": True,
    }
    fact = lambda name, args: {  # noqa: E731
        "fact": "Refunds are processed within five business days."
    }

    assistant = build_assistant(Settings(mcp_readonly_allowlist=("lookup_fact",)))
    assistant.base_registry = extend_assistant(
        assistant.base_registry, [spec], fact, ("lookup_fact",)
    )
    answer = assistant.ask("look up the company fact for the refund window")
    assert "five business days" in answer["answer"]
    assert answer["audit"] == ["ran: lookup_fact"]


def test_the_same_discovered_tool_pauses_when_the_operator_never_reviewed_it():
    """Same server, same self-description, no allowlist entry: the answer is an
    approval pause rather than a fact. The tool's own claim about itself made no
    difference, which is the property worth having."""
    from assistant.mcp_client import extend_assistant
    from assistant.service import build_assistant
    from assistant.settings import Settings

    assistant = build_assistant(Settings())
    assistant.base_registry = extend_assistant(
        assistant.base_registry,
        [
            {
                "name": "lookup_fact",
                "description": "Look up a company fact by topic. Use for policy questions.",
                "required_args": ("topic",),
                "read_only": True,  # the server insists
            }
        ],
        lambda name, args: {"fact": "Refunds are processed within five business days."},
    )
    answer = assistant.ask("look up the company fact for the refund window")
    assert answer["pending"]["tool"] == "lookup_fact"
    assert "five business days" not in answer["answer"]
