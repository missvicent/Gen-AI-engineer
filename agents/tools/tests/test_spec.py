from agent_tools.spec import spec_from_function
from agent_tools.tools import draft_reply


def test_spec_comes_from_the_docstring_and_type_hints():
    spec = spec_from_function(draft_reply)
    assert spec["name"] == "draft_reply"
    assert "does NOT send" in spec["description"]
    assert spec["parameters"] == {
        "type": "object",
        "properties": {"note_id": {"type": "string"}, "body": {"type": "string"}},
        "required": ["note_id", "body"],
    }


def test_blank_hides_the_description_only():
    spec = spec_from_function(draft_reply, blank=True)
    assert spec["description"] == ""
    assert spec["parameters"] == spec_from_function(draft_reply)["parameters"]
