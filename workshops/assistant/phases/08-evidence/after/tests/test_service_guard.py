"""The model-in-the-loop guard: what it may do, and what it must never do.

Every test here is offline. The guard is a callable returning a bool, so a
"model" that always says INJECTION, or always errors, or takes the scenic route
through a prompt-injection attempt of its own, is three lines of Python — and
testing the WIRING is the point. Whether a given 9B model classifies well is a
question for the eval suite; whether a well-behaved answer can unblock something
the deterministic screen refused is a question for this file, and the answer
must be no regardless of which model you plug in.
"""

from assistant import guardrails
from assistant.settings import Settings


def test_health_says_which_screen_is_live():
    from assistant.service import build_assistant

    assert build_assistant(Settings()).tier()["guard"] == "regex-only"
    hardened = build_assistant(
        Settings(ollama_host="http://ollama:11434", guard_model="llama-guard3:8b")
    )
    assert hardened.tier()["guard"] == "llama-guard3:8b"


def test_the_guard_covers_every_untrusted_channel_not_just_the_question():
    """The failure this wiring exists to prevent.

    A guard bolted onto `/ask` and nowhere else leaves retrieved documents,
    tool output and ingested files screened by the regex alone — which is to say
    it leaves the indirect channels, the ones an attacker actually uses, on the
    old filter. The assistant therefore takes ONE screen and hands it to all of
    them, and this test refuses to let that become three.
    """
    from assistant.service import build_assistant

    seen: list[str] = []

    def recording(text: str) -> tuple[bool, str]:
        seen.append(text)
        return guardrails.screen(text)

    assistant = build_assistant(Settings())
    assistant.screen = recording

    assistant.ingest(["refunds take five business days"], "alice")
    assert seen, "ingested documents never reached the screen"

    seen.clear()
    assistant.ask("how long do refunds take", "alice")
    assert any("refunds take five" in text for text in seen), (
        "the retrieved document never reached the screen"
    )
    assert any("how long do refunds" in text for text in seen), (
        "the question never reached the screen"
    )
