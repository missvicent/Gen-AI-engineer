"""Real local summarizer via Ollama. `make test-integration` (needs ollama serve)."""

import pytest

pytestmark = pytest.mark.integration

DOC = (
    "Acme refund policy, Q3 2026. Refunds are processed within five business "
    "days of approval. Approval requires a manager signature for amounts over $500."
)


def test_real_local_model_produces_a_context_line():
    from contextual_chunks.contextual import contextualize, local_summarizer

    out = contextualize(
        DOC, "Refunds are processed within five business days.", local_summarizer()
    )
    first_line = out.split("\n")[0]
    assert len(first_line) > 10
    assert first_line != "Refunds are processed within five business days."
